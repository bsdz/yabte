# Setup basic logging to capture any output
import logging
import unittest
from decimal import Decimal

import pandas as pd

from yabte.backtest import (
    Book,
    OHLCAsset,
    OrderSizeType,
    PositionalOrder,
    PositionalOrderCheckType,
    SimpleOrder,
    Strategy,
    StrategyRunner,
    Trade,
)
from yabte.backtest.tests._parity import (
    ParityAssertionsMixin,
    skip_or_fail_without_cpp,
)

logging.basicConfig(level=logging.INFO)


class TestParity(ParityAssertionsMixin, unittest.TestCase):
    def setUp(self):
        # Create deterministic data
        dates = pd.date_range(start="2024-01-01", periods=10, freq="B")
        prices = [float(100 + i) for i in range(10)]
        data = []
        for p in prices:
            # O, H, L, C, V
            data.append([p, p + 1.0, p - 1.0, p, 1000.0])

        self.df = pd.DataFrame(
            data,
            columns=pd.MultiIndex.from_product(
                [["GOOG"], ["Open", "High", "Low", "Close", "Volume"]]
            ),
            index=dates,
        )
        self.asset = OHLCAsset(name="GOOG", denom="USD")
        self.initial_cash = Decimal("100000")

    def run_strategy(self, engine_type):
        class ParityCheckStrategy(Strategy):
            def on_close(self):
                ix = self.data.index.get_loc(self.ts)
                symbol = "GOOG"

                # Day 1: Buy 10% of Book Value
                if ix == 1:
                    self.orders.append(
                        PositionalOrder(
                            asset_name=symbol,
                            size=10,
                            size_type=OrderSizeType.BOOK_PERCENT,
                            check_type=PositionalOrderCheckType.POS_TQ_DIFFER,
                        )
                    )
                # Day 3: Switch to Notional 5000
                elif ix == 3:
                    self.orders.append(
                        PositionalOrder(
                            asset_name=symbol,
                            size=5000,
                            size_type=OrderSizeType.NOTIONAL,
                            check_type=PositionalOrderCheckType.POS_TQ_DIFFER,
                        )
                    )
                # Day 5: Fixed Quantity 20
                elif ix == 5:
                    self.orders.append(
                        PositionalOrder(
                            asset_name=symbol,
                            size=20,
                            size_type=OrderSizeType.QUANTITY,
                            check_type=PositionalOrderCheckType.POS_TQ_DIFFER,
                        )
                    )
                # Day 7: Close
                elif ix == 7:
                    self.orders.append(
                        PositionalOrder(
                            asset_name=symbol,
                            size=0,
                            size_type=OrderSizeType.QUANTITY,
                            check_type=PositionalOrderCheckType.POS_TQ_DIFFER,
                        )
                    )

        book = Book(name="Main", cash=self.initial_cash)
        strat = ParityCheckStrategy()
        sr = StrategyRunner(
            data=self.df,
            assets=[self.asset],
            strategies=[strat],
            books=[book],
            engine=engine_type,
        )
        return sr.run()

    def test_parity_book_history_and_transactions(self):
        skip_or_fail_without_cpp(self)
        res_py = self.run_strategy("python")
        res_cpp = self.run_strategy("cpp")
        self.assertResultParity(res_py, res_cpp, context="parity")


class TestTimezoneParity(ParityAssertionsMixin, unittest.TestCase):
    """A timezone-aware index groups end-of-day tasks by local date on both engines."""

    def run_tz(self, tz, engine, hours):
        idx = pd.DatetimeIndex(
            [
                day + pd.Timedelta(h)
                for day in pd.bdate_range("2025-01-06", periods=3)
                for h in hours
            ],
            tz=tz,
        )
        px = [100.0 + i for i in range(len(idx))]
        data = pd.DataFrame(
            {("A", f): px for f in ["Open", "High", "Low", "Close"]}, index=idx
        )

        class Buy(Strategy):
            def on_open(self):
                if len(self.data) == 1:
                    self.orders.append(PositionalOrder(asset_name="A", size=10))

        return StrategyRunner(
            data=data,
            assets=[OHLCAsset(name="A")],
            strategies=[Buy()],
            books=[Book(name="Main", cash="-1000", rate=Decimal("0.001"))],
            engine=engine,
        ).run()

    def test_local_dates(self):
        skip_or_fail_without_cpp(self)
        # evening bars fall on the next UTC date in New York; morning bars on the
        # previous one in Tokyo
        hours = ["09:30:00", "12:00:00", "15:30:00", "19:30:00"]
        for tz in [None, "America/New_York", "Asia/Tokyo"]:
            with self.subTest(tz=tz):
                res_py = self.run_tz(tz, "python", hours)
                res_cpp = self.run_tz(tz, "cpp", hours)
                self.assertEqual(len(res_py.book_history), 3)
                self.assertEqual(
                    res_cpp.book_history.index.tz, res_py.book_history.index.tz
                )
                self.assertResultParity(res_py, res_cpp, context=str(tz))

    def test_ambiguous_local_times_raise(self):
        skip_or_fail_without_cpp(self)
        # US clocks go back at 02:00 on 2025-11-02, so 01:30 occurs twice
        idx = pd.date_range(
            "2025-11-02 00:30", periods=4, freq="30min", tz="America/New_York"
        )
        data = pd.DataFrame(
            {("A", f): 100.0 for f in ["Open", "High", "Low", "Close"]}, index=idx
        )
        with self.assertRaisesRegex(ValueError, "repeats a local time"):
            StrategyRunner(
                data=data,
                assets=[OHLCAsset(name="A")],
                strategies=[Strategy()],
                engine="cpp",
            ).run()


class TestResultParity(ParityAssertionsMixin, unittest.TestCase):
    """Both engines return the orders and strategy copies of each run."""

    def setUp(self):
        idx = pd.bdate_range("2025-01-06", periods=8)
        px = [100.0 + i for i in range(len(idx))]
        self.data = pd.DataFrame(
            {("A", f): px for f in ["Open", "High", "Low", "Close"]}, index=idx
        )

        class Trader(Strategy):
            book_name = None

            def on_open(self):
                n = len(self.data)
                if n in (1, 5):
                    self.orders.append(
                        SimpleOrder(
                            asset_name="A",
                            size=10,
                            label=f"buy{n}",
                            book=self.book_name,
                        )
                    )
                if n == 3:
                    for size in (5, 7):
                        self.orders.append(
                            PositionalOrder(
                                asset_name="A", size=size, label=f"pos{size}", key="k"
                            )
                        )

        self.Trader = Trader

    def runner(self, engine, strategy):
        return StrategyRunner(
            data=self.data,
            assets=[OHLCAsset(name="A")],
            strategies=[strategy],
            books=[Book(name="Main", cash="100000")],
            engine=engine,
        )

    def test_orders_and_strategies(self):
        skip_or_fail_without_cpp(self)
        results = {}
        for engine in ["python", "cpp"]:
            strategy = self.Trader()
            srr = self.runner(engine, strategy).run()
            with self.subTest(engine=engine):
                self.assertIsNone(strategy.params, "the caller's strategy was changed")
                self.assertIsNot(srr.strategies[0], strategy)
                self.assertIsInstance(srr.strategies[0], self.Trader)
            results[engine] = [
                (type(o).__name__, o.label, o.status.name) for o in srr.orders_processed
            ]
            self.assertEqual(len(srr.orders_unprocessed), 0)
        self.assertEqual(results["python"], results["cpp"])
        self.assertIn(("PositionalOrder", "pos5", "REPLACED"), results["cpp"])

    def test_cpp_batch_results_are_separate(self):
        skip_or_fail_without_cpp(self)
        from yabte.backtest.runner_cpp import CppStrategyRunner

        srrs = CppStrategyRunner(
            data=self.data,
            assets=[OHLCAsset(name="A")],
            strategies=[self.Trader()],
            books=[Book(name="Main", cash="100000")],
        ).run_batch([{}, {}], num_threads=2)
        self.assertIsNot(srrs[0].strategies[0], srrs[1].strategies[0])
        self.assertEqual([len(r.orders_processed) for r in srrs], [4, 4])
        self.assertIsNot(srrs[0].orders_processed[0], srrs[1].orders_processed[0])

    def test_unknown_book_raises(self):
        skip_or_fail_without_cpp(self)
        strategy = self.Trader()
        strategy.book_name = "Missing"
        for engine in ["python", "cpp"]:
            with self.subTest(engine=engine):
                with self.assertRaisesRegex(ValueError, "unknown book Missing"):
                    self.runner(engine, strategy).run()


if __name__ == "__main__":
    unittest.main()
