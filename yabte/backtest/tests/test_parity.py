# Setup basic logging to capture any output
import logging
import unittest
from decimal import Decimal

import numpy as np
import pandas as pd

from yabte.backtest import (
    Book,
    CostType,
    MissingPriceError,
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


class TestFillPriceParity(ParityAssertionsMixin, unittest.TestCase):
    """`OHLCAsset.fill_price` picks the same fill on both engines."""

    def setUp(self):
        idx = pd.bdate_range("2025-01-06", periods=6)
        self.data = pd.DataFrame(
            {
                ("A", "Open"): [100.0, 102.0, 101.0, 105.0, 104.0, 106.0],
                ("A", "High"): [103.0, 104.0, 106.0, 107.0, 108.0, 109.0],
                ("A", "Low"): [99.0, 100.0, 100.5, 103.0, 102.0, 104.0],
                ("A", "Close"): [101.0, 103.0, 104.0, 106.0, 105.0, 107.0],
            },
            index=idx,
        )

        class Trader(Strategy):
            def on_open(self):
                if len(self.data) in (1, 4):
                    self.orders.append(SimpleOrder(asset_name="A", size=10))

            def on_close(self):
                if len(self.data) == 2:
                    self.orders.append(SimpleOrder(asset_name="A", size=-5))

        self.Trader = Trader

    def run_engine(self, engine, fill_price, data=None):
        return StrategyRunner(
            data=self.data if data is None else data,
            assets=[OHLCAsset(name="A", fill_price=fill_price)],
            strategies=[self.Trader()],
            books=[Book(name="Main", cash="100000")],
            engine=engine,
        ).run()

    def test_fill_prices(self):
        skip_or_fail_without_cpp(self)
        # orders fill on rows 0, 2 (placed at row 1's close) and 3
        expected = {
            "midpoint": [Decimal("101"), Decimal("103.25"), Decimal("105")],
            "open": [Decimal("100"), Decimal("101"), Decimal("105")],
            "close": [Decimal("101"), Decimal("104"), Decimal("106")],
        }
        for fill_price, prices in expected.items():
            with self.subTest(fill_price=fill_price):
                res_py = self.run_engine("python", fill_price)
                res_cpp = self.run_engine("cpp", fill_price)
                self.assertEqual(res_py.transaction_history.price.to_list(), prices)
                self.assertResultParity(res_py, res_cpp, context=fill_price)

    def test_open_fill_needs_an_open_price(self):
        skip_or_fail_without_cpp(self)
        data = self.data.copy()
        data.loc[data.index[3], ("A", "Open")] = float("nan")
        for engine in ["python", "cpp"]:
            with self.subTest(engine=engine):
                with self.assertRaisesRegex(MissingPriceError, "No traded price"):
                    self.run_engine(engine, "open", data)


class TestCloseFirstParity(ParityAssertionsMixin, unittest.TestCase):
    """`PositionalOrder(close_first=False)` trades only the difference."""

    TARGETS = ["5", "8.1", "8.1", "-2.7", "0.3", "0"]

    def run_engine(self, engine, close_first):
        idx = pd.bdate_range("2025-01-06", periods=len(self.TARGETS))
        data = pd.DataFrame(
            {
                ("A", f): [100.0 + i for i in range(len(idx))]
                for f in ["High", "Low", "Close"]
            },
            index=idx,
        )
        targets = self.TARGETS

        class Rebalance(Strategy):
            def on_open(self):
                target = targets[len(self.data) - 1]
                self.orders.append(
                    PositionalOrder(
                        asset_name="A", size=target, close_first=close_first
                    )
                )

        return StrategyRunner(
            data=data,
            assets=[OHLCAsset(name="A")],
            strategies=[Rebalance()],
            books=[Book(name="Main", cash="100000")],
            engine=engine,
        ).run()

    def test_trades(self):
        skip_or_fail_without_cpp(self)
        expected = {
            True: ["5", "-5", "8.1", "-8.1", "-2.7", "2.7", "0.3", "-0.3"],
            False: ["5", "3.1", "-10.8", "3", "-0.3"],
        }
        for close_first, quantities in expected.items():
            with self.subTest(close_first=close_first):
                res_py = self.run_engine("python", close_first)
                res_cpp = self.run_engine("cpp", close_first)
                self.assertEqual(
                    res_py.transaction_history.quantity.to_list(),
                    [Decimal(q) for q in quantities],
                )
                self.assertResultParity(res_py, res_cpp, context=str(close_first))


class TestFlatPositionParity(ParityAssertionsMixin, unittest.TestCase):
    """C++ positions are doubles, so sums of rounded trades drift off the asset's
    quantity grid unless snapped back to it.

    Unsnapped, a position that should be flat keeps a tiny residual, which needs a price
    once the asset's prices stop.
    """

    def run_engine(self, engine, order_for_day, days):
        idx = pd.bdate_range("2024-01-01", periods=days + 2)
        px = pd.Series(10.0, index=idx)
        px.iloc[days:] = np.nan  # e.g. an expired option
        data = pd.DataFrame({("X", f): px for f in ["High", "Low", "Close"]})

        class Trader(Strategy):
            def on_open(self):
                i = len(self.data) - 1
                if i < days:
                    self.orders.append(order_for_day(i))

        return StrategyRunner(
            data=data,
            assets=[OHLCAsset(name="X", quantity_round_dp=6)],
            strategies=[Trader()],
            books=[Book(name="Main", cash="0")],
            engine=engine,
        ).run()

    def check(self, order_for_day, days):
        skip_or_fail_without_cpp(self)
        res_py = self.run_engine("python", order_for_day, days)
        res_cpp = self.run_engine("cpp", order_for_day, days)
        self.assertResultParity(res_py, res_cpp)
        self.assertEqual(res_py.books[0].positions["X"], 0)

    def test_simple_orders_back_to_flat(self):
        sizes = ["0.1", "0.2", "-0.3"]
        self.check(lambda i: SimpleOrder(asset_name="X", size=sizes[i]), len(sizes))

    def test_close_first_false_back_to_flat(self):
        rng = np.random.default_rng(0)
        targets = [f"{x:.6f}" for x in rng.uniform(-5000, 5000, 39)] + ["0"]
        self.check(
            lambda i: PositionalOrder(
                asset_name="X", size=targets[i], close_first=False
            ),
            len(targets),
        )


class Recorder(Strategy):
    """Module level, so run_batch's worker processes can import it."""

    def on_open(self):
        self.record(open=self.data.A.Open.iloc[-1])

    def on_close(self):
        self.record(close=self.data.A.Close.iloc[-1], scale=self.params.get("scale", 1))


class TestRecordParity(unittest.TestCase):
    """`Strategy.record` gives the same table on both engines."""

    def setUp(self):
        idx = pd.bdate_range("2025-01-06", periods=4)
        self.data = pd.DataFrame(
            {
                ("A", "Open"): [1.0, 2.0, 3.0, 4.0],
                ("A", "Close"): [1.5, 2.5, 3.5, 4.5],
            },
            index=idx,
        )
        self.expected = pd.DataFrame(
            {"open": [1.0, 2.0, 3.0, 4.0], "close": [1.5, 2.5, 3.5, 4.5]}, index=idx
        )

    def runner(self, engine):
        return StrategyRunner(
            data=self.data,
            assets=[OHLCAsset(name="A")],
            strategies=[Recorder()],
            engine=engine,
        )

    def test_recorded(self):
        skip_or_fail_without_cpp(self)
        for engine in ["python", "cpp"]:
            with self.subTest(engine=engine):
                srr = self.runner(engine).run({"scale": 2})
                recorded = srr.strategies[0].recorded
                pd.testing.assert_frame_equal(
                    recorded[["open", "close"]], self.expected, check_freq=False
                )
                self.assertEqual(recorded.scale.to_list(), [2] * 4)

    def test_batch_runs_record_separately(self):
        skip_or_fail_without_cpp(self)
        for engine in ["python", "cpp"]:
            with self.subTest(engine=engine):
                srrs = self.runner(engine).run_batch([{"scale": 1}, {"scale": 3}])
                self.assertEqual(
                    [r.strategies[0].recorded.scale.iloc[-1] for r in srrs], [1, 3]
                )
                self.assertEqual([len(r.strategies[0].recorded) for r in srrs], [4, 4])

    def test_record_needs_a_timestamp(self):
        with self.assertRaisesRegex(RuntimeError, "needs a timestamp"):
            Strategy().record(x=1)
        self.assertTrue(Strategy().recorded.empty)


class Market:
    """A non-scalar run param, e.g. a market object a hedging strategy prices with."""

    def __init__(self, spot):
        self.spot = spot


class UsesObjectParams(Strategy):
    def on_open(self):
        p = self.params
        self.record(spot=p.market.spot, rolls=len(p.rolls))
        self.orders.append(PositionalOrder(asset_name="A", size=p.rolls[0]))


class TestNonScalarParams(ParityAssertionsMixin, unittest.TestCase):
    """Python strategies get their params unchanged on either engine.

    The C++ param map only holds numbers, strings and bools, so only native C++
    strategies are limited to those.
    """

    PARAMS = [
        {"market": Market(100), "rolls": [1, 2]},
        {"market": Market(101), "rolls": [3]},
    ]

    @classmethod
    def setUpClass(cls):
        idx = pd.bdate_range("2025-01-06", periods=4)
        cls.data = pd.DataFrame(
            {("A", f): [100.0, 101.0, 102.0, 103.0] for f in ["High", "Low", "Close"]},
            index=idx,
        )

    def runner(self, engine, strategy):
        return StrategyRunner(
            data=self.data,
            assets=[OHLCAsset(name="A")],
            strategies=[strategy],
            books=[Book(name="Main", cash="100000")],
            engine=engine,
        )

    def test_run(self):
        skip_or_fail_without_cpp(self)
        params = self.PARAMS[0]
        res_py = self.runner("python", UsesObjectParams()).run(params)
        res_cpp = self.runner("cpp", UsesObjectParams()).run(params)
        self.assertResultParity(res_py, res_cpp)
        self.assertEqual(res_cpp.strategies[0].params.rolls, [1, 2])
        self.assertEqual(res_cpp.strategies[0].recorded.spot.to_list(), [100] * 4)

    def test_run_batch(self):
        skip_or_fail_without_cpp(self)
        from yabte.backtest.runner_cpp import CppStrategyRunner

        sr = self.runner("cpp", UsesObjectParams())
        cpp = CppStrategyRunner(
            data=sr.data, assets=sr.assets, strategies=sr.strategies, books=sr.books
        )
        for res, params in zip(cpp.run_batch(self.PARAMS, num_threads=1), self.PARAMS):
            with self.subTest(spot=params["market"].spot):
                expected = self.runner("python", UsesObjectParams()).run(params)
                self.assertResultParity(expected, res)
                self.assertIs(res.strategies[0].params.market.__class__, Market)

    def test_native_strategy_names_the_param(self):
        skip_or_fail_without_cpp(self)
        from yabte.backtest.runner_cpp import yabte_cpp

        sr = self.runner("cpp", yabte_cpp.SMAXOStrategy("A", 1, 2))
        with self.assertRaisesRegex(TypeError, r"\['market', 'rolls'\].*native C\+\+"):
            sr.run(self.PARAMS[0])


class TestTransactionCostParity(ParityAssertionsMixin, unittest.TestCase):
    """Asset transaction costs, charged on each order's net quantity per asset."""

    TARGETS = ["5", "3", "3", "-2.5", "0"]

    def run_engine(self, engine, cost_type, cost, close_first=True, extra=None):
        idx = pd.bdate_range("2025-01-06", periods=len(self.TARGETS))
        data = pd.DataFrame(
            {
                ("A", f): [100.0 + i for i in range(len(idx))]
                for f in ["High", "Low", "Close"]
            },
            index=idx,
        )
        targets = self.TARGETS

        class Rebalance(Strategy):
            def on_open(self):
                i = len(self.data) - 1
                self.orders.append(
                    PositionalOrder(
                        asset_name="A",
                        size=targets[i],
                        close_first=close_first,
                        label=f"rebalance {i}",
                    )
                )
                if extra is not None and i == 1:
                    self.orders.append(extra())

        return StrategyRunner(
            data=data,
            assets=[OHLCAsset(name="A", cost=cost, cost_type=cost_type)],
            strategies=[Rebalance()],
            books=[Book(name="Main", cash="1000")],
            engine=engine,
        ).run()

    def costs(self, res):
        tx = res.transaction_history
        return tx[tx.type == "Cost"]

    def test_cost_types(self):
        skip_or_fail_without_cpp(self)
        # net quantities 5, -2, -5.5, 2.5 at prices 100, 101, 103, 104
        expected = {
            (CostType.BPS, "10"): ["0.5", "0.202", "0.5665", "0.26"],
            (CostType.PER_UNIT, "0.65"): ["3.25", "1.3", "3.575", "1.625"],
            (CostType.PER_ORDER, "1"): ["1", "1", "1", "1"],
        }
        for (cost_type, cost), amounts in expected.items():
            for close_first in (True, False):
                with self.subTest(cost_type=cost_type, close_first=close_first):
                    res_py = self.run_engine("python", cost_type, cost, close_first)
                    res_cpp = self.run_engine("cpp", cost_type, cost, close_first)
                    costs = self.costs(res_py)
                    self.assertEqual(
                        costs.total.to_list(), [-Decimal(a) for a in amounts]
                    )
                    self.assertEqual(
                        costs.charged_quantity.to_list(),
                        [Decimal(q) for q in ["5", "-2", "-5.5", "2.5"]],
                    )
                    self.assertEqual(
                        costs.order_label.to_list(),
                        ["rebalance 0", "rebalance 1", "rebalance 3", "rebalance 4"],
                    )
                    self.assertEqual(set(costs.cost_type), {cost_type})
                    self.assertEqual(
                        res_py.books[0].cash,
                        Decimal(1000) - sum(Decimal(a) for a in amounts)
                        # what the trades made: flat at the end
                        + (
                            -5 * Decimal(100)
                            + 2 * Decimal(101)
                            + Decimal("5.5") * 103
                            - Decimal("2.5") * 104
                        ),
                    )
                    self.assertResultParity(res_py, res_cpp, context=str(cost_type))
                    self.assertEqual(len(self.costs(res_cpp)), 4)

    def test_reporting(self):
        skip_or_fail_without_cpp(self)
        for engine in ("python", "cpp"):
            with self.subTest(engine=engine):
                res = self.run_engine(engine, CostType.PER_UNIT, "0.65")
                # costs paid so far: on net quantities 5, -2, -, -5.5 and 2.5
                self.assertEqual(
                    res.book_history.Main.costs.to_list(),
                    [3.25, 4.55, 4.55, 8.125, 9.75],
                )
                self.assertEqual(float(res.books[0].costs), 9.75)
                # close_first books close and reopen trades: 5, -5 +3, -3 -2.5, 2.5
                turnover = res.turnover.loc[("Main", "A")]
                self.assertEqual(turnover.trades, 6)
                self.assertEqual(float(turnover.quantity), 21)
                self.assertEqual(
                    float(turnover.notional),
                    5 * 100 + 8 * 101 + 5.5 * 103 + 2.5 * 104,
                )
                self.assertEqual(float(turnover.costs), 9.75)

    def test_no_cost_books_nothing(self):
        skip_or_fail_without_cpp(self)
        for engine in ("python", "cpp"):
            with self.subTest(engine=engine):
                res = self.run_engine(engine, CostType.BPS, "0")
                self.assertEqual(set(res.transaction_history.type), {"Trade"})
                self.assertEqual(set(res.book_history.Main.costs), {0})
                self.assertEqual(res.turnover.costs.to_list(), [0])

    def test_book_percent_sizes_on_cash_after_costs(self):
        skip_or_fail_without_cpp(self)

        def buy_half_of_cash():
            return SimpleOrder(
                asset_name="A", size=50, size_type=OrderSizeType.BOOK_PERCENT
            )

        res_py = self.run_engine(
            "python", CostType.PER_ORDER, "20", extra=buy_half_of_cash
        )
        res_cpp = self.run_engine(
            "cpp", CostType.PER_ORDER, "20", extra=buy_half_of_cash
        )
        self.assertResultParity(res_py, res_cpp)
        # day 1: cash after day 0's buy of 5 at 100 and its cost, then the rebalance
        # to 3 (sells 2 at 101) and its cost
        cash = Decimal(1000) - 500 - 20 + 202 - 20
        tx = res_py.transaction_history
        bought = tx[(tx.type == "Trade") & (tx.order_label.isna())].quantity.iloc[0]
        self.assertEqual(bought, round(cash / 2 / 101, 2))


class TestCppRejectsSubclasses(unittest.TestCase):
    """The C++ engine refuses subclasses whose overrides it would ignore."""

    def setUp(self):
        idx = pd.bdate_range("2025-01-06", periods=3)
        self.data = pd.DataFrame(
            {("A", f): [100.0, 101.0, 102.0] for f in ["Open", "High", "Low", "Close"]},
            index=idx,
        )

    def run_cpp(self, asset=None, order_cls=SimpleOrder):
        class Trader(Strategy):
            def on_open(self):
                self.orders.append(order_cls(asset_name="A", size=1))

        return StrategyRunner(
            data=self.data,
            assets=[asset or OHLCAsset(name="A")],
            strategies=[Trader()],
            engine="cpp",
        ).run()

    def test_asset_overriding_a_price_hook(self):
        skip_or_fail_without_cpp(self)

        class TradeAtOpen(OHLCAsset):
            def intraday_traded_price(self, asset_day_data, size=None):
                return Decimal(asset_day_data.Open)

        class StepFee(OHLCAsset):
            def transaction_cost(self, quantity, price):
                return Decimal(1 if abs(quantity) < 10 else 2)

        with self.assertRaisesRegex(NotImplementedError, "transaction_cost"):
            self.run_cpp(StepFee(name="A"))

        class WholeLots(OHLCAsset):
            def round_quantity(self, quantity):
                return Decimal(int(quantity))

        with self.assertRaisesRegex(NotImplementedError, "intraday_traded_price"):
            self.run_cpp(TradeAtOpen(name="A"))
        with self.assertRaisesRegex(NotImplementedError, "round_quantity"):
            self.run_cpp(WholeLots(name="A"))

    def test_hook_set_on_an_instance(self):
        skip_or_fail_without_cpp(self)
        asset = OHLCAsset(name="A")
        asset.end_of_day_price = lambda asset_day_data: Decimal(1)  # type: ignore[method-assign]
        with self.assertRaisesRegex(NotImplementedError, "end_of_day_price"):
            self.run_cpp(asset)

        class Trader(Strategy):
            def on_open(self):
                order = SimpleOrder(asset_name="A", size=1)
                order.pre_execute_check = lambda ts, tp: None  # type: ignore[method-assign]
                self.orders.append(order)

        with self.assertRaisesRegex(NotImplementedError, "pre_execute_check"):
            StrategyRunner(
                data=self.data,
                assets=[OHLCAsset(name="A")],
                strategies=[Trader()],
                engine="cpp",
            ).run()

    def test_asset_subclass_without_overrides_runs(self):
        skip_or_fail_without_cpp(self)

        class Tagged(OHLCAsset):
            pass

        self.assertEqual(len(self.run_cpp(Tagged(name="A")).transaction_history), 3)

    def test_order_overriding_a_hook(self):
        skip_or_fail_without_cpp(self)

        class LimitOrder(SimpleOrder):
            def pre_execute_check(self, ts, trade_price):
                return None

        class Target(PositionalOrder):
            def post_complete(self, trades):
                pass

        for order_cls, hook in [
            (LimitOrder, "pre_execute_check"),
            (Target, "post_complete"),
        ]:
            with self.subTest(order=order_cls.__name__):
                with self.assertRaisesRegex(NotImplementedError, hook):
                    self.run_cpp(order_cls=order_cls)


if __name__ == "__main__":
    unittest.main()
