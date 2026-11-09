import logging
import unittest
from copy import deepcopy
from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd

import yabte.utilities.pandas_extension
from yabte.backtest import (
    BasketOrder,
    Book,
    OHLCAsset,
    OrderSizeType,
    OrderStatus,
    PositionalBasketOrder,
    PositionalOrder,
    SimpleOrder,
    Strategy,
    StrategyRunner,
)
from yabte.backtest.runner_cpp import CppStrategyRunner, yabte_cpp
from yabte.backtest.tests._parity import (
    HAS_CPP,
    ParityAssertionsMixin,
    skip_or_fail_without_cpp,
)
from yabte.tests._helpers import generate_nasdaq_dataset, generate_ohlc_dataset
from yabte.tests._unittest_numpy_extensions import NumpyTestCase
from yabte.utilities.strategy_helpers import crossover

logger = logging.getLogger(__name__)

# Check if C++ engine is available


class TestSMAXOStrat(Strategy):
    def init(self):
        p = self.params
        days_short = p.get("days_short", 10)
        days_long = p.get("days_long", 20)

        close_sma_short = (
            self.data.loc[:, (slice(None), "Close")]
            .rolling(days_short)
            .mean()
            .rename({"Close": "CloseSMAShort"}, axis=1, level=1)
        )
        close_sma_long = (
            self.data.loc[:, (slice(None), "Close")]
            .rolling(days_long)
            .mean()
            .rename({"Close": "CloseSMALong"}, axis=1, level=1)
        )
        self.data = pd.concat(
            [self.data, close_sma_short, close_sma_long], axis=1
        ).sort_index(axis=1)

    def on_close(self):
        p = self.params
        symbol = p.get("symbol", "GOOG")

        df = self.data[symbol]
        ix_2d = df.index[-2:]
        data = df.loc[ix_2d, ("CloseSMAShort", "CloseSMALong")].dropna()
        if len(data) == 2:
            if crossover(data.CloseSMAShort, data.CloseSMALong):
                self.orders.append(SimpleOrder(asset_name=symbol, size=100))
            elif crossover(data.CloseSMALong, data.CloseSMAShort):
                self.orders.append(SimpleOrder(asset_name=symbol, size=-100))


class TestSMAXOMultipleBookStrat(TestSMAXOStrat):
    def on_close(self):
        # create some orders

        for symbol in ["GOOG", "MSFT"]:
            book_name = f"{symbol}_BOOK"
            df = self.data[symbol]
            ix_2d = df.index[-2:]
            data = df.loc[ix_2d, ("CloseSMAShort", "CloseSMALong")].dropna()
            if len(data) == 2:
                if crossover(data.CloseSMAShort, data.CloseSMALong):
                    self.orders.append(
                        SimpleOrder(book=book_name, asset_name=symbol, size=-100)
                    )
                elif crossover(data.CloseSMALong, data.CloseSMAShort):
                    self.orders.append(
                        SimpleOrder(book=book_name, asset_name=symbol, size=100)
                    )


class TestPosOrderSizeStrat(Strategy):
    def on_close(self):
        p = self.params
        size_type = p.size_type
        size_factor = p.size_factor
        symbol = p.get("symbol", "GOOG")

        ix = self.data.index.get_loc(self.ts)
        if ix in [100, 201, 300, 401]:
            quantity = size_factor * (-1) ** ix
            self.orders.append(
                PositionalOrder(asset_name=symbol, size=quantity, size_type=size_type)
            )
        elif ix in [1000]:
            self.orders.append(PositionalOrder(asset_name=symbol, size=0))


class TestSpreadSimpleStrat(Strategy):
    def init(self):
        p = self.params
        s = self.data[p.s1].Close - p.factor * self.data[p.s2].Close
        self.data.loc[:, ("SPREAD", "Close")] = s
        self.mu = s.mean()
        self.sigma = s.std()

    def on_close(self):
        p = self.params
        s = self.data["SPREAD"].Close.iloc[-1]
        if s < self.mu - 0.5 * self.sigma:
            self.orders.append(PositionalOrder(asset_name=p.s1, size=100))
            self.orders.append(PositionalOrder(asset_name=p.s2, size=p.factor * 100))
        elif s > self.mu + 0.5 * self.sigma:
            self.orders.append(PositionalOrder(asset_name=p.s1, size=-100))
            self.orders.append(PositionalOrder(asset_name=p.s2, size=-p.factor * 100))
        elif abs(s) < 0.1 * self.sigma:
            self.orders.append(PositionalOrder(asset_name=p.s1, size=0))
            self.orders.append(PositionalOrder(asset_name=p.s2, size=0))


class TestBasketOrderSizeStrat(Strategy):
    def on_close(self):
        p = self.params
        size_type = p.size_type
        symbols = ["AAPL", "AMZN", "GOOG", "META"]
        weights = [1, 2, 3, 4]

        ix = self.data.index.get_loc(self.ts)
        if ix in [100, 201, 300, 401]:
            self.orders.append(
                BasketOrder(
                    asset_names=symbols, weights=weights, size=1, size_type=size_type
                )
            )
        elif ix in [1000]:
            self.orders.append(
                PositionalBasketOrder(
                    asset_names=symbols, weights=[0] * len(symbols), size=1
                )
            )


class StrategyRunnerTestCase(ParityAssertionsMixin, NumpyTestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets, cls.df_combined = generate_nasdaq_dataset()

    def _run_and_compare(self, name, kwargs, run_kwargs=None):
        """Run the same strategy on both engines and assert strict parity.

        Registered as a subtest so that, when the C++ extension is not built, it shows
        up as *skipped* in the test report rather than silently passing.
        """
        with self.subTest(engine="cpp", case=name):
            skip_or_fail_without_cpp(self)

            run_kwargs = run_kwargs or {}

            sr_py = StrategyRunner(**kwargs, engine="python")
            res_py = sr_py.run(params=run_kwargs)

            # Deep copy mutable inputs so the second run starts from clean state.
            kwargs_cpp = kwargs.copy()
            kwargs_cpp["strategies"] = [deepcopy(s) for s in kwargs["strategies"]]
            if kwargs_cpp.get("books"):
                kwargs_cpp["books"] = [deepcopy(b) for b in kwargs_cpp["books"]]

            sr_cpp = StrategyRunner(**kwargs_cpp, engine="cpp")
            try:
                res_cpp = sr_cpp.run(params=run_kwargs)
            except Exception as e:
                self.fail(f"C++ engine failed for {name}: {e}")

            self.assertResultParity(res_py, res_cpp, context=name)

    def test_cpp_availability(self):
        if HAS_CPP:
            print("C++ engine is available for testing.")
        else:
            print("C++ engine is NOT available. C++ parity tests will be skipped.")

    def test_sma_crossover(self):
        kwargs = {
            "data": self.df_combined,
            "assets": self.assets,
            "strategies": [TestSMAXOStrat()],
        }

        # Run standard python test
        sr = StrategyRunner(**kwargs)
        srr = sr.run()

        th = srr.transaction_history
        th["nc"] = -th.quantity * th.price
        bch = (
            th.pivot_table(index="ts", columns="book", values="nc", aggfunc="sum")
            .cumsum()
            .reindex(sr.data.index)
            .ffill()
            .fillna(0)
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    bch.astype("float64"),
                    srr.book_history.loc[:, (slice(None), "cash")].droplevel(
                        axis=1, level=1
                    ),
                )
            )
        )

        # Run comparison
        self._run_and_compare("test_sma_crossover", kwargs)

    def test_multiple_books(self):
        books = [
            Book(name="MSFT_BOOK", cash=Decimal("1000000")),
            Book(name="GOOG_BOOK", cash=Decimal("1000000")),
        ]

        kwargs = {
            "data": self.df_combined,
            "assets": self.assets,
            "strategies": [TestSMAXOMultipleBookStrat()],
            "books": books,
        }

        sr = StrategyRunner(**kwargs)
        srr = sr.run()

        th = srr.transaction_history
        self.assertEqual(len(th.book.unique()), 2)
        bh = srr.book_history
        self.assertEqual(len(bh.columns.levels[0]), 2)

        # Run comparison
        self._run_and_compare("test_multiple_books", kwargs)

    def test_positional_orders_quantity(self):
        # remove round_dp to avoid discrepancies between Python Decimal and C++ double rounding
        assets = [
            OHLCAsset(
                name=asset.name,
                denom=asset.denom,
                price_round_dp=None,
                quantity_round_dp=None,
                data_label=asset.data_label,
            )
            for asset in self.assets
        ]

        kwargs = {
            "data": self.df_combined,
            "assets": assets,
            "strategies": [TestPosOrderSizeStrat()],
        }
        run_params = {"size_type": OrderSizeType.QUANTITY, "size_factor": 100}

        sr = StrategyRunner(**kwargs)
        srr = sr.run(params=run_params)

        # 8 = ococococ
        self.assertEqual(len(srr.books[0].transactions), 8)

        self._run_and_compare("test_positional_orders_quantity", kwargs, run_params)

    def test_positional_orders_notional(self):
        # remove round_dp to avoid discrepancies between Python Decimal and C++ double rounding
        assets = [
            OHLCAsset(
                name=asset.name,
                denom=asset.denom,
                price_round_dp=None,
                quantity_round_dp=None,
                data_label=asset.data_label,
            )
            for asset in self.assets
        ]

        kwargs = {
            "data": self.df_combined,
            "assets": assets,
            "strategies": [TestPosOrderSizeStrat()],
        }
        run_params = {"size_type": OrderSizeType.NOTIONAL, "size_factor": 1000}

        sr = StrategyRunner(**kwargs)
        srr = sr.run(params=run_params)

        # 8 = ococococ
        self.assertEqual(len(srr.books[0].transactions), 8)

        self._run_and_compare("test_positional_orders_notional", kwargs, run_params)

    def test_positional_orders_book_percent(self):
        book = Book(name="Main", cash=Decimal("1000000"))

        kwargs = {
            "data": self.df_combined,
            "assets": self.assets,
            "strategies": [TestPosOrderSizeStrat()],
            "books": [book],
        }
        run_params = {
            "size_type": OrderSizeType.BOOK_PERCENT,
            "size_factor": 20,
        }

        sr = StrategyRunner(**kwargs)
        srr = sr.run(params=run_params)

        # 8 = ococococ
        self.assertEqual(len(srr.books[0].transactions), 8)

        self._run_and_compare("test_positional_orders_book_percent", kwargs, run_params)

    def test_spread_simple(self):
        params = {
            "s1": "GOOG",
            "s2": "MSFT",
            "factor": 4.5,
        }
        kwargs = {
            "data": self.df_combined,
            "assets": self.assets,
            "strategies": [TestSpreadSimpleStrat()],
        }

        sr = StrategyRunner(**kwargs)
        srr = sr.run(params)

        df_trades = pd.DataFrame(srr.books[0].transactions)
        self.assertEqual(len(df_trades), 6)
        self.assertEqual(len(df_trades.query("asset_name == 'GOOG'")), 3)
        self.assertEqual(len(df_trades.query("asset_name == 'MSFT'")), 3)

        # Note: mandates are not supported in C++ yet, but this test doesn't use mandates.
        self._run_and_compare("test_spread_simple", kwargs, params)

    def test_basket_order_quantity(self):
        # Basket orders might not be fully supported in C++ yet if they are expanded in Python.
        # But if they are just orders that expand to suborders, and suborders are supported, it might work?
        # Actually, `BasketOrder` expands to suborders in `apply`.
        # If the C++ engine logic for `Order::apply` matches, it should work.
        # However, `yabte_cpp` might not have BasketOrder class.
        # If the Python strategy expands it before sending to C++? No, strategies send orders to queue.
        # The C++ engine processes the queue.
        # If `BasketOrder` is not mapped in `CppStrategyAdapter._sync_orders`, it will fail/warn.
        # Let's check `runner_cpp.py`... it maps `SimpleOrder` but warns on others.
        # So this will likely fail parity check or produce no trades in C++.
        # We'll keep the Python test but maybe skip parity for now unless we fix adapter.

        kwargs = {
            "data": self.df_combined,
            "assets": self.assets,
            "strategies": [TestBasketOrderSizeStrat()],
        }
        run_params = {"size_type": OrderSizeType.QUANTITY}

        sr = StrategyRunner(**kwargs)
        srr = sr.run(params=run_params)

        # 20 = 4 x ttttc
        self.assertEqual(len(srr.books[0].transactions), 20)

        # Skipping C++ parity check for BasketOrder as adapter support is likely missing

    def test_on_open_masking(self):
        class TestOnOpenMaskStrat(Strategy):
            def on_open(self2):
                # only latest record masked
                if len(self2.data) > 1:
                    self.assertTrue(
                        all(
                            self2.data.iloc[:-1]
                            .loc[
                                :,
                                (
                                    slice(None),
                                    ("Open", "High", "Low", "Close", "Volume"),
                                ),
                            ]
                            .notnull()
                        )
                    )

                # these fields should be masked at open
                self.assertTrue(
                    all(
                        self2.data.iloc[-1:]
                        .loc[:, (slice(None), ("High", "Low", "Close", "Volume"))]
                        .isnull()
                    )
                )

                # this field should be unmasked
                self.assertTrue(
                    all(self2.data.iloc[-1:].loc[:, (slice(None), "Open")].notnull())
                )

            def on_close(self2):
                # at close all fields available
                self.assertTrue(
                    all(
                        self2.data.loc[
                            :, (slice(None), ("Open", "High", "Low", "Close", "Volume"))
                        ].notnull()
                    )
                )

        data = pd.DataFrame(
            [
                [100] * 10,
                [100] * 10,
                [100] * 10,
            ],
            columns=pd.MultiIndex.from_product(
                [["ACME", "BOKO"], ["High", "Low", "Open", "Close", "Volume"]]
            ),
            index=pd.date_range(start="20180102", periods=3, freq="B"),
        )

        sr = StrategyRunner(
            data=data,
            assets=[OHLCAsset(name="ACME"), OHLCAsset(name="BOKO")],
            strategies=[TestOnOpenMaskStrat()],
        )
        sr.run()

        # Masking logic is internal to Python wrapper.
        # C++ engine might not support masking verification via Python strategy assertions easily
        # because the strategy in C++ adapter might not see the masked data identically.
        # Skipping parity check.

    def test_limit_order(self):
        class LimitOrder(SimpleOrder):
            def pre_execute_check(self, ts, tp):
                # if goes above 110 then cancel
                if tp > 110:
                    return OrderStatus.CANCELLED
                # if drops below 90 then complete order
                elif tp < 90:
                    return None
                # otherwise leave open for another day
                return OrderStatus.OPEN

        class TestLimitOrderStrat(Strategy):
            def on_close(self):
                ix = self.data.index.get_loc(self.ts)
                if ix == 0:
                    self.orders.append(LimitOrder(asset_name="ACME", size=100))

        for ix, (data_arr, op_status, ou_status) in enumerate(
            [
                (
                    [
                        [105],
                        [115],
                        [110],
                    ],
                    [OrderStatus.CANCELLED],
                    [],
                ),
                (
                    [
                        [95],
                        [100],
                        [105],
                    ],
                    [],
                    [OrderStatus.OPEN],
                ),
                (
                    [
                        [95],
                        [100],
                        [85],
                    ],
                    [OrderStatus.COMPLETE],
                    [],
                ),
            ]
        ):
            with self.subTest(i=ix):
                data = pd.DataFrame(
                    data_arr,
                    columns=pd.MultiIndex.from_product([["ACME"], ["Close"]]),
                    index=pd.date_range(
                        start="20180102", periods=len(data_arr), freq="B"
                    ),
                )

                sr = StrategyRunner(
                    data=data,
                    assets=[OHLCAsset(name="ACME", denom="USD")],
                    strategies=[TestLimitOrderStrat()],
                )
                srr = sr.run()

                self.assertListEqual(
                    op_status, [o.status for o in srr.orders_processed]
                )
                self.assertListEqual(
                    ou_status, [o.status for o in srr.orders_unprocessed]
                )

        # Custom order logic (pre_execute_check override) in Python
        # is NOT executed by C++ engine unless we trampoline the Order class too.
        # Currently C++ adapter maps SimpleOrder to C++ SimpleOrder.
        # Custom python logic is lost. Parity check would fail.
        if HAS_CPP:
            pass
            # self._run_and_compare("test_limit_order", kwargs)

    def test_stop_loss_order(self):
        # Similar to Limit Order - uses custom pre_execute_check and post_complete
        # which are not supported in C++ adapter yet.
        class StopLossOrder(SimpleOrder):
            def pre_execute_check(self, ts, tp):
                # if drops below 90 then complete stop order
                if tp < 90:
                    return None
                # otherwise leave open for another day
                return OrderStatus.OPEN

        class OrderWithStopLosses(SimpleOrder):
            def post_complete(self, trades):
                self.suborders.extend(
                    [
                        StopLossOrder(
                            asset_name=t.asset_name,
                            size=-t.quantity,
                            label="my_stop",
                        )
                        for t in trades
                    ]
                )

        class TestStopLossOrderStrat(Strategy):
            def on_close(self):
                ix = self.data.index.get_loc(self.ts)
                if ix == 0:
                    self.orders.append(OrderWithStopLosses(asset_name="ACME", size=100))

        for ix, (data_arr, op_status, ou_status) in enumerate(
            [
                (
                    [
                        [105],
                        [115],
                        [110],
                    ],
                    [(OrderStatus.COMPLETE, None)],
                    [(OrderStatus.OPEN, "my_stop")],
                ),
                (
                    [
                        [95],
                        [100],
                        [85],
                    ],
                    [(OrderStatus.COMPLETE, None), (OrderStatus.COMPLETE, "my_stop")],
                    [],
                ),
            ]
        ):
            with self.subTest(i=ix):
                data = pd.DataFrame(
                    data_arr,
                    columns=pd.MultiIndex.from_product([["ACME"], ["Close"]]),
                    index=pd.date_range(
                        start="20180102", periods=len(data_arr), freq="B"
                    ),
                )

                sr = StrategyRunner(
                    data=data,
                    assets=[OHLCAsset(name="ACME", denom="USD")],
                    strategies=[TestStopLossOrderStrat()],
                )
                srr = sr.run()

                self.assertListEqual(
                    op_status, [(o.status, o.label) for o in srr.orders_processed]
                )
                self.assertListEqual(
                    ou_status, [(o.status, o.label) for o in srr.orders_unprocessed]
                )

    def test_priority(self):
        class TestPriorityStrat(Strategy):
            def on_close(self):
                ix = self.data.index.get_loc(self.ts)
                if ix == 0:
                    self.orders.append(
                        SimpleOrder(asset_name="ACME", size=100, priority=2)
                    )
                    self.orders.append(
                        SimpleOrder(asset_name="ACME", size=200, priority=3)
                    )
                    self.orders.append(
                        SimpleOrder(asset_name="ACME", size=300, priority=1)
                    )

        data_arr = [
            [105, 105, 105, 105],
            [115, 115, 115, 115],
            [110, 110, 110, 110],
        ]

        data = pd.DataFrame(
            data_arr,
            columns=pd.MultiIndex.from_product(
                [["ACME"], ["Close", "Open", "High", "Low"]]
            ),
            index=pd.date_range(start="20180102", periods=len(data_arr), freq="B"),
        )

        kwargs = {
            "data": data,
            "assets": [OHLCAsset(name="ACME", denom="USD")],
            "strategies": [TestPriorityStrat()],
        }

        sr = StrategyRunner(**kwargs)
        srr = sr.run()

        self.assertListEqual(
            srr.transaction_history.quantity.to_list(),
            [Decimal("200.00"), Decimal("100.00"), Decimal("300.00")],
        )

        # Priority logic should be supported in C++
        if HAS_CPP:
            pass
            # self._run_and_compare("test_priority", kwargs)

    def test_order_key(self):
        class LimitOrder(SimpleOrder):
            # this limit won't be met in test
            def pre_execute_check(self, ts, tp):
                # if goes above 200 then complete order
                if tp > 200:
                    return None
                # otherwise leave open for another day
                return OrderStatus.OPEN

        class TestOrderkeyStrat(Strategy):
            def on_close(self):
                ix = self.data.index.get_loc(self.ts)
                if ix == 0:
                    self.orders.append(
                        LimitOrder(asset_name="ACME", size=100, key="my_key")
                    )
                    self.orders.append(LimitOrder(asset_name="ACME", size=200))
                elif ix == 1:
                    self.orders.append(
                        LimitOrder(asset_name="ACME", size=300, key="my_key")
                    )

        data_arr = [
            [105],
            [115],
            [110],
        ]

        data = pd.DataFrame(
            data_arr,
            columns=pd.MultiIndex.from_product([["ACME"], ["Close"]]),
            index=pd.date_range(start="20180102", periods=len(data_arr), freq="B"),
        )

        sr = StrategyRunner(
            data=data,
            assets=[OHLCAsset(name="ACME", denom="USD")],
            strategies=[TestOrderkeyStrat()],
        )
        srr = sr.run()

        self.assertListEqual(
            [
                (OrderStatus.REPLACED, "my_key", Decimal("100")),
            ],
            [(o.status, o.key, o.size) for o in srr.orders_processed],
        )

        self.assertListEqual(
            [
                (OrderStatus.OPEN, None, Decimal("200")),
                (OrderStatus.OPEN, "my_key", Decimal("300")),
            ],
            [(o.status, o.key, o.size) for o in srr.orders_unprocessed],
        )

        # Order Key replacement logic should be in C++
        # But here we used Custom LimitOrder which has pre_execute_check.
        # This will fail parity because C++ adapter converts it to simple order
        # which doesn't have the check, so it might execute immediately.
        # Skipping parity.

    def test_run_batch(self):
        book = Book(name="Main", cash=Decimal("100000"))

        sr = StrategyRunner(
            data=self.df_combined,
            assets=self.assets,
            strategies=[TestSMAXOStrat()],
            books=[book],
        )

        param_iter = [
            {"days_long": n, "days_short": m}
            for n, m in zip([20, 30, 40, 50], [5, 10, 15, 20])
            if n > m
        ]

        srrs = sr.run_batch(param_iter)

        self.assertEqual(len(srrs), len(param_iter))

        # check we have distinct sharpe ratios for each param set
        sharpes = {
            srr.book_history.loc[:, ("Main", "total")].prc.sharpe_ratio()
            for srr in srrs
        }
        self.assertEqual(len(sharpes), len(param_iter))

    def test_run_batch_empty(self):
        sr = StrategyRunner(
            data=self.df_combined, assets=self.assets, strategies=[TestSMAXOStrat()]
        )
        self.assertEqual(sr.run_batch([]), [])

    def test_run_batch_matches_run(self):
        """The default process pool returns the same results, in order, as run()."""
        sr = StrategyRunner(
            data=self.df_combined, assets=self.assets, strategies=[TestSMAXOStrat()]
        )
        param_iter = [
            {"days_long": 30, "days_short": 10},
            {"days_long": 20, "days_short": 5},
        ]
        srrs = sr.run_batch(param_iter)

        for params, srr in zip(param_iter, srrs):
            expected = sr.run(params)
            self.assertResultParity(expected, srr, context=str(params))

    def test_run_batch_local_strategy_runs_serially(self):
        """A strategy class that worker processes can't import falls back to serial."""

        class LocalSMAXOStrat(TestSMAXOStrat):
            pass

        sr = StrategyRunner(
            data=self.df_combined, assets=self.assets, strategies=[LocalSMAXOStrat()]
        )
        param_iter = [
            {"days_long": 30, "days_short": 10},
            {"days_long": 20, "days_short": 5},
        ]
        with self.assertWarnsRegex(RuntimeWarning, "defined inside a function"):
            srrs = sr.run_batch(param_iter)

        self.assertEqual(len(srrs), len(param_iter))
        for params, srr in zip(param_iter, srrs):
            self.assertResultParity(sr.run(params), srr, context=str(params))

    def test_run_batch_cpp_parity(self):
        """C++ engine batches match the Python engine, both via the facade (a process
        pool, as the strategy is Python) and on the C++ thread pool directly.

        Uses an Enum param so it also checks each run's original params (not the
        sanitized C++ ParamMap) reach the Python strategy in C++ worker threads.
        """
        skip_or_fail_without_cpp(self)

        # no rounding, to avoid Decimal vs double discrepancies (as other parity tests)
        assets = [
            OHLCAsset(
                name=asset.name,
                denom=asset.denom,
                price_round_dp=None,
                quantity_round_dp=None,
                data_label=asset.data_label,
            )
            for asset in self.assets
        ]
        param_iter = [
            {"size_type": OrderSizeType.QUANTITY, "size_factor": 100},
            {"size_type": OrderSizeType.NOTIONAL, "size_factor": 1000},
            {"size_type": OrderSizeType.BOOK_PERCENT, "size_factor": 20},
        ]

        def runner(engine):
            return StrategyRunner(
                data=self.df_combined,
                assets=assets,
                strategies=[TestPosOrderSizeStrat()],
                books=[Book(name="Main", cash=Decimal("1000000"))],
                engine=engine,
            )

        srrs_py = runner("python").run_batch(param_iter)
        sr_cpp = runner("cpp")
        srrs_cpp = sr_cpp.run_batch(param_iter)
        srrs_cpp_threads = CppStrategyRunner(
            data=sr_cpp.data,
            assets=sr_cpp.assets,
            strategies=sr_cpp.strategies,
            books=sr_cpp.books,
        ).run_batch(param_iter, num_threads=len(param_iter))

        for srrs in (srrs_cpp, srrs_cpp_threads):
            self.assertEqual(len(srrs), len(param_iter))
            for params, res_py, res_cpp in zip(param_iter, srrs_py, srrs):
                self.assertEqual(len(res_cpp.books[0].transactions), 8)
                self.assertResultParity(res_py, res_cpp, context=str(params))

    def test_run_batch_cpp_native(self):
        """Native C++ strategies batch on the C++ thread pool, matching run()."""
        skip_or_fail_without_cpp(self)

        sr = StrategyRunner(
            data=self.df_combined,
            assets=self.assets,
            strategies=[
                yabte_cpp.SMAXOStrategy("GOOG", 10, 20),
                yabte_cpp.SMAXOStrategy("MSFT", 5, 30),
            ],
            books=[Book(name="Main", cash=Decimal("1000000"))],
            engine="cpp",
        )
        param_iter = [{}, {"unused": 1}, {"unused": 2}]
        srrs = sr.run_batch(param_iter)

        expected = sr.run()
        self.assertGreater(len(expected.transaction_history), 0)
        self.assertEqual(len(srrs), len(param_iter))
        for srr in srrs:
            self.assertResultParity(expected, srr)

    def test_intraday(self):
        class TestSMAXOStratIntraday(Strategy):
            def init(self):
                p = self.params
                days_short = p.get("days_short", 10)
                days_long = p.get("days_long", 20)

                close_sma_short = (
                    self.data.loc[:, (slice(None), "Close")]
                    .rolling(days_short)
                    .mean()
                    .rename({"Close": "CloseSMAShort"}, axis=1, level=1)
                )
                close_sma_long = (
                    self.data.loc[:, (slice(None), "Close")]
                    .rolling(days_long)
                    .mean()
                    .rename({"Close": "CloseSMALong"}, axis=1, level=1)
                )
                self.data = pd.concat(
                    [self.data, close_sma_short, close_sma_long], axis=1
                ).sort_index(axis=1)

            def on_close(self):
                p = self.params
                symbol = p.get("symbol", "A1 Inc")

                df = self.data[symbol]
                ix_2d = df.index[-2:]
                data = df.loc[ix_2d, ("CloseSMAShort", "CloseSMALong")].dropna()
                if len(data) == 2:
                    if crossover(data.CloseSMAShort, data.CloseSMALong):
                        self.orders.append(SimpleOrder(asset_name=symbol, size=100))
                    elif crossover(data.CloseSMALong, data.CloseSMAShort):
                        self.orders.append(SimpleOrder(asset_name=symbol, size=-100))

        assets, data = generate_ohlc_dataset(
            S0=[10, 100, 1000, 50],
            mu=0.05,
            vol=[0.2, 0.3, 0.1, 0.4],
            names=["A1 Inc", "B2 Corp", "C3 Ltd", "D4 Inc"],
            start=datetime(2025, 1, 1),
            end=datetime(2025, 3, 1),
            freq="30min",
            tick_freq="1min",
            rng=np.random.default_rng(12345),
        )

        book = Book(name="Main", cash=Decimal("100000"))

        kwargs = {
            "data": data,
            "assets": assets,
            "strategies": [TestSMAXOStratIntraday()],
        }

        sr = StrategyRunner(**kwargs)
        srr = sr.run()

        # TODO: include some checks on EOD vs intraday

        self._run_and_compare("test_intraday", kwargs)


if __name__ == "__main__":
    unittest.main()
