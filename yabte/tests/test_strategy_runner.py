import logging
import unittest
from decimal import Decimal

import numpy as np
import pandas as pd

from yabte.backtest import (
    Asset,
    BasketOrder,
    Book,
    Order,
    OrderSizeType,
    OrderStatus,
    PositionalBasketOrder,
    PositionalOrder,
    Strategy,
    StrategyRunner,
)
from yabte.tests._helpers import generate_nasdaq_dataset
from yabte.utilities.strategy_helpers import crossover

logger = logging.getLogger(__name__)


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
                self.orders.append(Order(asset_name=symbol, size=100))
            elif crossover(data.CloseSMALong, data.CloseSMAShort):
                self.orders.append(Order(asset_name=symbol, size=-100))


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
                        Order(book=book_name, asset_name=symbol, size=-100)
                    )
                elif crossover(data.CloseSMALong, data.CloseSMAShort):
                    self.orders.append(
                        Order(book=book_name, asset_name=symbol, size=100)
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
        s = self.data["SPREAD"].Close[-1]
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


class StrategyRunnerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets, cls.df_combined = generate_nasdaq_dataset()

    def test_sma_crossover(self):
        sr = StrategyRunner(
            data=self.df_combined,
            assets=self.assets,
            strat_classes=[TestSMAXOStrat],
        )
        sr.run()

        th = sr.transaction_history
        th["nc"] = -th.quantity * th.price
        bch = (
            th.pivot_table(index="ts", columns="book", values="nc", aggfunc="sum")
            .cumsum()
            .reindex(sr.data.index)
            .fillna(method="ffill")
            .fillna(0)
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    bch.astype("float64"),
                    sr.book_history.loc[:, (slice(None), "cash")].droplevel(
                        axis=1, level=1
                    ),
                )
            )
        )

    def test_multiple_books(self):
        books = [
            Book(name="MSFT_BOOK", cash=Decimal("1000000")),
            Book(name="GOOG_BOOK", cash=Decimal("1000000")),
        ]

        sr = StrategyRunner(
            data=self.df_combined,
            assets=self.assets,
            strat_classes=[TestSMAXOMultipleBookStrat],
            books=books,
        )
        sr.run()

        th = sr.transaction_history
        self.assertEqual(len(th.book.unique()), 2)
        bh = sr.book_history
        self.assertEqual(len(bh.columns.levels[0]), 2)

    def test_positional_orders_quantity(self):
        # test using quantities
        sr = StrategyRunner(
            data=self.df_combined,
            assets=self.assets,
            strat_classes=[TestPosOrderSizeStrat],
            strat_params={"size_type": OrderSizeType.QUANTITY, "size_factor": 100},
        )
        sr.run()

        # 8 = ococococ
        self.assertEqual(len(sr.books[0].transactions), 8)

    def test_positional_orders_notional(self):
        # test using notionals
        sr = StrategyRunner(
            data=self.df_combined,
            assets=self.assets,
            strat_classes=[TestPosOrderSizeStrat],
            strat_params={"size_type": OrderSizeType.NOTIONAL, "size_factor": 1000},
        )
        sr.run()

        # 8 = ococococ
        self.assertEqual(len(sr.books[0].transactions), 8)

    def test_positional_orders_book_percent(self):
        # test using book percent

        # books default to zero cash
        book = Book(name="Main", cash=Decimal("1000000"))

        sr = StrategyRunner(
            data=self.df_combined,
            assets=self.assets,
            strat_classes=[TestPosOrderSizeStrat],
            strat_params={
                "size_type": OrderSizeType.BOOK_PERCENT,
                "size_factor": 10000,
            },
            books=[book],
        )
        sr.run()

        # 8 = ococococ
        self.assertEqual(len(sr.books[0].transactions), 8)

    def test_spread_simple(self):
        strat_params = {
            "s1": "GOOG",
            "s2": "MSFT",
            "factor": 4.5,
        }
        sr = StrategyRunner(
            data=self.df_combined,
            assets=self.assets,
            strat_classes=[TestSpreadSimpleStrat],
            strat_params=strat_params,
        )
        sr.run()

        df_trades = pd.DataFrame(sr.books[0].transactions)
        self.assertEqual(len(df_trades), 6)
        self.assertEqual(len(df_trades.query("asset_name == 'GOOG'")), 3)
        self.assertEqual(len(df_trades.query("asset_name == 'MSFT'")), 3)

    def test_basket_order_quantity(self):
        # test using quantities
        sr = StrategyRunner(
            data=self.df_combined,
            assets=self.assets,
            strat_classes=[TestBasketOrderSizeStrat],
            strat_params={"size_type": OrderSizeType.QUANTITY},
        )
        sr.run()

        # 20 = 4 x ttttc
        self.assertEqual(len(sr.books[0].transactions), 20)

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
            assets=[Asset(name="ACME"), Asset(name="BOKO")],
            strat_classes=[TestOnOpenMaskStrat],
        )
        sr.run()

    def test_limit_order(self):
        class LimitOrder(Order):
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
                    assets=[Asset(name="ACME", denom="USD")],
                    strat_classes=[TestLimitOrderStrat],
                )
                sr.run()

                self.assertListEqual(op_status, [o.status for o in sr.orders_processed])
                self.assertListEqual(
                    ou_status, [o.status for o in sr.orders_unprocessed]
                )

    def test_stop_loss_order(self):
        class StopLossOrder(Order):
            def pre_execute_check(self, ts, tp):
                # if drops below 90 then complete stop order
                if tp < 90:
                    return None
                # otherwise leave open for another day
                return OrderStatus.OPEN

        class OrderWithStopLosses(Order):
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
                    assets=[Asset(name="ACME", denom="USD")],
                    strat_classes=[TestStopLossOrderStrat],
                )
                sr.run()

                self.assertListEqual(
                    op_status, [(o.status, o.label) for o in sr.orders_processed]
                )
                self.assertListEqual(
                    ou_status, [(o.status, o.label) for o in sr.orders_unprocessed]
                )

    def test_priority(self):
        class TestPriorityStrat(Strategy):
            def on_close(self):
                ix = self.data.index.get_loc(self.ts)
                if ix == 0:
                    self.orders.append(Order(asset_name="ACME", size=100, priority=2))
                    self.orders.append(Order(asset_name="ACME", size=200, priority=3))
                    self.orders.append(Order(asset_name="ACME", size=300, priority=1))

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
            assets=[Asset(name="ACME", denom="USD")],
            strat_classes=[TestPriorityStrat],
        )
        sr.run()

        self.assertListEqual(
            sr.transaction_history.quantity.to_list(),
            [Decimal("200.00"), Decimal("100.00"), Decimal("300.00")],
        )

    def test_order_key(self):
        class LimitOrder(Order):
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
            assets=[Asset(name="ACME", denom="USD")],
            strat_classes=[TestOrderkeyStrat],
        )
        sr.run()

        self.assertListEqual(
            [
                (OrderStatus.REPLACED, "my_key", Decimal("100")),
            ],
            [(o.status, o.key, o.size) for o in sr.orders_processed],
        )

        self.assertListEqual(
            [
                (OrderStatus.OPEN, None, Decimal("200")),
                (OrderStatus.OPEN, "my_key", Decimal("300")),
            ],
            [(o.status, o.key, o.size) for o in sr.orders_unprocessed],
        )


if __name__ == "__main__":
    unittest.main()
