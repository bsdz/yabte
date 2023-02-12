import logging
import unittest
from decimal import Decimal

import numpy as np
import pandas as pd

from tests._helpers import generate_nasdaq_dataset
from yabte.backtest import (
    BasketOrder,
    Book,
    Order,
    OrderSizeType,
    PositionalBasketOrder,
    PositionalOrder,
    Strategy,
    StrategyRunner,
    OrderStatus,
)
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
        cls.asset_meta, cls.df_combined = generate_nasdaq_dataset()

    def test_sma_crossover(self):
        sr = StrategyRunner(
            data=self.df_combined, asset_meta=self.asset_meta, strats=[TestSMAXOStrat]
        )
        sr.run()

        th = sr.trade_history
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

    def test_positional_orders_quantity(self):
        # test using quantities
        sr = StrategyRunner(
            data=self.df_combined,
            asset_meta=self.asset_meta,
            strats=[TestPosOrderSizeStrat],
            strat_params={"size_type": OrderSizeType.QUANTITY, "size_factor": 100},
        )
        sr.run()

        # 8 = ococococ
        self.assertEqual(len(sr.books[0].trades), 8)

    def test_positional_orders_notional(self):
        # test using notionals
        sr = StrategyRunner(
            data=self.df_combined,
            asset_meta=self.asset_meta,
            strats=[TestPosOrderSizeStrat],
            strat_params={"size_type": OrderSizeType.NOTIONAL, "size_factor": 1000},
        )
        sr.run()

        # 8 = ococococ
        self.assertEqual(len(sr.books[0].trades), 8)

    def test_positional_orders_book_percent(self):
        # test using book percent

        # books default to zero cash
        book = Book(name="PrimaryBook", cash=Decimal("1000000"))

        sr = StrategyRunner(
            data=self.df_combined,
            asset_meta=self.asset_meta,
            strats=[TestPosOrderSizeStrat],
            strat_params={
                "size_type": OrderSizeType.BOOK_PERCENT,
                "size_factor": 10000,
            },
            books=[book],
        )
        sr.run()

        # 8 = ococococ
        self.assertEqual(len(sr.books[0].trades), 8)

    def test_spread_simple(self):
        strat_params = {
            "s1": "GOOG",
            "s2": "MSFT",
            "factor": 4.5,
        }
        sr = StrategyRunner(
            data=self.df_combined,
            asset_meta=self.asset_meta,
            strats=[TestSpreadSimpleStrat],
            strat_params=strat_params,
        )
        sr.run()

        df_trades = pd.DataFrame(sr.books[0].trades)
        self.assertEqual(len(df_trades), 6)
        self.assertEqual(len(df_trades.query("asset_name == 'GOOG'")), 3)
        self.assertEqual(len(df_trades.query("asset_name == 'MSFT'")), 3)

    def test_basket_order_quantity(self):
        # test using quantities
        sr = StrategyRunner(
            data=self.df_combined,
            asset_meta=self.asset_meta,
            strats=[TestBasketOrderSizeStrat],
            strat_params={"size_type": OrderSizeType.QUANTITY},
        )
        sr.run()

        # 20 = 4 x ttttc
        self.assertEqual(len(sr.books[0].trades), 20)
        pass

    def test_limit_order(self):
        class TestLimitOrderStrat(Strategy):
            def on_close(self):
                def my_limit_func(tp):
                    # if goes above 110 then cancel
                    if tp > 110:
                        return OrderStatus.CANCELLED
                    # if drops below 90 then compelete order
                    elif tp < 90:
                        return None
                    # otherwise leave open for another day
                    return OrderStatus.OPEN

                ix = self.data.index.get_loc(self.ts)
                if ix == 0:
                    self.orders.append(
                        Order(asset_name="ACME", size=100, exec_cond=my_limit_func)
                    )

        for data_arr, op_status, ou_status in [
            (
                [
                    [105, 105, 105, 105, 100],
                    [115, 115, 115, 115, 100],
                    [110, 110, 110, 110, 100],
                ],
                [OrderStatus.CANCELLED],
                [],
            ),
            (
                [
                    [95, 95, 95, 95, 100],
                    [100, 100, 100, 100, 100],
                    [105, 105, 105, 105, 100],
                ],
                [],
                [OrderStatus.OPEN],
            ),
            (
                [
                    [95, 95, 95, 95, 100],
                    [100, 100, 100, 100, 100],
                    [85, 85, 85, 85, 100],
                ],
                [OrderStatus.COMPLETE],
                [],
            ),
        ]:
            with self.subTest():
                data = pd.DataFrame(
                    data_arr,
                    columns=pd.MultiIndex.from_product(
                        [["ACME"], ["Open", "High", "Low", "Close", "Volume"]]
                    ),
                    index=pd.date_range(
                        start="20180102", periods=len(data_arr), freq="B"
                    ),
                )

                sr = StrategyRunner(
                    data=data,
                    asset_meta={"ACME": {"denom": "USD"}},
                    strats=[TestLimitOrderStrat],
                    # strat_params={"size_type": OrderSizeType.QUANTITY},
                )
                sr.run()

                self.assertListEqual(op_status, [o.status for o in sr.orders_processed])
                self.assertListEqual(
                    ou_status, [o.status for o in sr.orders_unprocessed]
                )


if __name__ == "__main__":
    unittest.main()
