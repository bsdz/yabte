import unittest
from pathlib import Path
import logging
from copy import deepcopy
from decimal import Decimal

import numpy as np
import pandas as pd

from yabte import Strategy, StrategyRunner, Order, PositionalOrder, OrderSizeType, Book
from yabte.utils import crossover

logger = logging.getLogger(__name__)

data_dir = Path(__file__).parent / "data"


class SmokeStrat1(Strategy):
    def init(self):
        csma10 = (
            self.data.loc[:, (slice(None), "Close")]
            .rolling(10)
            .mean()
            .rename({"Close": "CloseSMA10"}, axis=1, level=1)
        )
        csma20 = (
            self.data.loc[:, (slice(None), "Close")]
            .rolling(20)
            .mean()
            .rename({"Close": "CloseSMA20"}, axis=1, level=1)
        )
        self.data = pd.concat([self.data, csma10, csma20], axis=1).sort_index(axis=1)

    def on_close(self):
        df_goog = self.data["GOOG"]
        ix_2d = df_goog.index[-2:]
        data = df_goog.loc[ix_2d, ("CloseSMA10", "CloseSMA20")].dropna()
        if len(data) == 2:
            if crossover(data.CloseSMA10, data.CloseSMA20):
                self.orders.append(Order(asset_name="GOOG", size=100))
            elif crossover(data.CloseSMA20, data.CloseSMA10):
                self.orders.append(Order(asset_name="GOOG", size=-100))


class XOStrat1(Strategy):
    def on_close(self):
        df_goog = self.data["GOOG"]
        ix = df_goog.index.get_loc(self.ts)
        if ix in [100, 201, 300, 401]:
            quantity = 100 * (-1) ** ix
            self.orders.append(PositionalOrder(asset_name="GOOG", size=quantity))
        elif ix in [1000]:
            self.orders.append(PositionalOrder(asset_name="GOOG", size=0))


class XOStrat2(Strategy):
    def on_close(self):
        df_goog = self.data["GOOG"]
        ix = df_goog.index.get_loc(self.ts)
        if ix in [100, 201, 300, 401]:
            notional = 10000 * (-1) ** ix
            self.orders.append(PositionalOrder(asset_name="GOOG", size=notional, size_type=OrderSizeType.NOTIONAL))
        elif ix in [1000]:
            self.orders.append(PositionalOrder(asset_name="GOOG", size=0))

class XOStrat3(Strategy):
    def on_close(self):
        df_goog = self.data["GOOG"]
        ix = df_goog.index.get_loc(self.ts)
        if ix in [100, 201, 300, 401]:
            notional = 10000 * (-1) ** ix
            self.orders.append(PositionalOrder(asset_name="GOOG", size="100", size_type=OrderSizeType.BOOK_PERCENT))
        elif ix in [1000]:
            self.orders.append(PositionalOrder(asset_name="GOOG", size=0))

class SpreadStrat1(Strategy):
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
            self.orders.append(
                PositionalOrder(asset_name=p.s2, size=-p.factor * 100)
            )
        elif abs(s) < 0.1 * self.sigma:
            self.orders.append(PositionalOrder(asset_name=p.s1, size=0))
            self.orders.append(PositionalOrder(asset_name=p.s2, size=0))


class StrategyRunnerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asset_meta = {}
        dfs = []
        for csv_pth in (data_dir / "nasdaq").glob("*.csv"):
            name = csv_pth.stem
            asset_meta[name] = {"denom": "USD"}
            df = pd.read_csv(csv_pth, index_col=0, parse_dates=[0])
            df.columns = pd.MultiIndex.from_tuples(
                [(name, f) for f in df.columns]
            )
            dfs.append(df)

        cls.asset_meta = asset_meta
        cls.df_combined = pd.concat(dfs, axis=1)

    def test_smoke(self):
        sr = StrategyRunner(
            data=self.df_combined, asset_meta=self.asset_meta, strats=[SmokeStrat1]
        )
        sr.run()

        th = sr.trade_history
        th["nc"] = - th.quantity * th.price
        bch = th.pivot_table(index="ts", columns="book", values="nc", aggfunc="sum").cumsum().reindex(sr.data.index).fillna(method="ffill").fillna(0) 
        self.assertTrue(np.all(np.isclose(bch.astype("float64"), sr.book_history.loc[:, (slice(None), "cash")].droplevel(axis=1, level=1))))

    def test_exclusive_orders_quantity(self):
        # test using quantities
        sr = StrategyRunner(
            data=self.df_combined, asset_meta=self.asset_meta, strats=[XOStrat1]
        )
        sr.run()

        self.assertEqual(len(sr.books[0].trades), 8)

    def test_exclusive_orders_notional(self):
        # test using notionals
        sr = StrategyRunner(
            data=self.df_combined, asset_meta=self.asset_meta, strats=[XOStrat2]
        )
        sr.run()

        self.assertEqual(len(sr.books[0].trades), 8)

    def test_exclusive_orders_book_percent(self):
        # test using book percent

        # books default to zero cash
        book = Book(name="PrimaryBook", cash=Decimal("1000000"))

        sr = StrategyRunner(
            data=self.df_combined, asset_meta=self.asset_meta, strats=[XOStrat3], books=[book]
        )
        sr.run()

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
            strats=[SpreadStrat1],
            strat_params=strat_params,
        )
        sr.run()

        df_trades = pd.DataFrame(sr.books[0].trades)
        self.assertEqual(len(df_trades), 6)
        self.assertEqual(len(df_trades.query("asset_name == 'GOOG'")), 3)
        self.assertEqual(len(df_trades.query("asset_name == 'MSFT'")), 3)


if __name__ == "__main__":
    unittest.main()
