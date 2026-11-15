import unittest
from decimal import Decimal

import numpy as np
import pandas as pd

from yabte.backtest import (
    Book,
    MissingPriceError,
    OHLCAsset,
    OrderSizeType,
    PositionalOrder,
    SimpleOrder,
    Strategy,
    StrategyRunner,
    find_missing_data,
)
from yabte.backtest.tests._parity import skip_or_fail_without_cpp


class PlannedOrders(Strategy):
    """Places `plan[day]` orders at the open of each (1-based) day."""

    plan: dict = {}

    def on_open(self):
        for order in self.plan.get(len(self.data), []):
            self.orders.append(order())


def make_data():
    """Two assets over five days; ACME has no prices on day 3 (2024-01-03)."""
    ix = pd.date_range("2024-01-01", periods=5, freq="D")
    prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    data = {}
    for name in ["ACME", "OTHER"]:
        for field in ["Open", "High", "Low", "Close"]:
            data[(name, field)] = prices.copy()
    df = pd.DataFrame(data, index=ix)
    df.loc["2024-01-03", "ACME"] = np.nan
    return df


class MissingPricesPythonTestCase(unittest.TestCase):
    engine = "python"

    def run_plan(self, plan):
        return StrategyRunner(
            data=make_data(),
            assets=[OHLCAsset(name="ACME"), OHLCAsset(name="OTHER")],
            # a subclass, as the C++ engine only takes scalar params
            strategies=[type("Plan", (PlannedOrders,), {"plan": plan})()],
            books=[Book(name="Main", cash=Decimal("1000"))],
            engine=self.engine,
        ).run()

    def assertMissingPrice(self, plan, kind):
        with self.assertRaises(MissingPriceError) as cm:
            self.run_plan(plan)
        self.assertIsInstance(cm.exception, ValueError)
        self.assertIn(
            f"No {kind} price for ACME on 2024-01-03 00:00:00", str(cm.exception)
        )

    def test_order_executing_without_a_price_raises(self):
        self.assertMissingPrice(
            {3: [lambda: SimpleOrder(asset_name="ACME", size=1)]}, "traded"
        )

    def test_notional_order_without_a_price_raises(self):
        order = lambda: SimpleOrder(
            asset_name="ACME", size=100, size_type=OrderSizeType.NOTIONAL
        )
        self.assertMissingPrice({3: [order]}, "traded")

    def test_positional_order_without_a_price_raises(self):
        self.assertMissingPrice(
            {3: [lambda: PositionalOrder(asset_name="ACME", size=1)]}, "traded"
        )

    def test_holding_an_asset_without_a_close_raises(self):
        self.assertMissingPrice(
            {1: [lambda: SimpleOrder(asset_name="ACME", size=1)]}, "end of day"
        )

    def test_gaps_in_unused_prices_are_ignored(self):
        """A flat position, or an asset never traded, needs no price."""
        srr = self.run_plan(
            {
                1: [
                    lambda: SimpleOrder(asset_name="ACME", size=1),
                    lambda: SimpleOrder(asset_name="OTHER", size=1),
                ],
                2: [lambda: SimpleOrder(asset_name="ACME", size=-1)],
                4: [lambda: SimpleOrder(asset_name="ACME", size=1)],
            }
        )
        self.assertEqual(len(srr.transaction_history), 4)
        self.assertFalse(srr.book_history.isna().any().any())


class MissingPricesCppTestCase(MissingPricesPythonTestCase):
    engine = "cpp"

    def setUp(self):
        skip_or_fail_without_cpp(self)


class FindMissingDataTestCase(unittest.TestCase):
    def test_lists_missing_required_fields(self):
        data = make_data()
        data.loc["2024-01-05", ("OTHER", "Close")] = np.nan
        missing = find_missing_data(
            data, [OHLCAsset(name="ACME"), OHLCAsset(name="OTHER")]
        )
        self.assertEqual(
            list(missing.itertuples(index=False, name=None)),
            [
                (pd.Timestamp("2024-01-03"), "ACME", "Close"),
                (pd.Timestamp("2024-01-05"), "OTHER", "Close"),
            ],
        )

    def test_uses_data_label_and_reports_missing_columns(self):
        data = make_data().drop(columns=[("OTHER", "Close")])
        missing = find_missing_data(
            data,
            [OHLCAsset(name="ACME2", data_label="ACME"), OHLCAsset(name="OTHER")],
        )
        self.assertEqual(
            missing.groupby("asset_name").size().to_dict(),
            {
                "ACME2": 1,
                "OTHER": 5,
            },
        )

    def test_nothing_missing(self):
        data = make_data().fillna(1.0)
        missing = find_missing_data(data, [OHLCAsset(name="ACME")])
        self.assertTrue(missing.empty)
        self.assertEqual(list(missing.columns), ["ts", "asset_name", "field"])


if __name__ == "__main__":
    unittest.main()
