import unittest
from decimal import Decimal

import numpy as np
import pandas as pd

from yabte.backtest import CostType, MissingPriceError, OHLCAsset, find_missing_data

# Check if C++ engine is available
from yabte.backtest.runner_cpp import yabte_cpp as yabte_cpp_backtest

HAS_CPP = yabte_cpp_backtest is not None


class AssetTestCase(unittest.TestCase):
    def test_fill_price(self):
        row = pd.Series({"Open": 10.0, "High": 14.0, "Low": 11.0, "Close": 13.0})
        for fill_price, expected in [
            ("midpoint", "12.5"),
            ("open", "10"),
            ("close", "13"),
        ]:
            with self.subTest(fill_price=fill_price):
                asset = OHLCAsset(name="A", fill_price=fill_price)
                self.assertEqual(asset.intraday_traded_price(row), Decimal(expected))
        with self.assertRaisesRegex(ValueError, "fill_price must be one of"):
            OHLCAsset(name="A", fill_price="vwap")  # type: ignore[arg-type]

    def test_transaction_cost(self):
        for cost_type, cost, expected in [
            (CostType.BPS, "10", "0.5"),
            (CostType.PER_UNIT, "0.65", "3.25"),
            (CostType.PER_ORDER, "1", "1"),
        ]:
            with self.subTest(cost_type=cost_type):
                asset = OHLCAsset(name="A", cost=cost, cost_type=cost_type)
                self.assertEqual(
                    asset.transaction_cost(Decimal(-5), Decimal(100)),
                    Decimal(expected),
                )
                self.assertEqual(asset.transaction_cost(Decimal(0), Decimal(100)), 0)
        self.assertEqual(
            OHLCAsset(name="A").transaction_cost(Decimal(5), Decimal(1)), 0
        )
        # like OrderSizeType, the enum can be given by name
        self.assertEqual(
            OHLCAsset(name="A", cost_type="per_unit").cost_type,  # type: ignore[arg-type]
            CostType.PER_UNIT,
        )
        with self.assertRaisesRegex(ValueError, "cost must not be negative"):
            OHLCAsset(name="A", cost=-1)

    def test_data_cost(self):
        ts = pd.Timestamp("2025-01-06")
        row = pd.Series({"HalfSpread": 0.05, "Gap": np.nan, "Bad": -1.0})
        q, p = Decimal(-4), Decimal(100)
        for field_type, expected in [
            (CostType.PER_UNIT, Decimal("0.2")),
            (CostType.BPS, Decimal("0.002")),
            (CostType.PER_ORDER, Decimal("0.05")),
        ]:
            with self.subTest(field_type=field_type):
                asset = OHLCAsset(
                    name="A", cost_field="HalfSpread", cost_field_type=field_type
                )
                self.assertEqual(asset.data_cost(q, p, row, ts), expected)
        self.assertEqual(OHLCAsset(name="A").data_cost(q, p, row, ts), 0)
        # nothing traded: the value isn't read
        gap = OHLCAsset(name="A", cost_field="Gap")
        self.assertEqual(gap.data_cost(Decimal(0), p, row, ts), 0)
        for field in ("Gap", "Missing"):
            with self.assertRaisesRegex(MissingPriceError, f"No {field} value for A"):
                OHLCAsset(name="A", cost_field=field).data_cost(q, p, row, ts)
        with self.assertRaisesRegex(ValueError, "Bad for A on .* is negative"):
            OHLCAsset(name="A", cost_field="Bad").data_cost(q, p, row, ts)
        self.assertEqual(
            OHLCAsset(name="A", cost_field_type="bps").cost_field_type,  # type: ignore[arg-type]
            CostType.BPS,
        )

    def test_open_fill_requires_open(self):
        data = pd.DataFrame(
            {("A", "Close"): [1.0, 2.0]}, index=pd.bdate_range("2025-01-06", periods=2)
        )
        self.assertTrue(find_missing_data(data, [OHLCAsset(name="A")]).empty)
        missing = find_missing_data(data, [OHLCAsset(name="A", fill_price="open")])
        self.assertEqual(set(missing.field), {"Open"})
        with self.assertRaisesRegex(ValueError, "missing {'Open'}"):
            OHLCAsset(name="A", fill_price="open").check_and_fix_data(data["A"])

    def test_round_quantity(self):
        # 1. Test Python Asset
        py_asset = OHLCAsset(name="TEST", quantity_round_dp=2)

        # Test basic rounding
        self.assertEqual(py_asset.round_quantity(1.234), 1.23)
        self.assertEqual(py_asset.round_quantity(1.236), 1.24)

        # Test Banker's Rounding with exact binary fractions
        # 0.125 -> 0.12 (even)
        self.assertEqual(py_asset.round_quantity(0.125), 0.12)
        # 0.375 -> 0.38 (even)
        self.assertEqual(py_asset.round_quantity(0.375), 0.38)

        # Test the problematic case
        # 9.255
        # In Python: 9.255 is approx 9.254999999999999. round(9.255, 2) -> 9.25
        print(f"Python round(9.255, 2) = {round(9.255, 2)}")

        # 2. Test C++ Asset (if available)
        if HAS_CPP:
            cpp_asset = yabte_cpp_backtest.OHLCAsset("TEST", "USD", 2, 2)

            # Helper to compare
            def check_cpp(val, expected):
                # C++ returns double, python expected is Decimal or string
                res = cpp_asset.round_quantity(val)
                # Compare with small epsilon for float equality
                self.assertAlmostEqual(
                    res,
                    float(expected),
                    places=10,
                    msg=f"C++ round({val}) mismatch. Got {res}, expected {expected}",
                )

            check_cpp(1.234, 1.23)
            check_cpp(1.236, 1.24)

            # Banker's Rounding
            check_cpp(0.125, 0.12)
            check_cpp(0.375, 0.38)

            # Check parity for 9.255
            # If Python gives 9.25, C++ should too
            py_res = round(9.255, 2)
            check_cpp(9.255, py_res)

            # Check parity for 1.225 (which gave 1.23 in Python)
            py_res_2 = round(1.225, 2)  # 1.23
            check_cpp(1.225, py_res_2)


if __name__ == "__main__":
    unittest.main()
