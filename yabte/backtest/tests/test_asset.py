import unittest
from decimal import Decimal

import pandas as pd

from yabte.backtest import OHLCAsset, find_missing_data

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
