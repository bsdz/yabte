import unittest
from decimal import Decimal

from yabte.backtest import OHLCAsset

# Check if C++ engine is available
try:
    import _yabte_backtest_lib as yabte_cpp_backtest

    HAS_CPP = True
except ImportError:
    HAS_CPP = False


class AssetTestCase(unittest.TestCase):
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
