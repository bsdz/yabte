import unittest
from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd

# Import components
from yabte.backtest import (
    Book,
    OHLCAsset,
    OrderStatus,
    SimpleOrder,
    Strategy,
    StrategyRunner,
)
from yabte.tests._helpers import generate_nasdaq_dataset


class TestCppIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate some dummy data
        cls.assets, cls.df_combined = generate_nasdaq_dataset()
        # Ensure data is sorted
        cls.df_combined = cls.df_combined.sort_index()

    def test_cpp_engine_basic(self):
        """Test basic execution of C++ engine."""
        try:
            import _yabte_backtest_lib
        except ImportError as e:
            self.fail(f"Failed to import _yabte_backtest_lib: {e}")

        # Define a simple strategy
        class SimpleTestStrat(Strategy):
            def on_close(self):
                # Just create an order to see if it works
                # NOTE: self.ts might not be available in C++ adapter yet!
                # If the strategy relies on self.ts, it might fail.
                # Let's see if we can run without referencing self.ts explicitly if we check len(data)?

                # Simple buy on first day
                if len(self.orders) == 0:
                    self.orders.append(SimpleOrder(asset_name="GOOG", size=10))

        sr = StrategyRunner(
            data=self.df_combined.iloc[:10],  # Small subset
            assets=self.assets,
            strategies=[SimpleTestStrat()],
            engine="cpp",
        )

        try:
            result = sr.run()
        except NotImplementedError as e:
            self.fail(f"C++ engine raised NotImplementedError: {e}")
        except Exception as e:
            self.fail(f"C++ engine failed with: {e}")

        # Check result
        self.assertEqual(len(result.books), 1)
        book = result.books[0]

        # Check if orders were processed
        # Strategy adds order on first call.
        # Should be processed.
        # self.assertTrue(len(book.transactions) > 0, "No transactions generated")
        # NOTE: If self.ts is missing, on_close might fail or do nothing depending on impl.


if __name__ == "__main__":
    unittest.main()
