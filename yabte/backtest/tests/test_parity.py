# Setup basic logging to capture any output
import logging
import unittest
from decimal import Decimal

import numpy as np
import pandas as pd

from yabte.backtest import (
    Book,
    OHLCAsset,
    OrderSizeType,
    PositionalOrder,
    PositionalOrderCheckType,
    Strategy,
    StrategyRunner,
    Trade,
)

logging.basicConfig(level=logging.INFO)


class TestParity(unittest.TestCase):
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
        try:
            import _yabte_backtest_lib
        except ImportError:
            self.skipTest("C++ extension not built")

        res_py = self.run_strategy("python")
        res_cpp = self.run_strategy("cpp")

        # 1. Compare Book History
        hist_py = res_py.book_history
        hist_cpp = res_cpp.book_history

        # Align indices
        hist_cpp = hist_cpp.reindex(hist_py.index).fillna(0)

        # Check numeric columns
        for col in ["cash", "mtm", "total"]:
            if col not in hist_py.columns:
                continue

            py_vals = hist_py[col].values.astype(float)
            cpp_vals = hist_cpp[col].values.astype(float)

            np.testing.assert_allclose(
                py_vals,
                cpp_vals,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Mismatch in book history column {col}",
            )

        # 2. Compare Transactions
        tx_py = res_py.transaction_history.sort_values("ts").reset_index(drop=True)
        tx_cpp = res_cpp.transaction_history.sort_values("ts").reset_index(drop=True)

        self.assertEqual(len(tx_py), len(tx_cpp), "Transaction count mismatch")

        for i in range(len(tx_py)):
            t_py = tx_py.iloc[i]
            t_cpp = tx_cpp.iloc[i]

            self.assertEqual(t_py.ts, t_cpp.ts, f"Timestamp mismatch at index {i}")
            self.assertEqual(
                t_py.asset_name, t_cpp.asset_name, f"Asset mismatch at index {i}"
            )

            # Quantity Check
            self.assertTrue(
                np.isclose(
                    float(t_py.quantity), float(t_cpp.quantity), rtol=1e-5, atol=1e-5
                ),
                f"Quantity mismatch at {i}: {t_py.quantity} vs {t_cpp.quantity}",
            )

            # Price Check
            self.assertTrue(
                np.isclose(float(t_py.price), float(t_cpp.price), rtol=1e-5, atol=1e-5),
                f"Price mismatch at {i}: {t_py.price} vs {t_cpp.price}",
            )

            # Total Check (The main fix we implemented)
            # Use strict Decimal comparison if possible, or high precision float
            # Since we rounded to 2dp in C++ and Python formatted string, they should be very close
            # We treat them as floats for the assertion to allow minor epsilon if any,
            # but ideally they are exact string matches for 2dp.

            self.assertTrue(
                np.isclose(float(t_py.total), float(t_cpp.total), rtol=1e-5, atol=1e-5),
                f"Total mismatch at {i}: {t_py.total} vs {t_cpp.total}",
            )


if __name__ == "__main__":
    unittest.main()
