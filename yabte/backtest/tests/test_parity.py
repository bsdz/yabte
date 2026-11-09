# Setup basic logging to capture any output
import logging
import unittest
from decimal import Decimal

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
from yabte.backtest.tests._parity import (
    ParityAssertionsMixin,
    skip_or_fail_without_cpp,
)

logging.basicConfig(level=logging.INFO)


class TestParity(ParityAssertionsMixin, unittest.TestCase):
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
        skip_or_fail_without_cpp(self)
        res_py = self.run_strategy("python")
        res_cpp = self.run_strategy("cpp")
        self.assertResultParity(res_py, res_cpp, context="parity")


if __name__ == "__main__":
    unittest.main()
