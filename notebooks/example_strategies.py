"""Strategies shared by the example notebooks.

They live in a module, not in a notebook cell, because
:py:meth:`yabte.backtest.StrategyRunner.run_batch` runs Python strategies in worker
processes, and a worker can only rebuild a strategy whose class it can import. A class
defined in a notebook can't be imported, so ``run_batch`` falls back to running
serially.
"""

import pandas as pd

from yabte.backtest import SimpleOrder, Strategy
from yabte.utilities.strategy_helpers import crossover


class SMAXO(Strategy):
    """Simple moving average crossover.

    Buys 100 when the short SMA crosses above the long, sells 100 when it crosses below.
    Parameters: ``days_short`` (default 10), ``days_long`` (default 20) and ``symbols``
    (default: every asset in the data).
    """

    def init(self):
        p = self.params
        days_short = p.get("days_short", 10)
        days_long = p.get("days_long", 20)
        self.symbols = p.get("symbols", None) or list(self.data.columns.levels[0])

        close = self.data.loc[:, (slice(None), "Close")]
        sma_short = (
            close.rolling(days_short)
            .mean()
            .rename({"Close": "CloseSMAShort"}, axis=1, level=1)
        )
        sma_long = (
            close.rolling(days_long)
            .mean()
            .rename({"Close": "CloseSMALong"}, axis=1, level=1)
        )
        self.data = pd.concat([self.data, sma_short, sma_long], axis=1).sort_index(
            axis=1
        )

    def on_close(self):
        for symbol in self.symbols:
            df = self.data[symbol]
            data = df.loc[df.index[-2:], ["CloseSMAShort", "CloseSMALong"]].dropna()
            if len(data) == 2:
                if crossover(data.CloseSMAShort, data.CloseSMALong):
                    self.orders.append(SimpleOrder(asset_name=symbol, size=100))
                elif crossover(data.CloseSMALong, data.CloseSMAShort):
                    self.orders.append(SimpleOrder(asset_name=symbol, size=-100))
