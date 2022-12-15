# yabte - Yet Another BackTesting Engine

Python module for backtesting trading strategies.

Support event driven backtesting, ie `on_open`, `on_close`, etc. Also supports multiple assets.

Very basic statistics like book cash, mtm and total value. Currently, everything else needs to be deferred to a 3rd party module like `empyrical`.

There are some basic tests but use at your own peril. It's not production level code.

## Core dependencies

The core module uses pandas.

## Installation

```bash
pip install yatbe
```

## Usage

Below is an example usage (the performance of the example strategy won't be good).

```python
from pathlib import Path
import pandas as pd

import yabte
from yabte import Strategy, StrategyRunner, Order
from yabte.utils import crossover

data_dir = Path(yabte.__file__).parents[1] / "tests/data"


class SmokeStrat1(Strategy):
    def init(self):
        # enhance data with simple moving averages
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
        # create some orders
        data_2d = self.data.iloc[-2:]
        for sym in ["GOOG", "MSFT"]:
            data = data_2d[sym].loc[:, ("CloseSMA10", "CloseSMA20")].dropna()
            if len(data) == 2:
                if crossover(data.CloseSMA10, data.CloseSMA20):
                    self.orders.append(Order(asset_name=sym, size=100))
                elif crossover(data.CloseSMA20, data.CloseSMA10):
                    self.orders.append(Order(asset_name=sym, size=-100))


# load some data
asset_meta = {"GOOG": {"denom": "USD"}, "MSFT": {"denom": "USD"}}

df_goog = pd.read_csv(data_dir / "GOOG.csv", index_col=0, parse_dates=[0])
df_goog.columns = pd.MultiIndex.from_tuples([("GOOG", f) for f in df_goog.columns])

df_msft = pd.read_csv(data_dir / "MSFT.csv", index_col=0, parse_dates=[0])
df_msft.columns = pd.MultiIndex.from_tuples([("MSFT", f) for f in df_msft.columns])

# run our strategy
sr = StrategyRunner(
    data=pd.concat([df_goog, df_msft], axis=1),
    asset_meta=asset_meta,
    strats=[SmokeStrat1],
)
sr.run()

# see the trades or book history
th = sr.trade_history
bch = sr.book_history.loc[:, (slice(None), "cash")]
```