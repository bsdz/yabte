# yabte - Yet Another BackTesting Engine

Python module for backtesting trading strategies.

Support event driven backtesting, ie `on_open`, `on_close`, etc. Also supports multiple assets.

Very basic statistics like book cash, mtm and total value. Currently, everything else needs to be deferred to a 3rd party module like `empyrical`.

There are some basic tests but use at your own peril. It's not production level code.

## Core dependencies

The core module uses pandas and scipy.

## Installation

```bash
pip install yatbe
```

## Usage

Below is an example usage (the performance of the example strategy won't be good).

```python
import pandas as pd

from yabte.backtest import Book, SimpleOrder, Strategy, StrategyRunner
from yabte.tests._helpers import generate_nasdaq_dataset
from yabte.utilities.plot.plotly.strategy_runner import plot_strategy_runner_result
from yabte.utilities.strategy_helpers import crossover


class SMAXO(Strategy):
    def init(self):
        # enhance data with simple moving averages

        p = self.params
        days_short = p.get("days_short", 10)
        days_long = p.get("days_long", 20)

        close_sma_short = (
            self.data.loc[:, (slice(None), "Close")]
            .rolling(days_short)
            .mean()
            .rename({"Close": "CloseSMAShort"}, axis=1, level=1)
        )
        close_sma_long = (
            self.data.loc[:, (slice(None), "Close")]
            .rolling(days_long)
            .mean()
            .rename({"Close": "CloseSMALong"}, axis=1, level=1)
        )
        self.data = pd.concat(
            [self.data, close_sma_short, close_sma_long], axis=1
        ).sort_index(axis=1)

    def on_close(self):
        # create some orders

        for symbol in ["GOOG", "MSFT"]:
            df = self.data[symbol]
            ix_2d = df.index[-2:]
            data = df.loc[ix_2d, ("CloseSMAShort", "CloseSMALong")].dropna()
            if len(data) == 2:
                if crossover(data.CloseSMAShort, data.CloseSMALong):
                    self.orders.append(SimpleOrder(asset_name=symbol, size=-100))
                elif crossover(data.CloseSMALong, data.CloseSMAShort):
                    self.orders.append(SimpleOrder(asset_name=symbol, size=100))


# load some data
assets, df_combined = generate_nasdaq_dataset()

# create a book with 100000 cash
book = Book(name="Main", cash="100000")

# run our strategy
sr = StrategyRunner(
    data=df_combined,
    assets=assets,
    strategies=[SMAXO()],
    books=[book],
)
srr = sr.run()

# see the trades or book history
th = srr.transaction_history
bch = srr.book_history.loc[:, (slice(None), "cash")]

# plot the trades against book value
plot_strategy_runner_result(srr, sr)
```

![Output from code](https://raw.githubusercontent.com/bsdz/yabte/main/readme_image.png)

## Examples

Jupyter notebook examples can be found under the [notebooks folder](https://github.com/bsdz/yabte/tree/main/notebooks).

## Documentation

Documentation can be found on [Read the Docs](https://yabte.readthedocs.io/en/latest/).


## Development

Before commit run following format commands in project folder:

```bash
poetry run black .
poetry run isort . --profile black
poetry run docformatter . --recursive --in-place --black --exclude _unittest_numpy_extensions.py
```
