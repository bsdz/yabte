yabte - Yet Another Backtesting Engine
======================================

yabte is an event-driven backtesting library for trading strategies over multiple
assets and books. Input data is a pandas DataFrame of daily or intraday bars, and
custom asset classes can support other data fields. A C++ engine runs the same
strategies faster, and parameter sweeps run in parallel. It also includes utilities
for portfolio optimisation and data simulation.


Installation
============

The package is on PyPI and can be installed with pip:

.. code-block:: text

   pip install yabte

Wheels include the C++ engine for CPython 3.12 to 3.14 on Linux x86_64. A source
install tries to build it and falls back to pure Python if that fails; set
``YABTE_BUILD_CPP=0`` to skip the build.

Example
=======

A typical strategy looks as follows:

.. code-block:: python

   import pandas as pd

   from yabte.backtest import Book, SimpleOrder, Strategy, StrategyRunner
   from yabte.tests._helpers import generate_nasdaq_dataset
   from yabte.utilities.strategy_helpers import crossover


   # All strategies inherit from the Strategy base class.
   class SMAXO(Strategy):

       # This method is called before any trading commences.
       def init(self):
           # Parameters are passed to this strategy via self.params.
           p = self.params
           days_short = p.get("days_short", 10)
           days_long = p.get("days_long", 20)

           # Data is passed in via self.data.
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

           # During initialisation one can update self.data.
           self.data = pd.concat(
               [self.data, close_sma_short, close_sma_long], axis=1
           ).sort_index(axis=1)

       def on_open(self):
           # Called at the open of each bar. self.data holds all data up to now,
           # with this bar's high, low and close hidden. Orders placed here
           # execute on this bar.
           pass

       def on_close(self):
           # Called at the close of each bar, with the whole bar visible.
           for symbol in ["GOOG", "MSFT"]:
               df = self.data[symbol]
               ix_2d = df.index[-2:]
               data = df.loc[ix_2d, ("CloseSMAShort", "CloseSMALong")].dropna()
               if len(data) == 2:
                   if crossover(data.CloseSMAShort, data.CloseSMALong):
                       # Orders appended to self.orders here execute on the
                       # next bar. There are other order types, e.g. positional
                       # and basket, and some support hooks for running checks
                       # such as limits.
                       self.orders.append(SimpleOrder(asset_name=symbol, size=-100))
                   elif crossover(data.CloseSMALong, data.CloseSMAShort):
                       self.orders.append(SimpleOrder(asset_name=symbol, size=100))


To run our strategy, we use a StrategyRunner:

.. code-block:: python

   # Sample Nasdaq data: a list of OHLCAsset, and a DataFrame with a two level
   # column index (asset name and field) and a timestamp row index. OHLCAsset
   # prices anything with Open, High, Low, Close and Volume fields; subclass
   # Asset to support other fields, e.g. Volatility.
   assets, df_combined = generate_nasdaq_dataset()

   sr = StrategyRunner(
       data=df_combined,
       assets=assets,
       strategies=[SMAXO()],
       books=[Book(name="Main", cash="100000")],
   )
   srr = sr.run()

Pass ``engine="cpp"`` to ``StrategyRunner`` to use the C++ engine. Once a runner has
completed we can access the book value history and transaction history from the
result:

.. code-block:: python

   th = srr.transaction_history
   bch = srr.book_history


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   yabte.backtest
   yabte.utilities
   yabte.utilities.portopt
   yabte.utilities.simulation

Thumbnails gallery
==================

.. nbgallery::
   notebooks/Readme_Example
   notebooks/Engines_and_Parameter_Sweeps
   notebooks/Orders_and_Assets
   notebooks/Books_Interest_and_Mandates
   notebooks/Intraday_Bars
   notebooks/Simulation
   notebooks/Portfolio_Optimization
   notebooks/Delta_Hedging
   notebooks/Pyfolio_Example

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
