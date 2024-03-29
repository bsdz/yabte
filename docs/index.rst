yabte - Yet Another Backtesting Engine
======================================

yabte is an event-driven trade backtesting library supporting multiple assets.
It supports Panda's dataframes for input data and is easily customisable for other formats.
It includes utilities for portfolio optimisation and data simulation. 


Installation
============

The package is kept on PyPi and can be installed via pip:

.. code-block:: text

   pip install yatbe

or alternatively using poetry:

.. code-block:: text

   poetry add yabte

Example
=======

A typical strategy looks as follows:

.. code-block:: python

   from yabte.backtest import Strategy, StrategyRunner, SimpleOrder, Book, OHLCAsset
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
         # This is called on open with self.data containing all
         # historical data until this time. Any orders in
         # this method will be executed before the on_close
         # method is called.
         pass

      def on_close(self):
         # Call on close of trading.
         for symbol in ["GOOG", "MSFT"]:
               df = self.data[symbol]
               ix_2d = df.index[-2:]
               data = df.loc[ix_2d, ("CloseSMAShort", "CloseSMALong")].dropna()
               if len(data) == 2:
                  if crossover(data.CloseSMAShort, data.CloseSMALong):
                     # Appending orders to the self.orders queue will be
                     # executed the next day. There are different types
                     # of order, e.g. positional and basket and some orders
                     # support hooks for running arbitrary checks for limits etc.
                     self.orders.append(SimpleOrder(asset_name=symbol, size=-100))
                  elif crossover(data.CloseSMALong, data.CloseSMAShort):
                     self.orders.append(SimpleOrder(asset_name=symbol, size=100))


To run our strategy, we use a StrategyRunner:

.. code-block:: python

   # The OHLCAsset class is any pricable object with accompanying High, Low, Close,
   # Open & Volume field data. One can subclass OHLCAsset to support different field
   # types, e.g. Volatility.
   assets = [OHLCAsset(name="GOOG", denom="USD"), OHLCAsset(name="MSFT", denom="USD")]

   sr = StrategyRunner(
      # Data is provided as a pandas dataframe with two level
      # column index (asset name & field) and single level row
      # index containing timestamps.
      data=pd.concat([df_goog, df_msft], axis=1),
      assets=assets,
      strat_classes=[SMAXO],
      books=[book],
   )
   sr.run()


Once a runner has completed we can access book value history and transaction history
via our instance:

.. code-block:: python

   th = sr.transaction_history
   bch = sr.book_history


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   yabte.backtest
   yabte.utilities.portopt
   yabte.utilities.simulation

Thumbnails gallery
==================

.. nbgallery::
   notebooks/Portfolio_Optimization
   notebooks/Delta_Hedging

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
