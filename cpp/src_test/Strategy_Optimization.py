#!/usr/bin/env python
# coding: utf-8

# In[1]:


from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa

import yabte
import yabte.backtest as ybt
import yabte.utilities.pandas_extension
from yabte.tests._helpers import generate_nasdaq_dataset
from yabte.utilities.strategy_helpers import crossover

yabte.__version__


# In[2]:


# load some data
ybt_assets, df_combined = generate_nasdaq_dataset()

# convert to arrow table
ar_combined = pa.Table.from_pandas(df_combined)


# In[3]:


param_iter = [
    {"days_long": n, "days_short": m}
    for n, m in zip([20, 30, 40, 50], [5, 10, 15, 20])
    if n > m
]


# In[4]:


class SMAXO_YBT(ybt.Strategy):
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

        # for symbol in ["GOOG", "MSFT"]:
        for symbol in ["GOOG"]:
            df = self.data[symbol]
            ix_2d = df.index[-2:]
            data = df.loc[ix_2d, ("CloseSMAShort", "CloseSMALong")].dropna()
            if len(data) == 2:
                if crossover(data.CloseSMAShort, data.CloseSMALong):
                    self.orders.append(ybt.SimpleOrder(asset_name=symbol, size=-100))
                elif crossover(data.CloseSMALong, data.CloseSMAShort):
                    self.orders.append(ybt.SimpleOrder(asset_name=symbol, size=100))


# In[5]:


# create a book with 100000 cash
book = ybt.Book(name="Main", cash="100000")

# run our strategy
sr2 = ybt.StrategyRunner(
    data=df_combined,
    assets=ybt_assets,
    strategies=[SMAXO_YBT()],
    books=[book],
)
# srrs2 = sr2.run_batch(param_iter)


# In[6]:


# [srr.book_history.loc[:, (slice(None), "total")].prc.sharpe_ratio() for srr in srrs2]


# In[7]:


import sys

sys.path.append("/home/blair/projects/yabte/cpp/build/Release/pybind")
import yabte_cpp_backtest as ybt_cpp

ybt_cpp.__version__


# In[8]:


class SMAXO_YBTCPP(ybt_cpp.Strategy):
    def __init__(self):
        ybt_cpp.Strategy.__init__(self)

    def clone(self):
        cloned = SMAXO_YBTCPP.__new__(SMAXO_YBTCPP)
        ybt_cpp.Strategy.__init__(cloned, self)
        cloned.__dict__.update(self.__dict__)
        return cloned

    def extend_data(self, data):
        # enhance data with simple moving averages

        df = data.to_pandas()

        p = self.params
        days_short = p["days_short"]
        days_long = p["days_long"]

        close_sma_short = (
            df.loc[:, (slice(None), "Close")]
            .rolling(days_short)
            .mean()
            .rename({"Close": "CloseSMAShort"}, axis=1, level=1)
        )
        close_sma_long = (
            df.loc[:, (slice(None), "Close")]
            .rolling(days_long)
            .mean()
            .rename({"Close": "CloseSMALong"}, axis=1, level=1)
        )
        new_df = pd.concat([close_sma_short, close_sma_long], axis=1).sort_index(axis=1)

        return pa.Table.from_pandas(new_df)

    def init(self): ...

    def on_close(self):
        # create some orders

        p = self.params
        days_long = p["days_long"]

        if (L := len(self.data)) < days_long + 1:
            return

        # for symbol in ["GOOG", "MSFT"]:
        for symbol in ["GOOG"]:
            key_long = f"('{symbol}', 'CloseSMALong')"
            key_short = f"('{symbol}', 'CloseSMAShort')"

            close_sma_short = self.data[key_short].slice(L - 2, 2).to_pandas()
            close_sma_long = self.data[key_long].slice(L - 2, 2).to_pandas()

            if crossover(close_sma_short, close_sma_long):
                self.orders.append(ybt_cpp.SimpleOrder(asset_name=symbol, size=-100))
            elif crossover(close_sma_long, close_sma_short):
                self.orders.append(ybt_cpp.SimpleOrder(asset_name=symbol, size=100))


# In[ ]:


# recreate cpp assets from python assets
ybtcpp_assets = [ybt_cpp.OHLCAsset(name=a.name) for a in ybt_assets]

# create a strategy
strategies = [SMAXO_YBTCPP()]

# create a book with 100000 cash
books = [ybt_cpp.Book(name="Main", cash=100000)]

# run the strategy
sr1 = ybt_cpp.StrategyRunner(ar_combined, ybtcpp_assets, strategies, books)
srrs1 = sr1.run_batch(param_iter, 2)


# In[ ]:
