import logging
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd

from .asset import Asset, AssetName
from .book import Book, BookMandate
from .order import Order, OrderStatus

logger = logging.getLogger(__name__)

__all__ = ["Strategy", "StrategyRunner"]


class Orders(deque):
    pass


@dataclass(kw_only=True)
class Strategy:
    """Trading strategy base class."""

    orders: Orders
    params: pd.Series
    books: List[Book]  # TODO: wrap this to make read only

    _ts = None
    _data_lock = True
    _ts_lock = True
    _mask_hlc = False

    @property
    def ts(self):
        """Stores the current timestamp."""
        return self._ts

    @ts.setter
    def ts(self, value: pd.Timestamp):
        if self._ts_lock:
            raise RuntimeError("Attempt to write ts")
        self._ts = value

    @property
    def data(self) -> pd.DataFrame:
        """Provides window of data available up to current
        timestamp `self.ts` and masks out data not availble
        at open (like high, low, close, volume)."""
        if not self.ts:
            return self._data
        else:
            df_t = self._data.loc[: self.ts, :]
            if not self._mask_hlc:
                data = df_t
            else:
                row_indexer = df_t.index == df_t.index[-1]
                col_indexer = df_t.columns.isin(
                    ["High", "Low", "Close", "Volume"], level=1
                )
                mask = row_indexer[:, None] & col_indexer
                data = df_t.mask(mask)
            return data

    @data.setter
    def data(self, value: pd.DataFrame):
        if self._data_lock:
            raise RuntimeError("Attempt to write data post init")
        self._data = value

    def init(self):
        """Initialise internal variables & enhance data for strategy."""
        pass

    def on_open(self):
        """Executed on open every day. Use `self.ts` to determine
        current timestamp. Data available at this timestamp is
        accessible from `self.data`."""
        pass

    def on_close(self):
        """Executed on close every day. Use `self.ts` to determine
        current timestamp. Data available at this timestamp is
        accessible from `self.data`."""
        pass


@dataclass(kw_only=True)
class StrategyRunner:
    """Encapsulates the execution of multiple strategies.

    Orders are captured in `orders_processed` and `orders_unprocessed`.
    `books` is a list of books and if none provided a single book is
    created call 'PrimaryBook'. After execution summary book and trade
    histories are captured in `book_history` and `trade_history`."""

    data: pd.DataFrame = field()
    asset_meta: Dict[AssetName, Dict[str, Any]]
    strats: List[Type[Strategy]]  # TODO: rename to strat_classes
    mandates: Dict[AssetName, BookMandate] = field(default_factory=dict)
    strat_params: Dict[str, Any] = field(default_factory=dict)

    orders_unprocessed: Orders = field(default_factory=Orders)
    orders_processed: List[Order] = field(default_factory=list)
    books: List[Book] = field(default_factory=list)
    strategies: List[Strategy] = field(default_factory=list)

    book_history: Optional[pd.DataFrame] = None
    trade_history: Optional[pd.DataFrame] = None

    def __post_init__(self):
        # check data structure correct
        df = self.data
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("data index must be a datetimeindex")
        if not df.index.is_monotonic_increasing:
            raise ValueError("data needs to have increasing index")
        # colum level 1 = asset, level 2 = field
        if not isinstance(df.columns, pd.MultiIndex):
            raise ValueError("data columns must be multindex asset/field")
        if len(df.columns.levels) != 2:
            raise ValueError("data columns multiindex must have 2 levels")
        required_fields = ["High", "Low", "Open", "Close", "Volume"]
        # check each asset as required fields
        if not np.all(
            pd.MultiIndex.from_tuples(
                product(df.columns.levels[0], required_fields)
            ).isin(df.columns)
        ):
            raise ValueError("data columns multiindex requires fields HLOCV")
        # TODO: check assets match asset_meta

        # set up books
        if not self.books:
            self.books = [Book(name="PrimaryBook", mandates=self.mandates)]

    def run(self):
        """Execute each strategy through time."""

        # TODO: make available where necessary
        asset_map = {
            name: Asset(name=name, **kwds) for name, kwds in self.asset_meta.items()
        }

        # calendar
        calendar = self.data.index

        # set up strategies
        self.strategies = [
            cls(
                orders=self.orders_unprocessed,
                params=pd.Series(self.strat_params, dtype=np.object0),
                books=self.books,  # TODO: make readonly wrapper
            )
            for cls in self.strats
        ]
        for strat in self.strategies:
            strat._data_lock = False
            strat.data = deepcopy(self.data)
            strat.init()
            strat._data_lock = True

        # run event loop
        book_history = defaultdict(list)

        for ts in calendar:
            logger.info(f"Processing timestep {ts}")

            # open
            for strat in self.strategies:
                # provide window
                strat._ts_lock = False
                strat._ts = ts
                strat._ts_lock = True
                strat._mask_hlc = True
                strat.on_open()
                strat._mask_hlc = False

            # order applied with ts's data
            day_data = self.data.loc[
                ts, (slice(None), ["High", "Low", "Open", "Close"])
            ]

            # process orders
            while self.orders_unprocessed:
                order = self.orders_unprocessed.popleft()

                if order.book is None:
                    order.book = self.books[0]

                order.apply(ts, day_data, asset_map)

                if order.status == OrderStatus.OPEN:
                    self.orders_unprocessed.append(order)
                else:
                    self.orders_processed.append(order)

            # close
            for strat in self.strategies:
                # provide window
                strat._ts_lock = False
                strat._ts = ts
                strat._ts_lock = True
                strat.on_close()

            # capture stats
            for book in self.books:
                cash = float(book.cash)
                mtm = sum(
                    day_data[an].Close * float(q) for an, q in book.positions.items()
                )
                book_history[(book.name, "cash")].append(cash)
                book_history[(book.name, "mtm")].append(mtm)
                book_history[(book.name, "value")].append(cash + mtm)

        self.book_history = pd.DataFrame(book_history, index=calendar)

        self.trade_history = pd.concat(
            [pd.DataFrame(bk.trades).assign(book=bk.name) for bk in self.books]
        )
