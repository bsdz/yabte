import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import pandas as pd

# TODO: use explicit imports until mypyc fixes attribute lookups in dataclass
# (https://github.com/mypyc/mypyc/issues/1000)
from pandas import DataFrame, Series, Timestamp  # type: ignore

from .asset import Asset, AssetName
from .book import Book, BookMandate, BookName
from .order import Order, Orders, OrderStatus
from .strategy import Strategy

logger = logging.getLogger(__name__)

__all__ = ["StrategyRunner"]


def _check_data(df, asset_map):
    """Check data structure correct."""

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("data index must be a datetimeindex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("data needs to have increasing index")
    if not df.index.is_unique:
        raise ValueError("data index must be unique")

    # column level 1 = asset, level 2 = field
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("data columns must be multindex asset/field")
    if len(df.columns.levels) != 2:
        raise ValueError("data columns multiindex must have 2 levels")

    # for cartesian products
    data_labels_data = set(df.columns.levels[0])
    data_labels_asset = {a.data_label for a in asset_map.values()}
    assets_missing_data = data_labels_asset - data_labels_data
    if len(assets_missing_data):
        raise ValueError(
            f"some assets are missing corresponding data: {assets_missing_data}"
        )

    # check and fix data for each asset
    dfs = {
        asset.data_label: asset.check_and_fix_data(asset._filter_data(df))
        for asset_name, asset in asset_map.items()
    }

    return pd.concat(dfs, axis=1)


@dataclass(kw_only=True)
class StrategyRunner:
    """Encapsulates the execution of multiple strategies.

    Orders are captured in `orders_processed` and `orders_unprocessed`.
    `books` is a list of books and if none provided a single book is
    created called 'Main'. After execution summary book and trade
    histories are captured in `book_history` and `transaction_history`.
    """

    data: pd.DataFrame = field()
    """Dataframe of price data including columns High, Low, Open, Close, Volume for each
    asset.

    Both asset name and field make a multiindex column. The index should consist of
    order pandas timestamps.
    """

    assets: List[Asset]
    """Assets available to strategy."""

    strat_classes: List[Type[Strategy]]
    """Strategy classes to be called within this runner."""

    mandates: Dict[AssetName, BookMandate] = field(default_factory=dict)
    """Dictionary of asset mandates (experimental)."""

    strat_params: Dict[str, Any] = field(default_factory=dict)
    """Parameters passed to all strategies."""

    books: List[Book] = field(default_factory=list)
    """Books available to strategies.

    If not supplied will be populated with single book named 'Main' denominated in USD.
    """

    @property
    def book_map(self) -> Dict[BookName, Book]:
        """Mapping from book name to book instance."""
        return {book.name: book for book in self.books}

    @property
    def asset_map(self) -> Dict[AssetName, Asset]:
        """Mapping from asset name to asset instance."""
        return {asset.name: asset for asset in self.assets}

    _orders_unprocessed: Orders = field(default_factory=Orders)

    @property
    def orders_unprocessed(self) -> Orders:
        """Unprocessed orders queue."""
        return self._orders_unprocessed

    _orders_processed: List[Order] = field(default_factory=list)

    @property
    def orders_processed(self) -> List[Order]:
        """Processed orders list."""
        return self._orders_processed

    _strategies: List[Strategy] = field(default_factory=list)

    @property
    def strategies(self) -> List[Strategy]:
        """List of instantiated strategies."""
        return self._strategies

    _book_history: Optional[pd.DataFrame] = None

    @property
    def book_history(self) -> pd.DataFrame:
        """Dataframe with book cash, mtm and total value history."""
        return pd.concat({b.name: b.history for b in self.books}, axis=1)

    @property
    def transaction_history(self) -> pd.DataFrame:
        """Dataframe with trade history."""
        return pd.concat(
            [pd.DataFrame(bk.transactions).assign(book=bk.name) for bk in self.books]
        )

    def __post_init__(self):
        self.data = _check_data(self.data, self.asset_map)

        # set up books
        if not self.books:
            self.books = [Book(name="Main", mandates=self.mandates)]

    def run(self):
        """Execute each strategy through time."""

        # made available where necessary
        asset_map = self.asset_map
        book_map = self.book_map

        # calendar
        calendar = self.data.index

        # set up strategies
        self._strategies = [
            cls(
                orders=self._orders_unprocessed,
                params=pd.Series(self.strat_params, dtype=object),
                books=book_map,
                assets=asset_map,
            )
            for cls in self.strat_classes
        ]
        for strat in self._strategies:
            strat._data_lock = False
            strat.data = deepcopy(self.data)
            strat.init()
            strat._data_lock = True

        # run event loop
        for ts in calendar:
            logger.info(f"Processing timestep {ts}")

            # open
            for strat in self._strategies:
                # provide window
                strat._set_ts(ts)
                strat._mask_open = True
                strat.on_open()
                strat._mask_open = False

            # order applied with ts's data
            day_data = self.data.loc[ts, :]

            # sort orders by priority
            self._orders_unprocessed.sort_by_priority()

            # process orders
            orders_next_ts = []
            while self._orders_unprocessed:
                order = self._orders_unprocessed.popleft()

                # set book attribute if needed
                if not isinstance(order.book, Book):
                    # fall back to first available book
                    order.book = book_map.get(order.book, self.books[0])

                order.apply(ts, day_data, asset_map)

                # add any child orders to next ts
                orders_next_ts.extend(order.suborders)

                if order.status == OrderStatus.OPEN:
                    orders_next_ts.append(order)
                else:
                    self.orders_processed.append(order)

            # extend with orders for next ts
            self._orders_unprocessed.extend(orders_next_ts)

            # remove older duplicate orders
            replaced = self._orders_unprocessed.remove_duplicate_keys()
            self.orders_processed.extend(replaced)

            # close
            for strat in self._strategies:
                # provide window
                strat._set_ts(ts)
                strat.on_close()

            # run book end-of-day tasks
            for book in self.books:
                book.eod_tasks(ts, day_data, asset_map)
