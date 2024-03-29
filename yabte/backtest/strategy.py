import logging
from dataclasses import dataclass
from itertools import chain, product
from typing import Dict

import pandas as pd
from mypy_extensions import mypyc_attr

# TODO: use explicit imports until mypyc fixes attribute lookups in dataclass
# (https://github.com/mypyc/mypyc/issues/1000)
from pandas import DataFrame, Series, Timestamp  # type: ignore

from .asset import ADFI_AVAILABLE_AT_OPEN, Asset, AssetName
from .book import Book, BookName
from .order import Orders

logger = logging.getLogger(__name__)

__all__ = ["Strategy"]


@mypyc_attr(allow_interpreted_subclasses=True)
@dataclass(kw_only=True)
class Strategy:
    """Trading strategy base class.

    This class should be derived from and will be instantiated by
    :py:class:`StrategyRunner`.
    """

    orders: Orders | None = None
    """Double ended queue of orders."""

    params: pd.Series | None = None
    """Parameters supplied to strategy."""

    books: Dict[BookName, Book] | None = None
    """Dictionary of books."""

    assets: Dict[AssetName, Asset] | None = None
    """Dictionary of assets."""

    _ts: pd.Timestamp | None = None
    _data_lock: bool = True
    _mask_open: bool = False

    @property
    def ts(self) -> pd.Timestamp | None:
        """Stores the current timestamp."""
        return self._ts

    def _set_ts(self, ts: pd.Timestamp):
        """Internal method to update timestep to current `ts`"""
        self._ts = ts

    def _get_col_indexer(self):
        # cache this call for hopefully a small speed up
        if not hasattr(self, "_col_indexer"):
            mix = pd.MultiIndex.from_tuples(
                chain(
                    *[
                        product(
                            [asset.data_label],
                            asset._get_fields(ADFI_AVAILABLE_AT_OPEN),
                        )
                        for asset_name, asset in self.assets.items()
                    ]
                )
            )
            self._col_indexer = ~self._data.columns.isin(mix)
        return self._col_indexer

    @property
    def data(self) -> pd.DataFrame:
        """Provides window of data available up to current timestamp `self.ts` and masks
        out data not availble at open (like high, low, close, volume)."""
        if not self.ts:
            return self._data
        else:
            df_t = self._data.loc[: self.ts, :]  # type: ignore[misc]
            if not self._mask_open:
                data = df_t
            else:
                row_indexer = df_t.index == df_t.index[-1]
                # generate mask from asset instances, some assets
                # might support different field masks
                col_indexer = self._get_col_indexer()
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
        """Executed on open every day.

        Use `self.ts` to determine current timestamp. Data available at
        this timestamp is accessible from `self.data`.
        """
        pass

    def on_close(self):
        """Executed on close every day.

        Use `self.ts` to determine current timestamp. Data available at
        this timestamp is accessible from `self.data`.
        """
        pass
