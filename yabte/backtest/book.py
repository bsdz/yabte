import logging
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from itertools import groupby
from typing import Any, Dict, List

import pandas as pd

from ._helpers import ensure_decimal
from .asset import Asset, AssetName
from .trade import Trade

logger = logging.getLogger(__name__)


__all__ = ["Book"]


@dataclass(kw_only=True)
class BookMandate:
    def check(self, current_pos, quantity):
        raise NotImplementedError()


class BookName(str):
    """Book name string."""


@dataclass(kw_only=True)
class Book:
    """Record of asset trades and positions including cash.

    A `name` can be provided and the default `cash` value can be changed
    to a non-zero amount.
    """

    name: BookName
    """Name of book."""

    mandates: Dict[AssetName, BookMandate] = field(default_factory=dict)
    """Dictionary mapping assets to mandates (experimental)."""

    positions: Dict[AssetName, Decimal] = field(
        default_factory=lambda: defaultdict(Decimal)
    )
    """Dictionary tracking positions of each asset."""

    trades: List[Trade] = field(default_factory=list)
    """List of executed trades."""

    denom: str = "USD"
    """Book currency."""

    cash: Decimal = Decimal(0)
    """Cash value of book."""

    _history: List[List[Any]] = field(default_factory=list)

    @property
    def history(self) -> pd.DataFrame:
        """Dataframe with book cash, mtm and total value history."""
        return pd.DataFrame(
            self._history, columns=["ts", "cash", "mtm", "value"]
        ).set_index("ts")

    def __post_init__(self):
        self.cash = ensure_decimal(self.cash)

    def test_trades(self, trades: List[Trade]) -> bool:
        """Checks whether list of trades will be successful by not failing any
        mandates."""
        for asset_name, asset_trades in groupby(trades, lambda t: t.asset_name):
            if asset_name in self.mandates:
                total_quantity = sum(t.quantity for t in asset_trades)
                if not self.mandates[asset_name].check(
                    self.positions[asset_name], total_quantity
                ):
                    return False
        return True

    def add_trades(self, trades: List[Trade]):
        """Records the `trades` and adjusts internal dictionary of positions
        and value of cash accordingly."""
        for trade in trades:
            self.trades.append(trade)
            self.positions[trade.asset_name] += trade.quantity
            self.cash -= trade.quantity * trade.price

    def eod_tasks(
        self, ts: pd.Timestamp, day_data: pd.DataFrame, asset_map: Dict[str, Asset]
    ):
        """Run end of day tasks such as book keeping.

        TODO: support daily interest accumulation.
        """
        cash = float(self.cash)
        mtm = sum(day_data[an].Close * float(q) for an, q in self.positions.items())
        self._history.append([ts, cash, mtm, cash + mtm])
