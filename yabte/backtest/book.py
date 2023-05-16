import logging
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from itertools import groupby
from typing import Any, Dict, List, Sequence, TypeAlias

import pandas as pd

from ._helpers import ensure_decimal
from .asset import Asset, AssetName
from .transaction import CashTransaction, Trade, Transaction

logger = logging.getLogger(__name__)


__all__ = ["Book"]


@dataclass(kw_only=True)
class BookMandate:
    def check(self, current_pos, quantity) -> bool:
        raise NotImplementedError()


BookName: TypeAlias = str
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

    transactions: List[Transaction] = field(default_factory=list)
    """List of executed trades."""

    denom: str = "USD"
    """Book currency."""

    cash: Decimal = Decimal(0)
    """Cash value of book."""

    rate: Decimal = Decimal(0)
    """Daily interest rate applied to cash at end of day."""

    interest_round_dp: int = 3
    """Number of decimal places to round interest."""

    _history: List[List[Any]] = field(default_factory=list)

    @property
    def history(self) -> pd.DataFrame:
        """Dataframe with book cash, mtm and total value history."""
        return pd.DataFrame(
            self._history, columns=["ts", "cash", "mtm", "total"]
        ).set_index("ts")

    def __post_init__(self):
        self.cash = ensure_decimal(self.cash)
        self.rate = ensure_decimal(self.rate)

    def test_trades(self, trades: Sequence[Trade]) -> bool:
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

    def add_transactions(self, transactions: Sequence[Transaction]):
        """Records the `transactions` and adjusts internal dictionary of
        positions and value of cash accordingly."""
        for tran in transactions:
            if isinstance(tran, Trade):
                self.positions[tran.asset_name] += tran.quantity
                self.cash += tran.total
            elif isinstance(tran, CashTransaction):
                self.cash += tran.total
            else:
                raise ValueError(f"Unsupport transaction class: {type(tran)}")

            self.transactions.append(tran)

    def eod_tasks(
        self, ts: pd.Timestamp, day_data: pd.DataFrame, asset_map: Dict[str, Asset]
    ):
        """Run end of day tasks such as book keeping."""
        # accumulate continously compounded interest
        interest = round(self.cash * (self.rate.exp() - 1), self.interest_round_dp)
        if self.rate != 0 and interest != 0:
            self.add_transactions(
                [
                    CashTransaction(
                        ts=ts,
                        total=interest,
                        desc=f"interest payment on cash {self.cash:.2f}",
                    )
                ]
            )
        cash = float(self.cash)
        mtm = sum(
            day_data[asset_map[an].data_label].Close * float(q)
            for an, q in self.positions.items()
        )
        self._history.append([ts, cash, mtm, cash + mtm])
