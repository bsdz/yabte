from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .asset import AssetName, CostType

logger = logging.getLogger(__name__)

__all__ = ["CashTransaction", "Cost", "Trade"]


@dataclass(frozen=True, kw_only=True)
class Transaction:
    """A frozen record of a transaction."""

    ts: pd.Timestamp
    """Transaction time."""

    total: Decimal = Decimal(0)
    """Total transaction value.

    Negative values are costs and positive values are benefits.
    """

    desc: str = ""
    """Description."""

    def __post_init__(self):
        # since frozen we need to use obj method to cast values
        if not isinstance(self.total, Decimal):
            object.__setattr__(self, "total", Decimal(self.total))


@dataclass(frozen=True, kw_only=True)
class CashTransaction(Transaction):
    """A frozen record of a cash transaction."""


@dataclass(frozen=True, kw_only=True)
class Trade(Transaction):
    """A frozen record of a trade transaction.

    A negative `quantity` represents a sell trade and a positive `quantity` represents a
    buy trade.
    """

    quantity: Decimal
    """Traded quantity."""

    price: Decimal
    """Traded price."""

    asset_name: AssetName
    """Traded asset."""

    order_label: str | None
    """Associated Order label."""

    def __post_init__(self):
        super().__post_init__()

        if self.quantity == 0:
            raise ValueError("trade quantity cannot be zero")

        # since frozen we need to use obj method to cast values
        if not isinstance(self.price, Decimal):
            object.__setattr__(self, "price", Decimal(self.price))
        if not isinstance(self.quantity, Decimal):
            object.__setattr__(self, "quantity", Decimal(self.quantity))

        # total represents cost/benefit when buy/sell
        object.__setattr__(self, "total", -self.quantity * self.price)
        object.__setattr__(
            self,
            "desc",
            f"{'sell' if self.quantity < 0 else 'buy'} {self.asset_name}",
        )


@dataclass(frozen=True, kw_only=True)
class Cost(Transaction):
    """A frozen record of the transaction cost of one order's trades in one asset.

    Its `total` is minus the cost. It has no `quantity`, so summing the `quantity`
    column of `transaction_history` still gives positions.
    """

    asset_name: AssetName
    """Traded asset."""

    order_label: str | None
    """Associated Order label."""

    charged_quantity: Decimal
    """The order's net traded quantity in the asset, which the cost is charged on."""

    cost_type: CostType
    """How the cost was charged."""

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.charged_quantity, Decimal):
            object.__setattr__(self, "charged_quantity", Decimal(self.charged_quantity))
        if not self.desc:
            object.__setattr__(self, "desc", f"cost {self.asset_name}")
