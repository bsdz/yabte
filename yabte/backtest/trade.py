import logging
from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from .asset import AssetName

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class Trade:
    """A frozen record of the transaction."""

    ts: pd.Timestamp
    """Trade time."""

    quantity: Decimal
    """Traded quantity."""

    price: Decimal
    """Traded price."""

    asset_name: AssetName
    """Traded asset."""

    def __post_init__(self):
        if self.quantity == 0:
            raise ValueError("trade quantity cannot be zero")
        # since frozen we need to use obj method to cast values
        if not isinstance(self.quantity, Decimal):
            object.__setattr__(self, "quantity", Decimal(self.quantity))
        if not isinstance(self.price, Decimal):
            object.__setattr__(self, "price", Decimal(self.price))
