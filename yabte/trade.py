from decimal import Decimal
import logging
from dataclasses import dataclass

import pandas as pd

from .asset import AssetName

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class Trade:
    ts: pd.Timestamp
    quantity: Decimal
    price: Decimal
    asset_name: AssetName

    def __post_init__(self):
        if self.quantity == 0:
            raise ValueError("trade quantity cannot be zero")
        # since frozen we need to use obj method to cast values
        if not isinstance(self.quantity, Decimal):
            object.__setattr__(self, "quantity", Decimal(self.quantity))
        if not isinstance(self.price, Decimal):
            object.__setattr__(self, "price", Decimal(self.price))
