from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

__all__ = ["Asset"]


class AssetName(str):
    """Asset name string."""


@dataclass(kw_only=True)
class Asset:
    """Anything that has a price.

    The price currency is `denom` and default rounding controlled by
    `price_round_dp`. The quantity rounding is controled by
    `quantity_round_dp`.
    """

    name: AssetName
    """Name string."""

    denom: str = "USD"
    """Denominated currency."""

    price_round_dp: int = 2
    """Number of decimal places to round prices to."""

    quantity_round_dp: int = 2
    """Number of decimal places to round quantities to."""

    def intraday_traded_price(self, asset_day_data) -> Decimal:
        """Calculate price during market hours with given row of
        `asset_day_data`."""
        if pd.notnull(asset_day_data.Low) and pd.notnull(asset_day_data.High):
            p = Decimal((asset_day_data.Low + asset_day_data.High) / 2)
        else:
            p = Decimal(asset_day_data.Close)
        return round(p, self.price_round_dp)

    def round_quantity(self, quantity) -> Decimal:
        """Round `quantity`."""
        return round(quantity, self.quantity_round_dp)
