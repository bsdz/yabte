from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

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

    @property
    def fields_available_at_open(self) -> Sequence[str]:
        """A sequence of field names available at open.

        Any fields not in this sequence will be masked out.
        """
        return ["Open"]

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

    def check_and_fix_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Checks dataframe `data` has correct fields and fixes columns where
        necessary."""

        # TODO: check low <= open, high, close & high >= open, low, close
        # TODO: check vol >= 0

        # check each asset has required fields
        required_fields = {"Close"}
        missing_req_fields = required_fields - set(data.columns)
        if len(missing_req_fields):
            raise ValueError(
                f"data columns multiindex requires fields {required_fields} and missing {missing_req_fields}"
            )

        # reindex columns with expected fields
        expected_fields = ["High", "Low", "Open", "Close", "Volume"]
        return data.reindex(expected_fields, axis=1)
