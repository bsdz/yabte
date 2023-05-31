from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence, TypeAlias

import pandas as pd
from mypy_extensions import mypyc_attr

__all__ = ["Asset"]


AssetName: TypeAlias = str
"""Asset name string."""


@mypyc_attr(allow_interpreted_subclasses=True)
@dataclass(kw_only=True)
class AssetBase:
    """Anything that has a price."""

    name: AssetName
    """Name string."""

    denom: str = "USD"
    """Denominated currency."""

    price_round_dp: int = 2
    """Number of decimal places to round prices to."""

    quantity_round_dp: int = 2
    """Number of decimal places to round quantities to."""

    data_label: str | None = None
    """`StrategyRunner.data` column index 1st level label.

    Defaults to `name`
    """

    def __post_init__(self):
        if self.data_label is None:
            self.data_label = self.name

    def round_quantity(self, quantity) -> Decimal:
        """Round `quantity`."""
        return round(quantity, self.quantity_round_dp)

    @property
    def fields_available_at_open(self) -> Sequence[str]:
        """A sequence of field names available at open.

        Any fields not in this sequence will be masked out.
        """
        return []

    def intraday_traded_price(self, asset_day_data) -> Decimal:
        """Calculate price during market hours with given row of
        `asset_day_data`."""
        raise NotImplementedError("The apply methods needs to be implemented.")

    def check_and_fix_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Checks dataframe `data` has correct fields and fixes columns where
        necessary."""
        raise NotImplementedError("The apply methods needs to be implemented.")


@mypyc_attr(allow_interpreted_subclasses=True)
@dataclass(kw_only=True)
class Asset(AssetBase):
    """Assets whose price history is represented by High, Low, Open, Close and
    Volume fields."""

    @property
    def fields_available_at_open(self) -> Sequence[str]:
        return ["Open"]

    def intraday_traded_price(self, asset_day_data) -> Decimal:
        if pd.notnull(asset_day_data.Low) and pd.notnull(asset_day_data.High):
            p = Decimal((asset_day_data.Low + asset_day_data.High) / 2)
        else:
            p = Decimal(asset_day_data.Close)
        return round(p, self.price_round_dp)

    def check_and_fix_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # TODO: check low <= open, high, close & high >= open, low, close
        # TODO: check volume >= 0

        # check each asset has required fields
        required_fields = {"Close"}
        missing_req_fields = required_fields - set(data.columns)
        if len(missing_req_fields):
            raise ValueError(
                f"data columns index requires fields {required_fields} and missing {missing_req_fields}"
            )

        # reindex columns with expected fields + additional fields
        expected_fields = ["High", "Low", "Open", "Close", "Volume"]
        other_fields = list(set(data.columns) - set(expected_fields))
        return data.reindex(expected_fields + other_fields, axis=1)
