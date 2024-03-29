from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, TypeAlias, TypeVar, Union, cast

import pandas as pd
from mypy_extensions import mypyc_attr

__all__ = ["OHLCAsset"]


# use ints until mypyc supports IntFlag
# https://github.com/mypyc/mypyc/issues/1022
AssetDataFieldInfo = int
ADFI_AVAILABLE_AT_CLOSE: int = 1
ADFI_AVAILABLE_AT_OPEN: int = 2
ADFI_REQUIRED: int = 4

T = TypeVar("T", bound=Union[pd.Series, pd.DataFrame])

AssetName: TypeAlias = str
"""Asset name string."""


@mypyc_attr(allow_interpreted_subclasses=True)
@dataclass(kw_only=True)
class Asset:
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

    def intraday_traded_price(
        self, asset_day_data: pd.Series, size: Decimal | None = None
    ) -> Decimal:
        """Calculate price during market hours with given row of `asset_day_data` and
        the order `size`.

        The `size` can be used to
        determine a price from say, bid / ask spreads.
        """
        raise NotImplementedError(
            "The intraday_traded_price method needs to be implemented."
        )

    def end_of_day_price(self, asset_day_data: pd.Series) -> Decimal:
        """Calculate price at end of day with given row of `asset_day_data`."""
        raise NotImplementedError(
            "The end_of_day_price method needs to be implemented."
        )

    def check_and_fix_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Checks dataframe `data` has correct fields and fixes columns where
        necessary."""
        raise NotImplementedError(
            "The check_and_fix_data method needs to be implemented."
        )

    def data_fields(self) -> list[tuple[str, AssetDataFieldInfo]]:
        """List of data fields and their availability."""
        raise NotImplementedError("The data_fields method needs to be implemented.")

    def _get_fields(self, field_info: AssetDataFieldInfo) -> list[str]:
        """Internal method to get fields from `data_fields` with `field_info`."""
        return [f for f, fi in self.data_fields() if fi & field_info]

    def _filter_data(self, data: T) -> T:
        """Internal method to filter `data` columns and return only those relevant to
        pricing."""
        assert isinstance(self.data_label, str)
        return cast(T, data[self.data_label])


@mypyc_attr(allow_interpreted_subclasses=True)
@dataclass(kw_only=True)
class OHLCAsset(Asset):
    """Assets whose price history is represented by High, Low, Open, Close and Volume
    fields."""

    def data_fields(self) -> list[tuple[str, AssetDataFieldInfo]]:
        return [
            ("High", ADFI_AVAILABLE_AT_CLOSE),
            ("Low", ADFI_AVAILABLE_AT_CLOSE),
            ("Open", ADFI_AVAILABLE_AT_CLOSE | ADFI_AVAILABLE_AT_OPEN),
            ("Close", ADFI_AVAILABLE_AT_CLOSE | ADFI_REQUIRED),
            ("Volume", ADFI_AVAILABLE_AT_CLOSE),
        ]

    def intraday_traded_price(
        self, asset_day_data: pd.Series, size: Decimal | None = None
    ) -> Decimal:
        if pd.notnull(asset_day_data.Low) and pd.notnull(asset_day_data.High):
            p = Decimal((asset_day_data.Low + asset_day_data.High) / 2)
        else:
            p = Decimal(asset_day_data.Close)
        return round(p, self.price_round_dp)

    def end_of_day_price(self, asset_day_data: pd.Series) -> Decimal:
        return round(Decimal(asset_day_data.Close), self.price_round_dp)

    def check_and_fix_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # TODO: check low <= open, high, close & high >= open, low, close
        # TODO: check volume >= 0

        # check each asset has required fields
        required_fields = self._get_fields(ADFI_REQUIRED)
        missing_req_fields = set(required_fields) - set(data.columns)
        if len(missing_req_fields):
            raise ValueError(
                f"data columns index requires fields {required_fields}"
                f" and missing {missing_req_fields}"
            )

        # reindex columns with expected fields + additional fields
        expected_fields = self._get_fields(ADFI_AVAILABLE_AT_CLOSE)
        other_fields = list(set(data.columns) - set(expected_fields))
        return data.reindex(expected_fields + other_fields, axis=1)
