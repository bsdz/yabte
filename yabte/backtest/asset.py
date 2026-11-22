from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, TypeAlias, TypeVar, Union, cast, get_args

import pandas as pd

from ._helpers import round_dp

__all__ = ["OHLCAsset", "FillPrice", "MissingPriceError", "find_missing_data"]


# TODO: switch to IntFlag since no longer using mypyc
AssetDataFieldInfo = int
ADFI_AVAILABLE_AT_CLOSE: int = 1
ADFI_AVAILABLE_AT_OPEN: int = 2
ADFI_REQUIRED: int = 4

AssetName: TypeAlias = str
"""Asset name string."""

T = TypeVar("T", bound=Union[pd.Series, pd.DataFrame])

FillPrice: TypeAlias = Literal["midpoint", "open", "close"]
"""Which price `OHLCAsset` orders fill at; see `OHLCAsset.fill_price`."""


class MissingPriceError(ValueError):
    """Raised when a backtest needs a price an asset can't give: an order executes on a
    day with no price, or a book holds the asset at a close with no price."""


def _require_price(price, asset_name: str, ts: pd.Timestamp, kind: str) -> Decimal:
    """Return `price` as a `Decimal`, raising `MissingPriceError` if it's missing.

    The engines call this on every price they use, so it covers custom assets too.
    """
    if price is None or not (price := Decimal(price)).is_finite():
        raise MissingPriceError(
            f"No {kind} price for {asset_name} on "
            f"{pd.Timestamp(ts):%Y-%m-%d %H:%M:%S}; drop or fill the missing data"
            " (find_missing_data lists the gaps)"
        )
    return price


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
        return round_dp(quantity, self.quantity_round_dp)

    def intraday_traded_price(
        self, asset_day_data: pd.Series, size: Decimal | None = None
    ) -> Decimal:
        """Calculate price during market hours with given row of `asset_day_data` and
        the order `size`.

        The `size` can be used to determine a price from say, bid / ask spreads.
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


@dataclass(kw_only=True)
class OHLCAsset(Asset):
    """Assets whose price history is represented by High, Low, Open, Close and Volume
    fields."""

    fill_price: FillPrice = "midpoint"
    """The price orders fill at.

    - ``"midpoint"`` (the default): the mean of the row's `High` and `Low`, or its
      `Close` if either is missing.
    - ``"open"``: the row's `Open`, which is then a required field. Use it for orders
      placed in `on_open`: they execute on the same row, and the other prices aren't
      known at the open yet.
    - ``"close"``: the row's `Close`.
    """

    def __post_init__(self):
        super().__post_init__()
        if self.fill_price not in get_args(FillPrice):
            raise ValueError(
                f"fill_price must be one of {get_args(FillPrice)}, "
                f"not {self.fill_price!r}"
            )

    def data_fields(self) -> list[tuple[str, AssetDataFieldInfo]]:
        open_required = ADFI_REQUIRED if self.fill_price == "open" else 0
        return [
            ("High", ADFI_AVAILABLE_AT_CLOSE),
            ("Low", ADFI_AVAILABLE_AT_CLOSE),
            ("Open", ADFI_AVAILABLE_AT_CLOSE | ADFI_AVAILABLE_AT_OPEN | open_required),
            ("Close", ADFI_AVAILABLE_AT_CLOSE | ADFI_REQUIRED),
            ("Volume", ADFI_AVAILABLE_AT_CLOSE),
        ]

    def intraday_traded_price(
        self, asset_day_data: pd.Series, size: Decimal | None = None
    ) -> Decimal:
        if self.fill_price == "open":
            p = Decimal(asset_day_data.Open)
        elif self.fill_price == "close":
            p = Decimal(asset_day_data.Close)
        elif pd.notnull(asset_day_data.Low) and pd.notnull(asset_day_data.High):
            p = Decimal((asset_day_data.Low + asset_day_data.High) / 2)
        else:
            p = Decimal(asset_day_data.Close)

        return round_dp(p, self.price_round_dp)

    def end_of_day_price(self, asset_day_data: pd.Series) -> Decimal:
        return round_dp(Decimal(asset_day_data.Close), self.price_round_dp)

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


def find_missing_data(data: pd.DataFrame, assets: list[Asset]) -> pd.DataFrame:
    """List the missing values of each asset's required fields in `data`.

    `data` and `assets` are as passed to `StrategyRunner`. Required fields are those
    that `Asset.data_fields` flags with `ADFI_REQUIRED` (`Close` for `OHLCAsset`).
    Returns one row per missing value, with columns `ts`, `asset_name` and `field`, so
    `find_missing_data(...).groupby("asset_name").ts.agg(["min", "max", "count"])`
    summarises the gaps.

    Gaps aren't necessarily errors: the engines only raise `MissingPriceError` when a
    run needs the price, because an order executes or the book holds the asset that
    day. Use this to find and fix gaps before a run rather than at the first failure.
    """
    rows: list[tuple[pd.Timestamp, str, str]] = []
    for asset in assets:
        assert isinstance(asset.data_label, str)
        for field in asset._get_fields(ADFI_REQUIRED):
            col = (asset.data_label, field)
            if col not in data.columns:
                missing = data.index
            else:
                missing = data.index[data[col].isna().to_numpy()]
            rows.extend((ts, asset.name, field) for ts in missing)
    return pd.DataFrame(rows, columns=["ts", "asset_name", "field"])
