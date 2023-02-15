from dataclasses import dataclass

__all__ = ["Asset"]


class AssetName(str):
    """Asset name string."""


@dataclass(kw_only=True)
class Asset:
    """Anything that has a price.

    The price currency is `denom` and default rounding
    controlled by `price_round_dp`. The quantity rounding
    is controled by `quantity_round_dp`.
    """

    name: AssetName
    """Name string."""

    denom: str
    """Denominated currency."""

    price_round_dp: int = 2
    """Number of decimal places to round prices to."""

    quantity_round_dp: int = 2
    """Number of decimal places to round quantities to."""
