from dataclasses import dataclass

__all__ = ["Asset"]


class AssetName(str):
    pass


@dataclass(kw_only=True)
class Asset:
    """Anything that has a price.

    The price currency is `denom` and default rounding
    controlled by `price_round_dp`. The quantity rounding
    is controled by `quantity_round_dp`.
    """

    name: AssetName
    denom: str
    price_round_dp: int = 2
    quantity_round_dp: int = 2
