import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


class AssetName(str):
    pass


@dataclass(kw_only=True)
class Asset:
    name: AssetName
    denom: str
    price_round_dp: int = 2
    quantity_round_dp: int = 2
