import logging
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from itertools import groupby
from typing import Dict, List

from ._helpers import ensure_decimal
from .asset import AssetName
from .trade import Trade

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class BookMandate:
    def check(self, current_pos, quantity):
        raise NotImplementedError()


class BookName(str):
    pass


@dataclass(kw_only=True)
class Book:
    name: BookName
    mandates: Dict[AssetName, BookMandate] = field(default_factory=dict)
    positions: Dict[AssetName, Decimal] = field(
        default_factory=lambda: defaultdict(Decimal)
    )
    trades: List[Trade] = field(default_factory=list)
    cash: Decimal = Decimal(0)

    def __post_init__(self):
        self.cash = ensure_decimal(self.cash)

    def test_trades(self, trades: List[Trade]) -> bool:
        for asset_name, asset_trades in groupby(trades, lambda t: t.asset_name):
            if asset_name in self.mandates:
                total_quantity = sum(t.quantity for t in asset_trades)
                if not self.mandates[asset_name].check( self.positions[asset_name], total_quantity):
                    return False
        return True

    def add_trades(self, trades: List[Trade]):
        for trade in trades:
            self.trades.append(trade)
            self.positions[trade.asset_name] += trade.quantity
            self.cash -= trade.quantity * trade.price
