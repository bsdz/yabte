from typing import Dict, Optional, Tuple, List
from decimal import Decimal
import logging
from enum import Enum
from dataclasses import dataclass, field

import pandas as pd

from ._helpers import ensure_decimal, ensure_enum

from .asset import AssetName, Asset
from .book import Book
from .trade import Trade

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    MANDATE_FAILED = 1
    CANCELLED = 2
    OPEN = 3
    COMPLETE = 4


class OrderSizeType(Enum):
    QUANTITY = 1
    NOTIONAL = 2
    BOOK_PERCENT = 3


def _intraday_traded_price(asset_day_data) -> Decimal:
    # TODO: support pluggable alternatives
    if pd.notnull(asset_day_data.Low) and pd.notnull(asset_day_data.High):
        return Decimal((asset_day_data.Low + asset_day_data.High) / 2)
    else:
        return Decimal(asset_day_data.Close)


@dataclass(kw_only=True)
class OrderBase:
    status: OrderStatus = OrderStatus.OPEN
    book: Optional[Book] = field(repr=False, default=None)
    # priority: Optional[int] = 0  # TODO

    def __post_init__(self):
        # TODO: support being a BookName
        pass


@dataclass(kw_only=True)
class Order(OrderBase):
    asset_name: AssetName
    size: Decimal
    size_type: OrderSizeType = OrderSizeType.QUANTITY
    # TODO: support below types of order
    # limit_price: Optional[Decimal] = None
    # stop_price: Optional[Decimal] = None
    # stop_loss_price: Optional[Decimal] = None
    # take_profit_price: Optional[Decimal] = None

    def __post_init__(self):
        super().__post_init__()
        self.size = ensure_decimal(self.size)
        self.size_type = ensure_enum(self.size_type, OrderSizeType)

    def _calc_quantity_price(self, day_data, asset_map) -> Tuple[Decimal, Decimal]:
        asset = asset_map[self.asset_name]
        asset_day_data = day_data[self.asset_name]
        trade_price = _intraday_traded_price(asset_day_data)

        if self.size_type == OrderSizeType.QUANTITY:
            quantity = self.size
        elif self.size_type == OrderSizeType.NOTIONAL:
            quantity = self.size / trade_price
        elif self.size_type == OrderSizeType.BOOK_PERCENT:
            assert self.book is not None  # to please mypy
            quantity = self.book.cash * self.size / 100 / trade_price
        else:
            raise RuntimeError("Unsupported size type")

        return round(quantity, asset.quantity_round_dp), round(
            trade_price, asset.price_round_dp
        )

    def apply(
        self, ts: pd.Timestamp, day_data: pd.DataFrame, asset_map: Dict[str, Asset]
    ):
        if not self.book:
            raise RuntimeError("Cannot apply order without book")

        trade_quantity, trade_price = self._calc_quantity_price(day_data, asset_map)

        trades = [
            Trade(
                asset_name=self.asset_name,
                ts=ts,
                quantity=trade_quantity,
                price=trade_price,
            )
        ]

        if self.book.test_trades(trades):
            self.book.add_trades(trades)
            self.status = OrderStatus.COMPLETE
        else:
            self.status = OrderStatus.MANDATE_FAILED


class PositionalOrderCheckType(Enum):
    UNEQUAL = 1
    NONZERO = 2


@dataclass(kw_only=True)
class PositionalOrder(Order):
    """Closes out existing positions, ensures current position is `quantity`"""

    check_type: PositionalOrderCheckType = PositionalOrderCheckType.UNEQUAL

    def __post_init__(self):
        super().__post_init__()
        self.check_type = ensure_enum(self.check_type, PositionalOrderCheckType)

    def apply(
        self, ts: pd.Timestamp, day_data: pd.DataFrame, asset_map: Dict[str, Asset]
    ):
        if not self.book:
            raise RuntimeError("Cannot apply order without book")

        trade_quantity, trade_price = self._calc_quantity_price(day_data, asset_map)

        current_position = self.book.positions[self.asset_name]

        if self.check_type == PositionalOrderCheckType.UNEQUAL:
            needs_trades = current_position != trade_quantity
        elif self.check_type == PositionalOrderCheckType.NONZERO:
            needs_trades = current_position == 0
        else:
            raise RuntimeError(f"Unexpected check type {self.check_type}")

        trades = []

        if needs_trades:  # otherwise we're done
            if current_position:
                # close out existing position
                trades.append(
                    Trade(
                        asset_name=self.asset_name,
                        ts=ts,
                        quantity=-current_position,
                        price=trade_price,
                    )
                )
            if trade_quantity != 0:
                trades.append(
                    Trade(
                        asset_name=self.asset_name,
                        ts=ts,
                        quantity=trade_quantity,
                        price=trade_price,
                    )
                )

        if self.book.test_trades(trades):
            self.book.add_trades(trades)
            self.status = OrderStatus.COMPLETE
        else:
            self.status = OrderStatus.MANDATE_FAILED


@dataclass
class BasketOrder(OrderBase):
    asset_names: List[AssetName]
    weights: List[Decimal]
    size: Decimal
    size_type: OrderSizeType = OrderSizeType.QUANTITY

    def __post_init__(self):
        super().__post_init__()
        self.weights = [ensure_decimal(w) for w in self.weights]
        self.size = ensure_decimal(self.size)
        self.size_type = ensure_enum(self.size_type, OrderSizeType)

    def _calc_quantity_price(
        self, day_data, asset_map
    ) -> List[Tuple[Decimal, Decimal]]:
        assets = [asset_map[an] for an in self.asset_names]
        assets_day_data = [day_data[an] for an in self.asset_names]
        trade_prices = [_intraday_traded_price(add) for add in assets_day_data]

        if self.size_type == OrderSizeType.QUANTITY:
            quantities = [self.size * w for w in self.weights]
        elif self.size_type == OrderSizeType.NOTIONAL:
            # size = k * sum(w_i * p_i)
            tp_weighted = sum(w * p for w, p in zip(self.weights, trade_prices))
            k = self.size / tp_weighted
            quantities = [k * w for w in self.weights]
        elif self.size_type == OrderSizeType.BOOK_PERCENT:
            assert self.book is not None  # to please mypy
            tp_weighted = sum(w * p for w, p in zip(self.weights, trade_prices))
            bk_notional = self.book.cash * self.size / 100
            k = bk_notional / tp_weighted
            quantities = [k * w for w in self.weights]
        else:
            raise RuntimeError("Unsupported size type")

        return [
            (round(q, a.quantity_round_dp), round(tp, a.price_round_dp))
            for a, q, tp in zip(assets, quantities, trade_prices)
        ]

    def apply(
        self, ts: pd.Timestamp, day_data: pd.DataFrame, asset_map: Dict[str, Asset]
    ):
        if not self.book:
            raise RuntimeError("Cannot apply order without book")

        trade_quantities, trade_prices = self._calc_quantity_price(day_data, asset_map)

        trades = [
            Trade(
                asset_name=an,
                ts=ts,
                quantity=tq,
                price=tp,
            )
            for an, tq, tp in zip(self.asset_names, trade_quantities, trade_prices)
        ]

        if self.book.test_trades(trades):
            self.book.add_trades(trades)
            self.status = OrderStatus.COMPLETE
        else:
            self.status = OrderStatus.MANDATE_FAILED
