from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from ._helpers import ensure_decimal, ensure_enum
from .asset import Asset, AssetName
from .book import Book
from .trade import Trade

logger = logging.getLogger(__name__)

__all__ = ["Order", "PositionalOrder", "BasketOrder", "PositionalBasketOrder"]


class OrderStatus(Enum):
    """Various statuses"""

    MANDATE_FAILED = 1
    """Order failed due to mandate."""

    CANCELLED = 2
    """Order was cancelled."""

    OPEN = 3
    """Order is open."""

    COMPLETE = 4
    """Order completed succesfully."""


class OrderSizeType(Enum):
    """Various size types."""

    QUANTITY = 1
    """Size is a quantity."""

    NOTIONAL = 2
    """Size represent notional amount."""

    BOOK_PERCENT = 3
    """Size is a percentage of book value."""


def _intraday_traded_price(asset_day_data) -> Decimal:
    # TODO: support pluggable alternatives
    if pd.notnull(asset_day_data.Low) and pd.notnull(asset_day_data.High):
        return Decimal((asset_day_data.Low + asset_day_data.High) / 2)
    else:
        return Decimal(asset_day_data.Close)


@dataclass(kw_only=True)
class OrderBase:
    """Base class for all orders."""

    status: OrderStatus = OrderStatus.OPEN
    """Status of order."""

    book: Optional[Book] = field(repr=False, default=None)
    """Target book."""

    suborders: List[OrderBase] = field(default_factory=list)
    """Additional orders to be executed the following timestep."""

    label: Optional[str] = None
    """Label to assist in matching / filtering."""

    priority: int = 0
    """Each day orders are sorted by this field and executed in order."""

    def __post_init__(self):
        # TODO: support being a BookName
        pass

    def apply(
        self, ts: pd.Timestamp, day_data: pd.DataFrame, asset_map: Dict[str, Asset]
    ):
        """Applies order to `self.book` for time `ts` using provided `day_data` and
        dictionary of asset information `asset_map`.
        """
        raise NotImplementedError("The apply methods needs to be implemented.")


@dataclass(kw_only=True)
class Order(OrderBase):
    """Simple market order."""

    asset_name: AssetName
    """Asset name for order."""

    size: Decimal
    """Order size."""

    size_type: OrderSizeType = OrderSizeType.QUANTITY
    """Order size type. Can be a quantity, notional or book percent."""

    pre_exec_cond: Optional[Callable[[Decimal], Optional[OrderStatus]]] = None
    """Callable that if set, is called with the calculated
    trade price before the trade is executed. If it returns `None`,
    the trade is executed as normal. It can return `OrderStatus.CANCELLED`
    to indicate the trade should be cancelled or `OrderStatus.OPEN` to
    indicate the trade should not be executed in the current timestep
    and processed in the following timestep."""

    post_complete: Optional[Callable[[List[Trade]], List[OrderBase]]] = None
    """Callable that if set, is called after and with trades that have
    been successfully booked. It can return a list of new orders to be
    executed the following timestep."""

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

        if self.pre_exec_cond is not None:
            new_status = self.pre_exec_cond(trade_price)
            if new_status is not None:
                self.status = new_status
                return

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
            if self.post_complete is not None:
                self.suborders.extend(self.post_complete(trades))

        else:
            self.status = OrderStatus.MANDATE_FAILED


class PositionalOrderCheckType(Enum):
    POS_TQ_DIFFER = 1
    ZERO_POS = 2


@dataclass(kw_only=True)
class PositionalOrder(Order):
    """Ensures current position is `size` and will close out existing positions
    to achieve this."""

    check_type: PositionalOrderCheckType = PositionalOrderCheckType.POS_TQ_DIFFER
    """Condition type to determine if a trade is required."""

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

        if self.check_type == PositionalOrderCheckType.POS_TQ_DIFFER:
            needs_trades = current_position != trade_quantity
        elif self.check_type == PositionalOrderCheckType.ZERO_POS:
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
    """Combine multiple assets into a single order."""

    asset_names: List[AssetName]
    """List of asset names in basket."""

    weights: List[Decimal]
    """Corresponding weights for each asset."""

    size: Decimal
    """Combined size of order."""

    size_type: OrderSizeType = OrderSizeType.QUANTITY
    """Size type."""

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
            # TODO: size is ignored, perhaps use a scaling factor?
            # NOTE: we could use self.book.mtm but would be from previous day
            book_mtm = sum(
                [
                    self.book.positions.get(a.name, 0) * tp
                    for a, tp in zip(assets, trade_prices)
                ]
            )
            book_value = self.book.cash + book_mtm
            quantities = [
                book_value * w / 100 / tp for w, tp in zip(self.weights, trade_prices)
            ]
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

        trade_quantity_prices = self._calc_quantity_price(day_data, asset_map)

        trades = [
            Trade(
                asset_name=an,
                ts=ts,
                quantity=tq,
                price=tp,
            )
            for an, (tq, tp) in zip(self.asset_names, trade_quantity_prices)
        ]

        if self.book.test_trades(trades):
            self.book.add_trades(trades)
            self.status = OrderStatus.COMPLETE
        else:
            self.status = OrderStatus.MANDATE_FAILED


@dataclass(kw_only=True)
class PositionalBasketOrder(BasketOrder):
    """Similar to a :py:class:`BasketOrder` but will close out existing positions
    if they do not match requested weights."""

    check_type: PositionalOrderCheckType = PositionalOrderCheckType.POS_TQ_DIFFER

    def apply(
        self, ts: pd.Timestamp, day_data: pd.DataFrame, asset_map: Dict[str, Asset]
    ):
        if not self.book:
            raise RuntimeError("Cannot apply order without book")

        trade_quantity_prices = self._calc_quantity_price(day_data, asset_map)

        current_positions = [self.book.positions[an] for an in self.asset_names]

        if self.check_type == PositionalOrderCheckType.POS_TQ_DIFFER:
            needs_trades = any(
                p != tq for p, (tq, tp) in zip(current_positions, trade_quantity_prices)
            )
        elif self.check_type == PositionalOrderCheckType.ZERO_POS:
            needs_trades = any(p == 0 for p in current_positions)
        else:
            raise RuntimeError(f"Unexpected check type {self.check_type}")

        trades = []

        if needs_trades:  # otherwise we're done
            for asset_name, current_position, (trade_quantity, trade_price) in zip(
                self.asset_names, current_positions, trade_quantity_prices
            ):
                if current_position:
                    # close out existing position
                    trades.append(
                        Trade(
                            asset_name=asset_name,
                            ts=ts,
                            quantity=-current_position,
                            price=trade_price,
                        )
                    )
                if trade_quantity != 0:
                    trades.append(
                        Trade(
                            asset_name=asset_name,
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
