from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd
from mypy_extensions import mypyc_attr

from ._helpers import ensure_decimal, ensure_enum
from .asset import Asset, AssetName
from .book import Book, BookName
from .transaction import Trade

logger = logging.getLogger(__name__)

__all__ = ["Order", "PositionalOrder", "BasketOrder", "PositionalBasketOrder"]


class OrderStatus(Enum):
    """Various statuses."""

    MANDATE_FAILED = 1
    """Order failed due to mandate."""

    CANCELLED = 2
    """Order was cancelled."""

    OPEN = 3
    """Order is open."""

    COMPLETE = 4
    """Order completed succesfully."""

    REPLACED = 5
    """Order was replaced."""


class OrderSizeType(Enum):
    """Various size types."""

    QUANTITY = 1
    """Size is a quantity."""

    NOTIONAL = 2
    """Size represent notional amount."""

    BOOK_PERCENT = 3
    """Size is a percentage of book value."""


@mypyc_attr(allow_interpreted_subclasses=True)
@dataclass(kw_only=True)
class OrderBase:
    """Base class for all orders."""

    status: OrderStatus = OrderStatus.OPEN
    """Status of order."""

    book: BookName | Book | None = field(repr=False, default=None)
    """Target book."""

    suborders: List[OrderBase] = field(default_factory=list)
    """Additional orders to be executed the following timestep."""

    label: Optional[str] = None
    """Label to assist in matching / filtering."""

    priority: int = 0
    """Each day orders are sorted by this field and executed in order."""

    key: Optional[str] = None
    """Unique key for this order.

    If a key is set then only the newest order with this key is kept.
    Older orders with the same key will be removed.
    """

    def __post_init__(self):
        pass

    def _book_trades(self, trades):
        # test then book trades, do any post complete tasks
        if self.book.test_trades(trades):
            self.book.add_transactions(trades)
            self.status = OrderStatus.COMPLETE
            self.post_complete(trades)

        else:
            self.status = OrderStatus.MANDATE_FAILED

    def post_complete(self, trades: List[Trade]):
        """Called after and with trades that have been successfully booked.

        It can append new orders to suborders for execution in the
        following timestep.
        """
        pass

    def apply(
        self, ts: pd.Timestamp, day_data: pd.DataFrame, asset_map: Dict[str, Asset]
    ):
        """Applies order to `self.book` for time `ts` using provided `day_data`
        and dictionary of asset information `asset_map`."""
        raise NotImplementedError("The apply methods needs to be implemented.")


@mypyc_attr(allow_interpreted_subclasses=True)
@dataclass(kw_only=True)
class Order(OrderBase):
    """Simple market order."""

    asset_name: AssetName
    """Asset name for order."""

    size: Decimal
    """Order size."""

    size_type: OrderSizeType = OrderSizeType.QUANTITY
    """Order size type.

    Can be a quantity, notional or book percent.
    """

    def __post_init__(self):
        super().__post_init__()
        self.size = ensure_decimal(self.size)
        self.size_type = ensure_enum(self.size_type, OrderSizeType)

    def _calc_quantity_price(self, day_data, asset_map) -> Tuple[Decimal, Decimal]:
        asset = asset_map[self.asset_name]
        asset_day_data = day_data[asset.data_label]
        trade_price = asset.intraday_traded_price(asset_day_data)

        if self.size_type == OrderSizeType.QUANTITY:
            quantity = self.size
        elif self.size_type == OrderSizeType.NOTIONAL:
            quantity = self.size / trade_price
        elif self.size_type == OrderSizeType.BOOK_PERCENT:
            assert isinstance(self.book, Book)  # to please mypy
            quantity = self.book.cash * self.size / 100 / trade_price
        else:
            raise RuntimeError("Unsupported size type")

        return asset.round_quantity(quantity), trade_price

    def pre_execute_check(
        self, ts: pd.Timestamp, trade_price: Decimal
    ) -> Optional[OrderStatus]:
        """Called with the current timestep and calculated trade price before
        the trade is executed.

        If it returns `None`, the trade is executed as normal. It can
        return `OrderStatus.CANCELLED` to indicate the trade should be
        cancelled or `OrderStatus.OPEN` to indicate the trade should not
        be executed in the current timestep and processed in the
        following timestep.
        """
        return None

    def apply(
        self, ts: pd.Timestamp, day_data: pd.DataFrame, asset_map: Dict[str, Asset]
    ):
        if not self.book or not isinstance(self.book, Book):
            raise RuntimeError("Cannot apply order without book instance")

        trade_quantity, trade_price = self._calc_quantity_price(day_data, asset_map)

        if (new_status := self.pre_execute_check(ts, trade_price)) is not None:
            self.status = new_status
            return

        trades = [
            Trade(
                asset_name=self.asset_name,
                ts=ts,
                quantity=trade_quantity,
                price=trade_price,
                order_label=self.label,
            )
        ]

        self._book_trades(trades)


class PositionalOrderCheckType(Enum):
    POS_TQ_DIFFER = 1
    ZERO_POS = 2


@mypyc_attr(allow_interpreted_subclasses=True)
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
        if not self.book or not isinstance(self.book, Book):
            raise RuntimeError("Cannot apply order without book instance")

        trade_quantity, trade_price = self._calc_quantity_price(day_data, asset_map)

        if (new_status := self.pre_execute_check(ts, trade_price)) is not None:
            self.status = new_status
            return

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
                        order_label=self.label,
                    )
                )
            if trade_quantity != 0:
                trades.append(
                    Trade(
                        asset_name=self.asset_name,
                        ts=ts,
                        quantity=trade_quantity,
                        price=trade_price,
                        order_label=self.label,
                    )
                )

        self._book_trades(trades)


@mypyc_attr(allow_interpreted_subclasses=True)
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
        assets_day_data = [day_data[a.data_label] for a in assets]
        trade_prices = [
            asset.intraday_traded_price(add)
            for asset, add in zip(assets, assets_day_data)
        ]

        if self.size_type == OrderSizeType.QUANTITY:
            quantities = [self.size * w for w in self.weights]
        elif self.size_type == OrderSizeType.NOTIONAL:
            # size = k * sum(w_i * p_i)
            tp_weighted = sum(w * p for w, p in zip(self.weights, trade_prices))
            k = self.size / tp_weighted
            quantities = [k * w for w in self.weights]
        elif self.size_type == OrderSizeType.BOOK_PERCENT:
            assert isinstance(self.book, Book)  # to please mypy
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
            (a.round_quantity(q), tp)
            for a, q, tp in zip(assets, quantities, trade_prices)
        ]

    def apply(
        self, ts: pd.Timestamp, day_data: pd.DataFrame, asset_map: Dict[str, Asset]
    ):
        if not self.book or not isinstance(self.book, Book):
            raise RuntimeError("Cannot apply order without book instance")

        trade_quantity_prices = self._calc_quantity_price(day_data, asset_map)

        trades = [
            Trade(
                asset_name=an,
                ts=ts,
                quantity=tq,
                price=tp,
                order_label=self.label,
            )
            for an, (tq, tp) in zip(self.asset_names, trade_quantity_prices)
        ]

        self._book_trades(trades)


@mypyc_attr(allow_interpreted_subclasses=True)
@dataclass(kw_only=True)
class PositionalBasketOrder(BasketOrder):
    """Similar to a :py:class:`BasketOrder` but will close out existing
    positions if they do not match requested weights."""

    check_type: PositionalOrderCheckType = PositionalOrderCheckType.POS_TQ_DIFFER

    def apply(
        self, ts: pd.Timestamp, day_data: pd.DataFrame, asset_map: Dict[str, Asset]
    ):
        if not self.book or not isinstance(self.book, Book):
            raise RuntimeError("Cannot apply order without book instance")

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
                            order_label=self.label,
                        )
                    )
                if trade_quantity != 0:
                    trades.append(
                        Trade(
                            asset_name=asset_name,
                            ts=ts,
                            quantity=trade_quantity,
                            price=trade_price,
                            order_label=self.label,
                        )
                    )

        self._book_trades(trades)
