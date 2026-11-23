from __future__ import annotations

import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from ._helpers import ensure_decimal, ensure_enum
from .asset import Asset, AssetName, _require_price
from .book import Book, BookName
from .transaction import Cost, Trade

logger = logging.getLogger(__name__)

__all__ = ["SimpleOrder", "PositionalOrder", "BasketOrder", "PositionalBasketOrder"]


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
    """Size is a percentage of the book's cash.

    For basket orders, each weight is instead a percentage of the book's cash plus its
    positions in the basket's assets, valued at the traded prices, and size is ignored.
    """


class Orders:
    """Double ended queue of orders."""

    def __init__(self):
        self.deque = deque()

    def __len__(self):
        return len(self.deque)

    def __iter__(self):
        return iter(self.deque)

    def popleft(self):
        return self.deque.popleft()

    def append(self, order: Order):
        return self.deque.append(order)

    def extend(self, orders: Iterable[Order]):
        return self.deque.extend(orders)

    def sort_by_priority(self):
        """Sorts orders by order priority."""
        ou_sorted = sorted(self.deque, key=lambda o: o.priority, reverse=True)
        self.deque.clear()
        self.deque.extend(ou_sorted)

    def remove_duplicate_keys(self) -> List[Order]:
        """Remove older orders with same key.

        Returns a list of orders than were removed with status set to REPLACED.
        """
        removed = []
        cntr = Counter(o.key for o in self.deque if o.key is not None)
        if any(v > 1 for v in cntr.values()):
            kept = []
            while self.deque:
                o = self.deque.popleft()
                if o.key in cntr and cntr[o.key] > 1:
                    o.status = OrderStatus.REPLACED
                    removed.append(o)
                    cntr[o.key] -= 1
                else:
                    kept.append(o)
            self.deque.clear()
            self.deque.extend(kept)

        return removed


@dataclass(kw_only=True)
class Order:
    """Base class for all orders."""

    status: OrderStatus = OrderStatus.OPEN
    """Status of order."""

    book: BookName | Book | None = field(repr=False, default=None)
    """Target book."""

    suborders: List[Order] = field(default_factory=list)
    """Additional orders to be executed the following timestep."""

    label: Optional[str] = None
    """Label to assist in matching / filtering."""

    priority: int = 0
    """Each day orders are sorted by this field and executed in order."""

    key: Optional[str] = None
    """Unique key for this order.

    If a key is set then only the newest order with this key is kept: before each
    timestep's orders execute, and again after, older queued orders with the same key
    are removed with status `REPLACED`.
    """

    def __post_init__(self):
        pass

    def _costs(self, trades: List[Trade], asset_map: Dict[str, Asset]) -> List[Cost]:
        """Each asset's transaction cost, charged on the net quantity `trades` trade in
        it, so closing and reopening a position pays only on the difference."""
        net: Dict[AssetName, Decimal] = {}
        prices: Dict[AssetName, Decimal] = {}
        for t in trades:
            net[t.asset_name] = net.get(t.asset_name, Decimal(0)) + t.quantity
            prices.setdefault(t.asset_name, t.price)
        costs = []
        for asset_name, quantity in net.items():
            asset = asset_map[asset_name]
            cost = asset.transaction_cost(quantity, prices[asset_name])
            if cost != 0:
                costs.append(
                    Cost(
                        ts=trades[0].ts,
                        total=-cost,
                        asset_name=asset_name,
                        order_label=self.label,
                        charged_quantity=quantity,
                        cost_type=asset.cost_type,
                    )
                )
        return costs

    def _book_trades(self, trades: List[Trade], asset_map: Dict[str, Asset]):
        # test then book trades and their costs, do any post complete tasks
        assert isinstance(self.book, Book)  # apply has resolved the book name
        if self.book.test_trades(trades):
            self.book.add_transactions([*trades, *self._costs(trades, asset_map)])
            self.status = OrderStatus.COMPLETE
            self.post_complete(trades)

        else:
            self.status = OrderStatus.MANDATE_FAILED

    def post_complete(self, trades: List[Trade]):
        """Called after and with trades that have been successfully booked.

        It can append new orders to suborders for execution in the following timestep.
        """
        pass

    def apply(self, ts: pd.Timestamp, day_data: pd.Series, asset_map: Dict[str, Asset]):
        """Applies order to `self.book` for time `ts` using provided `day_data` and
        dictionary of asset information `asset_map`."""
        raise NotImplementedError("The apply methods needs to be implemented.")


@dataclass(kw_only=True)
class SimpleOrder(Order):
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

    def _calc_quantity_price(
        self, day_data: pd.Series, asset_map: Dict[str, Asset]
    ) -> Tuple[Decimal, Decimal]:
        asset = asset_map[self.asset_name]
        asset_day_data = asset._filter_data(day_data)
        # set to ts since MultiIndexRowWrapper doesn't support .name access for timestamp, and some asset price methods might need it
        asset_day_data.name = day_data.timestamp
        trade_price = _require_price(
            asset.intraday_traded_price(asset_day_data, size=self.size),
            asset.name,
            day_data.timestamp,
            "traded",
        )

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
        """Called with the current timestep and calculated trade price before the trade
        is executed.

        If it returns `None`, the trade is executed as normal. It can return
        `OrderStatus.CANCELLED` to indicate the trade should be cancelled or
        `OrderStatus.OPEN` to indicate the trade should not be executed in the current
        timestep and processed in the following timestep.
        """
        return None

    def apply(self, ts: pd.Timestamp, day_data: pd.Series, asset_map: Dict[str, Asset]):
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

        self._book_trades(trades, asset_map)


class PositionalOrderCheckType(Enum):
    POS_TQ_DIFFER = 1
    ZERO_POS = 2


def _trades_to_target(
    order: Order,
    ts: pd.Timestamp,
    asset: Asset,
    current_position: Decimal,
    trade_quantity: Decimal,
    trade_price: Decimal,
    close_first: bool,
) -> List[Trade]:
    """Trades taking `current_position` to `trade_quantity`: a close and an open if
    `close_first`, otherwise one trade for the difference."""

    def trade(quantity):
        return Trade(
            asset_name=asset.name,
            ts=ts,
            quantity=quantity,
            price=trade_price,
            order_label=order.label,
        )

    if not close_first:
        difference = asset.round_quantity(trade_quantity - current_position)
        return [trade(difference)] if difference != 0 else []
    trades = []
    if current_position:
        trades.append(trade(-current_position))  # close out existing position
    if trade_quantity != 0:
        trades.append(trade(trade_quantity))
    return trades


@dataclass(kw_only=True)
class PositionalOrder(SimpleOrder):
    """Ensures current position is `size` and will close out existing positions to
    achieve this."""

    check_type: PositionalOrderCheckType = PositionalOrderCheckType.POS_TQ_DIFFER
    """Condition type to determine if a trade is required."""

    close_first: bool = True
    """How to reach the target when a trade is needed.

    `True` closes the current position and opens the target, booking two trades. `False`
    books one trade for the difference, as a rebalance would.
    """

    def __post_init__(self):
        super().__post_init__()
        self.check_type = ensure_enum(self.check_type, PositionalOrderCheckType)

    def apply(self, ts: pd.Timestamp, day_data: pd.Series, asset_map: Dict[str, Asset]):
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
            trades = _trades_to_target(
                self,
                ts,
                asset_map[self.asset_name],
                current_position,
                trade_quantity,
                trade_price,
                self.close_first,
            )

        self._book_trades(trades, asset_map)


@dataclass
class BasketOrder(Order):
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
        self, day_data: pd.Series, asset_map: Dict[str, Asset]
    ) -> List[Tuple[Decimal, Decimal]]:
        assets = [asset_map[an] for an in self.asset_names]
        assets_day_data = [a._filter_data(day_data) for a in assets]
        trade_prices = [
            _require_price(
                asset.intraday_traded_price(add, size=self.size),
                asset.name,
                day_data.timestamp,
                "traded",
            )
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

    def apply(self, ts: pd.Timestamp, day_data: pd.Series, asset_map: Dict[str, Asset]):
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

        self._book_trades(trades, asset_map)


@dataclass(kw_only=True)
class PositionalBasketOrder(BasketOrder):
    """Similar to a :py:class:`BasketOrder` but will close out existing positions if
    they do not match requested weights."""

    check_type: PositionalOrderCheckType = PositionalOrderCheckType.POS_TQ_DIFFER
    """Condition type to determine if a trade is required."""

    close_first: bool = True
    """As :py:attr:`PositionalOrder.close_first`, for each asset in the basket."""

    def apply(self, ts: pd.Timestamp, day_data: pd.Series, asset_map: Dict[str, Asset]):
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
                trades += _trades_to_target(
                    self,
                    ts,
                    asset_map[asset_name],
                    current_position,
                    trade_quantity,
                    trade_price,
                    self.close_first,
                )

        self._book_trades(trades, asset_map)
