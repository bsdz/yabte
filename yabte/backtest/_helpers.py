import logging
from decimal import Decimal
from enum import Enum
from typing import Any, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Enum)


def ensure_enum(value: Any, enum_type: Type[T]) -> T:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        return enum_type[value.upper()]
    if isinstance(value, int):
        return enum_type(value)
    raise ValueError(f"Unexpected enum type {value} for {enum_type}")


def ensure_decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (str, float, int)):
        return Decimal(value)
    raise ValueError(f"Unexpected decimal type {value}")
