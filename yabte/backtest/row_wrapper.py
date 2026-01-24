from typing import Any, Dict, Mapping, Union

import numpy as np
import pandas as pd
import pyarrow as pa


class MultiIndexRowWrapper:
    """Wraps a PyArrow row (as a dict) to simulate pandas MultiIndex row access.

    This allows accessing `row['Asset'].Close` or `row['Asset']['Close']` while actually
    reading from a flattened PyArrow structure where keys are "Asset.Close".
    """

    def __init__(self, row_dict: Mapping[str, Any]):
        self._row = row_dict
        self._cache: Dict[str, Any] = {}

    def __getitem__(self, key: str) -> Union[pd.Series, Any]:
        """Access asset data or direct fields."""
        if key in self._cache:
            return self._cache[key]

        # Try to find all keys starting with "key."
        # This implies 'key' is an Asset name.
        prefix = f"{key}."
        relevant_items = {
            k[len(prefix) :]: v for k, v in self._row.items() if k.startswith(prefix)
        }

        if relevant_items:
            # Construct a Series-like object for the asset
            # Using simple dict access or a lightweight object is faster than pd.Series
            # But strategies might expect pd.Series methods.
            # For max compatibility, we return a pd.Series.
            # This is overhead, but "Python Mode" is legacy/compatibility mode.

            # Note: PyArrow often converts None/NaN to Python None.
            # Pandas/Numpy operations often expect np.nan for missing floats.
            # `pd.Series` constructor handles `None` -> `NaN` for float dtypes usually,
            # but if data is mixed or object, it might keep None.
            # The error `TypeError: conversion from NoneType to Decimal is not supported`
            # implies something was None that shouldn't be.

            # Helper to convert None to NaN for numeric consistency with pandas
            clean_items = {
                k: (np.nan if v is None else v) for k, v in relevant_items.items()
            }

            series = pd.Series(clean_items)

            # Ensure "Close", "Open", etc. are accessible as attributes

            # pd.Series allows attribute access if index is valid identifiers.
            self._cache[key] = series
            return series

        # If not an asset prefix, maybe it's a direct column (unlikely in current schema but possible)
        if key in self._row:
            return self._row[key]

        # If key is an AssetName but not in row (e.g. no data for this asset on this day)
        # We should return a Series with NaNs or similar, OR raise KeyError.
        # Original df.loc[ts] would return a Series where indices are (Asset, Field).
        # Accessing df.loc[ts][Asset] returns a Series of fields.
        # If data is missing, what happens?
        # In `_check_data`, we ensure assets match data columns.
        # But if a specific row has Nones/NaNs, that's fine.
        # If the *asset* is completely missing from columns, it would fail earlier.

        # If we are here, key wasn't found as "key.Field" prefix.
        # This implies the asset isn't in the flattened columns.
        # Return empty series? Or raise?
        # Standard pandas behavior: if key not in index, raises KeyError.
        raise KeyError(f"Key '{key}' not found in row data.")

    def __iter__(self):
        return iter(self._row)

    def __len__(self):
        return len(self._row)
