from copy import deepcopy
from decimal import Decimal
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa

from .asset import Asset, AssetName
from .asset import OHLCAsset as PyOHLCAsset
from .book import Book, BookMandate
from .order import Order, Orders
from .row_wrapper import MultiIndexRowWrapper
from .strategy import Strategy
from .strategyrunner import StrategyRunnerResult

try:
    import _yabte_backtest_lib as yabte_cpp
except ImportError:
    yabte_cpp = None


class CppStrategyRunner:
    """Delegate for C++ strategy execution."""

    def __init__(
        self,
        data: pd.DataFrame,
        assets: List[Asset],
        strategies: List[Strategy],
        mandates: Dict[AssetName, BookMandate] = None,
        books: List[Book] = None,
    ):
        if not yabte_cpp:
            raise RuntimeError(
                "yabte_cpp extension not found. Please build the C++ extension."
            )

        self.data = data
        self.assets = assets
        self.strategies = strategies
        self.mandates = mandates or {}
        self.books = books or []

        if self.mandates:
            raise ValueError("Mandates are not supported in C++ mode.")

        # Ensure books are initialized if empty
        if not self.books:
            self.books = [Book(name="Main")]

    def _convert_dataframe_to_pyarrow(self, df: pd.DataFrame) -> pa.Table:
        """Convert multi-index columns DataFrame to flat PyArrow table."""
        df_flat = df.copy()

        # Flatten MultiIndex columns: ('AAPL', 'Close') -> 'AAPL.Close'
        if isinstance(df.columns, pd.MultiIndex):
            df_flat.columns = [f"{col[0]}.{col[1]}" for col in df.columns]

        # Ensure index name is set and reset index to make it a column
        if df_flat.index.name is None:
            df_flat.index.name = "Date"

        # Reset index to convert index to a regular column
        df_flat = df_flat.reset_index()

        # Explicitly rename the first column to "Date" if it isn't already (just to be safe)
        # reset_index uses the index name, so it should be correct, but let's be sure.
        if df_flat.columns[0] != "Date":
            df_flat.rename(columns={df_flat.columns[0]: "Date"}, inplace=True)

        # Convert to PyArrow Table
        # We preserve the pandas schema metadata which helps round-tripping if needed,
        # but critically we need the "Date" column to be present in the schema.
        table = pa.Table.from_pandas(df_flat, preserve_index=False)
        return table

    def run(self, params: Dict[str, Any] = None) -> StrategyRunnerResult:
        pa_table = self._convert_dataframe_to_pyarrow(self.data)

        # Sanitize params: Convert Enums to their values for C++ binding compatibility
        sanitized_params = {}
        if params:
            for k, v in params.items():
                if hasattr(v, "value"):  # Handle Enum
                    sanitized_params[k] = v.value
                else:
                    sanitized_params[k] = v

        # print(f"DEBUG: pa_table type: {type(pa_table)}")
        # print(f"DEBUG: pa_table: {pa_table}")

        # Validate that we have a valid PyArrow table
        if pa_table is None:
            raise ValueError("Conversion to PyArrow table resulted in None")

        # 1. Convert Assets
        cpp_assets = []
        for asset in self.assets:
            if isinstance(asset, PyOHLCAsset):
                cpp_asset = yabte_cpp.OHLCAsset(
                    asset.name,
                    asset.denom,
                    asset.price_round_dp,
                    asset.quantity_round_dp,
                    asset.data_label
                    or asset.name,  # Use asset.name as fallback for data_label
                )
            else:
                cpp_asset = yabte_cpp.OHLCAsset(
                    asset.name,
                    asset.denom,
                    asset.price_round_dp,
                    asset.quantity_round_dp,
                    asset.data_label or asset.name,
                )
            cpp_assets.append(cpp_asset)

        # 2. Convert Strategies (Adapter)
        cpp_strategies = []
        for strat in self.strategies:
            strat.params = pd.Series(params or {}, dtype=object)
            # IMPORTANT: C++ engine doesn't inject data into python strategy object automatically.
            # We must set `strat.data` here manually, matching `PythonStrategyRunner`.
            # We provide a copy of the data.
            # Unlock data write for init
            strat._data_lock = False
            strat.data = deepcopy(
                self.data
            )  # Use public setter which might do other things, but unlock first

            cpp_strat = CppStrategyAdapter(strat)
            cpp_strategies.append(cpp_strat)

        # 3. Convert Books
        cpp_books = []
        for book in self.books:
            cpp_book = yabte_cpp.Book(
                book.name,
                book.denom,
                float(book.cash),
                float(book.rate),
                book.interest_round_dp,
                None,  # mandates
            )
            cpp_books.append(cpp_book)

        # 4. Run C++ Engine
        runner = yabte_cpp.StrategyRunner(
            pa_table, cpp_assets, cpp_strategies, cpp_books
        )

        cpp_result = runner.run(sanitized_params)

        # 5. Convert Result back to Python
        result_books = []

        # Helper to convert Arrow Table to Python list of lists for book history
        def table_to_history_list(table):
            if not table:
                return []
            df = table.to_pandas()
            # Ensure columns order matches Python expectation: [ts, cash, mtm, total]
            # Assuming C++ returns them in this order or with these names.
            # If names are reliable:
            return df[["ts", "cash", "mtm", "total"]].values.tolist()

        # Re-construct Python books
        for i, cb in enumerate(cpp_result.books):
            # Create Python Book
            pb = Book(
                name=cb.name,
                denom=cb.denom,
                cash=cb.cash,
                rate=cb.rate,
                interest_round_dp=cb.interest_round_dp,
            )

            # Transactions
            # Imports
            from .transaction import CashTransaction, Trade

            py_trans = []
            for t in cb.transactions:
                if isinstance(t, yabte_cpp.Trade):
                    # Filter out zero quantity trades which might be returned by C++ engine
                    # but are invalid in Python Trade objects.
                    if t.quantity == 0:
                        continue

                    # Manually handle total precision from C++ if needed.
                    # Even though C++ rounds the internal double, converting Double -> Decimal
                    # directly introduces floating point noise. We format to string first.
                    # We assume 2dp based on the C++ rounding logic for Trades.
                    py_t = Trade(
                        ts=t.ts,
                        quantity=t.quantity,
                        price=t.price,
                        asset_name=t.asset_name,
                        order_label=t.order_label,
                        total=Decimal(f"{t.total:.2f}"),
                        desc=t.desc,
                    )
                    py_trans.append(py_t)
                elif isinstance(t, yabte_cpp.CashTransaction):
                    py_ct = CashTransaction(ts=t.ts, total=t.total, desc=t.desc)
                    py_trans.append(py_ct)
                else:
                    pass

            pb.transactions = py_trans

            # History
            cb_hist_table = cb.history
            if cb_hist_table:
                pb._history = table_to_history_list(cb_hist_table)

            result_books.append(pb)

        # Construct Result
        res = StrategyRunnerResult(
            books=result_books,
            strategies=self.strategies,
            assets=self.assets,
        )

        return res


class CppStrategyAdapter(yabte_cpp.Strategy):
    """Adapts a pure Python Strategy to be callable by C++ engine."""

    def __init__(self, python_strategy: Strategy):
        super().__init__()
        self.python_strategy = python_strategy

    def clone(self):
        """Clone method required by C++ PyStrategy trampoline."""
        # Note: self.python_strategy MUST be deepcopy-able.
        new_strat = deepcopy(self.python_strategy)
        return CppStrategyAdapter(new_strat)

    def init(self):
        # Initialize Python-side orders queue if not present
        if self.python_strategy.orders is None:
            self.python_strategy.orders = Orders()

        self.python_strategy.init()

    def _sync_orders(self):
        """Convert Python orders to C++ orders and push to C++ deque."""
        if not self.python_strategy.orders:
            return

        while len(self.python_strategy.orders) > 0:
            py_ord = self.python_strategy.orders.popleft()

            # DEBUG: Print order details
            # print(f"DEBUG: Syncing Order: {py_ord} at TS: {self.python_strategy.ts}")

            # Convert
            cpp_ord = None

            from .order import (
                OrderSizeType,
                PositionalOrder,
                PositionalOrderCheckType,
                SimpleOrder,
            )

            # Handle PositionalOrder by treating it as SimpleOrder for C++ (if possible)
            # OR we need to map PositionalOrder specifically if C++ supports it.
            # yabte_cpp seems to only expose SimpleOrder via binding based on inspection?
            # If C++ supports PositionalOrder, we should map it.
            # Assuming for now only SimpleOrder is robustly exposed or PositionalOrder shares base.
            # Map OrderSizeType
            size_type_map = {
                OrderSizeType.QUANTITY: yabte_cpp.OrderSizeType.QUANTITY,
                OrderSizeType.NOTIONAL: yabte_cpp.OrderSizeType.NOTIONAL,
                OrderSizeType.BOOK_PERCENT: yabte_cpp.OrderSizeType.BOOK_PERCENT,
            }

            # Map PositionalOrderCheckType
            check_type_map = {
                PositionalOrderCheckType.POS_TQ_DIFFER: yabte_cpp.PositionalOrderCheckType.POS_TQ_DIFFER,
                PositionalOrderCheckType.ZERO_POS: yabte_cpp.PositionalOrderCheckType.ZERO_POS,
            }

            # Basic attributes
            asset_name = py_ord.asset_name
            size = float(py_ord.size)
            label = py_ord.label
            priority = py_ord.priority
            key = py_ord.key
            book_name_arg = None
            if py_ord.book:
                if isinstance(py_ord.book, str):
                    book_name_arg = py_ord.book
                elif hasattr(py_ord.book, "name"):
                    book_name_arg = py_ord.book.name

            # Determine C++ Order Class and Size Type
            if isinstance(py_ord, PositionalOrder):
                # PositionalOrder also has size_type
                cpp_size_type = size_type_map.get(
                    py_ord.size_type, yabte_cpp.OrderSizeType.QUANTITY
                )
                cpp_check_type = check_type_map.get(
                    py_ord.check_type, yabte_cpp.PositionalOrderCheckType.POS_TQ_DIFFER
                )

                # Check if C++ side has PositionalOrder
                if hasattr(yabte_cpp, "PositionalOrder"):
                    try:
                        cpp_ord = yabte_cpp.PositionalOrder(
                            asset_name,
                            size,
                            cpp_size_type,
                            cpp_check_type,
                            book_name_arg,
                            label,
                            priority,
                            key,
                        )
                    except Exception as e:
                        print(f"ERROR creating C++ PositionalOrder: {e}")
                else:
                    # Fallback to SimpleOrder if PositionalOrder is not exposed
                    # WARNING: This might change behavior if PositionalOrder has special logic not captured by SimpleOrder + size_type
                    # In Python, PositionalOrder is just a subclass of SimpleOrder that allows different size types.
                    # SimpleOrder in Python defaults to QUANTITY but can take others.
                    # SimpleOrder in C++ takes OrderSizeType.
                    # So mapping PositionalOrder to SimpleOrder in C++ should be valid if logic is same.

                    try:
                        cpp_ord = yabte_cpp.SimpleOrder(
                            asset_name,
                            size,
                            cpp_size_type,
                            book_name_arg,
                            label,
                            priority,
                            key,
                        )
                    except Exception as e:
                        print(
                            f"ERROR creating C++ SimpleOrder (fallback for Positional): {e}"
                        )

            elif isinstance(py_ord, SimpleOrder):
                cpp_size_type = size_type_map.get(
                    py_ord.size_type, yabte_cpp.OrderSizeType.QUANTITY
                )
                try:
                    cpp_ord = yabte_cpp.SimpleOrder(
                        asset_name,
                        size,
                        cpp_size_type,
                        book_name_arg,
                        label,
                        priority,
                        key,
                    )
                except Exception as e:
                    print(f"ERROR creating C++ SimpleOrder: {e}")

            if cpp_ord:
                try:
                    self.orders.append(cpp_ord)
                except Exception as e:
                    print(f"ERROR appending to C++ orders: {e}")
            else:
                print(
                    f"WARNING: C++ Engine skipping unsupported order type: {type(py_ord)}"
                )

    def on_open(self):
        # Update timestamp on Python strategy if C++ engine hasn't done it (which it hasn't)
        # We need the current timestamp from the C++ engine to be consistent.
        # However, `yabte_cpp` Strategy bindings don't expose a way to get the current processing TS easily
        # unless we add it to the `Strategy` class in C++ and expose it.

        # HACK: The C++ engine sets `strategy->data_` to a slice ending at current time.
        # We can infer the timestamp from the last row of `self.data`.
        # `self.data` is a wrapper around `self.data_` (Arrow Table) in C++?
        # In `yabte_cpp.Strategy` binding:
        # .def_property_readonly("data", [](const Strategy &s) -> py::handle { return arrow::py::wrap_table(s.data_); })
        # So `self.data` returns a PyArrow table of the sliced data.

        # Let's get the timestamp from the last row of the PyArrow table.
        # Note: `self.data` here refers to the property on the C++ object (adapter base).
        # We need to access it via `super().data` or similar if we were subclassing directly,
        # but `CppStrategyAdapter` inherits from `yabte_cpp.Strategy`.

        # Check if `self.data` (from C++ base) is available and not None/Empty
        try:
            # self.data is a property of yabte_cpp.Strategy.
            # It returns a pyarrow.Table.
            current_data_slice = self.data
            if current_data_slice and current_data_slice.num_rows > 0:
                # Get last row's "Date" column
                # Assuming "Date" is the timestamp column name we enforced.
                date_col = current_data_slice["Date"]
                current_ts = date_col[
                    -1
                ].as_py()  # Get last element as python datetime/timestamp

                # Set it on the python strategy
                self.python_strategy._ts = pd.Timestamp(current_ts)
        except Exception as e:
            # print(f"WARNING: Could not infer timestamp in on_open: {e}")
            pass

        self.python_strategy.on_open()
        self._sync_orders()

    def on_close(self):
        # Same inference for on_close
        try:
            current_data_slice = self.data
            if current_data_slice and current_data_slice.num_rows > 0:
                date_col = current_data_slice["Date"]
                current_ts = date_col[-1].as_py()
                self.python_strategy._ts = pd.Timestamp(current_ts)
        except Exception as e:
            # print(f"WARNING: Could not infer timestamp in on_close: {e}")
            pass

        self.python_strategy.on_close()
        self._sync_orders()

    def extend_data(self, data):
        pass
