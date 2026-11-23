import numbers
import os
from copy import deepcopy
from decimal import Decimal
from typing import Any, Dict, Iterable, List, cast

import numpy as np
import pandas as pd
import pyarrow as pa

from .asset import Asset, AssetName
from .asset import OHLCAsset as PyOHLCAsset
from .book import Book, BookMandate
from .order import Orders, OrderStatus
from .row_wrapper import MultiIndexRowWrapper
from .strategy import Strategy
from .strategyrunner import StrategyRunnerResult

try:
    from . import _yabte_backtest_lib as yabte_cpp  # type: ignore[attr-defined]
except ImportError:
    try:
        import _yabte_backtest_lib as yabte_cpp
    except ImportError:
        yabte_cpp = None

# Extra key added to each ParamMap in ``run_batch`` so a Python strategy, running in
# a C++ worker thread, can recover the original (unsanitized) params for its run.
_BATCH_INDEX_KEY = "__yabte_batch_index__"


def _sanitize_params(params: Dict[str, Any] | None) -> Dict[str, Any]:
    """Convert Enums to their values for C++ binding compatibility."""
    return {
        k: (v.value if hasattr(v, "value") else v) for k, v in (params or {}).items()
    }


def _run_cpp(run, params: Any) -> Any:
    """Call the C++ `run` or `run_batch`, naming the params C++ can't hold if it rejects
    them.

    Only native C++ strategies read the C++ param map, so this only matters for runs
    that include one.
    """
    try:
        return run(params)
    except TypeError as e:
        maps = params if isinstance(params, list) else [params]
        bad = sorted(
            {
                k
                for m in maps
                for k, v in m.items()
                if v is not None and not isinstance(v, (str, numbers.Number))
            }
        )
        if not bad:
            raise
        raise TypeError(
            f"params {bad} can't be passed to a native C++ strategy, which only "
            "takes numbers, strings and bools. Set them on the strategy instead, "
            "or run only Python strategies, which receive their params unchanged."
        ) from e


# Methods the C++ engine reimplements. A subclass that overrides one would run, but
# with the base class's behaviour, so the C++ engine refuses it.
_ASSET_HOOKS = (
    "intraday_traded_price",
    "end_of_day_price",
    "data_fields",
    "check_and_fix_data",
    "round_quantity",
    "transaction_cost",
)
_ORDER_HOOKS = (
    "pre_execute_check",
    "post_complete",
    "apply",
    "_calc_quantity_price",
    "_book_trades",
)


def _overridden(obj: object, base: type, hooks: Iterable[str]) -> List[str]:
    """The `hooks` that `obj`'s class, or `obj` itself, overrides relative to `base`."""
    own = vars(obj) if hasattr(obj, "__dict__") else {}
    return [
        h for h in hooks if h in own or getattr(type(obj), h) is not getattr(base, h)
    ]


def _check_cpp_assets(assets: Iterable[Asset]) -> None:
    """Raise `NotImplementedError` for assets the C++ engine would price wrongly."""
    for asset in assets:
        if not isinstance(asset, PyOHLCAsset):
            raise NotImplementedError(
                f"{type(asset).__name__} ({asset.name}) is not an OHLCAsset; the C++ "
                'engine only prices OHLCAsset. Use engine="python".'
            )
        if hooks := _overridden(asset, PyOHLCAsset, _ASSET_HOOKS):
            raise NotImplementedError(
                f"{type(asset).__name__} ({asset.name}) overrides {', '.join(hooks)}, "
                'which the C++ engine would ignore. Use engine="python", or for a '
                "different fill price, OHLCAsset(fill_price=...)."
            )


def _check_cpp_order(order: object, base: type) -> None:
    """Raise `NotImplementedError` for an order the C++ engine would run wrongly."""
    if hooks := _overridden(order, base, _ORDER_HOOKS):
        raise NotImplementedError(
            f"{type(order).__name__} overrides {', '.join(hooks)}, which the C++ "
            'engine would ignore. Use engine="python".'
        )


class CppStrategyRunner:
    """Delegate for C++ strategy execution."""

    def __init__(
        self,
        data: pd.DataFrame,
        assets: List[Asset],
        strategies: List[Strategy],
        mandates: Dict[AssetName, BookMandate] | None = None,
        books: List[Book] | None = None,
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

        # The C++ engine has no timezones: it finds day boundaries by flooring each
        # timestamp to a day. Give it local wall-clock times so its days are the same
        # local dates the Python engine groups by, and map its output timestamps back.
        self._local_to_aware = None
        index = data.index
        if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
            local = index.tz_localize(None)
            if not local.is_unique:
                repeated = local[local.duplicated()][0]
                raise ValueError(
                    f"data index repeats a local time ({repeated}), as happens when "
                    "clocks go back. The C++ engine needs unique local times; convert "
                    "the index to UTC or use the Python engine."
                )
            ns = local.as_unit("ns").to_numpy().astype("int64").tolist()
            self._local_to_aware = dict(zip(ns, index))

        if self.mandates or any(book.mandates for book in self.books):
            raise ValueError("Mandates are not supported in C++ mode.")
        _check_cpp_assets(self.assets)

        # Ensure books are initialized if empty
        if not self.books:
            self.books = [Book(name="Main")]

    def _convert_dataframe_to_pyarrow(self, df: pd.DataFrame) -> pa.Table:
        """Convert multi-index columns DataFrame to flat PyArrow table."""
        df_flat = df.copy()
        if self._local_to_aware is not None:
            df_flat.index = pd.DatetimeIndex(df_flat.index).tz_localize(None)

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

    def _has_native_strategies(self) -> bool:
        return any(isinstance(s, yabte_cpp.Strategy) for s in self.strategies)

    def run(self, params: Dict[str, Any] | None = None) -> StrategyRunnerResult:
        runner, _keepalive = self._build_runner(params)
        # Python strategies get `params` unchanged on the Python side, so the C++
        # map, which only holds scalars, is only needed by native C++ strategies
        cpp_params = _sanitize_params(params) if self._has_native_strategies() else {}
        return self._convert_result(_run_cpp(runner.run, cpp_params))

    def run_batch(
        self,
        params_iterable: Iterable[Dict[str, Any]],
        num_threads: int | None = None,
    ) -> List[StrategyRunnerResult]:
        """Run each parameter set on the C++ engine's thread pool.

        The data, assets and books are converted once and shared by every run.
        `num_threads` defaults to ``min(len(params), cpu count)``. Native C++ strategies
        run fully in parallel. Python strategies need the GIL for every callback, so
        threads contend for it and run slower than ``num_threads=1``;
        :py:meth:`StrategyRunner.run_batch` uses processes for those instead.
        """
        params_list = [dict(p or {}) for p in params_iterable]
        if not params_list:
            return []

        if num_threads is None:
            num_threads = min(len(params_list), os.cpu_count() or 1)

        runner, _keepalive = self._build_runner(None, batch_params=params_list)
        native = self._has_native_strategies()
        cpp_params = [
            {**(_sanitize_params(p) if native else {}), _BATCH_INDEX_KEY: i}
            for i, p in enumerate(params_list)
        ]
        results = _run_cpp(lambda ps: runner.run_batch(ps, num_threads), cpp_params)
        return [self._convert_result(r) for r in results]

    def _build_runner(
        self,
        params: Dict[str, Any] | None,
        batch_params: List[Dict[str, Any]] | None = None,
    ):
        """Convert data, assets, strategies and books to a C++ StrategyRunner.

        Returns ``(runner, cpp_strategies)``. Callers must hold ``cpp_strategies``
        until the run finishes: C++ only owns the C++ half of each Python adapter,
        so if the Python objects are freed, the trampolines can no longer find the
        Python overrides (e.g. ``clone``).
        """
        pa_table = self._convert_dataframe_to_pyarrow(self.data)

        # print(f"DEBUG: pa_table type: {type(pa_table)}")
        # print(f"DEBUG: pa_table: {pa_table}")

        # Validate that we have a valid PyArrow table
        if pa_table is None:
            raise ValueError("Conversion to PyArrow table resulted in None")

        # 1. Convert Assets (_check_cpp_assets has rejected anything else)
        fill_price_map = {
            "midpoint": yabte_cpp.FillPrice.MIDPOINT,
            "open": yabte_cpp.FillPrice.OPEN,
            "close": yabte_cpp.FillPrice.CLOSE,
        }
        cpp_assets = [
            yabte_cpp.OHLCAsset(
                asset.name,
                asset.denom,
                asset.price_round_dp,
                asset.quantity_round_dp,
                asset.data_label or asset.name,
                fill_price_map[asset.fill_price],
                float(asset.cost),
                yabte_cpp.CostType(asset.cost_type.value),
            )
            for asset in cast(List[PyOHLCAsset], self.assets)
        ]

        # 2. Convert Strategies (Adapter)
        cpp_strategies = []
        for strat in self.strategies:
            # Check if strat is already a C++ strategy instance
            if yabte_cpp and isinstance(strat, yabte_cpp.Strategy):
                # It's already a C++ object.
                # We skip setting params/data/assets because C++ strategies
                # should be self-contained or initialized differently.
                # If they support param injection via property, we could try setting it,
                # but currently our bindings expose params as readonly.
                cpp_strategies.append(strat)
                continue

            # configure a copy, leaving the caller's strategy untouched
            strat = deepcopy(strat)
            strat.params = pd.Series(params or {}, dtype=object)
            # IMPORTANT: C++ engine doesn't inject data into python strategy object automatically.
            # We must set `strat.data` here manually, matching `PythonStrategyRunner`.
            # We provide a copy of the data.
            # Unlock data write for init
            strat._data_lock = False
            strat.data = deepcopy(
                self.data
            )  # Use public setter which might do other things, but unlock first

            # Also inject assets map into strategy, which PythonStrategyRunner does
            asset_map = {asset.name: asset for asset in self.assets}
            strat.assets = asset_map

            cpp_strat = CppStrategyAdapter(strat, batch_params)
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

        # 4. Build C++ Engine
        runner = yabte_cpp.StrategyRunner(
            pa_table, cpp_assets, cpp_strategies, cpp_books
        )
        return runner, cpp_strategies

    def _ts(self, ts_ns: int) -> pd.Timestamp:
        """Timestamp from the C++ engine, in the data's timezone if it has one."""
        if self._local_to_aware is not None:
            return self._local_to_aware[ts_ns]
        return pd.Timestamp(ts_ns, unit="ns")

    def _convert_result(self, cpp_result) -> StrategyRunnerResult:
        """Convert a C++ StrategyRunnerResult back to Python."""
        result_books = []

        # Helper to convert Arrow Table to Python list of lists for book history
        def table_to_history_list(table):
            if not table:
                return []
            df = table.to_pandas()
            if self._local_to_aware is not None:
                df["ts"] = [self._ts(ts.value) for ts in df.ts]
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
            from .asset import CostType
            from .transaction import CashTransaction, Cost, Trade, Transaction

            py_trans: List[Transaction] = []
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
                        ts=self._ts(t.ts_ns),
                        quantity=t.quantity,
                        price=t.price,
                        asset_name=t.asset_name,
                        order_label=t.order_label,
                        total=Decimal(f"{t.total:.2f}"),
                        desc=t.desc,
                    )
                    py_trans.append(py_t)
                elif isinstance(t, yabte_cpp.Cost):
                    py_trans.append(
                        Cost(
                            ts=self._ts(t.ts_ns),
                            total=Decimal(str(t.total)),
                            desc=t.desc,
                            asset_name=t.asset_name,
                            order_label=t.order_label,
                            charged_quantity=Decimal(str(t.charged_quantity)),
                            cost_type=CostType(int(t.cost_type)),
                        )
                    )
                elif isinstance(t, yabte_cpp.CashTransaction):
                    py_ct = CashTransaction(
                        ts=self._ts(t.ts_ns), total=t.total, desc=t.desc
                    )
                    py_trans.append(py_ct)
                else:
                    pass

            pb.transactions = py_trans

            # History
            cb_hist_table = cb.history
            if cb_hist_table:
                pb._history = table_to_history_list(cb_hist_table)

            result_books.append(pb)

        # This run's strategies: C++ cloned each one, so for Python strategies the
        # adapter's copy is the one that ran.
        strategies = []
        py_orders = {}
        for s in cpp_result.strategies:
            if isinstance(s, CppStrategyAdapter):
                strategies.append(s.python_strategy)
                py_orders.update(s._py_orders)
            else:
                strategies.append(s)

        def to_python(cpp_ord):
            # orders a Python strategy placed come back as the Python order, with
            # the engine's status; others (from native C++ strategies) as they are
            py_ord = py_orders.get(id(cpp_ord))
            if py_ord is None:
                return cpp_ord
            py_ord.status = OrderStatus(cpp_ord.status)
            return py_ord

        orders_unprocessed = Orders()
        orders_unprocessed.extend(to_python(o) for o in cpp_result.orders_unprocessed)
        return StrategyRunnerResult(
            books=result_books,
            strategies=strategies,
            assets=deepcopy(self.assets),
            _orders_unprocessed=orders_unprocessed,
            _orders_processed=[to_python(o) for o in cpp_result.orders_processed],
        )


class CppStrategyAdapter(yabte_cpp.Strategy if yabte_cpp else object):  # type: ignore[misc]
    """Adapts a pure Python Strategy to be callable by C++ engine."""

    def __init__(
        self,
        python_strategy: Strategy,
        batch_params: List[Dict[str, Any]] | None = None,
    ):
        if yabte_cpp:
            super().__init__()
        self.python_strategy = python_strategy
        # Each C++ order this adapter made, by id, with the Python order it came
        # from. Holding the C++ object keeps its id unique and lets pybind return
        # the same Python object for it in the result.
        self._py_orders: Dict[int, Any] = {}
        self._cpp_orders: List[Any] = []
        # Original params for each run of a batch, indexed by _BATCH_INDEX_KEY.
        self._batch_params = batch_params

    def clone(self):
        """Clone method required by C++ PyStrategy trampoline."""
        # Note: self.python_strategy MUST be deepcopy-able.
        new_strat = deepcopy(self.python_strategy)
        return CppStrategyAdapter(new_strat, self._batch_params)

    def init(self):
        # In a batch, give this run's own params to the Python strategy. They come
        # from the original list rather than the C++ ParamMap so that values such as
        # Enums keep their Python types, matching the Python engine.
        if self._batch_params is not None:
            ix = self.params[_BATCH_INDEX_KEY]
            self.python_strategy.params = pd.Series(
                self._batch_params[ix], dtype=object
            )

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

            # Reject anything the C++ engine has no counterpart for before
            # touching order attributes, so the caller gets a clear error
            # instead of an AttributeError from the conversion below.
            # BasketOrder/PositionalBasketOrder derive from Order, not
            # SimpleOrder, and describe several assets at once.
            if not isinstance(py_ord, SimpleOrder):
                raise NotImplementedError(
                    f"{type(py_ord).__name__} is not supported by the C++ "
                    'engine; use engine="python" for this strategy.'
                )

            _check_cpp_order(
                py_ord,
                PositionalOrder if isinstance(py_ord, PositionalOrder) else SimpleOrder,
            )

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
                            py_ord.close_first,
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"could not convert {type(py_ord).__name__} to a "
                            f"C++ PositionalOrder: {e}"
                        ) from e
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
                        raise RuntimeError(
                            f"could not convert {type(py_ord).__name__} to a "
                            f"C++ PositionalOrder: {e}"
                        ) from e

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
                    raise RuntimeError(
                        f"could not convert {type(py_ord).__name__} to a "
                        f"C++ SimpleOrder: {e}"
                    ) from e

            if cpp_ord is None:
                # Unreachable: unsupported types are rejected above and the
                # conversions raise on failure. Guard anyway, because silently
                # dropping an order would let a backtest trade nothing.
                raise AssertionError(
                    f"no C++ order produced for {type(py_ord).__name__}"
                )
            self._py_orders[id(cpp_ord)] = py_ord
            self._cpp_orders.append(cpp_ord)
            self.orders.append(cpp_ord)

    def _set_window(self):
        """Point the Python strategy at the engine's current timestep.

        The C++ engine keeps the full table on ``self.data`` and tracks position via
        ``current_idx``. The Python ``Strategy.data`` property already windows lazily on
        ``self.ts`` (``self._data.loc[: self.ts]``), so all that's needed to mirror
        ``PythonStrategyRunner`` is a correct ``_ts``. Reading it from the pandas index
        is O(1) and avoids an Arrow round-trip.
        """
        idx = self.current_idx
        if idx < 0:
            return
        self.python_strategy._set_ts(self.python_strategy._data.index[idx])

    def on_open(self):
        self._set_window()
        self.python_strategy._mask_open = True
        try:
            self.python_strategy.on_open()
        finally:
            self.python_strategy._mask_open = False
        self._sync_orders()

    def on_close(self):
        self._set_window()
        self.python_strategy.on_close()
        self._sync_orders()

    def extend_data(self, data):
        pass
