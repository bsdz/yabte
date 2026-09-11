# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`yabte` ("Yet Another BackTesting Engine") is a Python library for backtesting trading
strategies, event-driven (`on_open`/`on_close` hooks), with multi-asset/multi-book
support and pluggable order types. A C++ backend (via pybind11) provides an
accelerated, mostly-parallel implementation of the same object model for
performance-critical runs; the pure-Python engine remains the default and the
reference implementation. Licensed CC-BY-NC-4.0 — not for use at a bank, hedge fund,
commodity house, etc. without permission from the author.

## Development workflow

This project uses `nox` for task automation and Poetry-style dependency groups declared
in `pyproject.toml`. Prefer `nox` sessions over invoking tools manually so environments
stay consistent.

```bash
nox -s do-lint          # apply black, isort, docformatter
nox -s check_lint        # check formatting/imports/docstrings without modifying files
nox -s test               # build nothing; run full python unittest suite + coverage report
nox -s build-cpp -- debug     # conan install + cmake configure/build + ctest, Debug
nox -s build-cpp -- release   # same, Release
nox -s clean-cpp -- debug|release
```

`nox -s test` expects the C++ extension to already be built at
`cpp/build/{Debug,Release}/pybind` (it adds that path to `PYTHONPATH`); run `build-cpp`
first if `runner_cpp` / `engine="cpp"` tests need to pass. Default nox sessions (bare
`nox`) run `do-lint`, `build-cpp(debug)`, then `test-3.13(debug)`.

To pass a filename pattern to the test session: `nox -s test -- "test_asset.py"`
(matches unittest discover's `-p` pattern argument).

### Running Python tests directly

The project uses standard-library `unittest`, not pytest (pytest is only a dev
dependency, not the runner). Tests live alongside the code they test (e.g.
`yabte/backtest/tests/`, `yabte/utilities/portopt/tests/`), not in one central tree.

```bash
python -m unittest yabte.backtest.tests.test_strategy_runner.StrategyRunnerTestCase
python -m unittest yabte.backtest.tests.test_strategy_runner.StrategyRunnerTestCase.test_some_case
python -m unittest discover -v -s ./yabte/ -p "test_*.py"
```

Install dev deps first: `pip install .[dev]` or `poetry install`. If exercising the C++
engine, set `PYTHONPATH=cpp/build/Debug/pybind` (or `Release`) so `runner_cpp.py` can
import the compiled `_yabte_backtest_lib` extension — it falls back to `yabte_cpp =
None` and raises at runtime if not found.

If a nox-managed environment already exists, its interpreter (e.g.
`.nox/test-3-13-debug/bin/python`) has all dev/notebook dependencies preinstalled and is
the fastest way to run one-off commands without reinstalling anything.

### C++ build (fast path for iteration)

`nox -s build-cpp` creates a fresh environment each time, which can be slow. For
iterating on C++ directly, activate a venv first (cmake needs to find the right Python
headers/libs) and use system `conan` (v2.x):

```bash
source .venv_build/bin/activate
cd cpp
conan install . --output-folder=build/Release --build=missing --settings=build_type=Release
cd build/Release
cmake ../.. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cd src_test && ctest --output-on-failure -C Release
```

C++23, CMake 3.20+, Conan 2.x, GoogleTest, pybind11. Headers in `cpp/include/`, sources
in `cpp/src/`, tests in `cpp/src_test/`.

**Two Arrows problem:** when linking the C++ extension, it must use the `pyarrow`
shared libraries from the installed Python wheel (`libarrow_python.so`, located via
`find_library` in the pyarrow package dir in `CMakeLists.txt`), not Conan's static
Arrow libs — otherwise passing `pyarrow.Table` objects between Python and C++ fails with
`RuntimeError: Error converting pyarrow table to arrow table: ... Could not unwrap
Table...`.

## Code style

- Formatting is enforced by `black` (line length 88) + `isort` (`profile = "black"`) +
  `docformatter`; run `nox -s do-lint` before committing (`_unittest_numpy_extensions.py`
  is excluded from docformatter).
- `mypy` is configured in `pyproject.toml`; `scipy`/`plotly`/`matplotlib` are exempted
  from missing-import errors.
- Optional heavy dependencies (e.g. `scipy`) are imported in a `try/except` that sets a
  `HAS_X` flag; corresponding tests use `@unittest.skipUnless(HAS_X, "needs X")`.
- Numerical tests inherit `yabte.tests._unittest_numpy_extensions.NumpyTestCase` and use
  `self.numpyAssertAllclose(expected, actual, atol=...)` rather than plain equality.
  Seed RNGs deterministically (`np.random.default_rng(SEED)`) in `setUp`.
- Extend `pandas.DataFrame` behavior via a registered accessor
  (`@pd.api.extensions.register_dataframe_accessor(...)`, see
  `yabte/utilities/pandas_extension.py`), not by subclassing `DataFrame`.

## Architecture

### Core object model (`yabte/backtest/`)

The engine is built around five cooperating dataclasses, all re-exported from
`yabte/backtest/__init__.py`:

- **`Asset`** (`asset.py`) — anything with a price. `OHLCAsset` is the concrete
  implementation supporting High/Low/Open/Close/Volume fields; subclass `Asset` to
  support other schemas (e.g. bid/ask, custom fields). `data_fields()` declares which
  fields exist and when they're visible (`ADFI_AVAILABLE_AT_OPEN` /
  `ADFI_AVAILABLE_AT_CLOSE` / `ADFI_REQUIRED` bitflags) — this drives the open/close data
  masking described below.
- **`Book`** (`book.py`) — tracks cash, positions, and transaction history for one
  ledger. `eod_tasks()` accrues continuously-compounded interest and marks-to-market
  using each asset's `end_of_day_price`.
- **`Order`** family (`order.py`) — `SimpleOrder` (market order, `QUANTITY`/`NOTIONAL`/
  `BOOK_PERCENT` sizing), `PositionalOrder` (trades to reach a target position),
  `BasketOrder`/`PositionalBasketOrder` (multi-asset weighted variants). Orders are
  processed from an `Orders` double-ended queue each timestep, sorted by `priority`;
  `key` lets a newer order supersede an older one with the same key
  (`remove_duplicate_keys`). `apply()` computes trades and calls `_book_trades()`, which
  checks the book's mandates before booking.
- **`Strategy`** (`strategy.py`) — user-authored subclasses implement `init()` (run once,
  can enrich `self.data`), `on_open()`, and `on_close()` (run every timestep). `self.data`
  is a *view* of the full dataset windowed up to `self.ts`; during `on_open()` an
  additional mask hides same-day fields not yet available at open (anything not in
  `ADFI_AVAILABLE_AT_OPEN`) so strategies can't peek at the day's close/high/low early.
  `self.orders`, `self.books`, `self.assets` are injected by the runner.
- **`StrategyRunner`** (`strategyrunner.py`) — the facade. Validates `data` (must have a
  unique, monotonic `DatetimeIndex` and a 2-level `(asset, field)` column MultiIndex
  covering every asset), then dispatches to an engine and returns a
  `StrategyRunnerResult` (deep copies of assets/books/strategies plus processed/
  unprocessed orders — the result is decoupled from the runner's own state).
  `run_batch()` runs a parameter sweep across a `ProcessPoolExecutor`/`ThreadPoolExecutor`
  for optimization.

### Two engines, one model

`StrategyRunner(engine=...)` selects between:
- `"python"` (default, `PythonStrategyRunner` in `strategyrunner.py`) — the reference
  implementation. Its per-timestep loop converts the flattened DataFrame to a PyArrow
  table and iterates row-by-row via `MultiIndexRowWrapper` (`row_wrapper.py`) instead of
  repeated `df.loc[ts]` lookups, for speed.
- `"cpp"` (`CppStrategyRunner` in `runner_cpp.py`) — converts data/assets/books to the
  compiled `_yabte_backtest_lib` pybind11 extension (built from `cpp/`) and runs the
  event loop in C++. **Mandates are not supported in this engine** (raises `ValueError`
  if any are passed). The extension is optional at import time; `runner_cpp` degrades
  gracefully to `yabte_cpp = None` and only raises when the engine is actually
  instantiated.

Because both engines must produce identical results for the same strategy, changes to
order/book/asset semantics need to be mirrored in both the Python classes
(`yabte/backtest/*.py`) and the corresponding C++ classes (`cpp/include/YABTE/BackTest/`,
`cpp/src/YABTE/BackTest/`), and validated by `yabte/backtest/tests/test_parity.py`
(same strategy run against both engines, results compared) in addition to
`test_cpp_integration.py` and the C++ unit tests in `cpp/src_test/`.

### `yabte/utilities/`

Supporting, mostly-independent modules: `pandas_extension.py` (DataFrame accessors),
`portopt/` (portfolio optimization: hierarchical risk parity, inverse volatility,
minimum variance), `simulation/` (GBM, Heston, Wiener process generators for synthetic
test data), `strategy_helpers.py` (e.g. `crossover`), `pyfolio.py` and `plot/{matplotlib,
plotly}/` (result visualization — third-party stats like Sharpe/drawdown are expected to
come from `empyrical`/`pyfolio`, not this library).

### Benchmarks

`yabte/tests/benchmark/` (`benchmark_cpp_vs_py.py`, `profile_scenarios.py`, `portopt.py`)
compares engine performance; not part of the standard `unittest discover` correctness
suite.
