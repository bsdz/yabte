"""Strict engine-parity assertions shared by the Python-vs-C++ comparison tests.

These are deliberately unforgiving: nothing is reindexed or filled. A row that is
missing, shifted or mislabelled on either side is a failure, never a zero.
"""

import os

import numpy as np
import pandas as pd

try:
    import _yabte_backtest_lib  # noqa: F401

    HAS_CPP = True
except ImportError:
    HAS_CPP = False

# Set YABTE_REQUIRE_CPP=1 to turn "C++ engine not built, skipping" into a
# failure. CI sets it so a missing extension cannot quietly reduce the parity
# suite to nothing -- which is exactly how several engine bugs went unnoticed.
REQUIRE_CPP = os.environ.get("YABTE_REQUIRE_CPP", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def skip_or_fail_without_cpp(test_case):
    """Skip the calling test unless the C++ engine is required but missing."""
    if HAS_CPP:
        return
    if REQUIRE_CPP:
        test_case.fail(
            "YABTE_REQUIRE_CPP is set but the C++ extension could not be "
            "imported, so engine parity was not tested. Build it with "
            "`nox -s 'build-cpp(debug)'` and put cpp/build/Debug/pybind on "
            "PYTHONPATH."
        )
    test_case.skipTest("C++ engine not available")


# Columns that together identify a transaction; used for canonical ordering and
# for exact/near-exact comparison. ``desc`` is free text and deliberately excluded.
TX_EXACT_COLUMNS = ["ts", "book", "asset_name"]
TX_NUMERIC_COLUMNS = ["quantity", "price", "total"]


def canonical_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    """Return ``tx`` in a deterministic order with normalised dtypes.

    Both engines may emit same-day trades in a different order, so sort on every
    identifying column rather than ``ts`` alone. Cash transactions (interest) have no
    asset/quantity/price; those are normalised to ``""``/``NaN`` so a frame that mixes
    trades and cash rows compares cleanly.
    """
    tx = tx.copy()
    for col in TX_EXACT_COLUMNS:
        if col not in tx:
            tx[col] = ""
    for col in TX_NUMERIC_COLUMNS:
        if col not in tx:
            tx[col] = np.nan
    tx["ts"] = pd.to_datetime(tx["ts"])
    tx["book"] = tx["book"].fillna("").astype(str)
    tx["asset_name"] = tx["asset_name"].fillna("").astype(str)
    for col in TX_NUMERIC_COLUMNS:
        tx[col] = tx[col].astype(float)
    return (
        tx.sort_values(TX_EXACT_COLUMNS + TX_NUMERIC_COLUMNS, kind="stable")
        .reset_index(drop=True)
        .loc[:, TX_EXACT_COLUMNS + TX_NUMERIC_COLUMNS]
    )


class ParityAssertionsMixin:
    """Assertions comparing a Python-engine result against a C++-engine result.

    Mix into a ``unittest.TestCase``. The Python result is the reference; messages are
    phrased as C++ deviating from it.
    """

    def assertBookHistoryParity(
        self,
        hist_py: pd.DataFrame,
        hist_cpp: pd.DataFrame,
        context="",
        rtol=1e-3,
        atol=1e-3,
    ):
        # Index must match by value. Resolution may differ (pandas>=3 defaults the
        # DatetimeIndex to ``us``; the C++ history comes back as ``ns``) and
        # ``Index.equals`` compares values, not units.
        if not hist_py.index.equals(hist_cpp.index):
            self.fail(
                f"{context}: book history index mismatch\n"
                f"  python: n={len(hist_py)} {_index_summary(hist_py.index)}\n"
                f"  c++   : n={len(hist_cpp)} {_index_summary(hist_cpp.index)}"
            )

        self.assertEqual(
            set(hist_py.columns),
            set(hist_cpp.columns),
            f"{context}: book history columns mismatch",
        )
        # Reorder columns only; rows have already been proven identical.
        hist_cpp = hist_cpp[hist_py.columns]

        for col in hist_py.columns:
            py_vals = hist_py[col].to_numpy(dtype=float)
            cpp_vals = hist_cpp[col].to_numpy(dtype=float)
            # equal_nan=False: a NaN on either side is a discrepancy.
            close = np.isclose(py_vals, cpp_vals, rtol=rtol, atol=atol)
            if not close.all():
                i = int(np.argmax(~close))
                self.fail(
                    f"{context}: book history mismatch in column {col}: "
                    f"{(~close).sum()} of {len(close)} rows differ; first at "
                    f"{hist_py.index[i]}: python={py_vals[i]!r} c++={cpp_vals[i]!r}"
                )

    def assertTransactionParity(
        self,
        tx_py: pd.DataFrame,
        tx_cpp: pd.DataFrame,
        context="",
        rtol=1e-5,
        atol=1e-5,
    ):
        self.assertEqual(
            len(tx_py), len(tx_cpp), f"{context}: transaction count mismatch"
        )
        if tx_py.empty:
            return

        tx_py = canonical_transactions(tx_py)
        tx_cpp = canonical_transactions(tx_cpp)

        for col in TX_EXACT_COLUMNS:
            py_vals = tx_py[col].to_numpy()
            cpp_vals = tx_cpp[col].to_numpy()
            diff = py_vals != cpp_vals
            if diff.any():
                i = int(np.argmax(diff))
                self.fail(
                    f"{context}: transaction {col} mismatch: {diff.sum()} of "
                    f"{len(diff)} rows differ; first at row {i}:\n"
                    f"  python: {tx_py.iloc[i].to_dict()}\n"
                    f"  c++   : {tx_cpp.iloc[i].to_dict()}"
                )

        for col in TX_NUMERIC_COLUMNS:
            py_vals = tx_py[col].to_numpy()
            cpp_vals = tx_cpp[col].to_numpy()
            # equal_nan=True only so that cash rows (no quantity/price on either
            # side) compare equal; a NaN against a number still fails.
            close = np.isclose(py_vals, cpp_vals, rtol=rtol, atol=atol, equal_nan=True)
            if not close.all():
                i = int(np.argmax(~close))
                self.fail(
                    f"{context}: transaction {col} mismatch: {(~close).sum()} of "
                    f"{len(close)} rows differ; first at row {i}:\n"
                    f"  python: {tx_py.iloc[i].to_dict()}\n"
                    f"  c++   : {tx_cpp.iloc[i].to_dict()}"
                )

    def assertResultParity(self, res_py, res_cpp, context=""):
        """Compare two ``StrategyRunnerResult`` objects: book history then trades."""
        self.assertBookHistoryParity(
            res_py.book_history, res_cpp.book_history, context=context
        )
        self.assertTransactionParity(
            res_py.transaction_history, res_cpp.transaction_history, context=context
        )


def _index_summary(index: pd.Index) -> str:
    if len(index) == 0:
        return "(empty)"
    return f"[{index[0]} .. {index[-1]}] dtype={index.dtype}"
