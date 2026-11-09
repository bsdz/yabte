"""Tests of the parity assertions themselves.

Each case starts from a pair of results that *should* compare equal, introduces one
specific discrepancy, and asserts the harness rejects it. Several of these were accepted
by the previous harness (which reindexed and zero-filled the book history and compared
transactions only by count and per-asset sum).
"""

import unittest
from decimal import Decimal

import pandas as pd

from yabte.backtest.tests._parity import ParityAssertionsMixin


class _Result:
    """Minimal stand-in for ``StrategyRunnerResult``."""

    def __init__(self, book_history, transaction_history):
        self.book_history = book_history
        self.transaction_history = transaction_history


def _book_history(index, cash):
    cols = pd.MultiIndex.from_product([["Main"], ["cash", "mtm", "total"]])
    return pd.DataFrame(
        {
            ("Main", "cash"): cash,
            ("Main", "mtm"): [0.0] * len(index),
            ("Main", "total"): cash,
        },
        index=index,
        columns=cols,
    )


def _transactions(rows):
    return pd.DataFrame(
        rows, columns=["ts", "book", "asset_name", "quantity", "price", "total"]
    )


class ParityHarnessTestCase(ParityAssertionsMixin, unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2024-01-01", periods=4, freq="B")
        self.cash = [100.0, 90.0, 90.0, 110.0]
        d = self.index
        self.tx = [
            (d[1], "Main", "GOOG", Decimal(1), Decimal(10), Decimal(-10)),
            (d[1], "Main", "MSFT", Decimal(2), Decimal(5), Decimal(-10)),
            (d[3], "Main", "GOOG", Decimal(-1), Decimal(30), Decimal(30)),
        ]

    def _pair(self, cpp_index=None, cpp_cash=None, cpp_tx=None):
        py = _Result(_book_history(self.index, self.cash), _transactions(self.tx))
        cpp = _Result(
            _book_history(
                self.index if cpp_index is None else cpp_index,
                self.cash if cpp_cash is None else cpp_cash,
            ),
            _transactions(self.tx if cpp_tx is None else cpp_tx),
        )
        return py, cpp

    # -- should pass -------------------------------------------------------

    def test_identical_results_pass(self):
        self.assertResultParity(*self._pair())

    def test_index_resolution_is_not_a_discrepancy(self):
        # pandas>=3 defaults to ``us``; C++ history returns ``ns``.
        py, cpp = self._pair(cpp_index=self.index.as_unit("ns"))
        py.book_history.index = py.book_history.index.as_unit("us")
        self.assertResultParity(py, cpp)

    def test_transaction_order_within_a_day_is_not_a_discrepancy(self):
        reordered = [self.tx[1], self.tx[0], self.tx[2]]
        self.assertResultParity(*self._pair(cpp_tx=reordered))

    def test_decimal_vs_float_noise_is_not_a_discrepancy(self):
        cash = [c + 2e-12 for c in self.cash]
        self.assertResultParity(*self._pair(cpp_cash=cash))

    # -- should fail -------------------------------------------------------

    def test_shifted_index_fails(self):
        # The old harness reindexed to the Python index and filled with 0, so a
        # C++ history on entirely wrong dates compared as "all zeros".
        shifted = self.index + pd.Timedelta(days=1)
        with self.assertRaisesRegex(AssertionError, "index mismatch"):
            self.assertResultParity(*self._pair(cpp_index=shifted))

    def test_missing_row_fails(self):
        py, cpp = self._pair()
        cpp.book_history = cpp.book_history.iloc[:-1]
        with self.assertRaisesRegex(AssertionError, "index mismatch"):
            self.assertResultParity(py, cpp)

    def test_book_value_mismatch_fails(self):
        cash = list(self.cash)
        cash[2] += 1.0
        with self.assertRaisesRegex(AssertionError, r"column \('Main', 'cash'\)"):
            self.assertResultParity(*self._pair(cpp_cash=cash))

    def test_transaction_count_mismatch_fails(self):
        with self.assertRaisesRegex(AssertionError, "count mismatch"):
            self.assertResultParity(*self._pair(cpp_tx=self.tx[:-1]))

    def test_offsetting_wrong_trades_fail(self):
        # Same count, same per-asset net quantity, wrong trades: accepted by the
        # old count-and-sum comparison.
        d = self.index
        tx = [
            (d[1], "Main", "GOOG", Decimal(-1), Decimal(10), Decimal(10)),
            (d[1], "Main", "MSFT", Decimal(2), Decimal(5), Decimal(-10)),
            (d[3], "Main", "GOOG", Decimal(1), Decimal(30), Decimal(-30)),
        ]
        with self.assertRaisesRegex(AssertionError, "transaction .* mismatch"):
            self.assertResultParity(*self._pair(cpp_tx=tx))

    def test_trade_on_wrong_day_fails(self):
        d = self.index
        tx = list(self.tx)
        tx[2] = (d[2],) + tx[2][1:]
        with self.assertRaisesRegex(AssertionError, "transaction ts mismatch"):
            self.assertResultParity(*self._pair(cpp_tx=tx))

    def test_trade_on_wrong_book_fails(self):
        tx = list(self.tx)
        tx[0] = (tx[0][0], "Other") + tx[0][2:]
        with self.assertRaisesRegex(AssertionError, "transaction book mismatch"):
            self.assertResultParity(*self._pair(cpp_tx=tx))

    def test_trade_price_mismatch_fails(self):
        tx = list(self.tx)
        tx[1] = tx[1][:4] + (Decimal("5.01"), tx[1][5])
        with self.assertRaisesRegex(AssertionError, "transaction price mismatch"):
            self.assertResultParity(*self._pair(cpp_tx=tx))


if __name__ == "__main__":
    unittest.main()
