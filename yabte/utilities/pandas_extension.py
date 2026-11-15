"""Pandas accessors for prices and scaling.

Importing this module registers ``.prc`` on DataFrames and Series, and ``.scl`` on
DataFrames::

    import yabte.utilities.pandas_extension  # noqa: F401

    closes.prc.log_returns
"""

import numpy as np
import pandas as pd


@pd.api.extensions.register_dataframe_accessor("scl")
class ScaleAccessor:
    """``df.scl``: rescaling a DataFrame column by column."""

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @property
    def standard(self):
        """Z-scores: each column minus its mean, divided by its standard deviation."""
        return (self._obj - self._obj.mean()) / self._obj.std()


@pd.api.extensions.register_dataframe_accessor("prc")
@pd.api.extensions.register_series_accessor("prc")
class PriceAccessor:
    """``obj.prc``: returns and statistics from prices, one column per asset."""

    # TODO: add ledoit cov (via sklearn)
    # http://www.ledoit.net/honey.pdf

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        pass

    @property
    def log_returns(self):
        """Log returns between rows, without the first row."""
        if (self._obj < 0).any(axis=None):
            raise AttributeError("Prices must be non-negative for log returns")
        return np.log((self._obj / self._obj.shift())[1:])

    @property
    def returns(self):
        """Simple returns between rows, without the first row."""
        return self._obj.pct_change(fill_method=None)[1:]

    @property
    def frequency(self):
        """Periods a year from the smallest gap in the index: 252 for daily, 52 for
        weekly and 12 for monthly data, otherwise `None`."""
        days = pd.Timedelta(np.diff(self._obj.index).min()).days
        if days == 1:
            return 252
        elif days == 7:
            return 52
        elif 28 <= days <= 31:
            return 12

    def capm_returns(self, risk_free_rate=0):
        """Annualised expected returns from CAPM.

        The market is the equally weighted average of the columns' returns, and each
        column's beta is estimated against it. `risk_free_rate` is annual.
        """
        returns = self.returns
        returns_mkt = returns.mean(axis=1).rename("MKT")

        # concat market retutns & compute sampel covariance matrix
        cov = pd.concat([returns, returns_mkt], axis=1).cov()
        betas = cov.MKT.drop("MKT") / cov.MKT.MKT

        return (
            risk_free_rate
            + betas * (returns_mkt.mean() * self.frequency - risk_free_rate)
        ).rename("CAPM")

    def sharpe_ratio(self, risk_free_rate=0, use_log_returns=True):
        """Annualised Sharpe ratio per column.

        `risk_free_rate` is per period, not annual.
        """
        ann_factor = np.sqrt(self.frequency)
        returns = self.log_returns if use_log_returns else self.returns
        return ann_factor * (returns.mean() - risk_free_rate) / returns.std()

    def null_blips(self, sd=5, sdd=7):
        """Set one-row price spikes to NaN, in place.

        A value is a blip when its z-score exceeds `sd` and the z-score jumps by more
        than `sdd` into it and back out of it the next row. Prints each value it removes
        and returns the modified object.
        """
        df = self._obj
        z = df.scl.standard
        zd = z.diff()
        # TODO support blips longer than 1 day?
        for col, series in df[z.abs() > sd].dropna(how="all").dropna(axis=1).items():
            for row, val in series.items():
                row_ix = df.index.get_loc(row)
                col_ix = df.columns.get_loc(col)
                zd0 = zd.iloc[row_ix, col_ix]
                zd1 = zd.iloc[row_ix + 1, col_ix]
                if (
                    np.sign(zd0) != np.sign(zd1)
                    and np.abs(zd0) > sdd
                    and np.abs(zd1) > sdd
                ):
                    print(f"nullifying blip at {col} {row}")
                    df.iloc[row_ix, col_ix] = np.nan

        return self._obj
