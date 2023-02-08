import numpy as np
import pandas as pd


@pd.api.extensions.register_dataframe_accessor("scl")
class ScaleAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @property
    def standard(self):
        return (self._obj - self._obj.mean()) / self._obj.std()


@pd.api.extensions.register_dataframe_accessor("prc")
class PriceAccessor:
    # TODO add ledoit cov (via sklearn)
    # http://www.ledoit.net/honey.pdf

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if (obj < 0).any(axis=None):
            raise AttributeError("Prices must be non-negative")

    @property
    def log_returns(self):
        return np.log((self._obj / self._obj.shift())[1:])

    @property
    def returns(self):
        return self._obj.pct_change()[1:]

    @property
    def frequency(self):
        days = pd.Timedelta(np.diff(self._obj.index).min()).days
        if days == 1:
            return 252
        elif days == 7:
            return 52

    def capm_returns(self, risk_free_rate=0):
        returns = self.returns
        returns_mkt = returns.mean(axis=1).rename("MKT")

        # concat market retutns & compute sampel covariance matrix
        cov = pd.concat([returns, returns_mkt], axis=1).cov()
        betas = cov.MKT.drop("MKT") / cov.MKT.MKT

        return (
            risk_free_rate
            + betas * (returns_mkt.mean() * self.frequency - risk_free_rate)
        ).rename("CAPM")

    def null_blips(self, sd=5, sdd=7):
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
