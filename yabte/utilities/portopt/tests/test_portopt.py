import unittest

import numpy as np
import numpy.linalg as la

import yabte.utilities.pandas_extension  # noqa
from yabte.tests._helpers import generate_nasdaq_dataset
from yabte.tests._unittest_numpy_extensions import NumpyTestCase
from yabte.utilities.lagrangian import Lagrangian
from yabte.utilities.portopt.hierarchical_risk_parity import hrp
from yabte.utilities.portopt.inverse_volatility import inverse_volatility
from yabte.utilities.portopt.minimum_variance import (
    minimum_variance,
    minimum_variance_numeric,
    minimum_variance_numeric_slsqp,
)


class PortOptTestCase(NumpyTestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets, cls.df_combined = generate_nasdaq_dataset()
        cls.closes = cls.df_combined.loc[:, (slice(None), "Close")].droplevel(
            axis=1, level=1
        )
        cls.returns = cls.closes.prc.log_returns

    def test_min_var(self):
        Sigma = self.returns.cov()
        mu = self.closes.prc.capm_returns()
        r = 0.1

        w = minimum_variance(Sigma, mu, r)
        wn = minimum_variance_numeric(Sigma, mu, r)
        wn2 = minimum_variance_numeric_slsqp(Sigma, mu, r)

        self.numpyAssertAllclose(w, wn)
        # slsqp not as numerically accurate as other two methods
        self.numpyAssertAllclose(w, wn2, rtol=1e-06)

    def test_hrp(self):
        R = self.returns.corr()
        sigma = self.returns.std()

        w = hrp(R, sigma)

        self.numpyAssertAllclose(w.sum(), 1)

    def test_ivp(self):
        Sigma = self.returns.cov()

        w = inverse_volatility(Sigma)

        self.numpyAssertAllclose(w.sum(), 1)


if __name__ == "__main__":
    unittest.main()
