import unittest

import numpy as np
import numpy.linalg as la

import tests._unittest_numpy_extensions  # noqa
import yabte.portopt.pandas_extension  # noqa
from tests._helpers import generate_nasdaq_dataset
from yabte.portopt.lagrangian import Lagrangian
from yabte.portopt.minimum_variance import (
    minimum_variance,
    minimum_variance_numeric,
    minimum_variance_numeric_slsqp,
)


class LagrangianTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.asset_meta, cls.df_combined = generate_nasdaq_dataset()
        cls.closes = cls.df_combined.loc[:, (slice(None), "Close")].droplevel(axis=1, level=1)
 

    def test_lagrangian(self):
        Sigma = self.closes.price.log_returns.cov()
        mu = self.closes.price.capm_returns()
        r = 0.1

        # solve algebraically
        m = len(mu)
        ones = np.ones(m)
        SigmaInv = la.inv(Sigma)
        A = mu.T @ SigmaInv @ ones
        B = mu.T @ SigmaInv @ mu
        C = ones.T @ SigmaInv @ ones
        D = B*C - A*A
        l1 = (C*r - A)/D
        l2 = (B - A*r)/D
        w = SigmaInv@(l1 * mu + l2 * ones)

        # sanity checks
        self.numpyAssertAllclose(w.sum(), 1)
        self.numpyAssertAllclose(w @ mu, r)

        # test numerical
        L = Lagrangian(
            objective=lambda x: x.T @ Sigma @ x / 2,
            constraints=[
                lambda x: r - x.T @ mu,
                lambda x: 1 - x.T @ ones,
            ],
            x0=np.ones(m)/m
        )
        wn = L.fit()

        self.numpyAssertAllclose(wn, w)

    def test_min_var(self):
        Sigma = self.closes.price.log_returns.cov()
        mu = self.closes.price.capm_returns()
        r = 0.1

        mv1 = minimum_variance(Sigma, mu, r)
        mv2 = minimum_variance_numeric(Sigma, mu, r)
        mv3 = minimum_variance_numeric_slsqp(Sigma, mu, r)

        x = 1


if __name__ == "__main__":
    unittest.main()