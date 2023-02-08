import unittest

import numpy as np
import numpy.linalg as la

import tests._unittest_numpy_extensions  # noqa
import yabte.utilities.pandas_extension  # noqa
from tests._helpers import generate_nasdaq_dataset
from yabte.utilities.lagrangian import Lagrangian


class UtilitiesTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.asset_meta, cls.df_combined = generate_nasdaq_dataset()
        cls.closes = cls.df_combined.loc[:, (slice(None), "Close")].droplevel(axis=1, level=1)
        cls.returns = cls.closes.price.log_returns

    def test_lagrangian(self):
        Sigma = self.returns.cov()
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

if __name__ == "__main__":
    unittest.main()