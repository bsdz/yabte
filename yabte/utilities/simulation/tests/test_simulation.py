import unittest

import numpy as np

from yabte.tests._unittest_numpy_extensions import NumpyTestCase
from yabte.utilities.simulation.geometric_brownian_motion import gbm_simulate_paths
from yabte.utilities.simulation.heston import heston_simulate_paths
from yabte.utilities.simulation.weiner import weiner_simulate_paths

HAS_SCIPY = True
try:
    import scipy.stats as stats
except:
    HAS_SCIPY = False


def _gbm_recover_weiner(T, N, M, r, p, sigma):
    # duplicate copies of time axis to simplify broadcasting later
    ts = np.linspace(0, T, N, endpoint=False)
    ts = np.repeat(ts[:, np.newaxis], M, axis=1)[:, :, np.newaxis]
    dt = T / N
    r_ = np.atleast_1d(r)
    sigma_ = np.atleast_1d(sigma)
    ws = (np.log(p) - (r_ - sigma_**2 / 2) * ts) / sigma_ / np.sqrt(dt)
    return ws


class SimulationTestCase(NumpyTestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12345)
        super().setUp()

    def test_weiner_1d(self):
        R = [[1]]
        N = 1000
        M = 100

        # simulate data
        # p[steps, sims, path]
        ws = weiner_simulate_paths(n_steps=N, n_sims=M, stdev=1, R=R, rng=self.rng)
        dws = np.diff(ws, axis=0)

        # test weiner properities
        self.assertTrue((ws[0] == 0).all())

    def test_weiner_simple(self):
        R = [[1, 0.9], [0.9, 1]]
        N = 101
        M = 3

        # simulate data
        # p[steps, sims, path]
        ws = weiner_simulate_paths(n_steps=N, n_sims=M, stdev=1, R=R, rng=self.rng)
        dws = np.diff(ws, axis=0)

        # test weiner properities
        self.assertTrue((ws[0] == 0).all())

        # check correlation (TODO: tolerance a bit poor here)
        for m in range(M):
            R_ = np.corrcoef(dws[:, m, 0], dws[:, m, 1])
            self.numpyAssertAllclose(R, R_, atol=0.03)

    @unittest.skipUnless(HAS_SCIPY, "needs scipy")
    def test_weiner_simple_ks(self):
        R = [[1, 0.9], [0.9, 1]]
        N = 101
        M = 3
        K = len(R)

        # simulate data
        # p[steps, sims, path]
        ws = weiner_simulate_paths(n_steps=N, n_sims=M, stdev=1, R=R, rng=self.rng)
        dws = np.diff(ws, axis=0)

        # normal differences
        for m, k in zip(range(M), range(K)):
            ksr = stats.kstest(dws[:, m, k], "norm", args=(0, 1))
            self.assertGreater(ksr.pvalue, 0.01)

    def test_gbm_simple(self):
        r = [0.05, 0.01]
        R = [[1, 0.9], [0.9, 1]]
        sigma = 0.2
        N = 101
        T = N / 365
        M = 3
        K = len(r)

        # simulate data
        # p[steps, sims, path]
        p = gbm_simulate_paths(
            S0=1, mu=r, sigma=sigma, R=R, T=T, n_steps=N, n_sims=M, rng=self.rng
        )

        # recover weiner process
        ws = _gbm_recover_weiner(T, N, M, r, p, sigma)
        dws = np.diff(ws, axis=0)

        # test weiner properities
        self.assertTrue((ws[0] == 0).all())

        # check correlation
        for m in range(M):
            R_ = np.corrcoef(dws[:, m, 0], dws[:, m, 1])
            self.numpyAssertAllclose(R, R_, atol=0.02)

    @unittest.skipUnless(HAS_SCIPY, "needs scipy")
    def test_gbm_simple_ks(self):
        r = [0.05, 0.01]
        R = [[1, 0.9], [0.9, 1]]
        sigma = 0.2
        N = 101
        T = N / 365
        M = 3
        K = len(r)

        # simulate data
        # p[steps, sims, path]
        p = gbm_simulate_paths(
            S0=1, mu=r, sigma=sigma, R=R, T=T, n_steps=N, n_sims=M, rng=self.rng
        )

        # recover weiner process
        ws = _gbm_recover_weiner(T, N, M, r, p, sigma)
        dws = np.diff(ws, axis=0)

        # normal differences
        for m, k in zip(range(M), range(K)):
            ksr = stats.kstest(dws[:, m, k], "norm", args=(0, 1))
            self.assertGreater(ksr.pvalue, 0.01)

    @unittest.skipUnless(HAS_SCIPY, "needs scipy")
    def test_gbm_1d_moments(self):
        r = 0.05
        R = 1
        sigma = 0.2
        N = 1000
        T = 1
        M = 1000
        S0 = 50

        # simulate data
        # p[steps, sims, path]

        P = [
            gbm_simulate_paths(
                S0=S0, mu=r, sigma=sigma, R=R, T=T, n_steps=N, n_sims=M, rng=self.rng
            )
            for x in range(1000)
        ]

        # analytical moments
        E_S_T = S0 * np.exp(r * T)
        Var_S_T = S0**2 * np.exp(2 * r * T) * (np.exp(sigma**2 * T) - 1)

        # empirical moments
        # S_T = P[-1, :, 0]
        # E_S_T_emp = np.mean(S_T)
        # Var_S_T_emp = np.var(S_T)

        S_T_s = np.c_[P][:, -1, :, 0]
        E_S_T_emp_s = np.mean(S_T_s, axis=1)

        # i.e. CLM => e_M * sqrt(M) / sigma = Z ~ N(0, 1); where e_M = E_S_T_emp - E_S_T
        Z = (E_S_T_emp_s - E_S_T) * np.sqrt(M) / np.sqrt(Var_S_T)

        print(stats.kstest(Z, "norm"))

        # check error is small to within 99% probability
        # i.e. CLM => e_M * sqrt(M) / sigma = Z ~ N(0, 1); where e_M = E_S_T_emp - E_S_T
        # since P(|Z| < s) = P(-s < Z < s) = Phi(s) - Phi(-s) = 1 - 2 * Phi(-s)
        # and P(|Z| < s) = P(|e_M * sqrt(M) / sigma| < s) = P(|e_M| < s * sig / sqrt(M))
        # choose analytical Variance for sig^2

        # confidence interval
        # P(|E_m| < eps) = P(|E_m|* np.sqrt(M) / sigma < eps * np.sqrt(M) / sigma))
        # = P(|Z| < eps * np.sqrt(M) / sigma) = 1 - alpha
        # since P(|Z| < s) = P(-s < Z < s) = Phi(s) - Phi(-s) = 1 - 2 * Phi(-s)
        # and choose analytical Variance for sigma^2
        # => P(|E_m| < eps) = 1 - 2 * Phi(-eps * np.sqrt(M) / sigma) = 1 - alpha
        # => 2 * Phi(-eps * np.sqrt(M) / sigma) = alpha
        # => Phi(-eps * np.sqrt(M) / sigma) = alpha / 2
        # => -eps * np.sqrt(M) / sigma = phi(alpha / 2)

        # https://math.arizona.edu/~tgk/mc/book_chap2.pdf

        # 1- alpha =
        # = P(|E_m| < B * sqrt(Var_S_T) / sqrt(M))
        # = P(|E_m| * sqrt(M) / sqrt(Var_S_T) < B)
        # = P(|Z| < B) = 1 - 2 * Phi(B)
        # => B = Phi^-1(alpha / 2)

        # e = E_S_T_emp - E_S_T
        # z = np.abs(e) * np.sqrt(M) / np.sqrt(Var_S_T)  # implied standard deviations
        # P = 1 - 2 * stats.norm.cdf(-s)
        # self.assertLessEqual(P, 1)
        # self.assertGreater(P, 0.85)  # TODO: how to get this to 0.95?

        # TODO: statistical test to check |Var_S_T_emp - Var_S_T|?

    def test_heston_smoke(self):
        kappa = 4
        theta = 0.02
        v0 = 0.02
        sigma = 0.9
        r = 0.02
        S0 = 100
        T = 1
        R = 0.9

        S, v = heston_simulate_paths(
            S0=S0,
            v0=v0,
            mu=r,
            kappa=kappa,
            theta=theta,
            xi=sigma,
            R=np.array([[1, R], [R, 1]]),
            T=T,
            n_steps=100,
            n_sims=1,
        )

        # TODO: better tests
        self.assertEqual(sum(S.shape), 101)
        self.assertEqual(sum(v.shape), 101)


if __name__ == "__main__":
    unittest.main()
