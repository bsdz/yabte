import numpy as np
import numpy.linalg as la

from .lagrangian import Lagrangian


def minimum_variance(Sigma, mu, r):
    m = len(mu)
    ones = np.ones(m)

    SigmaInv = la.inv(Sigma)
    A = mu.T @ SigmaInv @ ones
    B = mu.T @ SigmaInv @ mu
    C = ones.T @ SigmaInv @ ones
    D = B * C - A * A
    l1 = (C * r - A) / D
    l2 = (B - A * r) / D
    return SigmaInv @ (l1 * mu + l2 * ones)


def minimum_variance_numeric(Sigma, mu, r):
    m = len(mu)
    ones = np.ones(m)

    L = Lagrangian(
        objective=lambda x: x.T @ Sigma @ x / 2,
        constraints=[
            lambda x: r - x.T @ mu,
            lambda x: 1 - x.T @ ones,
        ],
        x0=ones / m,
    )
    return L.fit()


def minimum_variance_numeric_slsqp(Sigma, mu, r):
    from scipy.optimize import minimize

    m = len(mu)
    ones = np.ones(m)

    res = minimize(
        lambda x: x.T @ Sigma @ x / 2,
        ones / m,
        method="SLSQP",
        constraints=(
            {"type": "eq", "fun": lambda x: r - x.T @ mu, "jac": lambda x: -mu.values},
            {"type": "eq", "fun": lambda x: 1 - x.T @ ones, "jac": lambda x: -ones},
        ),
        tol=1e-15,
    )

    return res.x
