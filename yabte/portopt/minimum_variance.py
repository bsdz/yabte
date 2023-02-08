r"""Calculate portfolio weights by minimizing variance. 

That is to minimize the following expression subject to various constraints.

.. math::

   \frac{1}{2} w' \Sigma w

Typical constraints are achieving a target return, i.e. :math:`w' \mu = r`, and that
all weights sum to one, i.e. :math:`\Sigma_i w_i = 1`.
   
"""

import numpy as np
import numpy.linalg as la

from ..utilities.lagrangian import Lagrangian


def minimum_variance(Sigma: np.ndarray, mu: np.ndarray, r: float) -> np.ndarray:
    """Calculate weights using Lagrangian multipliers and
    algebraic closed form solution."""
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


def minimum_variance_numeric(Sigma: np.ndarray, mu: np.ndarray, r: float) -> np.ndarray:
    """Calculate weights using Lagrangian multipliers and
    numeric solution (using scipy's root function)."""
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


def minimum_variance_numeric_slsqp(
    Sigma: np.ndarray, mu: np.ndarray, r: float
) -> np.ndarray:
    """Calculate weights using Lagrangian multipliers and
    numeric solution (using scipy's minimize function)."""
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
