r"""Geometric Brownian motion simulation.

Simulate stochastic process :math:`S_t` where,

.. math::

   dS_t = \mu S_t dt + \sigma S_t dW_t

where :math:`\mu` is the drift, :math:`\sigma` is the volatility
and :math:`dW_t` is a Weiner process.
"""

import numpy as np


def gbm_simulate_paths(
    mu: np.ndarray,
    sigma: np.ndarray,
    R: np.ndarray,
    T: int,
    n_steps: int,
    n_sims: int,
    rng=None,
):
    """Generate simulated paths using vectorised numpy calls.

    `mu` is the drift, `sigma` is volatility, `R` a correlation
    matrix, `T` is the time span, `n_steps` is how many time steps,
    `n_sims` the number of simulations and `rng` a numpy random
    number generator (optional).

    TODO: support Euler–Maruyama / Milstein / Antithetic.
    """

    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    R = np.atleast_2d(R)

    if rng is None:
        rng = np.random.default_rng()

    r = mu  # mu = rf in risk neutral framework
    dt = T / n_steps
    k = len(mu)

    # simulate 'n_sims' price paths of `k` sized asset groups with 'n_steps' timesteps
    dws = rng.multivariate_normal(
        mean=np.zeros_like(mu), cov=R, size=(n_steps, n_sims), check_valid="raise"
    )

    # use closed form solution

    # duplicate copies of time axis to simplify broadcasting later
    ts = np.linspace(0, T, n_steps, endpoint=False)
    ts = np.repeat(ts[:, np.newaxis], n_sims, axis=1)[:, :, np.newaxis]

    # use cumsum as speed up
    ws = np.cumsum(dws, axis=0)

    # start all at one by prefixing newaxis
    return np.concatenate(
        [
            np.ones((1, n_sims, k)),
            np.exp((r - sigma**2 / 2) * ts + sigma * ws * np.sqrt(dt))[:-1, :],
        ]
    )
