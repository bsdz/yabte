r"""Heston Stochastic Volatility simulation.

Simulate stochastic process :math:`S_t` where,

.. math::

   dS_t = \mu S_t dt + \sqrt{\nu_t} S_t dW_t^S \\
   d\nu_t = \kappa(\theta - \nu_t) dt + \xi \sqrt{\nu_t} dW_t^{\nu}

where :math:`\mu` is the drift, :math:`\nu_t` is the variance (volatility squared)
and :math:`dW_t^S`, :math:`dW_t^{\nu}` are Weiner processes. :math:`\theta` is long
term variance and :math:`\kappa` is mean reversion rate. :math:`\xi` is volatility of
volatility.
"""

import numpy as np

from yabte.utilities.simulation.weiner import weiner_simulate_paths


def heston_simulate_paths(
    S0: float,
    v0: float,
    mu: float,
    kappa: float,
    theta: float,
    xi: float,
    R: np.ndarray,
    T: int,
    n_steps: int,
    n_sims: int,
    rng=None,
):
    """Generate simulated paths.

    `S0` and `v0` are initial values, `mu` is the drift, `kappa` is the mean reversion
    rate, `theta` is long term variance, `xi` is volatility of volatility, `R` a 2x2
    correlation matrix between the price and variance Weiner processes, `T` is the time
    span, `n_steps` is how many time steps (including t=0, so `dt = T / (n_steps -
    1)`), `n_sims` the number of simulations and `rng` a numpy random number generator
    (optional).

    Uses a full truncation Euler scheme for the variance and a log-Euler scheme for the
    price, so `S` is a martingale after discounting by `mu`. Returns `(S, v)`, each of
    shape `(n_steps, n_sims)`, where `v` is the (non-negative) variance.

    TODO: support Milstein / QE / Antithetic.
    """
    if rng is None:
        rng = np.random.default_rng()

    r = mu  # mu = rf in risk neutral framework
    dt = T / (n_steps - 1)

    # ws[t, sim, path]; path[0] = price, path[1] = variance
    S_ix, v_ix = 0, 1
    ws = weiner_simulate_paths(
        n_steps=n_steps, n_sims=n_sims, stdev=np.sqrt(dt), R=R, rng=rng
    )
    dws = np.diff(ws, axis=0)

    # v[t, sim], log_S[t, sim]
    v = np.empty(shape=(n_steps, n_sims))
    log_S = np.empty(shape=(n_steps, n_sims))
    v[0, :] = v0
    log_S[0, :] = np.log(S0)
    for i in range(1, n_steps):
        vp = np.clip(v[i - 1, :], 0, None)
        sqrt_vp = np.sqrt(vp)
        v[i, :] = (
            v[i - 1, :] + kappa * (theta - vp) * dt + xi * sqrt_vp * dws[i - 1, :, v_ix]
        )
        log_S[i, :] = (
            log_S[i - 1, :] + (r - vp / 2) * dt + sqrt_vp * dws[i - 1, :, S_ix]
        )

    return np.exp(log_S), np.clip(v, 0, None)
