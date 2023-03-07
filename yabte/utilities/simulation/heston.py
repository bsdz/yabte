r"""Heston Stochastic Volatility simulation.

Simulate stochastic process :math:`S_t` where,

.. math::

   dS_t = \mu S_t dt + \sqrt{\nu_t} S_t dW_t^S \\
   d\nu_t = \kappa(\theta - \nu_t) dt + \xi \sqrt{\nu_t} dW_t^{\nu}

where :math:`\mu` is the drift, :math:`\nu_t` is the variance (volatility squared)
and :math:`dW_t^S`, :math:`dW_t^{\nu_t}` are Weiner processes. :math:`\theta` is long
term variance and :math:`\kappa` is mean reversion rate. :math:`xi` is volatility of
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

    `S0` and `v0` are initial values, `mu` is the drift, `xi` is volatility of volatility, `theta` is long term variance, `R` a correlation
    matrix, `T` is the time span, `n_steps` is how many time steps,
    `n_sims` the number of simulations and `rng` a numpy random
    number generator (optional).

    TODO: support Euler–Maruyama / Milstein / Antithetic.
    """
    if rng is None:
        rng = np.random.default_rng()

    r = mu  # mu = rf in risk neutral framework
    dt = T / n_steps

    ts = np.linspace(0, T, n_steps, endpoint=False)

    # simulate 'n_sims' price paths of `k` sized asset groups with 'n_steps' timesteps
    # ws[t, sim, path]; path[0] = vol, path[1] = price
    S_ix, v_ix = 0, 1
    ws = weiner_simulate_paths(
        n_steps=n_steps, n_sims=n_sims, stdev=np.sqrt(dt), R=R, rng=rng
    )

    # calc vol
    # v[t, sim]
    v = np.zeros(shape=(n_steps, n_sims))
    v[0, :] = v0
    for i in range(1, n_steps):
        vp = v[i - 1, :]
        vp_max = np.clip(vp, 0, None)
        v[i, :] = (
            vp
            + kappa * (theta - vp_max) * ts[i]
            + xi * np.sqrt(vp_max) * ws[i, :, v_ix]
        )

    # S[t, sim]
    S = S0 * np.exp((r - v**2) * ts.reshape(n_steps, -1) + v * ws[:, :, S_ix])

    return S, v
