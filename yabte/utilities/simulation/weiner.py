r"""Weiner process.

Weiner process :math:`W_t` such that,

#. :math:`W_0 = 0`
#. :math:`\forall t>0, u \ge  0, s<t \Rightarrow W_{t+u}-W_{t} \text{ independent of } W_s`
#. :math:`W_{t+u}-W_{t}\sim {\mathcal {N}}(0,u)`
#. :math:`W_t` is continuous
"""

import numpy as np


def weiner_simulate_paths(
    n_steps: int,
    n_sims: int = 1,
    stdev: float = 1,
    R: np.ndarray = np.array([[1]]),
    rng=None,
):
    """Generate simulated Weiner paths.

    `stdev` is the increment size, `R` a correlation matrix, `n_steps`
    is how many time steps, `n_sims` the number of simulations and `rng`
    a numpy random number generator (optional).
    """

    R = np.atleast_2d(R)

    if rng is None:
        rng = np.random.default_rng()

    k = R.shape[0]

    # simulate 'n_sims' price paths of `k` sized asset groups with 'n_steps' timesteps
    dws = rng.multivariate_normal(
        mean=np.zeros(k), cov=R, size=(n_steps - 1, n_sims), check_valid="raise"
    )

    # use cumsum as speed up
    ws = np.concatenate([np.zeros((1, n_sims, k)), np.cumsum(dws, axis=0)])

    return stdev * ws
