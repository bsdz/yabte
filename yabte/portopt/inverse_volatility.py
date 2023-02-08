r"""Calculate portfolio weights by inverting variance. 

That is,

.. math::

   w = \frac{\sigma^{-1}}{1'\sigma^{-1}}

where :math:`\sigma^2 = diag(\Sigma)`.
   
"""

import numpy as np


def inverse_volatility(cov: np.ndarray) -> np.ndarray:
    """Calculate weights using inverse variance."""
    sigma_inv = np.sqrt(np.diag(cov)) ** -1
    return sigma_inv / (np.ones_like(sigma_inv) @ sigma_inv)
