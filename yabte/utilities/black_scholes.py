"""Black-Scholes prices and greeks for European options, vectorised over numpy
arrays."""

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr

__all__ = ["BlackScholes", "black_scholes"]


class BlackScholes(NamedTuple):
    """Price and greeks, each an array broadcast from the inputs."""

    price: np.ndarray
    delta: np.ndarray
    """Change in price per unit change in the underlying."""

    gamma: np.ndarray
    """Change in delta per unit change in the underlying."""

    vega: np.ndarray
    """Change in price per unit change in volatility (1.0, not 1%)."""


def black_scholes(
    S: ArrayLike,
    K: ArrayLike,
    tau: ArrayLike,
    sigma: ArrayLike,
    r: ArrayLike = 0.0,
    call: ArrayLike = True,
) -> BlackScholes:
    """Black-Scholes price, delta, gamma and vega of European options.

    `S` is the underlying price, `K` the strike, `tau` the time to expiry in years,
    `sigma` the annual volatility and `r` the continuously compounded annual rate.
    `call` selects calls (`True`) or puts (`False`). The arguments broadcast against
    each other, so one call can price a whole path or a grid of strikes.

    At expiry (`tau` of 0), or with zero volatility, the price is the discounted
    intrinsic value, delta is 0 or ±1, and gamma and vega are 0.
    """
    S, K, tau, sigma, r, call = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (S, K, tau, sigma, r)),
        np.asarray(call, dtype=bool),
    )
    sign = np.where(call, 1.0, -1.0)
    discount = np.exp(-r * tau)
    sd = sigma * np.sqrt(tau)
    live = sd > 0

    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * tau) / sd
        d2 = d1 - sd
        pdf = np.exp(-(d1**2) / 2) / np.sqrt(2 * np.pi)
        price = sign * (S * ndtr(sign * d1) - K * discount * ndtr(sign * d2))
        delta = sign * ndtr(sign * d1)
        gamma = pdf / (S * sd)
        vega = S * pdf * np.sqrt(tau)

    # no time value left: the forward's intrinsic value
    forward_itm = sign * (S - K * discount) > 0
    return BlackScholes(
        price=np.where(live, price, np.maximum(sign * (S - K * discount), 0.0)),
        delta=np.where(live, delta, np.where(forward_itm, sign, 0.0)),
        gamma=np.where(live, gamma, 0.0),
        vega=np.where(live, vega, 0.0),
    )
