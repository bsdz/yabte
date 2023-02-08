r"""Calculate portfolio weights with hierarchical risk parity. 

That is to employ hierarchical tree clustering on the correlation distance
matrix and quasi-diagonalisation followed by recursive bisection to determine
the weights. See [LP] for further details.


References
----------
.. [LP] López de Prado, M. (2016). Building Diversified Portfolios that
    Outperform Out of Sample. The Journal of Portfolio Management, 42(4),
    59–69. https://doi.org/10.3905/jpm.2016.42.4.059
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree

# following 3 functions taken directly from paper [LP]


def _getIVP(cov, **kargs):
    # Compute the inverse-variance portfolio
    ivp = 1.0 / np.diag(cov)
    ivp /= ivp.sum()
    return ivp


def _getClusterVar(cov, cItems):
    # Compute variance per cluster
    cov_ = cov.loc[cItems, cItems]  # matrix slice
    w_ = _getIVP(cov_).reshape(-1, 1)
    cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
    return cVar


def _getRecBipart(cov, sortIx):
    # Compute HRP alloc
    w = pd.Series(1, index=sortIx)
    cItems = [sortIx]  # initialize all items in one cluster
    while len(cItems) > 0:
        cItems = [
            i[j:k]
            for i in cItems
            for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
            if len(i) > 1
        ]  # bi-section
        for i in range(0, len(cItems), 2):  # parse in pairs
            cItems0 = cItems[i]  # cluster 1
            cItems1 = cItems[i + 1]  # cluster 2
            cVar0 = _getClusterVar(cov, cItems0)
            cVar1 = _getClusterVar(cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0] *= alpha  # weight 1
            w[cItems1] *= 1 - alpha  # weight 2
    return w


def hrp(corr: pd.DataFrame, sigma: np.ndarray) -> np.ndarray:
    """Calculate weights using hierarchical risk parity and
    scipy's linkage/to_tree functions."""
    cov = np.diag(sigma) @ corr @ np.diag(sigma)
    cov.index, cov.columns = corr.index, corr.columns
    rho = corr.values
    D = np.sqrt((1 - rho) / 2)
    I, J = np.triu_indices_from(D, 1)
    link = linkage(np.sqrt(np.sum((D[I] - D[J]) ** 2, axis=1)))
    ix_sorted = to_tree(link, rd=False).pre_order()
    cols_sorted = corr.columns[ix_sorted]
    return _getRecBipart(cov, cols_sorted)
