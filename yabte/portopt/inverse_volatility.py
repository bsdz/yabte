import numpy as np

def inverse_volatility(cov):
    sigma_inv = np.sqrt(np.diag(cov))**-1
    return sigma_inv / (np.ones_like(sigma_inv) @ sigma_inv)