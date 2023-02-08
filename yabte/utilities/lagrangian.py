from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
from scipy.optimize import OptimizeResult, root
from scipy.optimize._numdiff import approx_derivative


@dataclass(kw_only=True)
class Lagrangian:
    x0: np.ndarray
    objective: Callable[[np.ndarray], float]
    constraints: List[Callable[[np.ndarray], float]] = field(default_factory=list)
    optimize_result: Optional[OptimizeResult] = None

    def f(self, x):
        return self.objective(x)

    def g(self, x):
        return np.array([f(x) for f in self.constraints])

    def f_grad(self, x):
        return approx_derivative(self.f, x)

    def g_jac(self, x):
        return approx_derivative(self.g, x)

    def H(self, z):
        x, l = np.split(z, (self.x0.shape[0],))
        eq1 = self.f_grad(x)
        eq2 = []
        if self.constraints:
            eq1 += self.g_jac(x).T @ l
            eq2 = self.g(x)
        return np.array([*eq1, *eq2])

    def fit(self):
        n_constraints = len(self.constraints)
        res = root(self.H, x0=np.r_[self.x0, [1] * n_constraints])
        self.optimize_result = res
        return res.x[:-n_constraints]
