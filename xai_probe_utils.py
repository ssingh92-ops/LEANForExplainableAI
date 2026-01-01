"""
xai_probe_utils.py

Utilities to fit and evaluate linear probes on representations.

Lean mapping (informal):
- rep : X → Z                       (model.encode / model.represent)
- probe w : Z → ℝ (linear)          (fit_linear_probe returns weights + bias)
- ProbeSpecRep: ∀x, |c(x) - w(rep(x))| ≤ δ
  We compute an *empirical* δ on a dataset as:
      δ_hat = max_i |c_i - w(z_i)|

This is the "Python witness" side that you can use to justify / instantiate
your Lean specs (with appropriate assumptions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class LinearProbe:
    """
    Simple linear probe: y = z @ w + b
    (not necessarily continuous linear map in the Lean sense unless b=0;
     in practice probes often include bias, so we track it explicitly).
    """
    w: np.ndarray          # shape (k,)
    b: float = 0.0
    name: str = "probe"

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return z @ self.w + self.b


def fit_linear_probe_ridge(z: np.ndarray, c: np.ndarray, *, ridge: float = 1e-3, fit_intercept: bool = True) -> LinearProbe:
    """
    Ridge regression closed form.

    z: (n,k) representations
    c: (n,) concept values

    If fit_intercept:
      augment z with a column of ones and solve ridge on the augmented matrix.
    """
    z = np.asarray(z, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64).reshape(-1)

    if fit_intercept:
        Z = np.concatenate([z, np.ones((z.shape[0], 1))], axis=1)
    else:
        Z = z

    k = Z.shape[1]
    A = Z.T @ Z + ridge * np.eye(k)
    b = Z.T @ c
    theta = np.linalg.solve(A, b)

    if fit_intercept:
        w = theta[:-1]
        b0 = float(theta[-1])
    else:
        w = theta
        b0 = 0.0

    return LinearProbe(w=w, b=b0, name="ridge_probe")


def probe_metrics(z: np.ndarray, c: np.ndarray, probe: LinearProbe) -> Dict[str, float]:
    """
    Compute common probe metrics.

    Returns:
      - mae
      - rmse
      - r2
      - delta_max (empirical sup error) = max_i |c_i - probe(z_i)|
    """
    z = np.asarray(z, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64).reshape(-1)
    pred = probe(z).reshape(-1)

    err = c - pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    delta_max = float(np.max(np.abs(err)))

    # R^2
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((c - np.mean(c))**2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"mae": mae, "rmse": rmse, "r2": r2, "delta_max": delta_max}
