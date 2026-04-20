from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _loglik(params: np.ndarray, r: np.ndarray) -> float:
    omega, alpha, beta = params
    # Penalize stationarity violations inside the objective
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1.0:
        return 1e10
    n = len(r)
    sigma2 = np.empty(n)
    sigma2[0] = omega / (1.0 - alpha - beta)  # unconditional variance

    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]

    if np.any(sigma2 <= 0):
        return 1e10

    return 0.5 * np.sum(np.log(sigma2) + r ** 2 / sigma2)


def fit(returns: pd.Series) -> tuple[float, float, float]:
    """
    Fit GARCH(1,1) by MLE.
    Returns (omega, alpha, beta) with alpha + beta < 1 (stationarity).
    """
    r = (returns - returns.mean()).values
    var0 = float(np.var(r))

    best = None
    # Try several starting points for robustness
    for a0, b0 in [(0.05, 0.90), (0.10, 0.80), (0.15, 0.75)]:
        x0 = np.array([var0 * (1 - a0 - b0), a0, b0])
        res = minimize(
            _loglik, x0, args=(r,),
            method="L-BFGS-B",
            bounds=[(1e-9, None), (1e-9, 0.999), (1e-9, 0.999)],
            options={"ftol": 1e-14, "maxiter": 1000},
        )
        if res.success or best is None or res.fun < best.fun:
            best = res

    omega, alpha, beta = best.x
    if alpha + beta >= 1.0:
        raise RuntimeError("GARCH fit: stationarity constraint violated.")

    return float(omega), float(alpha), float(beta)


def conditional_variance(returns: pd.Series, omega: float, alpha: float, beta: float) -> np.ndarray:
    """Compute the conditional variance series sigma^2_t."""
    r = (returns - returns.mean()).values
    n = len(r)
    sigma2 = np.empty(n)
    sigma2[0] = np.var(r)

    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]

    return sigma2


def persistence(alpha: float, beta: float) -> float:
    return alpha + beta


def half_life(alpha: float, beta: float) -> float:
    """Expected number of days for a vol shock to decay by half."""
    p = persistence(alpha, beta)
    if p >= 1.0:
        return float("inf")
    return np.log(0.5) / np.log(p)


def forecast_variance(omega: float, alpha: float, beta: float, sigma2_last: float, r_last: float, h: int = 1) -> float:
    """
    Multi-step variance forecast.
    For h=1: sigma^2_{t+1} = omega + alpha * r_t^2 + beta * sigma^2_t
    For h>1: iterate using the unconditional mean for expected r^2.
    """
    p = persistence(alpha, beta)
    var_unc = omega / (1.0 - p)
    sigma2 = omega + alpha * r_last ** 2 + beta * sigma2_last
    for _ in range(h - 1):
        sigma2 = omega + p * sigma2 + (1 - p) * var_unc
    return sigma2
