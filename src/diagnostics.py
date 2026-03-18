from __future__ import annotations
import numpy as np
import pandas as pd


def standardized_residuals(returns: pd.Series, sigma2: np.ndarray) -> np.ndarray:
    r = (returns - returns.mean()).values
    return r / np.sqrt(sigma2)


def ljung_box(residuals: np.ndarray, lags: int = 10) -> pd.DataFrame:
    """Ljung-Box test on squared residuals — checks for remaining ARCH effects."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    return acorr_ljungbox(residuals ** 2, lags=lags, return_df=True)


def kupiec_pof(n_exceptions: int, n_obs: int, alpha: float = 0.95) -> dict:
    """
    Kupiec proportion-of-failures test.
    H0: true exception rate = 1 - alpha.
    Returns test stat and p-value.
    """
    from scipy.stats import chi2
    p = 1 - alpha
    if n_exceptions == 0:
        return {"stat": np.nan, "p_value": np.nan, "reject_H0": False}
    p_hat = n_exceptions / n_obs
    lr = -2 * (
        n_exceptions * np.log(p / p_hat)
        + (n_obs - n_exceptions) * np.log((1 - p) / (1 - p_hat))
    )
    p_value = 1 - chi2.cdf(lr, df=1)
    return {"stat": round(lr, 4), "p_value": round(p_value, 4), "reject_H0": p_value < 0.05}
