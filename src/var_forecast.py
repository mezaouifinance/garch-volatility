from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
from .garch import fit, conditional_variance


def static_var(returns: pd.Series, alpha: float = 0.95) -> float:
    """Parametric VaR assuming constant normal distribution."""
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    return float(-(mu + norm.ppf(1 - alpha) * sigma))


def dynamic_var(returns: pd.Series, alpha: float = 0.95) -> pd.Series:
    """VaR using GARCH(1,1) conditional volatility."""
    omega, a, b = fit(returns)
    sigma2 = conditional_variance(returns, omega, a, b)
    mu = returns.mean()
    var = -(mu + norm.ppf(1 - alpha) * np.sqrt(sigma2))
    return pd.Series(var, index=returns.index)


def exception_rate(returns: pd.Series, var_series) -> float:
    if np.isscalar(var_series):
        return float((-returns > var_series).mean())
    return float((-returns > var_series).mean())


def compare(returns: pd.Series, alpha: float = 0.95) -> pd.DataFrame:
    sv = static_var(returns, alpha)
    dv = dynamic_var(returns, alpha)
    return pd.DataFrame({
        "return": returns.values,
        "static_var": sv,
        "dynamic_var": dv.values,
        "exc_static": (-returns > sv).values,
        "exc_dynamic": (-returns > dv).values,
    }, index=returns.index)
