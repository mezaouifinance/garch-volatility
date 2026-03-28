import numpy as np
import pandas as pd
import pytest
from src.garch import fit, conditional_variance, persistence, half_life
from src.var_forecast import static_var, dynamic_var, exception_rate
from src.diagnostics import standardized_residuals, kupiec_pof

rng = np.random.default_rng(0)
RETURNS = pd.Series(rng.normal(0, 0.01, 600))


def test_fit_stationarity():
    omega, alpha, beta = fit(RETURNS)
    assert omega > 0
    assert alpha > 0
    assert beta > 0
    assert persistence(alpha, beta) < 1.0


def test_conditional_variance_positive():
    omega, alpha, beta = fit(RETURNS)
    sigma2 = conditional_variance(RETURNS, omega, alpha, beta)
    assert len(sigma2) == len(RETURNS)
    assert (sigma2 > 0).all()


def test_half_life_finite():
    omega, alpha, beta = fit(RETURNS)
    hl = half_life(alpha, beta)
    assert np.isfinite(hl)
    assert hl > 0


def test_static_var_positive():
    assert static_var(RETURNS, alpha=0.95) > 0
    assert static_var(RETURNS, alpha=0.99) > static_var(RETURNS, alpha=0.95)


def test_dynamic_var_shape_and_sign():
    dv = dynamic_var(RETURNS, alpha=0.95)
    assert len(dv) == len(RETURNS)
    assert (dv > 0).all()


def test_exception_rate_in_range():
    dv = dynamic_var(RETURNS, alpha=0.95)
    rate = exception_rate(RETURNS, dv)
    assert 0.0 <= rate <= 1.0


def test_kupiec_pof_structure():
    result = kupiec_pof(n_exceptions=25, n_obs=500, alpha=0.95)
    assert "stat" in result
    assert "p_value" in result
    assert "reject_H0" in result


def test_standardized_residuals_unit_variance():
    omega, alpha, beta = fit(RETURNS)
    sigma2 = conditional_variance(RETURNS, omega, alpha, beta)
    z = standardized_residuals(RETURNS, sigma2)
    assert abs(np.std(z) - 1.0) < 0.2
