"""
Quick GARCH(1,1) analysis on SPY daily returns.
Run: python analysis.py
"""
import numpy as np
from src.data import load_returns
from src.garch import fit, conditional_variance, persistence, half_life
from src.var_forecast import static_var, dynamic_var, exception_rate, compare
from src.diagnostics import standardized_residuals, kupiec_pof


def main():
    print("Loading SPY returns (2018-2024)...")
    returns = load_returns("SPY", start="2018-01-01", end="2024-01-01")
    print(f"  {len(returns)} observations\n")

    print("Fitting GARCH(1,1)...")
    omega, alpha, beta = fit(returns)
    print(f"  omega = {omega:.6f}")
    print(f"  alpha = {alpha:.4f}")
    print(f"  beta  = {beta:.4f}")
    print(f"  persistence (alpha+beta) = {persistence(alpha, beta):.4f}")
    print(f"  vol shock half-life      = {half_life(alpha, beta):.1f} days\n")

    sigma2 = conditional_variance(returns, omega, alpha, beta)

    sv = static_var(returns, alpha=0.95)
    dv = dynamic_var(returns, alpha=0.95)

    exc_static = exception_rate(returns, sv)
    exc_dynamic = exception_rate(returns, dv)

    print("VaR comparison (95% confidence):")
    print(f"  Static VaR  : {sv:.4f}  |  exception rate: {exc_static:.2%}")
    print(f"  Dynamic VaR : mean={dv.mean():.4f}  |  exception rate: {exc_dynamic:.2%}")
    print(f"  (expected: 5.00%)\n")

    kup_static  = kupiec_pof(int((returns < -sv).sum()), len(returns), alpha=0.95)
    kup_dynamic = kupiec_pof(int((-returns > dv).sum()), len(returns), alpha=0.95)
    print("Kupiec POF test:")
    print(f"  Static  -> stat={kup_static['stat']}, p={kup_static['p_value']}, reject H0: {kup_static['reject_H0']}")
    print(f"  Dynamic -> stat={kup_dynamic['stat']}, p={kup_dynamic['p_value']}, reject H0: {kup_dynamic['reject_H0']}")


if __name__ == "__main__":
    main()
