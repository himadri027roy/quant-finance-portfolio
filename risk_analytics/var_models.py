"""
Value-at-Risk (VaR) & Expected Shortfall (ES)
===============================================
Three approaches to VaR estimation:
  1. Historical Simulation
  2. Parametric (Variance-Covariance)
  3. Monte Carlo Simulation

Also includes:
  - Expected Shortfall (CVaR)
  - Backtesting VaR with Kupiec's POF test
  - Stress testing

Reference:
  Basel Committee on Banking Supervision (2019). Minimum Capital Requirements
  for Market Risk.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# ─── Historical Simulation ────────────────────────────────────────────────────

def var_historical(returns: pd.Series, confidence: float = 0.99,
                   horizon: int = 1) -> float:
    """
    Historical Simulation VaR.

    Non-parametric: uses empirical return distribution directly.
    No distributional assumption required.

    Parameters
    ----------
    returns    : Daily P&L returns (as fractions)
    confidence : Confidence level (e.g., 0.99 = 99%)
    horizon    : Holding period in days (scaled by sqrt rule)

    Returns
    -------
    float : VaR (positive number = loss)
    """
    q = np.quantile(returns, 1 - confidence)
    return -q * np.sqrt(horizon)


def es_historical(returns: pd.Series, confidence: float = 0.99,
                  horizon: int = 1) -> float:
    """
    Expected Shortfall (CVaR) via Historical Simulation.
    Average loss beyond the VaR threshold.
    """
    q = np.quantile(returns, 1 - confidence)
    tail_losses = returns[returns <= q]
    return -tail_losses.mean() * np.sqrt(horizon)


# ─── Parametric (Normal) ──────────────────────────────────────────────────────

def var_parametric(returns: pd.Series, confidence: float = 0.99,
                   horizon: int = 1) -> float:
    """
    Parametric (Variance-Covariance) VaR assuming Normal distribution.

    VaR = mu - z * sigma  where z = norm.ppf(1 - confidence)
    """
    mu    = returns.mean()
    sigma = returns.std()
    z     = stats.norm.ppf(1 - confidence)
    return -(mu + z * sigma) * np.sqrt(horizon)


def es_parametric(returns: pd.Series, confidence: float = 0.99,
                  horizon: int = 1) -> float:
    """
    Parametric Expected Shortfall under Normal distribution.

    ES = -mu + sigma * phi(z) / (1 - confidence)
    where phi is the standard normal PDF, z = norm.ppf(1-confidence).
    """
    mu    = returns.mean()
    sigma = returns.std()
    z     = stats.norm.ppf(1 - confidence)
    es    = -mu + sigma * stats.norm.pdf(z) / (1 - confidence)
    return es * np.sqrt(horizon)


# ─── Monte Carlo VaR ─────────────────────────────────────────────────────────

def var_monte_carlo(returns: pd.Series, confidence: float = 0.99,
                    horizon: int = 1, n_simulations: int = 100_000,
                    seed: int = 42) -> float:
    """
    Monte Carlo VaR.

    Fits a Normal distribution to historical returns and simulates
    the loss distribution over the holding period.
    """
    np.random.seed(seed)
    mu, sigma = returns.mean(), returns.std()
    simulated  = np.random.normal(mu * horizon,
                                   sigma * np.sqrt(horizon),
                                   n_simulations)
    return -np.quantile(simulated, 1 - confidence)


def es_monte_carlo(returns: pd.Series, confidence: float = 0.99,
                   horizon: int = 1, n_simulations: int = 100_000,
                   seed: int = 42) -> float:
    """Monte Carlo Expected Shortfall."""
    np.random.seed(seed)
    mu, sigma  = returns.mean(), returns.std()
    simulated  = np.random.normal(mu * horizon,
                                   sigma * np.sqrt(horizon),
                                   n_simulations)
    threshold  = np.quantile(simulated, 1 - confidence)
    return -simulated[simulated <= threshold].mean()


# ─── Backtesting (Kupiec POF Test) ───────────────────────────────────────────

def kupiec_pof_test(returns: pd.Series, var_series: pd.Series,
                    confidence: float = 0.99) -> dict:
    """
    Kupiec's Proportion of Failures (POF) Test for VaR backtesting.

    H0: The observed violation rate equals the expected rate (1 - confidence).
    Reject H0 if LR statistic > chi-squared critical value (df=1).

    Parameters
    ----------
    returns    : Realised daily returns
    var_series : Forecasted daily VaR (positive = loss)
    confidence : VaR confidence level

    Returns
    -------
    dict with violations, LR statistic, p-value, and test result
    """
    violations = (returns < -var_series).sum()
    T          = len(returns)
    p_hat      = violations / T
    p_expected = 1 - confidence

    if p_hat == 0 or p_hat == 1:
        return {"violations": violations, "violation_rate": 0,
                "lr_stat": np.nan, "p_value": np.nan, "reject_h0": False}

    lr = -2 * (
        violations * np.log(p_expected / p_hat) +
        (T - violations) * np.log((1 - p_expected) / (1 - p_hat))
    )
    p_value  = 1 - stats.chi2.cdf(lr, df=1)
    critical = stats.chi2.ppf(0.95, df=1)

    return {
        "n_observations":   T,
        "violations":       int(violations),
        "violation_rate":   round(p_hat * 100, 3),
        "expected_rate":    round(p_expected * 100, 3),
        "lr_statistic":     round(lr, 4),
        "p_value":          round(p_value, 4),
        "critical_value":   round(critical, 4),
        "reject_h0":        lr > critical,
    }


# ─── Comparison & Visualisation ──────────────────────────────────────────────

def var_comparison(returns: pd.Series, confidence: float = 0.99,
                   portfolio_value: float = 1_000_000) -> None:
    """
    Print and plot a comparison of all three VaR methods.
    """
    hist  = var_historical(returns, confidence)
    param = var_parametric(returns, confidence)
    mc    = var_monte_carlo(returns, confidence)
    es_h  = es_historical(returns, confidence)
    es_p  = es_parametric(returns, confidence)
    es_m  = es_monte_carlo(returns, confidence)

    print("=" * 55)
    print(f"  VaR & ES Comparison  (confidence={confidence*100:.0f}%)")
    print("=" * 55)
    print(f"  {'Method':<22} {'VaR':>10} {'VaR ($)':>12} {'ES':>10}")
    print("-" * 55)
    for name, v, e in [("Historical",    hist,  es_h),
                        ("Parametric",    param, es_p),
                        ("Monte Carlo",   mc,    es_m)]:
        print(f"  {name:<22} {v:>10.6f} "
              f"${v*portfolio_value:>10,.0f} {e:>10.6f}")
    print("=" * 55)

    # Plot return distribution with VaR lines
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(returns, bins=80, color="steelblue", alpha=0.7,
            density=True, label="Daily Returns")

    x = np.linspace(returns.min(), returns.max(), 500)
    ax.plot(x, stats.norm.pdf(x, returns.mean(), returns.std()),
            "k--", linewidth=1.5, label="Normal Fit")

    colors = {"Historical": "crimson", "Parametric": "darkorange", "Monte Carlo": "purple"}
    for name, v in [("Historical", hist), ("Parametric", param), ("Monte Carlo", mc)]:
        ax.axvline(-v, color=colors[name], linewidth=1.5,
                   linestyle="--", label=f"{name} VaR: {v:.4f}")

    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Density")
    ax.set_title(f"Return Distribution & VaR Estimates ({confidence*100:.0f}% Confidence)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("var_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: var_comparison.png")


if __name__ == "__main__":
    np.random.seed(42)
    returns = pd.Series(
        np.random.normal(0.0005, 0.015, 1000),
        name="returns"
    )

    var_comparison(returns, confidence=0.99)

    # Kupiec test
    rolling_var = pd.Series(
        [var_parametric(returns.iloc[max(0, i-252):i]) for i in range(1, len(returns)+1)],
        index=returns.index
    )
    kup = kupiec_pof_test(returns, rolling_var, confidence=0.99)
    print("\n=== Kupiec POF Backtest ===")
    for k, v in kup.items():
        print(f"  {k:<22}: {v}")
