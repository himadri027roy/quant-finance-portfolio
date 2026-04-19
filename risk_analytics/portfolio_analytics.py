"""
Portfolio Analytics
====================
Mean-Variance Optimisation (Markowitz) and portfolio risk metrics.

Includes:
  - Efficient Frontier
  - Minimum Variance Portfolio
  - Maximum Sharpe Ratio Portfolio
  - Correlation / Covariance analysis
  - Portfolio stress testing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def portfolio_stats(weights: np.ndarray, mean_returns: np.ndarray,
                    cov_matrix: np.ndarray, risk_free_rate: float = 0.05,
                    trading_days: int = 252) -> tuple:
    """
    Compute annualised portfolio return, volatility, and Sharpe ratio.

    Parameters
    ----------
    weights       : Asset weight vector (sums to 1)
    mean_returns  : Vector of mean daily returns
    cov_matrix    : Covariance matrix of daily returns
    risk_free_rate: Annual risk-free rate

    Returns
    -------
    (ann_return, ann_vol, sharpe)
    """
    ann_return = np.dot(weights, mean_returns) * trading_days
    ann_vol    = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(trading_days)
    sharpe     = (ann_return - risk_free_rate) / ann_vol
    return ann_return, ann_vol, sharpe


def max_sharpe_portfolio(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                          risk_free_rate: float = 0.05,
                          allow_short: bool = False) -> dict:
    """
    Find the Maximum Sharpe Ratio (Tangency) Portfolio via numerical optimisation.
    """
    n = len(mean_returns)
    bounds = ((-1, 1),) * n if allow_short else ((0, 1),) * n

    def neg_sharpe(w):
        r, v, s = portfolio_stats(w, mean_returns, cov_matrix, risk_free_rate)
        return -s

    result = minimize(
        neg_sharpe,
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        options={"ftol": 1e-12, "maxiter": 1000}
    )

    w = result.x
    r, v, s = portfolio_stats(w, mean_returns, cov_matrix, risk_free_rate)
    return {"weights": w, "return": r, "volatility": v, "sharpe": s}


def min_variance_portfolio(mean_returns: np.ndarray,
                            cov_matrix: np.ndarray,
                            allow_short: bool = False) -> dict:
    """
    Find the Global Minimum Variance Portfolio.
    """
    n = len(mean_returns)
    bounds = ((-1, 1),) * n if allow_short else ((0, 1),) * n

    result = minimize(
        lambda w: w @ cov_matrix @ w,
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        options={"ftol": 1e-12, "maxiter": 1000}
    )

    w = result.x
    r, v, s = portfolio_stats(w, mean_returns, cov_matrix)
    return {"weights": w, "return": r, "volatility": v, "sharpe": s}


def efficient_frontier(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                        n_portfolios: int = 10_000,
                        risk_free_rate: float = 0.05,
                        seed: int = 42) -> pd.DataFrame:
    """
    Monte Carlo simulation of random portfolios to trace the efficient frontier.

    Returns DataFrame with return, volatility, Sharpe for each portfolio.
    """
    np.random.seed(seed)
    n = len(mean_returns)
    records = []

    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n))
        r, v, s = portfolio_stats(w, mean_returns, cov_matrix, risk_free_rate)
        records.append({"return": r, "volatility": v, "sharpe": s,
                        **{f"w{i}": w[i] for i in range(n)}})

    return pd.DataFrame(records)


def plot_efficient_frontier(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                             asset_names: list, risk_free_rate: float = 0.05) -> None:
    """
    Plot the efficient frontier, tangency portfolio, and min-variance portfolio.
    """
    ef   = efficient_frontier(mean_returns, cov_matrix,
                               risk_free_rate=risk_free_rate)
    msr  = max_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate)
    gmv  = min_variance_portfolio(mean_returns, cov_matrix)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Mean-Variance Efficient Frontier", fontsize=14, fontweight="bold")

    # Scatter of random portfolios
    sc = axes[0].scatter(ef["volatility"] * 100, ef["return"] * 100,
                          c=ef["sharpe"], cmap="viridis", alpha=0.5, s=5)
    plt.colorbar(sc, ax=axes[0], label="Sharpe Ratio")

    axes[0].scatter(msr["volatility"] * 100, msr["return"] * 100,
                    color="gold", edgecolors="black", s=200, zorder=5,
                    label=f"Max Sharpe ({msr['sharpe']:.2f})", marker="*")
    axes[0].scatter(gmv["volatility"] * 100, gmv["return"] * 100,
                    color="crimson", edgecolors="black", s=150, zorder=5,
                    label=f"Min Variance", marker="D")

    # Capital Market Line
    vols = np.linspace(0, ef["volatility"].max() * 100, 100)
    cml  = risk_free_rate * 100 + msr["sharpe"] * vols
    axes[0].plot(vols, cml, "k--", linewidth=1, label="Capital Market Line")

    axes[0].set_xlabel("Volatility (%)")
    axes[0].set_ylabel("Expected Return (%)")
    axes[0].set_title("Efficient Frontier")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Max Sharpe weights bar chart
    axes[1].bar(asset_names, msr["weights"] * 100, color="steelblue", edgecolor="black")
    axes[1].set_title("Max Sharpe Portfolio Weights")
    axes[1].set_ylabel("Weight (%)")
    axes[1].set_xlabel("Asset")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("efficient_frontier.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: efficient_frontier.png")


def stress_test(weights: np.ndarray, mean_returns: np.ndarray,
                cov_matrix: np.ndarray,
                scenarios: dict, asset_names: list) -> pd.DataFrame:
    """
    Stress test a portfolio under hypothetical scenarios.

    Parameters
    ----------
    scenarios : dict of {scenario_name: array of asset return shocks}

    Returns
    -------
    DataFrame with portfolio P&L under each scenario
    """
    results = []
    for name, shocks in scenarios.items():
        portfolio_loss = np.dot(weights, shocks)
        results.append({"Scenario": name, "Portfolio Return (%)": round(portfolio_loss * 100, 2)})
    return pd.DataFrame(results).set_index("Scenario")


if __name__ == "__main__":
    np.random.seed(42)
    n_assets = 5
    asset_names = ["Equity", "Bonds", "Gold", "Real Estate", "Commodities"]

    # Synthetic parameters
    mean_returns = np.array([0.08, 0.03, 0.04, 0.06, 0.05]) / 252
    corr = np.array([
        [1.00,  0.10,  0.05,  0.50,  0.30],
        [0.10,  1.00, -0.15,  0.20, -0.05],
        [0.05, -0.15,  1.00,  0.00,  0.20],
        [0.50,  0.20,  0.00,  1.00,  0.25],
        [0.30, -0.05,  0.20,  0.25,  1.00],
    ])
    vols = np.array([0.18, 0.05, 0.15, 0.12, 0.20]) / np.sqrt(252)
    cov_matrix = np.outer(vols, vols) * corr

    plot_efficient_frontier(mean_returns, cov_matrix, asset_names)

    msr = max_sharpe_portfolio(mean_returns, cov_matrix)
    print("\n=== Max Sharpe Portfolio ===")
    for asset, w in zip(asset_names, msr["weights"]):
        print(f"  {asset:<15}: {w*100:.1f}%")
    print(f"  Return    : {msr['return']*100:.2f}%")
    print(f"  Volatility: {msr['volatility']*100:.2f}%")
    print(f"  Sharpe    : {msr['sharpe']:.4f}")

    # Stress test
    scenarios = {
        "2008 GFC":          np.array([-0.40, 0.05, 0.15, -0.30, -0.25]),
        "COVID Crash (2020)":np.array([-0.35, 0.08, 0.10, -0.20, -0.30]),
        "Rate Shock (+300bps)": np.array([-0.15, -0.12, 0.05, -0.10, 0.00]),
        "Equity Rally":      np.array([ 0.30, -0.02, -0.05, 0.15, 0.10]),
    }
    st = stress_test(msr["weights"], mean_returns, cov_matrix, scenarios, asset_names)
    print("\n=== Stress Test Results ===")
    print(st.to_string())
