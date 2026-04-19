"""
Monte Carlo Options Pricing
============================
Prices European and Asian options via Monte Carlo simulation
under Geometric Brownian Motion (GBM).

Also includes:
  - Variance reduction via Antithetic Variates
  - Benchmarking against Black-Scholes analytical prices
  - Convergence analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from black_scholes import call_price as bs_call, put_price as bs_put


def simulate_gbm(S0: float, r: float, sigma: float, T: float,
                 n_steps: int, n_paths: int,
                 antithetic: bool = True,
                 seed: int = 42) -> np.ndarray:
    """
    Simulate stock price paths under GBM.

    dS = r*S*dt + sigma*S*dW

    Parameters
    ----------
    S0         : Initial stock price
    r          : Risk-free rate
    sigma      : Volatility
    T          : Time to maturity (years)
    n_steps    : Number of time steps
    n_paths    : Number of simulation paths
    antithetic : Use antithetic variates for variance reduction
    seed       : Random seed for reproducibility

    Returns
    -------
    np.ndarray : Shape (n_paths, n_steps+1) of simulated prices
    """
    np.random.seed(seed)
    dt = T / n_steps
    n = n_paths // 2 if antithetic else n_paths

    Z = np.random.standard_normal((n, n_steps))
    if antithetic:
        Z = np.concatenate([Z, -Z], axis=0)

    log_returns = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(np.cumsum(log_returns, axis=1))
    return paths


def price_european(S: float, K: float, T: float, r: float, sigma: float,
                   n_paths: int = 100_000, n_steps: int = 252,
                   option_type: str = "call",
                   antithetic: bool = True,
                   seed: int = 42) -> dict:
    """
    Monte Carlo price for a European option.

    Returns price, standard error, and 95% confidence interval.
    """
    paths = simulate_gbm(S, r, sigma, T, n_steps, n_paths, antithetic, seed)
    S_T = paths[:, -1]

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)

    discounted = np.exp(-r * T) * payoffs
    price = discounted.mean()
    stderr = discounted.std() / np.sqrt(n_paths)

    return {
        "price":  price,
        "stderr": stderr,
        "ci_95":  (price - 1.96 * stderr, price + 1.96 * stderr),
    }


def price_asian(S: float, K: float, T: float, r: float, sigma: float,
                n_paths: int = 100_000, n_steps: int = 252,
                option_type: str = "call",
                averaging: str = "arithmetic",
                seed: int = 42) -> dict:
    """
    Monte Carlo price for an Asian option (average price).

    Asian options have no closed-form solution — MC is the standard approach.

    Parameters
    ----------
    averaging : 'arithmetic' or 'geometric' averaging of path
    """
    paths = simulate_gbm(S, r, sigma, T, n_steps, n_paths, seed=seed)

    if averaging == "arithmetic":
        avg = paths[:, 1:].mean(axis=1)
    else:
        avg = np.exp(np.log(paths[:, 1:]).mean(axis=1))

    if option_type == "call":
        payoffs = np.maximum(avg - K, 0)
    else:
        payoffs = np.maximum(K - avg, 0)

    discounted = np.exp(-r * T) * payoffs
    price = discounted.mean()
    stderr = discounted.std() / np.sqrt(n_paths)

    return {
        "price":  price,
        "stderr": stderr,
        "ci_95":  (price - 1.96 * stderr, price + 1.96 * stderr),
    }


def convergence_analysis(S: float, K: float, T: float, r: float, sigma: float,
                          max_paths: int = 100_000,
                          option_type: str = "call") -> None:
    """
    Plot Monte Carlo convergence vs Black-Scholes analytical price.
    Shows how MC error decreases as number of paths increases (1/sqrt(N)).
    """
    bs_price = bs_call(S, K, T, r, sigma) if option_type == "call" else bs_put(S, K, T, r, sigma)
    path_counts = np.logspace(2, np.log10(max_paths), 30, dtype=int)

    mc_prices, errors, stderrs = [], [], []
    for n in path_counts:
        res = price_european(S, K, T, r, sigma, n_paths=n,
                             option_type=option_type, seed=42)
        mc_prices.append(res["price"])
        errors.append(abs(res["price"] - bs_price))
        stderrs.append(res["stderr"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Monte Carlo Convergence Analysis", fontsize=14, fontweight="bold")

    axes[0].plot(path_counts, mc_prices, label="MC Price", color="royalblue")
    axes[0].axhline(bs_price, color="crimson", linestyle="--", label=f"BS Price ({bs_price:.4f})")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Number of Paths")
    axes[0].set_ylabel("Option Price")
    axes[0].set_title("Price Convergence")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].loglog(path_counts, errors, label="|MC - BS|", color="royalblue")
    axes[1].loglog(path_counts, stderrs, label="Std Error", color="orange", linestyle="--")
    ref = errors[0] * (path_counts[0] / np.array(path_counts)) ** 0.5
    axes[1].loglog(path_counts, ref, "k:", label="1/√N reference")
    axes[1].set_xlabel("Number of Paths")
    axes[1].set_ylabel("Error")
    axes[1].set_title("Error Convergence (log-log)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("mc_convergence.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: mc_convergence.png")


def benchmark_vs_black_scholes(S: float = 100, K: float = 100,
                                T: float = 1.0, r: float = 0.05,
                                sigma: float = 0.2) -> None:
    """Print side-by-side comparison of MC vs analytical BS prices."""
    print("=" * 55)
    print("  Monte Carlo vs Black-Scholes Benchmark")
    print("=" * 55)
    print(f"  S={S}, K={K}, T={T}, r={r}, sigma={sigma}")
    print("-" * 55)

    for otype in ["call", "put"]:
        bs = bs_call(S, K, T, r, sigma) if otype == "call" else bs_put(S, K, T, r, sigma)
        mc = price_european(S, K, T, r, sigma, n_paths=500_000, option_type=otype)
        err = abs(mc["price"] - bs)
        ci = mc["ci_95"]
        in_ci = ci[0] <= bs <= ci[1]
        print(f"\n  {otype.upper()}")
        print(f"    BS  Price : {bs:.6f}")
        print(f"    MC  Price : {mc['price']:.6f}  ± {mc['stderr']:.6f}")
        print(f"    95% CI    : [{ci[0]:.6f}, {ci[1]:.6f}]")
        print(f"    Abs Error : {err:.6f}")
        print(f"    BS in CI  : {in_ci}")
    print("=" * 55)


if __name__ == "__main__":
    benchmark_vs_black_scholes()
    convergence_analysis(S=100, K=100, T=1.0, r=0.05, sigma=0.2)

    print("\n=== Asian Option (Arithmetic Average Call) ===")
    asian = price_asian(100, 100, 1.0, 0.05, 0.2)
    print(f"  Price  : {asian['price']:.6f}")
    print(f"  Stderr : {asian['stderr']:.6f}")
    print(f"  95% CI : [{asian['ci_95'][0]:.6f}, {asian['ci_95'][1]:.6f}]")
