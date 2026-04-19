"""
Black-Scholes Options Pricing Model
=====================================
Analytical pricing for European call and put options.
Includes Greeks: Delta, Gamma, Vega, Theta, Rho.

Reference: Black, F. & Scholes, M. (1973). The Pricing of Options and
           Corporate Liabilities. Journal of Political Economy.
"""

import numpy as np
from scipy.stats import norm


def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d1 term of the Black-Scholes formula."""
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d2 term of the Black-Scholes formula."""
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes European call option price.

    Parameters
    ----------
    S     : Current stock price
    K     : Strike price
    T     : Time to expiry (in years)
    r     : Risk-free interest rate (annualised)
    sigma : Volatility (annualised)

    Returns
    -------
    float : Call option price
    """
    if T <= 0:
        return max(S - K, 0.0)
    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)
    return S * norm.cdf(_d1) - K * np.exp(-r * T) * norm.cdf(_d2)


def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes European put option price.

    Uses put-call parity: P = C - S + K * exp(-rT)
    """
    return call_price(S, K, T, r, sigma) - S + K * np.exp(-r * T)


# ─── Greeks ──────────────────────────────────────────────────────────────────

def delta(S: float, K: float, T: float, r: float, sigma: float,
          option_type: str = "call") -> float:
    """Rate of change of option price with respect to S."""
    _d1 = d1(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(_d1)
    return norm.cdf(_d1) - 1


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Rate of change of delta with respect to S (same for call and put)."""
    _d1 = d1(S, K, T, r, sigma)
    return norm.pdf(_d1) / (S * sigma * np.sqrt(T))


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Sensitivity of option price to a 1% change in volatility."""
    _d1 = d1(S, K, T, r, sigma)
    return S * norm.pdf(_d1) * np.sqrt(T) * 0.01


def theta(S: float, K: float, T: float, r: float, sigma: float,
          option_type: str = "call") -> float:
    """Time decay: change in option price per calendar day."""
    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)
    term1 = -(S * norm.pdf(_d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        return (term1 - r * K * np.exp(-r * T) * norm.cdf(_d2)) / 365
    return (term1 + r * K * np.exp(-r * T) * norm.cdf(-_d2)) / 365


def rho(S: float, K: float, T: float, r: float, sigma: float,
        option_type: str = "call") -> float:
    """Sensitivity of option price to a 1% change in risk-free rate."""
    _d2 = d2(S, K, T, r, sigma)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(_d2) * 0.01
    return -K * T * np.exp(-r * T) * norm.cdf(-_d2) * 0.01


def implied_volatility(market_price: float, S: float, K: float, T: float,
                        r: float, option_type: str = "call",
                        tol: float = 1e-6, max_iter: int = 500) -> float:
    """
    Newton-Raphson solver for Implied Volatility.

    Parameters
    ----------
    market_price : Observed market price of the option
    tol          : Convergence tolerance
    max_iter     : Maximum iterations

    Returns
    -------
    float : Implied volatility, or NaN if not converged
    """
    sigma = 0.2  # initial guess
    price_fn = call_price if option_type == "call" else put_price

    for _ in range(max_iter):
        price = price_fn(S, K, T, r, sigma)
        v = vega(S, K, T, r, sigma) / 0.01  # undo the 1% scaling
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        if v < 1e-10:
            break
        sigma -= diff / v
        sigma = max(sigma, 1e-6)

    return float("nan")


def greeks_summary(S: float, K: float, T: float, r: float,
                   sigma: float) -> dict:
    """Return all Greeks for both call and put in a dictionary."""
    return {
        "call": {
            "price": call_price(S, K, T, r, sigma),
            "delta": delta(S, K, T, r, sigma, "call"),
            "gamma": gamma(S, K, T, r, sigma),
            "vega":  vega(S, K, T, r, sigma),
            "theta": theta(S, K, T, r, sigma, "call"),
            "rho":   rho(S, K, T, r, sigma, "call"),
        },
        "put": {
            "price": put_price(S, K, T, r, sigma),
            "delta": delta(S, K, T, r, sigma, "put"),
            "gamma": gamma(S, K, T, r, sigma),
            "vega":  vega(S, K, T, r, sigma),
            "theta": theta(S, K, T, r, sigma, "put"),
            "rho":   rho(S, K, T, r, sigma, "put"),
        },
    }


if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    g = greeks_summary(S, K, T, r, sigma)
    print("=== Black-Scholes Pricing ===")
    for otype, vals in g.items():
        print(f"\n  {otype.upper()}")
        for k, v in vals.items():
            print(f"    {k:8s}: {v:.6f}")

    iv = implied_volatility(g["call"]["price"], S, K, T, r, "call")
    print(f"\n  Implied Vol (call): {iv:.6f}  (expected: {sigma})")
