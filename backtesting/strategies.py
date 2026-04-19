"""
Trading Strategies
===================
Signal generators for the backtesting engine.

Strategies:
  1. Momentum (Time-Series)
  2. Mean Reversion (Bollinger Bands)
  3. Moving Average Crossover (SMA/EMA)
  4. RSI (Relative Strength Index)
"""

import numpy as np
import pandas as pd


def momentum_strategy(prices: pd.Series, lookback: int = 20) -> pd.Series:
    """
    Time-Series Momentum Strategy.

    Signal: +1 if trailing return > 0, else -1.
    Based on Moskowitz, Ooi & Pedersen (2012) — TSMOM.

    Parameters
    ----------
    prices   : Daily closing prices
    lookback : Lookback window (trading days)
    """
    trailing_return = prices.pct_change(lookback)
    signal = np.sign(trailing_return)
    return signal.rename("momentum_signal")


def mean_reversion_strategy(prices: pd.Series,
                             window: int = 20,
                             n_std: float = 2.0) -> pd.Series:
    """
    Bollinger Band Mean Reversion Strategy.

    Signal: -1 (short) when price > upper band, +1 (long) when price < lower band.
    Mean revert to the rolling moving average.

    Parameters
    ----------
    prices : Daily closing prices
    window : Rolling window for mean and std
    n_std  : Number of standard deviations for bands
    """
    rolling_mean = prices.rolling(window).mean()
    rolling_std  = prices.rolling(window).std()
    upper = rolling_mean + n_std * rolling_std
    lower = rolling_mean - n_std * rolling_std

    signal = pd.Series(0.0, index=prices.index)
    signal[prices > upper] = -1.0
    signal[prices < lower] =  1.0
    return signal.rename("mean_reversion_signal")


def ma_crossover_strategy(prices: pd.Series,
                           fast: int = 10,
                           slow: int = 50,
                           use_ema: bool = True) -> pd.Series:
    """
    Moving Average Crossover Strategy.

    Signal: +1 when fast MA > slow MA (uptrend), -1 otherwise.

    Parameters
    ----------
    fast    : Fast MA window
    slow    : Slow MA window
    use_ema : Use EMA (True) or SMA (False)
    """
    if use_ema:
        fast_ma = prices.ewm(span=fast, adjust=False).mean()
        slow_ma = prices.ewm(span=slow, adjust=False).mean()
    else:
        fast_ma = prices.rolling(fast).mean()
        slow_ma = prices.rolling(slow).mean()

    signal = np.sign(fast_ma - slow_ma)
    return signal.rename("ma_crossover_signal")


def rsi_strategy(prices: pd.Series,
                 period: int = 14,
                 overbought: float = 70.0,
                 oversold: float = 30.0) -> pd.Series:
    """
    RSI (Relative Strength Index) Contrarian Strategy.

    Signal: +1 when RSI < oversold (buy dip), -1 when RSI > overbought (sell rally).

    Parameters
    ----------
    period     : RSI period (default 14 days)
    overbought : RSI level above which we go short
    oversold   : RSI level below which we go long
    """
    delta  = prices.diff()
    gain   = delta.clip(lower=0).rolling(period).mean()
    loss   = (-delta.clip(upper=0)).rolling(period).mean()
    rs     = gain / loss.replace(0, np.nan)
    rsi    = 100 - (100 / (1 + rs))

    signal = pd.Series(0.0, index=prices.index)
    signal[rsi < oversold]   =  1.0
    signal[rsi > overbought] = -1.0
    return signal.rename("rsi_signal")


def generate_synthetic_prices(n_days: int = 1000,
                               S0: float = 100.0,
                               mu: float = 0.08,
                               sigma: float = 0.20,
                               seed: int = 42) -> pd.Series:
    """
    Generate synthetic GBM price series for testing.

    Parameters
    ----------
    n_days : Number of trading days
    S0     : Initial price
    mu     : Annual drift
    sigma  : Annual volatility
    seed   : Random seed
    """
    np.random.seed(seed)
    dt = 1 / 252
    returns = np.exp((mu - 0.5 * sigma ** 2) * dt +
                     sigma * np.sqrt(dt) * np.random.randn(n_days))
    prices = S0 * np.cumprod(returns)
    dates  = pd.bdate_range(start="2020-01-01", periods=n_days)
    return pd.Series(prices, index=dates, name="price")


if __name__ == "__main__":
    from engine import BacktestEngine

    prices = generate_synthetic_prices(n_days=1260, seed=42)

    strategies = {
        "Momentum (20d)":          momentum_strategy(prices, lookback=20),
        "Mean Reversion (BB 2σ)":  mean_reversion_strategy(prices),
        "EMA Crossover (10/50)":   ma_crossover_strategy(prices, fast=10, slow=50),
        "RSI (14d)":               rsi_strategy(prices),
    }

    print("\n" + "=" * 60)
    print("  Strategy Comparison on Synthetic GBM Prices")
    print("=" * 60)

    results = {}
    for name, signal in strategies.items():
        bt = BacktestEngine(prices, signal, transaction_cost=0.001)
        m  = bt.metrics()
        results[name] = m
        print(f"\n  {name}")
        print(f"    Sharpe: {m['Sharpe Ratio']:>7.4f}  |  "
              f"Ann. Ret: {m['Ann. Return (%)']:>6.2f}%  |  "
              f"Max DD: {m['Max Drawdown (%)']:>6.2f}%  |  "
              f"Trades: {m['Number of Trades']}")

    print("=" * 60)
