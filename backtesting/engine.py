"""
Backtesting Engine
===================
A vectorised backtesting framework for systematic trading strategies.

Features:
  - Position sizing
  - Transaction cost modelling
  - Portfolio-level P&L tracking
  - Performance metrics (Sharpe, Sortino, Max Drawdown, CAGR)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


class BacktestEngine:
    """
    Vectorised backtesting engine.

    Parameters
    ----------
    prices       : pd.Series or pd.DataFrame of asset prices (daily close)
    signals      : pd.Series of position signals (+1 long, -1 short, 0 flat)
    transaction_cost : One-way cost as a fraction (e.g., 0.001 = 10 bps)
    initial_capital  : Starting portfolio value in USD
    """

    def __init__(self, prices: pd.Series, signals: pd.Series,
                 transaction_cost: float = 0.001,
                 initial_capital: float = 1_000_000.0):
        self.prices = prices.copy()
        self.signals = signals.reindex(prices.index).fillna(0)
        self.cost = transaction_cost
        self.capital = initial_capital
        self.results: Optional[pd.DataFrame] = None

    def run(self) -> pd.DataFrame:
        """Execute backtest and return results DataFrame."""
        df = pd.DataFrame(index=self.prices.index)
        df["price"]  = self.prices
        df["signal"] = self.signals

        # Daily returns
        df["returns"] = df["price"].pct_change()

        # Position: signal is applied with 1-day lag to avoid look-ahead bias
        df["position"] = df["signal"].shift(1).fillna(0)

        # Trade indicator: position changed
        df["trade"] = df["position"].diff().abs()

        # Strategy gross and net returns
        df["gross_return"] = df["position"] * df["returns"]
        df["net_return"]   = df["gross_return"] - df["trade"] * self.cost

        # Cumulative portfolio value
        df["cum_return"]   = (1 + df["net_return"]).cumprod()
        df["portfolio"]    = self.capital * df["cum_return"]

        # Drawdown
        rolling_max        = df["portfolio"].cummax()
        df["drawdown"]     = (df["portfolio"] - rolling_max) / rolling_max

        self.results = df
        return df

    def metrics(self) -> dict:
        """Compute key performance metrics."""
        if self.results is None:
            self.run()

        r = self.results["net_return"].dropna()
        port = self.results["portfolio"].dropna()

        trading_days = 252
        ann_return = (port.iloc[-1] / port.iloc[0]) ** (trading_days / len(r)) - 1
        ann_vol    = r.std() * np.sqrt(trading_days)
        sharpe     = ann_return / ann_vol if ann_vol > 0 else 0

        downside   = r[r < 0].std() * np.sqrt(trading_days)
        sortino    = ann_return / downside if downside > 0 else 0

        max_dd     = self.results["drawdown"].min()
        calmar     = ann_return / abs(max_dd) if max_dd != 0 else 0
        n_trades   = (self.results["trade"] > 0).sum()
        win_rate   = (r[self.results["position"].shift(1) != 0] > 0).mean()

        bh_return  = (self.results["price"].iloc[-1] /
                      self.results["price"].iloc[0]) - 1

        return {
            "Total Return (%)":      round((port.iloc[-1] / self.capital - 1) * 100, 2),
            "Ann. Return (%)":       round(ann_return * 100, 2),
            "Ann. Volatility (%)":   round(ann_vol * 100, 2),
            "Sharpe Ratio":          round(sharpe, 4),
            "Sortino Ratio":         round(sortino, 4),
            "Calmar Ratio":          round(calmar, 4),
            "Max Drawdown (%)":      round(max_dd * 100, 2),
            "Win Rate (%)":          round(win_rate * 100, 2),
            "Number of Trades":      int(n_trades),
            "Buy & Hold Return (%)": round(bh_return * 100, 2),
        }

    def plot(self, title: str = "Strategy Backtest") -> None:
        """Plot portfolio value, drawdown, and signals."""
        if self.results is None:
            self.run()

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(title, fontsize=14, fontweight="bold")

        # Portfolio vs buy-and-hold
        bh = self.capital * (1 + self.results["returns"]).cumprod()
        axes[0].plot(self.results["portfolio"], label="Strategy", color="royalblue", linewidth=1.5)
        axes[0].plot(bh, label="Buy & Hold", color="gray", linestyle="--", linewidth=1)
        axes[0].set_ylabel("Portfolio Value ($)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Drawdown
        axes[1].fill_between(self.results.index, self.results["drawdown"] * 100,
                             color="crimson", alpha=0.5)
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].grid(True, alpha=0.3)

        # Signal
        axes[2].plot(self.results["position"], color="seagreen", linewidth=0.8)
        axes[2].axhline(0, color="black", linewidth=0.5)
        axes[2].set_ylabel("Position")
        axes[2].set_xlabel("Date")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        fname = title.lower().replace(" ", "_") + ".png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved: {fname}")

    def print_metrics(self) -> None:
        m = self.metrics()
        print("\n" + "=" * 40)
        print("  Performance Metrics")
        print("=" * 40)
        for k, v in m.items():
            print(f"  {k:<28}: {v}")
        print("=" * 40)
