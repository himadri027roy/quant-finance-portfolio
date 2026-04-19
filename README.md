# Quant Finance Portfolio

A Python-based quantitative finance toolkit demonstrating skills in options pricing, backtesting, and risk analytics — built to showcase expertise relevant to **Quantitative Researcher** and **Model Validation** roles.

## Modules

### 1. Options Pricing (`options_pricing/`)

| File | Description |
|---|---|
| `black_scholes.py` | Analytical Black-Scholes pricing for European options + full Greeks (Delta, Gamma, Vega, Theta, Rho) + Implied Volatility solver (Newton-Raphson) |
| `monte_carlo.py` | Monte Carlo pricing for European and Asian options with antithetic variates, benchmarking vs BS, and convergence analysis |

**Key concepts:** Geometric Brownian Motion, risk-neutral pricing, variance reduction, model validation via benchmarking.

---

### 2. Backtesting Engine (`backtesting/`)

| File | Description |
|---|---|
| `engine.py` | Vectorised backtesting engine with transaction costs, P&L tracking, drawdown, and full performance metrics |
| `strategies.py` | Four signal generators: Momentum, Mean Reversion (Bollinger Bands), MA Crossover (EMA/SMA), RSI |

**Key concepts:** Look-ahead bias prevention, Sharpe/Sortino/Calmar ratios, strategy comparison, synthetic GBM data generation.

---

### 3. Risk Analytics (`risk_analytics/`)

| File | Description |
|---|---|
| `var_models.py` | VaR and Expected Shortfall via Historical Simulation, Parametric (Normal), and Monte Carlo; Kupiec POF backtesting |
| `portfolio_analytics.py` | Markowitz Mean-Variance Optimisation, Efficient Frontier, Max Sharpe portfolio, stress testing (GFC, COVID, rate shock) |

**Key concepts:** Basel III/SR 11-7 model validation, Kupiec test, stress testing, mean-variance optimisation, Capital Market Line.

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/quant-finance-portfolio.git
cd quant-finance-portfolio

# Install dependencies
pip install -r requirements.txt

# Run options pricing demo
cd options_pricing
python black_scholes.py
python monte_carlo.py

# Run backtesting demo
cd ../backtesting
python strategies.py

# Run risk analytics demo
cd ../risk_analytics
python var_models.py
python portfolio_analytics.py
```

---

## Sample Results

### Black-Scholes vs Monte Carlo (500k paths, antithetic variates)
```
Method       BS Price    MC Price      Abs Error
Call         10.4506     10.4491       0.0015
Put           5.5735      5.5722       0.0013
```

### Strategy Comparison (Synthetic GBM, 5 years, 10 bps cost)
```
Strategy               Sharpe   Ann. Return   Max DD
Momentum (20d)          0.82       12.4%      -14.2%
EMA Crossover (10/50)   0.74       10.8%      -16.5%
Mean Reversion (BB)     0.61        8.3%      -18.9%
RSI (14d)               0.55        7.1%      -19.4%
```

### VaR Comparison (99% confidence, 1-day horizon)
```
Method          VaR       VaR ($1M portfolio)
Historical      3.42%     $34,200
Parametric      3.48%     $34,800
Monte Carlo     3.45%     $34,500
```

---

## Skills Demonstrated

- **Python:** NumPy, pandas, SciPy, Matplotlib, scikit-learn
- **Mathematical modelling:** GBM, stochastic calculus, PDEs, Monte Carlo
- **Model validation:** Benchmarking, convergence analysis, Kupiec POF test
- **Statistical analysis:** Time-series, hypothesis testing, regression, optimisation
- **Risk management:** VaR, ES, stress testing, portfolio optimisation
- **Finance:** Derivatives pricing, Greeks, portfolio theory, Basel III concepts

---

## Author

**Your Name** — PhD in Physics (IIT Kanpur) | CFA Level I
[LinkedIn](https://linkedin.com/in/yourprofile) | [Google Scholar](https://scholar.google.com)
