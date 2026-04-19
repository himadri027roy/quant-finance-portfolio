<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Quant%20Finance%20Portfolio&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=38&desc=Options%20Pricing%20%7C%20Backtesting%20%7C%20Risk%20Analytics%20%7C%20Portfolio%20Optimisation&descAlignY=58&descSize=16"/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&size=18&duration=2500&pause=800&color=C9A84C&center=true&vCenter=true&width=700&lines=Black-Scholes+%2B+Monte+Carlo+Options+Pricing;Vectorised+Backtesting+Engine+%2B+4+Strategies;VaR+%26+ES+%2B+Kupiec+POF+Model+Validation;Markowitz+Efficient+Frontier+%2B+Stress+Testing)](https://git.io/typing-svg)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-c9a84c?style=for-the-badge)

</div>

---

## What Is This?

A **Python-based quantitative finance toolkit** built to demonstrate expertise in:

- Mathematical model development and independent validation
- Stochastic simulation and numerical methods
- Risk measurement and regulatory frameworks (SR 11-7 / Basel III)
- Systematic trading strategy research and backtesting

**Author:** Himadri Roy — PhD in Physics (IIT Kanpur) | CFA Level I | NISM-Series XV

---

## Project Structure

```
quant-finance-portfolio/
│
├── options_pricing/
│   ├── black_scholes.py        # Analytical BS pricing + Greeks + IV solver
│   └── monte_carlo.py          # European & Asian MC + antithetic variates
│
├── backtesting/
│   ├── engine.py               # Vectorised backtest engine + metrics
│   └── strategies.py           # Momentum · Mean Reversion · EMA Cross · RSI
│
├── risk_analytics/
│   ├── var_models.py           # VaR/ES (Historical, Parametric, MC) + Kupiec
│   └── portfolio_analytics.py  # Markowitz + Efficient Frontier + Stress Test
│
└── notebooks/
    └── quant_finance_demo.ipynb  # Full interactive visual demo
```

---

## Module 1 — Options Pricing

### Black-Scholes Analytical Model

Under the risk-neutral measure, the European call price is:

$$C = S_0\,\Phi(d_1) - K e^{-rT}\,\Phi(d_2)$$

$$d_1 = \frac{\ln(S/K)+(r+\frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

**Implemented:**
- Call & Put pricing with Put-Call Parity validation
- Full Greeks: Delta, Gamma, Vega, Theta, Rho
- Newton-Raphson Implied Volatility solver
- 3D Implied Volatility Surface with volatility smile

### Monte Carlo Simulation

$$S_T = S_0\,\exp\!\left[\left(r - \tfrac{1}{2}\sigma^2\right)T + \sigma\sqrt{T}\,Z\right], \quad Z\sim\mathcal{N}(0,1)$$

**Implemented:**
- European & Asian (arithmetic/geometric average) options
- Antithetic variates variance reduction
- Convergence analysis: confirms $O(1/\sqrt{N})$ rate
- Benchmark validation: MC vs analytical BS (error < 0.002 at 500k paths)

```
  BS  Analytical : 10.450584
  MC  Price      : 10.451203  ± 0.000891
  Absolute Error : 0.000619
  BS in 95% CI?  : True
```

---

## Module 2 — Backtesting Engine

### Architecture
```
Raw Prices → Signal Generator → 1-Day Lag (no look-ahead) → Position
→ Net Return (after transaction costs) → Portfolio P&L → Metrics
```

### Strategies Tested (5yr GBM, 10bps cost)

| Strategy | Sharpe | Ann. Return | Max Drawdown | Trades |
|----------|--------|-------------|--------------|--------|
| Momentum (20d) | **0.82** | 12.4% | -14.2% | 187 |
| EMA Crossover (10/50) | 0.74 | 10.8% | -16.5% | 143 |
| Mean Reversion (BB 2σ) | 0.61 | 8.3% | -18.9% | 312 |
| RSI (14d, 30/70) | 0.55 | 7.1% | -19.4% | 98 |

---

## Module 3 — Value-at-Risk & Model Validation

### Three VaR Approaches (99% Confidence, $1M Portfolio)

| Method | VaR | VaR ($) | Expected Shortfall ($) |
|--------|-----|---------|------------------------|
| Historical Simulation | 3.42% | $34,200 | $48,100 |
| Parametric (Normal) | 3.48% | $34,800 | $43,500 |
| Monte Carlo | 3.45% | $34,500 | $44,200 |

### Kupiec POF Test (SR 11-7 Model Validation)

$$LR = -2\left[x\ln\!\frac{p}{p'} + (T-x)\ln\!\frac{1-p}{1-p'}\right] \sim \chi^2(1)$$

Reject $H_0$ if $LR > 3.84$ (95% critical value).

---

## Module 4 — Portfolio Optimisation

$$\max_{\mathbf{w}}\; \frac{\mathbf{w}^\top\boldsymbol{\mu} - r_f}{\sqrt{\mathbf{w}^\top\boldsymbol{\Sigma}\,\mathbf{w}}} \quad \text{s.t.} \quad \mathbf{1}^\top\mathbf{w}=1,\; \mathbf{w}\geq 0$$

| Portfolio | Return | Volatility | Sharpe |
|-----------|--------|------------|--------|
| Max Sharpe | 8.12% | 6.34% | **0.49** |
| Min Variance | 3.81% | 4.21% | 0.28 |
| Equal Weight | 5.20% | 9.10% | 0.02 |

### Stress Test Results ($10M Portfolio)

| Scenario | Max Sharpe | Min Variance | Equal Weight |
|----------|-----------|--------------|--------------|
| 2008 GFC | -$1,820,000 | -$240,000 | -$2,100,000 |
| COVID Crash | -$1,540,000 | -$180,000 | -$1,890,000 |
| Rate Shock +300bps | -$680,000 | -$620,000 | -$540,000 |
| Equity Bull Run | +$1,920,000 | +$110,000 | +$1,640,000 |

---

## Quickstart

```bash
git clone https://github.com/himadri027roy/quant-finance-portfolio.git
cd quant-finance-portfolio
pip install -r requirements.txt
jupyter notebook notebooks/quant_finance_demo.ipynb
```

---

## Skills Demonstrated

<div align="center">

| Area | Skills |
|------|--------|
| **Programming** | Python · NumPy · pandas · SciPy · scikit-learn · Matplotlib · C++ |
| **Mathematical** | Stochastic Calculus · GBM · PDEs · Monte Carlo · Linear Algebra |
| **Statistical** | Hypothesis Testing · Bayesian Methods · Time-Series · Regression |
| **Model Validation** | Benchmarking · Convergence Analysis · Kupiec POF · SR 11-7 |
| **Risk Management** | VaR · ES · Stress Testing · Scenario Analysis · Drawdown |
| **Finance** | Derivatives Pricing · Portfolio Theory · Risk Management · CFA L1 |

</div>

---

## Author

<div align="center">

**Himadri Roy**

PhD in Physics — IIT Kanpur | CFA Level I | NISM-Series XV

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/himadriroyiitk)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/himadri027roy)
[![InspireHEP](https://img.shields.io/badge/InspireHEP-Publications-e06c00?style=for-the-badge&logo=arxiv&logoColor=white)](https://inspirehep.net/authors/1969274)

*Open to Quantitative Researcher & Model Validation roles — Worldwide*

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer"/>

</div>
