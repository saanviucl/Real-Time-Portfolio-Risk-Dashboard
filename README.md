# Real-Time Portfolio Risk Dashboard

Streamlit app for live portfolio analytics: market data, VaR/ES/Sharpe/Beta, stress tests, an advisor that surfaces regime-aware ideas, a what-if simulator, and a formulas tab.

## Features 

- **Input:** Manual table or CSV (`ticker,quantity`). Auto-detects asset class. Dedupes tickers.
- **Metrics:** Parametric & Historical **VaR**, **CVaR/ES**, **Sharpe**, **Beta**, **CAGR** (1d/1wk/1mo aware).
- **Visuals:** Allocation, Contribution to Risk, Rolling Vol & VaR, Drawdown, Risk–Return scatter.
- **Stress tests:** Uniform price shock, volatility spike, correlation breakdown; Stressed/Adjusted VaR.
- **Advisor:** Classifies regime (Crisis/High-Vol/Calm/Normal), scans S&P 500 top performers (1W/1M) via keyless Yahoo, scores by momentum/trend/diversification/low-vol.
- **What-If:** Add a ticker by quantity or target weight; see deltas for VaR/ES/Sharpe/Beta/Return/Vol plus a short recommendation.
- **Formulas:** LaTeX for methods used.


## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # add ALPHAVANTAGE_API_KEY if you want
streamlit run app.py
```

## Data input

**Manual:** Sidebar → Portfolio Input → *Manual entry* (editable table, big ticker dropdown).
**CSV:** Sidebar → *Upload CSV* (schema below). Duplicates are collapsed (quantities summed).

```csv
ticker,quantity
AAPL,10
MSFT,8
TSLA,4
TLT,5
GLD,3
```

## Controls

* Interval: `1d / 1wk / 1mo` (annualization adapts)
* Risk-free (annual %) → affects **Sharpe**
* Confidence & Horizon → **VaR/ES**
* Stress: price shock, vol x, correlation breakdown

## Methods

* **Annualized Return:** $(1+\bar r)^P-1$
* **Vol:** $\sigma\sqrt{P}$
* **Sharpe:** $((\bar r - rf_{\text{period}})/\sigma)\sqrt{P}$
* **CAGR:** $(\prod(1+r))^{1/\text{years}}-1$
* **Beta:** $\mathrm{Cov}(r_p,r_b)/\mathrm{Var}(r_b)$
* **Parametric VaR:** $-(\mu h + z_\alpha \sigma \sqrt{h})$
* **Historical VaR/ES:** empirical quantile & tail mean

## Files

`app.py`, `data.py`, `risk.py`, `utils.py`, `portfolio.csv`, `requirements.txt`, `.env.example`, `README.md`

## Demo

* Watch the demo on Youtube

[![Watch the demo](https://img.youtube.com/vi/qUH16xhL33k/hqdefault.jpg)](https://youtu.be/qUH16xhL33k "Watch the demo on YouTube")

* Try it yourself

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://real-time-portfolio-risk-dashboard-4kkrglhbtu9pcj7wqdbf8a.streamlit.app)
