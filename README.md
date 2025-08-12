# Real-Time Portfolio Risk Dashboard

Streamlit app for portfolio analytics: **VaR, CVaR/ES, Sharpe, Beta, CAGR**, rolling risk, drawdowns, stress tests. Data from **Yahoo Finance** (no key) or **Alpha Vantage** (optional).

## Features

* Manual entry **or** CSV upload (+ large ticker dropdown from NASDAQ lists)
* Metrics: Parametric & Historical **VaR**, **CVaR/ES**, **Sharpe** (rf-aware), **Beta**, **Annualized Return**, **CAGR**
* Visuals: allocation, **contribution to risk**, **rolling vol & VaR**, **drawdowns**, risk–return scatter
* Stress tests: vol multiplier, correlation breakdown, **uniform price shock** (+ Shock-Adjusted VaR)
* Caching for speed; optional **Ledoit–Wolf** covariance shrinkage

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
ticker,quantity,asset_class
AAPL,10,Equity
MSFT,8,Equity
TSLA,4,Equity
TLT,5,Bond
GLD,3,Commodity
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

[![Watch the demo](https://img.youtube.com/vi/qUH16xhL33k/hqdefault.jpg)](https://youtu.be/qUH16xhL33k "Watch the demo on YouTube")
