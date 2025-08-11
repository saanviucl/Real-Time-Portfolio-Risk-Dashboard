
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm

def clean_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    # columns: ticker, quantity, asset_class
    df = df.copy()
    df["ticker"] = df["ticker"].str.upper().str.strip()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)
    if "asset_class" not in df.columns:
        df["asset_class"] = "Unknown"
    return df

def portfolio_weights(latest_prices: pd.Series, positions: pd.DataFrame) -> pd.Series:
    # value weights
    value = latest_prices.reindex(positions["ticker"]).fillna(0) * positions.set_index("ticker")["quantity"]
    total = value.sum()
    if total <= 0:
        return value * 0
    return value / total

def parametric_var(returns: pd.DataFrame, weights: pd.Series, alpha: float=0.95, periods: int=1) -> float:
    # Gaussian VaR; returns are daily
    mu = returns.mean()
    cov = returns.cov()
    port_mu = float(weights @ mu)
    port_sigma = float(np.sqrt(weights.T @ cov.values @ weights))
    z = norm.ppf(1 - alpha)
    var = -(port_mu*periods + z * port_sigma * np.sqrt(periods))
    return max(0.0, var)

def historical_var(returns: pd.DataFrame, weights: pd.Series, alpha: float=0.95, periods: int=1) -> float:
    port_ret = returns @ weights
    # aggregate periods (sum of log-approx) -> simple sum is OK for small returns
    agg = port_ret.rolling(periods).sum().dropna()
    q = agg.quantile(1 - alpha)
    return max(0.0, -float(q))

def cvar_es(returns: pd.DataFrame, weights: pd.Series, alpha: float=0.95, periods: int=1) -> float:
    port_ret = returns @ weights
    agg = port_ret.rolling(periods).sum().dropna()
    cutoff = agg.quantile(1 - alpha)
    tail = agg[agg <= cutoff]
    if len(tail) == 0:
        return 0.0
    es = -float(tail.mean())
    return es

def beta_vs_benchmark(asset_returns: pd.Series, bench_returns: pd.Series) -> float:
    cov = np.cov(asset_returns.dropna(), bench_returns.dropna())
    if cov.shape != (2,2) or np.isnan(cov).any():
        return np.nan
    var_b = cov[1,1]
    if var_b <= 0:
        return np.nan
    return cov[0,1] / var_b

def sharpe_ratio(returns: pd.Series, rf_daily: float=0.0) -> float:
    excess = returns - rf_daily
    mu = excess.mean()
    sd = excess.std()
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(mu / sd) * np.sqrt(252)
