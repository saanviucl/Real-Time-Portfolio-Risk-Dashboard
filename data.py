from __future__ import annotations
import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from functools import lru_cache

ALPHA_BASE = "https://www.alphavantage.co/query"

def _fetch_yfinance_prices(tickers: list[str], start: str, end: str, interval: str="1d") -> pd.DataFrame:
    data = yf.download(tickers, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if isinstance(data, pd.DataFrame) and "Close" in data.columns:
        data = data["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data.dropna(how="all")

def _fetch_alpha_prices(ticker: str, outputsize: str="compact") -> pd.Series:
    key = os.getenv("ALPHAVANTAGE_API_KEY", "")
    if not key:
        raise RuntimeError("Missing ALPHAVANTAGE_API_KEY")
    params = {
        "function":"TIME_SERIES_DAILY_ADJUSTED",
        "symbol":ticker,
        "outputsize":outputsize,
        "apikey":key
    }
    r = requests.get(ALPHA_BASE, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    ts = js.get("Time Series (Daily)", {})
    if not ts:
        raise RuntimeError(f"Alpha Vantage returned no data for {ticker}")
    s = pd.Series({pd.Timestamp(k): float(v["5. adjusted close"]) for k,v in ts.items()})
    s.index.name = "Date"
    s = s.sort_index()
    s.name = ticker
    # respect rate limit
    time.sleep(12.5)
    return s

def fetch_prices(tickers: list[str], start: str, end: str, source: str="yahoo", interval: str="1d") -> pd.DataFrame:
    tickers = list(dict.fromkeys([t.strip().upper() for t in tickers if t.strip()]))
    if not tickers:
        return pd.DataFrame()
    if source == "alpha":
        frames = []
        for t in tickers:
            try:
                s = _fetch_alpha_prices(t, outputsize="full")
                frames.append(s)
            except Exception as e:
                print(f"Alpha fetch failed for {t}: {e}")
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, axis=1)
        return df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
    # default: Yahoo Finance
    return _fetch_yfinance_prices(tickers, start, end, interval=interval)

def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.sort_index().pct_change().dropna(how="all")
