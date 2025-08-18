from __future__ import annotations

import io
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from scipy.stats import norm

from data import fetch_prices, to_returns
from risk import (
    cvar_es,
    clean_portfolio,
    historical_var,
    parametric_var,
    portfolio_weights,
)
from utils import contrib_to_risk, percent
import warnings, logging, sys, contextlib

warnings.filterwarnings("ignore", category=FutureWarning, message=".*fill_method='pad'.*")

# Quiet yfinance logger
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

@contextlib.contextmanager
def _mute_stdout_stderr():
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err


try:
    from streamlit_autorefresh import st_autorefresh  
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

# Optional covariance shrinkage
try:
    from sklearn.covariance import LedoitWolf  
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


load_dotenv()
st.set_page_config(page_title="Real-Time Portfolio Risk Dashboard", layout="wide")

# CACHING HELPERS

@st.cache_data(ttl=86400, show_spinner=True)
def load_ticker_universe() -> list[str]:
    """Fetch a large equity ticker universe (NASDAQ + other listed)."""
    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]
    headers = {"User-Agent": "Mozilla/5.0 (portfolio-risk-dashboard)"}
    symbols = set()
    errors = []

    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), sep="|", engine="python")
            # Drop footer row like "File Creation Time: ..."
            first_col = df.columns[0]
            df = df[~df[first_col].astype(str).str.contains("File Creation Time", na=False)]
            cols = {c.lower(): c for c in df.columns}
            sym_col = cols.get("nasdaq symbol") or cols.get("act symbol") or cols.get("symbol") or first_col
            if "test issue" in cols:
                df = df[df[cols["test issue"]].astype(str).str.upper().str.startswith("N")]
            s = (
                df[sym_col]
                .astype(str)
                .str.strip()
                .str.upper()
                .replace("", np.nan)
                .dropna()
            )
            s = s[~s.str.contains(r"\s", regex=True)]
            symbols.update(s.tolist())
        except Exception as e:
            errors.append(f"{url}: {e}")

    symbols = sorted(symbols)
    if not symbols:
        st.warning(
            "Could not fetch NASDAQ symbol list. Using fallback list. "
            "Click 'Refresh symbols' to retry.\n\nErrors:\n- " + "\n- ".join(errors)
        )
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "JNJ", "TLT", "GLD", "SPY"]
    return symbols


@st.cache_data(ttl=900, show_spinner=False)
def load_prices_cached(tickers: list[str], start: date, end: date, source: str, interval: str) -> pd.DataFrame:
    # Silence yfinance’s “Failed downloads …” noise during fetch
    with _mute_stdout_stderr():
        return fetch_prices(tickers, str(start), str(end), source=source, interval=interval)


@st.cache_data(ttl=900, show_spinner=False)
def returns_cached(prices: pd.DataFrame) -> pd.DataFrame:
    return to_returns(prices)


# HELPER FUNCTIONS (Advisor / What-If)

def infer_asset_class(tkr: str) -> str:
    """Best-effort quick mapping for new tickers in What-If."""
    t = (tkr or "").upper().strip()
    if t in {"TLT", "IEF", "LQD", "HYG", "BND", "AGG", "SHY", "IEI", "ZROZ"}:
        return "Bond"
    if t in {"GLD", "SLV", "USO", "DBC"}:
        return "Commodity"
    if t in {"SPY", "QQQ", "DIA", "IWM", "VTI"} or t.startswith("^"):
        return "Equity"  # treat broad ETFs as equity risk for display
    if t.startswith("XL"):
        return "Equity"
    if t.endswith("-USD"):
        return "Crypto"
    if t.endswith("=X"):
        return "FX"
    return "Equity"


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    rs = (
        up.ewm(alpha=1 / window, adjust=False).mean()
        / dn.ewm(alpha=1 / window, adjust=False).mean().replace(0, np.nan)
    )
    return 100 - (100 / (1 + rs))


def trend_features(px: pd.Series) -> dict:
    s = px.dropna()
    if len(s) < 60:
        return {"trend_score": np.nan, "label": "n/a", "rsi": np.nan}
    ma50 = s.rolling(50).mean()
    ma200 = s.rolling(200).mean()
    score = 0.0
    score += 1.0 if s.iloc[-1] > ma50.iloc[-1] else -1.0
    score += 1.0 if s.iloc[-1] > ma200.iloc[-1] else -1.0
    r = float(rsi(s).iloc[-1])
    if 55 <= r <= 70:
        score += 0.5
    if r > 70:
        score -= 0.2
    if r < 45:
        score -= 0.5
    label = "Uptrend" if score >= 1.5 else ("Range" if score >= 0 else "Downtrend")
    return {"trend_score": score, "label": label, "rsi": r}


def classify_regime(bench_rets_daily: pd.Series) -> tuple[str, dict]:
    s = bench_rets_daily.dropna()
    if len(s) < 60:
        return "Unknown", {}
    cum = (1 + s).cumprod()
    dd = cum / cum.cummax() - 1.0
    mdd = float(dd.min())
    vol20_ann = float(s.rolling(20).std().iloc[-1] * np.sqrt(252))
    if mdd <= -0.20 or vol20_ann >= 0.35:
        reg = "Crisis"
    elif mdd <= -0.10 or vol20_ann >= 0.25:
        reg = "High-Vol"
    elif vol20_ann <= 0.15 and mdd > -0.05:
        reg = "Calm/Uptrend"
    else:
        reg = "Normal"
    return reg, {"ann_vol20": vol20_ann, "mdd": mdd}


def rank_pct(s: pd.Series) -> pd.Series:
    """Percentile rank in [0,1] with NaNs kept as NaN."""
    return s.rank(pct=True, method="average")


# Keyless Yahoo helpers (spark + screener)
Y_SPARK = "https://query1.finance.yahoo.com/v7/finance/spark"
Y_HEADERS = {"User-Agent": "Mozilla/5.0 (portfolio-risk-dashboard)"}

def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

@st.cache_data(ttl=1800, show_spinner=True)
def yahoo_top_performers(symbols: list[str], period: str = "1mo", interval: str = "1d", top_n: int = 50) -> pd.DataFrame:
    """
    Returns DataFrame columns: symbol, ret
    Computes simple return over the period using spark closes.
    """
    results = []
    for batch in chunked(symbols, 150):  # keep URL length sane
        params = {"symbols": ",".join(batch), "range": period, "interval": interval}
        r = requests.get(Y_SPARK, params=params, headers=Y_HEADERS, timeout=20)
        r.raise_for_status()
        js = r.json()
        arr = js.get("spark", {}).get("result", []) or js.get("result", [])
        for res in arr:
            sym = res.get("symbol")
            resp = (res.get("response") or [{}])[0]
            closes = ((resp.get("indicators") or {}).get("quote") or [{}])[0].get("close", [])
            # Find first/last non-NaN
            if not closes:
                continue
            c = pd.Series(closes).dropna()
            if len(c) < 2:
                continue
            ret = float(c.iloc[-1] / c.iloc[0] - 1.0)
            results.append({"symbol": sym, "ret": ret})
    df = pd.DataFrame(results)
    if df.empty:
        return df
    df = df.groupby("symbol", as_index=False)["ret"].mean()
    df = df.sort_values("ret", ascending=False).head(top_n)
    return df

@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_symbols() -> list[str]:
    """Keyless scrape of S&P 500 tickers from Wikipedia with fallback."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        # Find table with "Symbol" column
        for t in tables:
            if "Symbol" in t.columns:
                syms = t["Symbol"].astype(str).str.upper().str.strip().tolist()
                # Remove footnote dots etc.
                syms = [s.replace(".", "-") for s in syms]  # BRK.B -> BRK-B (Yahoo style)
                return syms
    except Exception:
        pass
    # Fallback subset
    return ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","LLY","JPM","WMT","XOM","V","MA","UNH",
            "HD","ORCL","COST","BAC","KO","PEP","CSCO","MRK","ABBV","NVO","ADBE","CRM","NFLX","INTC","AMD"]


@st.cache_data(ttl=900, show_spinner=False)
def advisor_scan(universe: list[str],
                 aligned_w: pd.Series,
                 start: date, end: date,
                 source: str,
                 regime: str) -> pd.DataFrame:
    """
    Regime-aware scoring using percentile ranks:
      score = wm*mom_pr + wt*trend_pr + wd*divers_pr + wv*lowvol_pr
    where:
      mom_pr = 6m momentum percentile,
      trend_pr = normalized trend_score in [0,1],
      divers_pr = 1 - |corr to portfolio| (clipped to [0,1]),
      lowvol_pr = 1 - vol_ann percentile (prefers lower vol).
    """
    if aligned_w.empty or len(universe) == 0:
        return pd.DataFrame()

    uni_px = load_prices_cached(universe, start, end, source, "1d")
    if uni_px.empty:
        return pd.DataFrame()
    uni_rets = returns_cached(uni_px)

    # daily portfolio for correlation baseline
    port_px = load_prices_cached(list(aligned_w.index), start, end, source, "1d")
    port_rets = returns_cached(port_px)
    pr_daily = (port_rets[aligned_w.index].fillna(0) @ aligned_w).dropna()

    rows = []
    for t in uni_px.columns:
        px = uni_px[t].dropna()
        r = uni_rets[t].dropna()
        feats = trend_features(px)
        mom_6m = (1 + r.tail(126)).prod() - 1 if len(r) >= 126 else np.nan
        vol_ann = r.std() * np.sqrt(252) if len(r) > 2 else np.nan
        corr = r.tail(min(126, len(r))).corr(pr_daily.tail(min(126, len(pr_daily)))) if not pr_daily.empty else np.nan
        rows.append({
            "ticker": t,
            "mom_6m": mom_6m,
            "vol_ann": vol_ann,
            "corr": corr,
            "trend_score": feats["trend_score"],
            "trend": feats["label"]
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Percentile/normalized features
    mom_pr = rank_pct(df["mom_6m"].replace([np.inf, -np.inf], np.nan))
    vol_pr = rank_pct(df["vol_ann"].replace([np.inf, -np.inf], np.nan))
    lowvol_pr = 1 - vol_pr
    trend_pr = ((df["trend_score"] + 3.0) / 6.0).clip(0, 1)  # map ~[-3,3] → [0,1]
    divers_pr = (1 - df["corr"].abs()).clip(lower=0, upper=1)

    # Regime-dependent weights
    if regime in ("Crisis", "High-Vol"):
        wm, wt, wd, wv = 0.25, 0.25, 0.30, 0.20
    elif regime == "Calm/Uptrend":
        wm, wt, wd, wv = 0.45, 0.35, 0.10, 0.10
    else:  # Normal / Unknown
        wm, wt, wd, wv = 0.35, 0.30, 0.20, 0.15

    score = wm*mom_pr.fillna(mom_pr.median()) \
          + wt*trend_pr.fillna(trend_pr.median()) \
          + wd*divers_pr.fillna(divers_pr.median()) \
          + wv*lowvol_pr.fillna(lowvol_pr.median())

    # Buckets (reason tags)
    bucket = np.where(df["corr"] < -0.10, "Hedge",
               np.where(df["corr"].abs() < 0.20, "Diversifier",
               np.where((df["trend_score"] >= 1.5) & (df["mom_6m"] > 0), "Momentum", "Neutral")))

    out = df.assign(
        score=score,
        bucket=bucket,
        mom_pr=mom_pr, lowvol_pr=lowvol_pr, trend_pr=trend_pr, divers_pr=divers_pr
    ).sort_values("score", ascending=False)
    return out


def aggregate_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Combine duplicate tickers by summing quantities (treat as buying more)."""
    if "asset_class" not in df.columns:
        df["asset_class"] = df["ticker"].apply(infer_asset_class)
    agg = (
        df.assign(ticker=df["ticker"].str.upper().str.strip())
        .groupby("ticker", as_index=False)
        .agg(quantity=("quantity", "sum"), asset_class=("asset_class", "first"))
    )
    return agg


def simulate_target_weight(
    base_asset_rets: pd.DataFrame,
    base_weights: pd.Series,
    add_ticker: str,
    target_weight: float,
    start: date,
    end: date,
    source: str,
    interval: str,
    alpha: float,
    horizon: int,
    periods_per_year: int,
    rf_periodic: float,
    bench_rets: pd.Series | None,
):
    """
    Add `add_ticker` to the portfolio at `target_weight` (of final portfolio)
    and return (var_p, es_h, ann_ret, ann_vol, sharpe, beta).
    """
    px_add = load_prices_cached([add_ticker], start, end, source, interval)
    rets_add = returns_cached(px_add)
    if px_add.empty or add_ticker not in px_add.columns:
        return None

    w_old = base_weights.copy()
    w_old = w_old.reindex(base_asset_rets.columns).fillna(0.0)
    w_new = (1 - target_weight) * w_old
    w_new = pd.concat([w_new, pd.Series({add_ticker: target_weight})], axis=0)

    ar = base_asset_rets.copy()
    if add_ticker not in ar.columns:
        ar = pd.concat([ar, rets_add], axis=1)
    ar = ar[w_new.index]  # align order

    pr = (ar @ w_new.fillna(0)).dropna()
    mu, sd = pr.mean(), pr.std()
    ann_ret = (1 + mu) ** periods_per_year - 1
    ann_vol = sd * np.sqrt(periods_per_year)
    sharpe = ((mu - rf_periodic) / sd) * np.sqrt(periods_per_year) if sd > 0 else np.nan

    var_p = parametric_var(ar, w_new.fillna(0), alpha=alpha, periods=horizon)
    es_h = cvar_es(ar, w_new.fillna(0), alpha=alpha, periods=horizon)

    beta = np.nan
    if bench_rets is not None and not bench_rets.empty:
        ab = pd.concat([pr, bench_rets], axis=1).dropna()
        if ab.shape[0] >= 3 and ab.iloc[:, 1].var() > 0:
            beta = ab.iloc[:, 0].cov(ab.iloc[:, 1]) / ab.iloc[:, 1].var()

    return {"var_p": var_p, "es_h": es_h, "ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "beta": beta}


# Prefilter (existing) for non-top-performer path 
@st.cache_data(ttl=1800, show_spinner=True)
def prefilter_universe(base_universe: list[str],
                       prelist: list[str],
                       start: date,
                       end: date,
                       source: str,
                       method: str,
                       seed: int,
                       prefilter_breadth: int) -> list[str]:
    """
    Returns an ordered universe = prelist + top_of_rest according to method:
      - "Top by Liquidity (proxy)": rank by last_price * ann_vol (from ~start..end)
      - "Random sample": seeded permutation
    Only evaluates up to `prefilter_breadth` from the *remaining* universe (after prelist),
    shuffled with `seed` to remove alphabetic bias.
    """
    remaining = [x for x in base_universe if x not in prelist]
    rng = np.random.RandomState(seed)
    remaining = list(rng.permutation(remaining)) 

    if method == "Random sample":
        return prelist + remaining[:prefilter_breadth]

    # method == "Top by Liquidity (proxy)"
    cand = remaining[:prefilter_breadth]
    try:
        px = load_prices_cached(cand, start, end, source, "1d")
        if px.empty:
            return prelist + cand  # fallback
        rets = returns_cached(px)
        last_px = px.ffill().iloc[-1]
        ann_vol = rets.std() * np.sqrt(252)
        score = (last_px * ann_vol).replace([np.inf, -np.inf], np.nan)
        top = score.dropna().sort_values(ascending=False).index.tolist()
        missing = [t for t in cand if t not in top]
        ordered = prelist + top + missing
        return ordered
    except Exception:
        return prelist + cand

# SIDEBAR UI
st.sidebar.title("📊 Portfolio Settings")
default_start = date.today() - timedelta(days=365 * 2)
default_end = date.today()

data_source = st.sidebar.selectbox(
    "Data Source", ["yahoo", "alpha"],
    help="Yahoo Finance requires no API key. Alpha Vantage is optional.",
)
refresh_seconds = st.sidebar.slider("Auto-refresh (seconds)", 0, 300, 60, help="0 disables auto-refresh")

with st.sidebar.expander("Portfolio Input", True):
    input_method = st.radio("How do you want to enter positions?",
                            ["Manual entry", "Upload CSV"], horizontal=True)

    ticker_universe = load_ticker_universe()
    st.caption("Add rows, pick tickers from the big dropdown, set quantities.")
    if st.button("🔄 Refresh symbols", use_container_width=True,
                 help="Clear cache and refetch the latest symbol list"):
        load_ticker_universe.clear()
        st.rerun()

    if input_method == "Upload CSV":
        upload = st.file_uploader("Upload portfolio CSV (columns: ticker, quantity)", type=["csv"])
        if upload is None:
            st.caption("Using included sample `portfolio.csv`")
            dfp = pd.read_csv("portfolio.csv")
            dfp = dfp[["ticker", "quantity"]]
        else:
            dfp = pd.read_csv(upload)[["ticker", "quantity"]]
        st.dataframe(dfp, use_container_width=True)
    else:
        if "dfp_manual" not in st.session_state:
            st.session_state.dfp_manual = pd.DataFrame(
                {"ticker": ["AAPL", "MSFT", "TSLA"], "quantity": [10, 8, 4]}
            )
        dfp_edit = st.data_editor(
            st.session_state.dfp_manual,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "ticker": st.column_config.SelectboxColumn(
                    "Ticker", help="Choose a symbol or type to search",
                    options=ticker_universe, required=True),
                "quantity": st.column_config.NumberColumn(
                    "Quantity", min_value=0.0, step=1.0, format="%.4f"),
            },
            hide_index=True,
        )
        dfp_edit = dfp_edit.replace("", np.nan).dropna(subset=["ticker"]).copy()
        st.session_state.dfp_manual = dfp_edit
        dfp = dfp_edit

bench = st.sidebar.text_input("Benchmark Ticker", value="^GSPC", help="Used for beta & comparisons")
start = st.sidebar.date_input("Start Date", value=default_start)
end = st.sidebar.date_input("End Date", value=default_end)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"])

rf_annual = st.sidebar.number_input("Risk-free rate (annual, %)", value=4.0, step=0.25)
PER_YEAR = {"1d": 252, "1wk": 52, "1mo": 12}
periods_per_year = PER_YEAR.get(interval, 252)
rf_periodic = (1 + rf_annual / 100.0) ** (1 / periods_per_year) - 1

alpha = st.sidebar.slider("Confidence level (VaR/CVaR)", 0.80, 0.995, 0.95)
horizon = st.sidebar.selectbox("VaR/ES Horizon (periods)", [1, 5, 10, 20], index=0)

use_shrink = st.sidebar.checkbox(
    "Use shrinkage covariance (Ledoit–Wolf)", value=False,
    help="Stabilizes covariances in small samples / high correlation (if scikit-learn installed)."
)

with st.sidebar.expander("Stress Scenarios", True):
    shock_pct = st.slider("Uniform Price Shock (%)", -50, 50, -10)
    vol_spike = st.slider("Volatility Multiplier (x)", 1.0, 5.0, 2.0)
    corr_break = st.slider("Correlation Breakdown (%)", 0, 100, 30)

if refresh_seconds > 0:
    if HAS_AUTOREFRESH:
        st_autorefresh(interval=refresh_seconds * 1000, key="data_refresh")
    else:
        st.sidebar.info("Auto-refresh requires `streamlit-autorefresh`.\n\n"
                        "Either set Auto-refresh to 0 or install it:\n"
                        "`pip install streamlit-autorefresh`")
        if st.sidebar.button("Manual Refresh"):
            st.rerun()

#                  DATA FETCHING / PREP

# Auto-detect asset class (no manual column anymore)
dfp["ticker"] = dfp["ticker"].str.upper().str.strip()
dfp["asset_class"] = dfp["ticker"].apply(infer_asset_class)
dfp = clean_portfolio(dfp)

# collapse duplicate tickers -> sums quantities (prevents reindex errors)
dfp = (
    dfp.assign(ticker=dfp["ticker"].str.upper().str.strip())
       .groupby("ticker", as_index=False)
       .agg(quantity=("quantity", "sum"),
            asset_class=("asset_class", "first"))
)

tickers = list(dfp["ticker"].unique())
if bench and bench.strip():
    tickers = list(dict.fromkeys(tickers + [bench]))

prices = load_prices_cached(tickers, start, end, data_source, interval)
st.title("Real-Time Portfolio Risk Dashboard")

if prices.empty:
    st.error("No price data returned. Check tickers, dates, or data source/API key.")
    st.stop()

rets = returns_cached(prices)
asset_cols = [t for t in dfp["ticker"].unique() if t in rets.columns]
asset_rets = rets[asset_cols].dropna(axis=1, thresh=3).copy()
asset_rets = asset_rets.loc[:, ~asset_rets.columns.duplicated()]  # safety
bench_rets = rets[bench].copy() if bench and bench in rets.columns else None

latest_prices = prices.ffill().iloc[-1]
weights = portfolio_weights(latest_prices, dfp)
if weights.index.has_duplicates:
    weights = weights.groupby(level=0).sum()
aligned_w = weights.reindex(asset_rets.columns).fillna(0)

portfolio_value = float((weights * latest_prices.reindex(weights.index)).sum())
if portfolio_value <= 0:
    st.error("Total portfolio value is zero. Please check quantities and prices.")
    st.stop()

# portfolio returns (selected frequency)
pr = (asset_rets @ aligned_w).dropna()

# covariance
if use_shrink and HAS_SKLEARN and not asset_rets.dropna().empty:
    lw = LedoitWolf().fit(asset_rets.dropna().values)
    cov_base = pd.DataFrame(lw.covariance_, index=asset_rets.columns, columns=asset_rets.columns)
else:
    if use_shrink and not HAS_SKLEARN:
        st.sidebar.warning("scikit-learn not installed; using sample covariance instead.")
    cov_base = asset_rets.cov()

port_vol = float(np.sqrt(max(0.0, aligned_w.T @ cov_base.values @ aligned_w)))

# TABS

tab_metrics, tab_visuals, tab_stress, tab_formulas, tab_advisor, tab_whatif, tab_raw = st.tabs(
    ["📐 Metrics", "📈 Visuals", "🚨 Stress Tests", "📚 Formulas", "🧠 Advisor", "🧪 What-If", "🧾 Data"]
)

# Header metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Holdings (value-weighted)", f"{portfolio_value:,.2f}")
with c2:
    st.metric("Positions", f"{(dfp['quantity'] > 0).sum()}")
with c3:
    st.metric("Non-zero weights", f"{(weights > 0).sum()}")
with c4:
    st.metric("Benchmark", bench if bench else "None")


# METRICS 
with tab_metrics:
    st.subheader("Key Risk Metrics")

    cA, cB, cC = st.columns(3)
    with cA:
        var_p = parametric_var(asset_rets, aligned_w, alpha=alpha, periods=horizon)
        st.metric(f"Parametric VaR ({int(alpha*100)}%, {horizon}×{interval})", percent(var_p))
    with cB:
        var_h = historical_var(asset_rets, aligned_w, alpha=alpha, periods=horizon)
        st.metric(f"Historical VaR ({int(alpha*100)}%, {horizon}×{interval})", percent(var_h))
    with cC:
        es_h = cvar_es(asset_rets, aligned_w, alpha=alpha, periods=horizon)
        st.metric(f"CVaR / ES ({int(alpha*100)}%, {horizon}×{interval})", percent(es_h))

    st.divider()
    st.subheader("Performance & Beta")
    mu, sd = pr.mean(), pr.std()
    ann_return = (1 + mu) ** periods_per_year - 1
    ann_vol = sd * np.sqrt(periods_per_year)
    sharpe = ((mu - rf_periodic) / sd) * np.sqrt(periods_per_year) if sd > 0 else np.nan

    if len(pr) > 1:
        total_return = float((1 + pr).prod() - 1)
        years = (pr.index[-1] - pr.index[0]).days / 365.25
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
    else:
        cagr = np.nan

    c1m, c2m = st.columns(2)
    with c1m:
        st.metric("Annualized Return", percent(ann_return))
        st.metric("Annualized Volatility", percent(ann_vol))
        st.metric("Sharpe Ratio", f"{sharpe:.2f}" if not np.isnan(sharpe) else "n/a")
        st.metric("CAGR (selected period)", f"{percent(cagr)}" if not np.isnan(cagr) else "n/a")
    with c2m:
        if bench_rets is not None and not bench_rets.empty and not pr.empty:
            aligned_beta = pd.concat([pr, bench_rets], axis=1).dropna()
            if aligned_beta.shape[0] >= 3 and aligned_beta.iloc[:, 1].var() > 0:
                port, bench_series = aligned_beta.iloc[:, 0], aligned_beta.iloc[:, 1]
                beta = port.cov(bench_series) / bench_series.var()
            else:
                beta = np.nan
            st.metric("Portfolio Beta vs Benchmark", f"{beta:.2f}" if not np.isnan(beta) else "n/a")
        else:
            st.write("Benchmark data unavailable.")

    metrics_df = pd.DataFrame(
        [{"metric": "Parametric VaR", "value": var_p},
         {"metric": "Historical VaR", "value": var_h},
         {"metric": "CVaR / ES", "value": es_h},
         {"metric": "Annualized Return", "value": ann_return},
         {"metric": "Annualized Volatility", "value": ann_vol},
         {"metric": "Sharpe Ratio", "value": sharpe},
         {"metric": "CAGR (selected period)", "value": cagr}]
    )
    st.download_button("Download metrics (CSV)", metrics_df.to_csv(index=False).encode(), "metrics.csv", "text/csv")

    with st.expander("Methodology (concise)"):
        st.markdown(
            f"""
- **Returns:** simple returns at `{interval}` frequency (adjusted closes).  
- **Weights:** value weights from latest prices × quantities.  
- **Parametric VaR:** −(μ·h + z·σ·√h). **Historical VaR:** empirical h-period quantile. **ES/CVaR:** tail mean.  
- **Sharpe:** ((μₚ − rf) / σₚ) × √periods_per_year; rf converted to per-period.  
- **Beta:** Cov(port, bench) / Var(bench) on aligned returns.  
- **CAGR:** time-accurate using first/last timestamps.  
            """
        )


# VISUALS
with tab_visuals:
    st.subheader("Key Visuals")

    # Allocation by asset class
    classes = dfp.set_index("ticker")["asset_class"].reindex(weights.index).fillna("Unknown")
    class_w = weights.groupby(classes).sum().sort_values(ascending=True)
    st.caption("Allocation by Asset Class")
    fig1, ax1 = plt.subplots(figsize=(6, 3.2))
    if not class_w.empty:
        ax1.barh(class_w.index, class_w.values)
        for i, v in enumerate(class_w.values):
            ax1.text(v, i, f"  {v:.2%}", va="center")
        ax1.set_title("Weights by Asset Class"); ax1.set_xlabel("Weight"); ax1.set_ylabel("")
        st.pyplot(fig1, use_container_width=True)
    else:
        st.write("No class weights available.")

    # Contribution to risk (% of total vol)
    wv = aligned_w
    ctr_raw = contrib_to_risk(wv, cov_base)
    port_vol_now = float(np.sqrt(max(0.0, aligned_w.T @ cov_base.values @ aligned_w)))
    ctr_pct = ctr_raw / (port_vol_now if port_vol_now > 0 else 1.0)
    cr = ctr_pct.sort_values(ascending=True)
    st.caption("Contribution to Risk (as % of portfolio volatility)")
    fig2, ax2 = plt.subplots(figsize=(6, 3.2))
    if not cr.empty:
        ax2.barh(cr.index, cr.values)
        for i, v in enumerate(cr.values):
            ax2.text(v, i, f"  {v:.2%}", va="center")
        ax2.set_title("Contribution to Total Volatility"); ax2.set_xlabel("Contribution (%)"); ax2.set_ylabel("")
        st.pyplot(fig2, use_container_width=True)
    else:
        st.write("No contribution-to-risk available.")

    # Rolling volatility (annualized)
    window = min(max(10, periods_per_year), len(pr)) if len(pr) > 0 else 0
    st.caption(f"Rolling Volatility (window ≈ {window}×{interval})")
    fig3, ax3 = plt.subplots(figsize=(7, 3.2))
    if window >= 5:
        roll_sd = pr.rolling(window).std()
        ax3.plot(roll_sd.index, roll_sd.values * np.sqrt(periods_per_year))
        ax3.set_title("Rolling Annualized Volatility"); ax3.set_xlabel("Date"); ax3.set_ylabel("Volatility (annualized)")
        st.pyplot(fig3, use_container_width=True)
    else:
        st.write("Not enough data for rolling volatility.")

    # Rolling parametric VaR (per period)
    fig4, ax4 = plt.subplots(figsize=(7, 3.2))
    if window >= 5:
        z_roll = norm.ppf(1 - alpha)
        roll_sd = pr.rolling(window).std()
        roll_var = -(z_roll * roll_sd)
        ax4.plot(roll_var.index, roll_var.values * 100.0)
        ax4.set_title(f"Rolling VaR ({int(alpha*100)}%, per {interval})"); ax4.set_xlabel("Date"); ax4.set_ylabel("VaR (%)")
        st.pyplot(fig4, use_container_width=True)
    else:
        st.write("Not enough data for rolling VaR.")

    # Drawdown curve
    st.caption("Drawdown (peak-to-trough)")
    fig5, ax5 = plt.subplots(figsize=(7, 3.2))
    if not pr.empty:
        cum = (1 + pr).cumprod(); peak = cum.cummax(); dd = cum / peak - 1.0
        ax5.plot(dd.index, dd.values)
        ax5.set_title("Portfolio Drawdown"); ax5.set_xlabel("Date"); ax5.set_ylabel("Drawdown")
        st.pyplot(fig5, use_container_width=True)
    else:
        st.write("Not enough data to compute drawdowns.")

    # Risk–Return scatter
    st.caption("Risk–Return (Annualized)")
    fig_sc, ax_sc = plt.subplots(figsize=(6.5, 4))
    if not asset_rets.empty:
        mu_i = asset_rets.mean(); sd_i = asset_rets.std()
        ann_ret_i = (1 + mu_i) ** periods_per_year - 1
        ann_vol_i = sd_i * np.sqrt(periods_per_year)
        ax_sc.scatter(ann_vol_i.values, ann_ret_i.values)
        for t, x, y in zip(ann_ret_i.index, ann_vol_i.values, ann_ret_i.values):
            ax_sc.annotate(t, (x, y), xytext=(3, 3), textcoords="offset points")
        if not pr.empty:
            mu_p, sd_p = pr.mean(), pr.std()
            ax_sc.scatter([sd_p * np.sqrt(periods_per_year)], [(1 + mu_p) ** periods_per_year - 1], marker="X", s=80)
            ax_sc.annotate("PORT", (sd_p * np.sqrt(periods_per_year), (1 + mu_p) ** periods_per_year - 1),
                           xytext=(3, 3), textcoords="offset points")
        ax_sc.set_title("Risk–Return (per asset, annualized)"); ax_sc.set_xlabel("Volatility"); ax_sc.set_ylabel("Return")
        st.pyplot(fig_sc, use_container_width=True)
    else:
        st.write("No per-asset returns available.")


# STRESS TESTS 
with tab_stress:
    st.subheader("Stress Testing")
    st.write("Apply uniform price shocks (scenario), volatility spikes, and correlation breakdown (distribution).")

    shocked_prices = prices.copy()
    shocked_prices.iloc[-1] = shocked_prices.iloc[-1] * (1 + shock_pct / 100.0)
    _ = returns_cached(shocked_prices)

    cov_stressed = cov_base * (vol_spike ** 2)
    corr = asset_rets.corr()
    corr_bd = corr.copy()
    off_diag = corr_bd.values - np.eye(len(corr_bd))
    corr_bd.values[:] = off_diag * (1 - corr_break / 100.0) + np.eye(len(corr_bd))

    stds = np.sqrt(np.diag(cov_stressed.values))
    cov_new = pd.DataFrame(np.outer(stds, stds) * corr_bd.values, index=cov_base.index, columns=cov_base.columns)

    z = norm.ppf(1 - alpha)
    port_sigma_new = float(np.sqrt(max(0.0, aligned_w.T @ cov_new.values @ aligned_w)))
    VaR_stressed = -(z * port_sigma_new)
    ShockLoss = -(shock_pct / 100.0)
    ShockAdjVaR = ShockLoss + VaR_stressed

    s1, s2, s3 = st.columns(3)
    with s1: st.metric(f"Stressed Parametric VaR ({int(alpha*100)}%, 1×{interval})", f"{VaR_stressed*100:.2f}%")
    with s2: st.metric("Instantaneous Shock Loss", f"{ShockLoss*100:.2f}%")
    with s3: st.metric(f"Shock-Adjusted VaR ({int(alpha*100)}%, 1×{interval})", f"{ShockAdjVaR*100:.2f}%")

    # Before vs After CTR%
    st.caption("Before vs After Risk Contribution (volatility-only)")
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(10, 3))

    pv_before = float(np.sqrt(max(0.0, aligned_w.T @ cov_base.values @ aligned_w)))
    ctr_before = contrib_to_risk(aligned_w, cov_base) / (pv_before if pv_before > 0 else 1.0)
    datB = ctr_before.to_frame(name="CTR%").T
    ax3a.imshow(datB.values, aspect="auto"); ax3a.set_title("Before")
    ax3a.set_xticks(np.arange(len(datB.columns))); ax3a.set_xticklabels(datB.columns, rotation=45, ha="right")
    ax3a.set_yticks([0]); ax3a.set_yticklabels(["CTR%"])
    for j, _ in enumerate(datB.columns): ax3a.text(j, 0, f"{datB.iloc[0, j]:.2%}", ha="center", va="center")

    pv_after = float(np.sqrt(max(0.0, aligned_w.T @ cov_new.values @ aligned_w)))
    ctr_after = contrib_to_risk(aligned_w, cov_new) / (pv_after if pv_after > 0 else 1.0)
    datA = ctr_after.to_frame(name="CTR%").T
    ax3b.imshow(datA.values, aspect="auto"); ax3b.set_title("After")
    ax3b.set_xticks(np.arange(len(datA.columns))); ax3b.set_xticklabels(datA.columns, rotation=45, ha="right")
    ax3b.set_yticks([0]); ax3b.set_yticklabels(["CTR%"])
    for j, _ in enumerate(datA.columns): ax3b.text(j, 0, f"{datA.iloc[0, j]:.2%}", ha="center", va="center")

    st.pyplot(fig3, use_container_width=True)

# FORMULAS & DEFINITIONS 
with tab_formulas:
    st.subheader("Formulas & Definitions")

    st.markdown("**Returns & Aggregates**")
    st.latex(r"r_t=\frac{P_t}{P_{t-1}}-1,\quad r^{(p)}_t=\sum_i w_i\,r_{i,t},\quad P\in\{252,52,12\}")

    st.markdown("**Performance**")
    st.latex(r"\text{Ann. return}=(1+\bar r)^P-1")
    st.latex(r"\text{Ann. vol}=\sigma\sqrt{P}")
    st.latex(r"\text{Sharpe}=\left(\frac{\bar r-r_f^{(\text{period})}}{\sigma}\right)\sqrt{P}")
    st.latex(r"\beta=\frac{\mathrm{Cov}(r_p,r_b)}{\mathrm{Var}(r_b)}")
    st.latex(r"\text{CAGR}=\left(\prod_t(1+r_t)\right)^{1/\text{years}}-1")

    st.markdown("**Tail Risk**")
    st.latex(r"\text{Parametric VaR}(\alpha,h)=-(\mu h+z_\alpha\sigma\sqrt{h})")
    st.latex(r"\text{Historical VaR: } \text{VaR}^{\text{hist}}_\alpha(h)=-Q_\alpha\!\Big(\sum_{k=1}^{h} r^{(p)}_{t-k+1}\Big)")
    st.latex(r"\text{ES/CVaR: } \text{ES}_\alpha(h)=\mathbb{E}\!\left[L\,|\,L>\text{VaR}_\alpha(h)\right]")

    st.markdown("**Weights & Attribution**")
    st.latex(r"w_i=\frac{\text{value}_i}{\sum_j \text{value}_j},\quad \sum_i w_i=1")
    st.latex(r"\sigma_p^2=w^\top\Sigma w,\qquad RC_i=\frac{w_i(\Sigma w)_i}{\sigma_p},\quad \text{CTR\%}_i=\frac{RC_i}{\sum_j RC_j}")

    st.markdown("**Rolling Volatility & Drawdown**")
    st.latex(r"\hat{\sigma}_{t,\text{win}}=\text{stdev}(r^{(p)}_{t-\text{win}+1:\,t}),\quad \text{ann. vol}=\hat{\sigma}_{t,\text{win}}\sqrt{P}")
    st.latex(r"\text{DD}_t=\frac{V_t}{\max_{s\le t} V_s}-1,\qquad \text{MDD}=\min_t \text{DD}_t")


# ADVISOR (Top performers via Yahoo spark) 
with tab_advisor:
    st.subheader("Advisor: regime-aware suggestions & reasons")

    # Regime from benchmark daily
    bench_daily = load_prices_cached([bench], start, end, data_source, "1d") if bench else pd.DataFrame()
    bench_daily_rets = returns_cached(bench_daily)
    bsr = bench_daily_rets.iloc[:, 0] if not bench_daily_rets.empty else bench_rets
    regime, rinfo = classify_regime(bsr) if bsr is not None else ("Unknown", {})
    cA, cB = st.columns(2)
    with cA:
        st.metric("Market regime", regime)
    with cB:
        if rinfo:
            st.caption(f"20d ann.vol ≈ {rinfo['ann_vol20']*100:.1f}% | Max DD ≈ {rinfo['mdd']*100:.1f}%")

    # Universe controls
    base_universe = load_ticker_universe()
    spx_universe = get_sp500_symbols()  # S&P 500 for top-perf path

    st.markdown("**Universe & selection**")
    colU1, colU2, colU3 = st.columns([2, 1, 1])
    with colU1:
        extra_txt = st.text_input("Extra tickers (comma-separated, optional)", value="")
    with colU2:
        include_holdings = st.checkbox("Include my current holdings", value=True)
    with colU3:
        use_top_perf = st.checkbox("Use Yahoo Top Performers (1W/1M)", value=True)

    seed = st.number_input("Random seed (for fallback random sampling)", min_value=0, max_value=10**9, value=42, step=1)
    prefilter_breadth = st.slider("Prefilter breadth (fallback, evaluated count)", 200, 1200, 800, 100)

    # Parse extras & prelist
    extras = [e.strip().upper() for e in extra_txt.split(",") if e.strip()]
    prelist = []
    if include_holdings:
        prelist += list(aligned_w.index)
    prelist += extras
    prelist = [p for p in prelist if p]
    prelist = list(dict.fromkeys(prelist))  # unique keep order

    # Build ordered universe:
    # If top-performers enabled -> get S&P500, compute 1W(=5d) & 1M leaders via spark, blend.
    # else -> fallback to previous prefilter over NASDAQ Trader combined universe.
    ordered_universe = []
    just_top_table = None
    try:
        if use_top_perf:
            # 1W & 1M leaders (keyless spark)
            spx = [s for s in spx_universe if s]  # clean
            tp_1w = yahoo_top_performers(spx, period="5d", interval="1d", top_n=50)
            tp_1m = yahoo_top_performers(spx, period="1mo", interval="1d", top_n=50)

            if not tp_1w.empty or not tp_1m.empty:
                # Blend by average rank
                d1 = tp_1w.rename(columns={"ret": "ret_1w"}).set_index("symbol")
                d2 = tp_1m.rename(columns={"ret": "ret_1m"}).set_index("symbol")
                blend = d1.join(d2, how="outer")
                # Rank high-to-low returns -> lower rank number is better; convert to percentile
                if "ret_1w" in blend:
                    blend["r1w_rank"] = (-blend["ret_1w"]).rank(method="average")
                else:
                    blend["r1w_rank"] = np.nan
                if "ret_1m" in blend:
                    blend["r1m_rank"] = (-blend["ret_1m"]).rank(method="average")
                else:
                    blend["r1m_rank"] = np.nan
                # Avg rank (ignore NaNs)
                blend["avg_rank"] = blend[["r1w_rank", "r1m_rank"]].mean(axis=1)
                blend = blend.sort_values("avg_rank")
                ordered = blend.index.tolist()

                # Ensure prelist first
                ordered_universe = prelist + [s for s in ordered if s not in prelist]

                # Keep a compact table to show users (top 25)
                just_top_table = blend.head(25).copy()
                just_top_table = just_top_table[["ret_1w","ret_1m"]].fillna(np.nan)
            else:
                ordered_universe = []
        else:
            ordered_universe = []
    except Exception:
        ordered_universe = []

    # Fallback if we couldn't build top-perf universe (network issues etc.)
    if not ordered_universe:
        method = "Top by Liquidity (proxy)"
        ordered_universe = prefilter_universe(base_universe, prelist, start, end, data_source,
                                              method, seed, prefilter_breadth)

    scan_n = st.slider("Scan universe size", 50, 250, 150, 10)
    rec_df = advisor_scan(ordered_universe[:scan_n], aligned_w, start, end, data_source, regime)

    if just_top_table is not None and not just_top_table.empty:
        st.markdown("**Yahoo Top Performers snapshot (blended 1W & 1M ranks)**")
        st.dataframe(
            just_top_table.assign(
                ret_1w=lambda x: x["ret_1w"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "n/a"),
                ret_1m=lambda x: x["ret_1m"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "n/a"),
            ),
            use_container_width=True,
        )

    if rec_df.empty:
        st.info("No candidates found (check date range).")
    else:
        # Split views
        mom = rec_df[rec_df["bucket"] == "Momentum"].head(10)
        div = rec_df[rec_df["bucket"] == "Diversifier"].head(10)
        hed = rec_df[rec_df["bucket"] == "Hedge"].head(6)

        st.markdown("**Momentum candidates**")
        st.dataframe(
            mom[["ticker","score","mom_6m","corr","vol_ann","trend","trend_score"]]
              .style.format({"score":"{:.2f}","mom_6m":"{:.1%}","corr":"{:.2f}","vol_ann":"{:.1%}"}),
            use_container_width=True,
        )

        st.markdown("**Diversifiers (low |corr|)**")
        st.dataframe(
            div[["ticker","score","corr","vol_ann","mom_6m","trend"]]
              .style.format({"score":"{:.2f}","corr":"{:.2f}","vol_ann":"{:.1%}","mom_6m":"{:.1%}"}),
            use_container_width=True,
        )

        st.markdown("**Hedges (negatively correlated)**")
        st.dataframe(
            hed[["ticker","score","corr","vol_ann","trend"]]
              .style.format({"score":"{:.2f}","corr":"{:.2f}","vol_ann":"{:.1%}"}),
            use_container_width=True,
        )

        st.success("Top picks now: " + ", ".join(rec_df.head(5)["ticker"].tolist()))

    st.caption(
        "When enabled, **Yahoo Top Performers** pulls 1W (≈5 trading days) and 1M leaders from the S&P 500 using a keyless endpoint, "
        "blends ranks, and then runs regime-aware scoring against your portfolio. If unavailable, the advisor falls back to a NASDAQ universe "
        "prefiltered by a liquidity proxy."
    )


# WHAT-IF 
with tab_whatif:
    st.subheader("What-If: add a position and see impact")

    cW1, cW2, cW3 = st.columns([2, 1, 1])
    with cW1:
        universe_all = load_ticker_universe()
        default_ix = universe_all.index("NVDA") if "NVDA" in universe_all else 0
        new_ticker = st.selectbox("Ticker to test", universe_all, index=default_ix)
    with cW2:
        mode = st.radio("Sizing mode", ["Quantity", "Target weight (%)"], horizontal=True)
    with cW3:
        size_input = st.number_input("Quantity or %", value=5.0, step=1.0)

    if st.button("Simulate"):
        try:
            # Candidate standalone diagnostics (daily)
            px_add_d = load_prices_cached([new_ticker], start, end, data_source, "1d")
            if px_add_d.empty or new_ticker not in px_add_d.columns:
                st.error("Could not fetch data for that ticker.")
            else:
                r_add_d = returns_cached(px_add_d)[new_ticker].dropna()

                # Portfolio daily series
                port_px_d = load_prices_cached(list(aligned_w.index), start, end, data_source, "1d")
                pr_d = (returns_cached(port_px_d)[aligned_w.index].fillna(0) @ aligned_w).dropna()

                mom_6m = (1 + r_add_d.tail(126)).prod() - 1 if len(r_add_d) >= 126 else np.nan
                vol_ann = r_add_d.std() * np.sqrt(252) if len(r_add_d) > 2 else np.nan
                corr_port = r_add_d.tail(min(126, len(r_add_d))).corr(pr_d.tail(min(126, len(pr_d)))) if not pr_d.empty else np.nan
                beta_add = np.nan
                if bench_rets is not None and not bench_rets.empty:
                    ab = pd.concat([r_add_d, bench_rets], axis=1).dropna()
                    if ab.shape[0] >= 3 and ab.iloc[:,1].var() > 0:
                        beta_add = ab.iloc[:,0].cov(ab.iloc[:,1]) / ab.iloc[:,1].var()
                cum = (1 + r_add_d).cumprod(); dd = cum / cum.cummax() - 1.0
                mdd = float(dd.min()) if not dd.empty else np.nan

                cA, cB, cC, cD = st.columns(4)
                with cA: st.metric("6m momentum", percent(mom_6m))
                with cB: st.metric("Ann. vol (daily)", percent(vol_ann))
                with cC: st.metric("Corr to portfolio", f"{corr_port:.2f}" if not np.isnan(corr_port) else "n/a")
                with cD: st.metric("Beta vs benchmark", f"{beta_add:.2f}" if not np.isnan(beta_add) else "n/a")

                # Convert input to target weight
                px_now_int = load_prices_cached([new_ticker], start, end, data_source, interval)
                price_now = float(px_now_int.ffill().iloc[-1][new_ticker])
                if mode == "Quantity":
                    add_val = float(size_input) * price_now
                    base_val = float((weights * latest_prices.reindex(weights.index)).sum())
                    target_weight = add_val / (base_val + add_val)
                else:
                    target_weight = float(size_input)/100.0

                # Base metrics for deltas
                base_var_p = parametric_var(asset_rets, aligned_w, alpha=alpha, periods=horizon)
                base_es = cvar_es(asset_rets, aligned_w, alpha=alpha, periods=horizon)
                base_mu, base_sd = pr.mean(), pr.std()
                base_ann_ret = (1 + base_mu) ** periods_per_year - 1
                base_ann_vol = base_sd * np.sqrt(periods_per_year)
                base_sharpe = ((base_mu - rf_periodic) / base_sd) * np.sqrt(periods_per_year) if base_sd > 0 else np.nan
                base_beta = np.nan
                if bench_rets is not None and not bench_rets.empty:
                    ab0 = pd.concat([pr, bench_rets], axis=1).dropna()
                    if ab0.shape[0] >= 3 and ab0.iloc[:,1].var() > 0:
                        base_beta = ab0.iloc[:,0].cov(ab0.iloc[:,1]) / ab0.iloc[:,1].var()

                # Requested size
                res_req = simulate_target_weight(
                    asset_rets, aligned_w, new_ticker, target_weight,
                    start, end, data_source, interval, alpha, horizon,
                    periods_per_year, rf_periodic, bench_rets
                )

                # Recommend size: largest weight (0.5–20%) that keeps VaR ≤ base; else best-Sharpe ≤20%
                grid = np.linspace(0.005, 0.20, 40)
                best_w, best_r, best_sharpe, best_w_sharpe, best_r_sharpe = None, None, -1e9, None, None
                for tw in grid:
                    r = simulate_target_weight(asset_rets, aligned_w, new_ticker, tw,
                                               start, end, data_source, interval, alpha, horizon,
                                               periods_per_year, rf_periodic, bench_rets)
                    if r is None:
                        continue
                    if r["var_p"] <= base_var_p + 1e-12:
                        best_w, best_r = tw, r  # keep last (largest) that meets constraint
                    if not np.isnan(r["sharpe"]) and r["sharpe"] > best_sharpe:
                        best_sharpe, best_w_sharpe, best_r_sharpe = r["sharpe"], tw, r

                if best_w is None:
                    best_w, best_r, rec_text = best_w_sharpe, best_r_sharpe, "Best Sharpe up to 20% cap"
                else:
                    rec_text = "Max size that does not increase VaR"

                # Show deltas for requested size
                if res_req is not None:
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Δ Parametric VaR", percent(res_req["var_p"] - base_var_p))
                        st.metric("Δ ES (historical)", percent(res_req["es_h"] - base_es))
                    with cols[1]:
                        st.metric("Δ Sharpe", f"{(res_req['sharpe'] - (0 if np.isnan(base_sharpe) else base_sharpe)):+.2f}")
                        st.metric("New Beta", f"{(res_req['beta']):.2f}" if not np.isnan(res_req["beta"]) else "n/a")
                    with cols[2]:
                        st.metric("Δ Ann. Vol", percent(res_req["ann_vol"] - base_ann_vol))
                        st.metric("Δ Ann. Return", percent(res_req["ann_ret"] - base_ann_ret))

                # Narrative recommendation
                if best_r is not None:
                    dv = best_r["var_p"] - base_var_p
                    ds = (0 if np.isnan(base_sharpe) else best_r["sharpe"] - base_sharpe)
                    dvol = best_r["ann_vol"] - base_ann_vol
                    corr_note = ("adds diversification (|corr|<0.25)"
                                 if (not np.isnan(corr_port) and abs(corr_port) < 0.25)
                                 else ("acts as a hedge (corr<−0.10)"
                                       if (not np.isnan(corr_port) and corr_port < -0.10)
                                       else "is highly correlated"))
                    tilt = ("improves Sharpe **and** reduces VaR"
                            if dv < 0 and ds > 0 else
                            "is defensive (lower VaR, lower Sharpe)"
                            if dv < 0 and ds <= 0 else
                            "raises Sharpe with higher VaR — size prudently"
                            if dv >= 0 and ds > 0 else
                            "worsens risk-adjusted profile")
                    vol_clause = "decreases" if dvol < 0 else ("increases" if dvol > 0 else "keeps")
                    mom_clause = ("positive 6m momentum" if not np.isnan(mom_6m) and mom_6m > 0
                                  else "weak/negative 6m momentum")
                    st.success(
                        f"**Recommendation:** target weight ≈ **{best_w*100:.1f}%**  (_{rec_text}_). "
                        f"{new_ticker} {corr_note}; {mom_clause}. "
                        f"At this size, annual vol {vol_clause} vs current, "
                        f"VaR({int(alpha*100)}%,{horizon}×{interval}) changes by {percent(dv)}, "
                        f"Sharpe changes by {(ds):+.2f}. "
                        f"Overall it **{tilt}**."
                    )

                st.caption("Sizing scan: 0.5%–20% in 0.5% steps. Prefers max weight with VaR ≤ base; otherwise best Sharpe ≤20%.")
        except Exception as e:
            st.error(f"Simulation failed: {e}")

# RAW DATA 
with tab_raw:
    st.subheader("Raw Data")
    st.write("Latest prices")
    st.dataframe(prices.tail(), use_container_width=True)
    st.write(f"Returns at selected interval ({interval})")
    st.dataframe(rets.tail(), use_container_width=True)

st.caption("Educational use only. Not investment advice.")
