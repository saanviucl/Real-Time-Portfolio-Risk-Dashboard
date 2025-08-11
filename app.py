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

try:
    from streamlit_autorefresh import st_autorefresh  
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False


try:
    from sklearn.covariance import LedoitWolf  # pip install scikit-learn
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

load_dotenv()
st.set_page_config(page_title="Real-Time Portfolio Risk Dashboard", layout="wide")


#CACHING HELPERS

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

            # Pick best symbol column name available
            cols = {c.lower(): c for c in df.columns}
            sym_col = (
                cols.get("nasdaq symbol")
                or cols.get("act symbol")
                or cols.get("symbol")
                or first_col
            )

            # Filter out test issues if present
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
            # Remove whitespace-containing oddities
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
    return fetch_prices(tickers, str(start), str(end), source=source, interval=interval)

@st.cache_data(ttl=900, show_spinner=False)
def returns_cached(prices: pd.DataFrame) -> pd.DataFrame:
    return to_returns(prices)

# SIDEBAR UI

st.sidebar.title("📊 Portfolio Settings")
default_start = date.today() - timedelta(days=365 * 2)
default_end = date.today()

data_source = st.sidebar.selectbox(
    "Data Source", ["yahoo", "alpha"],
    help="Yahoo Finance requires no API key. Alpha Vantage is optional.",
)
refresh_seconds = st.sidebar.slider("Auto-refresh (seconds)", 0, 300, 60, help="0 disables auto-refresh")

# Input method: Manual vs CSV 
with st.sidebar.expander("Portfolio Input", True):
    input_method = st.radio("How do you want to enter positions?",
                            ["Manual entry", "Upload CSV"], horizontal=True)

    # Load (cached) ticker universe
    ticker_universe = load_ticker_universe()
    st.caption("Add rows, pick tickers from the big dropdown, set quantities & asset class.")
    if st.button("🔄 Refresh symbols", use_container_width=True,
                 help="Clear cache and refetch the latest symbol list"):
        load_ticker_universe.clear()  # clears this function's cache
        st.rerun()

    if input_method == "Upload CSV":
        upload = st.file_uploader("Upload portfolio CSV", type=["csv"])
        if upload is None:
            st.caption("Using included sample `portfolio.csv`")
            dfp = pd.read_csv("portfolio.csv")
        else:
            dfp = pd.read_csv(upload)
        st.dataframe(dfp, use_container_width=True)

    else:
        # Initialize session state for manual entry
        if "dfp_manual" not in st.session_state:
            st.session_state.dfp_manual = pd.DataFrame(
                {
                    "ticker": ["AAPL", "MSFT", "TSLA"],
                    "quantity": [10, 8, 4],
                    "asset_class": ["Equity", "Equity", "Equity"],
                }
            )
        # Editable table
        dfp_edit = st.data_editor(
            st.session_state.dfp_manual,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "ticker": st.column_config.SelectboxColumn(
                    "Ticker",
                    help="Choose a symbol or type to search",
                    options=ticker_universe,
                    required=True,
                ),
                "quantity": st.column_config.NumberColumn(
                    "Quantity", min_value=0.0, step=1.0, format="%.4f"
                ),
                "asset_class": st.column_config.SelectboxColumn(
                    "Asset Class",
                    options=["Equity", "Bond", "Commodity", "Crypto", "FX", "Other"],
                    required=True,
                ),
            },
            hide_index=True,
        )
        # Clean and persist
        dfp_edit = dfp_edit.replace("", np.nan).dropna(subset=["ticker"]).copy()
        st.session_state.dfp_manual = dfp_edit
        dfp = dfp_edit

bench = st.sidebar.text_input("Benchmark Ticker", value="^GSPC", help="Used for beta & comparisons")
start = st.sidebar.date_input("Start Date", value=default_start)
end = st.sidebar.date_input("End Date", value=default_end)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"])

# Interval-aware risk-free
rf_annual = st.sidebar.number_input("Risk-free rate (annual, %)", value=4.0, step=0.25)
PER_YEAR = {"1d": 252, "1wk": 52, "1mo": 12}
periods_per_year = PER_YEAR.get(interval, 252)
rf_periodic = (1 + rf_annual / 100.0) ** (1 / periods_per_year) - 1

alpha = st.sidebar.slider("Confidence level (VaR/CVaR)", 0.80, 0.995, 0.95)
horizon = st.sidebar.selectbox("VaR/ES Horizon (periods)", [1, 5, 10, 20], index=0)

# shrinkage covariance
use_shrink = st.sidebar.checkbox("Use shrinkage covariance (Ledoit–Wolf)", value=False,
                        help="Stabilizes covariances in small samples / high correlation (if scikit-learn installed).")

with st.sidebar.expander("Stress Scenarios", True):
    shock_pct = st.slider("Uniform Price Shock (%)", -50, 50, -10)
    vol_spike = st.slider("Volatility Multiplier (x)", 1.0, 5.0, 2.0)
    corr_break = st.slider("Correlation Breakdown (%)", 0, 100, 30)

# Auto-refresh handling
if refresh_seconds > 0:
    if HAS_AUTOREFRESH:
        st_autorefresh(interval=refresh_seconds * 1000, key="data_refresh")
    else:
        st.sidebar.info(
            "Auto-refresh requires `streamlit-autorefresh`.\n\n"
            "Either set Auto-refresh to 0 or install it:\n"
            "`pip install streamlit-autorefresh`"
        )
        if st.sidebar.button("Manual Refresh"):
            st.rerun()

#      DATA FETCHING

dfp = clean_portfolio(dfp)

# DEDUPE PORTFOLIO: collapse duplicate tickers 
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

# Keep only assets with enough data points to compute stats
asset_cols = [t for t in dfp["ticker"].unique() if t in rets.columns]
asset_rets = rets[asset_cols].dropna(axis=1, thresh=3).copy()

# Ensure no duplicate return columns (safety)
asset_rets = asset_rets.loc[:, ~asset_rets.columns.duplicated()]

bench_rets = rets[bench].copy() if bench and bench in rets.columns else None

#   SNAPSHOT & WEIGHTS

latest_prices = prices.ffill().iloc[-1]
weights = portfolio_weights(latest_prices, dfp)  # value weights

# Merge duplicate weights if any (safety), then align
if weights.index.has_duplicates:
    weights = weights.groupby(level=0).sum()

aligned_w = weights.reindex(asset_rets.columns).fillna(0)

portfolio_value = float((weights * latest_prices.reindex(weights.index)).sum())
if portfolio_value <= 0:
    st.error("Total portfolio value is zero. Please check quantities and prices.")
    st.stop()

#  CORE SERIES & MATRICES

# Portfolio return series at selected frequency
pr = (asset_rets @ aligned_w).dropna()

# Covariance 
if use_shrink and HAS_SKLEARN and not asset_rets.dropna().empty:
    lw = LedoitWolf().fit(asset_rets.dropna().values)
    cov_base = pd.DataFrame(lw.covariance_, index=asset_rets.columns, columns=asset_rets.columns)
else:
    if use_shrink and not HAS_SKLEARN:
        st.sidebar.warning("scikit-learn not installed; using sample covariance instead.")
    cov_base = asset_rets.cov()

# Portfolio volatility (per period)
port_vol = float(np.sqrt(max(0.0, aligned_w.T @ cov_base.values @ aligned_w)))

# LAYOUT TABS

tab_metrics, tab_visuals, tab_stress, tab_raw = st.tabs(
    ["📐 Metrics", "📈 Visuals", "🚨 Stress Tests", "🧾 Data"]
)

#Top metrics header
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Holdings (value-weighted)", f"{portfolio_value:,.2f}")
with col2:
    st.metric("Positions", f"{(dfp['quantity'] > 0).sum()}")
with col3:
    st.metric("Non-zero weights", f"{(weights > 0).sum()}")
with col4:
    st.metric("Benchmark", bench if bench else "None")

#       METRICS TAB

with tab_metrics:
    st.subheader("Key Risk Metrics")

    # VaR / ES (portfolio-level)
    colA, colB, colC = st.columns(3)
    with colA:
        var_p = parametric_var(asset_rets, aligned_w, alpha=alpha, periods=horizon)
        st.metric(f"Parametric VaR ({int(alpha*100)}%, {horizon}×{interval})", f"{percent(var_p)}")
    with colB:
        var_h = historical_var(asset_rets, aligned_w, alpha=alpha, periods=horizon)
        st.metric(f"Historical VaR ({int(alpha*100)}%, {horizon}×{interval})", f"{percent(var_h)}")
    with colC:
        es_h = cvar_es(asset_rets, aligned_w, alpha=alpha, periods=horizon)
        st.metric(f"CVaR / ES ({int(alpha*100)}%, {horizon}×{interval})", f"{percent(es_h)}")

    st.divider()
    st.subheader("Performance & Beta")

    # Return/vol stats (interval-aware)
    mu = pr.mean()
    sd = pr.std()
    ann_return = (1 + mu) ** periods_per_year - 1
    ann_vol = sd * np.sqrt(periods_per_year)

    if sd > 0:
        sharpe = ((mu - rf_periodic) / sd) * np.sqrt(periods_per_year)
    else:
        sharpe = np.nan

    # Time-accurate CAGR using dates (handles gaps/holidays)
    if len(pr) > 1:
        total_return = float((1 + pr).prod() - 1)
        years = (pr.index[-1] - pr.index[0]).days / 365.25
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
    else:
        cagr = np.nan

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Annualized Return", f"{percent(ann_return)}")
        st.metric("Annualized Volatility", f"{percent(ann_vol)}")
        st.metric("Sharpe Ratio", f"{sharpe:.2f}" if not np.isnan(sharpe) else "n/a")
        st.metric("CAGR (selected period)", f"{percent(cagr)}" if not np.isnan(cagr) else "n/a")
    with col2:
        if bench_rets is not None and not bench_rets.empty and not pr.empty:
            try:
                aligned_beta = pd.concat([pr, bench_rets], axis=1).dropna()
                if aligned_beta.shape[0] >= 3 and aligned_beta.iloc[:, 1].var() > 0:
                    port, bench_series = aligned_beta.iloc[:, 0], aligned_beta.iloc[:, 1]
                    beta = port.cov(bench_series) / bench_series.var()
                else:
                    beta = np.nan
                st.metric("Portfolio Beta vs Benchmark", f"{beta:.2f}" if not np.isnan(beta) else "n/a")
            except Exception:
                st.write("Could not compute beta (insufficient data).")
        else:
            st.write("Benchmark data unavailable.")

    # Download metrics (CSV)
    metrics_df = pd.DataFrame([
        {"metric": "Parametric VaR", "value": var_p},
        {"metric": "Historical VaR", "value": var_h},
        {"metric": "CVaR / ES", "value": es_h},
        {"metric": "Annualized Return", "value": ann_return},
        {"metric": "Annualized Volatility", "value": ann_vol},
        {"metric": "Sharpe Ratio", "value": sharpe},
        {"metric": "CAGR (selected period)", "value": cagr},
        {"metric": "Beta vs Benchmark", "value": (beta if bench_rets is not None else np.nan)},
    ])
    csv_bytes = metrics_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download metrics (CSV)", data=csv_bytes, file_name="metrics.csv", mime="text/csv")

    with st.expander("Methodology (concise)"):
        st.markdown(
            f"""
- **Returns:** simple returns at `{interval}` frequency; prices are adjusted closes.
- **Weights:** value weights from latest prices × quantities.
- **Parametric VaR ({int(alpha*100)}%, h):** −(μ·h + z·σ·√h); μ, σ from portfolio return distribution.  
- **Historical VaR ({int(alpha*100)}%, h):** empirical h-period portfolio return quantile.  
- **CVaR / ES:** mean of returns ≤ VaR threshold (tail expectation).  
- **Sharpe:** ((μₚ − rf) / σₚ) × √periods_per_year; rf converted to per-period.  
- **Beta:** Cov(port, bench) / Var(bench) on aligned {interval} returns.  
- **CAGR:** ((∏(1+r))^(1/years) − 1), years from first/last timestamp.  
- **Stress tests:** Stressed VaR uses covariance scaled by vol multiplier and correlation shrink; uniform shock shown separately and additively.
            """
        )

#       VISUALS TAB

with tab_visuals:
    st.subheader("Key Visuals")

    # 1) Allocation by Asset Class (bar) 
    classes = dfp.set_index("ticker")["asset_class"].reindex(weights.index).fillna("Unknown")
    class_weights = weights.groupby(classes).sum().sort_values(ascending=True)

    st.caption("Allocation by Asset Class")
    fig_alloc, ax_alloc = plt.subplots(figsize=(6, 3.2))
    if not class_weights.empty:
        ax_alloc.barh(class_weights.index, class_weights.values)
        for i, v in enumerate(class_weights.values):
            ax_alloc.text(v, i, f"  {v:.2%}", va="center")
        ax_alloc.set_title("Weights by Asset Class")
        ax_alloc.set_xlabel("Weight")
        ax_alloc.set_ylabel("")
        st.pyplot(fig_alloc, use_container_width=True)
    else:
        st.write("No class weights available.")

    # 2) Contribution to Risk (normalized %) 
    wv = aligned_w
    ctr_raw = contrib_to_risk(wv, cov_base)   # contribution in vol units
    port_vol_now = float(np.sqrt(max(0.0, aligned_w.T @ cov_base.values @ aligned_w)))
    ctr_pct = ctr_raw / (port_vol_now if port_vol_now > 0 else 1.0)  # normalize to fraction of total vol
    cr = ctr_pct.sort_values(ascending=True)

    st.caption("Contribution to Risk (as % of portfolio volatility)")
    fig_ctr, ax_ctr = plt.subplots(figsize=(6, 3.2))
    if not cr.empty:
        ax_ctr.barh(cr.index, cr.values)
        for i, v in enumerate(cr.values):
            ax_ctr.text(v, i, f"  {v:.2%}", va="center")
        ax_ctr.set_title("Contribution to Total Volatility")
        ax_ctr.set_xlabel("Contribution (%)")
        ax_ctr.set_ylabel("")
        st.pyplot(fig_ctr, use_container_width=True)
    else:
        st.write("No contribution-to-risk available.")

    # 3) Rolling Volatility (annualized) 
    window = min(max(10, periods_per_year), len(pr)) if len(pr) > 0 else 0
    st.caption(f"Rolling Volatility (window ≈ {window}×{interval})")
    fig_rvol, ax_rvol = plt.subplots(figsize=(7, 3.2))
    if window >= 5:
        roll_sd = pr.rolling(window).std()
        ax_rvol.plot(roll_sd.index, roll_sd.values * np.sqrt(periods_per_year))
        ax_rvol.set_title("Rolling Annualized Volatility")
        ax_rvol.set_xlabel("Date")
        ax_rvol.set_ylabel("Volatility (annualized)")
        st.pyplot(fig_rvol, use_container_width=True)
    else:
        st.write("Not enough data for rolling volatility.")

    #  4) Rolling Parametric VaR (per period) 
    fig_rvar, ax_rvar = plt.subplots(figsize=(7, 3.2))
    if window >= 5:
        z_roll = norm.ppf(1 - alpha)
        roll_sd = pr.rolling(window).std()
        roll_var = -(z_roll * roll_sd)  # per-period VaR
        ax_rvar.plot(roll_var.index, (roll_var.values * 100.0))
        ax_rvar.set_title(f"Rolling VaR ({int(alpha*100)}%, per {interval})")
        ax_rvar.set_xlabel("Date")
        ax_rvar.set_ylabel("VaR (%)")
        st.pyplot(fig_rvar, use_container_width=True)
    else:
        st.write("Not enough data for rolling VaR.")

    # 5) Drawdown Curve 
    st.caption("Drawdown (peak-to-trough)")
    fig_dd, ax_dd = plt.subplots(figsize=(7, 3.2))
    if not pr.empty:
        cum = (1 + pr).cumprod()
        peak = cum.cummax()
        dd = cum / peak - 1.0
        ax_dd.plot(dd.index, dd.values)
        ax_dd.set_title("Portfolio Drawdown")
        ax_dd.set_xlabel("Date")
        ax_dd.set_ylabel("Drawdown")
        st.pyplot(fig_dd, use_container_width=True)
    else:
        st.write("Not enough data to compute drawdowns.")

    #  6) Risk–Return Scatter (assets + portfolio) 
    st.caption("Risk–Return (Annualized)")
    fig_sc, ax_sc = plt.subplots(figsize=(6.5, 4))
    if not asset_rets.empty:
        mu_i = asset_rets.mean()
        sd_i = asset_rets.std()
        ann_ret_i = (1 + mu_i) ** periods_per_year - 1
        ann_vol_i = sd_i * np.sqrt(periods_per_year)

        ax_sc.scatter(ann_vol_i.values, ann_ret_i.values)
        for t, x, y in zip(ann_ret_i.index, ann_vol_i.values, ann_ret_i.values):
            ax_sc.annotate(t, (x, y), xytext=(3, 3), textcoords="offset points")

        # Portfolio point
        if not pr.empty:
            mu_p = pr.mean()
            sd_p = pr.std()
            ann_ret_p = (1 + mu_p) ** periods_per_year - 1
            ann_vol_p = sd_p * np.sqrt(periods_per_year)
            ax_sc.scatter([ann_vol_p], [ann_ret_p], marker="X", s=80)
            ax_sc.annotate("PORT", (ann_vol_p, ann_ret_p), xytext=(3, 3), textcoords="offset points")

        ax_sc.set_title("Risk–Return (per asset, annualized)")
        ax_sc.set_xlabel("Volatility")
        ax_sc.set_ylabel("Return")
        st.pyplot(fig_sc, use_container_width=True)
    else:
        st.write("No per-asset returns available.")

#      STRESS TESTS TAB

with tab_stress:
    st.subheader("Stress Testing")
    st.write(
        "Apply uniform price shocks (scenario), volatility spikes, and correlation breakdown (distribution). "
        "We show three metrics so you can separate the scenario effect from distributional risk."
    )

    # One-off uniform price shock on the last day (reserved for future scenario paths)
    shocked_prices = prices.copy()
    shocked_prices.iloc[-1] = shocked_prices.iloc[-1] * (1 + shock_pct / 100.0)
    _ = returns_cached(shocked_prices)

    # Volatility spike (scale covariance)
    cov_stressed = cov_base * (vol_spike ** 2)

    # Correlation breakdown (shrink off-diagonals)
    corr = asset_rets.corr()
    corr_bd = corr.copy()
    off_diag = corr_bd.values - np.eye(len(corr_bd))
    corr_bd.values[:] = off_diag * (1 - corr_break / 100.0) + np.eye(len(corr_bd))

    # Rebuild covariance with stressed vol and broken correlations
    stds = np.sqrt(np.diag(cov_stressed.values))
    cov_new = np.outer(stds, stds) * corr_bd.values
    cov_new = pd.DataFrame(cov_new, index=cov_base.index, columns=cov_base.columns)

    # Stressed Parametric VaR (distribution-only, 1 period)
    z = norm.ppf(1 - alpha)
    port_sigma_new = float(np.sqrt(max(0.0, aligned_w.T @ cov_new.values @ aligned_w)))
    VaR_stressed = -(z * port_sigma_new)  # positive number (loss as %)

    # Instantaneous Shock Loss (scenario-only)
    mu_shift = shock_pct / 100.0
    ShockLoss = -mu_shift  # e.g., shock -10% -> +10% loss

    # Shock-Adjusted VaR (combined view)
    ShockAdjVaR = ShockLoss + VaR_stressed

    colS1, colS2, colS3 = st.columns(3)
    with colS1:
        st.metric(f"Stressed Parametric VaR ({int(alpha*100)}%, 1×{interval})", f"{VaR_stressed*100:.2f}%")
    with colS2:
        st.metric("Instantaneous Shock Loss", f"{ShockLoss*100:.2f}%")
    with colS3:
        st.metric(f"Shock-Adjusted VaR ({int(alpha*100)}%, 1×{interval})", f"{ShockAdjVaR*100:.2f}%")

    # Contribution to risk: before vs after (volatility-only)
    st.caption("Before vs After Risk Contribution (volatility-only)")
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(10, 3))

    # Before
    port_vol_before = float(np.sqrt(max(0.0, aligned_w.T @ cov_base.values @ aligned_w)))
    ctr_before = contrib_to_risk(aligned_w, cov_base) / (port_vol_before if port_vol_before > 0 else 1.0)
    datB = ctr_before.to_frame(name="CTR%").T
    ax3a.imshow(datB.values, aspect="auto")
    ax3a.set_title("Before")
    ax3a.set_xticks(np.arange(len(datB.columns)))
    ax3a.set_xticklabels(datB.columns, rotation=45, ha="right")
    ax3a.set_yticks([0]); ax3a.set_yticklabels(["CTR%"])
    for j, c in enumerate(datB.columns):
        ax3a.text(j, 0, f"{datB.iloc[0, j]:.2%}", ha="center", va="center")

    # After
    port_vol_after = float(np.sqrt(max(0.0, aligned_w.T @ cov_new.values @ aligned_w)))
    ctr_after = contrib_to_risk(aligned_w, cov_new) / (port_vol_after if port_vol_after > 0 else 1.0)
    datA = ctr_after.to_frame(name="CTR%").T
    ax3b.imshow(datA.values, aspect="auto")
    ax3b.set_title("After")
    ax3b.set_xticks(np.arange(len(datA.columns)))
    ax3b.set_xticklabels(datA.columns, rotation=45, ha="right")
    ax3b.set_yticks([0]); ax3b.set_yticklabels(["CTR%"])
    for j, c in enumerate(datA.columns):
        ax3b.text(j, 0, f"{datA.iloc[0, j]:.2%}", ha="center", va="center")

    st.pyplot(fig3, use_container_width=True)

#  DATA TAB

with tab_raw:
    st.subheader("Raw Data")
    st.write("Latest prices")
    st.dataframe(prices.tail(), use_container_width=True)
    st.write(f"Returns at selected interval ({interval})")
    st.dataframe(rets.tail(), use_container_width=True)

st.caption("Educational use only. Not investment advice.")
