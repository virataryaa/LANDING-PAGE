"""
Landing Page — Unified Commodity Dashboard
Commodity -> Exposure (Flat | Spread | Arb | Volatility | Risk) view.

This project computes nothing of its own — every panel is built from parquets
already produced by the Rollex, Futures, Arb, Roll Yield, and Options
projects. Because this app deploys to Streamlit Cloud from its OWN git repo
(the Cloud build has no access to the other projects' repos/Database
folders), `Code/ingest.py` copies the specific files this app needs into
this project's own `Database/` — same pattern the VaR project uses for the
same reason. Run ingest.py (or the Automator) after those source projects'
own daily updates, then push, to refresh this dashboard.

Pilot scope: Coffee (KC + LRC) only. Adding a commodity is adding one entry
to COMMODITIES plus its per-exposure source codes — no new plumbing needed.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Landing Page", layout="wide", initial_sidebar_state="collapsed")

# ── Local database (synced in by Code/ingest.py — see its docstring for why
#    this project keeps its own copy instead of reading sibling repos directly) ─
DB = Path(__file__).resolve().parents[1] / "Database"

KC_FACTOR = 22.0462           # ¢/lb -> $/MT, same conversion the Arb project uses
CONF_Z    = 2.3263            # one-tailed 99% VaR z-score, same as the VaR project
LOT_SIZES = {"KC": 375, "LRC": 10}

# ── Style (consistent with VaR Monitor's theme) ────────────────────────────────
NAVY, BLACK, GREEN, RED, GREY = "#0a2463", "#1d1d1f", "#16a34a", "#dc2626", "#9ca3af"
st.markdown("""<style>
  [data-testid="stAppViewContainer"],[data-testid="stMain"],.main{background:#fafafa!important;color:#1d1d1f!important}
  [data-testid="stHeader"]{background:transparent!important}
  .block-container{padding-top:2rem!important;padding-bottom:1.5rem;max-width:1440px}
  hr{border:none!important;border-top:1px solid #e8e8ed!important;margin:.4rem 0!important}
  h1,h2,h3{color:#1d1d1f!important;font-weight:500!important}
</style>""", unsafe_allow_html=True)

_D = dict(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
          font=dict(family="-apple-system,Helvetica Neue,sans-serif", color=BLACK, size=10))

def lbl(text):
    return (f"<div style='background:{NAVY};padding:5px 13px;border-radius:5px;"
            f"margin-bottom:8px'><span style='font-size:.78rem;font-weight:500;"
            f"letter-spacing:.07em;text-transform:uppercase;color:#dde4f0'>{text}</span></div>")

# ── Commodity registry — add a commodity by adding one entry here ─────────────
COMMODITIES = {
    "Coffee": {
        "legs":          {"KC": "Arabica", "LRC": "Robusta"},
        "rollex_codes":  {"KC": "KC", "LRC": "RC"},     # rollex_{code}.parquet
        "futures_codes": {"KC": "kc", "LRC": "rc"},     # {code}_futures.parquet
        "arb_codes":     ("KC", "RC"),                  # front_{code}.parquet
        "ry_codes":      ("KC", "RC"),                  # Roll Yield "Commodity" values
        "options_codes": ("KC", "LRC"),                 # {code}_options_ice.parquet
    },
}

# ── Loaders — thin, cached, read-only ──────────────────────────────────────────
@st.cache_data(ttl=1800)
def load_rollex(code: str) -> pd.DataFrame:
    df = pd.read_parquet(DB / f"rollex_{code}.parquet")[["rollex_px"]].reset_index()
    df.columns = ["Date", "Close"]
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").dropna()

@st.cache_data(ttl=1800)
def load_futures_oi(code_lower: str) -> pd.DataFrame:
    df = pd.read_parquet(DB / f"{code_lower}_futures.parquet", columns=["Date", "open_interest"])
    df["Date"] = pd.to_datetime(df["Date"])
    tot = df.groupby("Date")["open_interest"].sum(min_count=1).reset_index()
    return tot.sort_values("Date")

@st.cache_data(ttl=1800)
def load_front_price(code_lower: str) -> pd.DataFrame:
    """Active front-contract settlement — same method as the VaR project."""
    raw = pd.read_parquet(DB / f"{code_lower}_futures.parquet",
                           columns=["Date", "FND", "settlement", "ice_symbol"])
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw["FND"]  = pd.to_datetime(raw["FND"])
    raw = raw.dropna(subset=["settlement"])
    active = (raw[raw["FND"] >= raw["Date"]]
              .sort_values(["Date", "FND"])
              .groupby("Date")[["settlement", "ice_symbol"]]
              .first())
    return active

@st.cache_data(ttl=1800)
def load_arb_front(code: str) -> pd.DataFrame:
    return pd.read_parquet(DB / f"front_{code}.parquet")

@st.cache_data(ttl=1800)
def load_roll_yield() -> pd.DataFrame:
    df = pd.read_parquet(DB / "roll_yield_data.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_data(ttl=1800)
def load_options(code: str) -> pd.DataFrame:
    df = pd.read_parquet(DB / f"{code}_options_ice.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='font-family:\"Playfair Display\",Georgia,serif;color:#0a2463;"
    "font-weight:400;letter-spacing:-.01em;margin-bottom:2px'>Landing Page</h2>",
    unsafe_allow_html=True,
)
st.caption("One page per commodity, one tab per exposure — every chart reads live off "
           "the same parquets the standalone dashboards use.")
st.markdown("<hr>", unsafe_allow_html=True)

commodity = st.selectbox("Commodity", list(COMMODITIES.keys()))
cfg = COMMODITIES[commodity]
legs = list(cfg["legs"].keys())          # e.g. ["KC", "LRC"]

tab_flat, tab_spread, tab_arb, tab_vol, tab_risk = st.tabs(
    ["Flat", "Spread", "Arb", "Volatility", "Risk"]
)

# ══════════════════════════════════════════════════════════════════════════════
# FLAT — continuous price + total open interest, per leg
# ══════════════════════════════════════════════════════════════════════════════
with tab_flat:
    st.markdown(lbl(f"{commodity} — Continuous Price & Open Interest"), unsafe_allow_html=True)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                         vertical_spacing=0.05)
    colors = {legs[0]: NAVY, legs[1]: "#8b1a00"} if len(legs) > 1 else {legs[0]: NAVY}

    for leg in legs:
        rx = load_rollex(cfg["rollex_codes"][leg])
        fig.add_trace(go.Scatter(x=rx["Date"], y=rx["Close"], name=f"{leg} ({cfg['legs'][leg]})",
                                  line=dict(color=colors[leg], width=1.6)), row=1, col=1)
        oi = load_futures_oi(cfg["futures_codes"][leg])
        fig.add_trace(go.Bar(x=oi["Date"], y=oi["open_interest"], name=f"{leg} Total OI",
                              marker_color=colors[leg], opacity=0.5), row=2, col=1)

    fig.update_layout(height=520, barmode="overlay",
                       legend=dict(orientation="h", y=1.05, font=dict(size=9)),
                       margin=dict(t=10, b=10, l=4, r=4), **_D)
    fig.update_yaxes(title_text="Rollex Price", row=1, col=1, gridcolor="#f0f0f0")
    fig.update_yaxes(title_text="Total OI", row=2, col=1, gridcolor="#f0f0f0")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SPREAD — roll yield curve + 1yr roll yield history, per leg
# ══════════════════════════════════════════════════════════════════════════════
with tab_spread:
    st.markdown(lbl(f"{commodity} — Roll Yield"), unsafe_allow_html=True)
    ry = load_roll_yield()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**1-Year Roll Yield — History**")
        fig_ry = go.Figure()
        for code in cfg["ry_codes"]:
            s = ry[ry["Commodity"] == code].sort_values("Date")
            if s.empty:
                continue
            fig_ry.add_trace(go.Scatter(x=s["Date"], y=s["Roll_Yield_1yr"], name=code,
                                         line=dict(width=1.6)))
        fig_ry.update_layout(height=360, yaxis=dict(title="Roll Yield (1yr, %)", gridcolor="#f0f0f0"),
                             margin=dict(t=10, b=10, l=4, r=4), **_D)
        st.plotly_chart(fig_ry, use_container_width=True)

    with c2:
        st.markdown("**Current Forward Curve (c1 → c8)**")
        fig_curve = go.Figure()
        curve_cols = [f"c{i}" for i in range(1, 9)]
        for code in cfg["ry_codes"]:
            s = ry[ry["Commodity"] == code].sort_values("Date")
            if s.empty:
                continue
            last = s.iloc[-1]
            fig_curve.add_trace(go.Scatter(x=curve_cols, y=[last[c] for c in curve_cols],
                                            name=code, mode="lines+markers"))
        fig_curve.update_layout(height=360, yaxis=dict(title="Price", gridcolor="#f0f0f0"),
                                margin=dict(t=10, b=10, l=4, r=4), **_D)
        st.plotly_chart(fig_curve, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# ARB — cross-leg spread, same calc as the Arb project
# ══════════════════════════════════════════════════════════════════════════════
with tab_arb:
    if commodity == "Coffee":
        st.markdown(lbl("Arabica Premium over Robusta ($/MT)"), unsafe_allow_html=True)
        kc = load_arb_front("KC")["px1"]
        rc = load_arb_front("RC")["px1"]
        spread = ((kc * KC_FACTOR) - rc).dropna()
        window = st.slider("Rolling window (days)", 60, 504, 252, step=21, key="arb_win")
        mu, sig = spread.rolling(window).mean(), spread.rolling(window).std()

        fig_arb = go.Figure()
        fig_arb.add_trace(go.Scatter(x=spread.index, y=mu + 2*sig, name="+2σ",
                                     line=dict(color=RED, width=1, dash="dot")))
        fig_arb.add_trace(go.Scatter(x=spread.index, y=mu - 2*sig, name="-2σ",
                                     line=dict(color=GREEN, width=1, dash="dot")))
        fig_arb.add_trace(go.Scatter(x=spread.index, y=mu, name="Rolling Mean",
                                     line=dict(color=GREY, width=1)))
        fig_arb.add_trace(go.Scatter(x=spread.index, y=spread, name="Spread",
                                     line=dict(color=NAVY, width=1.8)))
        fig_arb.update_layout(height=440, yaxis=dict(title="$/MT", gridcolor="#f0f0f0"),
                              legend=dict(orientation="h", y=1.05, font=dict(size=9)),
                              margin=dict(t=10, b=10, l=4, r=4), **_D)
        st.plotly_chart(fig_arb, use_container_width=True)
        st.caption("Full pair/unit/z-score controls live in the standalone Arb dashboard — "
                   "this is the headline KC/RC read.")
    else:
        st.info("No Arb pair mapped for this commodity yet.")

# ══════════════════════════════════════════════════════════════════════════════
# VOLATILITY — simplified ATM-ish IV vs realized vol, per leg
# ══════════════════════════════════════════════════════════════════════════════
with tab_vol:
    st.markdown(lbl(f"{commodity} — Implied vs Realized Vol (simplified)"), unsafe_allow_html=True)
    st.caption("Simplified snapshot: near-the-money average IV of the nearest listed expiry vs. "
               "20-day realized vol of the continuous price. For full per-expiry term structure "
               "and Call/Put breakdown, use the standalone Options dashboard.")

    lookback = st.slider("Lookback (days)", 30, 365, 180, step=15, key="vol_lookback")
    fig_vol = go.Figure()
    for leg in legs:
        opt_code = {"KC": "KC", "LRC": "LRC"}.get(leg, leg)
        try:
            odf = load_options(opt_code)
        except FileNotFoundError:
            continue
        max_d = odf["date"].max()
        near_key = odf.loc[odf["date"] == max_d].sort_values(["expiry_year", "expiry_month"])
        if near_key.empty:
            continue
        fy, fm = near_key.iloc[0][["expiry_year", "expiry_month"]]
        front = odf[(odf["expiry_year"] == fy) & (odf["expiry_month"] == fm)].copy()
        front = front[front["date"] >= max_d - pd.Timedelta(days=lookback)]
        # Near-the-money band: within 5% of that day's median strike traded.
        daily_atm = front.dropna(subset=["strike"]).groupby("date")["strike"].median().rename("mid_strike")
        front = front.join(daily_atm, on="date")
        near = front[(front["strike"] >= front["mid_strike"] * 0.95) &
                     (front["strike"] <= front["mid_strike"] * 1.05)]
        iv_series = near.dropna(subset=["impvol"]).groupby("date")["impvol"].mean()

        rx = load_rollex(cfg["rollex_codes"][leg])
        rx = rx.set_index("Date").sort_index()
        log_ret = np.log(rx["Close"] / rx["Close"].shift(1))
        rv_series = (log_ret.rolling(20).std() * np.sqrt(252) * 100).tail(lookback)

        fig_vol.add_trace(go.Scatter(x=iv_series.index, y=iv_series.values, name=f"{leg} IV (ATM-ish)",
                                     line=dict(width=1.8)))
        fig_vol.add_trace(go.Scatter(x=rv_series.index, y=rv_series.values, name=f"{leg} RV (20d)",
                                     line=dict(width=1.4, dash="dot")))

    fig_vol.update_layout(height=440, yaxis=dict(title="Annualized Vol (%)", gridcolor="#f0f0f0"),
                          legend=dict(orientation="h", y=1.05, font=dict(size=9)),
                          margin=dict(t=10, b=10, l=4, r=4), **_D)
    st.plotly_chart(fig_vol, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# RISK — per-lot parametric VaR per leg + combined, same method as the VaR project
# ══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    st.markdown(lbl(f"{commodity} — 1-Day VaR · 99% Confidence · Per Lot"), unsafe_allow_html=True)
    window_label = st.radio("VaR Window", ["20D", "60D", "120D"], index=1, horizontal=True, key="risk_win")
    w = {"20D": 20, "60D": 60, "120D": 120}[window_label]

    var_series = {}
    fig_risk = go.Figure()
    for leg in legs:
        if leg not in LOT_SIZES:
            continue
        rx = load_rollex(cfg["rollex_codes"][leg]).set_index("Date").sort_index()
        full_idx = pd.bdate_range(rx.index.min(), rx.index.max())
        rx = rx.reindex(full_idx).ffill()
        log_ret = np.log(rx["Close"] / rx["Close"].shift(1))
        vol = log_ret.rolling(w).std()

        front = load_front_price(cfg["futures_codes"][leg])
        settle = front["settlement"].reindex(full_idx).ffill()

        var_s = settle * LOT_SIZES[leg] * vol * CONF_Z
        var_series[leg] = var_s
        fig_risk.add_trace(go.Scatter(x=var_s.index, y=var_s.round(0), name=f"{leg} VaR",
                                      line=dict(width=1.6)))

    if len(var_series) > 1:
        combined = pd.concat(var_series.values(), axis=1).ffill().sum(axis=1, min_count=len(var_series))
        fig_risk.add_trace(go.Scatter(x=combined.index, y=combined.round(0),
                                      name=f"{commodity} Combined (1 lot each)",
                                      line=dict(color=NAVY, width=2.4)))

    fig_risk.update_layout(height=440, yaxis=dict(title="VaR (USD / lot)", gridcolor="#f0f0f0"),
                           legend=dict(orientation="h", y=1.05, font=dict(size=9)),
                           margin=dict(t=10, b=10, l=4, r=4), **_D)
    st.plotly_chart(fig_risk, use_container_width=True)
    st.caption("Same parametric method as the standalone VaR Monitor: settlement × lot size × "
               "rolling-vol × 2.3263 (one-tailed 99%). Combined line assumes 1 lot per leg — "
               "use the standalone VaR Monitor's Monte Carlo tab for real position sizing.")
