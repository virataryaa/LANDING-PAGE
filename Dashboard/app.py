"""
Landing Page — Unified Commodity Dashboard
Commodity -> Exposure (Flat | Spread | Arb | Volatility | Risk) view.

This project computes nothing of its own — every panel here is a verified
port of a chart that already exists in one of the standalone dashboards
(Rollex, Futures, Arb, Roll Yield, Options, VaR): the source file was read
and the exact calculation copied, not approximated. The one exception is
Volatility's "IV vs RV", which is explicitly labeled in-app as a simplified
stand-in for the Options dashboard's real per-expiry futures-anchored logic
(that mapping is too tightly coupled to port quickly) — every other panel
matches its source app's math.

Because this app deploys to Streamlit Cloud from its OWN git repo (the
Cloud build has no access to the other projects' repos/Database folders),
`Code/ingest.py` copies the specific files this app needs into this
project's own `Database/` — same pattern the VaR project uses for the same
reason. Run ingest.py (or the Automator) after those source projects' own
daily updates, then push, to refresh this dashboard.

Pilot scope: Coffee (KC + LRC) only. Adding a commodity is adding one entry
to COMMODITIES plus its per-exposure source codes — no new plumbing needed.
"""

import numpy as np
import pandas as pd
from scipy import stats
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
ROLLS_YR  = {"KC": 5,   "LRC": 5}   # Roll Yield project's rolls-per-year assumption
MONTHS    = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ── Style (consistent with VaR Monitor's theme) ────────────────────────────────
NAVY, BLACK, GREEN, RED, GREY, AMBER = "#0a2463", "#1d1d1f", "#16a34a", "#dc2626", "#9ca3af", "#d97706"
DRED = "#8b0000"
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

def base_fig(height=380, yaxis_title=None):
    fig = go.Figure()
    fig.update_layout(height=height,
                       yaxis=dict(title=yaxis_title, gridcolor="#f0f0f0"),
                       legend=dict(orientation="h", y=1.05, font=dict(size=9)),
                       margin=dict(t=10, b=10, l=4, r=4), **_D)
    return fig

def year_month_heatmap(df: pd.DataFrame, date_col: str, value_col: str, title: str,
                        colorscale=None, zmid=None, key=None, pct=False):
    d = df.dropna(subset=[value_col]).copy()
    if d.empty:
        st.info("No data to build a heatmap from.")
        return
    d["Year"]  = d[date_col].dt.year
    d["Month"] = d[date_col].dt.month
    pivot = d.groupby(["Year", "Month"])[value_col].mean().reset_index().pivot(
        index="Year", columns="Month", values=value_col)
    pivot.columns = [MONTHS[m - 1] for m in pivot.columns]
    pivot = pivot.sort_index(ascending=False)
    z = pivot.values * (100 if pct else 1)
    suffix = "%" if pct else ""
    text_vals = [[f"{v:,.1f}{suffix}" if not pd.isna(v) else "" for v in row] for row in z]
    hm_kwargs = dict(zmid=zmid) if zmid is not None else {}
    fig_hm = go.Figure(go.Heatmap(
        z=z, x=list(pivot.columns), y=[str(y) for y in pivot.index],
        text=text_vals, texttemplate="%{text}", textfont=dict(size=8, color=BLACK),
        colorscale=colorscale or [[0.0, "#d4edda"], [0.4, "#fff3cd"], [0.7, "#f8d7a0"], [1.0, "#f5c6cb"]],
        showscale=True, colorbar=dict(thickness=10, len=0.8, tickfont=dict(size=8, color=BLACK)),
        hoverongaps=False, hovertemplate="<b>%{y} %{x}</b><br>" + title + ": %{z:,.1f}" + suffix + "<extra></extra>",
        **hm_kwargs,
    ))
    fig_hm.update_layout(height=max(280, len(pivot.index) * 26),
                         xaxis=dict(side="top", tickfont=dict(size=9, color=BLACK), showgrid=False),
                         yaxis=dict(tickfont=dict(size=9, color=BLACK), showgrid=False),
                         margin=dict(t=36, b=10, l=50, r=10), **_D)
    st.plotly_chart(fig_hm, use_container_width=True, key=key)

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
    """rollex_ret is the project's own precomputed return series — used directly
    (not re-derived from price diffs) so Vol/Return-Distribution panels match
    the Rollex dashboard's numbers exactly."""
    df = pd.read_parquet(DB / f"rollex_{code}.parquet")[["rollex_px", "rollex_ret"]].reset_index()
    df.columns = ["Date", "Close", "Ret"]
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").dropna(subset=["Close"])

@st.cache_data(ttl=1800)
def load_futures_full(code_lower: str) -> pd.DataFrame:
    df = pd.read_parquet(DB / f"{code_lower}_futures.parquet",
                          columns=["Date", "ice_symbol", "LTD", "settlement", "volume", "open_interest"])
    df["Date"] = pd.to_datetime(df["Date"])
    df["LTD"]  = pd.to_datetime(df["LTD"])
    return df

@st.cache_data(ttl=1800)
def load_futures_oi(code_lower: str) -> pd.DataFrame:
    df = load_futures_full(code_lower)
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
leg_colors = {legs[0]: NAVY, legs[1]: "#8b1a00"} if len(legs) > 1 else {legs[0]: NAVY}

tab_flat, tab_spread, tab_arb, tab_vol, tab_risk = st.tabs(
    ["Flat", "Spread", "Arb", "Volatility", "Risk"]
)

# ══════════════════════════════════════════════════════════════════════════════
# FLAT — ports of Rollex dashboard (Price&OI, Price&Vol, Indexed, Return Dist)
#        + Futures dashboard's self-contained "All Contracts Rolling Volume"
# ══════════════════════════════════════════════════════════════════════════════
with tab_flat:
    f_price, f_pv, f_idx, f_dist, f_vol = st.tabs(
        ["Price & OI", "Price & Vol", "Indexed Performance", "Return Distribution", "Rolling Volume"]
    )

    with f_price:
        st.markdown(lbl(f"{commodity} — Continuous Price & Open Interest"), unsafe_allow_html=True)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                             vertical_spacing=0.05)
        for leg in legs:
            rx = load_rollex(cfg["rollex_codes"][leg])
            fig.add_trace(go.Scatter(x=rx["Date"], y=rx["Close"], name=f"{leg} ({cfg['legs'][leg]})",
                                      line=dict(color=leg_colors[leg], width=1.6)), row=1, col=1)
            oi = load_futures_oi(cfg["futures_codes"][leg])
            fig.add_trace(go.Bar(x=oi["Date"], y=oi["open_interest"], name=f"{leg} Total OI",
                                  marker_color=leg_colors[leg], opacity=0.5), row=2, col=1)
        fig.update_layout(height=520, barmode="overlay",
                           legend=dict(orientation="h", y=1.05, font=dict(size=9)),
                           margin=dict(t=10, b=10, l=4, r=4), **_D)
        fig.update_yaxes(title_text="Rollex Price", row=1, col=1, gridcolor="#f0f0f0")
        fig.update_yaxes(title_text="Total OI", row=2, col=1, gridcolor="#f0f0f0")
        st.plotly_chart(fig, use_container_width=True)

    with f_pv:
        st.markdown(lbl(f"{commodity} — Rollex Price & Rolling Volatility"), unsafe_allow_html=True)
        st.caption("Port of the Rollex dashboard's Price & Vol tab: 20d/60d realized vol computed "
                   "from the project's own rollex_ret series, annualized ×√252.")
        for leg in legs:
            rx = load_rollex(cfg["rollex_codes"][leg]).set_index("Date").sort_index()
            rx["vol20"] = rx["Ret"].rolling(20).std() * np.sqrt(252) * 100
            rx["vol60"] = rx["Ret"].rolling(60).std() * np.sqrt(252) * 100
            fig_px = make_subplots(specs=[[{"secondary_y": True}]])
            fig_px.add_trace(go.Scatter(x=rx.index, y=rx["Close"], name=f"{leg} Rollex Px", mode="lines",
                                        line=dict(color=leg_colors[leg], width=2),
                                        fill="tozeroy", fillcolor="rgba(10,36,99,0.07)"), secondary_y=False)
            fig_px.add_trace(go.Scatter(x=rx.index, y=rx["vol20"], name="Vol 20d", mode="lines",
                                        line=dict(color=AMBER, width=1.5)), secondary_y=True)
            fig_px.add_trace(go.Scatter(x=rx.index, y=rx["vol60"], name="Vol 60d", mode="lines",
                                        line=dict(color="#888", width=1.2, dash="dot")), secondary_y=True)
            fig_px.update_layout(height=380, title=dict(text=leg, font=dict(size=11)),
                                 legend=dict(orientation="h", y=1.05, font=dict(size=8)),
                                 margin=dict(t=25, b=8, l=4, r=4), **_D)
            fig_px.update_yaxes(title_text="Price", secondary_y=False, gridcolor="#f0f0f0")
            fig_px.update_yaxes(title_text="Ann. Vol %", secondary_y=True, showgrid=False)
            st.plotly_chart(fig_px, use_container_width=True, key=f"pv_{leg}")

    with f_idx:
        st.markdown(lbl(f"{commodity} — Indexed Performance (Base=100)"), unsafe_allow_html=True)
        fig_idx = base_fig(height=440, yaxis_title="Indexed (Base=100)")
        for leg in legs:
            rx = load_rollex(cfg["rollex_codes"][leg]).set_index("Date").sort_index()
            indexed = rx["Close"] / rx["Close"].iloc[0] * 100
            fig_idx.add_trace(go.Scatter(x=indexed.index, y=indexed, name=leg,
                                         line=dict(color=leg_colors[leg], width=1.8)))
        fig_idx.add_hline(y=100, line_color="#cccccc", line_width=1)
        st.plotly_chart(fig_idx, use_container_width=True)

    with f_dist:
        st.markdown(lbl(f"{commodity} — Log Return Distribution & Z-Score"), unsafe_allow_html=True)
        leg_pick = st.selectbox("Leg", legs, key="dist_leg")
        rx = load_rollex(cfg["rollex_codes"][leg_pick]).set_index("Date").sort_index()
        log_rets = np.log1p(rx["Ret"]).dropna() * 100
        mu, sigma = log_rets.mean(), log_rets.std()
        latest_lr = log_rets.iloc[-1]
        z_lr = (latest_lr - mu) / sigma if sigma > 0 else 0
        skew_v  = float(stats.skew(log_rets))
        kurt_v  = float(stats.kurtosis(log_rets))
        _, p_jb = stats.jarque_bera(log_rets)

        z_col = DRED if abs(z_lr) > 2 else AMBER if abs(z_lr) > 1 else GREEN
        m1, m2, m3, m4, m5 = st.columns(5)
        def _stat(col, label, val, color=NAVY):
            col.markdown(f"<div style='background:#f0f2f8;border-radius:8px;padding:8px 14px'>"
                        f"<div style='font-size:.58rem;color:#6e6e73;text-transform:uppercase;"
                        f"letter-spacing:.1em'>{label}</div>"
                        f"<div style='font-size:.95rem;font-weight:700;color:{color}'>{val}</div></div>",
                        unsafe_allow_html=True)
        _stat(m1, "Latest Return", f"{latest_lr:+.2f}%")
        _stat(m2, "Z-Score", f"{z_lr:+.2f}σ", color=z_col)
        _stat(m3, "Skewness", f"{skew_v:+.3f}")
        _stat(m4, "Excess Kurtosis", f"{kurt_v:+.3f}")
        _stat(m5, "Jarque-Bera p", f"{p_jb:.4f}", color=DRED if p_jb < 0.05 else GREEN)
        st.markdown("<div style='margin:10px 0'></div>", unsafe_allow_html=True)

        x_range = np.linspace(log_rets.min(), log_rets.max(), 400)
        bin_width = (log_rets.max() - log_rets.min()) / 80
        y_curve = stats.norm.pdf(x_range, mu, sigma) * len(log_rets) * bin_width
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=log_rets, nbinsx=80, marker_color=leg_colors[leg_pick],
                                        opacity=0.72, name="Log returns"))
        fig_hist.add_trace(go.Scatter(x=x_range, y=y_curve, mode="lines", name="Normal fit",
                                      line=dict(color="#888", width=1.8, dash="dot")))
        sd_colors = {1: "#f0a500", 2: "#e07000", 3: DRED}
        for n_sd, color in sd_colors.items():
            for sign in [1, -1]:
                val = mu + sign * n_sd * sigma
                fig_hist.add_vline(x=val, line_color=color, line_width=1.2, line_dash="dash")
        fig_hist.add_vline(x=latest_lr, line_color=z_col, line_width=2.5,
                           annotation_text=f"Today {latest_lr:+.2f}% (z={z_lr:+.2f}σ)",
                           annotation_font=dict(size=9, color=z_col))
        fig_hist.update_layout(height=420, showlegend=False, margin=dict(t=30, b=8, l=4, r=4),
                               xaxis=dict(title="Log Return (%)", gridcolor="#f0f0f0"),
                               yaxis=dict(title="Frequency", gridcolor="#f0f0f0"), **_D)
        st.plotly_chart(fig_hist, use_container_width=True)

    with f_vol:
        st.markdown(lbl(f"{commodity} — Rolling Volume, All Contracts"), unsafe_allow_html=True)
        st.caption("Port of the Futures dashboard's 'All Contracts — Rolling Volume' section: "
                   "every contract that traded within the lookback window, rolling-averaged.")
        roll_n = st.slider("Rolling window (days)", 1, 30, 10, key="fvol_rolln")
        lookback = st.slider("Lookback (calendar days)", 30, 365, 120, step=10, key="fvol_lookback")
        for leg in legs:
            st.markdown(f"**{leg}**")
            df_all = load_futures_full(cfg["futures_codes"][leg])
            cutoff = df_all["Date"].max() - pd.Timedelta(days=lookback)
            ltd_full = df_all.groupby("ice_symbol")["LTD"].first()
            syms_in_window = df_all.loc[df_all["Date"] >= cutoff, "ice_symbol"].unique()
            relevant_syms = ltd_full[ltd_full.index.isin(syms_in_window)].sort_values().index.tolist()

            pieces = []
            for sym in relevant_syms:
                g = df_all[df_all["ice_symbol"] == sym].sort_values("Date").copy()
                g["_rv"] = g["volume"].rolling(roll_n, min_periods=1).mean()
                pieces.append(g[["Date", "ice_symbol", "_rv"]])
            vol_all = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
            if vol_all.empty:
                st.info("No volume data in this window.")
                continue
            vol_win = vol_all[vol_all["Date"] >= cutoff].copy()
            traded_totals = vol_win.groupby("ice_symbol")["_rv"].sum()
            active_win = [s for s in relevant_syms if traded_totals.get(s, 0) > 0]

            pivot = (vol_win.pivot_table(index="Date", columns="ice_symbol", values="_rv", aggfunc="mean")
                            .reindex(columns=active_win).fillna(0.0))
            totals = pivot.sum(axis=1)
            pct = pivot.div(totals.replace(0, pd.NA), axis=0) * 100

            c1, c2 = st.columns(2)
            with c1:
                fig_pct = go.Figure()
                for sym in active_win:
                    fig_pct.add_trace(go.Bar(x=pct.index, y=pct[sym], name=sym))
                fig_pct.update_layout(barmode="stack", height=380, bargap=0.02,
                                      title=dict(text="Rolling Volume Mix (%)", font=dict(size=11)),
                                      yaxis=dict(title="Share of Rolling Volume", range=[0, 100], ticksuffix="%",
                                                 gridcolor="#f0f0f0"),
                                      legend=dict(orientation="h", y=-0.2, font=dict(size=8)),
                                      margin=dict(t=25, b=50, l=4, r=4), **_D)
                st.plotly_chart(fig_pct, use_container_width=True, key=f"fvol_pct_{leg}")
            with c2:
                fig_abs = go.Figure()
                for sym in active_win:
                    fig_abs.add_trace(go.Bar(x=pivot.index, y=pivot[sym], name=sym))
                fig_abs.update_layout(barmode="stack", height=380, bargap=0.02,
                                      title=dict(text="Total Rolling Volume (stacked)", font=dict(size=11)),
                                      yaxis=dict(title=f"{roll_n}D Avg Volume", gridcolor="#f0f0f0"),
                                      legend=dict(orientation="h", y=-0.2, font=dict(size=8)),
                                      margin=dict(t=25, b=50, l=4, r=4), **_D)
                st.plotly_chart(fig_abs, use_container_width=True, key=f"fvol_abs_{leg}")

# ══════════════════════════════════════════════════════════════════════════════
# SPREAD — verified ports of the Roll Yield dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tab_spread:
    s_yield, s_rank, s_curve, s_heat, s_cost = st.tabs(
        ["Yield & Curve", "Ranking & Percentile", "Forward Curves", "Roll Yield Heatmap", "Roll Cost"]
    )
    ry = load_roll_yield()
    curve_cols = [f"c{i}" for i in range(1, 9)]

    with s_yield:
        st.markdown(lbl(f"{commodity} — Roll Yield"), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**1-Year Roll Yield — History**")
            fig_ry = base_fig(height=360, yaxis_title="Roll Yield (1yr, %)")
            for code in cfg["ry_codes"]:
                s = ry[ry["Commodity"] == code].sort_values("Date")
                if not s.empty:
                    fig_ry.add_trace(go.Scatter(x=s["Date"], y=(s["Roll_Yield_1yr"]*100).round(2),
                                                name=code, line=dict(width=1.6)))
            fig_ry.add_hline(y=0, line_dash="dot", line_color="#aaaaaa")
            st.plotly_chart(fig_ry, use_container_width=True)
        with c2:
            st.markdown("**Current Forward Curve (c1 → c8)**")
            fig_curve = base_fig(height=360, yaxis_title="Price")
            for code in cfg["ry_codes"]:
                s = ry[ry["Commodity"] == code].sort_values("Date")
                if not s.empty:
                    last = s.iloc[-1]
                    fig_curve.add_trace(go.Scatter(x=curve_cols, y=[last[c] for c in curve_cols],
                                                   name=code, mode="lines+markers"))
            st.plotly_chart(fig_curve, use_container_width=True)

    with s_rank:
        st.markdown(lbl(f"{commodity} — Roll Yield Ranking & Percentile"), unsafe_allow_html=True)
        latest_date = ry["Date"].max()
        df_latest = ry[ry["Date"] == latest_date].set_index("Commodity")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Ranking · {latest_date.strftime('%d/%m/%Y')}** (all Roll Yield commodities)")
            rank_rows = []
            for code in ry["Commodity"].unique():
                if code in df_latest.index:
                    v = df_latest.loc[code, "Roll_Yield_1yr"] * 100
                    rank_rows.append({"Commodity": code, "Roll Yield (1yr)": f"{v:+.1f}%", "_ry": v})
            rank_df = pd.DataFrame(rank_rows).sort_values("_ry", ascending=False).reset_index(drop=True)
            rank_df.insert(0, "Rank", rank_df.index + 1)
            fig_rank = go.Figure(go.Table(
                header=dict(values=["Rank", "Commodity", "Roll Yield (1yr)"],
                           fill_color=NAVY, font=dict(color="white", size=10), align="center", height=28),
                cells=dict(values=[rank_df["Rank"], rank_df["Commodity"], rank_df["Roll Yield (1yr)"]],
                          fill_color=[["white" if i % 2 == 0 else "#f5f5f7" for i in range(len(rank_df))]],
                          font=dict(color=[["black"]*len(rank_df), ["black"]*len(rank_df),
                                          [(GREEN if r > 0 else DRED) for r in rank_df["_ry"]]], size=10),
                          align="center", height=24),
            ))
            fig_rank.update_layout(height=340, margin=dict(t=0, b=0, l=0, r=0), **_D)
            st.plotly_chart(fig_rank, use_container_width=True)
        with c2:
            st.markdown("**Percentile vs Full History** (all Roll Yield commodities)")
            pct_rows = []
            for code in ry["Commodity"].unique():
                hist = ry[ry["Commodity"] == code]["Roll_Yield_1yr"].dropna()
                if hist.empty or code not in df_latest.index:
                    continue
                cur = df_latest.loc[code, "Roll_Yield_1yr"]
                pct = float((hist < cur).mean() * 100)
                pct_rows.append({"Commodity": code, "Percentile": round(pct, 1)})
            pct_df = pd.DataFrame(pct_rows).sort_values("Percentile")
            fig_pct = go.Figure(go.Bar(x=pct_df["Percentile"], y=pct_df["Commodity"], orientation="h",
                                       marker_color=[leg_colors.get(c.replace("RC","LRC"), GREY) for c in pct_df["Commodity"]],
                                       text=pct_df["Percentile"].map(lambda x: f"{x:.0f}th"),
                                       textposition="outside", textfont=dict(size=9)))
            fig_pct.add_vline(x=50, line_dash="dot", line_color="#aaaaaa")
            fig_pct.add_vline(x=80, line_dash="dot", line_color=AMBER)
            fig_pct.update_layout(height=340, xaxis=dict(range=[0, 115], ticksuffix="%"),
                                  margin=dict(t=0, b=0, l=4, r=60), **_D)
            st.plotly_chart(fig_pct, use_container_width=True)

    with s_curve:
        st.markdown(lbl(f"{commodity} — Forward Curves"), unsafe_allow_html=True)
        code_pick = st.selectbox("Contract", cfg["ry_codes"], key="curve_code")
        df_comm = ry[ry["Commodity"] == code_pick].sort_values("Date")
        all_dates_sorted = df_comm["Date"].drop_duplicates().sort_values()
        latest_4d = all_dates_sorted.iloc[-4:].tolist() if len(all_dates_sorted) >= 4 else all_dates_sorted.tolist()
        weekly_idx = list(range(-1, -len(all_dates_sorted), -5))[:4]
        latest_4w = [all_dates_sorted.iloc[i] for i in sorted(weekly_idx)]
        day_colors = ["#1d1d1f", DRED, "#82c982", "#aaaaaa"]

        def _curve_fig(dates, colors, title):
            fig = go.Figure()
            for d, col in zip(dates, colors):
                row = df_comm[df_comm["Date"] == d]
                if row.empty:
                    continue
                y = [row.iloc[0][c] for c in curve_cols]
                fig.add_trace(go.Scatter(x=curve_cols, y=y, mode="lines+markers", name=d.strftime("%d/%m/%Y"),
                                         line=dict(color=col, width=2), marker=dict(size=5)))
            fig.update_layout(title=dict(text=title, font=dict(size=11), x=0.5, xanchor="center"),
                              height=340, legend=dict(font=dict(size=8)), margin=dict(t=35, b=10, l=4, r=4), **_D)
            return fig

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            latest_row = df_comm[df_comm["Date"] == all_dates_sorted.iloc[-1]]
            y_latest = [latest_row.iloc[0][c] for c in curve_cols]
            fig_latest = go.Figure(go.Scatter(x=curve_cols, y=y_latest, mode="lines+markers",
                                              line=dict(color=leg_colors.get(code_pick.replace("RC","LRC"), NAVY), width=2.5)))
            fig_latest.update_layout(title=dict(text=f"Latest · {all_dates_sorted.iloc[-1].strftime('%d/%m/%Y')}",
                                                font=dict(size=11), x=0.5, xanchor="center"),
                                     height=340, margin=dict(t=35, b=10, l=4, r=4), **_D)
            st.plotly_chart(fig_latest, use_container_width=True)
        with fc2:
            st.plotly_chart(_curve_fig(latest_4d, day_colors, "Last 4 Days"), use_container_width=True)
        with fc3:
            st.plotly_chart(_curve_fig(latest_4w, day_colors, "Last 4 Weeks"), use_container_width=True)

    with s_heat:
        st.markdown(lbl(f"{commodity} — Roll Yield Heatmap (Monthly Avg)"), unsafe_allow_html=True)
        code_pick_hm = st.selectbox("Contract", cfg["ry_codes"], key="ry_heat_code")
        s = ry[ry["Commodity"] == code_pick_hm]
        year_month_heatmap(s, "Date", "Roll_Yield_1yr", "Avg Roll Yield", pct=True, zmid=0,
                           colorscale=[[0.0,"#8b0000"],[0.4,"#f5c6cb"],[0.5,"#ffffff"],[0.6,"#d4edda"],[1.0,"#1a6b1a"]],
                           key="ry_heatmap")

    with s_cost:
        st.markdown(lbl(f"{commodity} — Roll Cost (c2 − c1)"), unsafe_allow_html=True)
        st.caption("Positive = contango (rolling forward costs you); negative = backwardation (rolling pays you). "
                   "Same definition as the Roll Yield dashboard's Roll Cost tab.")
        df_rc = ry.copy()
        df_rc["roll_spread"] = df_rc["c2"] - df_rc["c1"]
        df_rc["roll_pct"] = (df_rc["roll_spread"] / df_rc["c1"] * 100).round(3)
        latest_rc = df_rc[df_rc["Date"] == df_rc["Date"].max()].set_index("Commodity")

        fig_rc_line = base_fig(height=340, yaxis_title="c2 − c1")
        for code in cfg["ry_codes"]:
            s = df_rc[df_rc["Commodity"] == code].sort_values("Date")
            fig_rc_line.add_trace(go.Scatter(x=s["Date"], y=s["roll_spread"].round(2), name=code, line=dict(width=1.6)))
        fig_rc_line.add_hline(y=0, line_dash="dot", line_color="#aaaaaa")
        st.plotly_chart(fig_rc_line, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Snapshot · {df_rc['Date'].max().strftime('%d/%m/%Y')}**")
            snap_rows = []
            for code in cfg["ry_codes"]:
                if code not in latest_rc.index:
                    continue
                spread = latest_rc.loc[code, "roll_spread"]
                pct    = latest_rc.loc[code, "roll_pct"]
                lot_key = code.replace("RC", "LRC")
                mult, rolls = LOT_SIZES.get(lot_key, 1), ROLLS_YR.get(lot_key, 1)
                dol_lot, ann_lot = spread * mult, spread * mult * rolls
                regime = "Contango" if spread > 0 else "Backwardation"
                snap_rows.append({"Commodity": code, "Spread": f"{spread:+.2f}", "Spread %": f"{pct:+.2f}%",
                                  "$/Lot": f"${dol_lot:+,.0f}", "Ann $/Lot": f"${ann_lot:+,.0f}", "Regime": regime})
            snap_df = pd.DataFrame(snap_rows)
            fig_snap = go.Figure(go.Table(
                header=dict(values=list(snap_df.columns), fill_color=NAVY, font=dict(color="white", size=9),
                           align="center", height=28),
                cells=dict(values=[snap_df[c] for c in snap_df.columns], align="center", height=24,
                          fill_color=[["white" if i % 2 == 0 else "#f5f5f7" for i in range(len(snap_df))]]),
            ))
            fig_snap.update_layout(height=200, margin=dict(t=0, b=0, l=0, r=0), **_D)
            st.plotly_chart(fig_snap, use_container_width=True)
        with c2:
            st.markdown("**Seasonality — Avg by Month**")
            seas_code = st.selectbox("Contract", cfg["ry_codes"], key="rc_seas_code")
            seas = df_rc[df_rc["Commodity"] == seas_code].copy()
            seas["Month"] = seas["Date"].dt.month
            seas_avg = seas.groupby("Month")["roll_spread"].mean().reindex(range(1, 13))
            colors_seas = [DRED if v > 0 else GREEN for v in seas_avg.fillna(0)]
            fig_seas = go.Figure(go.Bar(x=MONTHS, y=seas_avg.values.round(2), marker_color=colors_seas,
                                        text=[f"{v:.2f}" if not np.isnan(v) else "" for v in seas_avg.values],
                                        textposition="outside", textfont=dict(size=8)))
            fig_seas.add_hline(y=0, line_dash="dot", line_color="#aaaaaa")
            fig_seas.update_layout(height=280, yaxis=dict(title="Avg c2−c1", gridcolor="#f0f0f0"),
                                   margin=dict(t=10, b=10, l=4, r=4), **_D)
            st.plotly_chart(fig_seas, use_container_width=True)

        st.markdown("**Roll Spread Heatmap (Monthly Avg)**")
        cost_heat_code = st.selectbox("Contract", cfg["ry_codes"], key="cost_heat_code")
        s_cost_hm = df_rc[df_rc["Commodity"] == cost_heat_code].copy()
        year_month_heatmap(s_cost_hm, "Date", "roll_spread", "Avg Roll Spread",
                           colorscale=[[0.0,"#1a6b1a"],[0.4,"#d4edda"],[0.5,"#ffffff"],[0.6,"#f5c6cb"],[1.0,"#8b0000"]],
                           zmid=0, key="rollcost_heatmap")

# ══════════════════════════════════════════════════════════════════════════════
# ARB — full verified port of the Arb dashboard's KC/RC spread section
# ══════════════════════════════════════════════════════════════════════════════
with tab_arb:
    if commodity == "Coffee":
        a_spread, a_z, a_legs, a_scatter, a_ratio = st.tabs(
            ["Spread", "Z-Score", "Individual Legs", "Return Scatter", "Ratio"]
        )
        kc = load_arb_front("KC")["px1"]
        rc = load_arb_front("RC")["px1"]
        l1 = kc * KC_FACTOR   # KC in $/MT
        l2 = rc               # RC in $/MT
        spread = (l1 - l2).dropna()
        zscore_win = st.slider("Rolling / Z-score window (days)", 60, 504, 252, step=21, key="arb_win")
        mu, sig = spread.rolling(zscore_win).mean(), spread.rolling(zscore_win).std()
        z = (spread - mu) / sig

        with a_spread:
            st.markdown(lbl("Arabica Premium over Robusta ($/MT)"), unsafe_allow_html=True)
            fig_sp = base_fig(height=440, yaxis_title="$/MT")
            fig_sp.add_trace(go.Scatter(x=spread.index, y=mu + 2*sig, name="+2σ", line=dict(color=RED, width=1, dash="dot")))
            fig_sp.add_trace(go.Scatter(x=spread.index, y=mu + sig, name="+1σ", line=dict(color=AMBER, width=1, dash="dash")))
            fig_sp.add_trace(go.Scatter(x=spread.index, y=mu, name="Mean", line=dict(color=GREY, width=1.5)))
            fig_sp.add_trace(go.Scatter(x=spread.index, y=mu - sig, name="-1σ", line=dict(color=AMBER, width=1, dash="dash")))
            fig_sp.add_trace(go.Scatter(x=spread.index, y=mu - 2*sig, name="-2σ", line=dict(color=GREEN, width=1, dash="dot")))
            fig_sp.add_trace(go.Scatter(x=spread.index, y=spread, name="Spread", line=dict(color=NAVY, width=2)))
            st.plotly_chart(fig_sp, use_container_width=True)

        with a_z:
            st.markdown(lbl(f"Spread Z-Score ({zscore_win}d rolling)"), unsafe_allow_html=True)
            fig_z = base_fig(height=380, yaxis_title="Z-Score")
            fig_z.add_trace(go.Scatter(x=z.index, y=z, line=dict(color=NAVY, width=1.5), name="Z-Score"))
            fig_z.add_hline(y=0, line_color=GREY, line_width=1)
            fig_z.update_layout(yaxis=dict(range=[-4, 4], gridcolor="#f0f0f0"))
            st.plotly_chart(fig_z, use_container_width=True)

        with a_legs:
            st.markdown(lbl("Individual Legs ($/MT)"), unsafe_allow_html=True)
            fig_legs = base_fig(height=420, yaxis_title="$/MT")
            fig_legs.add_trace(go.Scatter(x=l1.index, y=l1, name="KC ($/MT)", line=dict(color=NAVY, width=1.5)))
            fig_legs.add_trace(go.Scatter(x=l2.index, y=l2, name="RC ($/MT)", line=dict(color=AMBER, width=1.5)))
            st.plotly_chart(fig_legs, use_container_width=True)

        with a_scatter:
            st.markdown(lbl("Daily Return Scatter"), unsafe_allow_html=True)
            dl1, dl2 = l1.diff().dropna(), l2.diff().dropna()
            scat = pd.concat([dl1.rename("leg1"), dl2.rename("leg2")], axis=1).dropna()
            if len(scat) < 10:
                st.info("Not enough data.")
            else:
                coeffs = np.polyfit(scat["leg1"], scat["leg2"], 1)
                x_line = np.linspace(scat["leg1"].min(), scat["leg1"].max(), 200)
                y_line = coeffs[0] * x_line + coeffs[1]
                r2 = scat["leg1"].corr(scat["leg2"]) ** 2
                cutoff = 60
                old_mask = scat.index < scat.index[-min(cutoff, len(scat))]
                recent, history = scat[~old_mask], scat[old_mask]

                fig_scat = go.Figure()
                fig_scat.add_trace(go.Scatter(x=history["leg1"], y=history["leg2"], mode="markers", name="History",
                                              marker=dict(color=GREY, size=4, opacity=0.45)))
                fig_scat.add_trace(go.Scatter(x=recent["leg1"], y=recent["leg2"], mode="markers",
                                              name=f"Last {min(cutoff, len(scat))}d",
                                              marker=dict(color=NAVY, size=6, opacity=0.85)))
                fig_scat.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Regression",
                                              line=dict(color=RED, width=1.5, dash="dash")))
                callout_n = 5
                latest = scat.iloc[-callout_n:]
                fig_scat.add_trace(go.Scatter(x=latest["leg1"], y=latest["leg2"], mode="markers",
                                              name=f"Last {callout_n} sessions",
                                              marker=dict(color=RED, size=10, symbol="circle-open", line=dict(color=RED, width=2))))
                fig_scat.add_hline(y=0, line_color="#f0f0f0", line_width=1)
                fig_scat.add_vline(x=0, line_color="#f0f0f0", line_width=1)
                fig_scat.update_layout(height=440, title=dict(text=f"R²={r2:.2f}", font=dict(size=11)),
                                       xaxis=dict(title="Δ KC ($/MT)", gridcolor="#f0f0f0"),
                                       yaxis=dict(title="Δ RC ($/MT)", gridcolor="#f0f0f0"),
                                       legend=dict(orientation="h", y=1.08, font=dict(size=8)),
                                       margin=dict(t=30, b=10, l=4, r=4), **_D)
                st.plotly_chart(fig_scat, use_container_width=True)

        with a_ratio:
            st.markdown(lbl("KC/RC Price Ratio (Arabica/Robusta)"), unsafe_allow_html=True)
            st.caption("Roasters blend the two; extreme ratios historically mean-revert as substitution economics kick in.")
            ratio = l1 / l2
            mu_r, sig_r = ratio.rolling(zscore_win).mean(), ratio.rolling(zscore_win).std()
            fig_ratio = base_fig(height=420, yaxis_title="KC/RC")
            fig_ratio.add_trace(go.Scatter(x=ratio.index, y=mu_r + sig_r, name="+1σ", line=dict(color=AMBER, width=1, dash="dash")))
            fig_ratio.add_trace(go.Scatter(x=ratio.index, y=mu_r - sig_r, name="-1σ", line=dict(color=AMBER, width=1, dash="dash")))
            fig_ratio.add_trace(go.Scatter(x=ratio.index, y=mu_r, name="Mean", line=dict(color=GREY, width=1)))
            fig_ratio.add_trace(go.Scatter(x=ratio.index, y=ratio, name="KC/RC Ratio", line=dict(color=NAVY, width=2)))
            st.plotly_chart(fig_ratio, use_container_width=True)
    else:
        st.info("No Arb pair mapped for this commodity yet.")

# ══════════════════════════════════════════════════════════════════════════════
# VOLATILITY — unchanged: simplified stand-in, explicitly labeled as such
# ══════════════════════════════════════════════════════════════════════════════
with tab_vol:
    st.markdown(lbl(f"{commodity} — Implied vs Realized Vol (simplified)"), unsafe_allow_html=True)
    st.caption("Simplified snapshot: near-the-money average IV of the nearest listed expiry vs. "
               "20-day realized vol of the continuous price. This is NOT a port of the Options "
               "dashboard's real per-expiry futures-anchored ATM logic — that mapping is tightly "
               "coupled to that app's own session state. Use the standalone Options dashboard for "
               "the real term structure.")

    lookback = st.slider("Lookback (days)", 30, 365, 180, step=15, key="vol_lookback")
    fig_vol = base_fig(height=440, yaxis_title="Annualized Vol (%)")
    for leg in legs:
        try:
            odf = load_options(leg)
        except FileNotFoundError:
            continue
        max_d = odf["date"].max()
        near_key = odf.loc[odf["date"] == max_d].sort_values(["expiry_year", "expiry_month"])
        if near_key.empty:
            continue
        fy, fm = near_key.iloc[0][["expiry_year", "expiry_month"]]
        front = odf[(odf["expiry_year"] == fy) & (odf["expiry_month"] == fm)].copy()
        front = front[front["date"] >= max_d - pd.Timedelta(days=lookback)]
        daily_atm = front.dropna(subset=["strike"]).groupby("date")["strike"].median().rename("mid_strike")
        front = front.join(daily_atm, on="date")
        near = front[(front["strike"] >= front["mid_strike"] * 0.95) &
                     (front["strike"] <= front["mid_strike"] * 1.05)]
        iv_series = near.dropna(subset=["impvol"]).groupby("date")["impvol"].mean()

        rx = load_rollex(cfg["rollex_codes"][leg]).set_index("Date").sort_index()
        log_ret = np.log(rx["Close"] / rx["Close"].shift(1))
        rv_series = (log_ret.rolling(20).std() * np.sqrt(252) * 100).tail(lookback)

        fig_vol.add_trace(go.Scatter(x=iv_series.index, y=iv_series.values, name=f"{leg} IV (ATM-ish)", line=dict(width=1.8)))
        fig_vol.add_trace(go.Scatter(x=rv_series.index, y=rv_series.values, name=f"{leg} RV (20d)", line=dict(width=1.4, dash="dot")))
    st.plotly_chart(fig_vol, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# RISK — verified ports of the VaR project's Parametric VaR tab
# ══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    r_var, r_pct, r_heat = st.tabs(["Parametric VaR", "Vol Percentile", "VaR Heatmap"])

    window_label = st.radio("VaR Window (applies to this whole tab)", ["20D", "60D", "120D"],
                            index=1, horizontal=True, key="risk_win")
    w = {"20D": 20, "60D": 60, "120D": 120}[window_label]

    def _leg_vol_var(leg):
        rx = load_rollex(cfg["rollex_codes"][leg]).set_index("Date").sort_index()
        full_idx = pd.bdate_range(rx.index.min(), rx.index.max())
        rx = rx.reindex(full_idx).ffill()
        log_ret = np.log(rx["Close"] / rx["Close"].shift(1))
        vol = log_ret.rolling(w).std()
        front = load_front_price(cfg["futures_codes"][leg])
        settle = front["settlement"].reindex(full_idx).ffill()
        var_s = settle * LOT_SIZES.get(leg, 1) * vol * CONF_Z
        return vol, var_s

    with r_var:
        st.markdown(lbl(f"{commodity} — 1-Day VaR · 99% Confidence · Per Lot"), unsafe_allow_html=True)
        var_series = {}
        fig_risk = base_fig(height=440, yaxis_title="VaR (USD / lot)")
        for leg in legs:
            if leg not in LOT_SIZES:
                continue
            _, var_s = _leg_vol_var(leg)
            var_series[leg] = var_s
            fig_risk.add_trace(go.Scatter(x=var_s.index, y=var_s.round(0), name=f"{leg} VaR", line=dict(width=1.6)))
        if len(var_series) > 1:
            combined = pd.concat(var_series.values(), axis=1).ffill().sum(axis=1, min_count=len(var_series))
            fig_risk.add_trace(go.Scatter(x=combined.index, y=combined.round(0),
                                          name=f"{commodity} Combined (1 lot each)", line=dict(color=NAVY, width=2.4)))
        st.plotly_chart(fig_risk, use_container_width=True)
        st.caption("Same parametric method as the standalone VaR Monitor: settlement × lot size × "
                   "rolling-vol × 2.3263 (one-tailed 99%). Combined line assumes 1 lot per leg.")

    with r_pct:
        st.markdown(lbl(f"{commodity} — Current Volatility Percentile vs Full History"), unsafe_allow_html=True)
        rows = []
        for leg in legs:
            vol, var_s = _leg_vol_var(leg)
            hist = vol.dropna()
            if hist.empty:
                continue
            cur = hist.iloc[-1]
            pct = float((hist < cur).mean() * 100)
            rows.append({"Leg": leg, "Percentile": pct, "Current VaR": f"${var_s.dropna().iloc[-1]:,.0f}"})
        pdf = pd.DataFrame(rows).sort_values("Percentile")
        fig_pct2 = go.Figure(go.Bar(
            x=pdf["Percentile"], y=pdf["Leg"], orientation="h",
            marker_color=[leg_colors[l] for l in pdf["Leg"]],
            text=pdf.apply(lambda r: f"{r['Percentile']:.0f}th | {r['Current VaR']}", axis=1),
            textposition="outside",
        ))
        fig_pct2.add_vline(x=50, line_dash="dot", line_color="#aaaaaa")
        fig_pct2.add_vline(x=80, line_dash="dot", line_color=AMBER)
        fig_pct2.update_layout(height=260, xaxis=dict(range=[0, 120], title="Percentile"),
                               margin=dict(t=10, b=10, l=4, r=120), **_D)
        st.plotly_chart(fig_pct2, use_container_width=True)

    with r_heat:
        st.markdown(lbl(f"{commodity} — VaR Seasonality (Year x Month)"), unsafe_allow_html=True)
        leg_pick_h = st.selectbox("Leg", legs, key="var_heat_leg")
        _, var_s = _leg_vol_var(leg_pick_h)
        vdf = var_s.rename("VaR").reset_index().rename(columns={"index": "Date"})
        year_month_heatmap(vdf, "Date", "VaR", "Avg VaR", key="var_heatmap")
