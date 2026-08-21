"""
Landing Page — Unified Commodity Dashboard
Commodity -> Exposure (Flat | Spread | Arb | Volatility | Risk) view, each
exposure broken into inner tabs so multiple panels fit without one giant
scroll.

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

Some panels here are deliberately simplified versions of a standalone
project's real logic (flagged in-app where that's the case) — e.g. Volatility
uses a median-strike ATM approximation rather than the Options dashboard's
full futures-anchored per-expiry mapping, and the Monte Carlo panel fixes
position size at 1 lot per leg rather than exposing a full position editor.
Use the standalone dashboards for that level of depth; this page is the
at-a-glance read across all five exposures at once.
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
MONTHS    = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ── Style (consistent with VaR Monitor's theme) ────────────────────────────────
NAVY, BLACK, GREEN, RED, GREY, AMBER = "#0a2463", "#1d1d1f", "#16a34a", "#dc2626", "#9ca3af", "#d97706"
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

def year_month_heatmap(df: pd.DataFrame, date_col: str, value_col: str, title: str, colorscale=None, key=None):
    """Shared Year x Month average-value heatmap, same shape as the VaR project's."""
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
    z = pivot.values
    text_vals = [[f"{v:,.1f}" if not pd.isna(v) else "" for v in row] for row in z]
    fig_hm = go.Figure(go.Heatmap(
        z=z, x=list(pivot.columns), y=[str(y) for y in pivot.index],
        text=text_vals, texttemplate="%{text}", textfont=dict(size=8, color=BLACK),
        colorscale=colorscale or [[0.0, "#d4edda"], [0.4, "#fff3cd"], [0.7, "#f8d7a0"], [1.0, "#f5c6cb"]],
        showscale=True, colorbar=dict(thickness=10, len=0.8, tickfont=dict(size=8, color=BLACK)),
        hoverongaps=False, hovertemplate="<b>%{y} %{x}</b><br>" + title + ": %{z:,.1f}<extra></extra>",
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
    df = pd.read_parquet(DB / f"rollex_{code}.parquet")[["rollex_px"]].reset_index()
    df.columns = ["Date", "Close"]
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").dropna()

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
# FLAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_flat:
    f_price, f_oi, f_vol, f_dist = st.tabs(
        ["Price & OI", "OI Market Share", "Volume", "Indexed & Distribution"]
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

    with f_oi:
        st.markdown(lbl(f"{commodity} — Open Interest by Contract Month"), unsafe_allow_html=True)
        oi_lookback = st.slider("Lookback (calendar days)", 60, 720, 270, step=30, key="oi_ms_lookback")
        cols = st.columns(len(legs))
        for leg, col in zip(legs, cols):
            with col:
                st.markdown(f"**{leg} — Top 6 contracts by current OI**")
                fdf = load_futures_full(cfg["futures_codes"][leg])
                max_d = fdf["Date"].max()
                fdf = fdf[fdf["Date"] >= max_d - pd.Timedelta(days=oi_lookback)]
                latest_oi = fdf[fdf["Date"] == max_d].dropna(subset=["open_interest"])
                top_syms = latest_oi.nlargest(6, "open_interest")["ice_symbol"].tolist()
                sub = fdf[fdf["ice_symbol"].isin(top_syms)]
                piv = sub.pivot_table(index="Date", columns="ice_symbol", values="open_interest", aggfunc="last")
                fig_ms = go.Figure()
                for c in piv.columns:
                    fig_ms.add_trace(go.Scatter(x=piv.index, y=piv[c], stackgroup="one", name=c))
                fig_ms.update_layout(height=360, yaxis=dict(title="Open Interest", gridcolor="#f0f0f0"),
                                     legend=dict(orientation="h", y=1.1, font=dict(size=8)),
                                     margin=dict(t=10, b=10, l=4, r=4), **_D)
                st.plotly_chart(fig_ms, use_container_width=True, key=f"oi_ms_{leg}")
        st.caption("Stacked OI by individual contract (top 6 currently, by open interest) — shows which "
                   "expiry the market is concentrated in, not just the aggregate total.")

    with f_vol:
        st.markdown(lbl(f"{commodity} — Volume"), unsafe_allow_html=True)
        vol_lookback = st.slider("Lookback (calendar days)", 60, 720, 270, step=30, key="vol_ms_lookback")
        roll_n = st.slider("Rolling window for total volume (days)", 5, 60, 20, step=5, key="vol_roll_n")
        cols = st.columns(len(legs))
        for leg, col in zip(legs, cols):
            with col:
                st.markdown(f"**{leg} — Volume by contract (top 6) + rolling total**")
                fdf = load_futures_full(cfg["futures_codes"][leg])
                max_d = fdf["Date"].max()
                fdf = fdf[fdf["Date"] >= max_d - pd.Timedelta(days=vol_lookback)]
                latest_oi = fdf[fdf["Date"] == max_d].dropna(subset=["open_interest"])
                top_syms = latest_oi.nlargest(6, "open_interest")["ice_symbol"].tolist()
                sub = fdf[fdf["ice_symbol"].isin(top_syms)]
                piv = sub.pivot_table(index="Date", columns="ice_symbol", values="volume", aggfunc="last")
                fig_v = make_subplots(specs=[[{"secondary_y": True}]])
                for c in piv.columns:
                    fig_v.add_trace(go.Bar(x=piv.index, y=piv[c], name=c), secondary_y=False)
                total_vol = fdf.groupby("Date")["volume"].sum(min_count=1)
                rolling_tot = total_vol.rolling(roll_n).mean()
                fig_v.add_trace(go.Scatter(x=rolling_tot.index, y=rolling_tot.values,
                                           name=f"{roll_n}D Avg (All Contracts)",
                                           line=dict(color=NAVY, width=2)), secondary_y=True)
                fig_v.update_layout(height=360, barmode="stack",
                                    legend=dict(orientation="h", y=1.1, font=dict(size=8)),
                                    margin=dict(t=10, b=10, l=4, r=4), **_D)
                fig_v.update_yaxes(title_text="Daily Volume", secondary_y=False, gridcolor="#f0f0f0")
                fig_v.update_yaxes(title_text=f"{roll_n}D Avg Total Volume", secondary_y=True, showgrid=False)
                st.plotly_chart(fig_v, use_container_width=True, key=f"vol_ms_{leg}")

    with f_dist:
        st.markdown(lbl(f"{commodity} — Indexed Price & Return Distribution"), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Indexed to 100**")
            fig_idx = base_fig(height=380, yaxis_title="Index (start = 100)")
            for leg in legs:
                rx = load_rollex(cfg["rollex_codes"][leg]).set_index("Date").sort_index()
                idx = (rx["Close"] / rx["Close"].iloc[0]) * 100
                fig_idx.add_trace(go.Scatter(x=idx.index, y=idx.values, name=leg,
                                             line=dict(color=leg_colors[leg], width=1.6)))
            st.plotly_chart(fig_idx, use_container_width=True)
        with c2:
            st.markdown("**Daily Log-Return Distribution**")
            fig_hist = base_fig(height=380, yaxis_title="Frequency")
            for leg in legs:
                rx = load_rollex(cfg["rollex_codes"][leg]).set_index("Date").sort_index()
                log_ret = (np.log(rx["Close"] / rx["Close"].shift(1)) * 100).dropna()
                fig_hist.add_trace(go.Histogram(x=log_ret, name=leg, opacity=0.6, nbinsx=80,
                                                marker_color=leg_colors[leg]))
            fig_hist.update_layout(barmode="overlay", xaxis=dict(title="Daily log return (%)"))
            st.plotly_chart(fig_hist, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SPREAD
# ══════════════════════════════════════════════════════════════════════════════
with tab_spread:
    s_yield, s_rank, s_heat, s_cost = st.tabs(
        ["Yield & Curve", "Ranking & Curve History", "Roll Yield Heatmap", "Roll Cost"]
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
                    fig_ry.add_trace(go.Scatter(x=s["Date"], y=s["Roll_Yield_1yr"], name=code,
                                                line=dict(width=1.6)))
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
        st.markdown(lbl(f"{commodity} — Roll Yield Percentile & Curve Shift"), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Current Roll Yield Percentile vs Own History**")
            rows = []
            for code in cfg["ry_codes"]:
                s = ry[ry["Commodity"] == code].sort_values("Date")["Roll_Yield_1yr"].dropna()
                if s.empty:
                    continue
                pct = float((s < s.iloc[-1]).mean() * 100)
                rows.append({"Code": code, "Percentile": pct, "Current": s.iloc[-1]})
            rdf = pd.DataFrame(rows)
            fig_pct = go.Figure(go.Bar(
                x=rdf["Percentile"], y=rdf["Code"], orientation="h",
                marker_color=[leg_colors.get(c.replace("RC", "LRC"), NAVY) for c in rdf["Code"]],
                text=rdf.apply(lambda r: f"{r['Percentile']:.0f}th | {r['Current']:.2f}%", axis=1),
                textposition="outside",
            ))
            fig_pct.add_vline(x=50, line_dash="dot", line_color="#aaaaaa")
            fig_pct.update_layout(height=280, xaxis=dict(range=[0, 120], title="Percentile"),
                                  margin=dict(t=10, b=10, l=4, r=100), **_D)
            st.plotly_chart(fig_pct, use_container_width=True)
        with c2:
            st.markdown("**Curve Shift — Today vs 1 Week Ago vs 4 Weeks Ago**")
            code_pick = st.selectbox("Contract", cfg["ry_codes"], key="curve_shift_code")
            s = ry[ry["Commodity"] == code_pick].sort_values("Date")
            fig_shift = base_fig(height=280, yaxis_title="Price")
            offsets = {"Today": 0, "1 week ago": -5, "4 weeks ago": -20}
            for label, off in offsets.items():
                if len(s) <= abs(off):
                    continue
                row = s.iloc[-1 + off] if off != 0 else s.iloc[-1]
                fig_shift.add_trace(go.Scatter(x=curve_cols, y=[row[c] for c in curve_cols],
                                               name=label, mode="lines+markers"))
            st.plotly_chart(fig_shift, use_container_width=True)

    with s_heat:
        st.markdown(lbl(f"{commodity} — Roll Yield Seasonality (Year x Month)"), unsafe_allow_html=True)
        code_pick_hm = st.selectbox("Contract", cfg["ry_codes"], key="ry_heat_code")
        s = ry[ry["Commodity"] == code_pick_hm]
        year_month_heatmap(s, "Date", "Roll_Yield_1yr", "Avg Roll Yield", key="ry_heatmap")

    with s_cost:
        st.markdown(lbl(f"{commodity} — Roll Cost (c1 vs c2)"), unsafe_allow_html=True)
        st.caption("Roll cost = front-month minus second-month price. Positive means rolling forward "
                   "costs you money (contango); negative means it pays you (backwardation).")
        c1, c2 = st.columns(2)
        for code, col in zip(cfg["ry_codes"], [c1, c2]):
            with col:
                s = ry[ry["Commodity"] == code].sort_values("Date").copy()
                s["roll_cost"] = s["c1"] - s["c2"]
                st.markdown(f"**{code} — Roll Cost Over Time**")
                fig_rc = base_fig(height=300, yaxis_title="c1 - c2")
                fig_rc.add_trace(go.Scatter(x=s["Date"], y=s["roll_cost"], line=dict(color=leg_colors.get(code.replace("RC","LRC"), NAVY), width=1.4)))
                fig_rc.add_hline(y=0, line=dict(color="#aaaaaa", width=1))
                st.plotly_chart(fig_rc, use_container_width=True, key=f"rollcost_{code}")
        st.markdown("**Seasonality (Year x Month)**")
        code_pick_cost = st.selectbox("Contract", cfg["ry_codes"], key="cost_heat_code")
        s = ry[ry["Commodity"] == code_pick_cost].copy()
        s["roll_cost"] = s["c1"] - s["c2"]
        year_month_heatmap(s, "Date", "roll_cost", "Avg Roll Cost",
                           colorscale=[[0, "#d4edda"], [0.5, "#ffffff"], [1, "#f5c6cb"]],
                           key="rollcost_heatmap")

# ══════════════════════════════════════════════════════════════════════════════
# ARB
# ══════════════════════════════════════════════════════════════════════════════
with tab_arb:
    if commodity == "Coffee":
        a_spread, a_z, a_legs, a_scatter = st.tabs(["Spread", "Z-Score", "Legs & Ratio", "Return Scatter"])

        kc = load_arb_front("KC")["px1"]
        rc = load_arb_front("RC")["px1"]
        kc_mt = kc * KC_FACTOR
        spread = (kc_mt - rc).dropna()

        with a_spread:
            st.markdown(lbl("Arabica Premium over Robusta ($/MT)"), unsafe_allow_html=True)
            window = st.slider("Rolling window (days)", 60, 504, 252, step=21, key="arb_win")
            mu, sig = spread.rolling(window).mean(), spread.rolling(window).std()
            fig_arb = base_fig(height=440, yaxis_title="$/MT")
            fig_arb.add_trace(go.Scatter(x=spread.index, y=mu + 2*sig, name="+2σ",
                                         line=dict(color=RED, width=1, dash="dot")))
            fig_arb.add_trace(go.Scatter(x=spread.index, y=mu - 2*sig, name="-2σ",
                                         line=dict(color=GREEN, width=1, dash="dot")))
            fig_arb.add_trace(go.Scatter(x=spread.index, y=mu, name="Rolling Mean",
                                         line=dict(color=GREY, width=1)))
            fig_arb.add_trace(go.Scatter(x=spread.index, y=spread, name="Spread",
                                         line=dict(color=NAVY, width=1.8)))
            st.plotly_chart(fig_arb, use_container_width=True)
            st.caption("Full pair/unit controls live in the standalone Arb dashboard — "
                       "this is the headline KC/RC read.")

        with a_z:
            st.markdown(lbl("Spread Z-Score"), unsafe_allow_html=True)
            z_window = st.slider("Z-score lookback (days)", 60, 504, 252, step=21, key="arb_zwin")
            mu_z, sig_z = spread.rolling(z_window).mean(), spread.rolling(z_window).std()
            z = (spread - mu_z) / sig_z
            fig_z = base_fig(height=380, yaxis_title="Z-Score")
            fig_z.add_trace(go.Scatter(x=z.index, y=z, line=dict(color=NAVY, width=1.6), name="Z-Score"))
            for level, color in [(2, RED), (-2, GREEN), (0, GREY)]:
                fig_z.add_hline(y=level, line=dict(color=color, width=1, dash="dot"))
            st.plotly_chart(fig_z, use_container_width=True)

        with a_legs:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Individual Legs ($/MT)**")
                fig_legs = base_fig(height=380, yaxis_title="$/MT")
                fig_legs.add_trace(go.Scatter(x=kc_mt.index, y=kc_mt, name="KC ($/MT)", line=dict(color=NAVY, width=1.4)))
                fig_legs.add_trace(go.Scatter(x=rc.index, y=rc, name="RC ($/MT)", line=dict(color="#8b1a00", width=1.4)))
                st.plotly_chart(fig_legs, use_container_width=True)
            with c2:
                st.markdown("**KC / RC Ratio**")
                ratio = (kc_mt / rc).dropna()
                r_window = st.slider("Rolling window (days)", 60, 504, 252, step=21, key="ratio_win")
                r_mu, r_sig = ratio.rolling(r_window).mean(), ratio.rolling(r_window).std()
                fig_ratio = base_fig(height=380, yaxis_title="KC / RC")
                fig_ratio.add_trace(go.Scatter(x=ratio.index, y=r_mu + r_sig, name="+1σ", line=dict(color=RED, width=1, dash="dot")))
                fig_ratio.add_trace(go.Scatter(x=ratio.index, y=r_mu - r_sig, name="-1σ", line=dict(color=GREEN, width=1, dash="dot")))
                fig_ratio.add_trace(go.Scatter(x=ratio.index, y=ratio, name="Ratio", line=dict(color=NAVY, width=1.6)))
                st.plotly_chart(fig_ratio, use_container_width=True)

        with a_scatter:
            st.markdown(lbl("Daily Return Co-Movement"), unsafe_allow_html=True)
            ret_df = pd.concat([kc_mt.pct_change().rename("KC"), rc.pct_change().rename("RC")], axis=1).dropna()
            ret_df = ret_df.tail(504)
            if len(ret_df) > 2:
                coeffs = np.polyfit(ret_df["KC"], ret_df["RC"], 1)
                trend_x = np.linspace(ret_df["KC"].min(), ret_df["KC"].max(), 50)
                trend_y = coeffs[0] * trend_x + coeffs[1]
                r2 = np.corrcoef(ret_df["KC"], ret_df["RC"])[0, 1] ** 2
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(x=ret_df["KC"]*100, y=ret_df["RC"]*100, mode="markers",
                                            marker=dict(size=5, color=NAVY, opacity=0.5), name="Daily returns"))
                fig_sc.add_trace(go.Scatter(x=trend_x*100, y=trend_y*100, mode="lines",
                                            line=dict(color=RED, width=1.5), name=f"Fit (R²={r2:.2f})"))
                fig_sc.update_layout(height=420, xaxis=dict(title="KC daily return (%)", gridcolor="#f0f0f0"),
                                     yaxis=dict(title="RC daily return (%)", gridcolor="#f0f0f0"),
                                     margin=dict(t=10, b=10, l=4, r=4), **_D)
                st.plotly_chart(fig_sc, use_container_width=True)
                st.caption("Last 504 sessions (~2 years). A tight cluster around the fit line means the "
                           "legs move together; a wide scatter means the arb decouples from broad market moves.")
    else:
        st.info("No Arb pair mapped for this commodity yet.")

# ══════════════════════════════════════════════════════════════════════════════
# VOLATILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab_vol:
    v_ivrv, v_term, v_smile, v_flow = st.tabs(
        ["IV vs RV (Simplified)", "ATM Term Structure (Snapshot)", "Vol Smile", "OI / Volume by Strike"]
    )

    with v_ivrv:
        st.markdown(lbl(f"{commodity} — Implied vs Realized Vol (simplified)"), unsafe_allow_html=True)
        st.caption("Simplified snapshot: near-the-money average IV of the nearest listed expiry vs. "
                   "20-day realized vol of the continuous price. For full per-expiry term structure "
                   "and Call/Put breakdown, use the standalone Options dashboard.")
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

    with v_term:
        st.markdown(lbl(f"{commodity} — ATM Implied Vol Across Expiries (latest date)"), unsafe_allow_html=True)
        st.caption("Approximate ATM: nearest available strike to the median strike traded that day, "
                   "per expiry. Not the same futures-anchored mapping the Options dashboard uses for "
                   "serial-month expiries, but the general shape holds.")
        leg_pick = st.selectbox("Leg", legs, key="term_leg")
        try:
            odf = load_options(leg_pick)
            max_d = odf["date"].max()
            today_df = odf[odf["date"] == max_d].dropna(subset=["strike"])
            expiries = sorted(today_df[["expiry_year", "expiry_month"]].drop_duplicates().itertuples(index=False), key=lambda x: (x.expiry_year, x.expiry_month))[:6]
            rows = []
            for exp in expiries:
                edf = today_df[(today_df["expiry_year"] == exp.expiry_year) & (today_df["expiry_month"] == exp.expiry_month)]
                mid_strike = edf["strike"].median()
                edf = edf.copy()
                edf["dist"] = (edf["strike"] - mid_strike).abs()
                atm_row = edf.dropna(subset=["impvol"]).nsmallest(1, "dist")
                if not atm_row.empty:
                    rows.append({"Expiry": f"{MONTHS[int(exp.expiry_month)-1]} '{str(int(exp.expiry_year))[-2:]}",
                                 "ATM IV": float(atm_row.iloc[0]["impvol"]),
                                 "Strike": float(atm_row.iloc[0]["strike"])})
            if rows:
                tdf = pd.DataFrame(rows)
                fig_term = go.Figure(go.Scatter(x=tdf["Expiry"], y=tdf["ATM IV"], mode="lines+markers",
                                                line=dict(color=leg_colors[leg_pick], width=2),
                                                text=[f"Strike {s:.0f}" for s in tdf["Strike"]],
                                                hovertemplate="%{x}<br>ATM IV: %{y:.1f}%<br>%{text}<extra></extra>"))
                fig_term.update_layout(height=400, yaxis=dict(title="ATM IV (%)", gridcolor="#f0f0f0"),
                                       margin=dict(t=10, b=10, l=4, r=4), **_D)
                st.plotly_chart(fig_term, use_container_width=True)
            else:
                st.info("No IV data available for this leg's expiries on the latest date.")
        except FileNotFoundError:
            st.info(f"No options data available for {leg_pick}.")

    with v_smile:
        st.markdown(lbl(f"{commodity} — Vol Smile"), unsafe_allow_html=True)
        leg_pick2 = st.selectbox("Leg", legs, key="smile_leg")
        try:
            odf = load_options(leg_pick2)
            max_d = odf["date"].max()
            today_df = odf[odf["date"] == max_d].dropna(subset=["strike"])
            exp_opts = sorted(today_df[["expiry_year", "expiry_month"]].drop_duplicates().itertuples(index=False), key=lambda x: (x.expiry_year, x.expiry_month))
            exp_labels = {f"{MONTHS[int(e.expiry_month)-1]} '{str(int(e.expiry_year))[-2:]}": e for e in exp_opts}
            exp_pick = st.selectbox("Expiry", list(exp_labels.keys()), key="smile_exp")
            e = exp_labels[exp_pick]
            edf = today_df[(today_df["expiry_year"] == e.expiry_year) & (today_df["expiry_month"] == e.expiry_month)]
            fig_smile = base_fig(height=420, yaxis_title="Implied Vol (%)")
            for otype, color in [("C", GREEN), ("P", RED)]:
                sub = edf[edf["option_type"].astype(str).str.upper().str.startswith(otype)].dropna(subset=["impvol"]).sort_values("strike")
                if not sub.empty:
                    fig_smile.add_trace(go.Scatter(x=sub["strike"], y=sub["impvol"], mode="lines+markers",
                                                   name="Calls" if otype == "C" else "Puts",
                                                   line=dict(color=color, width=1.8)))
            fig_smile.update_layout(xaxis=dict(title="Strike", gridcolor="#f0f0f0"))
            st.plotly_chart(fig_smile, use_container_width=True)
        except FileNotFoundError:
            st.info(f"No options data available for {leg_pick2}.")

    with v_flow:
        st.markdown(lbl(f"{commodity} — OI & Volume by Strike (latest date, nearest expiry)"), unsafe_allow_html=True)
        leg_pick3 = st.selectbox("Leg", legs, key="flow_leg")
        try:
            odf = load_options(leg_pick3)
            max_d = odf["date"].max()
            today_df = odf[odf["date"] == max_d].dropna(subset=["strike"])
            near_key = today_df.sort_values(["expiry_year", "expiry_month"])
            if not near_key.empty:
                fy, fm = near_key.iloc[0][["expiry_year", "expiry_month"]]
                edf = today_df[(today_df["expiry_year"] == fy) & (today_df["expiry_month"] == fm)]
                calls = edf[edf["option_type"].astype(str).str.upper().str.startswith("C")].set_index("strike")[["oi", "volume"]].rename(columns={"oi": "Call OI", "volume": "Call Vol"})
                puts  = edf[edf["option_type"].astype(str).str.upper().str.startswith("P")].set_index("strike")[["oi", "volume"]].rename(columns={"oi": "Put OI", "volume": "Put Vol"})
                mid_strike = edf["strike"].median()
                merged = calls.join(puts, how="outer").sort_index()
                merged = merged[(merged.index >= mid_strike * 0.8) & (merged.index <= mid_strike * 1.2)]
                merged.index.name = "Strike"
                oi_cols  = [c for c in ["Call OI", "Put OI"] if merged[c].notna().any()]
                vol_cols = [c for c in ["Call Vol", "Put Vol"] if merged[c].notna().any()]
                styled = merged.style
                if oi_cols:
                    styled = styled.background_gradient(cmap="Greens", subset=oi_cols)
                if vol_cols:
                    styled = styled.background_gradient(cmap="Blues", subset=vol_cols)
                styled = styled.format("{:,.0f}", na_rep="")
                st.dataframe(styled, use_container_width=True, height=460)
            else:
                st.info("No expiry data available.")
        except FileNotFoundError:
            st.info(f"No options data available for {leg_pick3}.")

# ══════════════════════════════════════════════════════════════════════════════
# RISK
# ══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    r_var, r_pct, r_heat, r_mc = st.tabs(
        ["Parametric VaR", "Vol Percentile", "VaR Heatmap", "Monte Carlo (simplified)"]
    )

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
        return log_ret, vol, var_s, settle

    with r_var:
        st.markdown(lbl(f"{commodity} — 1-Day VaR · 99% Confidence · Per Lot"), unsafe_allow_html=True)
        var_series = {}
        fig_risk = base_fig(height=440, yaxis_title="VaR (USD / lot)")
        for leg in legs:
            if leg not in LOT_SIZES:
                continue
            _, _, var_s, _ = _leg_vol_var(leg)
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
            log_ret, vol, var_s, _ = _leg_vol_var(leg)
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
        _, _, var_s, _ = _leg_vol_var(leg_pick_h)
        vdf = var_s.rename("VaR").reset_index().rename(columns={"index": "Date"})
        year_month_heatmap(vdf, "Date", "VaR", "Avg VaR", key="var_heatmap")

    with r_mc:
        st.markdown(lbl(f"{commodity} — Portfolio VaR, Monte Carlo (1 lot per leg, fixed)"), unsafe_allow_html=True)
        st.caption("Simplified: fixed 1 lot per leg (choose direction below), normal returns, no fat-tail "
                   "toggle. For real position sizing and Student-t tails, use the standalone VaR Monitor.")
        dir_cols = st.columns(len(legs))
        directions = {}
        for leg, col in zip(legs, dir_cols):
            with col:
                directions[leg] = st.radio(f"{leg} direction", ["Long", "Short"], horizontal=True, key=f"mc_dir_{leg}")
        n_sims = st.select_slider("Simulations", [1000, 5000, 10000, 25000], value=10000, key="mc_nsims")

        ret_mx = pd.concat(
            [_leg_vol_var(leg)[0].rename(leg) for leg in legs if leg in LOT_SIZES], axis=1
        ).dropna()
        recent_r = ret_mx.tail(w)
        if len(recent_r) > 5 and recent_r.shape[1] >= 1:
            cov_mx = recent_r.cov().values
            try:
                L = np.linalg.cholesky(cov_mx)
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(cov_mx + np.eye(len(recent_r.columns)) * 1e-10)
            rng = np.random.default_rng(42)
            Z = rng.standard_normal((n_sims, len(recent_r.columns)))
            sim_ret = Z @ L.T

            latest_settle = {leg: load_front_price(cfg["futures_codes"][leg])["settlement"].dropna().iloc[-1] for leg in recent_r.columns}
            sign = {leg: (1 if directions[leg] == "Long" else -1) for leg in recent_r.columns}
            dollar_exp = np.array([sign[leg] * LOT_SIZES[leg] * latest_settle[leg] for leg in recent_r.columns])
            sim_pnl = sim_ret @ dollar_exp

            alpha = 0.01
            cutoff = float(np.percentile(sim_pnl, alpha * 100))
            port_var = max(-cutoff, 0.0)
            tail_mask = sim_pnl <= cutoff
            port_cvar = float(-sim_pnl[tail_mask].mean()) if tail_mask.any() else port_var

            k1, k2, k3 = st.columns(3)
            k1.metric("Portfolio VaR (99%)", f"${port_var:,.0f}")
            k2.metric("CVaR / Exp Shortfall", f"${port_cvar:,.0f}")
            k3.metric("Gross $ Exposure", f"${np.abs(dollar_exp).sum():,.0f}")

            c1, c2 = st.columns([3, 2])
            with c1:
                fig_hist2 = go.Figure()
                fig_hist2.add_trace(go.Histogram(x=sim_pnl[~tail_mask], nbinsx=80, marker_color="#2a6496",
                                                 opacity=0.55, name="Within VaR"))
                fig_hist2.add_trace(go.Histogram(x=sim_pnl[tail_mask], nbinsx=30, marker_color=RED,
                                                 opacity=0.85, name="Tail (>99%)"))
                fig_hist2.add_vline(x=cutoff, line=dict(color=RED, width=2, dash="dash"))
                fig_hist2.update_layout(barmode="overlay", height=380,
                                        xaxis=dict(title="1-Day P&L (USD)", tickformat="$,.0f"),
                                        yaxis=dict(title="Frequency", gridcolor="#f0f0f0"),
                                        legend=dict(orientation="h", y=-0.2, font=dict(size=8)),
                                        margin=dict(t=10, b=50, l=4, r=4), **_D)
                st.plotly_chart(fig_hist2, use_container_width=True)
            with c2:
                if recent_r.shape[1] > 1:
                    corr = recent_r.corr()
                    fig_corr2 = go.Figure(go.Heatmap(
                        z=corr.values, x=list(corr.columns), y=list(corr.columns),
                        text=[[f"{v:.2f}" for v in row] for row in corr.values],
                        texttemplate="%{text}", colorscale=[[0, RED], [0.5, "#ffffff"], [1, "#2a6496"]],
                        zmin=-1, zmax=1, showscale=True,
                    ))
                    fig_corr2.update_layout(height=380, margin=dict(t=10, b=10, l=4, r=4), **_D)
                    st.plotly_chart(fig_corr2, use_container_width=True)
                else:
                    st.info("Correlation needs at least 2 legs.")
        else:
            st.info("Not enough overlapping return history to run the simulation.")
