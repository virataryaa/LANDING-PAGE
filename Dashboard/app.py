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

st.set_page_config(page_title="Landing Page", layout="wide", initial_sidebar_state="expanded")

# ── Local database (synced in by Code/ingest.py — see its docstring for why
#    this project keeps its own copy instead of reading sibling repos directly) ─
DB = Path(__file__).resolve().parents[1] / "Database"

KC_FACTOR = 22.0462           # ¢/lb -> $/MT, same conversion the Arb project uses
CONF_Z    = 2.3263            # one-tailed 99% VaR z-score, same as the VaR project
LOT_SIZES = {"KC": 375, "LRC": 10}
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
  /* Top-level tab labels only (Flat/Spread/Arb/Volatility/Risk/Positioning/
     Currency) — excludes any button nested inside a tab's own content
     panel, so inner sub-tabs (Vol Percentile, Forward Curves, etc.) keep
     the default look. */
  div[data-testid="stTabs"]:not([data-testid="stTabsPanel"] div[data-testid="stTabs"]) > div[data-baseweb="tab-list"]{
    gap:8px!important;
  }
  button[data-baseweb="tab"]:not([data-testid="stTabsPanel"] button[data-baseweb="tab"]){
    padding:8px 18px!important;margin:0 2px 6px!important;border-radius:8px!important;
  }
  button[data-baseweb="tab"]:not([data-testid="stTabsPanel"] button[data-baseweb="tab"]):nth-of-type(-n+4){
    background:#dbeafe!important;
  }
  button[data-baseweb="tab"]:not([data-testid="stTabsPanel"] button[data-baseweb="tab"]):nth-of-type(n+5){
    background:#e5e7eb!important;
  }
</style>""", unsafe_allow_html=True)

_D = dict(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
          font=dict(family="-apple-system,Helvetica Neue,sans-serif", color=BLACK, size=10))

# ── Source links — URLs taken from the Homepage hub (icebreaker.html)'s LINKS
#    object, LSEG entries, so a user can jump to the real app for depth ───────
SOURCE_URLS = {
    "Rollex":      "https://interim-migration-rollex-nhawujes2vizbbacqtxkgf.streamlit.app/",
    "Roll Yield":  "https://interim-migration-rollyield-nr4yksj7oq3g3jygs73j7e.streamlit.app/",
    "Arb":         "https://interim-migration-arb-6fxwrbfx4539voujkhv6di.streamlit.app/",
    "Options":     "https://interim-migration-options-aa7ck2bsxhxgkbnczleggq.streamlit.app/",
    "Futures OI":  "https://interim-migration-futures-oi-3ojs8tugpuxycdfh9nrtdn.streamlit.app/",
    "VaR":         "https://interim-migration-var-st5pdh27fxp8feev8wuppw.streamlit.app/",
    "COT Comprehensive": "https://interim-migration-cot-all-dviic3fxsojoe9xvqmxzzs.streamlit.app/",
    "COT Distribution":  "https://interim-migration-cot-all-4yhhgv3dfzrsz9krjrlhyp.streamlit.app/",
    "Currency":    "https://interim-migration-currency-ejlqfktneqqxqqdmxz8dwp.streamlit.app/",
}

def source_link(*names):
    parts = " &nbsp;&nbsp;".join(
        f'<a href="{SOURCE_URLS[n]}" target="_blank" rel="noopener" '
        f'style="color:#0a2463;font-weight:500;font-size:.72rem;text-decoration:none;'
        f'letter-spacing:.02em">{n} &#8599;</a>' for n in names)
    st.markdown(f'<div style="text-align:right;margin:-8px 0 4px">{parts}</div>',
               unsafe_allow_html=True)

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

# ── Comprehensive Grid — verbatim port of the Futures dashboard's table ───────
def _cg_safe(v, default=1.0):
    v = float(v) if pd.notna(v) else default
    return v if v > 0 else default

def _cg_oi_heatmap_style(v, vmin, vmax):
    if pd.isna(v):
        return ""
    vmin = float(vmin) if pd.notna(vmin) else 0.0
    vmax = float(vmax) if pd.notna(vmax) else vmin + 1.0
    span = vmax - vmin
    t = min(max((float(v) - vmin) / span, 0.0), 1.0) if span > 0 else 0.0
    r = round(255 + t * (150 - 255)); g = round(255 + t * (200 - 255)); b = round(255 + t * (165 - 255))
    return f"background-color:rgb({r},{g},{b});color:#1a1a1a"

def _cg_bar_style(v, vmax, color):
    if pd.isna(v) or v == 0:
        return ""
    pct = min(abs(float(v)) / _cg_safe(vmax), 1.0) * 100
    return f"background:linear-gradient(to right, {color} {pct:.1f}%, transparent {pct:.1f}%)"

def _cg_diverging_bar_style(v, vmax, pos_color, neg_color):
    if pd.isna(v) or v == 0:
        return ""
    half_pct = min(abs(float(v)) / _cg_safe(vmax), 1.0) * 50
    if v >= 0:
        lo, hi, color = 50.0, 50.0 + half_pct, pos_color
    else:
        lo, hi, color = 50.0 - half_pct, 50.0, neg_color
    return (f"background:linear-gradient(to right, transparent {lo:.1f}%, "
            f"{color} {lo:.1f}%, {color} {hi:.1f}%, transparent {hi:.1f}%)")

def _cg_oi_chg_style(v, vmax):
    if pd.isna(v):
        return ""
    return _cg_diverging_bar_style(v, vmax, "rgba(22,163,74,0.55)", "rgba(220,38,38,0.55)")

def _cg_vol_style(v, vmax):
    return _cg_bar_style(v, vmax, "rgba(56,189,248,0.55)")

@st.cache_data(max_entries=50, show_spinner=False)
def build_comprehensive_grid_html(code_lower: str, table_lookback: int):
    """Verbatim port of the Futures dashboard's build_oi_vol_table_html —
    same column layout, same heatmap/bar styling, same box-shadow group
    dividers. Only the data source changed (this project's own Database/)."""
    df_all_tbl = load_futures_full(code_lower).sort_values(["ice_symbol", "Date"])
    df_all_tbl["oi_change"] = df_all_tbl.groupby("ice_symbol")["open_interest"].diff()
    df_all_tbl["px_change"] = df_all_tbl.groupby("ice_symbol")["settlement"].pct_change() * 100

    max_date_tbl = df_all_tbl["Date"].max()
    cutoff_tbl   = max_date_tbl - pd.Timedelta(days=table_lookback)
    win_tbl      = df_all_tbl[df_all_tbl["Date"] >= cutoff_tbl].copy()
    if win_tbl.empty:
        return None

    ltd_map   = win_tbl.groupby("ice_symbol")["LTD"].first()
    syms_tbl  = ltd_map.sort_values().index.tolist()
    dates_tbl = sorted(win_tbl["Date"].unique(), reverse=True)

    oi_piv  = (win_tbl.pivot_table(index="Date", columns="ice_symbol", values="open_interest", aggfunc="last")
               .reindex(index=dates_tbl, columns=syms_tbl))
    chg_piv = (win_tbl.pivot_table(index="Date", columns="ice_symbol", values="oi_change", aggfunc="last")
               .reindex(index=dates_tbl, columns=syms_tbl))
    vol_piv = (win_tbl.pivot_table(index="Date", columns="ice_symbol", values="volume", aggfunc="last")
               .reindex(index=dates_tbl, columns=syms_tbl))
    px_piv  = (win_tbl.pivot_table(index="Date", columns="ice_symbol", values="px_change", aggfunc="last")
               .reindex(index=dates_tbl, columns=syms_tbl))

    total_chg = chg_piv.sum(axis=1, min_count=1)
    total_vol = vol_piv.sum(axis=1, min_count=1)

    oi_col_min  = oi_piv.min(axis=0)
    oi_col_max  = oi_piv.max(axis=0)
    chg_col_max = chg_piv.abs().max(axis=0)
    vol_col_max = vol_piv.max(axis=0)
    px_col_max  = px_piv.abs().max(axis=0)
    total_chg_absmax = float(total_chg.abs().max()) if total_chg.notna().any() else 1.0
    total_vol_max    = float(total_vol.max())        if total_vol.notna().any() else 1.0
    total_chg_absmax = total_chg_absmax if total_chg_absmax > 0 else 1.0
    total_vol_max    = total_vol_max    if total_vol_max    > 0 else 1.0

    css = """
    <style>
    .oivol-wrap { overflow:auto; max-height:640px; border:1px solid #e5e7eb; border-radius:6px; }
    .oivol-tbl { border-collapse:collapse; font-size:9px; font-family:'Inter',sans-serif; white-space:nowrap; }
    .oivol-tbl th, .oivol-tbl td { padding:2px 5px; text-align:center; border-bottom:1px solid #f0f0f0; }
    .oivol-tbl th { position:sticky; top:0; background:#fafafa; font-weight:600; z-index:2; }
    .oivol-tbl .grp-h { background:#eef2f7; }
    .oivol-tbl .grp-start { box-shadow: inset 2px 0 0 0 #374151; }
    .oivol-tbl .date-cell { position:sticky; left:0; background:#fff; text-align:center;
                             font-weight:600; z-index:1; box-shadow: inset -2px 0 0 0 #374151; }
    .oivol-tbl .tot-cell { background:#fffbea; font-weight:600; }
    .oivol-tbl .sub-h { color:#888; font-weight:400; font-size:8px; }
    </style>
    """

    h1 = '<tr><th class="date-cell" rowspan="2">Date</th>'
    for s in syms_tbl:
        h1 += f'<th class="grp-h grp-start" colspan="4">{s}</th>'
    h1 += '<th class="tot-cell grp-start" colspan="2">Total</th></tr>'

    h2 = "<tr>"
    for s in syms_tbl:
        h2 += ('<th class="sub-h grp-h grp-start">OI</th>'
               '<th class="sub-h grp-h">ΔOI</th>'
               '<th class="sub-h grp-h">PxΔ%</th>'
               '<th class="sub-h grp-h">Vol</th>')
    h2 += '<th class="sub-h tot-cell grp-start">ΔOI</th><th class="sub-h tot-cell">Vol</th></tr>'

    rows = []
    for d in dates_tbl:
        d_str = pd.Timestamp(d).strftime("%d %b %Y")
        row = f'<tr><td class="date-cell">{d_str}</td>'
        for s in syms_tbl:
            oi_v, chg_v, vol_v, px_v = oi_piv.at[d, s], chg_piv.at[d, s], vol_piv.at[d, s], px_piv.at[d, s]
            oi_txt  = f"{oi_v:,.0f}"  if pd.notna(oi_v)  else ""
            chg_txt = f"{chg_v:+,.0f}" if pd.notna(chg_v) else ""
            vol_txt = f"{vol_v:,.0f}" if pd.notna(vol_v) else ""
            px_txt  = f"{px_v:+.2f}%" if pd.notna(px_v)  else ""
            row += f'<td class="grp-start" style="{_cg_oi_heatmap_style(oi_v, oi_col_min[s], oi_col_max[s])}">{oi_txt}</td>'
            row += f'<td style="{_cg_oi_chg_style(chg_v, chg_col_max[s])}">{chg_txt}</td>'
            row += f'<td style="{_cg_oi_chg_style(px_v, px_col_max[s])}">{px_txt}</td>'
            row += f'<td style="{_cg_vol_style(vol_v, vol_col_max[s])}">{vol_txt}</td>'
        tc, tv = total_chg.loc[d], total_vol.loc[d]
        tc_txt = f"{tc:+,.0f}" if pd.notna(tc) else ""
        tv_txt = f"{tv:,.0f}"  if pd.notna(tv) else ""
        row += f'<td class="tot-cell grp-start" style="{_cg_oi_chg_style(tc, total_chg_absmax)}">{tc_txt}</td>'
        row += f'<td class="tot-cell" style="{_cg_vol_style(tv, total_vol_max)}">{tv_txt}</td>'
        row += "</tr>"
        rows.append(row)

    return (css + f'<div class="oivol-wrap"><table class="oivol-tbl"><thead>{h1}{h2}</thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')

# ── Options butterfly table — verbatim port of the Options dashboard's
#    render_commodity_tab "OI Change + Volume" inner tab (butterfly_html +
#    _change_pivot/get_vol_pivot + oi_color/vol_color) ────────────────────────
BFY_CALL_CODES = {1:"A",2:"B",3:"C",4:"D",5:"E",6:"F",7:"G",8:"H",9:"I",10:"J",11:"K",12:"L"}
BFY_PUT_CODES  = {1:"M",2:"N",3:"O",4:"P",5:"Q",6:"R",7:"S",8:"T",9:"U",10:"V",11:"W",12:"X"}

def _bfy_month_keys(df):
    return (df[["expiry_month", "expiry_year"]].drop_duplicates()
            .sort_values(["expiry_year", "expiry_month"])
            .apply(lambda r: (int(r.expiry_month), int(r.expiry_year)), axis=1).tolist())

def _bfy_meta(df, opt):
    return (df[df["option_type"] == opt][["ric", "strike", "expiry_month", "expiry_year"]]
            .drop_duplicates()
            .assign(mk=lambda x: list(zip(x.expiry_month.astype(int), x.expiry_year.astype(int))))
            .set_index("ric"))

def _bfy_clean(pivot, month_keys):
    if pivot.empty:
        return pivot
    pivot = pivot.reindex(columns=month_keys)
    return pivot.apply(lambda c: pd.to_numeric(c, errors="coerce")).astype(float)

def _bfy_change_pivot(df, month_keys, opt, src, old_date, new_date):
    d1 = df[(df["date"].dt.date == old_date) & (df["option_type"] == opt)][["ric", src]].set_index("ric")
    d2 = df[(df["date"].dt.date == new_date) & (df["option_type"] == opt)][["ric", src]].set_index("ric")
    merged = d1.join(d2, how="outer", lsuffix="_1", rsuffix="_2")
    merged["val"] = pd.to_numeric(merged[src+"_2"], errors="coerce") - pd.to_numeric(merged[src+"_1"], errors="coerce")
    meta = _bfy_meta(df, opt)
    result = merged.join(meta[["strike", "mk"]]).dropna(subset=["strike"])
    result = result[result["mk"].notna()]
    piv = result.pivot_table(index="strike", columns="mk", values="val", aggfunc="first")
    return _bfy_clean(piv, month_keys).sort_index(ascending=False)

def _bfy_vol_pivot(df, month_keys, opt, old_date, new_date):
    lo, hi = min(old_date, new_date), max(old_date, new_date)
    sub = df[(df["option_type"] == opt) & (df["date"].dt.date >= lo) & (df["date"].dt.date <= hi)].copy()
    sub["mk"] = list(zip(sub["expiry_month"].astype(int), sub["expiry_year"].astype(int)))
    sub["volume"] = pd.to_numeric(sub["volume"], errors="coerce")
    piv = sub.groupby(["strike", "mk"])["volume"].sum().unstack("mk")
    return _bfy_clean(piv, month_keys).sort_index(ascending=False)

def _bfy_tot(piv):
    if piv.empty or piv.notna().to_numpy().sum() == 0:
        return float("nan")
    return float(piv.sum(skipna=True).sum())

def _bfy_alpha(v, mx):
    return round(0.15 + min(abs(float(v)) / max(mx, 0.01), 1.0) * 0.50, 2)

def _bfy_oi_color(val, mx):
    if pd.isna(val) or val == 0: return ""
    a = _bfy_alpha(val, mx)
    return (f"background:rgba(66,133,244,{a});color:#1a1a2e" if val > 0
            else f"background:rgba(220,75,75,{a});color:#1a1a2e")

def _bfy_vol_color(val, mx):
    if pd.isna(val) or val == 0: return ""
    return f"background:rgba(66,133,244,{_bfy_alpha(val, mx)});color:#1a1a2e"

_BFY_CSS = """<style>
.bft{border-collapse:collapse;font-size:11px;font-family:-apple-system,sans-serif}
.bft th,.bft td{white-space:nowrap;padding:2px 5px}
.bft th{font-weight:600;letter-spacing:.03em;font-size:10px;text-align:center}
.bft td{text-align:right;border:1px solid #f0f0f0;color:#1a1a2e}
.bft .sc{text-align:center;font-weight:700;font-size:11px;color:#1a1a2e;
         background:#f5f5f5;border-left:2px solid #ccc;border-right:2px solid #ccc}
.bft .sc-atm{background:#f59e0b!important;color:#1a1a2e!important;font-weight:900!important}
.bft tr.atm-row td{border-top:2px solid #f59e0b!important;border-bottom:2px solid #f59e0b!important}
.bft tfoot td{font-weight:700;border-top:2px solid #bbb}
.bft tfoot .sc{font-size:9px;color:#888;background:#efefef}
.ch{background:#dce8fb;color:#1a56cc}
.ph{background:#fde8e8;color:#c0392b}
.kch{background:#ebebeb;color:#555}
</style>"""

def bfy_butterfly_html(cpiv, ppiv, atm, cfn, month_keys, fmt="{:.0f}", footer=True, title="",
                       fixed_strikes=None, snap_tol=None):
    ccols, pcols = list(reversed(month_keys)), list(month_keys)
    strikes = list(fixed_strikes) if fixed_strikes is not None else sorted(
        set(cpiv.index.tolist() if not cpiv.empty else []) | set(ppiv.index.tolist() if not ppiv.empty else []))
    if not strikes:
        return "<p>No strikes in range.</p>"

    def _flat(p):
        return p.values.astype(float).flatten() if not p.empty else np.array([], dtype=float)
    av = np.concatenate([_flat(cpiv), _flat(ppiv)])
    av = av[~np.isnan(av)]
    mx = float(np.max(np.abs(av))) if len(av) > 0 else 1.0
    nc, npu = len(ccols), len(pcols)

    h1 = (f'<tr><th colspan="{nc}" class="ch">Call</th><th class="kch">{title}</th>'
          f'<th colspan="{npu}" class="ph">Put</th></tr>')
    h2 = ('<tr>' + "".join(f'<th class="ch" style="color:#999;font-weight:400">{BFY_CALL_CODES[m]}{str(y)[-2:]}</th>' for m, y in ccols)
          + '<th class="kch"></th>'
          + "".join(f'<th class="ph" style="color:#ccc;font-weight:400">{BFY_PUT_CODES[m]}{str(y)[-2:]}</th>' for m, y in pcols) + '</tr>')
    h3 = ('<tr>' + "".join(f'<th class="ch">{MONTHS[m-1]}</th>' for m, y in ccols) + '<th class="kch"></th>'
          + "".join(f'<th class="ph">{MONTHS[m-1]}</th>' for m, y in pcols) + '</tr>')

    idx_cache = {}
    def cv(piv, s, mk):
        if piv.empty or mk not in piv.columns:
            return np.nan
        if snap_tol is not None:
            pid = id(piv)
            if pid not in idx_cache:
                idx_cache[pid] = np.array(piv.index.tolist(), dtype=float)
            idx_arr = idx_cache[pid]
            if len(idx_arr) == 0:
                return np.nan
            diffs = np.abs(idx_arr - s)
            if diffs.min() > snap_tol:
                return np.nan
            s = idx_arr[diffs.argmin()]
        elif s not in piv.index:
            return np.nan
        v = piv.at[s, mk]
        return float(v) if not pd.isna(v) else np.nan

    def td(v):
        style = cfn(v, mx)
        txt = (fmt.format(v)) if not np.isnan(v) and v != 0 else ""
        return f'<td style="{style}">{txt}</td>'

    if len(strikes) >= 2:
        gaps = [abs(strikes[i]-strikes[i+1]) for i in range(len(strikes)-1)]
        atm_tol = min(gaps) * 0.6
    else:
        atm_tol = 1.0

    body = []
    for s in strikes:
        is_atm = atm is not None and abs(s - atm) < atm_tol
        sc = "sc sc-atm" if is_atm else "sc"
        tr_cls = ' class="atm-row"' if is_atm else ""
        lbl_s = int(s) if s == int(s) else s
        row = ("".join(td(cv(cpiv, s, mk)) for mk in ccols) + f'<td class="{sc}">{lbl_s}</td>'
               + "".join(td(cv(ppiv, s, mk)) for mk in pcols))
        body.append(f"<tr{tr_cls}>{row}</tr>")

    ft = ""
    if footer:
        def cs(piv, mk):
            if piv.empty or mk not in piv.columns or piv[mk].notna().sum() == 0:
                return float("nan")
            return float(piv[mk].sum(skipna=True))
        cft = "".join(td(cs(cpiv, mk)) for mk in ccols)
        pft = "".join(td(cs(ppiv, mk)) for mk in pcols)
        ft = f'<tfoot><tr>{cft}<td class="sc" style="font-size:9px;color:#888">TOT</td>{pft}</tr></tfoot>'

    est_h = max(400, (len(strikes) + 4) * 22 + 90)
    return (f'{_BFY_CSS}<div style="overflow-x:auto;overflow-y:auto;max-height:{est_h}px">'
            f'<table class="bft"><thead>{h1}{h2}{h3}</thead>{ft}<tbody>{"".join(body)}</tbody></table></div>')

# ── Rollex seasonality heatmap — verbatim port of ICE_Rollex.py's _simple_heatmap
#    (Monthly Returns Heatmap + Monthly Realized Vol Heatmap share this) ───────
RX_RET_CS = [[0.0, "#c0392b"], [0.45, "rgba(255,200,200,0.4)"], [0.5, "#f8f8f8"],
             [0.55, "rgba(200,235,200,0.4)"], [1.0, "#1a7a1a"]]
RX_VOL_CS = [[0.0, "#f0f0f0"], [0.01, "#1a7a1a"], [0.5, "#f8f8f8"], [1.0, "#c0392b"]]

def _rx_simple_heatmap(year_pivot, fmt_val, fmt_stat, colorscale, zmin, zmax, zmid, hover_suffix):
    piv = year_pivot.sort_index(ascending=True).copy()
    avg_row, std_row = piv.mean(skipna=True), piv.std(skipna=True)
    icv_row = avg_row / std_row
    y_labels = [str(y) for y in piv.index] + [""] + ["Avg", "Std", "ICV"]
    z_all, t_all = [], []
    for _, row in piv.iterrows():
        z_all.append([float(v) if pd.notna(v) else None for v in row])
        t_all.append([fmt_val(v) if pd.notna(v) else "" for v in row])
    z_all.append([None] * len(MONTHS)); t_all.append([""] * len(MONTHS))
    for lbl_s, row in zip(["Avg", "Std", "ICV"], [avg_row, std_row, icv_row]):
        z_all.append([0.0] * len(MONTHS))
        t_all.append([f"<b>{fmt_stat(lbl_s, v)}</b>" if lbl_s == "ICV" and pd.notna(v)
                     else (fmt_stat(lbl_s, v) if pd.notna(v) else "") for v in row])
    fig = go.Figure(go.Heatmap(
        z=z_all, x=MONTHS, y=y_labels, colorscale=colorscale, zmid=zmid, zmin=zmin, zmax=zmax,
        text=t_all, texttemplate="%{text}", textfont=dict(size=8.5),
        hovertemplate="%{y}  %{x}: <b>%{z:.2f}" + hover_suffix + "</b><extra></extra>",
        showscale=False, xgap=1, ygap=1))
    fig.update_layout(height=max(300, len(y_labels) * 24 + 80), margin=dict(t=35, b=8, l=50, r=4),
                      xaxis=dict(tickfont=dict(size=9), side="top", showgrid=False, showticklabels=False),
                      yaxis=dict(tickfont=dict(size=9), showgrid=False, autorange=True), **_D)
    for month in MONTHS:
        fig.add_annotation(x=month, y=1.02, xref="x", yref="paper", text=f"<b>{month}</b>",
                          showarrow=False, font=dict(size=9, color="#1d1d1f"), align="center")
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

# ── All Rollex commodities — for cross-commodity views (Correlation Matrix,
#    Roll Yield Ranking, COT Z-Score Matrix) that compare across the full set
#    rather than just the current page's commodity's legs ────────────────────
ALL_ROLLEX_COMMS  = ["KC", "RC", "CC", "LCC", "SB", "CT", "LSU"]
ALL_ROLLEX_NAMES  = {"KC":"KC — Arabica","RC":"RC — Robusta","CC":"CC — Cocoa (ICE)",
                     "LCC":"LCC — Cocoa (Liffe)","SB":"SB — Sugar #11","CT":"CT — Cotton",
                     "LSU":"LSU — White Sugar"}

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

# ── COT (Disaggregated, Futures-only — same basis the COT_ALL matrix uses so
#    every commodity, including RC which has no CIT report, sits on equal footing)
DISAGG_SPEC = {
    "Managed Money": {"long":"MM Long",       "short":"MM Short",       "net":"MM Net"},
    "Other Rept":    {"long":"Other Long",    "short":"Other Short",    "net":"Other Net"},
    "Non-Rep":       {"long":"Non Rep Long",  "short":"Non Rep Short",  "net":"Non Rep Net"},
    "Swap Dealers":  {"long":"Swap Long",     "short":"Swap Short",     "net":"Swap Net"},
    "MM + Other + Non-Rep": {"long":"MM+Other+NonRep Long","short":"MM+Other+NonRep Short","net":"MM+Other+NonRep Net"},
    "MM + Non-Rep":         {"long":"MM+NonRep Long",      "short":"MM+NonRep Short",      "net":"MM+NonRep Net"},
    "Commercial (Producer)": {"long":"Producer Long", "short":"Producer Short", "net":"Comm Net"},
}
COT_LOOKBACKS = [1, 3, 5, 10]

def _cot_derive_nets(df):
    pairs = [("MM Long","MM Short","MM Net"), ("Swap Long","Swap Short","Swap Net"),
             ("Other Long","Other Short","Other Net"), ("Producer Long","Producer Short","Comm Net")]
    for l, s, n in pairs:
        if l in df.columns and s in df.columns and n not in df.columns:
            df[n] = df[l] - df[s]
    return df

def _cot_add_pct(df):
    for col in list(df.columns):
        pct_col = f"Pct OI {col}"
        if (pct_col not in df.columns and "Total OI" in df.columns
                and col not in ("Date", "Commodity", "Crop") and not col.startswith(("Traders","Conc","Pct OI"))):
            try:
                df[pct_col] = (df[col] / df["Total OI"] * 100).round(2)
            except Exception:
                pass
    return df

@st.cache_data(ttl=1800)
def load_cot_disagg() -> pd.DataFrame:
    df = pd.read_parquet(DB / "cot_disagg_fut.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    num = [c for c in df.columns if c not in ("Date", "Commodity", "Crop")]
    df[num] = df[num].astype(float)
    df = _cot_derive_nets(df)
    for side in ("Long", "Short"):
        df[f"MM+Other+NonRep {side}"] = df.get(f"MM {side}", 0) + df.get(f"Other {side}", 0) + df.get(f"Non Rep {side}", 0)
    df["MM+Other+NonRep Net"] = df["MM+Other+NonRep Long"] - df["MM+Other+NonRep Short"]
    for side in ("Long", "Short"):
        df[f"MM+NonRep {side}"] = df.get(f"MM {side}", 0) + df.get(f"Non Rep {side}", 0)
    df["MM+NonRep Net"] = df["MM+NonRep Long"] - df["MM+NonRep Short"]
    df = _cot_add_pct(df)
    return df.sort_values(["Commodity", "Crop", "Date"]).reset_index(drop=True)

def _cot_zscore(series: pd.Series, years: int) -> float:
    if series.empty:
        return np.nan
    cutoff = series.index.max() - pd.DateOffset(years=years)
    window = series[series.index >= cutoff]
    if len(window) < 5 or window.std(ddof=0) == 0 or pd.isna(window.std(ddof=0)):
        return np.nan
    return float((series.iloc[-1] - window.mean()) / window.std(ddof=0))

def _cot_style_z(v):
    if pd.isna(v):
        return ""
    v = max(-3, min(3, v))
    if v >= 0:
        r, g, b = 255 - int(v/3*105), 235 - int(v/3*20), 130
    else:
        r, g, b = 250, 150 + int((v+3)/3*85), 120 + int((v+3)/3*60)
    return f"background-color:rgb({r},{g},{b});color:#1a1a2e"

# ── COT Recap — verbatim port of cot_app.py's _build_recap_df + _recap_html
#    (Disagg branch only, since this project only ingests the Disagg report) ──
_RECAP_GROUP_BG = {
    "Gross Positions": "#d1d5db", "NET": "#bae6fd", "SP": "#fed7aa",
    "MM+O+NR": "#a7f3d0", "OI": "#e5e7eb", "Rollex Px": "#fef3c7",
}
_RECAP_GROUP_TEXT = {}
_RECAP_COL_SUBSEP = {
    ("Gross Positions", "Non-Rep Long"), ("Gross Positions", "Swap Long"),
    ("Gross Positions", "Comm Long"), ("MM+O+NR", "Long"),
}
_COLUMN_TOOLTIPS = {
    ("MM+O+NR", "Long"): "MM Long + Other Long + Non-Rep Long",
    ("MM+O+NR", "Short"): "MM Short + Other Short + Non-Rep Short",
    ("NET", "MM+O+NR"): "MM Net + Other Net + Non-Rep Net",
    ("NET", "Rest"): "Other Net + Non-Rep Net",
    ("NET", "MM"): "Managed Money Net",
    ("NET", "Comm"): "Producer/Commercial Net",
}
_RECAP_CSS = """
<style>
.rtbl{border-collapse:collapse;font-size:.67rem;width:100%;font-family:-apple-system,sans-serif}
.rtbl th,.rtbl td{border:1px solid #e5e7eb;padding:2px 4px;text-align:center}
.rtbl td{white-space:nowrap}
.rtbl .grp{text-align:center;font-weight:700;font-size:.64rem;letter-spacing:.02em;white-space:normal;max-width:60px}
.rtbl .idx{text-align:left;font-weight:600;color:#374151;background:#f9fafb;min-width:52px;white-space:nowrap}
.rtbl .sub{background:#f9fafb;font-size:.60rem;color:#555;font-weight:600;text-align:center;white-space:normal;max-width:48px;line-height:1.25}
.rtbl tbody tr:hover td{background:#f0f9ff!important}
.rpos{color:#16a34a}.rneg{color:#dc2626}
.rtbl .gsep{box-shadow:inset 3px 0 0 #6b7280}
.rtbl .gsub{box-shadow:inset 1.5px 0 0 #b8c0cc}
</style>
"""

def _recap_html(df, signed=False, change_table=False, scroll=False, signed_groups=None,
                pct_groups=None, pct_subcols=None, signed_rows=None, z_rows=None, max_height=None):
    if df.empty: return ""
    cols = list(df.columns)
    groups, prev = [], None
    for c in cols:
        g = c[0]
        if g == prev: groups[-1][1] += 1
        else: groups.append([g, 1]); prev = g
    col_sep = []
    ci = 0
    for g, span in groups:
        for j in range(span):
            c = cols[ci + j]
            if j == 0: col_sep.append("gsep")
            elif c in _RECAP_COL_SUBSEP: col_sep.append("gsub")
            else: col_sep.append("")
        ci += span
    h1 = '<tr><th class="idx sub"></th>'
    for g, span in groups:
        bg = _RECAP_GROUP_BG.get(g, "#f9fafb")
        fg = _RECAP_GROUP_TEXT.get(g, "#111827")
        h1 += f'<th colspan="{span}" class="grp" style="background:{bg};color:{fg};box-shadow:inset 3px 0 0 #6b7280">{g}</th>'
    h1 += '</tr>'
    h2 = '<tr><th class="idx sub"></th>'
    for i, c in enumerate(cols):
        g = c[0]
        sep_cls = col_sep[i]
        tip = _COLUMN_TOOLTIPS.get(c)
        label = f'{c[1]} ⓘ' if tip else c[1]
        fsz = ";font-size:.62rem" if len(c[1]) > 9 else ""
        cls_str = f"sub {sep_cls}".strip()
        if g in _RECAP_GROUP_TEXT:
            bg = _RECAP_GROUP_BG.get(g, "#f9fafb"); fg = _RECAP_GROUP_TEXT[g]
            h2 += f'<th class="{cls_str}" style="background:{bg};color:{fg}{fsz}" title="{tip or ""}">{label}</th>'
        else:
            h2 += f'<th class="{cls_str}" style="font-size:.62rem" title="{tip or ""}">{label}</th>'
    h2 += '</tr>'
    body = ""
    for idx, row in df.iterrows():
        body += f'<tr><td class="idx">{idx}</td>'
        for i, c in enumerate(cols):
            sep_cls = col_sep[i]
            v = row[c]
            if pd.isna(v): body += f'<td class="{sep_cls}">—</td>'; continue
            is_z_row = z_rows and idx in z_rows
            use_signed = (signed or change_table or (signed_rows and idx in signed_rows)
                         or (signed_groups and isinstance(c, tuple) and c[0] in signed_groups))
            use_pct = ((pct_groups and isinstance(c, tuple) and c[0] in pct_groups) or
                      (pct_subcols and isinstance(c, tuple) and c in pct_subcols))
            fmt = ".2f" if is_z_row else ".1f"
            if use_signed:
                txt = f"{v:+{fmt}}"; cls = "rpos" if v > 0 else ("rneg" if v < 0 else "")
            elif use_pct:
                txt = f"{v:.1f}%"; cls = ""
            else:
                txt = f"{v:{fmt}}"; cls = ""
            full_cls = f"{cls} {sep_cls}".strip()
            body += f'<td class="{full_cls}">{txt}</td>'
        body += '</tr>'
    scroll_style = f"overflow-x:auto;overflow-y:auto;max-height:{max_height}px;" if max_height is not None \
        else ("overflow-x:auto;overflow-y:auto;max-height:420px;" if scroll else "overflow-x:auto;")
    return (f'{_RECAP_CSS}<div style="{scroll_style}margin-bottom:6px">'
            f'<table class="rtbl"><thead>{h1}{h2}</thead><tbody>{body}</tbody></table></div>')

def _build_recap_df(d):
    """Disagg-only version of cot_app.py's _build_recap_df."""
    d = d.sort_values("Date", ascending=True).reset_index(drop=True)
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    def gc(name):
        return d[name].astype(float) if name in d.columns else pd.Series(0.0, index=d.index)
    cols = {}
    for src, dst in [("MM Long","MM Long"),("MM Short","MM Short"),("Other Long","Other Long"),
                     ("Other Short","Other Short"),("Non Rep Long","Non-Rep Long"),
                     ("Non Rep Short","Non-Rep Short"),("Swap Long","Swap Long"),
                     ("Swap Short","Swap Short"),("Producer Long","Comm Long"),
                     ("Producer Short","Comm Short")]:
        if src in d.columns: cols[("Gross Positions", dst)] = gc(src) / 1000
    cols[("MM+O+NR", "Long")]  = (gc("MM Long")  + gc("Other Long")  + gc("Non Rep Long"))  / 1000
    cols[("MM+O+NR", "Short")] = (gc("MM Short") + gc("Other Short") + gc("Non Rep Short")) / 1000
    cols[("NET", "MM")]      = gc("MM Net")   / 1000
    cols[("NET", "Rest")]    = (gc("Other Net") + gc("Non Rep Net")) / 1000
    cols[("NET", "MM+O+NR")] = (gc("MM Net") + gc("Other Net") + gc("Non Rep Net")) / 1000
    cols[("NET", "Swap")]    = gc("Swap Net")  / 1000
    cols[("NET", "Comm")]    = gc("Comm Net")  / 1000
    for src, dst in [("MM Spread","MM Spread"),("Other Spread","Other Spread"),("Swap Spread","Swap Spread")]:
        if src in d.columns: cols[("SP", dst)] = gc(src) / 1000
    cols[("OI", "Total OI")] = gc("Total OI") / 1000
    cols[("Rollex Px", "Level")] = gc("Px")

    body = pd.DataFrame(cols)
    body.index = pd.to_datetime(d["Date"])
    body = body.iloc[::-1]

    row_1w, row_4w = {}, {}
    for c in body.columns:
        if len(body) >= 2: row_1w[c] = body.iloc[0][c] - body.iloc[1][c]
        if len(body) >= 5: row_4w[c] = body.iloc[0][c] - body.iloc[4][c]
    px_lvl = body[("Rollex Px", "Level")]
    body[("Rollex Px", "Δ% 1w")] = px_lvl.pct_change(-1) * 100

    row_z, row_avg, row_min, row_max = {}, {}, {}, {}
    for c in body.columns:
        series = body[c].replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) >= 4:
            mu, sigma = series.mean(), series.std()
            row_z[c] = (series.iloc[0] - mu) / sigma if sigma > 0 else 0.0
            row_avg[c], row_min[c], row_max[c] = mu, series.min(), series.max()

    summary = pd.DataFrame([row_1w, row_4w, row_z, row_avg, row_min, row_max],
                           index=["Δ 1w", "Δ 1m", "Z-Score", "Avg", "Min", "Max"], columns=body.columns)
    summary[("Rollex Px", "Δ% 1w")] = np.nan
    if len(px_lvl) >= 2 and px_lvl.iloc[1] != 0:
        summary.loc["Δ 1w", ("Rollex Px", "Δ% 1w")] = (px_lvl.iloc[0] / px_lvl.iloc[1] - 1) * 100
    if len(px_lvl) >= 5 and px_lvl.iloc[4] != 0:
        summary.loc["Δ 1m", ("Rollex Px", "Δ% 1w")] = (px_lvl.iloc[0] / px_lvl.iloc[4] - 1) * 100
    body.index = [f"{dt.day}-{dt.strftime('%b-%y')}" for dt in body.index]
    return summary, body

# ── COT Recap Charts — verbatim port of cot_app.py's render_recap_charts
#    (Disagg branch, 12-panel grid; the "Roll Yield vs Positioning" scatter
#    section at the bottom of the source function was left out) ──────────────
C_LONG, C_SHORT, C_NET = "#16a34a", "#dc2626", "#1a56db"
_BASE = dict(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="-apple-system,BlinkMacSystemFont,'Helvetica Neue',sans-serif",
                      color="#1a1a1a", size=11))
_PT_DARK_GREEN, _PT_LIGHT_GREEN = "#1a6b1a", "#7dce7d"
_PT_DARK_RED, _PT_LIGHT_RED     = "#8b0000", "#f4a0a0"
_PT_AMBER, _PT_BLACK, _PT_NAVY  = "#e8a020", "#1d1d1f", "#0a2463"
CONTRACT_SIZE = {"KC": 37500, "RC": 10}
CONTRACT_UNIT = {"KC": "lbs", "RC": "MT"}

def _pt_label(text):
    return (f"<div style='background:{_PT_NAVY};padding:5px 13px;border-radius:5px;"
            f"margin-bottom:8px'><span style='font-size:.78rem;font-weight:500;"
            f"letter-spacing:.07em;text-transform:uppercase;color:#dde4f0'>{text}</span></div>")

# ── Currency (Coffee only — Cocoa branch of the source app not needed here) ────
CCY_COUNTRIES_ARABICA = ["Brazil", "Colombia", "Honduras", "Ethiopia", "Peru"]
CCY_COUNTRIES_ROBUSTA = ["Vietnam", "Brazil", "Indonesia", "Uganda", "India"]
CCY_WEIGHTS_ARABICA = {"Brazil":54.0,"Colombia":20.7,"Honduras":7.4,"Ethiopia":11.9,"Peru":5.9}
CCY_WEIGHTS_ROBUSTA = {"Vietnam":47.0,"Brazil":11.0,"Indonesia":17.0,"Uganda":16.0,"India":9.0}
CCY_COLORS_ARABICA  = {"Brazil":"#4a7fb5","Colombia":"#e8c96a","Honduras":"#82c982","Ethiopia":"#e89090","Peru":"#c9a0dc"}
CCY_COLORS_ROBUSTA  = {"Vietnam":"#e07b39","Brazil":"#4a7fb5","Indonesia":"#7ec8c0","Uganda":"#a0aad4","India":"#f4a460"}

@st.cache_data(ttl=1800)
def load_currency() -> pd.DataFrame:
    df = pd.read_parquet(DB / "currency_data.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

def _ccy_rebase(series):
    first = series.dropna().iloc[0] if not series.dropna().empty else 1
    return series / first * 100

# ── Sidebar — commodity chooser ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h3 style='font-family:\"Playfair Display\",Georgia,serif;color:#0a2463;"
        "font-weight:400;letter-spacing:-.01em;margin-bottom:1rem'>Landing Page</h3>",
        unsafe_allow_html=True,
    )
    commodity = st.selectbox("Commodity", list(COMMODITIES.keys()), label_visibility="collapsed")

cfg = COMMODITIES[commodity]
legs = list(cfg["legs"].keys())          # e.g. ["KC", "LRC"]
leg_colors = {legs[0]: NAVY, legs[1]: "#8b1a00"} if len(legs) > 1 else {legs[0]: NAVY}

tab_flat, tab_spread, tab_arb, tab_vol, tab_risk, tab_pos, tab_ccy = st.tabs(
    ["Flat", "Spread", "Arb", "Volatility", "Risk", "Positioning", "Currency"]
)

# ══════════════════════════════════════════════════════════════════════════════
# FLAT — ports of Rollex dashboard (Price&OI, Price&Vol, Indexed, Return Dist)
#        + Futures dashboard's self-contained "All Contracts Rolling Volume"
# ══════════════════════════════════════════════════════════════════════════════
with tab_flat:
    source_link("Rollex", "Futures OI")
    f_idx, f_pv, f_vol, f_flow, f_seas, f_corr, f_dist, f_grid = st.tabs(
        ["Indexed Performance", "Price & Vol", "Rolling Volume", "OI & Volume Flow",
         "Seasonality", "Correlation Matrix", "Return Distribution", "Comprehensive Grid"]
    )

    with f_grid:
        st.markdown(lbl(f"{commodity} — Daily OI & Volume by Contract Month"), unsafe_allow_html=True)
        grid_lookback = st.slider("Lookback (calendar days)", 30, 365, 90, step=10, key="grid_lookback")
        for leg in legs:
            st.markdown(f"**{leg}**")
            html = build_comprehensive_grid_html(cfg["futures_codes"][leg], grid_lookback)
            if html is None:
                st.info("No data in this window.")
            else:
                st.markdown(html, unsafe_allow_html=True)

    with f_pv:
        st.markdown(lbl(f"{commodity} — Rollex Price & Rolling Volatility"), unsafe_allow_html=True)
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
        st.markdown(lbl("Indexed Performance (Base=100) — All Commodities"), unsafe_allow_html=True)
        idx_all = {}
        for c in ALL_ROLLEX_COMMS:
            try:
                idx_all[c] = load_rollex(c).set_index("Date").sort_index()["Close"]
            except FileNotFoundError:
                continue
        idx_min = min(s.index.min() for s in idx_all.values()).date()
        idx_max = max(s.index.max() for s in idx_all.values()).date()
        idx_range = st.slider("Date range", min_value=idx_min, max_value=idx_max,
                              value=(idx_min, idx_max), key="idx_daterange")

        fig_idx = base_fig(height=460, yaxis_title="Indexed (Base=100)")
        for c, s in idx_all.items():
            s = s[(s.index.date >= idx_range[0]) & (s.index.date <= idx_range[1])]
            if s.empty:
                continue
            indexed = s / s.iloc[0] * 100
            fig_idx.add_trace(go.Scatter(x=indexed.index, y=indexed, name=c,
                                         line=dict(color=leg_colors.get(c, None), width=1.8)
                                         if c in leg_colors else dict(width=1.4)))
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

    with f_flow:
        st.markdown(lbl(f"{commodity} — Daily OI Change vs Volume"), unsafe_allow_html=True)
        flow_lookback = st.slider("Lookback (calendar days)", 30, 365, 180, step=15, key="flow_lookback")
        oi_mode = st.radio("OI Δ (scatter)", ["Signed", "Absolute"], horizontal=True, key="flow_scatter_mode")
        for leg in legs:
            st.markdown(f"**{leg}**")
            df_all = load_futures_full(cfg["futures_codes"][leg])
            tot_oi = df_all.groupby("Date")["open_interest"].sum(min_count=1)
            tot_vol = df_all.groupby("Date")["volume"].sum(min_count=1)
            flow = pd.concat([tot_oi.rename("oi"), tot_vol.rename("vol")], axis=1).sort_index()
            flow["oi_change"] = flow["oi"].diff()
            flow = flow.dropna(subset=["oi_change"])
            max_d = flow.index.max()
            flow = flow[flow.index >= max_d - pd.Timedelta(days=flow_lookback)]
            if flow.empty:
                st.info("No data in this window.")
                continue

            c1, c2 = st.columns([3, 2])
            with c1:
                bar_colors = [GREEN if v >= 0 else RED for v in flow["oi_change"]]
                fig_flow = make_subplots(specs=[[{"secondary_y": True}]])
                fig_flow.add_trace(go.Bar(x=flow.index, y=flow["oi_change"], name="OI Change",
                                          marker_color=bar_colors), secondary_y=False)
                fig_flow.add_trace(go.Bar(x=flow.index, y=flow["vol"], name="Volume",
                                          marker_color="rgba(150,150,150,0.45)"), secondary_y=True)
                fig_flow.update_layout(height=380, barmode="overlay",
                                       legend=dict(orientation="h", y=1.08, font=dict(size=8)),
                                       margin=dict(t=25, b=8, l=4, r=4), **_D)
                fig_flow.update_yaxes(title_text="OI Change", secondary_y=False, gridcolor="#f0f0f0")
                fig_flow.update_yaxes(title_text="Volume", secondary_y=True, showgrid=False)
                st.plotly_chart(fig_flow, use_container_width=True, key=f"flow_bar_{leg}")
            with c2:
                x_vals = flow["oi_change"] if oi_mode == "Signed" else flow["oi_change"].abs()
                if len(flow) > 2 and x_vals.std() > 0:
                    coeffs = np.polyfit(x_vals, flow["vol"], 1)
                    x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
                    y_line = coeffs[0] * x_line + coeffs[1]
                    r2 = x_vals.corr(flow["vol"]) ** 2
                else:
                    x_line = y_line = None
                    r2 = float("nan")
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(x=x_vals[:-1], y=flow["vol"].iloc[:-1], mode="markers",
                                            marker=dict(color=leg_colors[leg], size=5, opacity=0.5), name="History"))
                fig_sc.add_trace(go.Scatter(x=[x_vals.iloc[-1]], y=[flow["vol"].iloc[-1]], mode="markers",
                                            marker=dict(color=RED, size=11, symbol="star"), name="Latest"))
                if x_line is not None:
                    fig_sc.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
                                                line=dict(color=GREY, width=1.5, dash="dash"), name=f"Fit (R²={r2:.2f})"))
                fig_sc.update_layout(height=380, title=dict(text="OI Δ vs Volume", font=dict(size=11)),
                                     xaxis=dict(title=f"OI Change ({oi_mode})", gridcolor="#f0f0f0"),
                                     yaxis=dict(title="Volume", gridcolor="#f0f0f0"),
                                     legend=dict(orientation="h", y=1.15, font=dict(size=8)),
                                     margin=dict(t=25, b=8, l=4, r=4), **_D)
                st.plotly_chart(fig_sc, use_container_width=True, key=f"flow_sc_{leg}")

    with f_seas:
        st.markdown(lbl(f"{commodity} — Monthly Returns Heatmap"), unsafe_allow_html=True)
        for leg in legs:
            st.markdown(f"**{leg}**")
            rx = load_rollex(cfg["rollex_codes"][leg]).set_index("Date").sort_index()
            monthly_s = rx["Close"].resample("ME").last().pct_change() * 100
            monthly_s.index = monthly_s.index.to_period("M")
            m_df = monthly_s.reset_index()
            m_df.columns = ["Period", "Return"]
            m_df["Year"], m_df["Month"] = m_df["Period"].dt.year, m_df["Period"].dt.month
            ret_pivot = m_df.pivot_table(index="Year", columns="Month", values="Return")
            ret_pivot.columns = [MONTHS[m-1] for m in ret_pivot.columns]
            ret_pivot = ret_pivot.reindex(columns=MONTHS).dropna(how="all")
            abs_max = ret_pivot.abs().max().max()
            fig_heat = _rx_simple_heatmap(ret_pivot, fmt_val=lambda v: f"{v:.1f}%",
                                          fmt_stat=lambda l, v: f"{v:.1f}%" if l in ("Avg","Std") else f"{v:.1f}",
                                          colorscale=RX_RET_CS, zmin=-abs_max, zmax=abs_max, zmid=0,
                                          hover_suffix="%")
            st.plotly_chart(fig_heat, use_container_width=True, key=f"seas_ret_{leg}")

    with f_corr:
        st.markdown(lbl("Return Correlation Matrix — All Commodities"), unsafe_allow_html=True)
        corr_lookback = st.slider("Lookback (calendar days)", 90, 1800, 730, step=90, key="corr_lookback")
        ret_data = {}
        for c in ALL_ROLLEX_COMMS:
            try:
                rx = load_rollex(c).set_index("Date").sort_index()
                ret_data[c] = rx["Ret"]
            except FileNotFoundError:
                continue
        ret_matrix = pd.DataFrame(ret_data).dropna()
        if len(ret_matrix) < 10:
            st.info("Not enough overlapping history to build a correlation matrix.")
        else:
            cutoff = ret_matrix.index.max() - pd.Timedelta(days=corr_lookback)
            ret_matrix = ret_matrix[ret_matrix.index >= cutoff]
            corr_matrix = ret_matrix.corr()
            labels = [ALL_ROLLEX_NAMES.get(c, c).split("—")[0].strip() for c in corr_matrix.columns]
            arr = corr_matrix.to_numpy(dtype=float, copy=True)
            np.fill_diagonal(arr, np.nan)
            z_mat = [[None if np.isnan(v) else v for v in row] for row in arr]
            t_mat = [["" if v is None else f"{v:.2f}" for v in row] for row in z_mat]
            corr_cs = [[0.0, RED], [0.45, "rgba(255,200,200,0.4)"], [0.5, "#f8f8f8"],
                      [0.55, "rgba(200,235,200,0.4)"], [1.0, NAVY]]
            fig_mat = go.Figure(go.Heatmap(
                z=z_mat, x=labels, y=labels, colorscale=corr_cs, zmin=-1, zmax=1, zmid=0,
                text=t_mat, texttemplate="%{text}", textfont=dict(size=11),
                hovertemplate="%{y} / %{x}: <b>%{z:.3f}</b><extra></extra>",
                showscale=True, colorbar=dict(thickness=12, len=0.8, tickfont=dict(size=9)),
                xgap=2, ygap=2))
            fig_mat.update_layout(height=420, margin=dict(t=10, b=8, l=4, r=4),
                                  xaxis=dict(tickfont=dict(size=10), side="bottom", showgrid=False),
                                  yaxis=dict(tickfont=dict(size=10), showgrid=False, autorange="reversed"),
                                  **_D)
            st.plotly_chart(fig_mat, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SPREAD — verified ports of the Roll Yield dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tab_spread:
    source_link("Roll Yield")
    ry = load_roll_yield()
    curve_cols = [f"c{i}" for i in range(1, 9)]

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

    st.markdown(lbl(f"{commodity} — Roll Yield Heatmap (Monthly Avg)"), unsafe_allow_html=True)
    code_pick_hm = st.selectbox("Contract", cfg["ry_codes"], key="ry_heat_code")
    s = ry[ry["Commodity"] == code_pick_hm]
    year_month_heatmap(s, "Date", "Roll_Yield_1yr", "Avg Roll Yield", pct=True, zmid=0,
                       colorscale=[[0.0,"#8b0000"],[0.4,"#f5c6cb"],[0.5,"#ffffff"],[0.6,"#d4edda"],[1.0,"#1a6b1a"]],
                       key="ry_heatmap")

# ══════════════════════════════════════════════════════════════════════════════
# ARB — full verified port of the Arb dashboard's KC/RC spread section
# ══════════════════════════════════════════════════════════════════════════════
with tab_arb:
    source_link("Arb")
    if commodity == "Coffee":
        kc = load_arb_front("KC")["px1"]
        rc = load_arb_front("RC")["px1"]
        l1 = kc * KC_FACTOR   # KC in $/MT
        l2 = rc               # RC in $/MT
        spread = (l1 - l2).dropna()
        zscore_win = st.slider("Rolling / Z-score window (days)", 60, 504, 252, step=21, key="arb_win")
        mu, sig = spread.rolling(zscore_win).mean(), spread.rolling(zscore_win).std()
        z = (spread - mu) / sig

        st.markdown(lbl("Arabica Premium over Robusta ($/MT)"), unsafe_allow_html=True)
        fig_sp = base_fig(height=440, yaxis_title="$/MT")
        fig_sp.add_trace(go.Scatter(x=spread.index, y=mu + 2*sig, name="+2σ", line=dict(color=RED, width=1, dash="dot")))
        fig_sp.add_trace(go.Scatter(x=spread.index, y=mu + sig, name="+1σ", line=dict(color=AMBER, width=1, dash="dash")))
        fig_sp.add_trace(go.Scatter(x=spread.index, y=mu, name="Mean", line=dict(color=GREY, width=1.5)))
        fig_sp.add_trace(go.Scatter(x=spread.index, y=mu - sig, name="-1σ", line=dict(color=AMBER, width=1, dash="dash")))
        fig_sp.add_trace(go.Scatter(x=spread.index, y=mu - 2*sig, name="-2σ", line=dict(color=GREEN, width=1, dash="dot")))
        fig_sp.add_trace(go.Scatter(x=spread.index, y=spread, name="Spread", line=dict(color=NAVY, width=2)))
        st.plotly_chart(fig_sp, use_container_width=True)

        st.markdown(lbl(f"Spread Z-Score ({zscore_win}d rolling)"), unsafe_allow_html=True)
        fig_z = base_fig(height=380, yaxis_title="Z-Score")
        fig_z.add_trace(go.Scatter(x=z.index, y=z, line=dict(color=NAVY, width=1.5), name="Z-Score"))
        fig_z.add_hline(y=0, line_color=GREY, line_width=1)
        fig_z.update_layout(yaxis=dict(range=[-4, 4], gridcolor="#f0f0f0"))
        st.plotly_chart(fig_z, use_container_width=True)

        st.markdown(lbl("Individual Legs ($/MT)"), unsafe_allow_html=True)
        fig_legs = base_fig(height=420, yaxis_title="$/MT")
        fig_legs.add_trace(go.Scatter(x=l1.index, y=l1, name="KC ($/MT)", line=dict(color=NAVY, width=1.5)))
        fig_legs.add_trace(go.Scatter(x=l2.index, y=l2, name="RC ($/MT)", line=dict(color=AMBER, width=1.5)))
        st.plotly_chart(fig_legs, use_container_width=True)

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

        st.markdown(lbl("KC/RC Price Ratio (Arabica/Robusta)"), unsafe_allow_html=True)
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
    source_link("Options", "Rollex")
    v_ivrv, v_rvseas, v_bfy = st.tabs(
        ["IV vs RV (Simplified)", "RV Seasonality", "OI Change + Volume Butterfly"]
    )

    with v_ivrv:
        st.markdown(lbl(f"{commodity} — Implied vs Realized Vol (simplified)"), unsafe_allow_html=True)

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

    with v_bfy:
        st.markdown(lbl(f"{commodity} — OI Change + Volume Butterfly"), unsafe_allow_html=True)
        leg_pick_b = st.selectbox("Leg", legs, key="bfy_leg")
        try:
            odf = load_options(leg_pick_b)
        except FileNotFoundError:
            odf = pd.DataFrame()
        if odf.empty:
            st.info(f"No options data available for {leg_pick_b}.")
        else:
            all_dates = sorted(odf["date"].dt.date.unique())
            default_old = all_dates[-1 - min(5, len(all_dates)-1)]
            c1, c2 = st.columns(2)
            with c1:
                old_date = st.select_slider("Old Date", options=all_dates, value=default_old, key="bfy_old_date")
            with c2:
                new_date = st.select_slider("New Date", options=all_dates, value=all_dates[-1], key="bfy_new_date")
            if old_date > new_date:
                old_date, new_date = new_date, old_date

            front = load_front_price(cfg["futures_codes"][leg_pick_b])
            price = float(front["settlement"].dropna().iloc[-1])
            all_strikes_data = sorted(odf["strike"].dropna().unique())
            if len(all_strikes_data) > 1:
                diffs = [all_strikes_data[i+1] - all_strikes_data[i] for i in range(len(all_strikes_data)-1)]
                step = sorted(diffs)[len(diffs)//2]
            else:
                step = 1.0
            atm = round(price / step) * step if step > 0 else price
            snap_tol = step / 2
            N = 35
            all_strikes = [round(atm + i*step, 6) for i in range(-N, N+1) if atm + i*step > 0]

            month_keys = _bfy_month_keys(odf)
            call_oi  = _bfy_change_pivot(odf, month_keys, "Call", "oi", old_date, new_date)
            put_oi   = _bfy_change_pivot(odf, month_keys, "Put",  "oi", old_date, new_date)
            call_vol = _bfy_vol_pivot(odf, month_keys, "Call", old_date, new_date)
            put_vol  = _bfy_vol_pivot(odf, month_keys, "Put",  old_date, new_date)

            c_oi, p_oi   = _bfy_tot(call_oi),  _bfy_tot(put_oi)
            c_vol, p_vol = _bfy_tot(call_vol), _bfy_tot(put_vol)
            def _bfy_fn(v):
                return "—" if pd.isna(v) else f"{v:,.0f}"
            cp_oi = (f"{abs(c_oi/p_oi):.2f}" if p_oi and not np.isnan(p_oi) and p_oi != 0 and not np.isnan(c_oi) else "—")

            items = [("ATM Price", f"{atm:,.2f}"), ("Old Date", old_date.strftime("%d %b %Y")),
                     ("New Date", new_date.strftime("%d %b %Y")), ("Call OI Δ", _bfy_fn(c_oi)),
                     ("Put OI Δ", _bfy_fn(p_oi)), ("Call Volume", _bfy_fn(c_vol)),
                     ("Put Volume", _bfy_fn(p_vol)), ("C/P OI Ratio", cp_oi)]
            st.markdown('<div style="display:flex;gap:24px;padding:6px 0 12px;border-bottom:1px solid #eee;flex-wrap:wrap">'
                       + "".join(f'<div><div style="font-size:9px;color:#888;letter-spacing:.07em;'
                                f'text-transform:uppercase;margin-bottom:2px">{k}</div>'
                                f'<div style="font-size:14px;font-weight:600;color:#1a1a2e">{v}</div></div>'
                                for k, v in items) + '</div>', unsafe_allow_html=True)

            cl, cr = st.columns(2)
            with cl:
                st.markdown("**OI Change**")
                st.markdown(bfy_butterfly_html(call_oi, put_oi, atm, _bfy_oi_color, month_keys,
                                              fmt="{:.0f}", footer=True, title=leg_pick_b,
                                              fixed_strikes=all_strikes, snap_tol=snap_tol),
                           unsafe_allow_html=True)
            with cr:
                st.markdown("**Volume**")
                st.markdown(bfy_butterfly_html(call_vol, put_vol, atm, _bfy_vol_color, month_keys,
                                              fmt="{:.0f}", footer=True, title=leg_pick_b,
                                              fixed_strikes=all_strikes, snap_tol=snap_tol),
                           unsafe_allow_html=True)

    with v_rvseas:
        st.markdown(lbl(f"{commodity} — Monthly Realized Volatility Heatmap"), unsafe_allow_html=True)
        rv_window = st.radio("Window", ["20d", "60d", "120d"], horizontal=True, key="rvseas_window")
        win = {"20d": 20, "60d": 60, "120d": 120}[rv_window]
        for leg in legs:
            st.markdown(f"**{leg}**")
            rx = load_rollex(cfg["rollex_codes"][leg]).set_index("Date").sort_index()
            rx["rv"] = rx["Ret"].rolling(win).std() * np.sqrt(252) * 100
            rv_monthly = rx["rv"].resample("ME").last()
            rv_monthly.index = rv_monthly.index.to_period("M")
            rv_df = rv_monthly.reset_index()
            rv_df.columns = ["Period", "RV"]
            rv_df["Year"], rv_df["Month"] = rv_df["Period"].dt.year, rv_df["Period"].dt.month
            rv_pivot = rv_df.pivot_table(index="Year", columns="Month", values="RV")
            rv_pivot.columns = [MONTHS[m-1] for m in rv_pivot.columns]
            rv_pivot = rv_pivot.reindex(columns=MONTHS).dropna(how="all")
            if rv_pivot.empty:
                st.info("Not enough history for this window.")
                continue
            rv_max, rv_min = float(rv_pivot.max().max()), float(rv_pivot.min().min())
            fig_rv_heat = _rx_simple_heatmap(rv_pivot, fmt_val=lambda v: f"{v:.1f}",
                                             fmt_stat=lambda l, v: f"{v:.1f}", colorscale=RX_VOL_CS,
                                             zmin=rv_min, zmax=rv_max, zmid=None, hover_suffix="%")
            st.plotly_chart(fig_rv_heat, use_container_width=True, key=f"rvseas_{leg}")

# ══════════════════════════════════════════════════════════════════════════════
# RISK — verified ports of the VaR project's Parametric VaR tab
# ══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    source_link("VaR")
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

# ══════════════════════════════════════════════════════════════════════════════
# POSITIONING — verified port of the COT_ALL spec_distribution_app's two tabs
# ══════════════════════════════════════════════════════════════════════════════
with tab_pos:
    source_link("COT Comprehensive", "COT Distribution")
    if commodity == "Coffee":
        p_recap, p_recap_ch, p_pain, p_matrix, p_dist = st.tabs(
            ["Recap", "Recap (Charts)", "Pain Trade Monitor", "Z-Score Matrix", "Distribution"]
        )
        cot = load_cot_disagg()

        with p_recap:
            st.markdown(lbl(f"{commodity} — COT Recap"), unsafe_allow_html=True)
            leg_pick_recap0 = st.selectbox("Leg", legs, key="recap_leg0")
            cot_code_r0 = "RC" if leg_pick_recap0 == "LRC" else leg_pick_recap0
            d_recap = cot[(cot["Commodity"] == cot_code_r0) & (cot["Crop"] == "All")]
            if d_recap.empty:
                st.warning("No data for the selected leg.")
            else:
                summary, body = _build_recap_df(d_recap)
                if body.empty:
                    st.warning("No data.")
                else:
                    view = body.iloc[:20]
                    _PX_PCT = {("Rollex Px", "Δ% 1w")}
                    with st.expander("Change summary  ·  k lots", expanded=True):
                        st.markdown(_recap_html(summary, signed_rows={"Δ 1w", "Δ 1m", "Z-Score"},
                                                z_rows={"Z-Score"}, pct_subcols=_PX_PCT, max_height=148),
                                   unsafe_allow_html=True)
                    with st.expander("Historical positions  ·  k lots", expanded=True):
                        st.markdown(_recap_html(view, scroll=True, pct_subcols=_PX_PCT), unsafe_allow_html=True)
                    with st.expander("Weekly change  ·  k lots", expanded=True):
                        chg = view.diff(-1)
                        st.markdown(_recap_html(chg, signed=True, change_table=True, scroll=True,
                                                pct_subcols=_PX_PCT), unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(lbl(f"{commodity} — Quick Recap, All Categories"), unsafe_allow_html=True)
            recap_rows = []
            for cat_name, cat_cols in DISAGG_SPEC.items():
                row = {"Category": cat_name}
                for leg in legs:
                    cot_code = "RC" if leg == "LRC" else leg
                    d = cot[(cot["Commodity"] == cot_code) & (cot["Crop"] == "All")].sort_values("Date")
                    net_col = cat_cols["net"]
                    if net_col not in d.columns:
                        row[f"{leg} Net"] = row[f"{leg} 1wk Δ"] = row[f"{leg} Z (3y)"] = None
                        continue
                    s = pd.to_numeric(d.set_index("Date")[net_col], errors="coerce").dropna() / 1000.0
                    if len(s) < 2:
                        row[f"{leg} Net"] = row[f"{leg} 1wk Δ"] = row[f"{leg} Z (3y)"] = None
                        continue
                    row[f"{leg} Net"] = round(s.iloc[-1], 1)
                    row[f"{leg} 1wk Δ"] = round(s.iloc[-1] - s.iloc[-2], 1)
                    row[f"{leg} Z (3y)"] = round(_cot_zscore(s, 3), 2)
                recap_rows.append(row)
            recap_df = pd.DataFrame(recap_rows)
            st.dataframe(recap_df, use_container_width=True, hide_index=True)

            st.markdown("**Net Positioning — History**")
            leg_pick_recap = st.selectbox("Leg", legs, key="recap_leg")
            cat_pick_recap = st.selectbox("Category", list(DISAGG_SPEC.keys()), key="recap_cat")
            cot_code_r = "RC" if leg_pick_recap == "LRC" else leg_pick_recap
            d_r = cot[(cot["Commodity"] == cot_code_r) & (cot["Crop"] == "All")].sort_values("Date")
            net_col_r = DISAGG_SPEC[cat_pick_recap]["net"]
            if net_col_r in d_r.columns:
                s_r = pd.to_numeric(d_r.set_index("Date")[net_col_r], errors="coerce").dropna() / 1000.0
                fig_net = base_fig(height=380, yaxis_title="Net (k lots)")
                fig_net.add_trace(go.Scatter(x=s_r.index, y=s_r.values, mode="lines",
                                             line=dict(color=leg_colors[leg_pick_recap], width=1.8),
                                             fill="tozeroy", fillcolor="rgba(26,86,219,0.08)"))
                fig_net.add_hline(y=0, line_color="#cccccc", line_width=1)
                st.plotly_chart(fig_net, use_container_width=True)
            else:
                st.info("No data for this leg/category.")

        with p_recap_ch:
            st.markdown(lbl(f"{commodity} — COT Recap Charts"), unsafe_allow_html=True)
            leg_pick_rc = st.selectbox("Leg", legs, key="recap_ch_leg")
            cot_code_rc = "RC" if leg_pick_rc == "LRC" else leg_pick_rc
            d_rc = cot[(cot["Commodity"] == cot_code_rc) & (cot["Crop"] == "All")].sort_values("Date").reset_index(drop=True)
            if d_rc.empty:
                st.warning("No data for the selected leg.")
            else:
                dates_rc = pd.to_datetime(d_rc["Date"])
                def gc_rc(name):
                    return d_rc[name].astype(float) if name in d_rc.columns else pd.Series(np.nan, index=d_rc.index)
                size = CONTRACT_SIZE.get(cot_code_rc, 1)
                unit = CONTRACT_UNIT.get(cot_code_rc, "MT")
                px_rc = gc_rc("Px")
                mult = (px_rc * size / 100 / 1_000_000) if unit == "lbs" else (px_rc * size / 1_000_000)
                oi_rc = gc_rc("Total OI").replace(0, np.nan)

                def _rc_line(title, series_dict, clrs=None):
                    dflt = [C_LONG, C_SHORT, C_NET, "#f59e0b", "#7c3aed"]
                    clrs = clrs or dflt
                    fig = go.Figure()
                    fig.update_layout(**_BASE, title=dict(text=f"{cot_code_rc} — {title}", font=dict(size=10, color="#374151")),
                                      height=260, margin=dict(l=40, r=8, t=36, b=48), showlegend=True,
                                      legend=dict(orientation="h", y=-0.28, font=dict(size=9)),
                                      xaxis=dict(showgrid=False, tickangle=-35, nticks=20),
                                      yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)"))
                    for i, (name, y) in enumerate(series_dict.items()):
                        fig.add_trace(go.Scatter(x=dates_rc, y=y, name=name, line=dict(color=clrs[i % len(clrs)], width=1.5)))
                    return fig

                mm_net, swap_net = gc_rc("MM Net"), gc_rc("Swap Net")
                rc_cols = st.columns(3)
                panels = [
                    ("MM Gross k lots", {"MM Long": gc_rc("MM Long")/1000, "MM Short": gc_rc("MM Short")/1000}, [C_LONG, C_SHORT]),
                    ("MM Gross % of OI", {"MM Long %": gc_rc("MM Long")/oi_rc*100, "MM Short %": gc_rc("MM Short")/oi_rc*100}, [C_LONG, C_SHORT]),
                    (f"MM Nominal M USD", {"MM Long": gc_rc("MM Long")*mult, "MM Short": gc_rc("MM Short")*mult}, [C_LONG, C_SHORT]),
                    ("Commercial Gross k lots", {"Prod Long": gc_rc("Producer Long")/1000, "Prod Short": gc_rc("Producer Short")/1000}, [C_LONG, C_SHORT]),
                    ("Commercial Gross % of OI", {"Prod Long %": gc_rc("Producer Long")/oi_rc*100, "Prod Short %": gc_rc("Producer Short")/oi_rc*100}, [C_LONG, C_SHORT]),
                    (f"Commercial Nominal M USD", {"Prod Long": gc_rc("Producer Long")*mult, "Prod Short": gc_rc("Producer Short")*mult}, [C_LONG, C_SHORT]),
                    ("Other Gross k lots", {"Other Long": gc_rc("Other Long")/1000, "Other Short": gc_rc("Other Short")/1000}, [C_LONG, C_SHORT]),
                    ("Other Gross % of OI", {"Other Long %": gc_rc("Other Long")/oi_rc*100, "Other Short %": gc_rc("Other Short")/oi_rc*100}, [C_LONG, C_SHORT]),
                    (f"Other Nominal M USD", {"Other Long": gc_rc("Other Long")*mult, "Other Short": gc_rc("Other Short")*mult}, [C_LONG, C_SHORT]),
                    ("MM Net & Swap Net & Other Net k lots", {"MM Net": mm_net/1000, "Swap Net": swap_net/1000, "Other Net": gc_rc("Other Net")/1000}, [C_NET, C_LONG, "#f59e0b"]),
                    ("# of Traders", {"MM Long": gc_rc("Traders MM Long"), "MM Short": gc_rc("Traders MM Short"),
                                      "Other Long": gc_rc("Traders Other Long"), "Other Short": gc_rc("Traders Other Short")},
                     [C_LONG, C_SHORT, "#f59e0b", "#7c3aed"]),
                    ("Other Spread k lots", {"Other Spread": gc_rc("Other Spread")/1000}, ["#f59e0b"]),
                ]
                for i, (title, series, clrs) in enumerate(panels):
                    with rc_cols[i % 3]:
                        st.plotly_chart(_rc_line(title, series, clrs), use_container_width=True, key=f"recap_ch_{i}")

        with p_pain:
            st.markdown(lbl(f"{commodity} — Pain Trade Monitor (first visual)"), unsafe_allow_html=True)
            leg_pick_pt = st.selectbox("Leg", legs, key="pain_leg")
            cot_code_pt = "RC" if leg_pick_pt == "LRC" else leg_pick_pt
            d_pt = cot[(cot["Commodity"] == cot_code_pt) & (cot["Crop"] == "All")].copy()
            if d_pt.empty:
                st.warning("No data for the selected leg.")
            else:
                incl = st.radio("Include Other Rept. in spec legs?",
                                ["Yes — MM + Non Rep + Other Rept.", "No — MM + Non Rep only"],
                                index=0, horizontal=True, key="pain_incl")
                use_third = incl.startswith("Yes")

                rx_daily = load_rollex(cfg["rollex_codes"][leg_pick_pt]).rename(columns={"Close": "Rollex"})
                df_pt = d_pt.sort_values("Date").reset_index(drop=True)
                df_pt["Rollex"] = pd.to_numeric(df_pt["Px"], errors="coerce")
                if "Total OI" in df_pt.columns:
                    oi_cap = pd.to_numeric(df_pt["Total OI"], errors="coerce").clip(lower=1)
                    for _pc in ["MM Long", "MM Short", "Other Long", "Other Short", "Non Rep Long", "Non Rep Short"]:
                        if _pc in df_pt.columns:
                            df_pt[_pc] = pd.to_numeric(df_pt[_pc], errors="coerce").clip(upper=oi_cap)

                if use_third:
                    gross_long  = (df_pt["MM Long"] + df_pt["Non Rep Long"] + df_pt["Other Long"]) / 1000
                    gross_short = (df_pt["MM Short"] + df_pt["Non Rep Short"] + df_pt["Other Short"]) / 1000
                    leg_label = "MM + Non Rep + Other Rept."
                else:
                    gross_long  = (df_pt["MM Long"] + df_pt["Non Rep Long"]) / 1000
                    gross_short = (df_pt["MM Short"] + df_pt["Non Rep Short"]) / 1000
                    leg_label = "MM + Non Rep"

                long_chg, short_chg = gross_long.diff(), gross_short.diff()
                df_pt["Long Add"]    =  long_chg.clip(lower=0)
                df_pt["Long Liq"]    =  long_chg.clip(upper=0)
                df_pt["Short Add"]   = -short_chg.clip(lower=0)
                df_pt["Short Cover"] = -short_chg.clip(upper=0)

                nw_opts = {"13w": 13, "26w": 26, "52w": 52}
                nw_sel = st.radio("Show last", list(nw_opts.keys()), index=0, horizontal=True, key="pain_nw")
                n_weeks = nw_opts[nw_sel]
                pt_max = df_pt["Date"].max()
                dff_pt = df_pt[df_pt["Date"] >= pt_max - pd.Timedelta(weeks=n_weeks)].copy()
                last_cot_date = dff_pt["Date"].max()
                last_cot_str = last_cot_date.strftime("%d/%m/%Y") if pd.notna(last_cot_date) else "—"
                latest_rx_str = rx_daily["Date"].max().strftime("%d/%m/%Y") if not rx_daily.empty else last_cot_str

                st.markdown(_pt_label(f"{cot_code_pt} — Spec Legs Weekly Change ({leg_label}) · Rollex (Right) "
                                     f"| COT as of {last_cot_str} · Rollex as of {latest_rx_str}"),
                           unsafe_allow_html=True)
                fig1 = make_subplots(specs=[[{"secondary_y": True}]])
                for col, c, name in [("Long Add", _PT_DARK_GREEN, "Long Add"), ("Long Liq", _PT_LIGHT_GREEN, "Long Liq."),
                                     ("Short Add", _PT_DARK_RED, "Short Add"), ("Short Cover", _PT_LIGHT_RED, "Short Cover")]:
                    fig1.add_trace(go.Bar(x=dff_pt["Date"], y=dff_pt[col], name=name, marker_color=c, opacity=0.92),
                                   secondary_y=False)
                rx_solid = dff_pt.dropna(subset=["Rollex"])
                fig1.add_trace(go.Scatter(x=rx_solid["Date"], y=rx_solid["Rollex"], name="Rollex (COT period)",
                                          mode="lines", line=dict(color=_PT_BLACK, width=2)), secondary_y=True)
                if not rx_solid.empty and not rx_daily.empty:
                    last_solid = rx_solid.iloc[-1:][["Date", "Rollex"]]
                    rx_after = rx_daily[rx_daily["Date"] > last_cot_date][["Date", "Rollex"]]
                    rx_ext = pd.concat([last_solid, rx_after]).sort_values("Date")
                    if len(rx_ext) > 1:
                        fig1.add_trace(go.Scatter(x=rx_ext["Date"], y=rx_ext["Rollex"],
                                                  name=f"Rollex post-COT ({latest_rx_str})", mode="lines",
                                                  line=dict(color=_PT_AMBER, width=2, dash="dot")), secondary_y=True)
                        last_pt = rx_ext.iloc[-1]
                        fig1.add_trace(go.Scatter(x=[last_pt["Date"]], y=[last_pt["Rollex"]], mode="markers+text",
                                                  marker=dict(color=_PT_AMBER, size=11, symbol="diamond",
                                                             line=dict(color=_PT_BLACK, width=1)),
                                                  text=[f"  {last_pt['Rollex']:.1f}"], textposition="middle right",
                                                  textfont=dict(size=10, color=_PT_AMBER), showlegend=False),
                                       secondary_y=True)
                x_left = dff_pt["Date"].min() - pd.Timedelta(days=2)
                x_right_anchor = rx_daily["Date"].max() if not rx_daily.empty else dff_pt["Date"].max()
                x_right = x_right_anchor + pd.Timedelta(days=5)
                fig1.update_layout(barmode="relative", height=420, margin=dict(t=10, b=10, l=4, r=4),
                                   legend=dict(orientation="h", y=1.06, x=0, font=dict(size=9)),
                                   xaxis=dict(showgrid=False, tickfont=dict(size=9), range=[x_left, x_right]),
                                   template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font=dict(family="-apple-system,Helvetica Neue,sans-serif", color=_PT_BLACK, size=10))
                fig1.update_yaxes(title_text="k Contracts", secondary_y=False, showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=9))
                fig1.update_yaxes(title_text="Rollex Price", secondary_y=True, showgrid=False, tickfont=dict(size=9))
                st.plotly_chart(fig1, use_container_width=True)

        with p_matrix:
            st.markdown(lbl("COT Z-Score Matrix — All Commodities, Disaggregated (Futures-only)"),
                       unsafe_allow_html=True)
            matrix_cat = st.selectbox("Category", list(DISAGG_SPEC.keys()), key="cot_matrix_cat")
            mcols = DISAGG_SPEC[matrix_cat]
            level_rows, chg_rows = {}, {}
            for cmm in cot["Commodity"].unique():
                d = cot[(cot["Commodity"] == cmm) & (cot["Crop"] == "All")].sort_values("Date")
                net_col = mcols["net"]
                if net_col not in d.columns or d.empty:
                    level_rows[cmm] = {y: np.nan for y in COT_LOOKBACKS}
                    chg_rows[cmm] = {y: np.nan for y in COT_LOOKBACKS}
                    continue
                level_s = pd.to_numeric(d.set_index("Date")[net_col], errors="coerce").dropna()
                chg_s = level_s.diff().dropna()
                level_rows[cmm] = {y: _cot_zscore(level_s, y) for y in COT_LOOKBACKS}
                chg_rows[cmm] = {y: _cot_zscore(chg_s, y) for y in COT_LOOKBACKS}
            level_df = pd.DataFrame(level_rows).T[COT_LOOKBACKS]
            chg_df = pd.DataFrame(chg_rows).T[COT_LOOKBACKS]

            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown(f"**{matrix_cat} Net — Z-score**")
                st.dataframe(level_df.style.map(_cot_style_z).format("{:.2f}"), use_container_width=True)
            with mc2:
                st.markdown(f"**{matrix_cat} Weekly Change — Z-score**")
                st.dataframe(chg_df.style.map(_cot_style_z).format("{:.2f}"), use_container_width=True)

        with p_dist:
            st.markdown(lbl(f"{commodity} — COT Positioning Distribution"), unsafe_allow_html=True)
            leg_pick = st.selectbox("Leg", legs, key="cot_dist_leg")
            cot_code = "RC" if leg_pick == "LRC" else leg_pick
            d = cot[(cot["Commodity"] == cot_code) & (cot["Crop"] == "All")].sort_values("Date")
            dist_cat = st.selectbox("Category", list(DISAGG_SPEC.keys()), key="cot_dist_cat")
            cols_cat = DISAGG_SPEC[dist_cat]
            lb_choice = st.radio("History window", ["All", "1y", "3y", "5y", "10y"],
                                 horizontal=True, key="cot_dist_lb")
            if lb_choice != "All":
                cutoff = d["Date"].max() - pd.DateOffset(years=int(lb_choice.replace("y", "")))
                d = d[d["Date"] >= cutoff]

            missing = [c for c in cols_cat.values() if c not in d.columns]
            if d.empty or missing:
                st.warning("No data available for this leg/category.")
            else:
                metrics = [("Net", cols_cat["net"], "#1a56db"), ("Long", cols_cat["long"], GREEN),
                          ("Short", cols_cat["short"], RED)]
                fig = make_subplots(rows=1, cols=3, subplot_titles=[m[0] for m in metrics])
                for i, (name, col, color) in enumerate(metrics, start=1):
                    s = (pd.to_numeric(d[col], errors="coerce").dropna() / 1000.0)
                    if s.empty:
                        continue
                    fig.add_trace(go.Histogram(x=s.values, marker_color=color, opacity=0.85,
                                               showlegend=False), row=1, col=i)
                    fig.add_vline(x=s.iloc[-1], line_dash="dash", line_color="#1a1a2e", row=1, col=i)
                fig.update_layout(height=380, bargap=0.04,
                                  yaxis=dict(title="Weeks (count)", gridcolor="#f0f0f0"),
                                  margin=dict(t=30, b=10, l=4, r=4), **_D)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Rollex Price — Weekly % Change**")
            rx = load_rollex(cfg["rollex_codes"][leg_pick])
            px_lvl = d[["Date"]].merge(rx.rename(columns={"Close": "rollex_px"}), on="Date", how="inner")
            px_chg = (px_lvl["rollex_px"].dropna().pct_change() * 100).dropna()
            if px_chg.empty:
                st.info("No overlapping weeks between COT dates and Rollex price data.")
            else:
                fig_px = go.Figure(go.Histogram(x=px_chg.values, marker_color=AMBER, opacity=0.85))
                fig_px.add_vline(x=px_chg.iloc[-1], line_color="#1a1a2e", line_width=2)
                fig_px.update_layout(height=320, xaxis=dict(title="Weekly Price Change %"),
                                     yaxis=dict(title="Weeks (count)", gridcolor="#f0f0f0"),
                                     margin=dict(t=15, b=10, l=4, r=4), **_D)
                st.plotly_chart(fig_px, use_container_width=True)
    else:
        st.info("No COT mapping for this commodity yet.")

# ══════════════════════════════════════════════════════════════════════════════
# CURRENCY — verified port of the Currency dashboard's Coffee branch
# ══════════════════════════════════════════════════════════════════════════════
with tab_ccy:
    source_link("Currency")
    if commodity == "Coffee":
        ccy = load_currency()
        min_d, max_d = ccy["Date"].min().date(), ccy["Date"].max().date()
        default_start = max(pd.Timestamp("2020-01-01").date(), min_d)
        d_start, d_end = st.slider("Date range", min_value=min_d, max_value=max_d,
                                   value=(default_start, max_d), key="ccy_daterange")
        dff = ccy[(ccy["Date"] >= pd.Timestamp(d_start)) & (ccy["Date"] <= pd.Timestamp(d_end))].copy()

        st.markdown(lbl("Producer-Country Currencies, Indexed to Start (=100)"), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Arabica Basket**")
            fig_ai = base_fig(height=340, yaxis_title="Indexed (start=100)")
            for c in CCY_COUNTRIES_ARABICA:
                fig_ai.add_trace(go.Scatter(x=dff["Date"], y=_ccy_rebase(dff[c]),
                                            name=f"{c} ({CCY_WEIGHTS_ARABICA[c]:.0f}%)",
                                            line=dict(color=CCY_COLORS_ARABICA.get(c,"#aaa"), width=1.4)))
            fig_ai.add_trace(go.Scatter(x=dff["Date"], y=_ccy_rebase(dff["Arabica_Idx"]), name="Arabica Index",
                                        line=dict(color=NAVY, width=2.5, dash="dash")))
            fig_ai.add_hline(y=100, line_color="#cccccc", line_dash="dot")
            st.plotly_chart(fig_ai, use_container_width=True)
        with c2:
            st.markdown("**Robusta Basket**")
            fig_ri = base_fig(height=340, yaxis_title="Indexed (start=100)")
            for c in CCY_COUNTRIES_ROBUSTA:
                fig_ri.add_trace(go.Scatter(x=dff["Date"], y=_ccy_rebase(dff[c]),
                                            name=f"{c} ({CCY_WEIGHTS_ROBUSTA[c]:.0f}%)",
                                            line=dict(color=CCY_COLORS_ROBUSTA.get(c,"#aaa"), width=1.4)))
            fig_ri.add_trace(go.Scatter(x=dff["Date"], y=_ccy_rebase(dff["Robusta_Idx"]), name="Robusta Index",
                                        line=dict(color="#8b1a00", width=2.5, dash="dash")))
            fig_ri.add_hline(y=100, line_color="#cccccc", line_dash="dot")
            st.plotly_chart(fig_ri, use_container_width=True)

        st.markdown(lbl("Currency Index vs Price"), unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        for col, idx_col, idx_name, price_col, price_label, idx_color in [
            (c3, "Arabica_Idx", "Arabica", "KC_Price", "KC Arabica (c/lb)", NAVY),
            (c4, "Robusta_Idx", "Robusta", "RC_Price", "RC Robusta ($/t)", "#8b1a00"),
        ]:
            with col:
                fig_p = make_subplots(specs=[[{"secondary_y": True}]])
                fig_p.add_trace(go.Scatter(x=dff["Date"], y=dff[idx_col], name=f"{idx_name} Idx",
                                           line=dict(color=idx_color, width=2)), secondary_y=False)
                fig_p.add_trace(go.Scatter(x=dff["Date"], y=dff[price_col], name=price_label,
                                           line=dict(color="#888", width=1.4, dash="dot")), secondary_y=True)
                fig_p.update_layout(height=280, title=dict(text=f"{idx_name} Index vs {price_label}", font=dict(size=11)),
                                    legend=dict(orientation="h", y=1.1, font=dict(size=8)),
                                    margin=dict(t=25, b=8, l=4, r=4), **_D)
                fig_p.update_yaxes(title_text=f"{idx_name} Index", secondary_y=False, gridcolor="#f0f0f0")
                fig_p.update_yaxes(title_text=price_label, secondary_y=True, showgrid=False)
                st.plotly_chart(fig_p, use_container_width=True, key=f"ccy_idxpx_{idx_name}")

        st.markdown(lbl("Arabica vs Robusta Currency Index"), unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        with c5:
            fig3 = base_fig(height=280)
            fig3.add_trace(go.Scatter(x=dff["Date"], y=dff["Arabica_Idx"], name="Arabica", line=dict(color=NAVY, width=2)))
            fig3.add_trace(go.Scatter(x=dff["Date"], y=dff["Robusta_Idx"], name="Robusta", line=dict(color="#8b1a00", width=2)))
            st.plotly_chart(fig3, use_container_width=True)
        with c6:
            st.markdown("**Spread: Arabica − Robusta Index**")
            fig4 = go.Figure(go.Scatter(x=dff["Date"], y=dff["Spread_Ara_Rob"], mode="lines",
                                        line=dict(color="#9b59b6", width=1.8), fill="tozeroy",
                                        fillcolor="rgba(155,89,182,0.07)"))
            fig4.add_hline(y=0, line_color="#cccccc")
            fig4.update_layout(height=280, margin=dict(t=10, b=10, l=4, r=4), **_D)
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown(lbl("Index vs Price — Correlation"), unsafe_allow_html=True)
        c7, c8 = st.columns(2)
        for col, idx_col, price_col, idx_name, price_name, color in [
            (c7, "Arabica_Idx", "KC_Price", "Arabica Index", "KC Arabica (c/lb)", NAVY),
            (c8, "Robusta_Idx", "RC_Price", "Robusta Index", "RC Robusta ($/t)", "#8b1a00"),
        ]:
            with col:
                valid = dff.dropna(subset=[idx_col, price_col])
                if valid.empty:
                    st.info(f"No data for {idx_name} vs {price_name}")
                    continue
                r = valid[idx_col].corr(valid[price_col])
                st.markdown(f"**{idx_name} vs {price_name} · r = {r:.3f}**")
                fig_sc = go.Figure(go.Scatter(x=valid[idx_col], y=valid[price_col], mode="markers",
                                              marker=dict(color=color, size=4, opacity=0.4)))
                z = np.polyfit(valid[idx_col], valid[price_col], 1)
                x_l = np.linspace(valid[idx_col].min(), valid[idx_col].max(), 100)
                fig_sc.add_trace(go.Scatter(x=x_l, y=np.polyval(z, x_l), mode="lines",
                                            line=dict(color="#cccccc", width=1.5, dash="dash")))
                fig_sc.update_layout(height=280, showlegend=False,
                                     xaxis=dict(title=idx_name, gridcolor="#f0f0f0"),
                                     yaxis=dict(title=price_name, gridcolor="#f0f0f0"),
                                     margin=dict(t=10, b=10, l=4, r=4), **_D)
                st.plotly_chart(fig_sc, use_container_width=True, key=f"ccy_scatter_{idx_name}")

        st.markdown(lbl("Latest Values & % Changes"), unsafe_allow_html=True)
        last_row, prev_row = dff.iloc[-1], dff.iloc[-2] if len(dff) >= 2 else dff.iloc[-1]
        y1_cut = dff[dff["Date"] <= (last_row["Date"] - pd.Timedelta(weeks=52))]
        y1_row = y1_cut.iloc[-1] if not y1_cut.empty else last_row
        all_countries = CCY_COUNTRIES_ARABICA + [c for c in CCY_COUNTRIES_ROBUSTA if c not in CCY_COUNTRIES_ARABICA]
        rows = []
        for c in all_countries + ["Arabica_Idx", "Robusta_Idx"]:
            if c not in dff.columns:
                continue
            label = c if c not in ("Arabica_Idx", "Robusta_Idx") else c.replace("_Idx", " Index")
            d1 = (last_row[c] - prev_row[c]) / prev_row[c] * 100 if prev_row[c] else 0
            y1 = (last_row[c] - y1_row[c]) / y1_row[c] * 100 if y1_row[c] else 0
            rows.append({"Name": label, "Latest": f"{last_row[c]:.4f}", "1D %": f"{d1:+.2f}%", "1Y %": f"{y1:+.1f}%"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No currency mapping for this commodity yet.")
