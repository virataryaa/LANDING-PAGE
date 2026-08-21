# Landing Page — Unified Commodity Dashboard

A single page per commodity, with one tab per exposure: **Flat | Spread |
Arb | Volatility | Risk**. This project computes nothing of its own — every
panel is built from parquets already produced by the Rollex, Futures, Arb,
Roll Yield, and Options projects.

**Why this project keeps its own `Database/` copy instead of reading those
projects' folders directly:** it deploys to Streamlit Cloud from its own git
repo, and Cloud only clones that one repo — it can't see the other projects'
repos or their `Database/` folders. So `Code/ingest.py` copies just the
files this app needs into `Database/` here (same reason and same pattern the
VaR project uses). Run it after the source projects' own daily updates, then
push, to refresh the deployed dashboard.

| Exposure | Source project | What it shows |
|---|---|---|
| Flat | Rollex + Futures | Continuous price and total open interest, per leg |
| Spread | Roll Yield | 1-year roll yield history + current c1→c8 forward curve |
| Arb | Arb | Cross-leg spread with rolling mean/±2σ bands (same calc as the standalone Arb dashboard) |
| Volatility | Options | Simplified near-the-money IV of the nearest expiry vs. 20-day realized vol |
| Risk | (recomputed inline) | Per-lot parametric VaR, same method as the VaR project (settlement × lot size × rolling vol × 2.3263), summed across legs for a combined read |

**Volatility is intentionally simplified** — it's a near-the-money average
of the nearest listed expiry, not the full per-expiry ATM term structure the
standalone Options dashboard builds. Good for "is vol elevated right now,"
not for term-structure or skew work — use the Options dashboard for that.

## Pilot scope

Coffee (KC + LRC) only. Adding another commodity is adding one entry to the
`COMMODITIES` dict in `Dashboard/app.py` with its per-exposure source codes
— no new plumbing.

## What's here

- **`Dashboard/app.py`** — the app.
- **`Dashboard/run_dashboard.bat`** — double-click to launch locally
  (`streamlit run app.py`).
- **`Code/ingest.py`** — copies 9 files (2 Rollex, 2 Futures, 2 Arb front,
  1 Roll Yield, 2 Options) from the sibling projects into this project's own
  `Database/`, mtime-gated (skips a file if the local copy is already
  newer).
- **`Database/`** — the local copy `ingest.py` produces. Committed to git so
  the deployed Cloud app has real data.
- **`Automator/run.bat`** — runs the ingest, commits + pushes `Database/` if
  anything changed, emails pass/fail via Outlook. Point Task Scheduler's
  Action at `cmd.exe` with argument
  `/c "...\Landing Page\Automator\run.bat"` — schedule it *after* Rollex,
  Futures, Arb, Roll Yield and Options have all finished their own daily
  runs.

## Running it

```bash
python Code/ingest.py              # sync parquets from the 5 source projects
streamlit run Dashboard/app.py
```

No LSEG/API session required — this project only reads parquets already
produced elsewhere.
