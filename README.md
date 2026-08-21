# Landing Page — Unified Commodity Dashboard

A single page per commodity, with one tab per exposure: **Flat | Spread |
Arb | Volatility | Risk**. This project holds no data of its own — every
panel reads directly, read-only, off the `Database/` folder of the sibling
Interim_Migration project that already computes it. Those projects each run
their own daily automator, so this app always sees their latest parquet.

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
- **`Code/check_sources.py`** — there's nothing to ingest here, so the daily
  automator instead confirms every source parquet this app depends on
  exists and isn't stale (>3 days old), and fails loudly if not.
- **`Automator/run.bat`** — runs the source check, emails pass/fail via
  Outlook. Point Task Scheduler's Action at `cmd.exe` with argument
  `/c "...\Landing Page\Automator\run.bat"`.

## Running it

```bash
streamlit run Dashboard/app.py
```

No LSEG/API session and no local database required — this project only
reads parquets already produced by Rollex, Futures, Arb, Roll Yield, and
Options.
