"""
Landing Page — Ingest / Sync
This project has no data pipeline of its own — it visualizes what the other
Interim_Migration projects already compute. Locally, app.py could just read
their Database/ folders directly (they sit right next to this one on disk).
But this project deploys to Streamlit Cloud from its OWN git repo, and the
Cloud build only clones that one repo — it has no access to the other
projects' repos or their Database folders. So this script copies the
specific files this app needs into this project's own Database/, and that
local copy is what gets committed and pushed (same pattern the VaR project
uses for the same reason).

Run this locally after the source projects' own daily automators complete,
then commit + push Database/ so the deployed Cloud app picks up the refresh.
"""
import shutil
import logging
import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

HERE    = Path(__file__).resolve().parent.parent   # .../Landing Page
ROOT    = HERE.parent                              # .../Interim_Migration
OUT_DIR = HERE / "Database"

SOURCES = [
    (ROOT / "Rollex" / "Database" / "rollex_KC.parquet",        OUT_DIR / "rollex_KC.parquet"),
    (ROOT / "Rollex" / "Database" / "rollex_RC.parquet",        OUT_DIR / "rollex_RC.parquet"),
    (ROOT / "Futures" / "Database" / "kc_futures.parquet",      OUT_DIR / "kc_futures.parquet"),
    (ROOT / "Futures" / "Database" / "rc_futures.parquet",      OUT_DIR / "rc_futures.parquet"),
    (ROOT / "Arb" / "Database" / "front_KC.parquet",            OUT_DIR / "front_KC.parquet"),
    (ROOT / "Arb" / "Database" / "front_RC.parquet",            OUT_DIR / "front_RC.parquet"),
    (ROOT / "Roll Yield" / "Database" / "roll_yield_data.parquet", OUT_DIR / "roll_yield_data.parquet"),
    (ROOT / "Options" / "Database" / "KC_options_ice.parquet",  OUT_DIR / "KC_options_ice.parquet"),
    (ROOT / "Options" / "Database" / "LRC_options_ice.parquet", OUT_DIR / "LRC_options_ice.parquet"),
]


def sync():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok, failed = 0, []

    for src, dst in SOURCES:
        if not src.exists():
            log.warning(f"  MISSING: {src}")
            failed.append(str(src))
            continue
        src_mtime = src.stat().st_mtime
        dst_mtime = dst.stat().st_mtime if dst.exists() else 0
        if src_mtime > dst_mtime:
            shutil.copy2(src, dst)
            log.info(f"  Copied : {dst.name}")
        else:
            log.info(f"  Up to date: {dst.name}")
        ok += 1

    log.info(f"Sync complete — {ok} files OK, {len(failed)} missing")
    if failed:
        raise RuntimeError(f"Missing source files: {failed}")


if __name__ == "__main__":
    log.info("=" * 50 + f"\nLanding Page Ingest | {datetime.date.today()}\n" + "=" * 50)
    sync()
    log.info("=" * 50)
