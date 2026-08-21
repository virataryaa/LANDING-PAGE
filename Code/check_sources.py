"""
Landing Page — Source Freshness Check
This project holds no data of its own; every panel reads live off sibling
projects' Database/ folders. There is nothing to ingest, so the daily
automator runs this instead: confirm every source parquet this app depends
on exists and was updated recently, and email/log if anything looks stale.
"""
import datetime
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[1]        # .../Landing Page
ROOT = BASE.parent                                # .../Interim_Migration

STALE_AFTER_DAYS = 3   # weekends + one missed run shouldn't false-alarm

SOURCES = [
    ROOT / "Rollex" / "Database" / "rollex_KC.parquet",
    ROOT / "Rollex" / "Database" / "rollex_RC.parquet",
    ROOT / "Futures" / "Database" / "kc_futures.parquet",
    ROOT / "Futures" / "Database" / "rc_futures.parquet",
    ROOT / "Arb" / "Database" / "front_KC.parquet",
    ROOT / "Arb" / "Database" / "front_RC.parquet",
    ROOT / "Roll Yield" / "Database" / "roll_yield_data.parquet",
    ROOT / "Options" / "Database" / "KC_options_ice.parquet",
    ROOT / "Options" / "Database" / "LRC_options_ice.parquet",
]


def check() -> bool:
    now = datetime.datetime.now()
    ok, missing, stale = 0, [], []

    for path in SOURCES:
        if not path.exists():
            missing.append(str(path))
            continue
        age_days = (now - datetime.datetime.fromtimestamp(path.stat().st_mtime)).days
        if age_days > STALE_AFTER_DAYS:
            stale.append(f"{path.name} (last updated {age_days}d ago)")
        else:
            ok += 1

    log.info(f"Checked {len(SOURCES)} source files — {ok} fresh, "
             f"{len(stale)} stale, {len(missing)} missing")
    for m in missing:
        log.warning(f"  MISSING: {m}")
    for s in stale:
        log.warning(f"  STALE: {s}")

    return not missing and not stale


if __name__ == "__main__":
    log.info("=" * 50 + f"\nLanding Page source check | {datetime.date.today()}\n" + "=" * 50)
    healthy = check()
    log.info("=" * 50)
    if not healthy:
        raise SystemExit(1)
