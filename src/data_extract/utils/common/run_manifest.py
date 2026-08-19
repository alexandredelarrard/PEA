"""
run_manifest.py  (src/data_extract/utils/common/run_manifest.py)
------------------------------------------------------------------
Single JSON checkpoint file (`data/extraction_manifest.json`) tracking, per DB
table, the last extraction run's date / ticker count / rows added. Two distinct
uses across the fetchers in `step_extract_all_data.py`:

  1. VISIBILITY (most fetchers): `record_run` is called once at the end purely
     for bookkeeping. These fetchers already decide their own extraction window
     from DB state (per-ticker max date, or bulk-period/accession dedup), which
     stays authoritative and is NOT replaced by this file. `fetch_13f` belongs
     here despite being an EDGAR filing-lister: it is all-filers, so
     `manifest_window`'s `ticker_count` trigger means nothing to it, and a full
     rescan would be ~528k filings (~16h). It resumes from the table's
     max(filing_date) minus its own bounded `lookback_days` instead.
  2. WINDOW CONTROL (the EDGAR filing-listing fetchers only: 13D, 8-K, DEF 14A
     edgar + LLM, fundamentals edgartools): these list each ticker's FULL
     `years_history` window every run and rely solely on post-hoc accession
     dedup (see `sec_utils.existing_filings`'s docstring). `manifest_window`
     gives them a narrower `since` cutoff (the last run's date, inclusive)
     instead of the fixed multi-year window.

     Bounded, not unconditional: `sec_utils.existing_filings` documents that an
     earlier version of these fetchers tried a permanent per-ticker cutoff and
     reverted it -- a filing missed by a bug, or one EDGAR posts out of date
     order, would stay missing forever once the window stops looking behind it.
     `manifest_window` therefore also forces a full-window relist (self-heal)
     whenever the table's ticker count changed (a new ticker needs its own full
     history) or `full_rescan_days` have elapsed since the last full relist --
     bounding any silently-missed filing to that window instead of forever.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.constants.constants import DATE_FORMAT
from src.context import Context
from src.data_store.schema import name_of

logger = logging.getLogger(__name__)

_MANIFEST_FILENAME = "extraction_manifest.json"


def _manifest_path(context: Context) -> Path:
    return Path(context.paths["DATA_STORE"]) / _MANIFEST_FILENAME


def _load(context: Context) -> dict:
    path = _manifest_path(context)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:                                        # noqa: BLE001
        logger.warning("extraction_manifest.json unreadable at %s -- starting fresh", path)
        return {}


def get_entry(context: Context, table) -> dict | None:
    """This table's last recorded run, or None on a first run / corrupt file.

    `name_of` because the JSON is keyed by the table NAME: callers now pass the `Table` object,
    and keying on it would both fail to serialize and orphan every existing manifest entry."""
    return _load(context).get(name_of(table))


def manifest_window(
    context: Context,
    table: str,
    ticker_count: int,
    fallback_since: pd.Timestamp,
    full_rescan_days: int,
) -> tuple[pd.Timestamp, bool]:
    """The `since` cutoff an EDGAR filing-lister should use, and whether this run
    counts as a full rescan (pass straight through to `record_run`).

    Falls back to `fallback_since` (the fetcher's usual years-history window) --
    marking `is_full_rescan=True` -- when there is no recorded run yet, the
    ticker universe changed size since that run, or the last full rescan is
    `>= full_rescan_days` old. Otherwise returns the entry's `last_run_date`
    (inclusive) with `is_full_rescan=False`."""
    entry = get_entry(context, table)
    if not entry or entry.get("ticker_count") != ticker_count:
        return fallback_since, True

    last_full = entry.get("last_full_rescan_date")
    if not last_full:
        return fallback_since, True
    try:
        age_days = (pd.Timestamp.today().normalize() - pd.Timestamp(last_full).normalize()).days
    except (TypeError, ValueError):
        return fallback_since, True
    if age_days >= full_rescan_days:
        return fallback_since, True

    last_run = entry.get("last_run_date")
    if not last_run:
        return fallback_since, True
    try:
        return pd.Timestamp(last_run).normalize(), False
    except (TypeError, ValueError):
        return fallback_since, True


def record_run(
    context: Context,
    table,
    ticker_count: int,
    rows_added: int,
    is_full_rescan: bool = False,
    run_date: pd.Timestamp | str | None = None,
) -> None:
    """Merge this table's run stats into the shared manifest (read-modify-write --
    every fetcher in a step run shares the one file, so this must not clobber
    sibling tables' entries). `last_run_date` is always set to `run_date`
    (default today); `last_full_rescan_date` is set to it too when
    `is_full_rescan` or this table has no prior entry, else left unchanged."""
    run_ts = pd.Timestamp(run_date).normalize() if run_date is not None else pd.Timestamp.today().normalize()
    run_date_str = run_ts.strftime(DATE_FORMAT)

    name = name_of(table)
    manifest = _load(context)
    prior = manifest.get(name) or {}
    last_full_rescan_date = (
        run_date_str if (is_full_rescan or not prior.get("last_full_rescan_date"))
        else prior["last_full_rescan_date"]
    )
    
    manifest[name] = {
        "last_run_date": run_date_str,
        "last_full_rescan_date": last_full_rescan_date,
        "ticker_count": int(ticker_count),
        "rows_added": int(rows_added),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _manifest_path(context).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
