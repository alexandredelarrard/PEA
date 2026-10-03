"""Per-table JSON checkpoint (`extraction_manifest.json`): last run date, ticker count, rows added.

Most fetchers call `record_run` for bookkeeping only; their DB frontier stays authoritative. The
per-ticker EDGAR filing listers also take their `since` from `manifest_window`, which forces a
full-window relist when the exact ticker membership changed or `full_rescan_days` elapsed. Writes
are atomic read-modify-write that never clobber sibling tables' entries.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.constants.constants import DATE_FORMAT
from src.context import Context
from src.data_store.schema import Table, name_of

logger = logging.getLogger(__name__)


def manifest_path(context: Context) -> Path:
    """The manifest JSON file under the data store directory."""
    return Path(context.paths["DATA_STORE"]) / Path(context.config.local.filename.extraction)


def _load_manifest(context: Context) -> dict:
    path = manifest_path(context)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        logger.warning("extraction_manifest.json unreadable at %s -- starting fresh", path)
        return {}


def _replace_with_retry(src: Path, dst: Path, attempts: int = 5, backoff_s: float = 0.1) -> None:
    """`os.replace(src, dst)`, retried on `PermissionError` (Windows refuses the rename while
    another process holds `dst` open); the last failure raises."""
    for attempt in range(1, attempts + 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts:
                raise
        time.sleep(backoff_s * attempt)


def _save_manifest(context: Context, manifest: dict) -> None:
    """Write `manifest` atomically (same-directory temp file, then `os.replace`), so a crash keeps
    the previous file; the temp file never outlives the call."""
    path = manifest_path(context)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        _replace_with_retry(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def get_entry(context: Context, table: Table | str) -> dict | None:
    """This table's last recorded run (keyed by table name), or None on a first run / corrupt file."""
    return _load_manifest(context).get(name_of(table))


def changed_scope_tickers(entry: dict | None, current: dict[str, str]) -> frozenset[str]:
    """Tickers whose identity-aware filing scope is new or changed."""
    prior = (entry or {}).get("identity_scope_fingerprints", {})
    return frozenset(ticker for ticker, fingerprint in current.items() if prior.get(ticker) != fingerprint)


def manifest_window(
    context: Context,
    table: Table | str,
    tickers: Sequence[str],
    *,
    fallback_since: pd.Timestamp,
    full_rescan_days: int,
) -> tuple[pd.Timestamp, bool]:
    """`(since, is_full_rescan)` for an EDGAR filing lister; pass `is_full_rescan` on to `record_run`.

    Returns `(fallback_since, True)` when there is no recorded run, the exact `tickers` set differs
    from the stored one (a same-size swap included), or the last full rescan is `>= full_rescan_days`
    old; otherwise the entry's `last_run_date` (inclusive) with False."""

    entry = get_entry(context, table)
    if not entry or entry.get("tickers") != sorted({str(ticker) for ticker in tickers}):
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
    table: Table | str,
    ticker_count: int,
    rows_added: int,
    is_full_rescan: bool = False,
    run_date: pd.Timestamp | str | None = None,
    backfill_window: tuple[str, str] | None = None,
    coverage_complete: bool = False,
    identity_scope_fingerprints: dict[str, str] | None = None,
    tickers: Iterable[str] | None = None,
) -> None:
    """Merge this table's run stats into the shared manifest without touching sibling tables.

    `last_run_date` becomes `run_date` (default today); `last_full_rescan_date` too when
    `is_full_rescan` or no prior one exists. A `backfill_window` run advances nothing: the window is
    appended to `backfills` and every other field is carried through unchanged."""
    run_ts = pd.Timestamp(run_date).normalize() if run_date is not None else pd.Timestamp.today().normalize()
    run_date_str = run_ts.strftime(DATE_FORMAT)

    name = name_of(table)
    manifest = _load_manifest(context)
    prior = manifest.get(name) or {}

    if backfill_window is not None:
        entry = dict(prior)
        entry["backfills"] = [
            *prior.get("backfills", []),
            {
                "window": f"{backfill_window[0]}:{backfill_window[1]}",
                "rows_added": int(rows_added),
                "run_date": run_date_str,
            },
        ]
        entry["updated_at"] = datetime.now(UTC).isoformat()
        manifest[name] = entry
        _save_manifest(context, manifest)
        return

    last_full_rescan_date = run_date_str if (is_full_rescan or not prior.get("last_full_rescan_date")) else prior["last_full_rescan_date"]

    entry = {
        **{k: v for k, v in prior.items() if k in {"backfills", "identity_scope_fingerprints", "tickers"}},
        "last_run_date": run_date_str,
        "last_full_rescan_date": last_full_rescan_date,
        "ticker_count": int(ticker_count),
        "rows_added": int(rows_added),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    if coverage_complete:
        entry["coverage_complete"] = True
    if identity_scope_fingerprints is not None:
        entry["identity_scope_fingerprints"] = dict(sorted(identity_scope_fingerprints.items()))
    if tickers is not None:
        entry["tickers"] = sorted({str(ticker) for ticker in tickers})
    manifest[name] = entry
    _save_manifest(context, manifest)
