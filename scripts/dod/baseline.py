"""
baseline.py  (scripts/dod/baseline.py)
--------------------------------------
The tracked "last known good" shape of each profiled table, so the DATA report's gates can say
*not worse than before* rather than merely *here are some numbers*.

    reports/baselines/data_profile.json
        {"<table>": {"recorded_at", "rows", "columns": [...],
                     "null_rate": {"<field>": 0.0..1.0}, "date_min", "date_max", "scope": {...}}}

Design notes
  * TRACKED IN GIT, not in the session state dir. A row count that must not regress is a
    property of the REPOSITORY's history, not of one Claude session -- it has to survive a
    reboot and be reviewable in a diff. (Session state, which is per-turn and churny, stays
    out of the OneDrive-synced tree; see report_common.state_dir.)
  * A MISSING BASELINE IS NOT A FAILURE. The first profile of a table establishes it: gates
    D2/D5 report N/A and say so. Silently passing would be a lie; failing would make the
    first run of every new table impossible.
  * ONLY A FULL-SCOPE RUN MAY UPDATE IT. A profile of two tickers says nothing about the row
    count of the whole table, so `--tickers AAPL,JPM` must never overwrite a full baseline --
    that is how a "not decreased" gate gets quietly neutered.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

BASELINE_RELPATH = Path("reports") / "baselines" / "data_profile.json"


def baseline_path(root: Path) -> Path:
    return root / BASELINE_RELPATH


def load_profile_baseline(root: Path) -> dict:
    """The whole baseline document, `{}` when it does not exist yet."""
    path = baseline_path(root)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_profile_baseline(root: Path, data: dict) -> Path:
    path = baseline_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def snapshot_from_profile(profile: dict) -> dict:
    """Reduce a full table profile to the few fields the gates compare."""
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": profile.get("rows"),
        "columns": sorted(profile.get("columns") or []),
        "null_rate": {f: s.get("null_rate") for f, s in (profile.get("fields") or {}).items()
                      if s.get("null_rate") is not None},
        "date_min": profile.get("date_min"),
        "date_max": profile.get("date_max"),
        "scope": profile.get("scope"),
    }


def is_full_scope(scope: dict) -> bool:
    """True only when the profile covered the whole table -- no ticker filter, no row limit,
    no `since`. See the module docstring: a partial scope must not overwrite the baseline."""
    return not (scope.get("tickers") or scope.get("limit") or scope.get("since"))
