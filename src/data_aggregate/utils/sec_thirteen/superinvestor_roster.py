"""Configured fixed cohorts and historical roster access."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.utils.superinvestor_roster import load_superinvestor_overrides, snapshot_as_of


def load_manager_cohorts(config_dir: Path | str, roster: set[str], k: int) -> dict[str, frozenset[str]]:
    """Validate the user-ordered raw-CIK list and select its fixed prefix."""
    # Read the one configured ordering before computing either cohort.
    path = Path(config_dir) / "superinvestors" / "ranked_managers.json"
    managers = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(managers, list) or not managers:
        raise ValueError("ranked_managers.json must contain a nonempty ordered CIK array")

    # Reject implicit normalization, duplicate identities and hidden replacements.
    if not all(isinstance(cik, str) and re.fullmatch(r"[0-9]{10}", cik) for cik in managers):
        raise ValueError("Every configured CIK must be an explicit ten-digit string")
    if len(set(managers)) != len(managers) or not set(managers).issubset(roster):
        raise ValueError("Configured CIKs must be unique members of the full raw-CIK roster")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(managers):
        raise ValueError("K must be a positive integer bounded by the ordered CIK list")
    return {"": frozenset(roster), "topk": frozenset(managers[:k])}


def roster_as_of(history: pd.DataFrame | None, as_of: object = None, *, config_dir: str | Path | None = None) -> set[str]:
    """The manager IDs on the roster at `as_of` -- THE point-in-time accessor.

    A set, so the `brk` / `BRK` duplicate (two Dataroma codes, one manager, CIK
    0001067983) collapses to one member, and so does a manager stored under either filer of
    its chain. Unresolved managers carry a NULL `cik` and are dropped here; the writer is
    what must fail loudly on them (they must never silently shrink the eligible pool, which
    is the survivorship bug this table exists to remove)."""
    snap = snapshot_as_of(history, as_of)
    if snap is None:
        return set()
    overrides = load_superinvestor_overrides(config_dir)
    return {c for raw in snap["cik"].dropna() if (c := overrides.manager_id(raw))}
