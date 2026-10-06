"""
superinvestor_roster.py  (src/utils/superinvestor_roster.py)
------------------------------------------------------------
THE read side of the `superinvestor_roster` table: which elite managers Dataroma
listed AS OF a given date.

Lives in `src/utils/` rather than beside the writer
(`data_extract/utils/institutionals/fetch_superinvestors.py`) because three subfolders
read it -- `data_aggregate` (elite 13F features), `strategies` (the replication sleeve)
and `data_extract` itself (the per-manager 13F walk scope) -- and cross-imports between
`src/` subfolders are not allowed. Same shape as `utils/universe.py`, which resolves the
ticker universe for every step from one place.

Point-in-time by construction: a snapshot row is what Dataroma published on
`snapshot_date`, so `roster_as_of(q)` reads the most recent snapshot AT OR BEFORE
`q` and never a later one. That is the only question a PIT manager selector asks,
and the flat `{cik: name}` JSON this replaces could not answer it -- applying
today's 81 names to 2013 silently drops the 23 managers Dataroma has since
dropped, six of whom carry real 13F history and two of whom (Arlington Value,
Wintergreen) are exactly the concentrated managers a concentration selector ranks
highest. Survivorship bias correlated with the selection rule is the worst kind.

Also owns the roster config under `<config_dir>/superinvestors/`: the hand resolutions
(`overrides.json`) and the committed Dataroma history (`dataroma_roster_history.json`).

MANAGER IDENTITY. A manager whose 13F filer CIK changed (Appaloosa, Ruane, Greenlight, Blue
Ridge) is one manager: its ID is the OLDEST filer CIK of its chain (`manager_ciks` in the
overrides), a singleton's ID is its own CIK. `roster_as_of` / `roster_map_as_of` return manager
IDs, `roster_cik_union` returns every filer CIK, and `to_manager_books` relabels a book read by
filer CIK to the manager ID, keeping each filer's rows only inside its dated window.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from functools import cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from src.constants.constants import DEFAULT_CONFIG_DIR
from src.context import Context
from src.data_store.schema import Tables
from src.utils.string import pad_cik, pad_cik_series

logger = logging.getLogger(__name__)

_COLS = ["snapshot_date", "dataroma_code", "manager_name", "cik"]

SUPERINVESTORS_CONFIG_SUBDIR = "superinvestors"
OVERRIDES_CONFIG_FILENAME = "overrides.json"
ROSTER_HISTORY_FILENAME = "dataroma_roster_history.json"


@dataclass(frozen=True)
class PeriodRange:
    """Inclusive quarter-end period range; `None` bounds are open."""

    start: date | None
    end: date | None

    def covers(self, period: object) -> bool:
        """Whether `period` (date, Timestamp or ISO string) lies inside the range."""
        day = pd.Timestamp(cast(Any, period)).date()
        return (self.start is None or day >= self.start) and (self.end is None or day <= self.end)


@dataclass(frozen=True)
class CikWindow:
    """One filer CIK of a manager chain and the periods in which it files that manager's book."""

    cik: str
    window: PeriodRange


@dataclass(frozen=True)
class InactiveRange:
    """A resolved code with no 13F activity over `window` (Q(snapshot) periods), for `reason`."""

    window: PeriodRange
    reason: str


@dataclass(frozen=True)
class SuperinvestorOverrides:
    """The roster hand configuration.

    `cik_by_code` wins over any stored or EDGAR resolution; `unresolvable` names the codes allowed
    to stay NULL; `inactive` names resolved codes without 13F activity over a dated range;
    `manager_ciks` maps a manager ID to its filer chain in chronological order."""

    cik_by_code: dict[str, str]
    unresolvable: dict[str, str]
    inactive: dict[str, InactiveRange] = field(default_factory=dict)
    manager_ciks: dict[str, tuple[CikWindow, ...]] = field(default_factory=dict)

    def manager_id(self, cik: object) -> str:
        """The manager ID of a filer CIK (padded); a CIK in no chain is its own ID, junk gives ''."""
        padded = pad_cik(cik)
        for mid, chain in self.manager_ciks.items():
            if any(w.cik == padded for w in chain):
                return mid
        return padded

    def chain(self, cik: object) -> tuple[CikWindow, ...]:
        """The filer chain of `cik`'s manager; empty for a singleton."""
        return self.manager_ciks.get(self.manager_id(cik), ())

    def member_at(self, cik: object, period: object) -> str:
        """The filer CIK of `cik`'s manager for `period`: the last chain member whose window starts at or
        before it (the first member before any start); a singleton is returned padded."""
        chain = self.chain(cik)
        if not chain:
            return pad_cik(cik)
        day = pd.Timestamp(cast(Any, period)).date()
        started = [w for w in chain if w.window.start is None or w.window.start <= day]
        return (started[-1] if started else chain[0]).cik


def _superinvestors_config_dir(config_dir: str | Path | None = None) -> Path:
    """The absolute `<config_dir>/superinvestors` directory; `None` means the default configs dir."""
    return Path(config_dir or DEFAULT_CONFIG_DIR).resolve() / SUPERINVESTORS_CONFIG_SUBDIR


def roster_history_path(config_dir: str | Path | None = None) -> Path:
    """The committed Dataroma roster history file under `_superinvestors_config_dir`."""
    return _superinvestors_config_dir(config_dir) / ROSTER_HISTORY_FILENAME


def load_superinvestor_overrides(config_dir: str | Path | None = None) -> SuperinvestorOverrides:
    """The superinvestor hand resolutions, cached per resolved config directory."""
    return _overrides_at(str(_superinvestors_config_dir(config_dir)))


@cache
def _overrides_at(superinvestors_dir: str) -> SuperinvestorOverrides:
    """`load_superinvestor_overrides`, keyed on the resolved `superinvestors` directory. Raises when a
    CIK is blank, a code is in two exception lists, or a chain is malformed (`_chain`)."""
    path = Path(superinvestors_dir) / OVERRIDES_CONFIG_FILENAME
    blob = json.loads(path.read_text(encoding="utf-8"))
    cik_by_code = {code: pad_cik(entry["cik"]) for code, entry in blob["cik_overrides"].items()}
    unresolvable = {code: str(reason) for code, reason in blob.get("unresolvable", {}).items()}
    inactive = {
        code: InactiveRange(_range(path, code, entry.get("from"), entry.get("to")), str(entry["reason"]))
        for code, entry in blob.get("inactive", {}).items()
    }
    blank = sorted(code for code, cik in cik_by_code.items() if not cik)
    both = sorted(set(cik_by_code) & set(unresolvable))
    twice = sorted(set(inactive) & set(unresolvable))
    if blank or both or twice:
        raise ValueError(f"{path}: blank CIK for {blank}; both overridden and unresolvable: {both}; both inactive and unresolvable: {twice}")
    manager_ciks = {pad_cik(mid): _chain(path, pad_cik(mid), entries) for mid, entries in blob.get("manager_ciks", {}).items()}
    members = [c for chain in manager_ciks.values() for c in {w.cik for w in chain}]
    shared = sorted({c for c in members if members.count(c) > 1})
    if shared:
        raise ValueError(f"{path}: CIK(s) {shared} listed in more than one chain")
    return SuperinvestorOverrides(cik_by_code=cik_by_code, unresolvable=unresolvable, inactive=inactive, manager_ciks=manager_ciks)


def _range(path: Path, label: str, start: str | None, end: str | None) -> PeriodRange:
    """`PeriodRange` from ISO `from` / `to` strings (None = open); raises when `from` is after `to`."""
    out = PeriodRange(date.fromisoformat(start) if start else None, date.fromisoformat(end) if end else None)
    if out.start and out.end and out.start > out.end:
        raise ValueError(f"{path}: {label}: from {out.start} is after to {out.end}")
    return out


def _chain(path: Path, manager_id: str, entries: list[dict[str, Any]]) -> tuple[CikWindow, ...]:
    """A manager's filer chain, validated: at least two members, listed chronologically with the
    manager ID (the oldest CIK) first, and no two windows sharing a period. A filer may return later
    in its own chain (A -> B -> A) with a second window."""
    chain = tuple(CikWindow(pad_cik(e["cik"]), _range(path, manager_id, e.get("from"), e.get("to"))) for e in entries)
    if len(chain) < 2:
        raise ValueError(f"{path}: chain {manager_id} needs at least two filer CIKs, got {len(chain)}")
    if chain[0].cik != manager_id:
        raise ValueError(f"{path}: chain {manager_id} must list its manager ID (the oldest filer CIK) first, got {chain[0].cik}")
    for prev, nxt in zip(chain, chain[1:], strict=False):
        if prev.window.end is None or nxt.window.start is None or nxt.window.start <= prev.window.end:
            raise ValueError(
                f"{path}: chain {manager_id}: windows of {prev.cik} (to {prev.window.end}) and {nxt.cik} "
                f"(from {nxt.window.start}) overlap or are not chronological; the manager ID must be the oldest filer CIK"
            )
    return chain


def _load(context: Context) -> pd.DataFrame | None:
    """The whole roster table (the capture-dated history plus each changed live snapshot) with
    `snapshot_date` coerced to Timestamp. None when the table has never been written.

    `snapshot_date` is a SQL DATE, so psycopg2 hands back `datetime.date` objects that
    compare unequal to the `Timestamp` every caller holds -- coerce once, here."""
    df = context.store.load(Tables.superinvestor_roster, columns=_COLS, optional=True)
    if df is None or df.empty:
        return None
    df = df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def _snapshot(context: Context, as_of=None) -> pd.DataFrame | None:
    """The rows of the most recent snapshot at or before `as_of` (the latest snapshot when
    `as_of` is None). None before the first snapshot -- an honest "Dataroma published no
    roster yet", not an empty roster to be silently ffilled backwards."""
    df = _load(context)
    if df is None:
        logger.warning("`%s` is empty -- run `data_extract superinvestors -F`.", Tables.superinvestor_roster)
        return None
    if as_of is not None:
        df = df[df["snapshot_date"] <= pd.Timestamp(as_of)]
        if df.empty:
            return None
    return cast(pd.DataFrame, df[df["snapshot_date"] == df["snapshot_date"].max()])


def config_dir_of(context: Context) -> str | Path | None:
    """The context's config directory, None (the default configs dir) when it has none."""
    return getattr(context, "config_dir", None)


def roster_as_of(context: Context, as_of=None) -> set[str]:
    """The manager IDs on the roster at `as_of` -- THE point-in-time accessor.

    A set, so the `brk` / `BRK` duplicate (two Dataroma codes, one manager, CIK
    0001067983) collapses to one member, and so does a manager stored under either filer of
    its chain. Unresolved managers carry a NULL `cik` and are dropped here; the writer is
    what must fail loudly on them (they must never silently shrink the eligible pool, which
    is the survivorship bug this table exists to remove)."""
    snap = _snapshot(context, as_of)
    if snap is None:
        return set()
    overrides = load_superinvestor_overrides(config_dir_of(context))
    return {c for raw in snap["cik"].dropna() if (c := overrides.manager_id(raw))}


def roster_map_as_of(context: Context, as_of=None) -> dict[str, str]:
    """`{manager_id: manager_name}` at `as_of` -- `roster_as_of` plus the display name, for
    the callers that log it or weight by it. One entry per manager: the duplicate-code pair
    keeps the first name in `dataroma_code` order, so the map is deterministic."""
    snap = _snapshot(context, as_of)
    if snap is None:
        return {}
    overrides = load_superinvestor_overrides(config_dir_of(context))
    out: dict[str, str] = {}
    for _code, name, raw in snap.sort_values("dataroma_code")[["dataroma_code", "manager_name", "cik"]].itertuples(index=False):
        if (cik := overrides.manager_id(raw)) and cik not in out:
            out[cik] = str(name)
    return out


def first_snapshot_date(context: Context) -> pd.Timestamp | None:
    """The earliest `snapshot_date` in the table, or None when it has never been written.

    The roster history starts after the first 13F books, so `roster_as_of(q)` is EMPTY for
    the earliest quarters and a caller that used it unguarded would zero every manager there.
    Callers floor their lookup at this date: extrapolating the oldest roster backwards is a
    compromise, but the alternative is today's roster, which is the survivorship bias this
    table exists to remove."""
    df = _load(context)
    if df is None:
        return None
    snapshot_dates = cast(pd.Series, df["snapshot_date"])
    return cast(pd.Timestamp, pd.Timestamp(snapshot_dates.min()))


def roster_cik_union(context: Context) -> set[str]:
    """Every filer CIK of every manager that was EVER on the roster, across all snapshots.

    NOT a point-in-time set and never a feature input: this is the per-manager 13F WALK
    SCOPE. Fetching only today's roster is what makes a dropped manager unrecoverable
    later, so the walk covers the union and the selector narrows it per quarter. A chain
    contributes every member, whichever one the snapshots stored."""
    df = _load(context)
    if df is None:
        return set()
    return filer_ciks(df["cik"].dropna(), config_dir_of(context))


def filer_ciks(manager_ids: Iterable[object], config_dir: str | Path | None = None) -> set[str]:
    """Every filer CIK of the given managers (any chain member stands for its manager), padded."""
    overrides = load_superinvestor_overrides(config_dir)
    out: set[str] = set()
    for raw in manager_ids:
        if not (cik := pad_cik(raw)):
            continue
        chain = overrides.chain(cik)
        out |= {w.cik for w in chain} if chain else {cik}
    return out


def to_manager_books(df: pd.DataFrame, config_dir: str | Path | None = None) -> pd.DataFrame:
    """A book read by filer CIK, keyed by manager ID: each chain member's rows are kept only for
    periods inside one of its windows and relabelled to the manager ID; other rows are untouched. Padded
    and unpadded CIK strings both match; `period` may be date, Timestamp or ISO string."""
    overrides = load_superinvestor_overrides(config_dir)
    windows = [(mid, w) for mid, chain in overrides.manager_ciks.items() for w in chain]
    if df.empty or not windows:
        return df
    padded = pad_cik_series(df["cik"]).to_numpy()
    chained = np.isin(padded, [w.cik for _, w in windows])
    if not chained.any():
        return df
    periods = pd.to_datetime(df["period"].where(chained)).to_numpy()
    label = np.full(len(df), None, dtype=object)
    for mid, w in windows:  # a returning filer has several windows
        hit = padded == w.cik
        if w.window.start is not None:
            hit &= periods >= np.datetime64(w.window.start)
        if w.window.end is not None:
            hit &= periods <= np.datetime64(w.window.end)
        label[hit] = mid
    keep = ~chained | pd.notna(label)
    out = df[keep].copy()
    out["cik"] = np.where(chained[keep], label[keep], out["cik"].to_numpy(dtype=object))
    return out
