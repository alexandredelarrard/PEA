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
"""
from __future__ import annotations

import logging

import pandas as pd

from src.context import Context
from src.data_store.schema import Tables
from src.utils.string import pad_cik

logger = logging.getLogger(__name__)

_COLS = ["snapshot_date", "dataroma_code", "manager_name", "cik"]


def _load(context: Context) -> pd.DataFrame | None:
    """The whole roster table (879 seeded rows + one snapshot per live scrape) with
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
        logger.warning("`%s` is empty -- run `data_extract superinvestors --seed`.",
                       Tables.superinvestor_roster)
        return None
    if as_of is not None:
        df = df[df["snapshot_date"] <= pd.Timestamp(as_of)]
        if df.empty:
            return None
    return df[df["snapshot_date"] == df["snapshot_date"].max()]


def roster_as_of(context: Context, as_of=None) -> set[str]:
    """The padded CIKs on the roster at `as_of` -- THE point-in-time accessor.

    A set, so the `brk` / `BRK` duplicate (two Dataroma codes, one manager, CIK
    0001067983) collapses to one member. Unresolved managers carry a NULL `cik` and are
    dropped here; the writer is what must fail loudly on them (they must never silently
    shrink the eligible pool, which is the survivorship bug this table exists to remove)."""
    snap = _snapshot(context, as_of)
    if snap is None:
        return set()
    return {c for raw in snap["cik"].dropna() if (c := pad_cik(raw))}


def roster_map_as_of(context: Context, as_of=None) -> dict[str, str]:
    """`{padded_cik: manager_name}` at `as_of` -- `roster_as_of` plus the display name, for
    the callers that log it or weight by it. One entry per CIK: the duplicate-code pair
    keeps the first name in `dataroma_code` order, so the map is deterministic."""
    snap = _snapshot(context, as_of)
    if snap is None:
        return {}
    out: dict[str, str] = {}
    for code, name, raw in snap.sort_values("dataroma_code")[
            ["dataroma_code", "manager_name", "cik"]].itertuples(index=False):
        if (cik := pad_cik(raw)) and cik not in out:
            out[cik] = str(name)
    return out


def first_snapshot_date(context: Context) -> pd.Timestamp | None:
    """The earliest `snapshot_date` in the table, or None when it has never been written.

    Dataroma's history starts in 2013 but the 13F books start 2011-09-30, so
    `roster_as_of(q)` is EMPTY for the first two years and a caller that used it unguarded
    would zero every manager there. Callers floor their lookup at this date: extrapolating
    the oldest roster backwards is a compromise, but the alternative is today's roster,
    which is the survivorship bias this table exists to remove."""
    df = _load(context)
    return None if df is None else pd.Timestamp(df["snapshot_date"].min())


def roster_cik_union(context: Context) -> set[str]:
    """Every CIK that was EVER on the roster, across all snapshots.

    NOT a point-in-time set and never a feature input: this is the per-manager 13F WALK
    SCOPE. Fetching only today's roster is what makes a dropped manager unrecoverable
    later, so the walk covers the union and the selector narrows it per quarter."""
    df = _load(context)
    if df is None:
        return set()
    return {c for raw in df["cik"].dropna() if (c := pad_cik(raw))}
