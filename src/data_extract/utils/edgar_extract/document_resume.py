"""EDGAR document work list: what each nightly run fetches for an EDGAR document table.

`document_worklist`: the local EDGAR index rows of the table's forms for
each key's filing scope (event CIKs and CIK windows), from the floor on and filed on or before the key's `ended_on`
(`Universe.bounds`), minus the accessions already
stored (empty-filing markers included; optionally only on the rows of one source); newest first and capped per run
(a `fill_cold` request lists a key with no stored row in full, outside the cap).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.identity import FilingScope, Identity
from src.data_extract.utils.common.incremental import stored_values
from src.data_extract.utils.common.registrant import listing_ciks, resolve_registrant_entries
from src.data_extract.utils.common.resume import (
    DONE_TABLE,
    KEY_ROWLESS,
    planner_bounds,
    stored_accessions,
)
from src.data_extract.utils.edgar_extract import edgar_index
from src.data_store.schema import Table
from src.utils.data_policy import window_policy
from src.utils.tickers_universe.universe import before_end
from src.utils.window import Window, resolve_window

logger = logging.getLogger(__name__)

#: `(context, keys, runtime_floor) -> {key: done accessions}`: a fetch's own stored-set rule.
DoneFn = Callable[[Context, Sequence[str], pd.Timestamp | None], dict[str, set[str]]]


@dataclass
class DocumentWork:
    """One table's document work list: index rows to read per key (oldest first), plus what `resume-plan` reports."""

    units: dict[str, pd.DataFrame]
    counts: dict[str, int] = field(default_factory=dict)
    uncapped: int = 0
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return sum(len(df) for df in self.units.values())


def _classify_units(df_listed: pd.DataFrame, df_units: pd.DataFrame) -> dict[str, int]:
    """Count missing source units before/after the actual completed filing frontier."""
    if df_units.empty:
        return {}
    stored = df_listed.loc[~df_listed["accession"].isin(df_units["accession"]), "filed"]
    if stored.empty:
        return {KEY_ROWLESS: len(df_units)}
    n_forward = int(df_units["filed"].ge(stored.max()).sum())
    return {"forward": n_forward, "gap": len(df_units) - n_forward}


def _cap(units: dict[str, pd.DataFrame], cap: int | None, desc: str) -> tuple[dict[str, pd.DataFrame], int]:
    """At most `cap` units in total, newest first, each key's list kept oldest first; an ERROR names the uncapped size."""
    uncapped = sum(len(df) for df in units.values())
    if cap is None or uncapped <= cap:
        return units, uncapped
    logger.error(
        "%s: work list of %d document(s) exceeds the per-run cap of %d; the newest %d run tonight, the rest on later runs", desc, uncapped, cap, cap
    )
    df_all = pd.concat([df.assign(_key=key) for key, df in units.items()], ignore_index=True)
    df_kept = df_all.sort_values(["filed", "accession"], ascending=False, kind="mergesort").head(cap)
    capped = {str(key): group.drop(columns="_key").sort_values(["filed", "accession"], ignore_index=True) for key, group in df_kept.groupby("_key")}
    return capped, uncapped


@dataclass(frozen=True)
class WorklistRequest:
    """Source keys/forms, identity, completion rules and the operational document cap."""

    keys: pd.DataFrame
    forms: Sequence[str]
    identity: Identity | None
    runtime_floor: pd.Timestamp | None = None
    done_scope: str = DONE_TABLE
    done_where: dict[str, object] | None = None
    cap: int | None = None
    done_accessions: DoneFn | None = None
    fill_cold: bool = False


def _cold_keys(context: Context, table: Table, keys: Sequence[str]) -> frozenset[str]:
    """The `keys` with no row in `table` (markers included); all of them when the table is absent."""
    return frozenset(keys) - stored_values(context, table, "ticker", where={"ticker": list(keys)})


def document_worklist(context: Context, table: Table, request: WorklistRequest) -> DocumentWork:
    """The documents `table` still lacks for the request's keys: index rows from the floor on, resolved through
    each key's filing scope (its roster CIK alone without `identity`), minus the accessions stored on the rows
    matching `done_where`, or by the request's own `done_accessions` (ignored under `full`)."""
    policy = window_policy(context.config, table, context.run.as_of)
    if policy is None or request.keys.empty:
        return DocumentWork(units={})
    scope = (("ticker", tuple(request.keys["ticker"].astype(str))),)
    plan = resolve_window(policy, context.run, None, pd.DatetimeIndex([]), scope)
    floor = plan.write.start
    if request.runtime_floor is not None:
        floor = max(floor, request.runtime_floor)
    if floor > plan.write.end:
        return DocumentWork(units={})
    window = Window(floor, plan.write.end)
    keys, forms, identity = request.keys, request.forms, request.identity
    runtime_floor = request.runtime_floor
    full = context.run.mode in {"full", "recovery"}
    bounds = planner_bounds(context)
    started = time.perf_counter()
    roster = dict(zip(keys["ticker"].astype(str), keys["cik"].astype(str), strict=True))
    scopes = {
        ticker: identity.filing_scope(ticker) if identity is not None else FilingScope.roster_only(ticker, cik) for ticker, cik in roster.items()
    }
    candidates = {ticker: listing_ciks(scope) for ticker, scope in scopes.items()}
    df_index = edgar_index.entries(context, {c for ciks in candidates.values() for c in ciks}, forms, window)
    by_cik = dict(tuple(df_index.groupby("cik", sort=False))) if not df_index.empty else {}
    read_index = time.perf_counter()
    if full:
        done: dict[str, set[str]] = {key: set() for key in roster}
    elif request.done_accessions is not None:
        done = request.done_accessions(context, list(roster), runtime_floor)
    else:
        done = stored_accessions(context, table, list(roster), request.done_scope, request.done_where)
    cold = _cold_keys(context, table, list(roster)) if request.fill_cold else frozenset()
    read_done = time.perf_counter()
    units: dict[str, pd.DataFrame] = {}
    counts: dict[str, int] = {}
    for ticker in roster:
        frames = [by_cik[c] for c in candidates[ticker] if c in by_cik]
        df_listed = resolve_registrant_entries(scopes[ticker], pd.concat(frames, ignore_index=True), forms) if frames else df_index.iloc[:0]
        df_listed = df_listed[before_end(pd.Series(ticker, index=df_listed.index), df_listed["filed"], bounds)]
        done_here = done.get(ticker, set()) & set(df_listed["accession"].astype(str))
        df_units = df_listed[~df_listed["accession"].isin(done_here)].reset_index(drop=True)
        for name, n in _classify_units(df_listed, df_units).items():
            counts[name] = counts.get(name, 0) + n
        if not df_units.empty:
            units[ticker] = df_units
    cold_units = {key: df for key, df in units.items() if key in cold}
    warm, uncapped = _cap({key: df for key, df in units.items() if key not in cold}, request.cap, table.name)
    n_cold = sum(map(len, cold_units.values()))
    if n_cold:
        logger.info(
            "%s: %d cold key(s) with no stored row list all %d of their document(s), outside the per-run cap",
            table.name,
            len(cold_units),
            n_cold,
        )
    units = {**warm, **cold_units}
    uncapped += n_cold
    timings = {"index_read": read_index - started, "done_read": read_done - read_index, "total": time.perf_counter() - started}
    return DocumentWork(units=units, counts=counts, uncapped=uncapped, timings=timings)
