"""
ledger.py  (src/validate/fundamentals/)
--------------------------------------------------------------------------------------------
The FIRST AND ONLY read-back of `fundamentals_check` / `_check_run` / `_check_status`.

## Why this module exists at all

Until now nothing read `fundamentals_check`. The validator wrote 11,926 rows a run and every
consumer -- the report, the register, the agents -- worked off the in-memory `ValidationRun`
instead. That is why "did this fix close anything?" was unanswerable without a hand-written
SQL query, and why a report could only ever describe the run that produced it.

This module is the counterpart to `substrate.py` and follows its discipline exactly: ONE place
that touches the store, projected reads, dates normalised once on load. A caller gets frames.

## `comparable_runs` is the load-bearing function

The loop's proposition is that a drop in row count PROVES a fix. That is only true between two
runs that looked at the same thing, so comparability is a hard test rather than a courtesy:
two runs are comparable iff their `scope_hash` matches. A 54-ticker baseline and a one-ticker
re-validation would otherwise report ~11,800 findings "closed".

When no comparable prior run exists, this module returns nothing and the report says so. A
first run must never render as a trend.

## Postgres DATE round-trips as `datetime.date`

Never as a `Timestamp`. An unguarded `frame["run_date"] > some_timestamp` then does something
surprising, and a parquet-cached test harness hides the whole bug class because parquet hands
back `Timestamp`. Normalised HERE, once, on load -- not in each caller, where the second
author to forget it reintroduces it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.data_store.schema import Tables

#: The `fundamentals_check` columns a report or a cluster build actually needs. A projection,
#: not a convenience: the ledger grows ~12k rows per run and `roll_up_children` / `role_uri` /
#: `root_anchor` are read by nothing here. AGENTS.md forbids an unprojected read outright.
FINDING_READ_COLUMNS: tuple[str, ...] = (
    "run_date", "run_id", "cluster_id", "check_name", "ticker", "field", "period_key",
    "finding_id", "tier", "severity", "substrate", "observed", "expected", "deviation",
    "source_concept", "resolution_method", "accession_number", "edgar_url", "detail")

#: Date columns per frame, forced to `datetime64[ns]` on load. See the module docstring.
_DATE_COLUMNS: dict[str, tuple[str, ...]] = {
    "findings": ("run_date",),
    "runs": ("run_date",),
    "status": ("decided_at",),
}


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One row of `fundamentals_check_run`, reduced to what a caller asks about a RUN.

    The per-check columns stay in the frame; this is the run-level view, which is identical on
    every one of a run's ~35 rows and is what `comparable_runs` compares.
    """

    run_id: str
    run_date: pd.Timestamp
    scope_hash: str
    scope_roster: str | None
    scope_tickers: int
    scope_fields: str
    scope_tiers: str

    @property
    def label(self) -> str:
        """`2026-08-24 (54 tickers, in_sample)` -- how a delta names the run it differenced."""
        roster = f", {self.scope_roster}" if self.scope_roster else ""
        return f"{self.run_date.date()} ({self.scope_tickers} tickers{roster})"


class Ledger:
    """Read-only access to the three validator tables. Construct from a live `Context`.

    Deliberately a class rather than loose functions: the frames are read once and reused by
    the cluster build, the delta and the report, and three separate module-level readers is
    exactly how `substrate.py`'s "re-reads its tables per check" risk gets reintroduced in a
    new place.
    """

    def __init__(self, findings: pd.DataFrame, runs: pd.DataFrame,
                 status: pd.DataFrame) -> None:
        self.findings = findings
        self.runs = runs
        self.status = status

    # ------------------------------------------------------------------------- loading ---
    @classmethod
    def load(cls, context, *, run_id: str | None = None,
             scope_hash: str | None = None) -> "Ledger":
        """Read the three tables, scoped as narrowly as the caller allows.

        `run_id` reads ONE run -- what `validate report --run-id X` needs. `scope_hash` reads
        every comparable run, which is what a delta needs. Neither given reads everything,
        which is only appropriate on a small ledger and is why both filters exist.
        """
        runs = context.store.load(Tables.fundamentals_check_run, columns=None, optional=True)
        runs = _normalise(runs, _DATE_COLUMNS["runs"])
        if scope_hash and not runs.empty:
            runs = runs[runs["scope_hash"] == scope_hash]

        where: dict[str, Any] | None = None
        if run_id:
            where = {"run_id": run_id}
        elif scope_hash and not runs.empty:
            where = {"run_id": sorted(runs["run_id"].dropna().unique())}
        findings = context.store.load(Tables.fundamentals_check,
                                      columns=list(FINDING_READ_COLUMNS),
                                      where=where, optional=True)
        status = context.store.load(Tables.fundamentals_check_status, columns=None,
                                    optional=True)
        return cls(findings=_normalise(findings, _DATE_COLUMNS["findings"]),
                   runs=runs,
                   status=_normalise(status, _DATE_COLUMNS["status"]))

    # --------------------------------------------------------------------------- runs ---
    def run(self, run_id: str) -> RunRecord | None:
        """One run's scope record, or None when the id is not in the table."""
        if self.runs.empty:
            return None
        rows = self.runs[self.runs["run_id"] == run_id]
        return _record(rows.iloc[0]) if len(rows) else None

    def comparable_runs(self, run_id: str) -> list[RunRecord]:
        """Every OTHER run whose scope hash matches `run_id`'s, newest first.

        This is the whole reason `scope_hash` is stored. A run scoped to different tickers,
        fields or tiers is not a worse comparison -- it is not a comparison, and the report
        omits the delta with a note rather than rendering a number that means nothing.
        """
        current = self.run(run_id)
        if current is None or self.runs.empty:
            return []
        peers = self.runs[(self.runs["scope_hash"] == current.scope_hash)
                          & (self.runs["run_id"] != run_id)]
        records = {row["run_id"]: _record(row) for _, row in peers.iterrows()}
        return sorted(records.values(), key=lambda r: r.run_date, reverse=True)

    def previous_comparable(self, run_id: str) -> RunRecord | None:
        """The most recent comparable run STRICTLY BEFORE this one, or None.

        Strictly before, by date: a run recorded later the same day at the same scope IS this
        run (same `run_id`), and comparing a run to a peer that came after it would report a
        fix as a regression.
        """
        current = self.run(run_id)
        if current is None:
            return None
        earlier = [r for r in self.comparable_runs(run_id) if r.run_date < current.run_date]
        return earlier[0] if earlier else None

    # ----------------------------------------------------------------------- findings ---
    def findings_for(self, run_id: str) -> pd.DataFrame:
        """One run's findings."""
        if self.findings.empty:
            return self.findings
        return self.findings[self.findings["run_id"] == run_id]

    def check_health(self, run_id: str) -> pd.DataFrame:
        """One run's per-check row: examined, queued, info, ceiling, abstained, over_ceiling.

        The check-health gate reads THIS rather than recomputing rates from the findings. A
        rate recomputed against a ceiling that has since moved answers a different question
        than the run asked, and an ABSTENTION cannot be recomputed from findings at all --
        a check that examined nothing leaves no rows to count.
        """
        if self.runs.empty:
            return self.runs
        return self.runs[self.runs["run_id"] == run_id].sort_values(["tier", "check_name"])

    def cluster_history(self, scope_hash: str) -> pd.DataFrame:
        """Per cluster, across every comparable run: `first_seen`, `last_seen`, `runs_open`.

        "How long has this been broken?" is the question that separates a defect somebody
        introduced last night from one that has survived nine runs and a triage pass. It is
        only answerable across runs of ONE scope -- a cluster absent from a narrower run did
        not close, it was never looked at.
        """
        empty = pd.DataFrame(columns=["cluster_id", "first_seen", "last_seen", "runs_open"])
        if self.findings.empty or self.runs.empty:
            return empty
        ids = set(self.runs.loc[self.runs["scope_hash"] == scope_hash, "run_id"].dropna())
        rows = self.findings[self.findings["run_id"].isin(ids)]
        if rows.empty:
            return empty
        grouped = rows.groupby("cluster_id").agg(
            first_seen=("run_date", "min"), last_seen=("run_date", "max"),
            runs_open=("run_id", "nunique"))
        return grouped.reset_index()

    # ------------------------------------------------------------------------- status ---
    def status_map(self) -> dict[str, dict[str, Any]]:
        """`{cluster_id: its status row}`. Empty when nobody has decided anything yet."""
        if self.status.empty:
            return {}
        return {str(row["cluster_id"]): dict(row) for _, row in self.status.iterrows()}


def _record(row) -> RunRecord:
    return RunRecord(
        run_id=str(row["run_id"]),
        run_date=pd.Timestamp(row["run_date"]),
        scope_hash=str(row.get("scope_hash") or ""),
        scope_roster=(str(row["scope_roster"]) if row.get("scope_roster") else None),
        scope_tickers=int(row.get("scope_tickers") or 0),
        scope_fields=str(row.get("scope_fields") or "[]"),
        scope_tiers=str(row.get("scope_tiers") or "[]"))


def _normalise(frame: pd.DataFrame | None, date_columns: tuple[str, ...]) -> pd.DataFrame:
    """An empty frame for a missing table, and every date column as `datetime64[ns]`.

    An EMPTY frame rather than None so a caller never has to branch on `is None` before it can
    ask a question -- the same choice `substrate.py` makes, for the same reason.
    """
    if frame is None:
        return pd.DataFrame()
    out = frame.copy()
    for column in date_columns:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    return out


__all__ = ["FINDING_READ_COLUMNS", "Ledger", "RunRecord"]
