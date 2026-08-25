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

## A WAIVER is a state, a FIX is an event, and they are read differently

`fundamentals_check_status` is keyed `(cluster_id, check_name)` and is MUTABLE: it says "this
finding is real and we tolerate it". `fundamentals_check_fix` is keyed
`(cluster_id, run_id_after)` and is APPEND-ONLY: it says "we intervened, here is what and
why". Neither ever removes a row from `fundamentals_check` -- a waiver is applied when a
report is RENDERED, and a fix row is read only to decide whether a settlement is claimable.

`qualifying_fix` is the SETTLEMENT PREDICATE and it lives here alone. Two copies of a rule
this load-bearing drift, and the copy the renderer uses would then disagree with the copy the
tests pin.

## Postgres DATE round-trips as `datetime.date`

Never as a `Timestamp`. An unguarded `frame["run_date"] > some_timestamp` then does something
surprising, and a parquet-cached test harness hides the whole bug class because parquet hands
back `Timestamp`. Normalised HERE, once, on load -- not in each caller, where the second
author to forget it reintroduces it.
"""
from __future__ import annotations

import json
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

#: The `fundamentals_check_fix` columns a settlement decision or a `fix show` needs -- which
#: is every one of them. Declared as a projection anyway rather than passing `columns=None`:
#: AGENTS.md's rule is about the READ being explicit, and a column added to the table later
#: must reach a reader through a deliberate edit here, not by silently widening the frame.
FIX_READ_COLUMNS: tuple[str, ...] = (
    "cluster_id", "run_id_after", "run_id_before", "scope_hash", "ticker", "field",
    "findings_before", "findings_after", "queued_before", "queued_after",
    "layer", "root_cause", "evidence", "commit_sha", "test_path", "decided_at")

#: Date columns per frame, forced to `datetime64[ns]` on load. See the module docstring.
_DATE_COLUMNS: dict[str, tuple[str, ...]] = {
    "findings": ("run_date",),
    "runs": ("run_date",),
    "status": ("decided_at",),
    "fixes": ("decided_at",),
}

#: What `check_name` means in a `fundamentals_check_status` row when it is empty: the waiver
#: covers the WHOLE cluster rather than one check. A sentinel rather than a NULL because a
#: Postgres primary key cannot contain one -- the same reason `fundamentals_check.period_key`
#: uses `''` for a ticker-level finding.
CLUSTER_WIDE = ""


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


@dataclass(frozen=True, slots=True)
class FixRecord:
    """One row of `fundamentals_check_fix` -- ONE INTERVENTION, and what it measurably closed.

    Frozen, like `RunRecord`, because the row it mirrors is append-only: a fix is an event
    that happened, and revising the record of an event is how a ledger stops being evidence.

    The four counts are STORED rather than recomputed, on `findings_at_decision`'s precedent.
    A count re-derived today answers a different question than the one the fix was measured
    against, and it stops being answerable at all once the ledger is pruned.
    """

    cluster_id: str
    run_id_after: str
    run_id_before: str
    scope_hash: str
    ticker: str
    field: str
    findings_before: int
    findings_after: int
    queued_before: int
    queued_after: int
    layer: str
    root_cause: str
    evidence: str
    commit_sha: str
    test_path: str
    decided_at: pd.Timestamp | None

    @property
    def improved(self) -> bool:
        """Did this fix reduce the QUEUE? The one thing settlement turns on.

        Queue severities only -- `info` is excluded upstream, because nothing reads an `info`
        finding as work and a fix that closed only `info` rows has not closed any work.
        A `False` here does NOT make the row illegitimate: correcting a wrong-but-plausible
        value where no check was firing is a real fix. It makes it unable to SETTLE.
        """
        return self.queued_after < self.queued_before

    @property
    def evidence_json(self) -> dict[str, Any]:
        """`evidence` parsed. `{}` when it is absent or unparseable, never a raised error.

        A reader asking what a fix cited must not be blocked by one malformed row written
        before the CLI enforced the shape.
        """
        try:
            blob = json.loads(self.evidence or "{}")
        except (TypeError, ValueError):
            return {}
        return blob if isinstance(blob, dict) else {}

    @property
    def summary(self) -> str:
        """`extraction: 55 -> 4 queue finding(s) @2fb6ef2` -- one line for a render."""
        return (f"{self.layer}: {self.queued_before} -> {self.queued_after} queue finding(s) "
                f"@{self.commit_sha[:7] if self.commit_sha else '?'}")


class Ledger:
    """Read-only access to the four validator tables. Construct from a live `Context`.

    Deliberately a class rather than loose functions: the frames are read once and reused by
    the cluster build, the delta and the report, and three separate module-level readers is
    exactly how `substrate.py`'s "re-reads its tables per check" risk gets reintroduced in a
    new place.
    """

    def __init__(self, findings: pd.DataFrame, runs: pd.DataFrame,
                 status: pd.DataFrame, fixes: pd.DataFrame | None = None) -> None:
        self.findings = findings
        self.runs = runs
        self.status = status
        #: Defaulted rather than required: every existing caller built a three-table Ledger,
        #: and a ledger with no fixes on record is the normal state, not a degraded one.
        self.fixes = pd.DataFrame() if fixes is None else fixes

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
        # Never scoped by run: a fix is keyed on the run that PROVED it, and a reader asking
        # "has this cluster ever been fixed?" must see rows from runs it did not ask about.
        # The table gains one row per intervention, so it stays small by construction.
        fixes = context.store.load(Tables.fundamentals_check_fix,
                                   columns=list(FIX_READ_COLUMNS), optional=True)
        return cls(findings=_normalise(findings, _DATE_COLUMNS["findings"]),
                   runs=runs,
                   status=_normalise(status, _DATE_COLUMNS["status"]),
                   fixes=_normalise(fixes, _DATE_COLUMNS["fixes"]))

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
    def status_map(self) -> dict[str, dict[str, dict[str, Any]]]:
        """`{cluster_id: {check_name: its status row}}`. `''` is the CLUSTER-WIDE waiver.

        NESTED, and that is the whole ripple of widening the primary key. Returning
        `{cluster_id: row}` under the wider key would let two waivers on one cluster collide
        and the last one read would silently win -- a `peer_ratio` waiver quietly becoming a
        `series_shape` waiver depending on row order.

        Empty when nobody has decided anything yet, which is the normal state.
        """
        if self.status.empty:
            return {}
        out: dict[str, dict[str, dict[str, Any]]] = {}
        for _, row in self.status.iterrows():
            check = str(row.get("check_name") or CLUSTER_WIDE)
            out.setdefault(str(row["cluster_id"]), {})[check] = dict(row)
        return out

    def waivers_for(self, cluster_id: str) -> dict[str, dict[str, Any]]:
        """`{check_name: row}` for ONE cluster. `''` covers every check on it."""
        return self.status_map().get(cluster_id, {})

    # --------------------------------------------------------------------------- fixes ---
    def fixes_for(self, cluster_id: str) -> list[FixRecord]:
        """Every recorded intervention on one cluster, NEWEST FIRST.

        A list rather than one row: the table is append-only, so a cluster fixed twice has
        two rows and the second did not un-happen the first. "What has been tried here?" is a
        history, and answering it with the latest row alone hides the attempt that failed.
        """
        if self.fixes.empty:
            return []
        rows = self.fixes[self.fixes["cluster_id"].astype(str) == cluster_id]
        records = [_fix(row) for _, row in rows.iterrows()]
        return sorted(records, key=lambda f: (f.decided_at or pd.Timestamp.min,
                                              f.run_id_after), reverse=True)

    def qualifying_fix(self, cluster_id: str, scope_hash: str) -> FixRecord | None:
        """THE SETTLEMENT PREDICATE. The newest fix that can carry a settlement, or None.

        Two conditions, both necessary, and this is the only place either is written down:

          * the fix was measured at THIS scope. A fix proven on a one-ticker re-validation
            cannot settle a cluster in a 54-ticker run -- that is not a stronger proof or a
            weaker one, it is a different measurement, and it is the same test
            `comparable_runs` makes for exactly the same reason;
          * it REDUCED THE QUEUE. A row where `queued_after >= queued_before` stays on the
            record -- correcting a wrong-but-plausible value where no check was firing is a
            legitimate fix -- but it proves nothing was closed, so it cannot settle anything.

        Permissive to record, strict to settle. Callers must not re-derive this: a second
        copy of the rule is a copy that drifts from the one the tests pin.
        """
        for record in self.fixes_for(cluster_id):
            if record.scope_hash == scope_hash and record.improved:
                return record
        return None


def _record(row) -> RunRecord:
    return RunRecord(
        run_id=str(row["run_id"]),
        run_date=pd.Timestamp(row["run_date"]),
        scope_hash=str(row.get("scope_hash") or ""),
        scope_roster=(str(row["scope_roster"]) if row.get("scope_roster") else None),
        scope_tickers=int(row.get("scope_tickers") or 0),
        scope_fields=str(row.get("scope_fields") or "[]"),
        scope_tiers=str(row.get("scope_tiers") or "[]"))


def _fix(row) -> FixRecord:
    """One `fundamentals_check_fix` row as a `FixRecord`, missing cells coerced not raised.

    A row written before a column existed reads as `''` / `0` rather than blowing up a
    report: the table is append-only, so old rows are exactly the ones that cannot be
    rewritten to match a newer shape.
    """
    decided = row.get("decided_at")
    return FixRecord(
        cluster_id=str(row["cluster_id"]),
        run_id_after=str(row.get("run_id_after") or ""),
        run_id_before=str(row.get("run_id_before") or ""),
        scope_hash=str(row.get("scope_hash") or ""),
        ticker=str(row.get("ticker") or ""),
        field=str(row.get("field") or ""),
        findings_before=_count(row.get("findings_before")),
        findings_after=_count(row.get("findings_after")),
        queued_before=_count(row.get("queued_before")),
        queued_after=_count(row.get("queued_after")),
        layer=str(row.get("layer") or ""),
        root_cause=str(row.get("root_cause") or ""),
        evidence=str(row.get("evidence") or ""),
        commit_sha=str(row.get("commit_sha") or ""),
        test_path=str(row.get("test_path") or ""),
        decided_at=(pd.Timestamp(decided) if pd.notna(decided) else None))


def _count(value) -> int:
    """A stored BIGINT as `int`. `pd.NA` / None / NaN -> 0, never a raised error."""
    return 0 if value is None or pd.isna(value) else int(value)


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
            out[column] = _nanoseconds(out[column])
    return out


def _nanoseconds(column: pd.Series) -> pd.Series:
    """One column as `datetime64[ns]`, whatever resolution it arrived in.

    The second cast is not redundant. Pandas 2.x INFERS the unit: a column of Postgres
    `datetime.date` objects converts to `datetime64[s]`, while the same column round-tripped
    through parquet comes back `datetime64[ns]`. Two frames of different resolution then
    merge and compare in ways that depend on which path loaded them, which is precisely the
    class of surprise this module normalises to remove -- so the unit is pinned, not inferred.
    """
    converted = pd.to_datetime(column, errors="coerce")
    return (converted.astype("datetime64[ns]")
            if pd.api.types.is_datetime64_any_dtype(converted) else converted)


__all__ = ["CLUSTER_WIDE", "FINDING_READ_COLUMNS", "FIX_READ_COLUMNS", "FixRecord",
           "Ledger", "RunRecord"]
