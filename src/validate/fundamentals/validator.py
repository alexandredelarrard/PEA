"""
validator.py  (src/validate/fundamentals/)
--------------------------------------------------------------------------------------------
`FundamentalsValidator`: run the registry over the substrates and hand back a run -- the
findings, what each check examined, and the SCOPE the run covered.

## THE ONE THING THIS CLASS PROMISES: IT NEVER MUTATES ANYTHING BUT ITS OWN THREE TABLES

Decision 40, and it is load-bearing rather than stylistic. v2's design had a "Layer A" that
nulled impossible values by UPDATE-ing `fundamentals_history` after the fact. That contradicts
the table's append-only contract in the most damaging possible way: a historical row would
change value after publication, so yesterday's cube and today's would disagree about the same
event, and `build_history.diff_against_stored` -- the guard that makes the append-only property
enforced rather than asserted -- would start reporting drift it had itself caused.

So the guards moved INTO the builder, where they run before the row is written, and this class
only ever reports. `run()` returns a `ValidationRun`; `write()` persists it. Nothing else.

## NOTHING IS SUBTRACTED (D5), and the JSON register that did the subtracting is GONE

Until this rebuild, `run()` subtracted findings that `configs/fundamentals/fundamentals_check
.json` had settled, so the queue shrank as work was accepted. It also made the ledger's own
row count meaningless: a run with fewer rows than yesterday's could mean the defect was fixed,
or that somebody had added a suppression, and no query could separate the two. The register's
own documentation opened with "THE REGISTER IS NOT A SUPPRESSION LIST", which is the kind of
thing a design only has to say when its shape makes the opposite easy.

Now every finding of every run is written. A `wontfix` is recorded in
`fundamentals_check_status` and applied when the report is RENDERED. The ledger says what the
checks found; the status table says what a human decided; and a row-count drop between two
runs OF THE SAME SCOPE has exactly one cause.

## Scope is recorded, because a delta without it is not evidence

`RunScope` hashes the tickers, fields and tiers. Two runs are comparable iff their
`scope_hash` matches -- see `scope.py`. Without that, re-validating a fix on one ticker after
a 54-ticker baseline reports ~11,800 findings "closed".

## Load once, pass the frame down (Phase 10's named risk)

`Substrates` is built once per run and handed to every check. A check has no `store`, so it
cannot re-read; `tests/validate/` asserts the load count, so a future author cannot add one.

## Nothing gates (decision 45)

`run()` has no failure mode that stops a caller. The nightly extraction runs to completion
whatever this finds -- one filer's bad quarter must never stall the other 499. That is the
SEC's own warn-over-reject precedent, taken deliberately.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Iterable

import pandas as pd

from src.data_extract.utils.fundamentals.kpi_catalogue import Catalogue, load_catalogue
from src.data_store.schema import Tables
from src.validate.fundamentals.checks import CHECK_REGISTRY, CheckSpec, checks_for
from src.validate.fundamentals.finding import (
    Finding, FINDING_COLUMNS, QUEUE_SEVERITIES, SEVERITY_ORDER, findings_frame)
from src.validate.fundamentals.scope import RunScope
from src.validate.fundamentals.substrate import Substrates, load as load_substrates

#: Every column of `fundamentals_check_run`, in order. Declared ONCE, like `FINDING_COLUMNS`,
#: so the DDL, the writer and the read-back cannot drift apart.
CHECK_RUN_COLUMNS: tuple[str, ...] = (
    "run_id", "run_date", "check_name",
    "scope_hash", "scope_roster", "scope_tickers", "scope_ticker_list",
    "scope_fields", "scope_tiers",
    "tier", "substrate",
    "examined", "queued", "info", "ceiling", "abstained", "over_ceiling",
)


@dataclass(frozen=True, slots=True)
class CheckOutcome:
    """What one check did: its findings, what it examined, and whether it over-fired.

    `examined` and `fire_rate` exist so the calibration report is a MEASUREMENT rather than a
    list. A check reporting 400 findings has said nothing until you know whether that is 400
    out of 3,000 or out of 3,000,000.
    """

    spec: CheckSpec
    findings: list[Finding]
    examined: int

    @property
    def queued(self) -> int:
        """Findings that actually reach the work queue -- `critical` / `high` / `medium`.

        `info` is excluded BY CONSTRUCTION, which is what makes `info` a usable home for
        `restatement_ledger`, `amendment_ledger` and every abstention notice.
        """
        return sum(1 for f in self.findings if f.severity in QUEUE_SEVERITIES)

    @property
    def info(self) -> int:
        """Findings that are declared and quantified but are not work."""
        return len(self.findings) - self.queued

    @property
    def fire_rate(self) -> float:
        """QUEUE findings per unit examined. 0.0 when nothing was examined -- an ABSTENTION,
        which must never be reported as a clean 0% pass.

        Queue findings and not TOTAL findings, and the distinction is the whole point of the
        ceiling. The ceiling exists to answer one question -- *is this check burying real
        findings under itself?* -- and an `info` finding cannot bury anything, because nothing
        reads it as work. The first calibration run measured `series_shape` at 29.1% and
        flagged it a threshold bug when 1,045 of its 1,632 findings were `info` gap codes that
        are benign by construction. That was the METRIC misreading the check.
        """
        return self.queued / self.examined if self.examined else 0.0

    @property
    def abstained(self) -> bool:
        """Did this check have nothing to look at? Reported distinctly from "found nothing"."""
        return self.examined == 0

    @property
    def over_ceiling(self) -> bool:
        """Is this check firing above the rate it DECLARED it expects?

        DQC_0118's lesson, enforced instead of left to a human reading a table: a check over
        its own ceiling has a threshold bug until proven otherwise, and it buries every real
        finding under itself.
        """
        return (not self.abstained
                and self.fire_rate > self.spec.expected_fire_rate_ceiling)


@dataclass
class ValidationRun:
    """One complete run: the frames to write, the per-check outcomes, and the scope covered."""

    run_date: pd.Timestamp
    findings: pd.DataFrame
    outcomes: list[CheckOutcome]
    scope: RunScope = dataclass_field(default_factory=RunScope)

    @property
    def run_id(self) -> str:
        """This run's identity. Every finding row carries it; the delta is computed on it."""
        return self.scope.run_id(self.run_date)

    @property
    def tickers(self) -> tuple[str, ...]:
        """The tickers actually loaded. "0 findings" and "0 tickers" must never look alike."""
        return self.scope.tickers

    @property
    def queue(self) -> pd.DataFrame:
        """Findings at a severity an agent works, worst first.

        `info` is excluded BY CONSTRUCTION -- that is what makes `info` a usable home for
        `restatement_ledger` and every probation-field finding without drowning the queue.
        NOTHING settled is removed here: a `wontfix` is applied when the report is rendered,
        so that this frame and the table it is written to always agree.
        """
        if self.findings.empty:
            return self.findings
        open_rows = self.findings[self.findings["severity"].isin(QUEUE_SEVERITIES)].copy()
        order = {severity: i for i, severity in enumerate(SEVERITY_ORDER)}
        open_rows["_rank"] = open_rows["severity"].map(order)
        return (open_rows.sort_values(["_rank", "check_name", "ticker", "field"])
                .drop(columns="_rank"))

    @property
    def over_ceiling(self) -> list[CheckOutcome]:
        """Checks that fired above their own declared ceiling -- threshold bugs, ranked."""
        return sorted((o for o in self.outcomes if o.over_ceiling),
                      key=lambda o: -o.fire_rate)

    @property
    def abstained(self) -> list[CheckOutcome]:
        """Checks that examined nothing. NOT a pass, and the report says which."""
        return sorted((o for o in self.outcomes if o.abstained), key=lambda o: o.spec.name)

    def check_runs(self) -> pd.DataFrame:
        """The `fundamentals_check_run` frame: one row per check, carrying the run's scope."""
        return check_run_frame(self)


def check_run_frame(run: ValidationRun) -> pd.DataFrame:
    """`run` as a `fundamentals_check_run`-shaped frame, columns pinned and dtypes forced.

    The dtype forcing is the same guard `findings_frame` carries and for the same reason:
    `sql/schema.sql` is applied only when Postgres INITIALISES a volume, so on a live one
    `store.save` creates a missing table from the FIRST frame it is handed -- and an all-None
    object column becomes TEXT, permanently. `scope_roster` is None on a `--tickers` run, and
    a first run of that shape would otherwise fix the column's type wrong forever.
    """
    scope_columns = run.scope.as_columns(run.run_date)
    rows = [{
        **scope_columns,
        "run_date": run.run_date.date(),
        "check_name": outcome.spec.name,
        "tier": outcome.spec.tier,
        "substrate": outcome.spec.substrate,
        "examined": outcome.examined,
        "queued": outcome.queued,
        "info": outcome.info,
        "ceiling": outcome.spec.expected_fire_rate_ceiling,
        "abstained": outcome.abstained,
        "over_ceiling": outcome.over_ceiling,
    } for outcome in run.outcomes]

    frame = pd.DataFrame(rows, columns=list(CHECK_RUN_COLUMNS))
    for column in ("scope_tickers", "tier", "examined", "queued", "info"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    frame["ceiling"] = pd.to_numeric(frame["ceiling"], errors="coerce").astype(float)
    for column in ("abstained", "over_ceiling"):
        frame[column] = frame[column].fillna(False).astype(bool)
    for column in ("run_id", "check_name", "scope_hash", "scope_roster",
                   "scope_ticker_list", "scope_fields", "scope_tiers", "substrate"):
        frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
    return frame


class FundamentalsValidator:
    """Runs the checks. Instantiate against a `Substrates` -- no DB, no CLI, no network.

    That constructor signature IS the testing strategy. Every check is exercised by building a
    synthetic `Substrates` with one planted violation and asserting that exactly one check
    fires, exactly once. A check that cannot be planted cannot be trusted.
    """

    def __init__(self, substrates: Substrates) -> None:
        self._substrates = substrates

    @classmethod
    def from_context(cls, context, *, tickers: list[str] | None = None,
                     config_dir: str = "./configs", tiers: Iterable[int] | None = None,
                     since=None, catalogue: Catalogue | None = None
                     ) -> "FundamentalsValidator":
        """Build from a live `Context`: the only path that touches the database.

        `tiers` is forwarded purely so a Tier-1-only run can skip loading
        `fundamentals_facts`, which is by far the largest read here -- decision 53's nightly
        shape is Tier 1 over the whole table and Tiers 2-3 only where a filing landed.
        """
        catalogue = catalogue or load_catalogue(config_dir)
        need_facts = tiers is None or any(t in (2, 3) for t in tiers) or 1 in (tiers or ())
        substrates = load_substrates(context, catalogue, tickers, since=since,
                                     need_facts=need_facts)
        return cls(substrates)

    # ------------------------------------------------------------------------------ run ---
    def run(self, *, tiers: Iterable[int] | None = None,
            names: Iterable[str] | None = None,
            fields: Iterable[str] | None = None,
            roster: str = "", run_date=None) -> ValidationRun:
        """Every selected check, in registry order. NOTHING is subtracted (D5).

        `fields` narrows to a single catalogue field -- the per-field ACCEPTANCE SHEET
        (decision 44). Adding a field to the catalogue is not finished when it resolves; it is
        finished when `validate fundamentals --field X --roster in_sample` is clean or its gaps
        are recorded with evidence.

        `roster` is carried into the run purely as a LABEL for the report; it takes no part in
        the scope hash, because two runs covering the same tickers are comparable whether or
        not someone renamed the roster in between.
        """
        # NOT `.normalize()`-d: `run_id` hashes the hour (see `RunScope.run_id`), so the
        # default run_date must carry real clock time or every run of a day would hash alike.
        run_date = pd.Timestamp(run_date) if run_date is not None else pd.Timestamp.today()
        wanted_fields = set(fields) if fields else None
        selected = checks_for(tiers=tiers, names=names)
        # The tiers ACTUALLY RUN, not the flag: `--tier` unset and `--tier 1,2,3` cover the
        # same ground and must hash alike, or a nightly run would never be comparable to the
        # explicit re-validation an agent does after a fix.
        scope = RunScope.build(
            tickers=self._substrates.tickers, fields=fields,
            tiers=tiers if tiers is not None else sorted({s.tier for s in selected}),
            roster=roster)
        run_id = scope.run_id(run_date)

        outcomes: list[CheckOutcome] = []
        produced_all: list[Finding] = []
        for spec in selected:
            produced = spec.fn(self._substrates)
            if wanted_fields is not None:
                produced = [f for f in produced if f.field in wanted_fields]
            produced_all.extend(produced)
            outcomes.append(CheckOutcome(
                spec=spec, findings=produced,
                examined=self._substrates.denominator(spec.name)))

        return ValidationRun(
            run_date=run_date,
            findings=findings_frame(produced_all, run_date, run_id),
            outcomes=outcomes, scope=scope)

    # ---------------------------------------------------------------------------- write ---
    @staticmethod
    def write(context, run: ValidationRun) -> int:
        """Persist the run: `fundamentals_check_run` first, then `fundamentals_check`.

        ## The run REPLACES ITS OWN ROWS before writing them

        `run_id` is (run_hour, scope), so a re-run within the same clock-hour at the same
        scope carries the same id -- which is exactly the shape of "fix it, rebuild,
        re-validate". A re-run more than an hour later gets a fresh id instead. Without
        the delete, findings that STOPPED firing would survive as leftovers from the morning's
        run and the measured delta would read 0. An upsert cannot remove a row that is no
        longer produced, so the run clears its own footprint and nobody else's.

        The run-metadata row is written FIRST and unconditionally. A run that found nothing
        must still record that it happened, at what scope, and which checks abstained --
        otherwise "the validator ran and found nothing" and "the validator did not run" are
        the same empty result, which is the one distinction this table exists to make.
        """
        run_id = run.run_id
        context.store.save(Tables.fundamentals_check_run, run.check_runs())

        if run_id:
            context.store.delete(Tables.fundamentals_check, {"run_id": run_id})
        if run.findings.empty:
            context.log.info("validate fundamentals: 0 findings, nothing written to %s "
                             "(run_id %s recorded)", Tables.fundamentals_check, run_id)
            return 0
        written = context.store.save(Tables.fundamentals_check, run.findings)
        context.log.info("validate fundamentals: wrote %d finding(s) to %s for run_id %s (%s)",
                         written, Tables.fundamentals_check, run_id, run.scope.describe())
        return written


def empty_findings() -> pd.DataFrame:
    """A correctly-shaped empty findings frame. For a caller that must return one regardless."""
    return pd.DataFrame(columns=list(FINDING_COLUMNS))


__all__ = ["CHECK_REGISTRY", "CHECK_RUN_COLUMNS", "CheckOutcome", "FundamentalsValidator",
           "RunScope", "ValidationRun", "check_run_frame", "empty_findings"]
