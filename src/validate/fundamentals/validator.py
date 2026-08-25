"""
validator.py  (src/validate/fundamentals/)
--------------------------------------------------------------------------------------------
`FundamentalsValidator`: run the registry over the substrates, subtract what is settled, and
hand back a ranked queue.

## THE ONE THING THIS CLASS PROMISES: IT NEVER MUTATES ANYTHING BUT `fundamentals_check`

Decision 40, and it is load-bearing rather than stylistic. v2's design had a "Layer A" that
nulled impossible values by UPDATE-ing `fundamentals_history` after the fact. That contradicts
the table's append-only contract in the most damaging possible way: a historical row would
change value after publication, so yesterday's cube and today's would disagree about the same
event, and `build_history.diff_against_stored` -- the guard that makes the append-only property
enforced rather than asserted -- would start reporting drift it had itself caused.

So the guards moved INTO the builder, where they run before the row is written, and this class
only ever reports. `run()` returns a frame; `write()` appends it. Nothing else.

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
from src.validate.fundamentals.check_register import CheckRegister, load_register
from src.validate.fundamentals.checks import CHECK_REGISTRY, CheckSpec, checks_for
from src.validate.fundamentals.finding import (
    Finding, FINDING_COLUMNS, QUEUE_SEVERITIES, SEVERITY_ORDER, findings_frame)
from src.validate.fundamentals.substrate import Substrates, load as load_substrates


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
    settled: int = 0

    @property
    def queued(self) -> int:
        """Findings that actually reach the work queue -- `critical` / `high` / `medium`.

        `info` is excluded BY CONSTRUCTION, which is what makes `info` a usable home for
        `register_cost`, `restatement_ledger`, `amendment_ledger` and every abstention notice.
        """
        return sum(1 for f in self.findings if f.severity in QUEUE_SEVERITIES)

    @property
    def fire_rate(self) -> float:
        """QUEUE findings per unit examined. 0.0 when nothing was examined -- an ABSTENTION,
        which must never be reported as a clean 0% pass.

        Queue findings and not TOTAL findings, and the distinction is the whole point of the
        ceiling. The ceiling exists to answer one question -- *is this check burying real
        findings under itself?* -- and an `info` finding cannot bury anything, because nothing
        reads it as work. The first calibration run measured `series_shape` at 29.1% and
        flagged it a threshold bug when 1,045 of its 1,632 findings were `info` gap codes that
        are benign by construction; `register_cost` is `info`-only and reported 825.9%. Both
        were the METRIC misreading the check, not the check misreading the data.
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
    """One complete run: the frame to write, the per-check outcomes, and the register's state."""

    run_date: pd.Timestamp
    findings: pd.DataFrame
    outcomes: list[CheckOutcome]
    register: CheckRegister
    tickers: tuple[str, ...] = ()
    settled_total: int = 0
    stale_entries: list = dataclass_field(default_factory=list)

    @property
    def queue(self) -> pd.DataFrame:
        """OPEN findings at a severity an agent works, worst first.

        `info` is excluded BY CONSTRUCTION -- that is what makes `info` a usable home for
        `register_cost`, `restatement_ledger` and every probation-field finding without
        drowning the queue. Settled findings are already gone; they were subtracted in `run()`.
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


class FundamentalsValidator:
    """Runs the checks. Instantiate against a `Substrates` -- no DB, no CLI, no network.

    That constructor signature IS the testing strategy. Every check is exercised by building a
    synthetic `Substrates` with one planted violation and asserting that exactly one check
    fires, exactly once. A check that cannot be planted cannot be trusted.
    """

    def __init__(self, substrates: Substrates, *,
                 register: CheckRegister | None = None) -> None:
        self._substrates = substrates
        self._register = register if register is not None else CheckRegister([])

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
        return cls(substrates, register=load_register(config_dir))

    # ------------------------------------------------------------------------------ run ---
    def run(self, *, tiers: Iterable[int] | None = None,
            names: Iterable[str] | None = None,
            fields: Iterable[str] | None = None,
            run_date=None) -> ValidationRun:
        """Every selected check, in registry order, with settled findings subtracted.

        `fields` narrows to a single catalogue field -- the per-field ACCEPTANCE SHEET
        (decision 44). Adding a field to the catalogue is not finished when it resolves; it is
        finished when `validate fundamentals --field X --roster in_sample` is clean or its gaps
        are recorded with evidence.
        """
        run_date = pd.Timestamp(run_date or pd.Timestamp.today().normalize())
        wanted_fields = set(fields) if fields else None
        outcomes: list[CheckOutcome] = []
        kept: list[Finding] = []
        fired_ids: list[str] = []
        settled_total = 0

        for spec in checks_for(tiers=tiers, names=names):
            produced = spec.fn(self._substrates)
            if wanted_fields is not None:
                produced = [f for f in produced if f.field in wanted_fields]
            fired_ids.extend(f.id for f in produced)
            open_findings = [f for f in produced if not self._register.is_settled(f.id)]
            settled = len(produced) - len(open_findings)
            settled_total += settled
            kept.extend(open_findings)
            outcomes.append(CheckOutcome(
                spec=spec, findings=open_findings,
                examined=self._substrates.denominator(spec.name), settled=settled))

        return ValidationRun(
            run_date=run_date, findings=findings_frame(kept, run_date), outcomes=outcomes,
            register=self._register, tickers=self._substrates.tickers,
            settled_total=settled_total,
            stale_entries=self._register.stale(fired_ids))

    # ---------------------------------------------------------------------------- write ---
    @staticmethod
    def write(context, run: ValidationRun) -> int:
        """Append the run's findings to `fundamentals_check`. The ONLY write in the package.

        An empty run writes nothing and says so. That is not a no-op worth optimising away --
        it is the difference between "the validator ran and found nothing" and "the validator
        did not run", and only the row count can tell them apart.
        """
        if run.findings.empty:
            context.log.info("validate fundamentals: 0 findings, nothing written to %s",
                             Tables.fundamentals_check)
            return 0
        written = context.store.save(Tables.fundamentals_check, run.findings)
        context.log.info("validate fundamentals: wrote %d finding(s) to %s for run_date %s",
                         written, Tables.fundamentals_check, run.run_date.date())
        return written


def empty_findings() -> pd.DataFrame:
    """A correctly-shaped empty findings frame. For a caller that must return one regardless."""
    return pd.DataFrame(columns=list(FINDING_COLUMNS))


__all__ = ["CHECK_REGISTRY", "CheckOutcome", "FundamentalsValidator", "ValidationRun",
           "empty_findings"]
