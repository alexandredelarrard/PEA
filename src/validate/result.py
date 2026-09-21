"""
result.py  (src/validate/result.py)
--------------------------------------------------------------------------------------------
The one shape every check in `src/validate/checks/` returns, and the exit code it maps to.

THREE statuses, not two. `abstain` exists because the failure mode this package is built
against is a check that measured nothing and reported a zero: a table that declares no clip
convention has no on-clip share, and "0 legs over the limit" reads identically whether the
check looked and found none or never looked at all. So a check missing its declaration
returns `abstain` and the CLI exits **3** -- a code no caller mistakes for a pass.

`scope` is mandatory and carries what the check was computed INDEPENDENTLY OF. That is what
makes a number evidence rather than an assertion, and it is what `data-check.md`'s Method note
is built from. A finding without its scope is a rumour with a decimal point.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

#: Process exit code per status. `abstain` is deliberately NOT 0: see the module docstring.
EXIT: dict[str, int] = {"pass": 0, "fail": 1, "abstain": 3}

#: The severity ladder of `.claude/commands/data-check.md` -- score 1-10 anchored on
#: PROVABILITY, then weighted by blast radius. Held here as a mapping so every check labels
#: the same score the same way; the prose defining each band stays in `data-check.md`.
_SEVERITY_BANDS: tuple[tuple[int, str], ...] = ((9, "critical"), (7, "high"), (4, "medium"), (1, "info"))

#: The lowest score that turns a run red. Findings BELOW it are `info` -- recorded in the
#: JSON, printed in the summary, and deliberately not a failure.
#:
#: ⚠ THIS EXISTS FOR A MEASURED FALSE POSITIVE. `coverage` must report the nine declared
#: universe exclusions (`FDXF GEHC GEV HONA KVUE Q SNDK SOLV VLTO`) so a reader can see they
#: were considered, but filing them as defects is exactly the false positive
#: `momentum/_scripts/08` recorded as D-08: they are absent BY DECLARATION. A check with
#: nowhere to put "I looked at this and it is fine" either hides the evidence or fails on it.
_FAIL_FLOOR: int = 4


def severity_for(score: int) -> str:
    """The `data-check.md` band a 1-10 score falls in."""
    for floor, name in _SEVERITY_BANDS:
        if score >= floor:
            return name
    return "info"


def jsonable(value: Any) -> Any:
    """`value` as something `json.dumps` accepts, recursively.

    Numpy scalars and `pd.Timestamp` both survive `dataclasses.asdict` untouched and both
    raise in `json.dumps`, and a check that computed a perfect answer then died writing it out
    has still measured nothing anybody can read."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        # JSON has no NaN/Infinity; `null` is the only honest encoding and `allow_nan=False`
        # would abort the write instead.
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass(frozen=True)
class Finding:
    """One defect, clustered by whatever the defect is a property OF -- a leg, a ticker, a
    pair -- never per row. One badly-adjusted ticker produces sixty failing rows, and sixty
    findings for one cause is how a report stops being read. The row counts live in
    `evidence`."""

    score: int
    severity: str
    field: str | None
    ticker: str | None
    observed: str
    expected: str
    evidence: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def at(cls, score: int, observed: str, expected: str, *, field: str | None = None,
           ticker: str | None = None, **evidence: Any) -> "Finding":
        """Build a finding with the severity derived from the score, so the two cannot drift."""
        return cls(score=score, severity=severity_for(score), field=field, ticker=ticker,
                   observed=observed, expected=expected, evidence=evidence)


@dataclass(frozen=True)
class CheckResult:
    """One check's outcome on one table."""

    check: str
    table: str
    status: str
    reason: str = ""
    scope: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def abstained(cls, check: str, table: str, reason: str, **scope: Any) -> "CheckResult":
        return cls(check=check, table=table, status="abstain", reason=reason, scope=scope)

    @classmethod
    def measured(cls, check: str, table: str, findings: list[Finding], *,
                 scope: dict[str, Any], metrics: dict[str, Any], reason: str = "") -> "CheckResult":
        """`pass` iff nothing above the `info` band was filed -- see `_FAIL_FLOOR`. A check
        that measured and found nothing is the only thing allowed to report a pass."""
        red = any(f.score >= _FAIL_FLOOR for f in findings)
        return cls(check=check, table=table, status="fail" if red else "pass",
                   reason=reason, scope=scope, findings=list(findings), metrics=metrics)

    @property
    def worst_score(self) -> int:
        return max((f.score for f in self.findings), default=0)

    def summary(self) -> str:
        head = f"[{self.status.upper()}] {self.check} on {self.table}"
        if self.status == "abstain":
            return f"{head} -- {self.reason}"
        rows = self.scope.get("rows")
        span = f"{self.scope.get('first_date')} -> {self.scope.get('last_date')}"
        line = (f"{head}: {len(self.findings)} finding(s), worst score {self.worst_score}"
                f"; scope {rows:,} rows, {self.scope.get('tickers')} tickers, {span}"
                if rows is not None else
                f"{head}: {len(self.findings)} finding(s), worst score {self.worst_score}")
        if self.reason:
            line += f"\n  note: {self.reason}"
        return line

    def to_json(self) -> dict[str, Any]:
        return jsonable({
            "check": self.check,
            "table": self.table,
            "status": self.status,
            "exit_code": EXIT[self.status],
            "reason": self.reason,
            "worst_score": self.worst_score,
            "scope": self.scope,
            "findings": [{"score": f.score, "severity": f.severity, "field": f.field,
                          "ticker": f.ticker, "observed": f.observed, "expected": f.expected,
                          "evidence": f.evidence} for f in self.findings],
            "metrics": self.metrics,
        })


def full_table_only(check: str, table: str, tickers: list[str] | None) -> CheckResult | None:
    """ABSTAIN rather than answer a full-table question about a subset.

    `-t/--tickers` is meaningful for the per-ticker checks and meaningless for the ones whose
    whole claim is "over every row of this table". Silently ignoring the flag would print a
    full-table verdict for a run the caller believes was scoped to three names; honouring it
    would print a three-name verdict under a check whose findings are worded as table-wide.
    Neither is a result, so the check declines to run."""
    if not tickers:
        return None
    return CheckResult.abstained(
        check, table,
        f"`{check}` always measures the full table, and -t/--tickers was given "
        f"({', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''}) -- a subset answer "
        f"under a table-wide finding is not a result. Drop -t, or use a per-ticker check.",
        tickers=tickers)


def gate(result: CheckResult) -> tuple[bool, str]:
    """`(ok, one-line reason)`, the shape `step_build_cube` already consumes from
    `utils/prices.gate`. An abstain is NOT ok: it has not earned a pass."""
    return result.status == "pass", result.summary()
