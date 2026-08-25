"""
finding.py  (src/validate/fundamentals/)
--------------------------------------------------------------------------------------------
ONE finding: the self-contained investigation packet plan-5b decision 47 specifies, plus the
`finding_id` that gives it an identity across runs.

## Why the payload is fat

The obvious design is an identity-only row plus an on-demand join back to `fundamentals_facts`
for provenance. It does not work here, and not marginally: a Tier-2 or Tier-3 finding on a
DERIVED value -- a trailing-twelve total, a `derived_identity` `totalLiabilities`, a computed
ratio -- has **no single fact row to join to**. Half the queue would arrive with an empty
provenance block and the reviewing agent would have to re-derive the number before it could
even read the finding. So the packet carries its own evidence.

The other rejected shape was a prose `message`. An agent parsing English to decide what to fix
is precisely the failure mode this rebuild exists to remove; `detail` is JSON, and the check
that wrote it declares what is in it.

## Why `finding_id` excludes `run_date` and `severity`

`fundamentals_check` is append-only and keyed on `run_date`, so the ROWS are per-run. What has
to survive across runs is the *finding*: `configs/fundamentals/fundamentals_check.json` records
a settled outcome against a `finding_id`, and that record has to keep matching tomorrow.

So the id hashes exactly the four components that make two observations "the same finding":
check, ticker, field, period_key. It deliberately does NOT include:

  * `run_date` -- an id that changed nightly would settle nothing;
  * `severity` -- a threshold retune moves severities constantly, and an accepted finding must
    not resurrect because a check was recalibrated;
  * `observed` -- the whole point is that the value can be re-measured and still be the same
    known finding. A CHANGED value on a settled finding is caught by the register's staleness
    report instead, which is the honest place for it.

## `period_key` is polymorphic, and that is deliberate

TEXT, by grain: the `as_of` for a history-grain check, the `period_end` for a facts-grain one,
`''` for a ticker-level check, and `start..end` for a series-grain one. One key column rather
than three nullable ones, because a Postgres PK cannot contain a NULL and a sentinel date
would be a lie about which period the finding is about.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field as dataclass_field
from typing import Any

import pandas as pd

#: The severity ladder (decision 49). PURELY the reviewing agent's queue order -- nothing
#: gates, so severity cannot mean "how bad is the impact"; it means **how sure are we the
#: number is wrong**. An impact ladder was rejected: it needs a per-field weighting nobody has
#: measured, and it hides small provable defects that indicate a systematic bug.
CRITICAL = "critical"   # provably wrong, or a structural contract is broken
HIGH = "high"           # probably wrong, and a NAMED mechanism says so
MEDIUM = "medium"       # a statistical candidate; look, do not assume
INFO = "info"           # declared, quantified, no action expected -- never enters the queue

#: Ordered worst-first. The report and the work queue both enumerate THIS, never a hand list.
SEVERITY_ORDER: tuple[str, ...] = (CRITICAL, HIGH, MEDIUM, INFO)

#: Severities an agent is expected to work. `info` is excluded by construction, which is what
#: makes `info` a usable place to put `register_cost`, `restatement_ledger` and every
#: probation-field finding without drowning the queue.
QUEUE_SEVERITIES: frozenset[str] = frozenset({CRITICAL, HIGH, MEDIUM})

#: EDGAR's canonical accession URL. A finding without one costs the reviewing agent a lookup
#: before it can read the filing, which is the one step decision 47 exists to remove.
_EDGAR_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{plain}/{accession}-index.htm"

#: Every column of `fundamentals_check`, in order. Declared ONCE so the empty-result path,
#: the populated path and the DDL cannot drift apart.
FINDING_COLUMNS: tuple[str, ...] = (
    # identity -- the PK, then the cross-run id
    "run_date", "check_name", "ticker", "field", "period_key", "finding_id",
    # classification
    "tier", "severity", "substrate",
    # the claim
    "observed", "expected", "deviation",
    # provenance: what the resolver did, so "is the CHECK wrong?" is answerable in one hop
    "as_of", "source_concept", "resolution_method", "roll_up_children", "root_anchor",
    "role_uri", "accession_number", "edgar_url",
    # check-specific evidence, JSON
    "detail",
)


def finding_id(check_name: str, ticker: str, field: str, period_key: str) -> str:
    """The stable 16-hex identity of a finding. See the module docstring for the exclusions.

    SHA-256 truncated to 16 hex characters: 64 bits over a population that will not exceed a
    few hundred thousand findings, so a collision is not a practical concern, and short
    enough to read in a report and paste into a config by hand.
    """
    payload = "\x1f".join((check_name, ticker, field or "", period_key or ""))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def edgar_url(cik: str | int | None, accession_number: str | None) -> str | None:
    """The filing's EDGAR index page, or None when either component is missing.

    Both forms of the accession are needed: the path segment has the dashes stripped and the
    file name keeps them. Getting that wrong yields a 404, which reads to an agent as "the
    filing is gone" rather than "the URL was built wrong" -- so it is built once, here.
    """
    if not cik or not accession_number:
        return None
    return _EDGAR_URL.format(cik=str(cik).lstrip("0") or "0",
                             plain=str(accession_number).replace("-", ""),
                             accession=accession_number)


def period_key_for_range(start, end) -> str:
    """`series_shape`'s grain: the span a series-level finding covers, as `start..end`."""
    return f"{_date(start)}..{_date(end)}"


def _date(value) -> str:
    """A date as `YYYY-MM-DD`, tolerating `datetime.date` (what Postgres DATE returns),
    `Timestamp`, and a string that is already in that shape."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(pd.Timestamp(value).date())


@dataclass(slots=True)
class Finding:
    """One finding, ready to become one `fundamentals_check` row.

    Constructed by a check; never by hand outside a test. `run_date` is stamped by the
    validator at write time rather than by the check, so every finding in one run shares it
    even if the run straddles midnight.
    """

    check_name: str
    ticker: str
    severity: str
    field: str = ""
    period_key: str = ""
    tier: int = 0
    substrate: str = ""

    observed: float | None = None
    expected: float | None = None
    deviation: float | None = None

    as_of: Any = None
    source_concept: str | None = None
    resolution_method: str | None = None
    roll_up_children: str | None = None
    root_anchor: str | None = None
    role_uri: str | None = None
    accession_number: str | None = None
    cik: str | None = None

    detail: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITY_ORDER:
            raise ValueError(f"{self.check_name}: severity {self.severity!r} is not one of "
                             f"{SEVERITY_ORDER}")

    @property
    def id(self) -> str:
        """This finding's cross-run identity."""
        return finding_id(self.check_name, self.ticker, self.field, self.period_key)

    def as_row(self, run_date) -> dict[str, Any]:
        """One `fundamentals_check` row. `detail` is serialised here and nowhere else."""
        return {
            "run_date": _date(run_date),
            "check_name": self.check_name,
            "ticker": self.ticker,
            "field": self.field or "",
            "period_key": self.period_key or "",
            "finding_id": self.id,
            "tier": int(self.tier),
            "severity": self.severity,
            "substrate": self.substrate,
            "observed": _float(self.observed),
            "expected": _float(self.expected),
            "deviation": _float(self.deviation),
            "as_of": _date(self.as_of) or None,
            "source_concept": self.source_concept,
            "resolution_method": self.resolution_method,
            "roll_up_children": self.roll_up_children,
            "root_anchor": self.root_anchor,
            "role_uri": self.role_uri,
            "accession_number": self.accession_number,
            "edgar_url": edgar_url(self.cik, self.accession_number),
            # `default=str` so a Timestamp or a numpy scalar that reached `detail` is written
            # rather than raising mid-run and losing every finding after it.
            "detail": json.dumps(self.detail, sort_keys=True, default=str),
        }


def _float(value) -> float | None:
    """A payload number as a plain float, or None. Never NaN.

    NaN in a DOUBLE PRECISION column is legal Postgres but reads as a value in every
    downstream frame; None is what "the check had nothing to report here" means.
    """
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(out) else out


def findings_frame(findings: list[Finding], run_date) -> pd.DataFrame:
    """`findings` as a `fundamentals_check`-shaped frame, columns pinned and dtypes forced.

    The dtype forcing is not cosmetic. `sql/schema.sql` is applied only when Postgres
    INITIALISES a volume, so on a live one `store.save` creates a missing table from the
    FIRST frame it is handed -- and an all-None `object` column becomes TEXT, permanently.
    That is how a real number once landed in this database as the string '1997000000.0'.
    A first run in which no check reports an `expected` is entirely ordinary.
    """
    frame = pd.DataFrame([f.as_row(run_date) for f in findings],
                         columns=list(FINDING_COLUMNS))
    for column in ("observed", "expected", "deviation"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    frame["tier"] = pd.to_numeric(frame["tier"], errors="coerce").astype("Int64")
    for column in frame.columns:
        if column in ("observed", "expected", "deviation", "tier"):
            continue
        frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
    return frame
