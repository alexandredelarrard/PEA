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

`fundamentals_check` is keyed on `run_date`, so the ROWS are per-run. What has to survive
across runs is the *finding*, because differencing two comparable runs is how a fix is proved
-- and a finding whose id changed between them would look closed and reopened at once.

So the id hashes exactly the four components that make two observations "the same finding":
check, ticker, field, period_key. It deliberately does NOT include:

  * `run_date` -- an id that changed nightly would make every delta empty;
  * `severity` -- a threshold retune moves severities constantly, and Phase 2 moved 347 of
    them in a single change. A finding must not lose its identity because a check was
    recalibrated;
  * `observed` -- the whole point is that the value can be re-measured and still be the same
    known finding.

`finding_id` also IS the primary key, hashed. That is not decoration: two findings sharing one
id are two rows that upsert onto each other, so `findings_frame` refuses them outright rather
than letting a run report more findings than it stored. See `DuplicateFindingError`.

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
#: makes `info` a usable place to put `restatement_ledger`, a declared exclusion cost and every
#: probation-field finding without drowning the queue. `clusters.py` weights it ZERO for the
#: same reason -- no volume of `info` can outrank one real finding.
QUEUE_SEVERITIES: frozenset[str] = frozenset({CRITICAL, HIGH, MEDIUM})

#: EDGAR's canonical accession URL. A finding without one costs the reviewing agent a lookup
#: before it can read the filing, which is the one step decision 47 exists to remove.
_EDGAR_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{plain}/{accession}-index.htm"

#: Every column of `fundamentals_check`, in order. Declared ONCE so the empty-result path,
#: the populated path and the DDL cannot drift apart.
FINDING_COLUMNS: tuple[str, ...] = (
    # identity -- the PK, then the cross-run id, the run's scope id, the DEFECT's id
    "run_date", "check_name", "ticker", "field", "period_key", "finding_id",
    "run_id", "cluster_id",
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


def cluster_id(ticker: str, field: str) -> str:
    """The stable 12-hex identity of a CLUSTER: one `(ticker, field)` defect.

    ## Why `check_name` is deliberately NOT in this key

    Because 11,926 findings were never 11,926 bugs. Run 2 measured 739 of 1,893 `(ticker,
    field)` series carrying 8,160 of 10,362 findings, and MCD `capex` alone tripping NINE
    checks for 54 findings. Nine checks are not nine issues -- they are nine witnesses to one
    defect, and a work queue keyed on the witness makes the same fix look like nine jobs.

    So `check_name` is EVIDENCE INSIDE a cluster (and the corroboration signal an agent weighs
    first), never part of its identity. `period_key` is excluded for the same reason: a field
    that is wrong for 40 quarters is one fix, not forty.

    Twelve hex characters rather than `finding_id`'s sixteen: the population is one per
    (ticker, field) pair -- ~30k universe-wide, not hundreds of thousands -- and 48 bits is
    ample over that, while a shorter id is what a human actually reads aloud and pastes into
    `validate status set`.
    """
    payload = "\x1f".join((ticker or "", field or ""))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


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

    @property
    def cluster(self) -> str:
        """The `(ticker, field)` DEFECT this finding is one witness to. See `cluster_id`."""
        return cluster_id(self.ticker, self.field)

    def as_row(self, run_date, run_id: str = "") -> dict[str, Any]:
        """One `fundamentals_check` row. `detail` is serialised here and nowhere else.

        `run_id` is stamped by the validator, like `run_date`: it identifies the SCOPE this
        run covered, and without it a row-count drop between two runs is ambiguous between
        "the fix worked" and "the second run looked at fewer tickers".
        """
        return {
            "run_date": _date(run_date),
            "check_name": self.check_name,
            "ticker": self.ticker,
            "field": self.field or "",
            "period_key": self.period_key or "",
            "finding_id": self.id,
            "run_id": run_id or None,
            "cluster_id": self.cluster,
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


def collapse_by_id(findings: list["Finding"], *, why: str) -> list["Finding"]:
    """One finding per `finding_id`, worst first, with what was merged recorded in `detail`.

    ## The grain mismatch this exists to close

    `finding_id` hashes `(check_name, ticker, field, period_key)` -- it IS the
    `fundamentals_check` primary key -- but several checks group on something FINER. Two
    findings then share one id, the second upserts silently over the first, and the run
    reports more findings than it stored.

    It has happened twice, on two different mechanisms:

      * TIER 3 groups on `(ticker, field, duration_type, period_end)`. A period end carrying
        both an `instant` and a `quarterly` observation produced two findings with one id.
        Run 2 emitted 12,462 and stored 11,926: `cross_vintage` lost 526, `q4_footing` 6,
        `leaf_vs_total` 4.
      * TIER 1's `cross_identity`, once it moved to `fundamentals_facts`, groups on
        `(ticker, accession_number, period_end)` -- and a filing carries COMPARATIVES, so one
        balance-sheet date appears in several filings. AMT's 2016-12-31 sheet appears in five.
        174 findings, 83 ids, 91 collisions.

    Widening `period_key` was the other option and was rejected both times: it changes the
    identity of every existing finding, so every settled outcome and every cross-run
    comparison would silently re-key. Collapsing keeps the identity and reports the merge,
    which is also the more honest shape -- one (ticker, field, period) IS one thing to look at,
    and reporting the same broken balance sheet five times because five filings repeated it is
    the DQC_0118 drowning this design exists to prevent.

    The survivor is the WORST severity first and the largest absolute deviation second. Not
    deviation alone: `cross_vintage` emits `info` for a restatement and `high` for a candidate
    defect on the same period, and a big benign restatement must never bury a small real one.
    """
    if len(findings) < 2:
        return findings
    order = {severity: i for i, severity in enumerate(SEVERITY_ORDER)}
    groups: dict[str, list["Finding"]] = {}
    for finding in findings:
        groups.setdefault(finding.id, []).append(finding)

    out: list["Finding"] = []
    for members in groups.values():
        if len(members) == 1:
            out.append(members[0])
            continue
        members = sorted(members, key=lambda f: (order.get(f.severity, 99),
                                                 -abs(f.deviation or 0.0)))
        winner = members[0]
        winner.detail = {
            **winner.detail,
            "n_collapsed": len(members),
            "collapsed": [{"duration_type": m.detail.get("duration_type"),
                           "annual_period_end": m.detail.get("annual_period_end"),
                           "period_end": m.detail.get("period_end"),
                           "severity": m.severity,
                           "observed": m.observed, "expected": m.expected,
                           "deviation": m.deviation,
                           "accession": m.accession_number} for m in members[1:]],
            "collapsed_why": why,
        }
        out.append(winner)
    return out


class DuplicateFindingError(ValueError):
    """Two findings sharing one `finding_id` in a single run.

    Raised, never warned, and that choice is the whole point of the class. `finding_id` is a
    hash of exactly the `fundamentals_check` PK, so two findings carrying one id are two rows
    that will UPSERT onto each other: the second silently overwrites the first and the run
    reports a number it did not write. Run 2 emitted 12,462 findings and stored 11,926 -- 536
    of them vanished exactly this way, in `cross_vintage` (526), `q4_footing` (6) and
    `leaf_vs_total` (4), and nothing in the system said so.

    The cause is always the same shape: a check GROUPS on a key wider than the one
    `finding_id` hashes -- those three grouped on `(ticker, field, duration_type, period_end)`
    while `period_key` carries only `period_end`. The fix belongs in the check, which must
    collapse to the finding's own grain and say what it collapsed; it does NOT belong here,
    because dropping a duplicate at write time is the silent overwrite with better manners.
    """


def findings_frame(findings: list[Finding], run_date, run_id: str = "") -> pd.DataFrame:
    """`findings` as a `fundamentals_check`-shaped frame, columns pinned and dtypes forced.

    Raises `DuplicateFindingError` when two findings share a `finding_id`; see that class for
    why an emitted count that exceeds the stored count is the bug this guard exists to make
    impossible rather than merely visible.

    The dtype forcing is not cosmetic. `sql/schema.sql` is applied only when Postgres
    INITIALISES a volume, so on a live one `store.save` creates a missing table from the
    FIRST frame it is handed -- and an all-None `object` column becomes TEXT, permanently.
    That is how a real number once landed in this database as the string '1997000000.0'.
    A first run in which no check reports an `expected` is entirely ordinary.
    """
    _assert_unique(findings)
    frame = pd.DataFrame([f.as_row(run_date, run_id) for f in findings],
                         columns=list(FINDING_COLUMNS))
    for column in ("observed", "expected", "deviation"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    frame["tier"] = pd.to_numeric(frame["tier"], errors="coerce").astype("Int64")
    for column in frame.columns:
        if column in ("observed", "expected", "deviation", "tier"):
            continue
        frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
    return frame


def _assert_unique(findings: list[Finding]) -> None:
    """Raise naming the offending keys, and the check that produced them.

    The message carries the KEYS rather than a count: "3 duplicates" sends an author looking
    through 30 checks, whereas `cross_vintage AAPL capex 2019-09-28` names the grain mismatch
    outright.
    """
    seen: dict[str, Finding] = {}
    clashes: list[tuple[Finding, Finding]] = []
    for finding in findings:
        first = seen.setdefault(finding.id, finding)
        if first is not finding:
            clashes.append((first, finding))
    if not clashes:
        return
    shown = "; ".join(
        f"{a.check_name} {a.ticker} {a.field} @ {a.period_key or 'ticker-level'} "
        f"[{a.id}]" for a, _ in clashes[:5])
    raise DuplicateFindingError(
        f"{len(clashes)} finding(s) share a finding_id with an earlier finding in this run: "
        f"{shown}{' ...' if len(clashes) > 5 else ''}. finding_id hashes exactly the "
        f"fundamentals_check PK, so these rows would UPSERT onto each other and the run "
        f"would report more findings than it stored. The check must collapse to "
        f"(ticker, field, period_key) and record what it collapsed -- see "
        f"DuplicateFindingError.")
