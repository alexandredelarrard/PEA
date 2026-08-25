"""
tier1_value.py  (src/validate/fundamentals/checks/)
--------------------------------------------------------------------------------------------
TIER 1 -- the deterministic tier. SIX contract checks on `fundamentals_history`, everything
else on `fundamentals_facts`.

Every check here is a RULE, not a statistic: it either holds or it does not, and a finding is
a statement about a contract rather than about a distribution. That is what makes the tier
cheap enough to run nightly over the whole table.

## THE SUBSTRATE SPLIT, and why it is not arbitrary

    fundamentals_history   grain, column_contract, code_vocabulary, unexplained_null,
                           pit_leak, coverage_universe
    fundamentals_facts     everything else, all 13 of them

The rule is: a check that asks about the TABLE reads history; a check that asks about a
NUMBER reads facts.

The eight value and coverage checks were moved here from history, and the reason is
provenance. `Finding.edgar_url` is built from `(cik, accession_number)`; `fundamentals_history`
has 69 columns and carries neither. Measured on the last history-based run: **0 of 1,437
Tier-1 findings had a URL**, against 77.8% on Tier 2 and 100% on Tier 3. A finding an agent
cannot trace to a filing cannot be investigated, so the whole tier was unactionable however
well it was ranked.

Nothing is lost by the move and something is gained. On the live 54-ticker table the balance
sheet fails to foot for the same seven filers on either substrate -- UNH, PGR, AMT, EQIX, VRT,
SPG, NVDA -- but facts exposes 4,763 testable statements against history's 3,229, because a
filing carries comparatives and each comparative is a separate published claim that can be
restated on its own. 144 breaks against 64, every one with an accession.

The SIX that stayed are the ones facts cannot express: a 69-column ORDERED contract (facts is
long), a null CELL (in facts a missing fact is an absent ROW), the reason-code vocabulary, and
the no-leakage snapshot grain. They all carry `expected_fire_rate_ceiling=0.0` and all fired
zero on the calibration run, which is exactly what a tripwire should do -- they exist to catch
a bug in `build_history`, which is the only defect class that is genuinely history's own.

ETN's 2012-11-14 row is the specimen of that class: `totalLiabilities` of -$8,237,223,652
against `totalAssets` of $4,776,348, tagged `derived_identity`, computed across the Irish
redomicile's holdco shell and a carried-forward equity from the operating company. It has no
counterpart in `facts` because no filer ever tagged it. That row is why history keeps its
tripwires, and why it keeps nothing else.

## `facts` is strictly as-filed, so the derived-total skip is gone

`cross_identity` on history had to read `fundamentals_reason_codes` and skip every row whose
`totalLiabilities` was `derived_identity` -- testing `A - E + E == A` is arithmetic and passes
on any numbers at all. On facts that cannot arise: all eight `resolution_method` values
(`linkbase_total`, `tag_primary`, `tag_fallback`, `linkbase_sum`, `linkbase_root`,
`field_sum`, `statement_leaf_sum`, `unresolved`) resolve from concepts the filer tagged, and
nothing in the table is computed from the identity being tested. The skip and its helper are
deleted.

## This module ABSORBED `scripts/verify_fundamentals_history.py`

That script's eight §5.8 gates are Tier 1, already written and already run against the live
tables, so re-implementing them here would have produced two implementations of one rule --
exactly what `reason_codes.py` exists to prevent. They are now checks
(`grain`, `column_contract`, `unexplained_null`, `filing_lag`, `amendment_ledger`,
`same_day_collapse`, `coverage_field`, `code_vocabulary`) and the script is deleted.

## The builder owns invariants; the validator REPORTS them (decision 40)

Three of these rules are already build-time assertions in `build_history.py` that FAIL the
build: the reason-code vocabulary, the grain, and the look-ahead lag. The validator does not
re-implement them and does not weaken them -- it calls the same code and reports the NUMBER,
so that a green build and a green validator are the same claim made twice rather than two
claims that can drift. If one of these ever fires here, something wrote to the table without
going through the builder.

## NOTHING HERE GATES (decision 45)

Not one finding blocks the nightly fill of `fundamentals_facts` / `fundamentals_history`. This
is the SEC's own warn-over-reject precedent, and it is a decision rather than an oversight:
one filer's bad quarter must never stall the other 499.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from src.data_extract.utils.fundamentals import reason_codes as rc
from src.data_extract.utils.fundamentals.kpi_catalogue import (
    HISTORY_KEYS, HISTORY_PROVENANCE, HISTORY_REGIME)
from src.validate.fundamentals.checks import (
    FACTS, GRAIN_CELL, GRAIN_ROW, GRAIN_SERIES, GRAIN_TICKER, HISTORY, check)
from src.validate.fundamentals.finding import (
    CRITICAL, Finding, HIGH, INFO, MEDIUM, collapse_by_id, period_key_for_range)
from src.validate.fundamentals.substrate import Substrates

# --------------------------------------------------------------------------- #
# thresholds -- code, not config, with their measurement in the comment         #
# --------------------------------------------------------------------------- #
# Decision: a threshold is BEHAVIOUR and lives in code; a measured baseline is a FACT and
# lives in `configs/fundamentals/fundamentals_baselines.json`. Phase 9 will retune these
# repeatedly and a `configs/` approval loop per tweak is friction with no safety benefit.

#: A 10-Q lands ~35-45 days after quarter end and a 10-K ~60-90. Past this the row is stamped
#: with a date at which its number was NOT the freshest available. Measured on the rebuilt
#: 54-ticker table: median 34d, p90 55d, and **1 row of 3,267** beyond it -- SMCI's delinquent
#: FY2017 10-K at 686d, filed during its Nasdaq delisting, which is real.
MAX_FILING_LAG_DAYS = 200

#: The filings-per-ticker-per-year band a continuously-listed US filer sits in: four 10-Qs and
#: a 10-K, minus the quarter the 10-K replaces, plus the occasional amendment.
FILINGS_PER_YEAR_BAND: tuple[float, float] = (3.0, 5.5)

#: `epsDiluted` beyond this is not a per-share amount -- it is a units error. BRK-A's ~$40,000
#: EPS is the highest real figure in the S&P 500 and it is not in this universe (BRK-B is);
#: $1,000 leaves an order of magnitude of headroom over every plausible filer here.
#:
#: CURRENTLY UNUSED, and kept deliberately. `impossible_value` moved to `fundamentals_facts`,
#: where `epsDiluted` does not exist -- it is one of the twelve columns `build_history`
#: DERIVES. The bound is still the right one and costs nothing to keep; it becomes live again
#: the moment EPS is either tagged into facts or given a derived-value check of its own.
MAX_ABS_EPS = 1_000.0

#: `cross_identity` tolerance. Balance sheets foot exactly in the filing; the slack is for
#: rounding when the filer tags in thousands and we store units, not for a real gap.
IDENTITY_TOLERANCE = 0.01

#: Above this relative gap the balance sheet is PROVABLY broken and the finding is `critical`;
#: below it the finding is `high`, because temporary/mezzanine equity could account for it and
#: we carry no column for that (see `cross_identity`).
#:
#: 10%, and it is a measurement rather than a round number. On the rebuilt 54-ticker table
#: (2026-08-24) the gaps that survive both equity bases are:
#:
#:     UNH 1.68% | EQIX 1.48% | PGR 1.41% | SPG 1.06% | NVDA 1.15% | AMT 3.28%
#:     ---------------------------- nothing at all between 3.3% and 95% -------------------
#:     VRT 95.55%  (the pre-merger SPAC: its trust IS temporary equity)
#:     ETN 172,559x (the 2012 Irish-redomicile holdco shell -- section 5a-2's open case)
#:
#: The mezzanine-explainable population tops out at 3.3%, so 10% leaves 3x headroom over
#: everything a redeemable-NCI or OP-unit balance could plausibly be, while sitting far below
#: the two rows that are genuinely broken. Redeemable NCI above a tenth of total assets is not
#: a real operating-company capital structure -- VRT is a shell, which is the exception that
#: SHOULD be critical.
IDENTITY_GROSS_BREAK = 0.10

#: Below this a relative comparison is meaningless: a $3 gap on a $2 base is a 150% error that
#: says nothing. Absolute, in reporting units.
IDENTITY_MIN_MAGNITUDE = 1_000.0

#: Below this many filers a regime cannot support a peer argument at all -- one insurer says
#: nothing about insurers. Inherited verbatim from the absorbed `audit_absence_evidence.py`.
MIN_PEERS_FOR_ABSENCE_VERDICT = 4

#: `coverage_field` fires when a field is null for at least this fraction of a ticker's
#: publication events. Not per-cell: see `coverage_field`'s docstring.
COVERAGE_NULL_RATE = 0.5

#: Fields whose absence is checked against the peer oracle. The derived ones are excluded --
#: a ratio is null exactly when its input is, and reporting both is one defect twice.
_TOP_LINES: tuple[str, ...] = ("totalRevenue", "stockholdersEquity", "totalAssets",
                               "totalLiabilities")

#: The balance-sheet identity's four legs. Tagged INSTANT: a balance sheet is a point in
#: time, and a duration-tagged `totalAssets` is a different measurement.
_IDENTITY_FIELDS: tuple[str, ...] = ("totalAssets", "totalLiabilities",
                                     "stockholdersEquity", "minorityInterest")

#: The gross-profit relation's three legs. DURATION-tagged, and the pivot keys on
#: `duration_type` and `period_start` as well as `period_end` so all three legs are read off
#: the SAME window -- an annual `grossProfit` against a quarterly `costOfRevenue` is a units
#: error dressed up as a finding, and on this table it would fire on nearly every filer.
_GROSS_PROFIT_FIELDS: tuple[str, ...] = ("grossProfit", "totalRevenue", "costOfRevenue")

#: The duration shapes an income-statement line can be tagged with. `other` is excluded: it
#: is the shape `unresolved` rows carry, and they hold no value by construction.
_DURATION_TYPES: tuple[str, ...] = ("quarterly", "annual", "ytd6", "ytd9")

#: A dimension MEMBER, AXIS or DOMAIN in a concept name. `dimensional_scope`'s detector -- see
#: that check for why this, and not a `dim_*` column, is what is checkable here.
_MEMBER_RE = re.compile(r"(Member|Axis|Domain)$")


# --------------------------------------------------------------------------- #
# 0. the statement pivot -- what makes a Tier-1 finding traceable to a filing  #
# --------------------------------------------------------------------------- #

def _statements(sub: Substrates, fields: tuple[str, ...], duration_types: tuple[str, ...],
                *, extra_keys: tuple[str, ...] = ()) -> pd.DataFrame:
    """`fundamentals_facts` pivoted to one row per FILED STATEMENT, wide over `fields`.

    The key is `(ticker, accession_number, period_end)` -- one balance sheet or one income
    statement as the filer actually published it, carrying its own `cik` and
    `accession_number`. That last part is the entire reason the value checks moved here: a
    finding on this grain yields an `edgar_url` a reviewer can open, and a finding on the
    history grain never can.

    ONE ACCESSION YIELDS SEVERAL STATEMENTS, and that is correct rather than duplication. A
    10-K carries its comparatives, each is a separate published claim, and a filer can restate
    one while leaving the others alone -- which is why `fundamentals_facts` keys on
    `period_end` and not on the fiscal LABELS, after those labels were measured losing 18,604
    rows to collisions.

    Rows with a NULL `value` are dropped before the pivot. They are the `unresolved` shape --
    64,462 of them, all tagged `other` and all carrying a `dc_code` -- and they mean "we
    looked and found nothing", which is a coverage fact rather than a value.
    """
    facts = sub.facts
    if facts.empty:
        return pd.DataFrame()
    rows = facts[facts["field"].isin(fields)
                 & facts["duration_type"].isin(duration_types)
                 & facts["value"].notna()]
    if rows.empty:
        return pd.DataFrame()
    keys = ["ticker", "accession_number", "period_end", "cik", "filing_date", *extra_keys]
    wide = rows.groupby(keys + ["field"])["value"].last().unstack("field").reset_index()
    for name in fields:
        if name not in wide.columns:
            wide[name] = np.nan
    return wide


def _filings(sub: Substrates) -> pd.DataFrame:
    """One row per `(ticker, accession_number)` -- the FILING, not its contents.

    `period_of_report` and not `period_end`: a 10-K's comparatives end years before the
    filing, so a lag measured against `period_end` reports every comparative in every annual
    report as a delinquent filing. `period_of_report` is the filing's own reporting date and
    is populated on all 316,136 rows.
    """
    facts = sub.facts
    if facts.empty:
        return pd.DataFrame()
    columns = ["cik", "form", "filing_date", "period_of_report", "is_amendment"]
    available = [c for c in columns if c in facts.columns]
    return (facts.groupby(["ticker", "accession_number"], as_index=False)[available].first())


# --------------------------------------------------------------------------- #
# 1. the grain, the contract, the vocabulary -- absorbed §5.8 gates            #
# --------------------------------------------------------------------------- #

@check(name="grain", tier=1, substrate=HISTORY, severity=CRITICAL, grain=GRAIN_ROW,
       expected_fire_rate_ceiling=0.0)
def grain(sub: Substrates) -> list[Finding]:
    """One row per (ticker, as_of); `fiscal_end` never regresses; `as_of` never precedes it.

    A ZERO ceiling, and that is the point. `build_history._assert_grain` already fails the
    build on all three, so under the publication-event grain none of them can fire -- which is
    exactly what makes this a good check rather than a redundant one. A hit means something
    wrote to `fundamentals_history` without going through the builder.

    The third rule is the LOOK-AHEAD LEAK, and it is why the grain was rebuilt at all: the
    previous build computed `as_of` as a median-of-spine heuristic and put ROP's 2009 year 59
    days BEFORE its own period end.
    """
    out: list[Finding] = []
    history = sub.history
    if history.empty:
        return out
    sub.denominator("grain", len(history))

    duplicated = history[history.duplicated(["ticker", "as_of"], keep=False)]
    for (ticker, as_of), rows in duplicated.groupby(["ticker", "as_of"]):
        out.append(Finding(
            check_name="grain", ticker=str(ticker), severity=CRITICAL, tier=1,
            substrate=HISTORY, field="", period_key=str(pd.Timestamp(as_of).date()),
            as_of=as_of, observed=float(len(rows)), expected=1.0,
            detail={"rule": "duplicate (ticker, as_of)",
                    "why": "the same-day collapse failed; two accessions on one date must "
                           "produce ONE row, with provenance resolved by form precedence"}))

    ordered = history.sort_values(["ticker", "as_of"])
    step = ordered.groupby("ticker")["fiscal_end"].diff()
    for _, row in ordered[step < pd.Timedelta(0)].iterrows():
        out.append(Finding(
            check_name="grain", ticker=str(row["ticker"]), severity=CRITICAL, tier=1,
            substrate=HISTORY, field="fiscal_end",
            period_key=str(pd.Timestamp(row["as_of"]).date()), as_of=row["as_of"],
            detail={"rule": "fiscal_end regressed",
                    "fiscal_end": str(pd.Timestamp(row["fiscal_end"]).date()),
                    "why": "the visible fact set only grows, so the newest period a filer "
                           "has reached cannot move backwards"}))

    lag = (history["as_of"] - history["fiscal_end"]).dt.days
    for _, row in history[lag < 0].iterrows():
        out.append(Finding(
            check_name="grain", ticker=str(row["ticker"]), severity=CRITICAL, tier=1,
            substrate=HISTORY, field="as_of",
            period_key=str(pd.Timestamp(row["as_of"]).date()), as_of=row["as_of"],
            observed=float((row["as_of"] - row["fiscal_end"]).days), expected=0.0,
            detail={"rule": "LOOK-AHEAD LEAK: as_of precedes fiscal_end",
                    "fiscal_end": str(pd.Timestamp(row["fiscal_end"]).date())}))
    return out


@check(name="column_contract", tier=1, substrate=HISTORY, severity=CRITICAL,
       grain=GRAIN_TICKER, expected_fire_rate_ceiling=0.0)
def column_contract(sub: Substrates) -> list[Finding]:
    """`fundamentals_history` has exactly the catalogue's columns, in the catalogue's order.

    ORDER, not just membership. The stored order IS the contract: a consumer reading the table
    positionally, or a `store.save` against a table whose columns drifted, produces values in
    the wrong columns rather than an error. `Catalogue.history_columns` is the one authority.
    """
    if sub.history.empty:
        return []
    contract, stored = sub.catalogue.history_columns, list(sub.history.columns)
    sub.denominator("column_contract", len(contract))
    if stored == contract:
        return []
    return [Finding(
        check_name="column_contract", ticker="", severity=CRITICAL, tier=1, substrate=HISTORY,
        observed=float(len(stored)), expected=float(len(contract)),
        detail={"missing": sorted(set(contract) - set(stored)),
                "extra": sorted(set(stored) - set(contract)),
                "same_order": stored == contract,
                "why": "the stored column ORDER is the contract, not just the column set"})]


@check(name="code_vocabulary", tier=1, substrate=HISTORY, severity=CRITICAL,
       grain=GRAIN_TICKER, expected_fire_rate_ceiling=0.0)
def code_vocabulary(sub: Substrates) -> list[Finding]:
    """Every stored `dc_code` is in `reason_codes.ALL_CODES`.

    A typo in a reason code is WORSE than no code at all: the zero-unexplained-nulls gate is a
    LEFT JOIN, so a misspelt code still produces a matching row and the cell reads as explained
    while nothing can interpret the explanation. `build_ticker` asserts this on every row it
    writes; this reports it over what is actually stored.
    """
    if sub.codes.empty:
        return []
    mix = sub.codes["dc_code"].value_counts()
    sub.denominator("code_vocabulary", int(mix.sum()))
    unknown = sorted(set(mix.index) - rc.ALL_CODES)
    return [Finding(
        check_name="code_vocabulary", ticker="", severity=CRITICAL, tier=1, substrate=HISTORY,
        field=str(code), observed=float(mix[code]),
        detail={"code": str(code), "declared_vocabulary_size": len(rc.ALL_CODES),
                "why": "a code outside the closed set makes a null read as explained while "
                       "nothing can interpret the explanation"})
        for code in unknown]


# --------------------------------------------------------------------------- #
# 2. nulls, leaks and scope                                                    #
# --------------------------------------------------------------------------- #

@check(name="unexplained_null", tier=1, substrate=HISTORY, severity=CRITICAL, grain=GRAIN_CELL,
       expected_fire_rate_ceiling=0.0)
def unexplained_null(sub: Substrates) -> list[Finding]:
    """No NULL value cell without a `fundamentals_reason_codes` row of its own.

    THE gate the whole reason-code layer exists to make passable, and the one number that says
    whether "why is this cell empty?" is answerable at all. Measured on the rebuilt 54-ticker
    table: 196,020 cells, 71,857 null (36.7%), **0 unexplained**.

    A high null RATE is not a defect -- a bank has no inventory and a REIT no cost of revenue.
    An unexplained one always is.
    """
    out: list[Finding] = []
    history, value_columns = sub.history, sub.value_columns
    if history.empty:
        return out
    sub.denominator("unexplained_null", len(history) * len(value_columns))

    explained = sub.coded()
    long = history.melt(id_vars=["ticker", "as_of"], value_vars=value_columns,
                        var_name="field", value_name="value")
    nulls = long[long["value"].isna()]
    for ticker, as_of, field in zip(nulls["ticker"], nulls["as_of"], nulls["field"]):
        if (ticker, as_of, field) in explained:
            continue
        out.append(Finding(
            check_name="unexplained_null", ticker=str(ticker), severity=CRITICAL, tier=1,
            substrate=HISTORY, field=str(field),
            period_key=str(pd.Timestamp(as_of).date()), as_of=as_of,
            detail={"why": "a null with no reason code is a value nobody can account for; "
                           "the builder asserts this, so a hit means the table was written "
                           "outside build_history"}))
    return out


@check(name="pit_leak", tier=1, substrate=HISTORY, severity=CRITICAL, grain=GRAIN_ROW,
       expected_fire_rate_ceiling=0.0)
def pit_leak(sub: Substrates) -> list[Finding]:
    """No publication event carries a fact filed AFTER its own `as_of`.

    The point-in-time property, stated as a query rather than trusted as a consequence of the
    algorithm. `build_ticker` replays every event from `filing_date <= as_of` positionally, so
    a leak is structurally impossible -- and that is precisely why it is worth asserting: if
    this fires, the replay's prefix assumption has been broken upstream.

    The APPEND-ONLY half of decision 42's guarantee -- "no stored row changed value since the
    last build" -- is `build_history.diff_against_stored`, which runs inside the builder where
    it can see both the stored and the rebuilt frame. It is not re-implemented here; the
    validator never holds a rebuilt frame, and a second implementation of one rule is the
    thing this package exists to avoid.
    """
    out: list[Finding] = []
    if sub.history.empty or sub.facts.empty:
        return out
    sub.denominator("pit_leak", len(sub.history))

    first_filing = sub.facts.groupby("ticker")["filing_date"].min()
    for ticker, rows in sub.history.groupby("ticker"):
        earliest = first_filing.get(ticker)
        if pd.isna(earliest):
            continue
        leaked = rows[rows["as_of"] < earliest]
        for _, row in leaked.iterrows():
            out.append(Finding(
                check_name="pit_leak", ticker=str(ticker), severity=CRITICAL, tier=1,
                substrate=HISTORY, period_key=str(pd.Timestamp(row["as_of"]).date()),
                as_of=row["as_of"],
                detail={"earliest_filing_date": str(pd.Timestamp(earliest).date()),
                        "why": "a publication event predating every filing it could have "
                               "been built from is a row assembled from the future"}))
    return out


@check(name="dimensional_scope", tier=1, substrate=FACTS, severity=CRITICAL, grain=GRAIN_CELL,
       expected_fire_rate_ceiling=0.0)
def dimensional_scope(sub: Substrates) -> list[Finding]:
    """No resolved fact was read off a dimension MEMBER rather than the consolidated group.

    The defect it guards: DTE tags capex ONLY dimensioned to `dte:DTEElectricMember`, so a
    relaxed dimensional filter reads one subsidiary's number as the group's -- 17% low and
    entirely plausible. Nothing downstream can tell.

    ## WHAT THIS CHECK CAN AND CANNOT SEE -- read before trusting a zero

    `entity_scope.consolidated_facts` DROPS every dimensioned row at extraction and does not
    keep the `dim_*` columns, so `fundamentals_facts` has no dimension columns to test. A
    filter regression would therefore arrive here as an ordinary-looking row with a
    subsidiary's value in it, and no column would say so.

    What IS decidable on the stored table is the provenance: a fact resolved through a
    member-scoped arc carries a `...Member` / `...Axis` / `...Domain` token in its
    `source_concept`, its `root_anchor`, or one of its `roll_up_children`. That is the
    signature this tests, and it is a real boundary assertion -- but it is NOT full coverage
    of the invariant. The primary defence stays where it belongs, in `entity_scope`, tested
    where it lives. This is the second lock, and the README says so out loud.
    """
    out: list[Finding] = []
    facts = sub.facts
    if facts.empty:
        return out
    resolved = facts[facts["value"].notna()]
    sub.denominator("dimensional_scope", len(resolved))

    # Vectorised pre-filter before the row walk. The invariant holds on essentially every row
    # -- that is the point of a zero-ceiling check -- so iterating 252,001 resolved rows in
    # Python to find nothing is pure cost. Three `str.contains` passes in C narrow it to the
    # candidates, and only those are unpacked (`roll_up_children` is JSON and needs parsing).
    columns = [c for c in ("source_concept", "root_anchor", "roll_up_children")
               if c in resolved.columns]
    suspect = pd.Series(False, index=resolved.index)
    for column in columns:
        suspect |= resolved[column].astype(str).str.contains(
            r"(?:Member|Axis|Domain)", regex=True, na=False)
    for _, row in resolved[suspect].iterrows():
        offenders = [token for token in _provenance_concepts(row) if _MEMBER_RE.search(token)]
        if not offenders:
            continue
        out.append(Finding(
            check_name="dimensional_scope", ticker=str(row["ticker"]), severity=CRITICAL,
            tier=1, substrate=FACTS, field=str(row["field"]),
            period_key=str(pd.Timestamp(row["period_end"]).date()),
            observed=_float(row["value"]),
            source_concept=_text(row.get("source_concept")),
            resolution_method=_text(row.get("resolution_method")),
            roll_up_children=_text(row.get("roll_up_children")),
            root_anchor=_text(row.get("root_anchor")), role_uri=_text(row.get("role_uri")),
            accession_number=_text(row.get("accession_number")),
            cik=sub.cik_for(str(row["ticker"])),
            detail={"member_tokens": offenders,
                    "why": "a value read off a dimension member is one subsidiary's, stored "
                           "as the consolidated group's"}))
    return out


def _provenance_concepts(row: pd.Series) -> list[str]:
    """Every concept name a fact row's provenance mentions, `roll_up_children` unpacked.

    `roll_up_children` is stored as JSON; a malformed one is treated as a single opaque token
    rather than raising, because a check that dies on one bad row reports nothing about the
    other twenty-eight million.
    """
    tokens = [str(row.get(col) or "") for col in ("source_concept", "root_anchor")]
    children = row.get("roll_up_children")
    if children:
        try:
            parsed = json.loads(children) if isinstance(children, str) else children
            tokens.extend(str(c) for c in (parsed if isinstance(parsed, list) else [parsed]))
        except (TypeError, ValueError, json.JSONDecodeError):
            tokens.append(str(children))
    return [t for t in tokens if t]


# --------------------------------------------------------------------------- #
# 3. coverage                                                                  #
# --------------------------------------------------------------------------- #

@check(name="coverage_universe", tier=1, substrate=HISTORY, severity=HIGH, grain=GRAIN_TICKER,
       expected_fire_rate_ceiling=0.0)
def coverage_universe(sub: Substrates) -> list[Finding]:
    """Every ticker the run was SCOPED to has at least one row.

    Reads `Substrates.tickers`, which is the roster asked for -- so a ticker that produced no
    rows at all is a finding rather than an absence from the report. "0 findings" and "0
    tickers loaded" must never look the same.
    """
    if not sub.tickers:
        return []
    sub.denominator("coverage_universe", len(sub.tickers))
    present = set(sub.history["ticker"].unique()) if not sub.history.empty else set()
    return [Finding(
        check_name="coverage_universe", ticker=str(ticker), severity=HIGH, tier=1,
        substrate=HISTORY, observed=0.0, expected=1.0,
        detail={"why": "the roster names this ticker and fundamentals_history has no row "
                       "for it -- the fetch or the build never reached it"})
        for ticker in sub.tickers if ticker not in present]


@check(name="coverage_quarters", tier=1, substrate=FACTS, severity=HIGH, grain=GRAIN_SERIES,
       expected_fire_rate_ceiling=0.10)
def coverage_quarters(sub: Substrates) -> list[Finding]:
    """A filer's FILINGS are contiguous on ITS OWN filing cadence.

    RE-SPECIFIED against v2, which compared to a calendar. A calendar grid is wrong for a
    52/53-week filer, for KR's 16-week Q1, and for every Jan / May / Sep year-end on the
    roster -- it reports a hole every year for filers that never missed a filing.

    The expectation is instead the filer's own: the median gap between its own consecutive
    `as_of` dates. A gap materially longer than twice that is a MISSING FILING, and the
    quarter-grid holes this catches are real (MAA 17, JNJ 8, GS 3, DE 1, VRT 1).

    CEILING 10%, RAISED FROM 2%. The denominator is TICKERS (54 on the live roster), so a
    single finding is already 1.85% and a 2% ceiling could not survive two of them. Measured:
    4 findings, 7.41%. The old number was a rate borrowed from a per-ROW check and applied to
    a per-ticker one.

    ## ON FACTS: distinct FILING DATES, which is the same series history reshaped

    `fundamentals_history` emits one row per date on which a value became public, so its
    `as_of` series and the filer's distinct `filing_date` series are the same dates -- the
    history grain IS this cadence, collapsed. Reading it here means the finding names the
    accession that CLOSED the gap, so a reviewer opens the filing on one side of the hole
    instead of being handed a bare date range.
    """
    out: list[Finding] = []
    filings = _filings(sub)
    if filings.empty:
        return out
    sub.denominator("coverage_quarters", filings["ticker"].nunique())

    for ticker, rows in filings.sort_values("filing_date").groupby("ticker"):
        events = (rows.drop_duplicates("filing_date")
                  .sort_values("filing_date").reset_index(drop=True))
        dates = pd.to_datetime(events["filing_date"])
        if len(dates) < 4:
            continue                      # too short to have a cadence; not a hole
        gaps = dates.diff().dt.days.dropna()
        typical = float(gaps.median())
        if typical <= 0:
            continue
        for position, gap in gaps.items():
            if gap <= 2 * typical:
                continue
            closing = events.loc[position]
            out.append(Finding(
                check_name="coverage_quarters", ticker=str(ticker), severity=HIGH, tier=1,
                substrate=FACTS, field="",
                period_key=period_key_for_range(dates.loc[position - 1], dates.loc[position]),
                as_of=dates.loc[position],
                accession_number=str(closing["accession_number"]),
                cik=str(closing.get("cik") or ""),
                observed=float(gap), expected=typical, deviation=float(gap) / typical,
                detail={"typical_gap_days": typical,
                        "gap_opens": str(dates.loc[position - 1].date()),
                        "gap_closes": str(dates.loc[position].date()),
                        "closing_form": str(closing.get("form") or ""),
                        "why": "measured against the FILER'S OWN cadence, not a calendar -- "
                               "a 52/53-week or non-December year-end is not a hole. The "
                               "accession named here is the filing that CLOSED the gap"}))
    return out


@check(name="coverage_field", tier=1, substrate=FACTS, severity=HIGH, grain=GRAIN_SERIES,
       expected_fire_rate_ceiling=0.25)
def coverage_field(sub: Substrates) -> list[Finding]:
    """A field is expected for a filer unless its REGIME's own filers say otherwise.

    ## The oracle -- absorbed from `scripts/audit_absence_evidence.py`, which is deleted

    The register cannot settle this: `expected_absent` is an asserted rule, not a filing. What
    CAN settle it, from stored facts alone and with zero rules, is the filer's PEERS. Per
    (regime, field), how many of the regime's filers ever resolve a number through a concept
    we recognise?

      STRUCTURAL  0 of N resolve it -> absence is a property of the regime.   no finding
      UNIVERSAL   N of N resolve it -> a NULL here is a DEFECT in our extraction.  `high`
      MIXED       some do, some do not -> only the filing can settle it.       `medium`

    MIXED IS THE MAJORITY AND THAT IS THE HONEST HEADLINE: 31 of 48 industrial fields resolve
    for some filers and not others. It is the validator's real work queue, and no config rule
    could ever decide it.

    CEILING 25%, RAISED FROM 10% ON EVIDENCE. The first calibration run measured 20.25%
    (656 of 3,240 series: 567 `medium` MIXED + 89 `high` UNIVERSAL) and the report called it a
    threshold bug. It was not -- the 10% ceiling contradicted section E-2 of the plan that
    commissioned this check, which states outright that MIXED is the largest bucket. A ceiling
    that disagrees with its own phase's measurement is the ceiling that is wrong.

    Below `MIN_PEERS_FOR_ABSENCE_VERDICT` filers the check ABSTAINS and says so. 0-of-4 does
    not become evidence by being written down -- `energy` and `utility` have four filers each
    on this roster, and that is why no `expected_absent` cell is written from this output.

    ## FIRES PER (ticker, field), NOT PER CELL

    A deviation from the plan's letter, forced by arithmetic: 71,857 null cells on a 54-ticker
    roster would be 71,857 findings, which is the DQC_0118 drowning this whole design exists to
    prevent. One finding per (ticker, field) carries the miss count and rate in `detail`, and
    the SHAPE of the gap -- interior hole vs late start vs went dark -- is `series_shape`'s
    job, which is exactly the division of labour the plan's own critique of per-cell firing
    argues for.

    ## ON FACTS, WHERE THE ORACLE ALREADY LIVED

    `_absence_verdicts` has always read `sub.facts` -- the peer evidence could never have come
    from history, because history stores a filer's own snapshot and says nothing about what
    its peers resolve. Only the null RATE was measured on history, so the check straddled two
    substrates and could report a rate its own oracle had no way to see.

    The denominator changes shape, and it is the honest one: in `fundamentals_facts` a missing
    field is an ABSENT ROW, not a null cell, so coverage is measured over the filer's own
    distinct `period_of_report` values -- how many of its reporting periods produced a number
    for this field. `modal_dc_code` now comes from facts' own `dc_code` column, which is
    populated on all 64,462 `unresolved` rows, so the diagnosis and the rate are finally read
    off the same table.
    """
    out: list[Finding] = []
    facts = sub.facts
    if facts.empty:
        return out

    regimes = sub.regime_by_ticker
    verdicts = _absence_verdicts(sub)
    fields = sub.facts_fields
    sub.denominator("coverage_field", facts["ticker"].nunique() * len(fields))

    for ticker, rows in facts.groupby("ticker"):
        regime = regimes.get(str(ticker))
        periods = rows["period_of_report"].nunique()
        if not periods:
            continue
        resolved = (rows[rows["value"].notna()]
                    .groupby("field")["period_of_report"].nunique())
        # ONE pass per ticker for both the diagnosis and the exhibit. Doing either per
        # FINDING costs a full mask of the 316k-row facts frame each time, and this check
        # produces ~682 of them.
        misses = rows[rows["value"].isna()].sort_values("filing_date")
        modal = (misses.dropna(subset=["dc_code"]).groupby("field")["dc_code"]
                 .agg(lambda codes: codes.mode().iloc[0]) if len(misses) else {})
        exhibits = misses.groupby("field").last() if len(misses) else None
        fallback = rows.sort_values("filing_date").iloc[-1]

        for field in fields:
            covered = int(resolved.get(field, 0))
            null_rate = 1.0 - covered / periods
            if null_rate < COVERAGE_NULL_RATE:
                continue
            verdict = verdicts.get((regime, field))
            if verdict in (None, "STRUCTURAL", "TOO FEW PEERS"):
                continue                       # no evidence, or evidence that it is expected
            # The EXHIBIT: the most recent filing in which this field failed to resolve. The
            # finding is series-grain, so no single accession "caused" it -- but a reviewer
            # has to open SOMETHING, and "the last filing we looked in and did not find it"
            # is the one that settles whether the caption is there and we missed it.
            hit = (exhibits.loc[field]
                   if exhibits is not None and field in exhibits.index else None)
            exhibit = hit if hit is not None else fallback
            out.append(Finding(
                check_name="coverage_field", ticker=str(ticker),
                severity=HIGH if verdict == "UNIVERSAL" else MEDIUM, tier=1,
                substrate=FACTS, field=field,
                period_key=period_key_for_range(rows["period_of_report"].min(),
                                                rows["period_of_report"].max()),
                as_of=exhibit["filing_date"],
                accession_number=_text(exhibit["accession_number"]),
                cik=_text(exhibit.get("cik")),
                observed=null_rate, expected=0.0,
                detail={"verdict": verdict, "regime": regime,
                        "periods_covered": covered,
                        "periods_reported": int(periods),
                        "modal_dc_code": _text(modal.get(field)) if len(misses) else None,
                        "exhibit_is": ("the latest filing in which this field did not resolve"
                                       if hit is not None else
                                       "the filer's latest filing -- this field produced no "
                                       "row at all, resolved or unresolved"),
                        "why": ("every filer in this regime resolves this field, so a miss "
                                "is a defect in OUR extraction"
                                if verdict == "UNIVERSAL" else
                                "some filers in this regime resolve it and some do not -- "
                                "only the filing can settle it")}))
    return out


def _absence_verdicts(sub: Substrates) -> dict[tuple[str | None, str], str]:
    """`{(regime, field): STRUCTURAL | UNIVERSAL | MIXED | TOO FEW PEERS}` from stored facts.

    The absorbed `audit_absence_evidence.py`, verbatim in logic and re-pointed at the
    substrate frame instead of the sweep parquet ledgers. Zero rules, zero config: it counts
    how many of a regime's filers ever resolved a NUMBER for the field.
    """
    facts = sub.facts
    if facts.empty or "regime" not in facts.columns:
        return {}
    resolved = facts.assign(resolved=facts["value"].notna())
    per_filer = (resolved.groupby(["regime", "field", "ticker"])["resolved"].any()
                 .reset_index())
    grouped = (per_filer.groupby(["regime", "field"])
               .agg(filers=("ticker", "nunique"), resolving=("resolved", "sum")))
    verdicts: dict[tuple[str | None, str], str] = {}
    for (regime, field), row in grouped.iterrows():
        if row["filers"] < MIN_PEERS_FOR_ABSENCE_VERDICT:
            verdicts[(regime, field)] = "TOO FEW PEERS"
        elif row["resolving"] == 0:
            verdicts[(regime, field)] = "STRUCTURAL"
        elif row["resolving"] == row["filers"]:
            verdicts[(regime, field)] = "UNIVERSAL"
        else:
            verdicts[(regime, field)] = "MIXED"
    return verdicts


# TOMBSTONE: `_modal_code(sub, ticker, field)` is deleted. It masked the whole 316k-row facts
# frame once per FINDING, and `coverage_field` -- its only caller -- produces ~682 of them.
# The diagnosis is now grouped once per ticker inside the check, alongside the exhibit
# accession, which needs the same scan. A `not_disclosed` code remains a statement about our
# concept MAP and never about the filing; that is what the exhibit is for.


@check(name="expected_absent_drift", tier=1, substrate=FACTS, severity=INFO,
       grain=GRAIN_SERIES, expected_fire_rate_ceiling=1.0)
def expected_absent_drift(sub: Substrates) -> list[Finding]:
    """A value is PRESENT where the register says the field is `expected_absent`.

    `info`, always. The register is a measurement, so it decays: a filer that starts tagging a
    caption its regime template does not require has not done anything wrong, and our register
    cell has simply gone stale. Reporting it is how the register stays honest without anyone
    having to re-derive it -- the same reasoning that makes `catalogue_exclusion_cost` visible
    every run rather than filed in a report nobody re-reads.
    """
    out: list[Finding] = []
    facts = sub.facts
    if facts.empty:
        return out
    regimes = sub.regime_by_ticker
    fields = sub.facts_fields
    sub.denominator("expected_absent_drift", facts["ticker"].nunique() * len(fields))

    for ticker, rows in facts.groupby("ticker"):
        regime = regimes.get(str(ticker))
        if not regime:
            continue
        tagged = rows[rows["value"].notna()]
        counts = tagged.groupby("field").size()
        for field in fields:
            if not sub.catalogue.expected_absent(regime, field):
                continue
            present = int(counts.get(field, 0))
            if not present:
                continue
            latest = tagged[tagged["field"] == field].sort_values("filing_date").iloc[-1]
            out.append(Finding(
                check_name="expected_absent_drift", ticker=str(ticker), severity=INFO, tier=1,
                substrate=FACTS, field=field,
                period_key=period_key_for_range(rows["period_of_report"].min(),
                                                rows["period_of_report"].max()),
                as_of=latest["filing_date"],
                accession_number=str(latest["accession_number"]),
                cik=str(latest.get("cik") or ""),
                source_concept=_text(latest.get("source_concept")),
                observed=float(present), expected=0.0,
                detail={"regime": regime, "facts_tagged": present,
                        "latest_concept": _text(latest.get("source_concept")),
                        "why": "the register declares this field absent for the regime and "
                               "the filer tags it anyway -- the register has drifted, and "
                               "PGR really does tag capex. `latest_concept` names the tag "
                               "the filer actually used, which is what settles it"}))
    return out


# --------------------------------------------------------------------------- #
# 4. identities and impossible values                                          #
# --------------------------------------------------------------------------- #

@check(name="cross_identity", tier=1, substrate=FACTS, severity=CRITICAL, grain=GRAIN_CELL,
       expected_fire_rate_ceiling=0.05)
def cross_identity(sub: Substrates) -> list[Finding]:
    """`Assets == Liabilities + Equity`, and the gross-profit relation -- at DIFFERENT severities.

    ## ON FACTS, PER FILED STATEMENT -- which is what makes a break investigable

    Each finding names one `(accession_number, period_end)`: one balance sheet, as the filer
    published it, with an `edgar_url` attached. The history version could not do that -- it
    had no accession to attach -- so its 254 findings, including all seven criticals, arrived
    with a NULL URL and no way for an agent to reach the filing.

    Measured on the live 54-ticker table, the move is strictly additive: the SAME seven filers
    fail on either substrate (UNH, PGR, AMT, EQIX, VRT, SPG, NVDA), and facts finds 144 breaks
    to history's 64 because it tests 4,763 statements to history's 3,229. A filing carries
    comparatives and each is a separate published claim.

    ## The derived-total skip is GONE, because facts cannot contain a derived total

    On history, `totalLiabilities` was computed as `totalAssets - stockholdersEquity` for the
    filers who never tag `us-gaap:Liabilities`, and testing the identity on such a row is
    testing `A - E + E == A` -- arithmetic, which passes on any numbers at all, including
    wrong ones. So the check read `fundamentals_reason_codes` and skipped every
    `derived_identity` row.

    `fundamentals_facts` is strictly as-filed: every `resolution_method` it carries resolves
    from a concept the filer tagged, and nothing in it is computed from the identity under
    test. The skip is not merely unnecessary, it is unexpressible -- and its absence is why
    ETN's -$8.2bn `derived_identity` liability, the loudest break on the history substrate,
    correctly has no facts counterpart at all.

    This remains the check the plan warns must not "helpfully" propose summing the liability
    legs to fix it: 0 of 44 10-Ks declare a `Liabilities` total, leg-sets vary by filer and by
    year, and an unlisted sibling is dropped SILENTLY -- a balance-sheet total short by a
    caption that looks entirely plausible.

    HCA's negative stockholders' equity is CORRECT and must pass; the identity is signed
    arithmetic and never assumes a sign.

    ## THE FIRST CALIBRATION RUN REWROTE THIS CHECK. Both halves were wrong.

    Run 1 (2026-08-24) fired 293 `critical` findings, 9.0% of rows, on a 2% ceiling. Challenged
    before the data was, per the rule -- and the check lost both arguments.

    **(a) The balance sheet has a THIRD component we do not carry.** The equity side can be
    tagged EX-NCI, so the identity is tested on BOTH bases now and passes if either foots --
    which silences 39 of 103. What survives is 64 rows on UNH (16), EQIX (12), PGR (12),
    AMT (12), VRT (7), NVDA (2), SPG (2), ETN (1), and the mechanism is **temporary /
    mezzanine equity**: redeemable NCI and OP units sit BETWEEN liabilities and equity under
    ASC 480-10-S99, so `Assets = Liabilities + TemporaryEquity + Equity`. `us-gaap:Liabilities`
    excludes it and `StockholdersEquity` excludes it, and the 69-column contract has no column
    for it. VRT is the purest case: its pre-merger SPAC trust IS temporary equity, which is why
    its gap is 95%.

    A gap we cannot decompose is NOT "provably wrong", so it cannot be `critical` -- that is
    what `critical` means on this ladder. It fires `high`, and the payload names all three
    candidate mechanisms so the reviewing agent does not re-derive them. The real repair is a
    CATALOGUE field (`temporaryEquity`), which is Phase 9's decision and not the validator's.

    Above `IDENTITY_GROSS_BREAK` no mezzanine component could account for the gap and it IS
    provable, so it stays `critical`. On the live table that is exactly two rows: ETN's 2012
    Irish-redomicile holdco shell (section 5a-2's open case, a 172,559x gap) and VRT's SPAC.

    **(b) `GrossProfit == Revenue - COGS` is NOT AN ACCOUNTING IDENTITY.** All 191 failures
    were `industrial`, at 15-74%: TMO 50 (33.6%), EQIX 48 (15.0%), CVS 31 (73.9%), CAT 31
    (39.3%), COST 28 (39.7%). `grossProfit` resolves from the filer's OWN `us-gaap:GrossProfit`
    tag, and each filer computes it on its own cost basis -- CVS excludes benefit costs, COST
    nets membership fees, CAT excludes certain items. Both numbers are right and the PREMISE
    was wrong. It is kept, because a large gap can still indicate a mis-resolved `costOfRevenue`,
    but at `medium`: a candidate, look, do not assume.
    """
    out: list[Finding] = []
    sheets = _statements(sub, _IDENTITY_FIELDS, ("instant",))
    income = _statements(sub, _GROSS_PROFIT_FIELDS, _DURATION_TYPES,
                         extra_keys=("duration_type", "period_start"))
    sub.denominator("cross_identity", len(sheets) + len(income))

    for row in sheets.itertuples():
        out.extend(_balance_sheet_finding(row))
    for row in income.itertuples():
        out.extend(_gross_profit_finding(row))
    # A FILING CARRIES COMPARATIVES, so one balance-sheet date appears in several accessions
    # -- AMT's 2016-12-31 sheet is in five of them, and all five report the same break.
    # `finding_id` stops at `period_key`, so without this they upsert onto each other: 174
    # findings, 83 ids, 91 silently lost. Collapsing is also the honest shape -- one broken
    # balance sheet is one thing to look at, however many filings repeated it.
    return collapse_by_id(out, why=(
        "this (ticker, field, period_end) appears in more than one filing, because a 10-K "
        "carries its comparatives. The worst is reported here with its accession; the other "
        "filings that repeat it are listed above"))


def _balance_sheet_finding(row) -> list[Finding]:
    """`Assets == Liabilities + Equity`, tested on BOTH equity bases. Empty if either foots.

    `row` is one FILED balance sheet from `_statements` -- an `itertuples` record, not a
    history `Series`, so it carries the accession and CIK the finding needs.

    Both bases, because we cannot tell from the stored row which element `stockholdersEquity`
    resolved through -- and asserting a rule we cannot verify is exactly what this check was
    once corrected for. If the books foot with NCI included OR excluded, they foot.
    """
    assets, liabilities = _float(getattr(row, "totalAssets", None)), \
        _float(getattr(row, "totalLiabilities", None))
    equity = _float(getattr(row, "stockholdersEquity", None))
    nci = _float(getattr(row, "minorityInterest", None))
    if assets is None or liabilities is None or equity is None:
        return []
    if abs(assets) < IDENTITY_MIN_MAGNITUDE:
        return []                     # a shell too small for a relative gap to mean anything

    ex_nci = liabilities + equity
    with_nci = ex_nci + (nci or 0.0)
    scale = max(abs(assets), abs(ex_nci), IDENTITY_MIN_MAGNITUDE)
    gaps = {"ex_nci": (assets - ex_nci) / scale, "with_nci": (assets - with_nci) / scale}
    if min(abs(g) for g in gaps.values()) <= IDENTITY_TOLERANCE:
        return []                     # foots on at least one basis

    # The SMALLER of the two gaps is the honest one to report: it is the residual that no
    # equity basis explains, which is the quantity a reviewer actually has to account for.
    basis, gap = min(gaps.items(), key=lambda kv: abs(kv[1]))
    return [Finding(
        check_name="cross_identity", ticker=str(row.ticker),
        severity=CRITICAL if abs(gap) > IDENTITY_GROSS_BREAK else HIGH, tier=1,
        substrate=FACTS, field="totalAssets",
        period_key=str(pd.Timestamp(row.period_end).date()), as_of=row.filing_date,
        accession_number=str(row.accession_number), cik=str(row.cik),
        observed=assets, expected=(ex_nci if basis == "ex_nci" else with_nci),
        deviation=gap,
        detail={"identity": "totalAssets == totalLiabilities + stockholdersEquity [+ NCI]",
                "best_basis": basis,
                "gap_ex_nci": gaps["ex_nci"], "gap_with_nci": gaps["with_nci"],
                "period_end": str(pd.Timestamp(row.period_end).date()),
                "parts": {"totalLiabilities": liabilities,
                          "stockholdersEquity": equity,
                          "minorityInterest": nci},
                "tolerance": IDENTITY_TOLERANCE,
                "candidate_mechanisms": [
                    "TEMPORARY / MEZZANINE EQUITY -- redeemable NCI and REIT OP units sit "
                    "between liabilities and equity under ASC 480-10-S99, and the 69-column "
                    "contract carries no column for it. THE LIKELIEST cause of a 1-4% gap",
                    "an ex-NCI equity element that minorityInterest did not close",
                    "a genuine resolution defect in one of the three inputs"],
                "why": "the books do not foot on EITHER equity basis. Not `critical` below "
                       f"{IDENTITY_GROSS_BREAK:.0%} because a gap we cannot decompose is not "
                       "PROVABLY wrong -- the missing column is a catalogue decision"})]


def _gross_profit_finding(row) -> list[Finding]:
    """`grossProfit` vs `totalRevenue - costOfRevenue`. `medium` -- a basis difference, not an
    identity. See `cross_identity`'s docstring for the five filers that proved it.

    All three legs come from ONE window of ONE filing: `_statements` keys the income pivot on
    `duration_type` and `period_start` as well as `period_end`, so an annual `grossProfit` is
    never differenced against a quarterly `costOfRevenue`. On the history substrate the row was
    a carry-forward snapshot and that guarantee did not exist.
    """
    gross = _float(getattr(row, "grossProfit", None))
    revenue = _float(getattr(row, "totalRevenue", None))
    cost = _float(getattr(row, "costOfRevenue", None))
    if gross is None or revenue is None or cost is None:
        return []
    observed, expected = gross, revenue - cost
    scale = max(abs(observed), abs(expected), IDENTITY_MIN_MAGNITUDE)
    if abs(observed - expected) <= IDENTITY_TOLERANCE * scale:
        return []
    return [Finding(
        check_name="cross_identity", ticker=str(row.ticker), severity=MEDIUM, tier=1,
        substrate=FACTS, field="grossProfit",
        period_key=str(pd.Timestamp(row.period_end).date()), as_of=row.filing_date,
        accession_number=str(row.accession_number), cik=str(row.cik),
        observed=observed, expected=expected, deviation=(observed - expected) / scale,
        detail={"relation": "grossProfit ~ totalRevenue - costOfRevenue",
                "duration_type": str(row.duration_type),
                "period_end": str(pd.Timestamp(row.period_end).date()),
                "parts": {"totalRevenue": revenue, "costOfRevenue": cost},
                "tolerance": IDENTITY_TOLERANCE,
                "known_false_positive_population":
                    "filers whose own us-gaap:GrossProfit tag uses a different cost basis "
                    "than our costOfRevenue -- CVS excludes benefit costs, COST nets "
                    "membership fees, CAT excludes certain items. Measured: 191 of 191 "
                    "failures on the first run were industrial filers of exactly this shape",
                "why": "NOT an accounting identity -- a basis difference, so `medium`. Still "
                       "worth a look: a large gap can indicate a mis-resolved costOfRevenue"})]


# TOMBSTONE: `_derived_liability_rows` is deleted. It skipped history rows whose
# `totalLiabilities` carried `derived_identity` / `derived_identity_nci_assumed_zero`, because
# testing `A - E + E == A` on them was arithmetic rather than a check. `fundamentals_facts` is
# strictly as-filed and holds no derived total, so there is nothing to skip -- and the check
# is stronger for it, since a filer that never tags `us-gaap:Liabilities` is now simply absent
# from the identity's population rather than silently excused inside it.


@check(name="impossible_value", tier=1, substrate=FACTS, severity=HIGH, grain=GRAIN_CELL,
       expected_fire_rate_ceiling=0.01)
def impossible_value(sub: Substrates) -> list[Finding]:
    """Values no filer could report and be right -- FLAG ONLY, nothing is nulled.

    This is the residue of v2's mutating "Layer A", moved out of the guard by decision 46. Four
    genuinely-impossible rules stay in `build_history.HARD_GUARDS` and null the value before it
    is written; EVERYTHING ELSE is reported here and left in the table for a human to judge.

    That split is the 745-row lesson made structural. v2's proposed `[-1, 1]` bound on
    `debtToEquity` nulls HCA's correct negative ratio -- its equity IS negative -- and every
    filer whose debt exceeds its equity. A rule that wrong must never be allowed to delete
    data; it can only ever raise a question.

    A negative revenue or cost line IS reported and IS sometimes correct (APA's -$467M revenue
    on an unrealised-FX root was a defect; VRT's zeros were not). That is what `high` means:
    probably wrong, a named mechanism says so, go and look.

    ## ON FACTS, so the finding names the tag that produced the number

    Each finding now carries `source_concept`, `resolution_method` and an `accession_number`.
    That is the difference between "APA's revenue is negative somewhere in 2015" and "APA
    tagged THIS concept in THIS filing and it resolved to -$467M" -- the second is a work item,
    the first is a rumour.

    THE `epsDiluted` RULE IS DROPPED. `epsDiluted` is one of the twelve columns
    `build_history` derives and it exists nowhere in `fundamentals_facts`, so the rule has no
    substrate to run on. `MAX_ABS_EPS` is kept with its reasoning intact against the day EPS
    gets a derived-value check. Nothing else is lost: the other three rules run on as-filed
    numbers and always could.
    """
    out: list[Finding] = []
    facts = sub.facts
    if facts.empty:
        return out
    sub.denominator("impossible_value", int(facts["value"].notna().sum()))

    rules: list[tuple[str, object, str]] = [
        ("totalRevenue", lambda v: v < 0,
         "a negative top line -- APA's -$467M came from an unrealised-FX root"),
        ("costOfRevenue", lambda v: v < 0, "a negative cost of sales"),
        ("totalAssets", lambda v: v < 0, "negative total assets"),
    ]
    for field, is_impossible, why in rules:
        rows = facts[(facts["field"] == field) & facts["value"].notna()]
        for row in rows.itertuples():
            value = float(row.value)
            if not is_impossible(value):
                continue
            out.append(Finding(
                check_name="impossible_value", ticker=str(row.ticker), severity=HIGH,
                tier=1, substrate=FACTS, field=field,
                period_key=str(pd.Timestamp(row.period_end).date()), as_of=row.filing_date,
                accession_number=str(row.accession_number), cik=str(row.cik),
                source_concept=_text(getattr(row, "source_concept", None)),
                resolution_method=_text(getattr(row, "resolution_method", None)),
                observed=value,
                detail={"rule": why,
                        "flag_only": True,
                        "duration_type": _text(getattr(row, "duration_type", None)),
                        "why": "reported, NOT nulled -- only the four impossible rules in "
                               "build_history.HARD_GUARDS ever delete a value"}))
    return out


# --------------------------------------------------------------------------- #
# 5. filing continuity, lag, amendments                                        #
# --------------------------------------------------------------------------- #

@check(name="filing_lag", tier=1, substrate=FACTS, severity=MEDIUM, grain=GRAIN_ROW,
       expected_fire_rate_ceiling=0.01)
def filing_lag(sub: Substrates) -> list[Finding]:
    """`filing_date - period_of_report` inside a real SEC filing window, per FILING.

    Past `MAX_FILING_LAG_DAYS` the filing reports a period for which a fresher number already
    existed. `medium` and not `high`: a delinquent filer is a real thing, and the one case
    beyond the bound on the rebuilt roster is SMCI's FY2017 10-K at 686 days, filed during its
    Nasdaq delisting. The check exists because an earlier build had a median lag of 401 days
    for ATO and lags out to 1,884 -- a broken grain, not a delinquent filer.

    ## `period_of_report`, NEVER `period_end` -- the trap this check would otherwise fall into

    A 10-K carries its comparatives, so `period_end` on a single accession ranges over several
    years while `filing_date` does not. Differencing those reports EVERY comparative in EVERY
    annual report as a delinquent filing -- thousands of findings, all of them arithmetic.
    `period_of_report` is the filing's own reporting date, it is populated on all 316,136 rows
    of the live table, and it is the only correct left-hand side here.
    """
    out: list[Finding] = []
    filings = _filings(sub)
    if filings.empty or "period_of_report" not in filings.columns:
        return out
    sub.denominator("filing_lag", len(filings))
    lag = (filings["filing_date"] - filings["period_of_report"]).dt.days
    for position in lag[lag > MAX_FILING_LAG_DAYS].index:
        row = filings.loc[position]
        out.append(Finding(
            check_name="filing_lag", ticker=str(row["ticker"]), severity=MEDIUM, tier=1,
            substrate=FACTS, field="",
            period_key=str(pd.Timestamp(row["period_of_report"]).date()),
            as_of=row["filing_date"], accession_number=str(row["accession_number"]),
            cik=str(row.get("cik") or ""), observed=float(lag[position]),
            expected=float(MAX_FILING_LAG_DAYS),
            detail={"period_of_report": str(pd.Timestamp(row["period_of_report"]).date()),
                    "form": str(row.get("form") or ""),
                    "why": "a 10-Q lands ~35-45 days after quarter end and a 10-K ~60-90; "
                           "beyond 200 the filing reports a period for which a fresher "
                           "number already existed -- or the filer is genuinely delinquent"}))
    return out


@check(name="filing_continuity", tier=1, substrate=FACTS, severity=HIGH, grain=GRAIN_TICKER,
       expected_fire_rate_ceiling=0.10)
def filing_continuity(sub: Substrates) -> list[Finding]:
    """Filings per ticker per year inside the 3.0-5.5 band, unless something EXCUSES it.

    The defect: a CIK truncation. APA, GOOGL and ETN each changed CIK -- a redomicile, a
    re-registration, a holding-company reorganisation -- and without a cutover entry the
    history simply starts late, silently, with no null for any gate to find.

    THREE EXCUSING MECHANISMS, and only a short history with NONE of them is a work item:

      1. a `fundamentals_cik_cutover.json` entry (the fix, already applied for the known ones);
      2. a recent first-trade or index-add date -- a 2019 IPO has no 2015 filings and never did;
      3. an `accepted` entry in the check register.

    (2) is not testable from these substrates -- the listing date is not in `fundamentals_*` --
    so it is reported as an OPEN QUESTION in the finding's `detail` rather than silently
    assumed either way. That is the honest shape: the check says "this filer's history is
    short and no cutover entry explains it", and the agent's first move is to look up whether
    it was listed.
    """
    out: list[Finding] = []
    facts = sub.facts
    if facts.empty:
        return out
    sub.denominator("filing_continuity", facts["ticker"].nunique())

    for ticker, rows in facts.groupby("ticker"):
        accessions = rows.drop_duplicates("accession_number")
        dates = pd.to_datetime(accessions["filing_date"]).dropna()
        if len(dates) < 8:
            continue                    # too few filings to have a rate at all
        years = (dates.max() - dates.min()).days / 365.25
        if years < 2:
            continue
        rate = len(dates) / years
        low, high = FILINGS_PER_YEAR_BAND
        if low <= rate <= high:
            continue
        out.append(Finding(
            check_name="filing_continuity", ticker=str(ticker), severity=HIGH, tier=1,
            substrate=FACTS, field="",
            period_key=period_key_for_range(dates.min(), dates.max()),
            observed=rate, expected=(low + high) / 2,
            cik=sub.cik_for(str(ticker)),
            detail={"filings": int(len(dates)), "years": round(years, 2),
                    "band": list(FILINGS_PER_YEAR_BAND),
                    "distinct_ciks": sorted(set(accessions["cik"].dropna().astype(str))),
                    "open_question": "is there a listing / index-add date that explains a "
                                     "short history? not answerable from fundamentals_* -- "
                                     "look it up before writing a cutover entry",
                    "why": "a rate below the band with no cik_cutover entry is a MISSING "
                           "CUTOVER ENTRY: the history starts late and silently"}))
    return out


@check(name="amendment_ledger", tier=1, substrate=FACTS, severity=INFO, grain=GRAIN_TICKER,
       expected_fire_rate_ceiling=1.0)
def amendment_ledger(sub: Substrates) -> list[Finding]:
    """Amendment accessions REFUSED by the 365-day cutoff, declared per ticker.

    `info`, and it is a declaration rather than a complaint. Decision 34 refuses an amendment
    landing more than a year after the original because a quarter stays inside a live TTM
    window for about twelve months, so a later restatement is a number a long/short model
    cannot learn from. That is a deliberate loss of real data, and the amount lost has to be
    VISIBLE every run rather than recorded once in a plan: 3 of 36 accessions (8%) on the
    rebuilt roster.
    """
    out: list[Finding] = []
    facts = sub.facts
    if facts.empty or "is_amendment" not in facts.columns:
        return out
    sub.denominator("amendment_ledger", facts["ticker"].nunique())

    amended = facts[facts["is_amendment"].fillna(False).astype(bool)]
    if amended.empty:
        return out
    originals = facts.groupby(["ticker", "period_of_report"])["filing_date"].min()
    joined = amended.join(originals.rename("original"), on=["ticker", "period_of_report"])
    lag = (joined["filing_date"] - joined["original"]).dt.days
    per_accession = lag.groupby([joined["ticker"], joined["accession_number"]]).max().dropna()

    for ticker, group in per_accession.groupby(level=0):
        refused = group[group > 365]
        if refused.empty:
            continue
        out.append(Finding(
            check_name="amendment_ledger", ticker=str(ticker), severity=INFO, tier=1,
            substrate=FACTS, observed=float(len(refused)), expected=0.0,
            cik=sub.cik_for(str(ticker)),
            detail={"refused_accessions": [str(a) for _, a in refused.index],
                    "max_lag_days": float(refused.max()),
                    "amendment_accessions": int(len(group)),
                    "why": "decision 34 refuses an amendment landing >365d after the "
                           "original; this is the real data that costs, declared"}))
    return out


@check(name="same_day_collapse", tier=1, substrate=FACTS, severity=INFO, grain=GRAIN_TICKER,
       expected_fire_rate_ceiling=1.0)
def same_day_collapse(sub: Substrates) -> list[Finding]:
    """(ticker, date) pairs carrying more than one accession, declared.

    `info`. The same-day collapse rule (decision 37) resolves two filings on one date into one
    row by form precedence, and this reports how often that rule is actually LOAD-BEARING: 9 of
    3,273 pairs on the rebuilt roster, up to 4 accessions on one day. A rule that never fires
    is a rule nobody is testing, so the count is published rather than assumed.
    """
    out: list[Finding] = []
    facts = sub.facts
    if facts.empty:
        return out
    per_day = facts.groupby(["ticker", "filing_date"])["accession_number"].nunique()
    sub.denominator("same_day_collapse", len(per_day))
    collapsed = per_day[per_day > 1]
    for (ticker, date), count in collapsed.items():
        out.append(Finding(
            check_name="same_day_collapse", ticker=str(ticker), severity=INFO, tier=1,
            substrate=FACTS, period_key=str(pd.Timestamp(date).date()),
            observed=float(count), expected=1.0, cik=sub.cik_for(str(ticker)),
            detail={"why": "form precedence (10-K > 10-K/A > 10-Q > 10-Q/A) decided this "
                           "row's provenance -- the rule is load-bearing, not theoretical"}))
    return out


# --------------------------------------------------------------------------- #
# 6. the declared costs -- `info`, visible every run                           #
# --------------------------------------------------------------------------- #
#
# NAMED FOR THE CATALOGUE, not for the finding ledger. Both checks below read
# `configs/fundamentals/fundamentals_kpis.json` -- its `never_use` exclusions and its
# `by_ticker` overrides. They were once called `register_cost` and `register_coverage`, and
# that name meant the CATALOGUE's exclusion register; when the separate settled-findings
# register (`fundamentals_check.json`) was retired, the collision was very nearly enough to
# get both of these deleted along with it. Renamed so the next reader cannot make that
# mistake: the only "register" left in this package is the CHECK_REGISTRY.

@check(name="catalogue_exclusion_cost", tier=1, substrate=FACTS, severity=INFO,
       grain=GRAIN_SERIES, expected_fire_rate_ceiling=1.0)
def catalogue_exclusion_cost(sub: Substrates) -> list[Finding]:
    """The QUANTIFIED cost of each catalogue `never_use` exclusion, republished every run.

    A `never_use` entry or an excluded extension leg buys correctness somewhere and costs
    coverage somewhere else, and the cost is only defensible while somebody can still see it.
    This check is the only place that number is published, which is why it survived the
    retirement of the similarly-named finding register.
    NEE's mixed acquisition-plus-capex line is the headline: excluding it understates 2018-19
    capex by up to **$5.2bn**. That number belongs in the nightly output, not in a plan.

    Reads the catalogue's own `never_use` blocks, so it cannot drift from what the resolver
    actually does. `info` -- no action expected, and none wanted.

    NO `edgar_url`, AND THAT IS CORRECT. This check and `catalogue_override_coverage` are
    diagnostics about OUR configuration, not about a filing: the subject is a `never_use`
    entry, and no accession caused it. They are marked `FACTS` because that is where their
    regime map is read from now, not because a filing is implicated. Every other check in this
    tier names an accession, and one that cannot should say why rather than look broken.
    """
    out: list[Finding] = []
    regimes = sub.regime_by_ticker
    if not regimes:
        return out
    # tickers x FIELDS. The first calibration run divided by tickers alone and reported an
    # 825.9% "fire rate" -- arithmetic, not behaviour, but it is precisely the kind of nonsense
    # that trains a reader to ignore the ceiling column.
    sub.denominator("catalogue_exclusion_cost",
                    len(regimes) * len(sub.catalogue.history_fields))
    for ticker, regime in sorted(regimes.items()):
        for field in sub.catalogue.history_fields:
            never_use = sub.catalogue.field(field).never_use(regime)
            if not never_use:
                continue
            out.append(Finding(
                check_name="catalogue_exclusion_cost", ticker=ticker, severity=INFO, tier=1,
                substrate=FACTS, field=field, observed=float(len(never_use)),
                detail={"regime": regime, "excluded_concepts": sorted(never_use),
                        "reasons": {k: str(v) for k, v in never_use.items()},
                        "why": "an exclusion buys correctness and costs coverage; the cost "
                               "stays visible so it can be re-argued"}))
    return out


@check(name="catalogue_override_coverage", tier=1, substrate=FACTS, severity=INFO,
       grain=GRAIN_TICKER, expected_fire_rate_ceiling=1.0)
def catalogue_override_coverage(sub: Substrates) -> list[Finding]:
    """Which filers run on a PARTIAL catalogue -- i.e. carry per-ticker overrides.

    `info`. 17 of ~500 filers have a `by_ticker` cell somewhere, which means their numbers rest
    on a hand-authored exception rather than on the general rule. That is not wrong; it is a
    smaller evidence base, and a reader comparing two tickers should know which one is which.
    """
    out: list[Finding] = []
    regimes = sub.regime_by_ticker
    if not regimes:
        return out
    sub.denominator("catalogue_override_coverage", len(regimes))
    for ticker in sorted(regimes):
        overridden = [f for f in sub.catalogue.history_fields
                      if sub.catalogue.filer_leaves(ticker, f) is not None
                      or sub.catalogue.periodicity_shapes(ticker, f) is not None]
        if not overridden:
            continue
        out.append(Finding(
            check_name="catalogue_override_coverage", ticker=ticker, severity=INFO, tier=1,
            substrate=FACTS, observed=float(len(overridden)),
            detail={"fields_with_by_ticker_overrides": overridden,
                    "why": "this filer's numbers rest on a hand-authored exception rather "
                           "than the general rule -- a smaller evidence base, declared"}))
    return out


@check(name="adjustment_unguarded", tier=1, substrate=FACTS, severity=INFO, grain=GRAIN_CELL,
       expected_fire_rate_ceiling=1.0)
def adjustment_unguarded(sub: Substrates) -> list[Finding]:
    """An adjustment that fired on SILENCE rather than on positive evidence.

    `info`, and the population is known: all 128 `ppeNet` lease adjustments. The distinction
    matters because an adjustment applied because a concept was ABSENT is an inference, while
    one applied because a concept was PRESENT and said so is a reading. Absence of a tag is
    not evidence -- the same principle that makes `not_disclosed` a statement about our concept
    map rather than about the filing.
    """
    out: list[Finding] = []
    facts = sub.facts
    if facts.empty or "adjustment" not in facts.columns:
        return out
    adjusted = facts[facts["adjustment"].notna() & (facts["adjustment"] != "")]
    sub.denominator("adjustment_unguarded", len(facts))
    for _, row in adjusted.iterrows():
        out.append(Finding(
            check_name="adjustment_unguarded", ticker=str(row["ticker"]), severity=INFO,
            tier=1, substrate=FACTS, field=str(row["field"]),
            period_key=str(pd.Timestamp(row["period_end"]).date()),
            observed=_float(row["value"]),
            source_concept=_text(row.get("source_concept")),
            resolution_method=_text(row.get("resolution_method")),
            accession_number=_text(row.get("accession_number")),
            cik=sub.cik_for(str(row["ticker"])),
            detail={"adjustment": _text(row.get("adjustment")),
                    "why": "an adjustment applied because a concept was ABSENT is an "
                           "inference, not a reading"}))
    return out


# --------------------------------------------------------------------------- #
# shared helpers                                                               #
# --------------------------------------------------------------------------- #

def _text(value) -> str | None:
    """A payload string as `str`, or None -- never the string "nan".

    `itertuples` and `.get` on a facts row hand back `float('nan')` for a missing object
    column, and `str(nan)` is "nan", which then travels into `source_concept` and reads as a
    concept name. The same shape as `_float`, for the same reason.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return str(value)


def _float(value) -> float | None:
    """A cell as a plain float, or None for NULL/NaN. Never propagates NaN into a payload."""
    if value is None or (not isinstance(value, (int, float, np.floating)) and pd.isna(value)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(out) else out


def _text(value) -> str | None:
    """A provenance cell as a string, or None. Keeps `pd.NA`/NaN out of the JSON payload."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value)
    return text if text and text.lower() not in ("nan", "none", "<na>") else None
