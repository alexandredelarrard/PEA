"""
tier3_internal.py  (src/validate/fundamentals/checks/)
--------------------------------------------------------------------------------------------
TIER 3 -- the PROVENANCE-INDEPENDENT tier, on `fundamentals_facts`.

Nobody validates by re-deriving a value and checking it matches itself. Every check here plays
the filer's own DISJOINT evidence against itself: a number we derived against a number the
filer published separately, one filing's statement of a period against another filing's, a
declared total against the leaves that foot to it. None of it needs an external source, and
that is exactly what makes deferring Tier 0 (XBRL-US DQC via Arelle) and Tier 4 (aggregators)
defensible rather than merely convenient.

## The eight checks and the v2 baselines they were calibrated against

| check                  | rule                                              | pre-4c baseline        |
|------------------------|---------------------------------------------------|------------------------|
| `holdout_q4`           | our derived Q4 vs the filer's OWN discrete Q4     | 591/752; 98.7/99.0% @1%|
| `annual_footing`       | four derived quarters vs the filer's annual       | 99.12% / 98.78% @2%    |
| `q4_footing`           | Q1+Q2+Q3+Q4 == FY, non-identity years only        | 99.9% of Q4 testable   |
| `leaf_vs_total`        | a leaf sum in one vintage vs a total in another   | 89/94; 76.4/78.7% exact|
| `cross_vintage`        | same (ticker, field, period) across filings       | 4.53% move >2%         |
| `derived_vs_asreported`| our epsDiluted / share counts vs the filer's      | 97.3% of 710 @0.5%     |
| `duplicate_fact`       | one filing tagging (concept, period) twice        | ORCL $7,623 vs $7,600  |
| `restatement_ledger`   | RECORD, never repair                              | BAC 98,581 -> 102,769  |

**Every one of those numbers is PRE-4c and is the number to BEAT, not an assertion.** They
predate the statement-role test, the `longTermDebt` reorder, the ORCL refusal, the CIK cutover
AND the Phase-5 PK fix that alone recovered 18,604 rows (5.5%) that were being silently
dropped. `configs/fundamentals/fundamentals_baselines.json` is where the re-measured versions
live, with their `n` and their substrate.

## RESTATEMENT vs DEFECT is separable WITHOUT external data

The single most useful property in this tier, and the reason `cross_vintage` is worth running:

  * a DERIVATION error leaves at least one quarter whose basis is not `as_reported`;
  * a RESTATEMENT leaves all four quarters as-filed, and they foot to the FIRST-FILED annual.

So `cross_vintage` classifies rather than merely flags, and `restatement_ledger` records the
restatements as `info` so that nobody later "fixes" our as-filed numbers toward a re-presented
aggregate. BAC's FY2023 revenue is 98,581M as filed and 102,769M re-presented; both are true,
and only the first one was knowable at the time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_extract.utils.fundamentals.periods import ANNUAL, QUARTERLY, YTD9
from src.validate.fundamentals.checks import FACTS, GRAIN_CELL, GRAIN_SERIES, check
from src.validate.fundamentals.finding import (
    CRITICAL, Finding, collapse_by_id, HIGH, INFO, MEDIUM, period_key_for_range)
from src.validate.fundamentals.substrate import Substrates

# --------------------------------------------------------------------------- #
# tolerances -- code, with the measurement that set each one                    #
# --------------------------------------------------------------------------- #

#: `holdout_q4` and `derived_vs_asreported`: within 1% is agreement. The v2 sweep measured
#: 98.73% / 98.99% of 591 forced derivations landing inside it, so the tolerance is where the
#: distribution already sits rather than a round number picked in advance.
HOLDOUT_TOLERANCE = 0.01

#: `annual_footing` and `q4_footing`: 2%. Looser than `holdout_q4` on purpose -- footing four
#: derived quarters accumulates four roundings, and the filer's own annual is itself rounded.
FOOTING_TOLERANCE = 0.02

#: `cross_vintage`: a re-statement of the SAME period across two filings moving by more than
#: this is worth reporting. 4.53% of annual windows moved >2% in the v2 sweep, and most of
#: those are genuine restatements rather than defects -- which is what the classifier is for.
VINTAGE_TOLERANCE = 0.02

#: `derived_vs_asreported`: our per-share figure against the filer's own tagged one. Tighter
#: still, because EPS is published to the cent and there is nothing to accumulate.
PER_SHARE_TOLERANCE = 0.005

#: Below this a relative comparison is meaningless -- a $3 move on a $2 base is a 150% error
#: that means nothing. Absolute, in reporting units.
MIN_COMPARABLE_MAGNITUDE = 1_000.0


# --------------------------------------------------------------------------- #
# the footing family                                                           #
# --------------------------------------------------------------------------- #

@check(name="q4_footing", tier=3, substrate=FACTS, severity=HIGH, grain=GRAIN_CELL,
       expected_fire_rate_ceiling=0.02)
def q4_footing(sub: Substrates) -> list[Finding]:
    """Q1 + Q2 + Q3 + Q4 == FY, on the filer's own tagged quarters only.

    ## Only on NON-IDENTITY years, and that restriction is the whole check

    Where our Q4 was DERIVED as `FY - YTD9`, footing the four quarters back to FY is an
    identity: it passes on any numbers at all, including wrong ones. The check therefore runs
    only on years where the filer tagged all four quarters itself, which is what makes a
    failure informative. On the v2 sweep 99.9% of Q4 rows were genuinely testable this way.

    The defect it catches: ORCL's FY2020 Q4 revenue at **$39,068M** -- a full-year `Revenues`
    fact stamped with a Q4 window, which foots to roughly four times the year.
    """
    out: list[Finding] = []
    facts = sub.facts
    if facts.empty:
        return out
    quarters = _latest_per_window(facts, QUARTERLY)
    annuals = _latest_per_window(facts, ANNUAL)
    if quarters.empty or annuals.empty:
        return out
    sub.denominator("q4_footing", len(annuals))
    by_series = _by_series(quarters)

    for _, annual in annuals.iterrows():
        window = _quarters_in(by_series.get((annual["ticker"], annual["field"])), annual)
        if len(window) != 4:
            continue                     # not four tagged quarters -> not testable, skip
        total = float(window["value"].sum())
        expected = float(annual["value"])
        if not _materially_different(total, expected, FOOTING_TOLERANCE):
            continue
        out.append(_finding_from(
            window.iloc[-1], sub, check_name="q4_footing", severity=HIGH, tier=3,
            observed=total, expected=expected,
            deviation=_relative(total, expected),
            detail={"quarters": [str(pd.Timestamp(p).date())
                                 for p in window["period_end"]],
                    "quarter_values": [float(v) for v in window["value"]],
                    "annual_period_end": str(pd.Timestamp(annual["period_end"]).date()),
                    "annual_accession": str(annual["accession_number"]),
                    "tolerance": FOOTING_TOLERANCE,
                    "why": "all four quarters are the FILER'S OWN, so this is not an "
                           "identity -- ORCL's FY2020 Q4 at $39,068M is this shape"}))
    return _collapse(out)


@check(name="annual_footing", tier=3, substrate=FACTS, severity=HIGH, grain=GRAIN_CELL,
       expected_fire_rate_ceiling=0.02)
def annual_footing(sub: Substrates) -> list[Finding]:
    """The filer's own YTD9 plus its own Q4 equals its own FY.

    The disjoint-evidence version of `q4_footing`: three numbers the filer published
    independently, on three different bases, that must reconcile. It runs where `q4_footing`
    cannot -- a filer that tags YTD9 and FY but not all four discrete quarters -- so the two
    together cover more years than either alone.

    Baseline: 99.12% / 98.78% within 2%.
    """
    out: list[Finding] = []
    facts = sub.facts
    if facts.empty:
        return out
    annuals = _latest_per_window(facts, ANNUAL)
    ytd9 = _latest_per_window(facts, YTD9)
    quarters = _latest_per_window(facts, QUARTERLY)
    if annuals.empty or ytd9.empty:
        return out
    sub.denominator("annual_footing", len(annuals))
    by_series = _by_series(quarters)

    ytd9_index = {(t, f, pd.Timestamp(p)): v for t, f, p, v in
                  zip(ytd9["ticker"], ytd9["field"], ytd9["period_end"], ytd9["value"])}
    for _, annual in annuals.iterrows():
        series = by_series.get((annual["ticker"], annual["field"]))
        ninth_month_end = _ytd9_end_for(series, annual)
        interim = ytd9_index.get((annual["ticker"], annual["field"], ninth_month_end))
        q4 = _q4_for(series, annual)
        if interim is None or q4 is None:
            continue
        total = float(interim) + float(q4)
        expected = float(annual["value"])
        if not _materially_different(total, expected, FOOTING_TOLERANCE):
            continue
        out.append(_finding_from(
            annual, sub, check_name="annual_footing", severity=HIGH, tier=3,
            observed=total, expected=expected, deviation=_relative(total, expected),
            detail={"ytd9": float(interim), "q4": float(q4),
                    "ytd9_period_end": str(ninth_month_end.date()),
                    "tolerance": FOOTING_TOLERANCE,
                    "why": "three numbers the filer published independently, on three "
                           "different bases, that must reconcile"}))
    return out


@check(name="holdout_q4", tier=3, substrate=FACTS, severity=HIGH, grain=GRAIN_CELL,
       expected_fire_rate_ceiling=0.02)
def holdout_q4(sub: Substrates) -> list[Finding]:
    """Force `Q4 = FY - YTD9` where the filer ALSO published its own discrete Q4, and compare.

    A genuine hold-out: the derivation the history build performs everywhere is re-run on the
    subset of years where the answer is independently known, and scored against it. That is
    the only way to put a number on how good the derivation IS, rather than on how often it
    runs. 591 of 752 cases were testable in the v2 sweep, and 98.73% / 98.99% landed within 1%.

    Where it disagrees, the usual mechanism is a mid-year basis change -- the FY and the YTD9
    were tagged on different bases -- which is `tag_switch_break`'s territory, and the finding
    says so.
    """
    out: list[Finding] = []
    facts = sub.facts
    if facts.empty:
        return out
    annuals = _latest_per_window(facts, ANNUAL)
    ytd9 = _latest_per_window(facts, YTD9)
    quarters = _latest_per_window(facts, QUARTERLY)
    if annuals.empty or ytd9.empty or quarters.empty:
        return out
    sub.denominator("holdout_q4", len(annuals))
    by_series = _by_series(quarters)

    ytd9_index = {(t, f, pd.Timestamp(p)): v for t, f, p, v in
                  zip(ytd9["ticker"], ytd9["field"], ytd9["period_end"], ytd9["value"])}
    for _, annual in annuals.iterrows():
        series = by_series.get((annual["ticker"], annual["field"]))
        published_q4 = _q4_for(series, annual)
        interim = ytd9_index.get((annual["ticker"], annual["field"],
                                  _ytd9_end_for(series, annual)))
        if published_q4 is None or interim is None:
            continue                       # not a hold-out case
        derived = float(annual["value"]) - float(interim)
        if not _materially_different(derived, float(published_q4), HOLDOUT_TOLERANCE):
            continue
        out.append(_finding_from(
            annual, sub, check_name="holdout_q4", severity=HIGH, tier=3,
            observed=derived, expected=float(published_q4),
            deviation=_relative(derived, float(published_q4)),
            detail={"derivation": "FY - YTD9", "annual": float(annual["value"]),
                    "ytd9": float(interim), "published_q4": float(published_q4),
                    "tolerance": HOLDOUT_TOLERANCE,
                    "likely_mechanism": "a mid-year basis change -- the FY and the YTD9 were "
                                        "tagged on different bases; see tag_switch_break",
                    "why": "the derivation the build performs everywhere, re-run where the "
                           "answer is independently known"}))
    return out


# --------------------------------------------------------------------------- #
# cross-vintage and the restatement ledger                                     #
# --------------------------------------------------------------------------- #

@check(name="cross_vintage", tier=3, substrate=FACTS, severity=HIGH, grain=GRAIN_CELL,
       expected_fire_rate_ceiling=0.06)
def cross_vintage(sub: Substrates) -> list[Finding]:
    """The same (ticker, field, period) tagged differently by two filings -- CLASSIFIED.

    ## Restatement vs defect, without any external data

    A filer re-presenting an earlier period is not a defect, and treating it as one is how a
    validator ends up "correcting" as-filed history toward an aggregate. The discriminator is
    the resolution basis, and it needs nothing outside these tables:

      * every vintage resolved AS REPORTED, disagreeing -> a RESTATEMENT. `info`. VLO's FY2012
        operating income is the worked case: first-filed $4,010M matches our four quarters to
        the dollar, last-restated $5,044M does not, and the first one is what was knowable;
      * at least one vintage resolved through a DERIVATION -> our arithmetic is in play, so it
        is a candidate DEFECT. `high`.

    Baseline: 4.53% of annual windows move by more than 2%.
    """
    out: list[Finding] = []
    facts = sub.facts
    if facts.empty:
        return out
    valued = facts[facts["value"].notna()]
    sub.denominator("cross_vintage", len(valued))

    keys = ["ticker", "field", "duration_type", "period_end"]
    grouped = _multi_row_groups(valued, keys).groupby(keys, sort=False)
    for (ticker, field, duration, period_end), rows in grouped:
        if len(rows) < 2:
            continue
        ordered = rows.sort_values("filing_date")
        first, last = float(ordered.iloc[0]["value"]), float(ordered.iloc[-1]["value"])
        if not _materially_different(last, first, VINTAGE_TOLERANCE):
            continue
        methods = {str(m) for m in ordered["resolution_method"].dropna()}
        as_reported_only = methods.issubset({"as_reported", "declared_total"}) and methods
        out.append(_finding_from(
            ordered.iloc[-1], sub, check_name="cross_vintage",
            severity=INFO if as_reported_only else HIGH, tier=3,
            observed=last, expected=first, deviation=_relative(last, first),
            detail={"classification": "restatement" if as_reported_only else "candidate defect",
                    "vintages": int(len(ordered)),
                    "first_filed": {"value": first,
                                    "filing_date": _date(ordered.iloc[0]["filing_date"]),
                                    "accession": str(ordered.iloc[0]["accession_number"])},
                    "last_filed": {"value": last,
                                   "filing_date": _date(ordered.iloc[-1]["filing_date"]),
                                   "accession": str(ordered.iloc[-1]["accession_number"])},
                    "resolution_methods": sorted(methods),
                    "duration_type": str(duration),
                    "why": ("every vintage is AS REPORTED, so the filer re-presented the "
                            "period -- both numbers are true and only the first was knowable "
                            "at the time"
                            if as_reported_only else
                            "at least one vintage was DERIVED, so our arithmetic is in play")}))
    return _collapse(out)


@check(name="restatement_ledger", tier=3, substrate=FACTS, severity=INFO, grain=GRAIN_SERIES,
       expected_fire_rate_ceiling=1.0)
def restatement_ledger(sub: Substrates) -> list[Finding]:
    """RECORD every restatement, per (ticker, field). NEVER repair one.

    This check exists so that nobody "fixes" our as-filed numbers toward a re-presented
    aggregate. BAC's FY2023 revenue is **98,581M as filed** and **102,769M re-presented**; both
    are true and only the first was knowable in early 2024, which is the only one a
    point-in-time model may use.

    `info`, permanently. A restatement is not a finding about our extraction; it is a fact
    about the filer, and the ledger is where it is written down.
    """
    out: list[Finding] = []
    facts = sub.facts
    if facts.empty:
        return out
    valued = facts[facts["value"].notna()]
    keys = ["ticker", "field", "duration_type", "period_end"]
    grouped = _multi_row_groups(valued, keys).groupby(keys, sort=False)

    per_series: dict[tuple[str, str], list[dict]] = {}
    for (ticker, field, _duration, period_end), rows in grouped:
        if len(rows) < 2:
            continue
        ordered = rows.sort_values("filing_date")
        first, last = float(ordered.iloc[0]["value"]), float(ordered.iloc[-1]["value"])
        if not _materially_different(last, first, VINTAGE_TOLERANCE):
            continue
        methods = {str(m) for m in ordered["resolution_method"].dropna()}
        if not (methods and methods.issubset({"as_reported", "declared_total"})):
            continue                       # a derived vintage is cross_vintage's, not ours
        per_series.setdefault((str(ticker), str(field)), []).append({
            "period_end": _date(period_end), "as_filed": first, "re_presented": last,
            "move_pct": round(_relative(last, first) * 100, 2)})

    sub.denominator("restatement_ledger", len(per_series) or 1)
    for (ticker, field), events in per_series.items():
        out.append(Finding(
            check_name="restatement_ledger", ticker=ticker, severity=INFO, tier=3,
            substrate=FACTS, field=field,
            period_key=period_key_for_range(min(e["period_end"] for e in events),
                                            max(e["period_end"] for e in events)),
            observed=float(len(events)),
            cik=sub.cik_for(ticker),
            detail={"restatements": events,
                    "why": "RECORDED, NEVER REPAIRED -- the as-filed number is the only one "
                           "that was knowable at the time, and it is the one a point-in-time "
                           "model may use"}))
    return out


# --------------------------------------------------------------------------- #
# the remaining three                                                          #
# --------------------------------------------------------------------------- #

@check(name="duplicate_fact", tier=3, substrate=FACTS, severity=HIGH, grain=GRAIN_CELL,
       expected_fire_rate_ceiling=0.01)
def duplicate_fact(sub: Substrates) -> list[Finding]:
    """ONE filing tagging the same (field, duration, period) twice, with DIFFERENT values.

    Not a cross-vintage disagreement -- the same accession, contradicting itself. ORCL is the
    case: $7,623M and $7,600M for one period in one filing. Whichever the resolver picked, it
    picked it by frame order.

    Structurally near-impossible on the stored table, because `period_end` is in the PK and the
    upsert collapses the pair -- which is the point. If this fires, the collapse chose silently
    and the loser is gone. The one shape that DOES survive is a nudged start date: the same
    window tagged twice with `period_start` a day apart, measured at 3 rows in 337,190.
    """
    out: list[Finding] = []
    facts = sub.facts
    if facts.empty:
        return out
    valued = facts[facts["value"].notna()]
    sub.denominator("duplicate_fact", len(valued))

    keys = ["ticker", "accession_number", "field", "duration_type", "period_end"]
    grouped = _multi_row_groups(valued, keys).groupby(keys, sort=False)
    for key, rows in grouped:
        values = {round(float(v), 6) for v in rows["value"]}
        if len(values) < 2:
            continue
        ticker, accession, field, _duration, period_end = key
        out.append(_finding_from(
            rows.iloc[0], sub, check_name="duplicate_fact", severity=HIGH, tier=3,
            observed=float(max(values)), expected=float(min(values)),
            deviation=_relative(max(values), min(values)),
            detail={"values": sorted(values), "accession": str(accession),
                    "period_starts": sorted({_date(p) for p in rows["period_start"]}),
                    "why": "one filing contradicting itself -- whichever value the resolver "
                           "kept, it kept it by frame order (ORCL: $7,623M vs $7,600M)"}))
    return out


@check(name="leaf_vs_total", tier=3, substrate=FACTS, severity=HIGH, grain=GRAIN_CELL,
       expected_fire_rate_ceiling=0.25)
def leaf_vs_total(sub: Substrates) -> list[Finding]:
    """A `statement_leaf_sum` in one vintage against a DECLARED TOTAL in another.

    Two different routes to the same number, taken by two different filings, compared. It is
    the check that says whether our roll-up arithmetic reproduces what the filer eventually
    declares outright -- and the answer is uncomfortable and worth publishing: 76.40% / 78.72%
    EXACT agreement over 89/94 comparable points on the v2 sweep.

    The high ceiling (25%) is honest rather than lax. A leaf sum and a declared total genuinely
    differ when the filer's linkbase omits a caption from the roll-up it nonetheless includes
    in the total, and that is a property of the FILING. The finding is the invitation to look;
    it is not a claim that the roll-up is wrong.
    """
    out: list[Finding] = []
    facts = sub.facts
    if facts.empty or "resolution_method" not in facts.columns:
        return out
    valued = facts[facts["value"].notna()]
    sub.denominator("leaf_vs_total", len(valued))

    keys = ["ticker", "field", "duration_type", "period_end"]
    grouped = _multi_row_groups(valued, keys).groupby(keys, sort=False)
    for (_ticker, _field, duration, _period_end), rows in grouped:
        methods = rows["resolution_method"].astype(str)
        leaves = rows[methods.str.contains("leaf", case=False, na=False)]
        totals = rows[methods.str.contains("total|as_reported", case=False, na=False)]
        if leaves.empty or totals.empty:
            continue
        leaf_value = float(leaves.sort_values("filing_date").iloc[-1]["value"])
        total_value = float(totals.sort_values("filing_date").iloc[-1]["value"])
        if not _materially_different(leaf_value, total_value, HOLDOUT_TOLERANCE):
            continue
        out.append(_finding_from(
            leaves.sort_values("filing_date").iloc[-1], sub, check_name="leaf_vs_total",
            severity=HIGH, tier=3, observed=leaf_value, expected=total_value,
            deviation=_relative(leaf_value, total_value),
            detail={"leaf_sum": leaf_value, "declared_total": total_value,
                    "duration_type": str(duration),
                    "total_accession": str(totals.sort_values("filing_date")
                                           .iloc[-1]["accession_number"]),
                    "tolerance": HOLDOUT_TOLERANCE,
                    "why": "two routes to one number, taken by two filings. A genuine gap is "
                           "a property of the FILER'S linkbase, not necessarily of ours"}))
    return _collapse(out)


@check(name="derived_vs_asreported", tier=3, substrate=FACTS, severity=HIGH, grain=GRAIN_CELL,
       expected_fire_rate_ceiling=0.05)
def derived_vs_asreported(sub: Substrates) -> list[Finding]:
    """Our per-share figures against the ones the filer tagged itself.

    `epsDiluted` and the weighted-average share counts are the two places where we compute a
    number the filer also publishes, so the comparison is direct and needs no reconciliation.
    97.3% of 710 points landed within 0.5% on the v2 sweep.

    A disagreement here is usually a SHARE-COUNT BASIS problem rather than an EPS one -- a
    share-day weighted count against the filer's own weighted average, or a split applied on
    one side and not the other (AAPL's derived FY2012 Q4 at 24.3bn shares is the 7:1 split,
    which is reason-coded `split_basis_mismatch` and is NOT this check's business).
    """
    out: list[Finding] = []
    facts = sub.facts
    if facts.empty:
        return out
    per_share = facts[facts["field"].isin({"epsDiluted", "dilutedShares", "basicShares"})
                      & facts["value"].notna()]
    if per_share.empty:
        return out
    sub.denominator("derived_vs_asreported", len(per_share))

    keys = ["ticker", "field", "duration_type", "period_end"]
    grouped = _multi_row_groups(per_share, keys).groupby(keys, sort=False)
    for _key, rows in grouped:
        methods = rows["resolution_method"].astype(str)
        derived = rows[methods.str.contains("derive|comput", case=False, na=False)]
        reported = rows[methods.str.contains("as_reported", case=False, na=False)]
        if derived.empty or reported.empty:
            continue
        ours = float(derived.sort_values("filing_date").iloc[-1]["value"])
        theirs = float(reported.sort_values("filing_date").iloc[-1]["value"])
        if abs(theirs) < 1e-9 or abs(ours - theirs) / abs(theirs) <= PER_SHARE_TOLERANCE:
            continue
        out.append(_finding_from(
            derived.sort_values("filing_date").iloc[-1], sub,
            check_name="derived_vs_asreported", severity=HIGH, tier=3,
            observed=ours, expected=theirs, deviation=_relative(ours, theirs),
            detail={"ours": ours, "filer": theirs, "tolerance": PER_SHARE_TOLERANCE,
                    "likely_mechanism": "a share-count BASIS difference (share-day weighting "
                                        "vs the filer's weighted average), or a split applied "
                                        "on one side only",
                    "why": "the one place we compute a number the filer also publishes, so "
                           "the comparison needs no reconciliation"}))
    return out


# --------------------------------------------------------------------------- #
# shared helpers                                                               #
# --------------------------------------------------------------------------- #

def _multi_row_groups(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """`frame` reduced to the rows whose key group has MORE THAN ONE row.

    A vectorised pre-filter, and on this table it is most of the runtime. `cross_vintage`,
    `restatement_ledger`, `duplicate_fact` and `leaf_vs_total` all ask a question that only a
    group of two or more can answer, and on 317,036 fact rows the overwhelming majority of
    groups are singletons -- a Python loop over ~250,000 groups to discard 95% of them costs
    minutes, while `duplicated(keep=False)` is one pass in C.
    """
    if frame.empty:
        return frame
    return frame[frame.duplicated(subset=keys, keep=False)]


def _latest_per_window(facts: pd.DataFrame, duration_type: str) -> pd.DataFrame:
    """The LATEST-filed valued row per (ticker, field, period_end) at one duration shape.

    Amendments and re-presentations collapse to one observation, which is what makes a footing
    comparison well-defined. `cross_vintage` and `restatement_ledger` deliberately do NOT use
    this -- disagreement across vintages is the thing they are looking at.
    """
    scoped = facts[(facts["duration_type"] == duration_type) & facts["value"].notna()]
    if scoped.empty:
        return scoped
    return (scoped.sort_values("filing_date")
            .drop_duplicates(subset=["ticker", "field", "period_end"], keep="last"))


def _by_series(frame: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    """`{(ticker, field): its rows, sorted by period_end}` -- built ONCE per run.

    NOT a convenience. The footing family walks ~31,000 annual rows on the live table, and the
    obvious implementation masks the ~63,000-row quarterly frame once per annual row: about
    2 billion element comparisons per check, times three checks that each need the lookup twice.
    Indexing once turns every one of those masks into a dict hit on a frame of ~60 rows.

    This is Phase 10's "the validator re-reads its data per check" risk in a second guise -- the
    substrate is loaded once, and then a check quietly rescans it per row.
    """
    if frame.empty:
        return {}
    return {key: rows.sort_values("period_end")
            for key, rows in frame.groupby(["ticker", "field"], sort=False)}


def _quarters_in(series: pd.DataFrame | None, annual: pd.Series) -> pd.DataFrame:
    """The tagged quarters lying inside one annual window, in period order.

    `series` is ONE (ticker, field)'s quarterly rows, already sorted -- see `_by_series`.

    Selected by CALENDAR WINDOW, never by `(fiscal_year, fiscal_period)`: the labels collide
    18,604 times in 337,190 rows, so a label join would silently pull a comparative year's
    quarter into this year's footing.
    """
    if series is None or series.empty:
        return pd.DataFrame(columns=["ticker", "field", "period_end", "value"])
    end = pd.Timestamp(annual["period_end"])
    start = pd.Timestamp(annual["period_start"]) if pd.notna(annual.get("period_start")) \
        else end - pd.Timedelta(days=366)
    return series[(series["period_end"] > start) & (series["period_end"] <= end)]


def _q4_for(series: pd.DataFrame | None, annual: pd.Series) -> float | None:
    """The filer's OWN discrete Q4 for an annual window: the quarter ending with the year."""
    if series is None or series.empty:
        return None
    match = series[series["period_end"] == pd.Timestamp(annual["period_end"])]
    return float(match.iloc[-1]["value"]) if len(match) else None


def _ytd9_end_for(series: pd.DataFrame | None, annual: pd.Series) -> pd.Timestamp:
    """The period end a YTD9 for this annual window would carry: the third quarter's end.

    Derived from the filer's OWN quarter grid rather than by subtracting 92 days, because a
    52/53-week filer's quarter ends walk and KR's Q1 is sixteen weeks long. The 92-day fallback
    is only for a filer with too few tagged quarters to have a grid.
    """
    window = _quarters_in(series, annual)
    if len(window) >= 2:
        return pd.Timestamp(window.iloc[-2]["period_end"])
    return pd.Timestamp(annual["period_end"]) - pd.Timedelta(days=92)


def _materially_different(observed: float, expected: float, tolerance: float) -> bool:
    """Is the gap real, at this tolerance and above the noise floor?

    Two guards, both needed. The relative test is what the tolerance means; the absolute floor
    stops a $3 difference on a $2 base being reported as a 150% error, which is arithmetically
    true and completely uninformative.
    """
    if not np.isfinite(observed) or not np.isfinite(expected):
        return False
    if max(abs(observed), abs(expected)) < MIN_COMPARABLE_MAGNITUDE:
        return False
    return abs(observed - expected) > tolerance * max(abs(expected), 1.0)


def _relative(observed: float, expected: float) -> float:
    """`(observed - expected) / |expected|`, or 0 when there is nothing to divide by."""
    base = abs(float(expected))
    return (float(observed) - float(expected)) / base if base else 0.0


def _date(value) -> str:
    """A date as `YYYY-MM-DD` -- Postgres DATE arrives as `datetime.date`, not `Timestamp`."""
    return "" if value is None or pd.isna(value) else str(pd.Timestamp(value).date())


def _collapse(findings: list[Finding]) -> list[Finding]:
    """`collapse_by_id` with THIS module's reason. See `finding.collapse_by_id`.

    Tier 3's collision is the duration shape: three checks here group on
    `(ticker, field, duration_type, period_end)` while `finding_id` stops at `period_end`.
    """
    return collapse_by_id(findings, why=(
        "this period end carries more than one duration shape; the worst is reported here "
        "and the rest are listed above, because the finding's grain is "
        "(ticker, field, period_end)"))


def _finding_from(row: pd.Series, sub: Substrates, *, check_name: str, severity: str,
                  tier: int, observed=None, expected=None, deviation=None,
                  detail: dict | None = None) -> Finding:
    """A `Finding` carrying one fact row's full provenance -- decision 47's packet."""
    return Finding(
        check_name=check_name, ticker=str(row["ticker"]), severity=severity, tier=tier,
        substrate=FACTS, field=str(row["field"]),
        period_key=_date(row["period_end"]), as_of=row.get("filing_date"),
        observed=_num(observed), expected=_num(expected), deviation=_num(deviation),
        source_concept=_str(row.get("source_concept")),
        resolution_method=_str(row.get("resolution_method")),
        roll_up_children=_str(row.get("roll_up_children")),
        root_anchor=_str(row.get("root_anchor")), role_uri=_str(row.get("role_uri")),
        accession_number=_str(row.get("accession_number")),
        cik=sub.cik_for(str(row["ticker"])), detail=detail or {})


def _num(value) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(out) else out


def _str(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value)
    return text if text and text.lower() not in ("nan", "none", "<na>") else None
