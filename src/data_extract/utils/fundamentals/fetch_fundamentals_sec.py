"""
fetch_fundamentals_sec.py (src/data_extract/utils/fundamentals/fetch_fundamentals_sec.py)
--------------------------------------------------------------------------------------------
Per-filing SEC XBRL walk -> `fundamentals_facts`. Replaces the 1,232-line
`fetch_fundamentals_edgar.py`, whose priority-ordered candidate-tag resolver this rebuild
exists to remove.

One row per catalogue FIELD per period per filing. **Strictly as-filed**: every value is a
number the filer actually tagged, on the period shape it tagged it with. Q4 = FY - YTD9 and
the YTD decumulation are Phase 4's job and happen in memory during the history build, so
this table stays a faithful record of what was published -- which is what makes the
publication-event grain and the no-leakage property of `fundamentals_history_sec` provable
rather than merely asserted.

Division of labour:
  * `xbrl_linkbase.py`  decides WHICH concept (or weighted set of concepts) is the field,
                        once per (filing, field), from the filer's own calculation linkbase.
  * `entity_scope.py`   decides WHICH facts belong to the consolidated registrant.
  * this module         turns those two answers into rows, per period.

Resume, never rescan: the shared `run_edgar_fetch` driver reads the stored accession set and
the extraction manifest's window, so a nightly run touches only genuinely new filings
(~5-8 universe-wide on a quiet night, ~20-80 at earnings peak). It also serializes the first
write to a cold table behind a lock -- `store.ensure_table` is a check-then-create with no
locking, and on a cold table concurrent workers otherwise race the CREATE and silently lose
whole tickers' rows.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from functools import partial

import pandas as pd

from src.constants.constants import FUNDAMENTALS_FORMS
from src.context import Context
from src.data_extract.utils.common.edgar_driver import (
    PROGRAMMING_ERRORS, new_filings, run_edgar_fetch,
)
from src.data_extract.utils.common.sec_utils import load_cik_mapping
from src.data_extract.utils.fundamentals import entity_scope as scope
from src.data_extract.utils.fundamentals.cik_cutover import (
    Cutover, cutover_filings, load_cutovers)
from src.data_extract.utils.fundamentals.fundamentals_employees import (
    employee_fact_frame, history_by_ticker, is_headcount_form)
from src.data_extract.utils.fundamentals.kpi_catalogue import Catalogue, load_catalogue
from src.data_extract.utils.fundamentals.periods import (
    AMBIGUOUS_DURATION, ANNUAL, OTHER_SHAPE, QUARTERLY, period_shape)
from src.data_extract.utils.fundamentals.reason_codes import (
    NOT_DISCLOSED, PERIOD_INTERSECTION_PARTIAL)
from src.data_extract.utils.fundamentals.xbrl_linkbase import (
    FIELD_SUM, INCOMPLETE_ROLL_UP, LINKBASE_SUM, NO_USABLE_PERIOD, STATEMENT_LEAF_SUM,
    UNRESOLVED, ArcGraph, Resolution, bare, calculation_arcs, resolve_field,
    segment_only_concepts, statement_arcs)
from src.data_store.schema import Table, Tables

logger = logging.getLogger(__name__)

_COLS = ["ticker", "accession_number", "field", "fiscal_year", "fiscal_period",
         "duration_type", "cik", "form", "filing_date", "is_amendment",
         "period_of_report", "regime", "period_start", "period_end", "period_days",
         "value", "unit", "decimals", "resolution_method", "source_concept",
         "roll_up_children", "root_anchor", "adjustment", "role_uri", "is_extension",
         "dc_code"]

#: Fiscal period recorded when the filer tagged none. No longer a PK column -- the key is the
#: calendar window now -- but kept, because an empty string and a NULL would sort and join as
#: two silent extra states where a named one reads as what it is.
UNLABELLED_PERIOD = "NA"


def _period_frame(facts: pd.DataFrame) -> pd.DataFrame:
    """Facts with the period columns normalised: one `duration_type`, one `period_days`,
    and `period_end` populated for instants (where edgartools leaves start/end NaT and
    carries the date in `period_instant` instead)."""
    out = facts.copy()
    # Guarantee the schema. edgartools does not always emit `fiscal_year`/`fiscal_period`
    # -- three KR filings ship neither -- and `_period_key` reads them off `itertuples`,
    # where a missing column is an AttributeError that kills the whole filing rather than
    # one field. Found by the 26-ticker sweep; the 6-ticker snapshot could not surface it.
    for column in ("fiscal_year", "fiscal_period", "unit_ref", "decimals"):
        if column not in out.columns:
            out[column] = None
    start = pd.to_datetime(out.get("period_start"), errors="coerce")
    end = pd.to_datetime(out.get("period_end"), errors="coerce")
    if "period_instant" in out.columns:
        instant = pd.to_datetime(out["period_instant"], errors="coerce")
        end = end.fillna(instant)
    out["period_start"] = start
    out["period_end"] = end
    out["period_days"] = (end - start).dt.days
    out["duration_type"] = [
        period_shape(str(pt), d)
        for pt, d in zip(out.get("period_type", ""), out["period_days"])]
    out["_bare"] = [scope.bare_concept(c) for c in out["concept"]]
    return out


def _period_key(row) -> tuple:
    return (row.fiscal_year, row.fiscal_period, row.duration_type,
            row.period_start, row.period_end)


#: `decimals` value meaning "exact" -- the finest precision an XBRL fact can declare.
_DECIMALS_EXACT = "INF"


def _precision(decimals) -> float:
    """`decimals` as a sortable precision, coarsest first. Higher is finer.

    XBRL's `decimals` counts digits to the RIGHT of the point, so it runs negative for
    figures reported in thousands (-3) or millions (-6) and `INF` means the fact is exact.
    An absent or unparseable value ranks lowest, so a fact that declares its precision
    always beats one that does not -- the honest ordering, since the whole point is to
    prefer the number that claims to be less rounded.
    """
    if decimals is None or (isinstance(decimals, float) and pd.isna(decimals)):
        return float("-inf")
    text = str(decimals).strip()
    if text.upper() == _DECIMALS_EXACT:
        return float("inf")
    try:
        return float(text)
    except ValueError:
        return float("-inf")


def _values_by_period(facts: pd.DataFrame, concept: str) -> dict[tuple, dict]:
    """concept -> {period key: the fact}. Namespaced match where the resolution carried a
    namespace (APA's `apa:RevenuesAndOther`), bare match otherwise.

    **The finer `decimals` wins a duplicate, not the later arc** (4c.3). One filing can tag
    the same (concept, period) twice at different precisions, and the old rule -- last one
    wins -- handed the choice to arc order: ORCL's FY2026 `Depreciation` arrives as both
    $7,623M and a rounded $7,600M, and taking the second is a **0.3% haircut** decided by
    nothing. The defect is route-independent, so it reaches every field.

    A disagreement is always RECORDED, on the surviving period as `duplicate_fact`, whether
    or not the tie-break changed the answer: two different numbers for one (concept, period)
    is a filer-side defect in its own right, and Phase 5b's `duplicate_fact` check needs the
    population rather than the repair. Identical re-tagging -- the common case, a
    re-presented comparative -- is not a disagreement and is not flagged.
    """
    column = "concept" if ":" in concept else "_bare"
    hits = facts[facts[column] == concept]
    out: dict[tuple, dict] = {}
    for row in hits.itertuples():
        key = _period_key(row)
        fact = {
            "value": float(row.numeric_value), "unit": getattr(row, "unit_ref", None),
            "decimals": getattr(row, "decimals", None),
            "fiscal_year": row.fiscal_year, "fiscal_period": row.fiscal_period,
            "duration_type": row.duration_type, "period_start": row.period_start,
            "period_end": row.period_end, "period_days": row.period_days,
        }
        prior = out.get(key)
        if prior is None:
            out[key] = fact
            continue
        winner, loser = ((fact, prior)
                         if _precision(fact["decimals"]) > _precision(prior["decimals"])
                         else (prior, fact))
        if winner["value"] != loser["value"]:
            seen = list(prior.get("duplicate_fact", []))
            seen.append({"concept": concept, "kept": winner["value"],
                         "kept_decimals": str(winner["decimals"]),
                         "dropped": loser["value"],
                         "dropped_decimals": str(loser["decimals"])})
            winner = {**winner, "duplicate_fact": seen}
        out[key] = winner
    return out


#: Forms whose STATEMENT periods are all ANNUAL. A quarterly-shaped fact in one of these did
#: not come off the face of a statement -- an annual report has no quarterly column -- so it
#: came from a note, and the notes publish quarters in exactly two shapes (`_lone_quarters`).
#: Spelled out and matched EXACTLY, like `FORM_PRECEDENCE` and `HEADCOUNT_FORMS`: only the four
#: forms in `FUNDAMENTALS_FORMS` are ever fetched, so a `startswith` prefix bought nothing and
#: read as though `10-KT`/`10-K405` had been considered and admitted, which they had not.
_ANNUAL_FORMS: tuple[str, ...] = ("10-K", "10-K/A")


_Window = tuple[tuple, pd.Timestamp, pd.Timestamp]


def _annual_windows(periods: dict[tuple, dict]) -> list[_Window]:
    """`(key, start, end)` for every ANNUAL period in the filing, coerced once.

    Hoisted out of `_covering_annual`, which runs once per quarter: every call used to re-walk
    the whole `periods` dict and rebuild the same `Timestamp`s, so an ASC 270 table of eight
    quarters re-parsed its handful of annual windows eight times over.
    """
    out = []
    for key, period in periods.items():
        if period.get("duration_type") != ANNUAL:
            continue
        low, high = pd.Timestamp(period["period_start"]), pd.Timestamp(period["period_end"])
        if pd.notna(low) and pd.notna(high):
            out.append((key, low, high))
    return out


def _covering_annual(windows: list[_Window], period: dict) -> tuple | None:
    """The key of the ANNUAL window in the same filing that CONTAINS `period`.

    The containment relation rather than the filer's `fiscal_year` label, because the label
    is per-fact and edgartools does not always populate it (three KR filings ship neither
    `fiscal_year` nor `fiscal_period`), while the windows are the PK and always present. It
    also settles a 52/53-week and a non-calendar issuer without a special case: ORCL's fourth
    quarter ends 2018-05-31 and so does its fiscal 2018, and containment simply holds.
    """
    start, end = pd.Timestamp(period["period_start"]), pd.Timestamp(period["period_end"])
    if pd.isna(start) or pd.isna(end):
        return None
    for key, low, high in windows:
        if low <= start and end <= high:
            return key
    return None


def _lone_quarters(periods: dict[tuple, dict],
                   filing_windows: list[_Window] | None = None) -> dict[tuple, tuple]:
    """`{key of a quarter that is the ONLY one of its fiscal year: key of that year}`.

    A 10-K carries quarterly contexts from a note, and there are only two notes that put a
    quarterly window on an income-statement concept:

      * the **ASC 270 / Item 302 quarterly financial data table**, which is a SERIES -- it
        publishes all four quarters of the year (usually of two years), so the concept lands
        with three siblings inside its own fiscal year; and
      * an **ASC 270-10-50-2 fourth-quarter-adjustment narrative**, which is a SENTENCE about
        a DISCRETE ITEM inside a quarter, and lands alone.

    So the count of same-year siblings separates a published quarter from a prose aside, with
    no list of concepts, no role-name matching and no second request. Boeing's fiscal 2011
    10-K (0001193125-12-048565) tags `us-gaap:Revenues` for all four quarters of 2010 AND
    2011 off the table, and `us-gaap:IncomeTaxExpenseBenefit` for Q4 of each year ALONE, off
    the sentence "during the fourth quarters of 2011 and 2010, we recorded tax benefits of
    $397 and $371 as a result of settling the 2004-2006 and 1998-2003 federal audits" -- and
    that filing's quarterly data table has no income tax row at all. The same context also
    carries `us-gaap:TaxAdjustmentsSettlementsAndUnusualProvisions` at -$397M, the same
    magnitude signed as what it is.

    **The fiscal calendar is the FILING's, not the field's own.** `filing_windows` is the
    union of every annual window the filing declares across ALL fields, and it is consulted
    only where THIS field declares none -- exactly the case the field-local rule could not
    judge, so it kept the row. ORCL's fiscal 2020-2022 10-Ks tag the full-year
    `us-gaap:Revenues` into a 91-day fourth-quarter context and publish no annual-window
    `Revenues` at all, so `_annual_windows` came back empty and the guard returned before
    reading a single quarter: **9 rows across 3 filings**, fiscal 2022 Q4 stored at $42,440M
    against a true $11,840M. The year is not in doubt and no inference is needed to date it
    -- the same filings carry the correct annual figure under
    `RevenueFromContractWithCustomerExcludingAssessedTax` on 2021-06-01..2022-05-31, the
    same $42,440M -- only the context the filer hung it on is wrong.

    Scoped to fields with no annual window of their own rather than unioned unconditionally,
    because the two differ and the difference is unread: replayed over the 54-ticker table
    the fallback drops **9 rows, all ORCL**, while an unconditional union drops **16 across
    7 (ticker, field) pairs** -- 7 further rows on DTE, EQIX, META and VLO that have the
    same prose-aside shape but no filing-level evidence behind them yet.

    A quarter with NO covering annual fact anywhere in the filing is still not judged and is
    kept: silence is not evidence, the same rule `xbrl_linkbase.is_note_only` and D1's
    condition 1 apply.
    """
    windows = _annual_windows(periods) or filing_windows or []
    if not windows:
        return {}
    years: dict[tuple, list[tuple]] = {}
    for key, period in periods.items():
        if period.get("duration_type") != QUARTERLY:
            continue
        year = _covering_annual(windows, period)
        if year is not None:
            years.setdefault(year, []).append(key)
    return {keys[0]: year for year, keys in years.items() if len(keys) == 1}


def _filing_annual_windows(values: dict[str, dict[tuple, dict]]) -> list[_Window]:
    """Every ANNUAL window the filing declares, over all fields, deduplicated by span.

    One filing states one fiscal calendar, so a window is worth keeping once however many
    fields tag it. Deduplicated on `(start, end)` rather than on the period key, which
    carries the field and would therefore repeat the same year ~50 times on a full
    catalogue and make `_covering_annual`'s scan that much longer for no extra evidence.
    """
    by_span: dict[tuple, _Window] = {}
    for periods in values.values():
        for key, low, high in _annual_windows(periods):
            by_span.setdefault((low, high), (key, low, high))
    return list(by_span.values())


def _drop_note_only_quarter(periods: dict[tuple, dict], *, form: str,
                            filing_windows: list[_Window] | None = None,
                            ) -> dict[tuple, dict]:
    """Refuse a quarterly fact an ANNUAL report published ALONE for its fiscal year.

    The value is a discrete item disclosed in prose, never the quarter's total, and storing
    it makes `fundamentals_facts` assert an as-filed quarter the filer never stated. It then
    does two further kinds of damage downstream, because `periods.py` ranks an `AS_REPORTED`
    quarter above every derived one and keeps the LATEST filing per window: the note quarter
    both DISPLACES the 10-Q's own face-statement quarter and PRE-EMPTS the `FY - YTD9`
    ladder, then propagates into four TTM windows.

    Measured over the 54-ticker table before the fix: **19 rows**, and BA `incomeTaxExpense`
    is 2 of them. Q4 2010 was stored at $371M against a true $1,196M - $1,359M = **-$163M**
    and Q4 2011 at $397M against $1,382M - $1,325M = **+$57M** -- both signs wrong, because
    a settlement BENEFIT was tagged with the expense element. Only 11 of the 19 are the sole
    source of their period and all 11 are provably wrong; the other 8 are exact duplicates
    of the same period from a 10-Q, so no window loses its number.

    **`filing_windows` -- the fiscal calendar is the filing's (cluster `2603621e89ab`).**
    That 11 was first written down as "BA 2, ORCL 9", and the ORCL half was never true: the
    rule dates a quarter by a covering annual window of THE SAME FIELD, and ORCL's fiscal
    2020-2022 10-Ks publish no annual-window `us-gaap:Revenues` at all, so `_lone_quarters`
    hit `if not windows: return {}` and judged nothing. The 9 rows survived the fix that
    claimed them and went on carrying a full year in a 91-day context -- 47 findings across
    7 checks. Passing the filing's own annual windows in as a fallback dates them without
    inference. Measured over the same table, the fallback drops **exactly those 9 rows and
    nothing else**.

    The form gate is load-bearing, not a nicety. A 10-Q's face statement carries exactly one
    quarterly context per fiscal year -- the current quarter, plus the prior-year comparative
    in its own year -- so every quarter in a 10-Q is "lone" and an ungated rule would delete
    the entire quarterly grain.

    This is a DIFFERENT mechanism from `periods._drop_annual_masquerading_as_quarter` and
    neither subsumes the other: D1/D1b test the quarter's value against the filer's own
    annual (agreement within 0.1%) and so cannot see $397M beside $1,382M, while this tests
    the note's SHAPE and cannot see a full-year number tagged into a quarterly context that
    the table publishes alongside its three siblings. The two overlap on ORCL and the layer
    is what separates them: D1b already refused those rows in `periods.py`, which is why
    `fundamentals_history_sec` never showed $42,440M as a quarter, but it runs on the way to
    HISTORY and leaves `fundamentals_facts` -- the substrate every Tier-2/3 check reads --
    still asserting the bad quarter. Refusing at the facts layer is what closes the cluster.
    """
    if str(form or "").upper() not in _ANNUAL_FORMS:
        return periods
    lone = _lone_quarters(periods, filing_windows)
    if not lone:
        return periods
    out = {key: period for key, period in periods.items() if key not in lone}
    for key, year in lone.items():
        if year not in out:
            continue
        dropped, host = periods[key], out[year]
        rejected = list(host.get("note_quarter_rejected", []))
        rejected.append({"period_end": str(pd.Timestamp(dropped["period_end"]).date()),
                         "value": dropped["value"]})
        out[year] = {**host, "note_quarter_rejected": rejected}
    return out


def _retry_without(name, resolution, catalogue, graph, available, regime, facts,
                   durations, zero_only, magnitudes, ticker, prefer_structure,
                   form: str, filing_windows: list[_Window],
                   ) -> tuple[Resolution, dict[tuple, dict], dict[tuple, dict]] | None:
    """Re-resolve `name` with the concept that yielded NOTHING withheld, or None.

    A concept every one of whose periods was refused did not resolve the field -- it only
    looked like it did, because `resolve_field` is period-agnostic by design and so ranks a
    tag on whether the filer USES it, not on whether what it says is usable. The catalogue
    already ranks a second answer; nothing was ever asking for it.

    ORCL is the measured case and the whole reason this exists. `totalRevenue` lists
    `fallback_concepts: ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
    ...]`. `us-gaap:Revenues` wins in the fiscal 2020-2022 10-Ks because it is present -- but
    every period it offers is a full year stamped into a 91-day context, so
    `_drop_note_only_quarter` refuses all three and the field resolves to nothing. The ASC 606
    element sits in the SAME filings on proper 364/365-day windows carrying the same figures,
    and is the very next candidate. Without the retry, fiscal 2020's $39,068M annual is in no
    filing we store, and `Q4 = FY - YTD9` cannot run for three years: the point-in-time Q4 at
    `as_of` 2020-06-22, 2021-06-21 and 2022-06-21 silently carried the PRIOR quarter instead
    (9,796 / 10,085 / 10,513 $M against ~10,440 / ~11,259 / 11,840).

    Withheld via `available` rather than a new resolver argument: that set is the filing's
    reported concepts keyed BARE, it is already the lever `resolve_field` reads to decide what
    a filing offers, and removing a name from it is exactly "pretend the filer never tagged
    this". No change to the resolver, and the retry runs the same two-pass zero guard.

    Returns None -- leaving the caller to record the refusal -- when the field has no second
    candidate, when the retry lands on the same concept, or when the retry's periods are
    refused in their turn. A retry that resolves to nothing is not an improvement over an
    honest `ambiguous_duration` stub.

    Deliberately NOT a loop over every remaining candidate. One retry covers the measured
    population (4 value-less stubs table-wide: ORCL x3, JPM x1) and a loop would need its own
    termination story; if a filer ever needs two, the finding will say so.
    """
    dead = resolution.concept
    if not dead:
        return None
    retry = resolve_field(catalogue.field(name), graph, available - {bare(dead)},
                          catalogue, regime, duration_concepts=durations,
                          zero_only=zero_only, magnitudes=magnitudes,
                          ticker=ticker, prefer_structure=prefer_structure)
    if not retry.resolved or retry.concept == dead:
        return None
    periods, refused = _materialise(retry, facts)
    kept = _drop_note_only_quarter(periods, form=form, filing_windows=filing_windows)
    return (retry, kept, refused) if kept else None


def _materialise(resolution: Resolution,
                 facts: pd.DataFrame) -> tuple[dict[tuple, dict], dict[tuple, dict]]:
    """Turn one field's resolution into `({period key: value + provenance}, {refused})`.

    A `linkbase_sum` emits a period ONLY where every leg is reported for that same period.
    A partial sum is not the total: dropping a leg is precisely the `shortTermDebt` defect
    this rebuild removes (the discarded leg was the LARGER one in 54.4% of the 2,017 cells
    that tag both legs with no total).

    `statement_leaf_sum` takes the SAME intersection, over the leaves route 3b actually
    chose for this filing rather than over a fixed list. The choice matters: a filer that
    tags its capex legs on the annual window but only one of them on the ytd9 window would
    otherwise emit a ytd9 that is short by a whole leg -- and short by a DIFFERENT amount
    each quarter, which is worse than absent because it survives every level check and
    corrupts only the growth rate. The cost is a dropped period, which
    `insufficient_quarters` already reports honestly.

    Keeping the intersection STRICT survives this change; what changes (register item 9 /
    B.6.6) is that the periods it drops are now RETURNED rather than discarded. Until
    Phase 5 a partial intersection produced no row and no code, and `rows_from_xbrl` only
    reason-coded a field with no periods AT ALL -- so a field that resolved for fourteen
    windows and was refused for the fifteenth said nothing about the fifteenth. Measured:
    **128 rows** across EQIX `capex` (40), EQIX `depAmort` (40), SCHW `cash` (34), NEE
    `ppeNet` (8) and VRT `depAmort` (6). The plan's estimate was 31.

    The SCHW case is the one a null-gate could never have caught: `cash` is an INSTANT
    field, so a dropped period does not leave a null in the history snapshot -- it leaves
    the previous balance carried forward, which is correct behaviour under the snapshot
    contract and indistinguishable from a genuinely unchanged balance. The refusal has to
    be recorded where it happens or it is not observable anywhere at all.
    """
    if resolution.method in (LINKBASE_SUM, STATEMENT_LEAF_SUM):
        legs = {c: _values_by_period(facts, c) for c, _ in resolution.children}
        every = set().union(*(set(v) for v in legs.values())) if legs else set()
        shared = set.intersection(*(set(v) for v in legs.values())) if legs else set()
        refused = {key: _refused_period(legs, key) for key in sorted(every - shared)}
        out = {}
        for key in shared:
            total = sum(legs[c][key]["value"] * w for c, w in resolution.children)
            base = legs[resolution.children[0][0]][key]
            # Union the duplicate ledger across the legs, not just the first one: a summed
            # field with a duplicate in its SECOND leg is exactly as affected, and taking
            # `base`'s copy alone would silently lose it.
            duplicates = [d for c, _ in resolution.children
                          for d in legs[c][key].get("duplicate_fact", [])]
            out[key] = {**base, "value": total}
            if duplicates:
                out[key]["duplicate_fact"] = duplicates
            else:
                out[key].pop("duplicate_fact", None)
    elif resolution.concept:
        out, refused = _values_by_period(facts, resolution.concept), {}
    else:
        return {}, {}

    for concept in resolution.subtract:
        for key, adjustment in _values_by_period(facts, concept).items():
            if key in out:
                out[key] = {**out[key], "value": out[key]["value"] - adjustment["value"]}
    return out, refused


def _refused_period(legs: dict[str, dict[tuple, dict]], key: tuple) -> dict:
    """A value-less period stub for a window the strict intersection refused.

    Every PK column is already inside the period key, so the stub needs no second pass over
    the facts frame; `period_days` and `unit` come from whichever leg DID report the window,
    since all of them describe the same one.
    """
    reported = next(legs[c][key] for c in legs if key in legs[c])
    return {"fiscal_year": reported["fiscal_year"],
            "fiscal_period": reported["fiscal_period"],
            "duration_type": reported["duration_type"],
            "period_start": reported["period_start"],
            "period_end": reported["period_end"],
            "period_days": reported["period_days"],
            "value": None, "unit": reported.get("unit"), "decimals": None}


def _compose(spec, component_fields: tuple[str, ...],
             resolved: dict[str, dict[tuple, dict]],
             ) -> tuple[dict[tuple, dict], str | None]:
    """Sum a field composed of OTHER CATALOGUE FIELDS (`totalDebt` from its four debt/lease
    legs, `ppeNet` from gross less accumulated depreciation).

    Missing components still count as zero -- `totalDebt` must not vanish because a filer
    has no finance leases -- but **only for components the catalogue does not mark as
    load-bearing**. Zero-filling indiscriminately is how a composed field turns into a
    confident wrong number, and both composed fields were doing it:

      * **`totalDebt`** reported a LEASE LIABILITY as total debt on **213 of 2,655
        in-sample rows (8.0%)** and 29 of 3,096 out of sample, whenever neither debt leg
        resolved: BRK-B $4.9-6.3bn, GS $2.1-2.4bn, META $7.6-16.7bn. PGR traces the whole
        failure -- correct at $1.9-2.7bn through 2016, NULL for 2017-18, then $179-211M from
        2019 once `longTermDebt` stops resolving and the sum quietly becomes the operating
        lease liability. `roll_up.require_any` now demands at least one debt leg.
      * **`ppeNet`** emitted `accumulatedDepreciation` ALONE as net PP&E on 86 GS rows and
        2 DTE rows ($7.7-40.8bn), and `ppeGross` alone on 8 PG rows. Net PP&E is a
        difference, so `roll_up.require_all` demands both.

    The requirement is tested PER PERIOD, so a filing keeps the periods it can support.
    Returns `(values, dc_code)`; the code is set whenever the field yields nothing at all,
    so a null always travels with its reason -- and the two reasons are NOT the same:

      * `incomplete_roll_up` -- some component resolved and a load-bearing one did not.
        The filer reported something; we cannot make the field out of it without inventing
        the missing part.
      * `not_disclosed` -- NO component resolved anywhere in the filing.

    Returning None in the second case (which this did until Phase 4b) leaves an
    UNEXPLAINED null, and the plan's "zero unexplained nulls" criterion is checkable only
    if it is literally zero. Measured on the Phase-4 ledgers: **79 rows per 7 tickers**,
    all `ppeNet` and `totalDebt`, i.e. ~1.5% of both fields' rows.
    """
    roll_up = spec.raw.get("roll_up") or {}
    require_all = bool(roll_up.get("require_all"))
    require_any = [n for n in roll_up.get("require_any", [])]

    keys: set[tuple] = set()
    for name in component_fields:
        keys |= set(resolved.get(name, {}))
    out = {}
    for key in sorted(keys):
        present = [name for name in component_fields if key in resolved.get(name, {})]
        if require_all and len(present) < len(component_fields):
            continue
        if require_any and not any(name in present for name in require_any):
            continue
        parts = [resolved[name][key] for name in present]
        out[key] = {**parts[0], "value": sum(p["value"] for p in parts)}
        duplicates = [d for part in parts for d in part.get("duplicate_fact", [])]
        if duplicates:
            out[key]["duplicate_fact"] = duplicates
        else:
            out[key].pop("duplicate_fact", None)
    if not out:
        return {}, (INCOMPLETE_ROLL_UP if keys else NOT_DISCLOSED)
    return out, None


def _adjustment_json(resolution: Resolution, period: dict | None = None) -> str | None:
    """The free-form provenance blob: what was subtracted, which candidates the
    statement-role test withheld, and whether the value survives only because a guard was
    relaxed.

    Every key rides here rather than in its own column because `fundamentals_facts` is a
    named risk zone and this needs no schema change to stay auditable --
    `adjustment::jsonb ? 'undeclared_rejected'` finds every row 4c.1 actually reordered,
    `? 'role_rejected'` every row its note-role half withheld a candidate on,
    `? 'segment_rejected'` every row where a concept was withheld because the filer declares
    it ONLY on a segment-information role -- the one withholding with no relaxation behind
    it, so on an unresolved row it is also the `dc_code`,
    `? 'zero_only_retained'` every row the zero guard did, `? 'basis_qualifier'` every row
    that answered on a concept the catalogue declares non-comparable (`basis_ex_iprd`), and
    `? 'duplicate_fact'` every (concept, period) this filer tagged twice at two values,
    and `? 'sibling_rejected'` every row where route 1 declined the catalogue's total
    because the filer declared it BESIDE one of this field's roll-up legs rather than above
    it -- each entry a `[total, leg]` pair, so "which filers mistag the superset element,
    and as what?" is one query rather than an archaeology exercise.

    `duplicate_fact` is the one PERIOD-level key: resolution is period-agnostic by design,
    but a duplicate is a property of one fact, so it arrives on the materialised period
    rather than on the `Resolution`.
    """
    blob: dict = {}
    if resolution.subtract:
        blob["subtract"] = list(resolution.subtract)
    if resolution.zero_only_retained:
        blob["zero_only_retained"] = True
    if resolution.role_rejected:
        blob["role_rejected"] = list(resolution.role_rejected)
    if resolution.role_only_retained:
        blob["role_only_retained"] = True
    if resolution.segment_rejected:
        blob["segment_rejected"] = list(resolution.segment_rejected)
    if resolution.undeclared_rejected:
        blob["undeclared_rejected"] = list(resolution.undeclared_rejected)
    if resolution.sibling_rejected:
        blob["sibling_rejected"] = [list(pair) for pair in resolution.sibling_rejected]
    if resolution.basis_qualifier:
        blob["basis_qualifier"] = resolution.basis_qualifier
    if period and period.get("note_quarter_rejected"):
        blob["note_quarter_rejected"] = period["note_quarter_rejected"]
    if period and period.get("duplicate_fact"):
        blob["duplicate_fact"] = period["duplicate_fact"]
    return json.dumps(blob) if blob else None


def _period_end(period: dict | None, stamp: "_FilingStamp") -> pd.Timestamp:
    """The row's `period_end`, guaranteed non-NULL because it is part of the PK.

    Falls back through the filing's `period_of_report` to its filing date. Both fallbacks are
    only ever reached by a row that carries no value -- a reason-coded absence, or the handful
    of duration facts (10 in 109,267) whose window is unreadable -- so a fallback can never
    displace a real measurement.
    """
    if period is not None and pd.notna(period.get("period_end")):
        return pd.Timestamp(period["period_end"])
    return stamp.reported if pd.notna(stamp.reported) else stamp.filed


@dataclass(frozen=True)
class _FilingStamp:
    """The five filing-level values every row of a filing repeats.

    Read once per filing rather than once per row, and there are hundreds of rows a filing.
    `period_of_report` is the one that mattered: it is a plain edgartools `@property` that
    goes back through `Filing.sgml()`, so asking each row for it re-derived the whole
    submission header.
    """

    accession_number: str
    form: str
    filed: pd.Timestamp
    reported: pd.Timestamp
    is_amendment: bool

    @classmethod
    def of(cls, filing) -> "_FilingStamp":
        return cls(
            accession_number=filing.accession_number,
            form=filing.form,
            filed=pd.Timestamp(filing.filing_date),
            reported=pd.to_datetime(getattr(filing, "period_of_report", None),
                                    errors="coerce"),
            is_amendment=str(filing.form).upper().endswith("/A"))


def _row(ticker: str, cik: str, stamp: _FilingStamp, regime: str | None, field: str,
         resolution: Resolution, period: dict | None, *,
         dc_code: str | None = None) -> dict:
    """One `fundamentals_facts` row.

    `dc_code` overrides the resolution's own, for the one case where they differ: a period
    the strict intersection refused on a field that resolved perfectly well elsewhere in the
    same filing. The resolution has no code (it resolved); the PERIOD does.
    """
    children = ([[c, w] for c, w in resolution.children] if resolution.children
                else None)
    return {
        "ticker": ticker, "cik": cik,
        "accession_number": stamp.accession_number, "field": field,
        "fiscal_year": int(period["fiscal_year"]) if period and pd.notna(
            period.get("fiscal_year")) else stamp.filed.year,
        "fiscal_period": (str(period["fiscal_period"]) if period and pd.notna(
            period.get("fiscal_period")) else UNLABELLED_PERIOD),
        "duration_type": period["duration_type"] if period else OTHER_SHAPE,
        "form": stamp.form,
        "filing_date": stamp.filed,
        "is_amendment": stamp.is_amendment,
        "period_of_report": stamp.reported,
        "regime": regime,
        "period_start": period["period_start"] if period else pd.NaT,
        # `period_end` is a PK column, so it cannot be NULL -- and a REASON-CODED row has no
        # period of its own by definition. It falls back to the filing's own period of report,
        # which is the honest reading ("this field was absent as of the period this filing
        # covers") and cannot collide: a field with any usable period emits no such row, and
        # the key already contains `field`.
        "period_end": _period_end(period, stamp),
        "period_days": period["period_days"] if period else None,
        "value": period["value"] if period else None,
        "unit": period.get("unit") if period else None,
        # `str(NaN)` is the string "nan", which joins and compares as a real value.
        "decimals": (str(period["decimals"])
                     if period and period.get("decimals") is not None
                     and pd.notna(period.get("decimals")) else None),
        "resolution_method": resolution.method,
        "source_concept": resolution.source_concept,
        "roll_up_children": json.dumps(children) if children else None,
        "root_anchor": resolution.anchor,
        "adjustment": _adjustment_json(resolution, period),
        "role_uri": resolution.role_uri,
        "is_extension": resolution.is_extension,
        "dc_code": dc_code or resolution.dc_code,
    }


def filing_rows(ticker: str, cik: str, filing, catalogue: Catalogue,
                gics: dict[str, str | None] | None, *,
                failures: list[tuple[str, str]] | None = None) -> list[dict]:
    """Every catalogue field, for every period, from one filing.

    Returns [] rather than raising on an unreadable filing: one bad filing must not abort a
    490-ticker walk, and its absence is visible as a gap in the accession set. It is also
    APPENDED to `failures` as `(accession, error)` when the caller supplies a list, so the
    gap is counted and logged rather than inferred later from a hole in the accessions.

    The two `except`s are deliberately different, and the split is the whole point:

      * `filing.xbrl()` is edgartools parsing the filer's XBRL. Anything at all can come out
        of a malformed submission, so that one swallows everything -- absorbing unreadable
        filings is what it exists for.
      * `rows_from_xbrl` is OUR resolver. `PROGRAMMING_ERRORS` out of it is a defect in this
        repo and is re-raised; only a data failure is swallowed and counted.
    """
    try:
        xbrl = filing.xbrl()
    except Exception as exc:                # noqa: BLE001 -- the filer's XBRL, not our code
        _note_failure(failures, filing, exc)
        return []
    if xbrl is None:
        return []
    try:
        return rows_from_xbrl(ticker, cik, filing, xbrl, catalogue, gics)
    except PROGRAMMING_ERRORS:
        raise                                           # our bug, not the filer's
    except Exception as exc:                            # noqa: BLE001 -- one bad filing
        _note_failure(failures, filing, exc)
        return []


def _note_failure(failures: list[tuple[str, str]] | None, filing, exc: Exception) -> None:
    """Record one unreadable filing. `accession_number` is read defensively because a filing
    object broken enough to fail the parse may not answer for its own accession either."""
    if failures is None:
        return
    failures.append((str(getattr(filing, "accession_number", "unknown")), str(exc)))


def rows_from_xbrl(ticker: str, cik: str, filing, xbrl, catalogue: Catalogue,
                   gics: dict[str, str | None] | None, *,
                   prefer_structure: bool = True) -> list[dict]:
    """`filing_rows` with the parsed XBRL handed in.

    Split out because `filing.xbrl()` is the pipeline's whole cost (1.4-5.8 s against
    `calculation_linkbase()`'s 0.003-0.006 s), so any audit that needs the same filing read
    under two resolution settings -- which 4c.1's before/after acceptance does, on 3,200
    filings -- must be able to pay for the parse once. `prefer_structure` is documented on
    `resolve_field`; production never passes False.
    """
    facts = scope.consolidated_facts(xbrl.facts.to_dataframe())
    if facts.empty:
        return []
    facts = _period_frame(facts)
    available = scope.reported_concepts(facts)
    # Two filing-level properties the resolver cannot derive from structure alone: which
    # concepts are FLOWS (so a balance-sheet total cannot pose as a revenue root) and which
    # are zero in every period they report (so a tagging artefact does not win, and a real
    # zero is not thrown away). Both are computed once per filing, like the graph itself.
    durations = scope.duration_concepts(facts)
    zero_only = scope.zero_only_concepts(facts)
    # Peak |value| per concept. Route 1 needs it to see a filer reporting its declared
    # "total" SMALLER than a component FASB puts inside it -- MCD's
    # `PaymentsToAcquireProductiveAssets` is $540.9M of restaurant acquisitions beside a
    # $2,393.7M capex line. Filing-level like the two above, so resolution stays
    # period-agnostic. See `xbrl_linkbase.sibling_leg`.
    magnitudes = scope.peak_magnitudes(facts)
    stamp = _FilingStamp.of(filing)
    # ONE `calculation_linkbase()` read, two views of it -- see `statement_arcs`.
    arcs = calculation_arcs(xbrl)
    graph = ArcGraph(statement_arcs(xbrl, arcs))
    # Read off the UNFILTERED linkbase, because `statement_arcs` has already dropped every
    # segment-note arc by the time the graph exists -- which is precisely why the graph's own
    # `is_note_only` cannot see this population. See `xbrl_linkbase.SEGMENT_ROLE`.
    segment_only = segment_only_concepts(arcs)
    regime = catalogue.regime_for(
        gics, [str(r) for r in graph.arcs.get("role_uri", pd.Series(dtype=str))])

    # Resolve every concept-backed field first; the composed ones (`totalDebt`, `ppeNet`)
    # then read those results rather than the facts.
    resolutions: dict[str, Resolution] = {}
    values: dict[str, dict[tuple, dict]] = {}
    #: field -> the periods route 3b's strict intersection refused (B.6.6). Kept separate
    #: from `values` so a composed field cannot accidentally sum a refused stub.
    refused: dict[str, dict[tuple, dict]] = {}
    for name in catalogue.extracted_fields:
        resolution = resolve_field(catalogue.field(name), graph, available,
                                   catalogue, regime, duration_concepts=durations,
                                   zero_only=zero_only, magnitudes=magnitudes,
                                   ticker=ticker, prefer_structure=prefer_structure,
                                   segment_only=segment_only)
        resolutions[name] = resolution
        if resolution.method != FIELD_SUM:
            values[name], refused[name] = _materialise(resolution, facts)
    # Before `_compose` reads these, so a composed field inherits the cleaned legs, and
    # after the loop rather than inside it: a lone quarter is dated against the FILING's
    # fiscal calendar, so every field has to be materialised before the first one is judged.
    # The gate here only skips the union scan on a quarterly report; the RULE it encodes
    # lives in `_drop_note_only_quarter`, which re-checks the form itself.
    #
    #: Fields the note guard emptied OUTRIGHT. They reach the stub below with a resolved
    #: concept and no period, which is `NO_USABLE_PERIOD`'s shape but not its meaning -- that
    #: code says `_materialise` FOUND none, and here we found some and refused them. ORCL is
    #: the whole population: `us-gaap:Revenues` resolves in three 10-Ks and every period it
    #: offers is a mislabelled year, so without this the only trace of the refusal would be a
    #: code that misdescribes it. The `note_quarter_rejected` marker cannot carry this and it
    #: is worth saying so, because it is the first thing a reader will reach for: that marker
    #: lands on the covering annual OF THE SAME FIELD, and having none is the whole premise.
    note_refused: set[str] = set()
    form = str(filing.form or "").upper()
    if form in _ANNUAL_FORMS:
        filing_windows = _filing_annual_windows(values)
        for name, periods in list(values.items()):
            kept = _drop_note_only_quarter(periods, form=form,
                                           filing_windows=filing_windows)
            if periods and not kept:
                retry = _retry_without(name, resolutions[name], catalogue, graph, available,
                                       regime, facts, durations, zero_only, magnitudes,
                                       ticker, prefer_structure, form, filing_windows)
                if retry is not None:
                    resolutions[name], values[name], refused[name] = retry
                    continue
                note_refused.add(name)
            values[name] = kept
    for name, resolution in list(resolutions.items()):
        if resolution.method == FIELD_SUM:
            composed, reason = _compose(catalogue.field(name),
                                        resolution.component_fields, values)
            values[name] = composed
            if reason:
                resolutions[name] = replace(resolution, method=UNRESOLVED, dc_code=reason)

    rows: list[dict] = []
    for name, resolution in resolutions.items():
        periods = values.get(name) or {}
        if not periods:
            # No value anywhere in this filing: emit ONE reason-coded row rather than
            # nothing, so a downstream null is always explained. This is what makes
            # "zero unexplained nulls" checkable instead of aspirational.
            #
            # A RESOLVED field reaching here has no `dc_code` of its own, and that is the
            # last hole in the criterion: the concept was picked, so nothing upstream calls
            # it absent, yet `_materialise` found no period for it. Measured on the
            # in-sample ledger: **1 row of 144,131** -- JPM's 2011 10-K `pretaxIncome`,
            # where `reported_concepts` matched `IncomeLossFromContinuingOperationsBefore
            # IncomeTaxesExtraordinaryItemsNoncontrollingInterest` BARE while
            # `_values_by_period` then matched it NAMESPACED and the filing's namespace was
            # not `us-gaap`. Coded rather than repaired: changing the matching is the risky
            # half (`bare()`'s own docstring records the multi-class share-count defect that
            # lives in exactly that code path), and a named code makes any future instance
            # of the class visible instead of silent.
            if resolution.resolved:
                resolution = replace(resolution, method=UNRESOLVED,
                                     dc_code=(AMBIGUOUS_DURATION if name in note_refused
                                              else NO_USABLE_PERIOD))
            rows.append(_row(ticker, cik, stamp, regime, name, resolution, None))
            continue
        rows.extend(_row(ticker, cik, stamp, regime, name, resolution, period)
                    for period in periods.values())
    # The periods route 3b refused, each as a value-less row carrying its own code. Emitted
    # for EVERY field, including the ones that resolved -- that is the whole of B.6.6.
    for name, periods in refused.items():
        # Disjoint by construction -- `refused` is `union - intersection` and `values` is the
        # intersection -- but asserted, because a key in both would write the same PK twice
        # and the dedup in `build_ticker_fundamentals` would silently keep the value-less one.
        assert not (set(periods) & set(values.get(name, {}))), (
            f"{ticker} {filing.accession_number} {name}: a refused period is also resolved")
        rows.extend(_row(ticker, cik, stamp, regime, name, resolutions[name], period,
                         dc_code=PERIOD_INTERSECTION_PARTIAL)
                    for period in periods.values())
    return rows


def build_ticker_fundamentals(ticker: str, cik: str, *, since: pd.Timestamp | None = None,
                              done_accessions: frozenset[str] = frozenset(),
                              catalogue: Catalogue, gics_by_ticker: dict[str, dict],
                              cutovers: dict[str, Cutover] | None = None,
                              headcounts: dict[str, list[int]] | None = None,
                              ) -> dict[Table, pd.DataFrame]:
    """One ticker's facts, walking BOTH registrants where it re-registered.

    `Company(ticker)` sees only the current registrant, so without the cutover register APA
    loses 2011-02 to 2021-05 and GOOGL 2011-2015 -- silently, with no error and no gap. The
    walk is DATED, never a union: Apache Corp kept filing its own 10-K/10-Q through
    2024-11-07 as a subsidiary, so a union would duplicate ~15 filings and blend two legal
    entities' consolidated statements. See `cik_cutover`.

    The `cik` recorded on each row is the registrant that actually FILED it, not the
    ticker's current one, so a row's provenance survives the boundary.
    """
    
    cutover = (cutovers or {}).get(ticker)
    filings = (cutover_filings(cutover, FUNDAMENTALS_FORMS, since, done_accessions)
               if cutover else new_filings(ticker, FUNDAMENTALS_FORMS, since,
                                           done_accessions))
    rows: list[dict] = []
    # Headcount rides the SAME walk (decision 35): the number is in the 10-K prose this loop
    # already has a handle on, so a separate fetcher would list, download and date those
    # filings a second time. The continuity guard is seeded from what is already stored and
    # grows as the walk goes -- `new_filings` is oldest-first, so each 10-K is judged against
    # every earlier one exactly as a full-history pass would judge it.
    accepted = list((headcounts or {}).get(ticker, []))
    staff: list[dict] = []
    # Unreadable filings, `(accession, error)`. Counted rather than merely skipped: a walk
    # that quietly drops filings and a walk that finds none look identical in the row count.
    failures: list[tuple[str, str]] = []
    for filing in filings:
        filing_cik = (cutover.cik_for(filing.filing_date) if cutover else cik)
        rows.extend(filing_rows(ticker, filing_cik, filing, catalogue,
                                gics_by_ticker.get(ticker), failures=failures))
        if not is_headcount_form(getattr(filing, "form", None)):
            continue
        parsed = employee_fact_frame(filing, accepted)
        if parsed is None:
            continue
        count = float(parsed["value"].iloc[0])
        accepted.append(int(count))
        staff.append({"ticker": ticker,
                      "as_of": pd.Timestamp(filing.filing_date), "employees": count})
    employees = pd.DataFrame(staff, columns=["ticker", "as_of", "employees"])
    if failures:
        logger.warning("%s: %d of %d filing(s) unreadable -- %s", ticker, len(failures),
                       len(filings),
                       ", ".join(f"{acc} ({err})" for acc, err in failures))
    # The line that would have caught the `cols` NameError in hour one instead of hour ten:
    # filings were walked and NOT ONE of them yielded a fact. Never a normal outcome -- every
    # 10-K/10-Q in `FUNDAMENTALS_FORMS` carries some catalogue field -- so it is an ERROR even
    # though the walk itself completed and the run will report success.
    if filings and not rows:
        logger.error("%s: 0 facts from %d filing(s) (%d unreadable) -- the ticker's whole "
                     "history is missing, not empty", ticker, len(filings), len(failures))
    df = pd.DataFrame(rows, columns=_COLS)
    if df.empty:
        return {Tables.fundamentals_facts: df, Tables.fundamentals_employees: employees}
    # One filing can tag the same field on the same WINDOW twice (a nudged boundary day).
    # Postgres rejects an upsert touching one PK row twice, so collapse here. Measured over
    # 337,190 swept facts, this now costs **3 rows** -- against 18,604 under the old
    # fiscal-label PK, of which 16,340 collisions held two genuinely different values.
    #
    # A CUTOVER cannot introduce a duplicate here -- the two CIK walks are split on a date
    # and are disjoint by construction -- but assert it rather than assume it: a silent
    # duplicate accession would double a period's facts and every downstream sum with them.
    before = df["accession_number"].nunique()
    df = df.drop_duplicates(subset=list(Tables.fundamentals_facts.pk), keep="last")
    if cutover and df["accession_number"].nunique() != before:
        raise ValueError(
            f"{ticker}: the {cutover.predecessor_cik} -> {cutover.successor_cik} cutover at "
            f"{cutover.cutover_date.date()} lost accessions in dedup "
            f"({before} -> {df['accession_number'].nunique()}); the two walks overlap")
    # Two 10-K/A amendments filed the same day would collide on the employees PK.
    employees = employees.drop_duplicates(subset=["ticker", "as_of"], keep="last")
    return {Tables.fundamentals_facts: df, Tables.fundamentals_employees: employees}


def fetch_fundamentals_sec(context: Context, tickers: list[str],
                           years_history: int, *, full: bool = False) -> None:
    # `context.config_dir` is the CLI's `-c` value, resolved once by `get_config_context`;
    # threading it explicitly is what lets a non-default `-c` actually reach the catalogue.
    catalogue = load_catalogue(context.config_dir)
    # All three GICS levels: the regimes config declares its membership at whichever level
    # is natural (bank/insurer by sub-industry, real_estate by industry group, utility and
    # energy by sector), so reading only one level mis-routes whole sectors.
    #
    # Off `load_cik_mapping`'s frame, which already carries them, and handed down to
    # `run_edgar_fetch` -- otherwise the universe is read twice in the same run, once here
    # for the regimes and once inside the driver for the CIKs.
    levels = ["sector", "industry_group", "sub_industry"]
    cik_map = load_cik_mapping(context, tickers)
    gics = {row.ticker: {lvl: getattr(row, lvl) for lvl in levels}
            for row in cik_map.itertuples()}
    # The headcount continuity guard's seed, and the ONE read of this table that is
    # deliberately unfiltered: `history_by_ticker` seeds a per-ticker median from every
    # stored headcount, and a `where=` on the run's ticker list would silently narrow the
    # continuity guard to the chunk being fetched. Three columns of an annual, ~500-ticker
    # table, so the whole-table read is bounded by construction.
    stored = context.store.load(Tables.fundamentals_employees,
                                columns=["ticker", "as_of", "employees"], optional=True)
    headcounts = history_by_ticker(
        stored.rename(columns={"as_of": "filing_date", "employees": "value"})
        if stored is not None else None)
    cutovers = load_cutovers(context.config_dir)
    if cutovers:
        context.log.info("fundamentals: %d CIK cutover(s) declared -- %s", len(cutovers),
                         ", ".join(f"{t} @{c.cutover_date.date()}"
                                   for t, c in sorted(cutovers.items())))
    run_edgar_fetch(
        context, tickers, years_history,
        # `fundamentals_facts` stays FIRST: it keys the manifest window and the accession
        # dedup set, and headcount is a by-product of the same filings.
        tables=(Tables.fundamentals_facts, Tables.fundamentals_employees),
        build=partial(build_ticker_fundamentals, catalogue=catalogue,
                      gics_by_ticker=gics, cutovers=cutovers, headcounts=headcounts),
        desc="fundamentals (linkbase)", full=full, cik_map=cik_map,
        max_workers=int(context.config.data_extract.fundamentals_workers))
