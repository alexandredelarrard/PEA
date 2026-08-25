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
publication-event grain and the no-leakage property of `fundamentals_history` provable
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
from dataclasses import replace
from functools import partial

import pandas as pd

from src.constants.constants import FUNDAMENTALS_FORMS
from src.context import Context
from src.data_extract.utils.common.edgar_driver import new_filings, run_edgar_fetch
from src.data_extract.utils.fundamentals import entity_scope as scope
from src.data_extract.utils.fundamentals.cik_cutover import (
    Cutover, cutover_filings, load_cutovers)
from src.data_extract.utils.fundamentals.fundamentals_employees import (
    employee_fact_frame, history_by_ticker, is_headcount_form)
from src.data_extract.utils.fundamentals.kpi_catalogue import Catalogue, load_catalogue
from src.data_extract.utils.fundamentals.periods import OTHER_SHAPE, period_shape
from src.data_extract.utils.fundamentals.reason_codes import (
    NOT_DISCLOSED, PERIOD_INTERSECTION_PARTIAL)
from src.data_extract.utils.fundamentals.xbrl_linkbase import (
    FIELD_SUM, INCOMPLETE_ROLL_UP, LINKBASE_SUM, NO_USABLE_PERIOD, STATEMENT_LEAF_SUM,
    UNRESOLVED, ArcGraph, Resolution, resolve_field, statement_arcs)
from src.data_store.schema import Table, Tables

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
    if resolution.undeclared_rejected:
        blob["undeclared_rejected"] = list(resolution.undeclared_rejected)
    if resolution.sibling_rejected:
        blob["sibling_rejected"] = [list(pair) for pair in resolution.sibling_rejected]
    if resolution.basis_qualifier:
        blob["basis_qualifier"] = resolution.basis_qualifier
    if period and period.get("duplicate_fact"):
        blob["duplicate_fact"] = period["duplicate_fact"]
    return json.dumps(blob) if blob else None


def _period_end(period: dict | None, filing) -> pd.Timestamp:
    """The row's `period_end`, guaranteed non-NULL because it is part of the PK.

    Falls back through the filing's `period_of_report` to its filing date. Both fallbacks are
    only ever reached by a row that carries no value -- a reason-coded absence, or the handful
    of duration facts (10 in 109,267) whose window is unreadable -- so a fallback can never
    displace a real measurement.
    """
    if period is not None and pd.notna(period.get("period_end")):
        return pd.Timestamp(period["period_end"])
    reported = pd.to_datetime(getattr(filing, "period_of_report", None), errors="coerce")
    return reported if pd.notna(reported) else pd.Timestamp(filing.filing_date)


def _row(ticker: str, cik: str, filing, regime: str | None, field: str,
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
        "accession_number": filing.accession_number, "field": field,
        "fiscal_year": int(period["fiscal_year"]) if period and pd.notna(
            period.get("fiscal_year")) else pd.Timestamp(filing.filing_date).year,
        "fiscal_period": (str(period["fiscal_period"]) if period and pd.notna(
            period.get("fiscal_period")) else UNLABELLED_PERIOD),
        "duration_type": period["duration_type"] if period else OTHER_SHAPE,
        "form": filing.form,
        "filing_date": pd.Timestamp(filing.filing_date),
        "is_amendment": str(filing.form).upper().endswith("/A"),
        "period_of_report": pd.to_datetime(getattr(filing, "period_of_report", None),
                                           errors="coerce"),
        "regime": regime,
        "period_start": period["period_start"] if period else pd.NaT,
        # `period_end` is a PK column, so it cannot be NULL -- and a REASON-CODED row has no
        # period of its own by definition. It falls back to the filing's own period of report,
        # which is the honest reading ("this field was absent as of the period this filing
        # covers") and cannot collide: a field with any usable period emits no such row, and
        # the key already contains `field`.
        "period_end": _period_end(period, filing),
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
                gics: dict[str, str | None] | None) -> list[dict]:
    """Every catalogue field, for every period, from one filing.

    Returns [] rather than raising on an unreadable filing: one bad filing must not abort a
    490-ticker walk, and its absence is visible as a gap in the accession set.
    """
    try:
        xbrl = filing.xbrl()
    except Exception:                                   # noqa: BLE001 -- unreadable XBRL
        return []
    if xbrl is None:
        return []
    return rows_from_xbrl(ticker, cik, filing, xbrl, catalogue, gics)


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
    graph = ArcGraph(statement_arcs(xbrl))
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
                                   ticker=ticker, prefer_structure=prefer_structure)
        resolutions[name] = resolution
        if resolution.method != FIELD_SUM:
            values[name], refused[name] = _materialise(resolution, facts)
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
                                     dc_code=NO_USABLE_PERIOD)
            rows.append(_row(ticker, cik, filing, regime, name, resolution, None))
            continue
        rows.extend(_row(ticker, cik, filing, regime, name, resolution, period)
                    for period in periods.values())
    # The periods route 3b refused, each as a value-less row carrying its own code. Emitted
    # for EVERY field, including the ones that resolved -- that is the whole of B.6.6.
    for name, periods in refused.items():
        # Disjoint by construction -- `refused` is `union - intersection` and `values` is the
        # intersection -- but asserted, because a key in both would write the same PK twice
        # and the dedup in `build_ticker_fundamentals` would silently keep the value-less one.
        assert not (set(periods) & set(values.get(name, {}))), (
            f"{ticker} {filing.accession_number} {name}: a refused period is also resolved")
        rows.extend(_row(ticker, cik, filing, regime, name, resolutions[name], period,
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
    for filing in filings:
        filing_cik = (cutover.cik_for(filing.filing_date) if cutover else cik)
        rows.extend(filing_rows(ticker, filing_cik, filing, catalogue,
                                gics_by_ticker.get(ticker)))
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
    # `Context` exposes no config directory (it is a named risk zone, and the CLI's `-c`
    # never reaches it), so the catalogue loads from its own default -- which is the same
    # `./configs` the CLI defaults to.
    catalogue = load_catalogue()
    # All three GICS levels: the regimes config declares its membership at whichever level
    # is natural (bank/insurer by sub-industry, real_estate by industry group, utility and
    # energy by sector), so reading only one level mis-routes whole sectors.
    levels = ["sector", "industry_group", "sub_industry"]
    universe = context.store.load(Tables.sp500_tickers, columns=["ticker", *levels],
                                  optional=True)
    gics = ({row.ticker: {lvl: getattr(row, lvl) for lvl in levels}
             for row in universe.itertuples()} if universe is not None else {})
    # The headcount continuity guard's seed. Read from the employees table itself rather
    # than from `fundamentals_facts`, where headcount no longer lives.
    stored = context.store.load(Tables.fundamentals_employees,
                                columns=["ticker", "as_of", "employees"], optional=True)
    headcounts = history_by_ticker(
        stored.rename(columns={"as_of": "filing_date", "employees": "value"})
        if stored is not None else None)
    cutovers = load_cutovers()
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
        desc="fundamentals (linkbase)", full=full,
        max_workers=int(context.config.data_extract.fundamentals_workers))
