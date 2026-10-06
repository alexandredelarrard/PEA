"""
Per-filing SEC XBRL walk -> `fundamentals_facts`: one row per catalogue field per period per filing.

Strictly as-filed: every value is a number the filer tagged, on the period shape it tagged it with; Q4 = FY - YTD9
and YTD decumulation happen later, in memory, in the history build. `xbrl_linkbase` picks the concept(s) for a
field, `entity_scope` picks the consolidated-registrant facts, and this module turns both into rows. A field or
period with no usable value is emitted as a value-less row carrying a `dc_code`, so every null has a reason.
`run_edgar_fetch` lists the index filings not yet stored; a filing with no XBRL, an unreadable one or one
with no consolidated facts becomes an empty-filing marker.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from functools import partial
from typing import Any, cast

import pandas as pd

from src.constants.constants import FUNDAMENTALS_FORMS
from src.context import Context
from src.data_extract.utils.common.edgar_driver import (
    EdgarFetch,
    EdgarScope,
    FilingStamp,
    run_edgar_fetch,
)
from src.data_extract.utils.common.parallel_fetch import PROGRAMMING_ERRORS
from src.data_extract.utils.common.sec_io import ParseFailureError, TransientReadError, filing_xbrl
from src.data_extract.utils.common.sec_utils import load_cik_mapping
from src.data_extract.utils.fundamentals import entity_scope as scope
from src.data_extract.utils.fundamentals.kpi_catalogue import Catalogue, load_catalogue
from src.data_extract.utils.fundamentals.periods import AMBIGUOUS_DURATION, ANNUAL, OTHER_SHAPE, QUARTERLY, period_shape
from src.data_extract.utils.fundamentals.reason_codes import NOT_DISCLOSED, PERIOD_INTERSECTION_PARTIAL
from src.data_extract.utils.fundamentals.xbrl_linkbase import (
    FIELD_SUM,
    INCOMPLETE_ROLL_UP,
    LINKBASE_SUM,
    NO_USABLE_PERIOD,
    STATEMENT_LEAF_SUM,
    UNRESOLVED,
    ArcGraph,
    Resolution,
    bare,
    calculation_arcs,
    resolve_field,
    segment_only_concepts,
    statement_arcs,
)
from src.data_store.schema import Table, Tables

logger = logging.getLogger(__name__)

_COLS = [
    "ticker",
    "accession_number",
    "field",
    "fiscal_year",
    "fiscal_period",
    "duration_type",
    "cik",
    "form",
    "filing_date",
    "is_amendment",
    "period_of_report",
    "regime",
    "period_start",
    "period_end",
    "period_days",
    "value",
    "unit",
    "decimals",
    "resolution_method",
    "source_concept",
    "roll_up_children",
    "root_anchor",
    "adjustment",
    "role_uri",
    "is_extension",
    "dc_code",
]

#: Fiscal period recorded when the filer tagged none (a named state rather than an empty string or NULL).
UNLABELLED_PERIOD = "NA"


def _period_frame(facts: pd.DataFrame) -> pd.DataFrame:
    """Facts with `duration_type`, `period_days` and `period_end` normalised (instants take `period_instant`)."""
    out = facts.copy()
    # edgartools may omit these columns, and `_period_key` reads them as attributes off `itertuples`.
    for column in ("fiscal_year", "fiscal_period", "unit_ref", "decimals"):
        if column not in out.columns:
            out[column] = None
    start = pd.to_datetime(cast(Any, out.get("period_start")), errors="coerce")
    end = pd.to_datetime(cast(Any, out.get("period_end")), errors="coerce")
    if "period_instant" in out.columns:
        instant = pd.to_datetime(out["period_instant"], errors="coerce")
        end = end.fillna(instant)
    out["period_start"] = start
    out["period_end"] = end
    out["period_days"] = (end - start).dt.days
    out["duration_type"] = [period_shape(str(pt), d) for pt, d in zip(out.get("period_type", ""), out["period_days"], strict=False)]
    out["_bare"] = [scope.bare_concept(c) for c in out["concept"]]
    return out


def _period_key(row) -> tuple:
    return (row.fiscal_year, row.fiscal_period, row.duration_type, row.period_start, row.period_end)


#: `decimals` value meaning "exact" -- the finest precision an XBRL fact can declare.
_DECIMALS_EXACT = "INF"


def _precision(decimals) -> float:
    """XBRL `decimals` as a sortable precision (higher is finer): `INF` is +inf, absent/unparseable is -inf."""
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
    """`{period key: fact}` for one concept; namespaced match when `concept` has a prefix, bare otherwise.

    A duplicate (concept, period) keeps the finer `decimals`, not the later one. Two DIFFERENT values are always
    recorded on the survivor as `duplicate_fact`; identical re-tagging is not flagged.
    """
    column = "concept" if ":" in concept else "_bare"
    hits = facts[facts[column] == concept]
    out: dict[tuple, dict] = {}
    for row in hits.itertuples():
        key = _period_key(row)
        fact = {
            "value": float(cast(Any, row.numeric_value)),
            "unit": getattr(row, "unit_ref", None),
            "decimals": getattr(row, "decimals", None),
            "fiscal_year": row.fiscal_year,
            "fiscal_period": row.fiscal_period,
            "duration_type": row.duration_type,
            "period_start": row.period_start,
            "period_end": row.period_end,
            "period_days": row.period_days,
        }
        prior = out.get(key)
        if prior is None:
            out[key] = fact
            continue
        winner, loser = (fact, prior) if _precision(fact["decimals"]) > _precision(prior["decimals"]) else (prior, fact)
        if winner["value"] != loser["value"]:
            seen = list(prior.get("duplicate_fact", []))
            seen.append(
                {
                    "concept": concept,
                    "kept": winner["value"],
                    "kept_decimals": str(winner["decimals"]),
                    "dropped": loser["value"],
                    "dropped_decimals": str(loser["decimals"]),
                }
            )
            winner = {**winner, "duplicate_fact": seen}
        out[key] = winner
    return out


#: Forms whose statement periods are all annual, so any quarterly fact in them comes from a note. Matched exactly.
_ANNUAL_FORMS: tuple[str, ...] = ("10-K", "10-K/A")


_Window = tuple[tuple, pd.Timestamp, pd.Timestamp]


def _annual_windows(periods: dict[tuple, dict]) -> list[_Window]:
    """`(key, start, end)` for every ANNUAL period with readable bounds."""
    out = []
    for key, period in periods.items():
        if period.get("duration_type") != ANNUAL:
            continue
        low, high = pd.Timestamp(period["period_start"]), pd.Timestamp(period["period_end"])
        if pd.notna(low) and pd.notna(high):
            out.append((key, low, high))
    return out


def _covering_annual(windows: list[_Window], period: dict) -> tuple | None:
    """Key of the annual window that contains `period`, or None; dated by containment, not the `fiscal_year` label."""
    start, end = pd.Timestamp(period["period_start"]), pd.Timestamp(period["period_end"])
    if pd.isna(start) or pd.isna(end):
        return None
    for key, low, high in windows:
        if low <= start and end <= high:
            return key
    return None


def _lone_quarters(periods: dict[tuple, dict], filing_windows: list[_Window] | None = None) -> dict[tuple, tuple]:
    """`{key of a quarter that is the ONLY one in its fiscal year: key of that year}`.

    An annual report's quarterly data table publishes a series (all four quarters), while a fourth-quarter
    narrative tags a discrete item that lands alone, so a lone quarter is a prose aside. Years come from this
    field's annual windows, falling back to the filing-wide `filing_windows` only when the field has none.
    A quarter with no covering annual window is kept (silence is not evidence).
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
    """Every annual window the filing declares across all fields, deduplicated on `(start, end)`."""
    by_span: dict[tuple, _Window] = {}
    for periods in values.values():
        for key, low, high in _annual_windows(periods):
            by_span.setdefault((low, high), (key, low, high))
    return list(by_span.values())


def _drop_note_only_quarter(
    periods: dict[tuple, dict],
    *,
    form: str,
    filing_windows: list[_Window] | None = None,
) -> dict[tuple, dict]:
    """Drop quarterly facts an ANNUAL report published alone for their fiscal year (see `_lone_quarters`).

    Such a value is a discrete item from a note, never the quarter's total. Gated on `_ANNUAL_FORMS`: every
    quarter in a 10-Q is "lone", so an ungated rule would delete the quarterly grain. Each dropped quarter is
    recorded as `note_quarter_rejected` on its covering annual period when that period survives. Complements,
    and does not replace, the value-based check in `periods._drop_annual_masquerading_as_quarter`.
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
        rejected.append({"period_end": str(pd.Timestamp(dropped["period_end"]).date()), "value": dropped["value"]})
        out[year] = {**host, "note_quarter_rejected": rejected}
    return out


def _retry_without(
    name,
    resolution,
    catalogue,
    graph,
    available,
    regime,
    facts,
    durations,
    zero_only,
    magnitudes,
    ticker,
    prefer_structure,
    form: str,
    filing_windows: list[_Window],
) -> tuple[Resolution, dict[tuple, dict], dict[tuple, dict]] | None:
    """Re-resolve `name` once with the concept whose periods were all refused withheld, or None.

    The concept is withheld by removing it from `available` (the filing's bare reported concepts), so the
    resolver itself is unchanged. Returns `(resolution, kept, refused)`, or None when there is no other
    candidate, the retry lands on the same concept, or its periods are refused too; the caller then records
    the refusal. A single retry, not a loop over every candidate.
    """
    dead = resolution.concept
    if not dead:
        return None
    retry = resolve_field(
        catalogue.field(name),
        graph,
        available - {bare(dead)},
        catalogue,
        regime,
        duration_concepts=durations,
        zero_only=zero_only,
        magnitudes=magnitudes,
        ticker=ticker,
        prefer_structure=prefer_structure,
    )
    if not retry.resolved or retry.concept == dead:
        return None
    periods, refused = _materialise(retry, facts)
    kept = _drop_note_only_quarter(periods, form=form, filing_windows=filing_windows)
    return (retry, kept, refused) if kept else None


def _materialise(resolution: Resolution, facts: pd.DataFrame) -> tuple[dict[tuple, dict], dict[tuple, dict]]:
    """Turn one field's resolution into `({period key: value + provenance}, {refused period key: stub})`.

    `linkbase_sum` and `statement_leaf_sum` emit a period ONLY where every leg reports that same period (a
    partial sum is not the total); the other periods come back as value-less refused stubs so the caller can
    code them. `subtract` concepts are then netted off the periods that have them.
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
            # Union duplicates across all legs, not just `base`'s.
            duplicates = [d for c, _ in resolution.children for d in legs[c][key].get("duplicate_fact", [])]
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
    """A value-less period stub for a refused window, its period columns copied from a leg that reported it."""
    reported = next(legs[c][key] for c in legs if key in legs[c])
    return {
        "fiscal_year": reported["fiscal_year"],
        "fiscal_period": reported["fiscal_period"],
        "duration_type": reported["duration_type"],
        "period_start": reported["period_start"],
        "period_end": reported["period_end"],
        "period_days": reported["period_days"],
        "value": None,
        "unit": reported.get("unit"),
        "decimals": None,
    }


def _compose(
    spec,
    component_fields: tuple[str, ...],
    resolved: dict[str, dict[tuple, dict]],
) -> tuple[dict[tuple, dict], str | None]:
    """Sum a field composed of other catalogue fields (e.g. `totalDebt`, `ppeNet`), per period.

    Missing components count as zero unless the catalogue's `roll_up.require_all` / `require_any` makes them
    load-bearing; the requirement is tested per period. Returns `(values, dc_code)`: when nothing survives the
    code is `incomplete_roll_up` if some component resolved, `not_disclosed` if none did, so a null always
    carries its reason.
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
    """The `adjustment` JSON provenance blob, or None when empty.

    Resolution-level keys: `subtract`, `zero_only_retained`, `role_rejected`, `role_only_retained`,
    `segment_rejected`, `undeclared_rejected`, `sibling_rejected` (`[total, leg]` pairs), `basis_qualifier`.
    Period-level keys, read off `period`: `note_quarter_rejected`, `duplicate_fact`.
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


def _period_end(period: dict | None, reported: pd.Timestamp, filed: pd.Timestamp) -> pd.Timestamp:
    """The row's `period_end`, never NULL (PK column): the period's own, else `period_of_report`, else filing date.

    Only value-less rows reach the fallbacks, so a fallback never displaces a measurement.
    """
    if period is not None and pd.notna(period.get("period_end")):
        return pd.Timestamp(period["period_end"])
    return reported if pd.notna(reported) else filed


def _row(
    ticker: str,
    stamp: FilingStamp,
    reported: pd.Timestamp,
    regime: str | None,
    field: str,
    resolution: Resolution,
    period: dict | None,
    *,
    dc_code: str | None = None,
) -> dict:
    """One `fundamentals_facts` row; `period=None` gives a value-less reason-coded row.

    `reported` is the filing's `period_of_report` (NaT when absent). `dc_code` overrides the resolution's own,
    for a refused period on a field that otherwise resolved.
    """
    children = [[c, w] for c, w in resolution.children] if resolution.children else None
    return {
        "ticker": ticker,
        "cik": stamp.cik,
        "accession_number": stamp.accession_number,
        "field": field,
        "fiscal_year": int(period["fiscal_year"]) if period and pd.notna(period.get("fiscal_year")) else stamp.filed.year,
        "fiscal_period": (str(period["fiscal_period"]) if period and pd.notna(period.get("fiscal_period")) else UNLABELLED_PERIOD),
        "duration_type": period["duration_type"] if period else OTHER_SHAPE,
        "form": stamp.form,
        "filing_date": stamp.filed,
        "is_amendment": stamp.is_amendment,
        "period_of_report": reported,
        "regime": regime,
        "period_start": period["period_start"] if period else pd.NaT,
        # PK column: a reason-coded row falls back to the filing's period of report (see `_period_end`).
        "period_end": _period_end(period, reported, stamp.filed),
        "period_days": period["period_days"] if period else None,
        "value": period["value"] if period else None,
        "unit": period.get("unit") if period else None,
        # `str(NaN)` is the string "nan", which joins and compares as a real value.
        "decimals": (str(period["decimals"]) if period and period.get("decimals") is not None and pd.notna(period.get("decimals")) else None),
        "resolution_method": resolution.method,
        "source_concept": resolution.source_concept,
        "roll_up_children": json.dumps(children) if children else None,
        "root_anchor": resolution.anchor,
        "adjustment": _adjustment_json(resolution, period),
        "role_uri": resolution.role_uri,
        "is_extension": resolution.is_extension,
        "dc_code": dc_code or resolution.dc_code,
    }


def filing_rows(ticker: str, stamp: FilingStamp, catalogue: Catalogue, gics: dict[str, str | None] | None) -> list[dict]:
    """Every catalogue field, for every period, from one filing (`stamp.filing`, filed by `stamp.cik`); [] without XBRL.

    The two failure classes differ on purpose: edgartools' `xbrl()` parses the filer's XBRL, so any
    exception from it is an unreadable filing (`ParseFailureError`; `TransientReadError` passes
    through), while `PROGRAMMING_ERRORS` from our own `rows_from_xbrl` are re-raised.
    """
    filing = stamp.filing
    try:
        xbrl = filing_xbrl(filing)
    except TransientReadError:
        raise
    except Exception as exc:  # noqa: BLE001 -- the filer's XBRL, not our code
        raise ParseFailureError(f"{stamp.accession_number}: XBRL unreadable ({type(exc).__name__}: {exc})") from exc
    if xbrl is None:
        return []
    try:
        return rows_from_xbrl(ticker, stamp, xbrl, catalogue, gics)
    except PROGRAMMING_ERRORS:
        raise  # our bug, not the filer's
    except Exception as exc:  # noqa: BLE001 -- one bad filing
        raise ParseFailureError(f"{stamp.accession_number}: facts unresolvable ({type(exc).__name__}: {exc})") from exc


def rows_from_xbrl(
    ticker: str, stamp: FilingStamp, xbrl, catalogue: Catalogue, gics: dict[str, str | None] | None, *, prefer_structure: bool = True
) -> list[dict]:
    """`filing_rows` with the already-parsed XBRL handed in, so an audit can reuse one parse.

    `stamp.cik` is the filer CIK. `prefer_structure` is documented on `resolve_field`; production keeps True.
    """
    facts = scope.consolidated_facts(xbrl.facts.to_dataframe())
    if facts.empty:
        return []
    facts = _period_frame(facts)
    available = scope.reported_concepts(facts)
    # Filing-level properties the resolver cannot derive from structure: flow concepts and all-zero concepts.
    durations = scope.duration_concepts(facts)
    zero_only = scope.zero_only_concepts(facts)
    # Peak |value| per concept, for `xbrl_linkbase.sibling_leg`; keeps resolution period-agnostic.
    magnitudes = scope.peak_magnitudes(facts)
    reported = pd.to_datetime(cast(Any, stamp.period_of_report), errors="coerce")
    # One `calculation_linkbase()` read, two views of it -- see `statement_arcs`.
    arcs = calculation_arcs(xbrl)
    graph = ArcGraph(statement_arcs(xbrl, arcs))
    # Unfiltered arcs: `statement_arcs` has already dropped the segment-note arcs. See `xbrl_linkbase.SEGMENT_ROLE`.
    segment_only = segment_only_concepts(arcs)
    regime = catalogue.regime_for(gics, [str(r) for r in graph.arcs.get("role_uri", pd.Series(dtype=str))])

    # Concept-backed fields first; composed fields (`FIELD_SUM`) then read those results.
    resolutions: dict[str, Resolution] = {}
    values: dict[str, dict[tuple, dict]] = {}
    #: field -> periods the strict intersection refused; kept apart so a composed field never sums a stub.
    refused: dict[str, dict[tuple, dict]] = {}
    for name in catalogue.extracted_fields:
        resolution = resolve_field(
            catalogue.field(name),
            graph,
            available,
            catalogue,
            regime,
            duration_concepts=durations,
            zero_only=zero_only,
            magnitudes=magnitudes,
            ticker=ticker,
            prefer_structure=prefer_structure,
            segment_only=segment_only,
        )
        resolutions[name] = resolution
        if resolution.method != FIELD_SUM:
            values[name], refused[name] = _materialise(resolution, facts)
    # After every field is materialised (lone quarters are dated on the filing's calendar), before `_compose`.
    #: Fields the note guard emptied outright; their stub is coded `AMBIGUOUS_DURATION`, not `NO_USABLE_PERIOD`.
    note_refused: set[str] = set()
    form = str(stamp.form or "").upper()
    if form in _ANNUAL_FORMS:
        filing_windows = _filing_annual_windows(values)
        for name, periods in list(values.items()):
            kept = _drop_note_only_quarter(periods, form=form, filing_windows=filing_windows)
            if not periods or kept:
                values[name] = kept
                continue
            retry = _retry_without(
                name,
                resolutions[name],
                catalogue,
                graph,
                available,
                regime,
                facts,
                durations,
                zero_only,
                magnitudes,
                ticker,
                prefer_structure,
                form,
                filing_windows,
            )
            if retry is not None:
                resolutions[name], values[name], refused[name] = retry
                continue
            note_refused.add(name)
            values[name] = kept
    for name, resolution in list(resolutions.items()):
        if resolution.method == FIELD_SUM:
            composed, reason = _compose(catalogue.field(name), resolution.component_fields, values)
            values[name] = composed
            if reason:
                resolutions[name] = replace(resolution, method=UNRESOLVED, dc_code=reason)

    rows: list[dict] = []
    for name, resolution in resolutions.items():
        periods = values.get(name) or {}
        if not periods:
            # No value in this filing: emit ONE reason-coded row so a downstream null is always explained;
            # a resolved field with no period gets its own code.
            if resolution.resolved:
                resolution = replace(resolution, method=UNRESOLVED, dc_code=(AMBIGUOUS_DURATION if name in note_refused else NO_USABLE_PERIOD))
            rows.append(_row(ticker, stamp, reported, regime, name, resolution, None))
            continue
        rows.extend(_row(ticker, stamp, reported, regime, name, resolution, period) for period in periods.values())
    # Refused periods, each a value-less row coded `PERIOD_INTERSECTION_PARTIAL`, for every field.
    for name, periods in refused.items():
        # Disjoint by construction; a shared key would write one PK twice and dedup could keep the stub.
        assert not (set(periods) & set(values.get(name, {}))), f"{ticker} {stamp.accession_number} {name}: a refused period is also resolved"
        rows.extend(
            _row(ticker, stamp, reported, regime, name, resolutions[name], period, dc_code=PERIOD_INTERSECTION_PARTIAL) for period in periods.values()
        )
    return rows


def parse_fundamentals(
    ticker: str,
    cik: str,
    stamp: FilingStamp,
    scope: EdgarScope,
    *,
    catalogue: Catalogue,
    gics_by_ticker: dict[str, dict],
) -> dict[Table, pd.DataFrame]:
    """One filing's `fundamentals_facts` rows, one per primary key (a filing can tag one field on one window twice)."""
    df = pd.DataFrame(filing_rows(ticker, stamp, catalogue, gics_by_ticker.get(ticker)), columns=_COLS)
    return {Tables.fundamentals_facts: df.drop_duplicates(subset=list(Tables.fundamentals_facts.pk), keep="last")}


def fundamentals_fetch(context: Context, cik_map: pd.DataFrame) -> EdgarFetch:
    """The `fundamentals_facts` fetch: the catalogue from `context.config_dir`, all three GICS levels of `cik_map`'s tickers."""
    catalogue = load_catalogue(str(context.config_dir))
    levels = ["sector", "industry_group", "sub_industry"]
    df_gics = cik_map[levels].set_index(cik_map["ticker"].map(str))
    gics = cast(dict[str, dict], df_gics[~df_gics.index.duplicated(keep="last")].to_dict("index"))
    return EdgarFetch(
        desc="fundamentals (linkbase)",
        tables=(Tables.fundamentals_facts,),
        forms=tuple(FUNDAMENTALS_FORMS),
        parse=partial(parse_fundamentals, catalogue=catalogue, gics_by_ticker=gics),
    )


def fetch_fundamentals_sec(
    context: Context, tickers: list[str], years_history: int, *, full: bool = False, as_of: pd.Timestamp | None = None, no_cap: bool = False
) -> None:
    """Fetch the `fundamentals_facts` documents not yet stored for `tickers` (see `run_edgar_fetch`)."""
    cik_map = load_cik_mapping(context, tickers)
    run_edgar_fetch(
        context,
        tickers,
        years_history,
        fundamentals_fetch(context, cik_map),
        full=full,
        cik_map=cik_map,
        max_workers=int(context.config.data_extract.fundamentals_workers),
        as_of=as_of,
        no_cap=no_cap,
    )
