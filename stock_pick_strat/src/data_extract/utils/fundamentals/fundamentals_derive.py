"""
fundamentals_derive.py
------------------------
Rebuilds `fundamentals_history` (unchanged shape/PK: `(ticker, as_of)`, ~150
columns -- see `schema_registry.py`) FROM the new accession-grain, amendment-
aware `fundamentals_facts` table, so every existing test and `data_aggregate`
consumer of `fundamentals_history` needs zero changes.

Reuses `fetch_fundamentals.py`'s proven `_spine_grid` / `_assemble_base` /
`_derive_history` UNCHANGED (they operate on a generic dict-of-[end, filed, val]
DataFrames shape that doesn't care whether the source was the companyfacts JSON
or `fundamentals_facts`) -- only the ADAPTER from `fundamentals_facts` rows into
that shape is new. This is the "leak-guard logic preserved, not reimplemented"
piece from the design: `_assemble_base`'s median-of-spine `as_of` computation and
its two leak-guard passes run exactly as before.
"""
from __future__ import annotations

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.sec_utils import load_cik_mapping
from src.data_extract.utils.fundamentals.fetch_fundamentals import (
    _assemble_base, _derive_history, _spine_grid,
)
from src.data_extract.utils.fundamentals.fundamentals_tags import (
    EXTRA_FLOW_TAGS, EXTRA_STOCK_TAGS, FLOW_TAGS, LATEST_DURATION_TAGS,
    STOCK_TAGS,
)

_FLOW_FIELDS: dict[str, list[str]] = {**FLOW_TAGS, **EXTRA_FLOW_TAGS}
_STOCK_FIELDS: dict[str, list[str]] = {**STOCK_TAGS, **EXTRA_STOCK_TAGS}


def _load_facts_for_ticker(context: Context, ticker: str) -> pd.DataFrame:
    """All `fundamentals_facts` rows for one ticker. `DataStore.load` has no
    server-side WHERE filter, so this loads the table then filters client-side --
    acceptable at this table's scale (per-filing grain, not per-day); revisit if
    the table grows large enough to need a scoped query."""
    df = context.store.load("fundamentals_facts")
    if df.empty:
        return df
    return df[df["ticker"] == ticker].copy()


def _resolve_latest_per_period(
    facts: pd.DataFrame, as_of_cutoff: pd.Timestamp | None,
) -> pd.DataFrame:
    """Collapse original + amendments to ONE row per (field, fiscal_year,
    fiscal_period, duration_type): the row with the LATEST filing_date <=
    as_of_cutoff (default: everything currently stored). This is the ONLY place
    amendment precedence is applied, and it never needs to walk
    `amends_accession` -- 'latest qualifying filing_date' already IS the
    point-in-time-correct answer. With zero amendments (every existing synthetic
    test fixture), this is identical to the old file's "earliest-filed-wins"
    behavior when there's exactly one disclosure per period -- the ~40 existing
    tests are unaffected; behavior only diverges on periods that actually have an
    amendment, which is new coverage, not a regression.

    No amendment value is ever exposed before its own filing date: this filter is
    a tautological guarantee (an amendment can only be selected once
    `as_of_cutoff` is at/past ITS OWN filing_date), and whatever it selects still
    passes through `_assemble_base`'s unchanged leak guards as a second,
    independent check.
    """
    if facts.empty:
        return facts
    key = ["field", "fiscal_year", "fiscal_period", "duration_type"]
    f = facts
    if as_of_cutoff is not None:
        f = facts[pd.to_datetime(facts["filing_date"]) <= as_of_cutoff]
    if f.empty:
        return f
    return f.sort_values("filing_date").drop_duplicates(subset=key, keep="last")


def _to_concept_series(resolved: pd.DataFrame, field: str, duration_types: tuple[str, ...]) -> pd.DataFrame:
    """One field's resolved rows -> [end, filed, val], the shape
    `_spine_grid`/`_assemble_base` expect (matches what the old companyfacts-JSON
    path's per-concept DataFrames looked like before this rewrite)."""
    empty = pd.DataFrame(columns=["end", "filed", "val"])
    if resolved.empty:
        return empty
    sub = resolved[(resolved["field"] == field) & (resolved["duration_type"].isin(duration_types))]
    sub = sub.dropna(subset=["period_end", "value"])
    if sub.empty:
        return empty
    out = (sub[["period_end", "filing_date", "value"]]
           .rename(columns={"period_end": "end", "filing_date": "filed", "value": "val"}))
    # `fundamentals_facts` round-trips filing_date/period_end through Postgres DATE
    # columns, which pandas can read back as plain `datetime.date` objects (object
    # dtype) rather than datetime64 -- `_assemble_base`'s row-wise max() over mixed
    # object-dtype `_filed` columns then breaks comparing `date` against a NaN
    # float. Coerce explicitly so every concept's `end`/`filed` is real datetime64.
    out["end"] = pd.to_datetime(out["end"])
    out["filed"] = pd.to_datetime(out["filed"])
    return out.sort_values("end").drop_duplicates(subset=["end"], keep="last").reset_index(drop=True)


def _option_overhang_from_facts(resolved: pd.DataFrame) -> pd.DataFrame:
    """(diluted - basic) / basic, matched on period `end` (already deduped to one
    value per fiscal period by `_resolve_latest_per_period`, so end-only matching
    is safe here -- unlike the old file's raw-fact version, which must match on
    both start AND end to avoid pairing a YTD basic against a quarterly diluted)."""
    b = _to_concept_series(resolved, "basicShares", ("instant",))
    d = _to_concept_series(resolved, "dilutedShares", ("instant",))
    if b.empty or d.empty:
        return pd.DataFrame(columns=["end", "filed", "val"])
    j = b.merge(d, on="end", suffixes=("_b", "_d"))
    j = j[(j["val_b"] > 0) & j["val_d"].notna()]
    if j.empty:
        return pd.DataFrame(columns=["end", "filed", "val"])
    j = j.assign(val=(j["val_d"] - j["val_b"]) / j["val_b"],
                filed=j[["filed_b", "filed_d"]].max(axis=1))
    return j[["end", "filed", "val"]].sort_values("end").reset_index(drop=True)


def derive_fundamentals_history(
    context: Context,
    ticker: str,
    as_of_cutoff: pd.Timestamp | None = None,
    sector: str | None = None,
    industry_group: str | None = None,
) -> pd.DataFrame:
    
    """Rebuild ONE ticker's `fundamentals_history`-shaped frame (exact existing
    PK/columns) from `fundamentals_facts`. `as_of_cutoff=None` (default) uses
    everything currently stored (today's best knowledge); a real timestamp
    simulates 'what this table would have looked like built on date D' for
    walk-forward / leak audits -- an axis orthogonal to the row's own `as_of`
    column (which answers "when did THIS period's data become public").
    """
    facts = _load_facts_for_ticker(context, ticker)
    if facts.empty:
        return pd.DataFrame()

    if sector is None or industry_group is None:
        mapping = load_cik_mapping(context)
        row = mapping[mapping["ticker"] == ticker]
        if not row.empty:
            sector = sector or row.iloc[0].get("sector")
            industry_group = industry_group or row.iloc[0].get("industry_group")

    resolved = _resolve_latest_per_period(facts, as_of_cutoff)
    if resolved.empty:
        return pd.DataFrame()

    flows = {f: _to_concept_series(resolved, f, ("quarterly",)) for f in _FLOW_FIELDS}
    annuals = {f: _to_concept_series(resolved, f, ("annual",)) for f in _FLOW_FIELDS}
    stocks = {f: _to_concept_series(resolved, f, ("instant",)) for f in _STOCK_FIELDS}
    shares = _to_concept_series(resolved, "sharesOutstanding", ("instant",))
    latest = {f: _to_concept_series(resolved, f, ("instant",)) for f in LATEST_DURATION_TAGS}
    overhang = _option_overhang_from_facts(resolved)

    ends = _spine_grid(flows, stocks)
    if ends is None:
        return pd.DataFrame()
    base = _assemble_base(ends, flows, annuals, stocks, shares, latest, overhang)
    if base.empty:
        return pd.DataFrame()
    return _derive_history(base, ticker, sector, industry_group)


def rebuild_fundamentals_history(context: Context, tickers: list[str]) -> pd.DataFrame:
    """Batch entry point: rebuild + persist `fundamentals_history` for `tickers`
    from `fundamentals_facts` (mirrors `fetch_fundamentals.py::build_fundamentals_history_sec`'s
    persistence pattern -- upsert on the registry PK `(ticker, as_of)`)."""
    mapping = load_cik_mapping(context)
    sector_by_ticker = {r["ticker"]: (r.get("sector"), r.get("industry_group"))
                       for _, r in mapping.iterrows()} if not mapping.empty else {}

    frames = []
    for ticker in tickers:
        sector, industry_group = sector_by_ticker.get(ticker, (None, None))
        hist = derive_fundamentals_history(context, ticker, sector=sector, industry_group=industry_group)
        if not hist.empty:
            frames.append(hist)

    if not frames:
        context.log.info("rebuild_fundamentals_history: no rows derived for %d ticker(s).", len(tickers))
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    context.store.save("fundamentals_history", out)
    context.log.info("rebuild_fundamentals_history: saved %d fundamentals_history rows for %d ticker(s).",
                     len(out), out["ticker"].nunique())
    return out
