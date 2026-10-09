"""
`fundamentals_facts` -> `fundamentals_history_sec` + `fundamentals_reason_codes`, on the publication-event grain.

`as_of` is a FILING DATE. Rules: (1) a row for every `(ticker, date)` on which >=1 value became public (an original
always qualifies); (2) an amendment emits a row only if it changes >=1 value and lands <= `MAX_AMENDMENT_LAG_DAYS`
after the original; (3) each row is a complete snapshot of latest-known values, built only from facts filed on or
before `as_of`, with every null explained by a reason code; (4) stored rows are immutable (`diff_against_stored`);
(5) same-day filings collapse to one row by `(ticker, date)`, provenance by `FORM_PRECEDENCE`; (6) across a CIK seam
one filer per fiscal period, the CIK whose stated window owns the period end (`keep_window_owner_filings`).
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from src.context import Context
from src.data_extract.utils.common.frame_sanitize import pin_dtypes
from src.data_extract.utils.common.identity import CikWindow, load_identity
from src.data_extract.utils.common.resume import recently_changed
from src.data_extract.utils.common.security_master import load_security_manual
from src.data_extract.utils.fundamentals import reason_codes as rc
from src.data_extract.utils.fundamentals.kpi_catalogue import HISTORY_KEYS, HISTORY_PROVENANCE, HISTORY_REGIME, Catalogue, load_catalogue
from src.data_extract.utils.fundamentals.periods import (
    INSTANT,
    InstantLookup,
    PeriodGuards,
    build_periods,
    fiscal_quarter_of_end,
    fiscal_year_ends,
    load_guards,
)
from src.data_store.schema import Tables
from src.utils.string import pad_cik_series

#: Form precedence for a same-day collapse; keeps `publication_form` a scalar.
FORM_PRECEDENCE: tuple[str, ...] = ("10-K", "10-K/A", "10-Q", "10-Q/A")

#: Structural, not tunable: a restated quarter stays inside some live TTM window for twelve months.
MAX_AMENDMENT_LAG_DAYS = 365

#: `{column: predicate meaning the value is IMPOSSIBLE}`, nulled by `_hard_guard` before the row is written.
#: Impossible only, never merely implausible (those are flag-only validator checks). Share counts refuse `<= 0`;
#: `totalAssets` refuses only `< 0`, since a shell can legitimately foot to zero.
HARD_GUARDS: dict[str, Callable[[float], bool]] = {
    "totalAssets": lambda v: v < 0,
    "sharesOutstanding": lambda v: v <= 0,
    "basicShares": lambda v: v <= 0,
    "dilutedShares": lambda v: v <= 0,
}

#: Identity of one as-filed measurement, used to test an amendment by value. Keyed on `period_end`, not the
#: fiscal labels, which are not unique inside a filing.
_VALUE_KEY: tuple[str, ...] = ("field", "duration_type", "period_end")

#: Every computed column, as `column -> (inputs, formula)`; must agree with `fundamentals_kpis.json`.
#: Flow inputs are TTM, so ratios are TTM over TTM.
_FORMULAS: dict[str, tuple[tuple[str, ...], Callable[[float, float], float]]] = {
    "ebitda": (("operatingIncome", "depAmort"), lambda a, b: a + b),
    "freeCashflow": (("operatingCashFlow", "capex"), lambda a, b: a - b),
    "epsDiluted": (("netIncome", "dilutedShares"), lambda a, b: a / b),
    "effectiveTaxRate": (("incomeTaxExpense", "pretaxIncome"), lambda a, b: a / b),
    "grossMargins": (("grossProfit", "totalRevenue"), lambda a, b: a / b),
    "operatingMargins": (("operatingIncome", "totalRevenue"), lambda a, b: a / b),
    "profitMargins": (("netIncome", "totalRevenue"), lambda a, b: a / b),
    "returnOnEquity": (("netIncome", "stockholdersEquity"), lambda a, b: a / b),
    "debtToEquity": (("totalDebt", "stockholdersEquity"), lambda a, b: a / b),
    "optionOverhang": (("dilutedShares", "basicShares"), lambda a, b: a / b - 1),
}

#: History columns read from the discrete quarter rather than the TTM, mapped to their source field.
_QUARTER_LABEL_COLUMNS: dict[str, str] = {"revenue_q": "totalRevenue", "netIncome_q": "netIncome"}

#: Formulas whose second operand is a denominator; a zero denominator yields None, never an infinity.
_RATIOS: frozenset[str] = frozenset(
    {"epsDiluted", "effectiveTaxRate", "grossMargins", "operatingMargins", "profitMargins", "returnOnEquity", "debtToEquity", "optionOverhang"}
)


@dataclass(frozen=True)
class TickerHistory:
    """One ticker's history and reason-code frames, built together: every null in `history` has a reason code."""

    history: pd.DataFrame
    reason_codes: pd.DataFrame


# --------------------------------------------------------------- the event ladder ---


def _same_value(before, after) -> bool:
    """Whether two as-filed values are equal, with NaN == NaN (a value-less row re-tagged changed nothing)."""
    if pd.isna(before) and pd.isna(after):
        return True
    if pd.isna(before) or pd.isna(after):
        return False
    return float(before) == float(after)


def _amended_fields(facts: pd.DataFrame, accession: str, filed: pd.Timestamp) -> list[str]:
    """Sorted fields this amendment changed, by value on `_VALUE_KEY`, against everything filed before it."""
    amendment = facts[facts["accession_number"] == accession]
    prior = facts[facts["filing_date"] < filed]
    if prior.empty:
        return sorted(set(amendment["field"]))
    latest = prior.sort_values("filing_date").drop_duplicates(subset=list(_VALUE_KEY), keep="last").set_index(list(_VALUE_KEY))["value"]
    moved: set[str] = set()
    for row in amendment.itertuples():
        key = tuple(getattr(row, c) for c in _VALUE_KEY)
        if key not in latest.index or not _same_value(latest.loc[key], row.value):
            moved.add(str(row.field))
    return sorted(moved)


def publication_events(facts: pd.DataFrame) -> pd.DataFrame:
    """One row per `(date)` on which this ticker made new information public.

    Returns `as_of`, `publication_form`, `is_amendment`, `amended_fiscal_end` and
    `amended_fields` -- the four provenance columns, already collapsed to the day.
    """
    if facts.empty:
        return pd.DataFrame(columns=["as_of", *HISTORY_PROVENANCE])
    first_by_period = facts.groupby("period_of_report", dropna=True)["filing_date"].min()
    rows = []
    for accession, group in facts.groupby("accession_number", sort=False):
        filed = pd.Timestamp(group["filing_date"].iloc[0])
        form = str(group["form"].iloc[0])
        period = group["period_of_report"].iloc[0]
        if not bool(group["is_amendment"].iloc[0]):
            rows.append({"as_of": filed, "publication_form": form, "is_amendment": False, "amended_fiscal_end": pd.NaT, "amended_fields": None})
            continue
        moved = _amended_fields(facts, str(accession), filed)
        if not moved:
            continue  # a no-op amendment publishes nothing
        original = first_by_period.get(period)
        if original is not None and pd.notna(original) and pd.notna(period) and (filed - pd.Timestamp(original)).days > MAX_AMENDMENT_LAG_DAYS:
            continue  # too late to move a published TTM
        rows.append(
            {
                "as_of": filed,
                "publication_form": form,
                "is_amendment": True,
                "amended_fiscal_end": pd.to_datetime(period, errors="coerce"),
                "amended_fields": ",".join(moved),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["as_of", *HISTORY_PROVENANCE])
    return _collapse_same_day(pd.DataFrame(rows))


def _collapse_same_day(events: pd.DataFrame) -> pd.DataFrame:
    """Rule 5: one event per day; provenance resolves by `FORM_PRECEDENCE`, amended fields are unioned."""
    rank = {form: i for i, form in enumerate(FORM_PRECEDENCE)}
    events = events.assign(_rank=[rank.get(f, len(rank)) for f in events["publication_form"]])
    out = []
    for as_of, group in events.sort_values("_rank").groupby("as_of", sort=True):
        fields = sorted({f for csv in group["amended_fields"].dropna() for f in str(csv).split(",") if f})
        out.append(
            {
                "as_of": as_of,
                "publication_form": group["publication_form"].iloc[0],
                "is_amendment": bool(group["is_amendment"].any()),
                "amended_fiscal_end": group["amended_fiscal_end"].max(),
                "amended_fields": ",".join(fields) or None,
            }
        )
    return pd.DataFrame(out).sort_values("as_of").reset_index(drop=True)


# ------------------------------------------------------------------- the snapshot ---


def carry_latest_known(facts: pd.DataFrame, ends, field: str, on: str = "period_end") -> pd.DataFrame:
    """`field`'s latest known value at each date in `ends` (backward as-of), as a frame of `on` and `field`.

    Ties on `on` keep the latest `filing_date`. The frame-at-a-time oracle that `periods.InstantLookup` is
    tested against.
    """
    # `merge_asof` refuses mixed datetime resolutions, so both sides are forced to nanoseconds.
    index = pd.DatetimeIndex(pd.to_datetime(ends)).astype("datetime64[ns]").sort_values()
    rows = facts[facts["field"] == field]
    out = pd.DataFrame({on: index})
    if rows.empty:
        out[field] = pd.NA
        return out
    ordered = (
        rows.assign(**{on: pd.to_datetime(rows[on]).astype("datetime64[ns]")})
        .sort_values([on, "filing_date"])
        .drop_duplicates(subset=[on], keep="last")[[on, "value"]]
        .rename(columns={"value": field})
        .dropna(subset=[on])
    )
    if ordered.empty:
        out[field] = pd.NA
        return out
    return pd.merge_asof(out, ordered.sort_values(on), on=on, direction="backward")


#: Max days between a TTM window's end and the row's `fiscal_end`: half a quarter, so only the SAME fiscal
#: quarter is admitted (the two dates come from different columns and can differ by days).
TTM_STALENESS_DAYS = 45


def _latest(frame: pd.DataFrame, field: str, column: str = "period_end") -> pd.Series | None:
    """The row of `frame` (a `build_periods` output) with the newest `column` for `field`, or None."""
    if frame is None or frame.empty or "field" not in frame.columns:
        return None
    rows = frame[frame["field"] == field]
    if rows.empty:
        return None
    dates = pd.to_datetime(rows[column])
    return rows.iloc[int(dates.to_numpy().argmax())]


def _is_stale(newest: pd.Series, period: pd.Timestamp) -> bool:
    """Whether this TTM window ends more than `TTM_STALENESS_DAYS` from `period`; False when `period` is unknown."""
    if pd.isna(period):
        return False
    end = pd.to_datetime(cast(Any, newest.get("period_end")), errors="coerce")
    return pd.notna(end) and abs((period - end).days) > TTM_STALENESS_DAYS


def _split_by_field(visible: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """`visible` split into one frame per field, each kept in `filing_date` order (callers rely on `.iloc[-1]`)."""
    return {str(field): group for field, group in visible.groupby("field", sort=False)}


def _facts_code(by_field: dict[str, pd.DataFrame], field: str) -> str | None:
    """The facts layer's `dc_code` for this field, preferring the latest filing that mentions it.

    `not_disclosed` when no visible filing mentions the field; None when none carries a code.
    """
    rows = by_field.get(field)
    if rows is None:
        return rc.NOT_DISCLOSED
    latest = rows[rows["filing_date"] == rows["filing_date"].max()]
    coded = latest["dc_code"].dropna()
    if not coded.empty:
        return str(coded.iloc[0])
    coded = rows["dc_code"].dropna()
    return str(coded.iloc[-1]) if not coded.empty else None


def _deduced_nci(by_field: dict[str, pd.DataFrame]) -> float | None:
    """NCI deduced as equity-incl-NCI minus equity-ex-NCI at the latest `period_end` tagged on both bases, or None.

    Used only inside `_total_liabilities_identity`; never written into the as-filed `minorityInterest` column.
    """
    equity = by_field.get("stockholdersEquity")
    if equity is None:
        return None
    rows = equity[equity["value"].notna()]
    if rows.empty:
        return None
    concepts = rows["source_concept"].fillna("").astype(str)
    incl = rows[concepts.str.contains(_EQUITY_INCL_NCI, regex=False)]
    ex = rows[~concepts.str.contains(_EQUITY_INCL_NCI, regex=False)]
    if incl.empty or ex.empty:
        return None
    # One value per period_end per basis, then the latest end carrying both.
    incl_by_end = incl.groupby("period_end")["value"].last()
    ex_by_end = ex.groupby("period_end")["value"].last()
    shared = incl_by_end.index.intersection(ex_by_end.index)
    if shared.empty:
        return None
    latest = max(shared)
    return float(incl_by_end.loc[latest]) - float(ex_by_end.loc[latest])


def _has_valued_fact(by_field: dict[str, pd.DataFrame], field: str) -> bool:
    """Whether any visible fact carries a value for this field ("found but unusable" vs `not_disclosed`)."""
    rows = by_field.get(field)
    return rows is not None and bool(rows["value"].notna().any())


def _qualifiers(by_field: dict[str, pd.DataFrame], field: str) -> list[str]:
    """Qualifier codes for a present value off its nominal basis, from the latest filing touching the field.

    Sources: a `dc_code` in `rc.IS_QUALIFIER`, and the `adjustment` JSON's `basis_qualifier` / `zero_only_retained`.
    """
    rows = by_field.get(field)
    if rows is None:
        return []
    filed = rows["filing_date"].to_numpy("datetime64[ns]")
    known = filed[~np.isnat(filed)]
    if known.size == 0:
        return []
    latest = filed == known.max()
    codes = rows["dc_code"].to_numpy(dtype=object)[latest]
    blobs = rows["adjustment"].to_numpy(dtype=object)[latest]
    found = {str(c) for c in codes[pd.notna(codes)] if str(c) in rc.IS_QUALIFIER}
    for blob in blobs[pd.notna(blobs)]:
        try:
            parsed = json.loads(blob)
        except (TypeError, ValueError):
            continue
        if parsed.get("basis_qualifier"):
            found.add(str(parsed["basis_qualifier"]))
        if parsed.get("zero_only_retained"):
            found.add(rc.ZERO_ONLY_RETAINED)
    return sorted(found)


#: Equity concept that already includes NCI; adding `minorityInterest` to it would double-count.
_EQUITY_INCL_NCI = "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"


#: Relative error above which the filer's own tags contradict `grossProfit = totalRevenue - costOfRevenue`.
GROSS_PROFIT_IDENTITY_TOLERANCE = 0.01


def _contradicts_gross_profit(visible: pd.DataFrame) -> bool:
    """Whether any visible window tags all three gross-profit terms and breaks the identity beyond tolerance.

    Point-in-time and per filer (reads only `visible`), joined on `(period_end, duration_type)`, not fiscal labels.
    """
    wanted = ("grossProfit", "totalRevenue", "costOfRevenue")
    rows = visible[visible["field"].isin(wanted) & visible["value"].notna()]
    if rows.empty:
        return False
    wide = rows.pivot_table(index=["period_end", "duration_type"], columns="field", values="value", aggfunc="max")
    if not all(field in wide.columns for field in wanted):
        return False
    wide = wide.dropna(subset=list(wanted))
    wide = wide[wide["grossProfit"] != 0]
    if wide.empty:
        return False
    error = (wide["totalRevenue"] - wide["costOfRevenue"] - wide["grossProfit"]).abs() / wide["grossProfit"].abs()
    return bool((error > GROSS_PROFIT_IDENTITY_TOLERANCE).any())


def _gross_profit_identity(row: dict, visible: pd.DataFrame) -> float | None:
    """`grossProfit` as TTM `totalRevenue - costOfRevenue` (the catalogue's `derived_fallback`), where no tag gave it.

    Computed here, not in the as-filed facts layer. None when either input is missing or
    `_contradicts_gross_profit` says the filer's own tags disagree.
    """
    revenue, cost = row.get("totalRevenue"), row.get("costOfRevenue")
    if revenue is None or cost is None or _contradicts_gross_profit(visible):
        return None
    return float(revenue) - float(cost)


def _total_liabilities_identity(row: dict, by_field: dict[str, pd.DataFrame]) -> tuple[float | None, str | None]:
    """`totalLiabilities` as `totalAssets - equity (incl. NCI)`, where no filer tag gave it; returns `(value, code)`.

    Computed here, never in the as-filed facts layer, and coded `derived_identity`. Unless the equity row's latest
    concept already includes NCI, adds `minorityInterest`, else `_deduced_nci`; with neither, NCI is assumed zero
    (`derived_identity_nci_zero`) only if no visible fact ever tagged an NCI value, otherwise `(None, None)`.
    """
    assets, equity = row.get("totalAssets"), row.get("stockholdersEquity")
    if assets is None or equity is None:
        return None, None
    rows = by_field.get("stockholdersEquity")
    concepts = rows[rows["value"].notna()]["source_concept"].dropna() if rows is not None else pd.Series(dtype=object)
    incl_nci = bool(len(concepts)) and _EQUITY_INCL_NCI in str(concepts.iloc[-1])
    basis = rc.DERIVED_IDENTITY
    if not incl_nci:
        nci = row.get("minorityInterest")
        if nci is None:
            # A deduction from two filed facts keeps the plain `derived_identity` code.
            nci = _deduced_nci(by_field)
        if nci is None:
            # Point-in-time: zero is assumed only while no visible fact has tagged an NCI value.
            if _has_valued_fact(by_field, "minorityInterest"):
                return None, None
            basis = rc.DERIVED_IDENTITY_NCI_ZERO
        else:
            equity = equity + nci
    return float(assets) - float(equity), basis


def _as_datetime(column: pd.Series) -> pd.Series:
    """`column` as timestamps, converting only when it is not already datetime."""
    return column if is_datetime64_any_dtype(column) else pd.to_datetime(column, errors="coerce")


def _latest_period_known(visible: pd.DataFrame, as_of: pd.Timestamp) -> pd.Timestamp:
    """`fiscal_end`: the latest `period_of_report` (else `period_end`) visible by `as_of`, capped at `as_of`.

    The cap prevents look-ahead from a filing dated before its period closes. Monotone non-decreasing over
    events; an amendment's restated period goes in `amended_fiscal_end` instead.
    """
    periods = _as_datetime(visible["period_of_report"])
    if periods.isna().all():
        periods = _as_datetime(visible["period_end"])
    known = periods[periods <= as_of]
    return cast(pd.Timestamp, known.max() if not known.empty else pd.NaT)


def _instant(lookup: InstantLookup, field: str, period) -> float | None:
    """A balance-sheet level's latest known value as of `period` (carried forward), None only if it has none.

    Equivalent to `carry_latest_known`, pinned by `test_instant_lookup_matches_merge_asof`.
    """
    return lookup.value(field, period)


def _ratio(column: str, numerator, denominator):
    """A `_FORMULAS` value, or None for a missing input or a zero `_RATIOS` denominator."""
    if numerator is None or denominator is None:
        return None
    if pd.isna(numerator) or pd.isna(denominator):
        return None
    if column in _RATIOS and float(denominator) == 0.0:
        return None
    return float(_FORMULAS[column][1](float(numerator), float(denominator)))


def _snapshot(
    ticker: str, visible: pd.DataFrame, event: pd.Series, catalogue: Catalogue, guards: PeriodGuards, narrow: pd.DataFrame | None = None
) -> tuple[dict, list[dict]]:
    """One complete history row plus its reason codes, from `visible` (every fact filed on or before `as_of`).

    `narrow` is the same rows projected to `PERIOD_COLUMNS` for `build_periods`.
    """
    refusals: list[dict] = []
    as_of = pd.Timestamp(event["as_of"])
    facts = narrow if narrow is not None else visible
    by_field = _split_by_field(visible)
    # The filer's own year ends as known at THIS event, shared with `build_periods`.
    year_ends = fiscal_year_ends(facts)
    quarters, ttm, instants = build_periods(facts, catalogue, guards, refusals, year_ends=year_ends)
    lookup = InstantLookup(instants)
    period = _latest_period_known(visible, as_of)
    # `visible` is a filing-date-sorted prefix, so the last non-null regime is the latest-filed one.
    regime = visible["regime"].dropna()
    regime = str(regime.iloc[-1]) if not regime.empty else None

    quarter = fiscal_quarter_of_end(period, year_ends)

    row: dict = {
        "ticker": ticker,
        "as_of": event["as_of"],
        "fiscal_end": period,
        "fiscal_quarter": quarter,
        HISTORY_REGIME: regime,
        **{c: event[c] for c in HISTORY_PROVENANCE},
    }
    codes: list[dict] = []

    def code(field: str, dc_code: str) -> None:
        codes.append(
            {
                "ticker": ticker,
                "as_of": event["as_of"],
                "field": field,
                "dc_code": dc_code,
                "combined_into": catalogue.combined_into(regime, ticker, field),
                # Payload of `failed_hard_guard` alone; `_hard_guard` fills it in.
                "rejected_value": None,
            }
        )

    for field in catalogue.history_fields:
        if field in _FORMULAS:
            continue  # computed once the inputs are in
        if catalogue.field(field).kind == INSTANT:
            # Aligned on `as_of`, not `fiscal_end`: the cover-page share count is dated after the period end.
            value, reason = _instant(lookup, field, as_of), None
        else:
            # The newest TTM must be this row's own quarter; an older window is never carried forward.
            newest = _latest(ttm, field)
            if newest is not None and _is_stale(newest, period):
                newest = None
                reason = rc.STALE_TTM
            else:
                reason = str(newest["dc_code"]) if newest is not None and pd.notna(newest.get("dc_code")) else None
            value = None if newest is None or pd.isna(newest["value"]) else float(newest["value"])
        if value is None and reason is None:
            reason = _facts_code(by_field, field)
        if value is None and reason is None and _has_valued_fact(by_field, field):
            # Disclosed but no full four-quarter window yet; guard refusals already carry their own code.
            reason = rc.INSUFFICIENT_QUARTERS
        row[field], gated = _gate(catalogue, regime, field, value)
        if row[field] is None:
            code(field, gated or reason or rc.NOT_DISCLOSED)
        else:
            for qualifier in _qualifiers(by_field, field):
                code(field, qualifier)
        _break_code(catalogue, field, period, code)

    # After the field loop, so `minorityInterest` (resolved after `totalLiabilities`) is available.
    if row.get("totalLiabilities") is None:
        row["totalLiabilities"], basis = _total_liabilities_identity(row, by_field)
        if row["totalLiabilities"] is not None:
            assert basis is not None
            # Replace the absence code with the derivation code (qualifiers stay).
            codes[:] = [c for c in codes if c["field"] != "totalLiabilities" or c["dc_code"] in rc.IS_QUALIFIER]
            code("totalLiabilities", basis)

    # Before `_FORMULAS`, so `grossMargins` can use it; `operatingIncome`'s `derived_fallback` is deliberately unwired.
    if row.get("grossProfit") is None:
        row["grossProfit"] = _gross_profit_identity(row, visible)
        if row["grossProfit"] is not None:
            # Replace the absence code with the derivation code (qualifiers stay).
            codes[:] = [c for c in codes if c["field"] != "grossProfit" or c["dc_code"] in rc.IS_QUALIFIER]
            code("grossProfit", rc.DERIVED_FALLBACK)

    for column, (inputs, _) in _FORMULAS.items():
        row[column] = _ratio(column, *(row.get(name) for name in inputs))
        if row[column] is None:
            missing = next((n for n in inputs if row.get(n) is None), None)
            code(column, next((c["dc_code"] for c in codes if c["field"] == missing), rc.NOT_DISCLOSED))

    for column, source in _QUARTER_LABEL_COLUMNS.items():
        newest = _latest(quarters, source)
        row[column] = None if newest is None or pd.isna(newest["value"]) else float(newest["value"])
        if row[column] is None:
            code(column, _facts_code(by_field, source) or rc.NOT_DISCLOSED)

    for refusal in refusals:
        code(refusal["field"], str(refusal["dc_code"]))
    # Last, so it also covers derived and computed columns.
    _hard_guard(ticker, event["as_of"], row, codes)
    return row, codes


def _hard_guard(ticker: str, as_of, row: dict, codes: list[dict]) -> None:
    """Null every `HARD_GUARDS` violation in `row` in place, before the write.

    The field's existing codes are replaced by one `failed_hard_guard` row carrying the refused number as
    `rejected_value` (a derived cell has no fact row to recover it from).
    """
    for field, is_impossible in HARD_GUARDS.items():
        value = row.get(field)
        if value is None or pd.isna(value) or not is_impossible(float(value)):
            continue
        row[field] = None
        codes[:] = [c for c in codes if c["field"] != field]
        codes.append(
            {"ticker": ticker, "as_of": as_of, "field": field, "dc_code": rc.FAILED_HARD_GUARD, "combined_into": None, "rejected_value": float(value)}
        )


def _gate(catalogue: Catalogue, regime: str | None, field: str, value: float | None) -> tuple[float | None, str | None]:
    """Regime gating (history layer only): returns `(value, code)`.

    Where the field is `expected_absent` for the regime, a `regime_gated` field is dropped with
    `not_applicable_for_regime`; otherwise the value is kept and only its absence gets that code.
    """
    if not regime or not catalogue.expected_absent(regime, field):
        return value, None
    if catalogue.field(field).regime_gated:
        return None, rc.NOT_APPLICABLE_FOR_REGIME
    return value, (None if value is not None else rc.NOT_APPLICABLE_FOR_REGIME)


def _break_code(catalogue: Catalogue, field: str, period, code) -> None:
    """Code `regime_break` when the field's definitional break date falls in the trailing year ending at `period`."""
    effective = catalogue.regime_break_effective(field)
    if effective is None or pd.isna(period):
        return
    if pd.Timestamp(period) - pd.Timedelta(days=365) < effective <= pd.Timestamp(period):
        code(field, rc.REGIME_BREAK)


# ---------------------------------------------------------------------- entry point ---


def build_ticker_history(ticker: str, facts, *, catalogue: Catalogue | None = None, guards: PeriodGuards | None = None) -> pd.DataFrame:
    """One ticker's `fundamentals_history_sec` frame: 69 columns, one row per publication event.

    `facts` is the ticker's `fundamentals_facts` rows; a `companyfacts` mapping is accepted for synthetic fixtures only.
    """
    return build_ticker(ticker, facts, catalogue=catalogue, guards=guards).history


def build_ticker(
    ticker: str, facts, *, catalogue: Catalogue | None = None, guards: PeriodGuards | None = None, after: pd.Timestamp | None = None
) -> TickerHistory:
    """`build_ticker_history` plus the dense reason-code side table.

    The facts frame is loaded once and sliced in memory; every event rebuilds its whole snapshot from facts with
    `filing_date <= as_of`, so no-leakage follows from the algorithm. `after` keeps only the events with
    `as_of > after`; each snapshot depends on its own prefix only, so those rows equal a full replay's. Asserts the
    69-column contract, known codes only, and the grain (`_assert_grain`).
    """
    catalogue = catalogue or load_catalogue()
    guards = guards or load_guards()
    frame = _normalise_facts(facts, catalogue)
    columns = catalogue.history_columns
    assert len(columns) == 69, f"the column contract is {len(columns)}, not 69"
    events = publication_events(frame)
    if after is not None and not events.empty:
        events = events[pd.to_datetime(events["as_of"]) > pd.Timestamp(after)].reset_index(drop=True)
    if events.empty:
        return TickerHistory(pd.DataFrame(columns=columns), pd.DataFrame(columns=list(_CODE_COLUMNS)))

    narrow = _period_projection(frame)
    filed = frame["filing_date"].to_numpy()
    rows, codes = [], []
    for _, event in events.iterrows():
        # `frame` is sorted by `filing_date`, so "filed on or before as_of" is a positional prefix.
        upto = int(filed.searchsorted(event["as_of"].to_datetime64(), side="right"))
        row, row_codes = _snapshot(ticker, frame.iloc[:upto], event, catalogue, guards, narrow.iloc[:upto])
        rows.append(row)
        codes.extend(row_codes)

    # Pinned dtypes (all-null columns too) for `diff_against_stored` and cold-table type inference.
    history = pin_dtypes(
        pd.DataFrame(rows).reindex(columns=columns),
        dates=("as_of", "fiscal_end", "amended_fiscal_end"),
        floats=[column for column in columns if column not in (*HISTORY_KEYS, HISTORY_REGIME, *HISTORY_PROVENANCE)],
        texts=("publication_form", "amended_fields", HISTORY_REGIME),
    )
    # Nullable Int64: a ticker with no fiscal calendar yet keeps a NULL quarter, not 0.
    history["fiscal_quarter"] = history["fiscal_quarter"].astype("Int64")
    history["is_amendment"] = history["is_amendment"].astype(bool)
    reason = pd.DataFrame(codes, columns=list(_CODE_COLUMNS)).drop_duplicates(subset=["ticker", "as_of", "field", "dc_code"])
    reason = pin_dtypes(reason, floats=("rejected_value",))
    unknown = sorted(set(reason["dc_code"]) - rc.ALL_CODES)
    assert not unknown, f"{ticker}: reason code(s) outside the declared set: {unknown}"
    _assert_grain(ticker, history)
    return TickerHistory(history, reason)


#: `fundamentals_reason_codes` grain `(ticker, as_of, field, dc_code)`, then its two non-key payloads.
_CODE_COLUMNS: tuple[str, ...] = ("ticker", "as_of", "field", "dc_code", "combined_into", "rejected_value")


def _assert_grain(ticker: str, history: pd.DataFrame) -> None:
    """Assert the grain: unique `(ticker, as_of)`, monotone `fiscal_end`, and `as_of >= fiscal_end` (no look-ahead)."""
    assert not history.duplicated(["ticker", "as_of"]).any(), f"{ticker}: two rows share an (ticker, as_of) -- the same-day collapse failed"
    ends = pd.to_datetime(history["fiscal_end"])
    assert (ends.diff().dropna() >= pd.Timedelta(0)).all(), f"{ticker}: fiscal_end is not monotone non-decreasing in as_of"
    lag = (pd.to_datetime(history["as_of"]) - ends).dt.days.dropna()
    assert (lag >= 0).all(), f"{ticker}: as_of precedes fiscal_end -- look-ahead leak"


#: The only columns `build_periods` reads; the replay projects to them because every engine filter copies its frame.
PERIOD_COLUMNS: tuple[str, ...] = (
    "ticker",
    "field",
    "duration_type",
    "period_start",
    "period_end",
    "period_days",
    "value",
    "filing_date",
    "source_concept",
    "fiscal_year",
    "fiscal_period",
)


#: Duration types that are a filing's own reporting period (a 10-Q's quarter, a 10-K's year).
OWN_PERIOD_DURATIONS = ("quarterly", "annual")
#: Days a filing's own duration may end after its stated period of report before the header is taken as wrong.
STATED_PERIOD_TOLERANCE_DAYS = 7


def _period_projection(frame: pd.DataFrame) -> pd.DataFrame:
    """`frame` reduced to `PERIOD_COLUMNS`, with string columns cast to object (cheap per-slice takes, unlike Arrow)."""
    out = frame[[c for c in PERIOD_COLUMNS if c in frame.columns]].copy()
    for column in ("field", "duration_type", "source_concept", "fiscal_period"):
        if column in out.columns and not pd.api.types.is_object_dtype(out[column]):
            out[column] = out[column].astype(object)
    return out


def _normalise_facts(facts, catalogue: Catalogue) -> pd.DataFrame:
    """The facts frame with timestamp dates, defaults for columns a fixture may lack, sorted by `filing_date`."""
    if not isinstance(facts, pd.DataFrame):
        facts = facts_frame_from_companyfacts(facts, catalogue)
    out = facts.copy()
    for column in ("filing_date", "period_of_report", "period_start", "period_end"):
        out[column] = pd.to_datetime(cast(Any, out.get(column)), errors="coerce")
    for column, default in (
        ("is_amendment", False),
        ("dc_code", None),
        ("adjustment", None),
        ("regime", None),
        ("form", "10-Q"),
        ("accession_number", ""),
        ("source_concept", None),
    ):
        if column not in out.columns:
            out[column] = default
    out["is_amendment"] = out["is_amendment"].fillna(False).astype(bool)
    if "period_of_report" in out and out["period_of_report"].isna().all():
        out["period_of_report"] = out["period_end"]
    return _own_period_of_report(out).sort_values("filing_date")


def _own_period_of_report(facts: pd.DataFrame) -> pd.DataFrame:
    """A filing whose quarterly or annual duration ends after its stated `period_of_report` takes that end.

    The SEC header's period is filer-typed and can name the prior fiscal year end on a 10-Q; the filing's own
    tagged duration is the period it reports. Only durations that ended by the filing date count: a
    forward-tagged context is not the period.
    """
    if "duration_type" not in facts.columns:
        return facts
    own = facts["duration_type"].isin(OWN_PERIOD_DURATIONS) & (facts["period_end"] <= facts["filing_date"])
    latest = facts["period_end"].where(own).groupby(facts["accession_number"]).transform("max")
    stale = latest > facts["period_of_report"] + pd.Timedelta(days=STATED_PERIOD_TOLERANCE_DAYS)
    facts.loc[stale, "period_of_report"] = latest[stale]
    return facts


def _companyfacts_rows(concept: str, payload: dict, field: str) -> list[dict]:
    """One `fundamentals_facts`-shaped row per companyfacts entry of `concept`, across all its units."""
    return [
        {
            "ticker": "FIXTURE",
            "accession_number": f"{concept}-{unit}-{i}",
            "field": field,
            "fiscal_year": pd.Timestamp(entry["end"]).year,
            "fiscal_period": entry.get("fp", "NA"),
            "form": entry.get("form", "10-Q"),
            "filing_date": entry.get("filed"),
            "is_amendment": False,
            "period_of_report": entry["end"],
            "regime": None,
            "period_start": entry.get("start"),
            "period_end": entry["end"],
            "value": entry.get("val"),
            "unit": unit,
            "source_concept": concept,
            "dc_code": None,
            "adjustment": None,
        }
        for unit, entries in (payload.get("units") or {}).items()
        for i, entry in enumerate(entries)
    ]


def facts_frame_from_companyfacts(blob: dict, catalogue: Catalogue) -> pd.DataFrame:
    """A `fundamentals_facts`-shaped frame from a raw `companyfacts` mapping, for synthetic fixtures ONLY.

    Never on the production path (companyfacts drops extension and dimensioned facts). Concepts map to fields via
    the catalogue's own declared concepts; undeclared concepts are skipped.
    """
    by_concept: dict[str, str] = {}
    for name in catalogue.extracted_fields:
        spec = catalogue.field(name)
        for concept in [spec.total_concept(), *spec.fallback_concepts()]:
            if concept:
                by_concept.setdefault(concept.split(":")[-1], name)
    rows = []
    for concepts in (blob.get("facts") or {}).values():
        for concept, payload in concepts.items():
            field = by_concept.get(concept.split(":")[-1])
            if field is not None:
                rows.extend(_companyfacts_rows(concept, payload, field))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # One accession per (filing date, form), so the event ladder sees filings and not facts.
    frame["accession_number"] = frame["filing_date"].astype(str) + "-" + frame["form"].astype(str)
    frame["period_days"] = (pd.to_datetime(frame["period_end"]) - pd.to_datetime(frame["period_start"])).dt.days
    frame["duration_type"] = [INSTANT if pd.isna(d) else ("annual" if d > 300 else "quarterly") for d in frame["period_days"]]
    return frame


# ------------------------------------------------------------------ the seam rule ---


def keep_window_owner_filings(facts: pd.DataFrame, windows: Sequence[CikWindow]) -> pd.DataFrame:
    """Rule 6: a filing is kept only when its filer CIK's seam-widened window admits its filing date (so a CIK with no
    window, such as an acquired target, contributes nothing), and across a seam rows of a fiscal period reported by
    several CIKs keep only the CIK whose stated window owns the period end.

    A period no other CIK reports (a margin filing alone in its period) is kept, as is a row with no CIK, date or period;
    with no window at all the facts are returned unchanged.
    """
    if not windows or facts.empty or "cik" not in facts.columns:
        return facts
    facts = facts[pd.Series(_filed_inside_window(facts, windows), index=facts.index, dtype=bool)]
    if len(windows) < 2:
        return facts
    ciks = pad_cik_series(facts["cik"]).tolist()
    periods = [None if pd.isna(day) else pd.Timestamp(day) for day in pd.to_datetime(facts["period_of_report"], errors="coerce")]
    reported = set(zip(periods, ciks, strict=True))
    owner = {day: next((window.cik for window in windows if window.owns(day)), None) for day in set(periods) if day is not None}
    drop = [
        day is not None and (holder := owner[day]) is not None and cik != holder and (day, holder) in reported
        for day, cik in zip(periods, ciks, strict=True)
    ]
    return facts[pd.Series([not dropped for dropped in drop], index=facts.index, dtype=bool)]


def _filed_inside_window(facts: pd.DataFrame, windows: Sequence[CikWindow]) -> list[bool]:
    """Per row: whether its filer CIK's seam-widened window admits its filing date (the `consolidating` policy)."""
    if "filing_date" not in facts.columns:
        return [True] * len(facts)
    filed = pd.to_datetime(facts["filing_date"], errors="coerce")
    return [
        not cik or pd.isna(day) or any(window.cik == cik and window.admits(pd.Timestamp(day)) for window in windows)
        for cik, day in zip(pad_cik_series(facts["cik"]), filed, strict=True)
    ]


def drop_reverse_acquisition_comparatives(facts: pd.DataFrame, seam: pd.Timestamp) -> pd.DataFrame:
    """Facts of a reverse acquisition's survivor minus the pre-seam periods its post-seam filings restate.

    A fact whose `period_end` is before `seam` in a filing whose `period_of_report` is on or after it is the
    accounting acquirer's comparative, not the traded security's history. The survivor's own amendment of a
    pre-seam period has a pre-seam `period_of_report` and is kept; a row with no date is kept.
    """
    if facts.empty:
        return facts
    ends = pd.to_datetime(facts["period_end"], errors="coerce")
    reports = pd.to_datetime(facts["period_of_report"], errors="coerce")
    return facts[~((ends < seam) & (reports >= seam))]


def _filer_count(facts: pd.DataFrame) -> int:
    """Distinct filer CIKs in a ticker's facts (0 when the projection has no `cik`)."""
    return int(pad_cik_series(facts["cik"].dropna()).nunique()) if "cik" in facts.columns else 0


# ------------------------------------------------------------------- immutability ---

#: The `fundamentals_facts` columns the replay reads (projected read, one ticker at a time).
FACT_COLUMNS: tuple[str, ...] = (
    "ticker",
    "cik",
    "accession_number",
    "field",
    "fiscal_year",
    "fiscal_period",
    "duration_type",
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
    "source_concept",
    "dc_code",
    "adjustment",
)


def diff_against_stored(stored: pd.DataFrame, rebuilt: pd.DataFrame) -> pd.DataFrame:
    """Every cell of an already-stored row that a rebuild would change, as `as_of, column, stored, rebuilt`.

    Enforces immutability, since `store.save` upserts on `(ticker, as_of)`. Compared exactly, with no tolerance
    (DOUBLE PRECISION round-trips bit for bit); NaN on both sides counts as equal.
    """
    if stored is None or stored.empty or rebuilt.empty:
        return pd.DataFrame(columns=["as_of", "column", "stored", "rebuilt"])
    # Postgres DATE columns return as `datetime.date`, which never equals a `Timestamp`.
    left = _keyed_by_as_of(stored)
    right = _keyed_by_as_of(rebuilt)
    shared_rows = left.index.intersection(right.index)
    shared_cols = [c for c in left.columns if c in right.columns and c != "ticker"]
    rows = []
    for as_of in shared_rows:
        for column in shared_cols:
            was, now = left.at[as_of, column], right.at[as_of, column]
            if pd.isna(was) and pd.isna(now):
                continue
            if pd.isna(was) or pd.isna(now) or was != now:
                rows.append({"as_of": as_of, "column": column, "stored": was, "rebuilt": now})
    return pd.DataFrame(rows, columns=["as_of", "column", "stored", "rebuilt"])


def _keyed_by_as_of(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ("as_of", "fiscal_end", "amended_fiscal_end"):
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    return out.set_index("as_of").sort_index()


def _unpublished_events(context, ticker: str, stored: pd.DataFrame, history: pd.DataFrame, codes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The rebuilt `(history, codes)` rows whose `as_of` is not stored yet.

    Raises ValueError, after logging the diff, if a stored row would change (history is append-only).
    """
    drift = diff_against_stored(stored, history)
    if not drift.empty:
        context.log.error(
            "history: %s would CHANGE %d already-published cell(s) "
            "across %d row(s) -- refusing to overwrite. Re-run with "
            "--rebuild-history to accept:\n%s",
            ticker,
            len(drift),
            drift["as_of"].nunique(),
            drift.head(20).to_string(),
        )
        raise ValueError(
            f"{ticker}: {len(drift)} stored fundamentals_history_sec cell(s) would change; history is append-only (pass --rebuild-history to rebuild)"
        )
    known = set(pd.to_datetime(stored["as_of"]))
    new = ~pd.to_datetime(history["as_of"]).isin(known)
    return history[new.values], codes[pd.to_datetime(codes["as_of"]).isin(set(history[new.values]["as_of"]))]


def _drop_history(context: Context, ticker: str) -> None:
    """Delete a rebuilt ticker's history and reason codes when its facts now yield no event."""
    deleted = context.store.delete(Tables.fundamentals_history_sec, {"ticker": ticker})
    context.store.delete(Tables.fundamentals_reason_codes, {"ticker": ticker})
    if deleted:
        context.log.warning("history: %s REBUILT -- %d row(s) deleted, no event left in its facts", ticker, deleted)


def _scope_changed_recently(context: Context, tickers: list[str], as_of: pd.Timestamp | None = None) -> frozenset[str]:
    """Tickers whose lineage `scope_changed_at` falls inside `resume.recently_changed`'s window on `as_of` (default today)."""
    identity = load_identity(context)
    stamps = {ticker: identity.filing_scope(ticker).scope_changed_at for ticker in tickers if ticker in identity.roster_cik}
    return frozenset(recently_changed(stamps, as_of))


# ------------------------------------------------------------------------ triage ---

#: Per-ticker history paths chosen by `_history_work`.
FULL, INCREMENTAL, CHECK, SKIP = "full", "incremental", "check", "skip"

#: The `fundamentals_facts` columns the triage reads: one filing date per row, and whether it is an original.
_TRIAGE_COLUMNS: tuple[str, ...] = ("ticker", "filing_date", "is_amendment")


@dataclass(frozen=True)
class HistoryWork:
    """One ticker's history path, its newest stored `as_of`, and its stored event list (`as_of` + provenance)."""

    path: str
    newest: pd.Timestamp | None = None
    stored_events: pd.DataFrame | None = None


def _stored_events(context: Context, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Each ticker's stored `as_of` + provenance rows, sorted by `as_of`; tickers with no history are absent."""
    if not tickers:
        return {}
    stored = context.store.load(
        Tables.fundamentals_history_sec, columns=["ticker", "as_of", *HISTORY_PROVENANCE], where={"ticker": list(tickers)}, optional=True
    )
    if stored is None:
        return {}
    stored = stored.assign(as_of=pd.to_datetime(stored["as_of"]).dt.normalize())
    return {str(ticker): group.drop(columns="ticker").sort_values("as_of").reset_index(drop=True) for ticker, group in stored.groupby("ticker")}


def _filing_dates(context: Context, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Each ticker's distinct `(filing_date, original)` pairs, streamed so no chunk outlives its reduction."""
    if not tickers:
        return {}
    parts = []
    for chunk in context.store.iter_load(Tables.fundamentals_facts, columns=list(_TRIAGE_COLUMNS), where={"ticker": list(tickers)}):
        filed = pd.to_datetime(chunk["filing_date"], errors="coerce").dt.normalize()
        original = ~chunk["is_amendment"].fillna(False).astype(bool)
        parts.append(pd.DataFrame({"ticker": chunk["ticker"].astype(str), "filing_date": filed, "original": original}).dropna().drop_duplicates())
    if not parts:
        return {}
    dates = pd.concat(parts, ignore_index=True).drop_duplicates()
    return {str(ticker): group.drop(columns="ticker") for ticker, group in dates.groupby("ticker")}


def _ticker_path(
    ticker: str, events: pd.DataFrame, filed: pd.DataFrame | None, *, read: Sequence[pd.Timestamp] | None, full_fetch: bool, recent: frozenset[str]
) -> str:
    """The path of a ticker with stored history (see `_history_work`)."""
    newest = pd.Timestamp(events["as_of"].max())
    if filed is None:
        return FULL
    if read is not None and (full_fetch or any(pd.Timestamp(day).normalize() <= newest for day in read)):
        return FULL
    stored = set(events["as_of"])
    upto = filed[filed["filing_date"] <= newest]
    # An original always publishes, so an unstored original or a stored date with no filing means the past moved.
    moved = not stored <= set(upto["filing_date"]) or not set(upto.loc[upto["original"], "filing_date"]) <= stored
    if (filed["filing_date"] > newest).any():
        return INCREMENTAL
    return CHECK if moved or ticker in recent else SKIP


def _history_work(
    context: Context,
    tickers: list[str],
    *,
    fetched: Mapping[str, Sequence[pd.Timestamp]] | None = None,
    full_fetch: bool = False,
    verify: bool = False,
    recent: frozenset[str] = frozenset(),
) -> dict[str, HistoryWork]:
    """Each ticker's history path, from two narrow reads and before any replay.

    `full`: no stored history, `verify`, a same-process fetch (`fetched`) read a filing dated on or before the newest
    stored row or read the ticker under `full_fetch`, or no facts. `incremental`: a filing newer than the newest stored
    row. `check`: lineage-recent (`recent`), or the stored dates and the filing dates disagree up to the newest stored
    row. `skip`: otherwise. `incremental` and `check` compare the rebuilt event list first (`_incremental_rows`).
    """
    stored = _stored_events(context, tickers)
    if verify:
        return {t: HistoryWork(FULL, stored[t]["as_of"].max(), stored[t]) if t in stored else HistoryWork(FULL) for t in tickers}
    dates = _filing_dates(context, sorted(stored))
    work: dict[str, HistoryWork] = {}
    for ticker in tickers:
        events = stored.get(ticker)
        if events is None:
            work[ticker] = HistoryWork(FULL)
            continue
        read = None if fetched is None else fetched.get(ticker)
        path = _ticker_path(ticker, events, dates.get(ticker), read=read, full_fetch=full_fetch, recent=recent)
        work[ticker] = HistoryWork(path, pd.Timestamp(events["as_of"].max()), events)
    return work


def _event_keys(events: pd.DataFrame) -> list[tuple]:
    """`(as_of, *HISTORY_PROVENANCE)` per event, normalised so stored (DB round-trip) and rebuilt frames compare equal."""

    def text(value: Any) -> str | None:
        return None if value is None or pd.isna(value) else str(value)

    def day(value: Any) -> pd.Timestamp | None:
        return None if value is None or pd.isna(value) else pd.Timestamp(value).normalize()

    keys = [
        (
            day(row["as_of"]),
            text(row["publication_form"]),
            bool(row["is_amendment"]) if pd.notna(row["is_amendment"]) else False,
            day(row["amended_fiscal_end"]),
            text(row["amended_fields"]),
        )
        for row in events[["as_of", *HISTORY_PROVENANCE]].to_dict("records")
    ]
    return sorted(keys, key=lambda key: key[0])


def _incremental_rows(ticker: str, df_facts: pd.DataFrame, work: HistoryWork, catalogue: Catalogue, guards: PeriodGuards) -> TickerHistory | None:
    """Rows of the events after `work.newest` only, or None when the rebuilt event list up to it differs from the stored one."""
    assert work.newest is not None and work.stored_events is not None
    events = publication_events(_normalise_facts(df_facts, catalogue))
    dates = pd.to_datetime(events["as_of"])
    if _event_keys(events[dates <= work.newest]) != _event_keys(work.stored_events):
        return None
    if not (dates > work.newest).any():
        return TickerHistory(pd.DataFrame(columns=catalogue.history_columns), pd.DataFrame(columns=list(_CODE_COLUMNS)))
    built = build_ticker(ticker, df_facts, catalogue=catalogue, guards=guards, after=work.newest)
    # Append-only: never upsert onto a stored row, whatever the build returned.
    new = pd.to_datetime(built.history["as_of"]) > work.newest
    return TickerHistory(built.history[new.values], built.reason_codes[pd.to_datetime(built.reason_codes["as_of"]) > work.newest])


# ---------------------------------------------------------------------- the build ---


def _ticker_facts(context: Context, ticker: str, seams: Mapping[str, pd.Timestamp]) -> pd.DataFrame | None:
    """The ticker's replay facts with the seam rule and, for a declared reverse acquisition (`seams`), the comparatives
    rule applied, or None when it has none stored."""
    df_facts = context.store.load(Tables.fundamentals_facts, columns=list(FACT_COLUMNS), where={"ticker": ticker}, optional=True)
    if df_facts is None:
        return df_facts
    if _filer_count(df_facts) > 1:
        df_kept = keep_window_owner_filings(df_facts, load_identity(context).filing_scope(ticker).windows)
        set_aside = sorted(set(df_facts["accession_number"]) - set(df_kept["accession_number"]))
        if set_aside:
            context.log.info(
                "history: %s seam rule set aside %d filing(s) outside the filer's window or of a period its window owner reports: %s",
                ticker,
                len(set_aside),
                ", ".join(set_aside),
            )
        df_facts = df_kept
    if ticker in seams:
        df_kept = drop_reverse_acquisition_comparatives(df_facts, seams[ticker])
        dropped = df_facts.loc[~df_facts.index.isin(df_kept.index), "accession_number"].value_counts().sort_index()
        if not dropped.empty:
            context.log.info(
                "history: %s reverse-acquisition rule dropped %d pre-seam comparative fact(s) of %d post-seam filing(s): %s",
                ticker,
                int(dropped.sum()),
                len(dropped),
                ", ".join(f"{accession} ({count})" for accession, count in dropped.items()),
            )
        df_facts = df_kept
    return df_facts


# ------------------------------------------------------------- the replay pool ---

#: Full builds in flight per pool worker; bounds the fact frames and results the parent holds.
_IN_FLIGHT_PER_WORKER = 2

#: A pool child's catalogue and guards, set once by `_init_replay_worker`.
_WORKER_STATE: dict[str, Any] = {}


def _init_replay_worker(catalogue: Catalogue, guards: PeriodGuards) -> None:
    """Pool initializer: keep the parent's catalogue and guards for every build in this child."""
    _WORKER_STATE["catalogue"] = catalogue
    _WORKER_STATE["guards"] = guards


def _build_in_worker(ticker: str, df_facts: pd.DataFrame) -> TickerHistory:
    """The pure full build of one ticker, run in a pool child (no store, no log)."""
    return build_ticker(ticker, df_facts, catalogue=_WORKER_STATE["catalogue"], guards=_WORKER_STATE["guards"])


def _replay_workers(context: Context) -> int:
    """The full-replay pool size, `data_extract.fundamentals_workers`; 1 when a test double's config lacks it."""
    section = getattr(getattr(context, "config", None), "data_extract", None)
    return max(int(getattr(section, "fundamentals_workers", 1)), 1)


@dataclass
class _Pending:
    """One non-skipped ticker waiting for its in-order save: a full build (future or done), or incremental rows."""

    ticker: str
    full: Future[TickerHistory] | TickerHistory | None = None
    rows: TickerHistory | None = None
    no_facts: bool = False


def _full_rows(
    context: Context,
    ticker: str,
    built: TickerHistory,
    *,
    rebuild_history: bool,
    rebuild: frozenset[str],
    catalogue: Catalogue,
) -> TickerHistory | None:
    """The rows of a full replay (`built`) to save after the drift guard, or None when the facts yield no event."""
    if built.history.empty:
        if ticker in rebuild:
            _drop_history(context, ticker)
        return None
    # Explicit projection so the read fails loudly if the table and the column contract diverge.
    df_stored = context.store.load(Tables.fundamentals_history_sec, columns=list(catalogue.history_columns), where={"ticker": ticker}, optional=True)
    df_history, df_codes = built.history, built.reason_codes
    drifted = ticker in rebuild and df_stored is not None and not diff_against_stored(df_stored, df_history).empty
    if rebuild_history or drifted:
        deleted = context.store.delete(Tables.fundamentals_history_sec, {"ticker": ticker})
        context.store.delete(Tables.fundamentals_reason_codes, {"ticker": ticker})
        context.log.warning(
            "history: %s REBUILT -- %d row(s) deleted and recomputed. "
            "Log this in the phase report: a rebuild re-derives numbers "
            "under whatever model is already trained on them.",
            ticker,
            deleted,
        )
    elif df_stored is not None:
        df_history, df_codes = _unpublished_events(context, ticker, df_stored, df_history, df_codes)
    return TickerHistory(df_history, df_codes)


def build_fundamentals_history(
    context: Context,
    tickers: list[str],
    *,
    rebuild_history: bool = False,
    verify_history: bool = False,
    fetched: Mapping[str, Sequence[pd.Timestamp]] | None = None,
    full_fetch: bool = False,
    as_of: pd.Timestamp | None = None,
) -> None:
    """`fundamentals_facts` -> `fundamentals_history_sec` + `fundamentals_reason_codes`, per ticker.

    Triage first (`_history_work`): a current ticker is skipped without reading its facts, a ticker with newer
    filings snapshots its new events only, and the rest replay in full. Append-only: a full replay saves only new
    `as_of` events, and a stored row that would change raises ValueError after logging the diff.
    `verify_history=True` replays every ticker in full under that guard. `rebuild_history=True` (CLI
    `--rebuild-history`) deletes the ticker's rows from both tables and rebuilds from stored facts, with no network;
    so does a ticker whose lineage scope changed recently (`resume.recently_changed` on `as_of`) when its recomputed
    history differs from the stored one. `fetched` (filing dates a same-process fetch read, per ticker) and
    `full_fetch` (that fetch ran with `-F`) route back-dated reads to the full replay. A ticker whose facts come from
    several CIKs reads its windows from the identity layer for the seam rule (`keep_window_owner_filings`); a declared
    reverse acquisition then drops the pre-seam comparatives of its survivor's post-seam filings.
    Full builds run in a process pool of `data_extract.fundamentals_workers` (in-process for 1 worker or a single
    ticker); the parent reads, guards, saves and logs, in ticker order.
    """
    catalogue = load_catalogue(str(context.config_dir))
    guards = load_guards(str(context.config_dir))
    seams = {entry.ticker: entry.seam_date for entry in load_security_manual(str(context.config_dir)).reverse_acquisitions}
    history_rows = codes_rows = 0
    rebuild = frozenset(tickers) if rebuild_history else _scope_changed_recently(context, tickers, as_of)
    if rebuild and not rebuild_history:
        context.log.info(
            "history: %d ticker lineage scope(s) changed recently -> rebuilt where the history moved: %s", len(rebuild), ", ".join(sorted(rebuild))
        )
    if rebuild_history:
        work = {ticker: HistoryWork(FULL) for ticker in tickers}
    else:
        work = _history_work(context, tickers, fetched=fetched, full_fetch=full_fetch, verify=verify_history, recent=rebuild)
    paths = pd.Series([item.path for item in work.values()], dtype=object).value_counts().to_dict()
    context.log.info(
        "history: triage of %d ticker(s): %d full, %d incremental, %d check, %d current (skipped, no replay)",
        len(tickers),
        paths.get(FULL, 0),
        paths.get(INCREMENTAL, 0),
        paths.get(CHECK, 0),
        paths.get(SKIP, 0),
    )
    active = [ticker for ticker in tickers if work[ticker].path != SKIP]
    workers = _replay_workers(context)
    use_pool = workers > 1 and len(active) > 1
    limit = workers * _IN_FLIGHT_PER_WORKER if use_pool else 1
    executor: ProcessPoolExecutor | None = None
    pending: deque[_Pending] = deque()

    def finish(entry: _Pending) -> None:
        """The parent's in-order tail of one ticker: drift guard, save, log."""
        nonlocal history_rows, codes_rows
        ticker = entry.ticker
        if entry.no_facts:
            context.log.info("history: %s has no stored facts -- skipped", ticker)
            if ticker in rebuild:
                _drop_history(context, ticker)
            return
        built = entry.rows
        if entry.full is not None:
            full = entry.full.result() if isinstance(entry.full, Future) else entry.full
            built = _full_rows(context, ticker, full, rebuild_history=rebuild_history, rebuild=rebuild, catalogue=catalogue)
        if built is None:
            return
        df_history, df_codes = built.history, built.reason_codes
        if df_history.empty:
            context.log.info("history: %s already current (0 new events)", ticker)
            return
        context.store.save(Tables.fundamentals_history_sec, df_history)
        if not df_codes.empty:
            context.store.save(Tables.fundamentals_reason_codes, df_codes)
        context.log.info("history: %s +%d event row(s), %d reason code(s)", ticker, len(df_history), len(df_codes))
        history_rows += len(df_history)
        codes_rows += len(df_codes)

    try:
        for ticker in active:
            item = work[ticker]
            df_facts = _ticker_facts(context, ticker, seams)
            entry = _Pending(ticker, no_facts=df_facts is None)
            if df_facts is not None and item.path in (INCREMENTAL, CHECK):
                entry.rows = _incremental_rows(ticker, df_facts, item, catalogue, guards)
                if entry.rows is None:
                    context.log.warning(
                        "history: %s event list differs from the stored one up to %s -> full replay", ticker, pd.Timestamp(item.newest).date()
                    )
            if df_facts is not None and entry.rows is None:
                if use_pool:
                    if executor is None:
                        executor = ProcessPoolExecutor(max_workers=workers, initializer=_init_replay_worker, initargs=(catalogue, guards))
                    entry.full = executor.submit(_build_in_worker, ticker, df_facts)
                else:
                    entry.full = build_ticker(ticker, df_facts, catalogue=catalogue, guards=guards)
            del df_facts
            pending.append(entry)
            # Saves stay in ticker order; the bound keeps at most `limit` builds (and their facts) in flight.
            while pending and (len(pending) >= limit or not isinstance(pending[0].full, Future) or pending[0].full.done()):
                finish(pending.popleft())
        while pending:
            finish(pending.popleft())
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    context.log.info("history: %d ticker(s), +%d event row(s), %d reason code(s)", len(tickers), history_rows, codes_rows)
