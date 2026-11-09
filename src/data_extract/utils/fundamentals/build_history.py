"""
`sec_fundamentals_facts` -> `sec_fundamentals_history`, on the publication-event grain, from original filings only.

`as_of` is a FILING DATE. Rules: (1) a row for every `(ticker, date)` on which an original 10-K or 10-Q was filed;
amendment facts, if any are stored, are ignored; (2) each row is a complete snapshot of latest-known values, built
only from facts filed on or before `as_of`; (3) stored rows are immutable (`diff_against_stored`); (4) same-day
filings collapse to one row by `(ticker, date)`, a 10-K winning `publication_form`; (5) across a CIK seam one filer
per fiscal period, the CIK whose stated window owns the period end (`keep_window_owner_filings`).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_object_dtype, is_string_dtype

from src.context import Context
from src.data_extract.utils.common.frame_sanitize import pin_dtypes
from src.data_extract.utils.common.identity import CikWindow, load_identity
from src.data_extract.utils.common.resume import recently_changed
from src.data_extract.utils.common.security_lines import load_security_manual
from src.data_extract.utils.fundamentals.kpi_catalogue import HISTORY_KEYS, HISTORY_PROVENANCE, HISTORY_REGIME, Catalogue, load_catalogue
from src.data_extract.utils.fundamentals.periods import (
    INSTANT,
    InstantLookup,
    PeriodGuards,
    field_periods,
    fiscal_quarter_of_end,
    fiscal_year_ends,
    instant_stock,
    is_duration,
    load_guards,
)
from src.data_store.schema import Tables
from src.utils.string import pad_cik_series

#: `{column: predicate meaning the value is IMPOSSIBLE}`, nulled by `_hard_guard` before the row is written.
#: Impossible only, never merely implausible (those are flag-only validator checks). Share counts refuse `<= 0`;
#: `totalAssets` refuses only `< 0`, since a shell can legitimately foot to zero.
HARD_GUARDS: dict[str, Callable[[float], bool]] = {
    "totalAssets": lambda v: v < 0,
    "sharesOutstanding": lambda v: v <= 0,
    "basicShares": lambda v: v <= 0,
    "dilutedShares": lambda v: v <= 0,
}

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


# --------------------------------------------------------------- the events ---


def _originals(facts: pd.DataFrame) -> pd.DataFrame:
    """`facts` without amendment filings (`form` ending `/A`): the history is built from originals only."""
    if facts.empty or "form" not in facts.columns:
        return facts
    return facts[~facts["form"].astype(str).str.endswith("/A")]


def publication_events(facts: pd.DataFrame) -> pd.DataFrame:
    """One row per filing date of this ticker's original filings: `as_of` and `publication_form`.

    Filings sharing a day collapse to one event; a 10-K wins `publication_form` over a 10-Q.
    """
    filings = facts.dropna(subset=["filing_date"]).drop_duplicates("accession_number")
    if filings.empty:
        return pd.DataFrame(columns=["as_of", *HISTORY_PROVENANCE])
    ranked = filings.assign(_quarterly=~filings["form"].astype(str).str.startswith("10-K"))
    events = ranked.sort_values(["filing_date", "_quarterly"], kind="stable").drop_duplicates("filing_date", keep="first")
    return pd.DataFrame({"as_of": pd.to_datetime(events["filing_date"]), "publication_form": events["form"].astype(str)}).reset_index(drop=True)


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


#: The fields whose visible facts the identities read (`_total_liabilities_identity` and its helpers).
_IDENTITY_FIELDS: tuple[str, ...] = ("stockholdersEquity", "minorityInterest")


@dataclass(frozen=True)
class _Visible:
    """What one snapshot reads, all derived from the facts filed on or before its `as_of`."""

    #: Every visible fact, in `filing_date` order.
    facts: pd.DataFrame
    #: The `_IDENTITY_FIELDS` facts, one frame per field in `filing_date` order (callers rely on `.iloc[-1]`).
    by_field: dict[str, pd.DataFrame]
    #: The newest TTM row and the newest discrete-quarter row of every duration field with a quarter.
    ttm: dict[str, pd.Series]
    quarters: dict[str, pd.Series]
    lookup: InstantLookup
    year_ends: list[pd.Timestamp]


class _Replay:
    """One ticker's facts indexed once for its event loop.

    `frame` is sorted by `filing_date`, so an event's visible facts are a positional prefix. A field's quarters and
    TTM depend only on that field's visible duration facts and the fiscal calendar, so they are recomputed only when
    the field gained a fact or the calendar moved, and otherwise reused from the previous event unchanged.
    """

    def __init__(self, frame: pd.DataFrame, catalogue: Catalogue, guards: PeriodGuards) -> None:
        self._frame = frame
        self._narrow = _period_projection(frame)
        self._catalogue = catalogue
        self._guards = guards
        self._filed = frame["filing_date"].to_numpy()
        # Ascending positions per field; a field's duration facts are the subset `build_periods` would group.
        self._positions: dict[str, np.ndarray] = {str(name): positions for name, positions in frame.groupby("field", sort=False).indices.items()}
        durations = is_duration(self._narrow).to_numpy()
        self._durations = {name: positions[durations[positions]] for name, positions in sorted(self._positions.items())}
        self._periods: dict[str, tuple[tuple[int, tuple[pd.Timestamp, ...]], pd.Series | None, pd.Series | None]] = {}

    def at(self, as_of: pd.Timestamp) -> _Visible:
        """The snapshot inputs of the event dated `as_of`."""
        upto = int(self._filed.searchsorted(as_of.to_datetime64(), side="right"))
        narrow = self._narrow.iloc[:upto]
        # The filer's own year ends as known at THIS event, shared by every field.
        year_ends = fiscal_year_ends(narrow)
        calendar = tuple(year_ends)
        ttm, quarters = {}, {}
        for name, positions in self._durations.items():
            count = int(positions.searchsorted(upto))
            if count == 0:
                continue
            key, newest_ttm, newest_quarter = self._periods.get(name, (None, None, None))
            if key != (count, calendar):
                periods = field_periods(self._narrow.iloc[positions[:count]], self._catalogue.field(name), self._guards, year_ends)
                newest_ttm, newest_quarter = (None, None) if periods is None else (_latest(periods[1], name), _latest(periods[0], name))
                self._periods[name] = ((count, calendar), newest_ttm, newest_quarter)
            if newest_ttm is not None:
                ttm[name] = newest_ttm
            if newest_quarter is not None:
                quarters[name] = newest_quarter
        by_field = {}
        for name in _IDENTITY_FIELDS:
            positions = self._positions.get(name)
            count = 0 if positions is None else int(positions.searchsorted(upto))
            if count:
                by_field[name] = self._frame.iloc[positions[:count]]
        return _Visible(self._frame.iloc[:upto], by_field, ttm, quarters, InstantLookup(instant_stock(narrow)), year_ends)


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
    """Whether any visible fact carries a value for this field."""
    rows = by_field.get(field)
    return rows is not None and bool(rows["value"].notna().any())


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


def _total_liabilities_identity(row: dict, by_field: dict[str, pd.DataFrame]) -> float | None:
    """`totalLiabilities` as `totalAssets - equity (incl. NCI)`, where no filer tag gave it.

    Computed here, never in the as-filed facts layer. Unless the equity row's latest concept already includes NCI,
    adds `minorityInterest`, else `_deduced_nci`; with neither, NCI is assumed zero only if no visible fact ever
    tagged an NCI value, otherwise None.
    """
    assets, equity = row.get("totalAssets"), row.get("stockholdersEquity")
    if assets is None or equity is None:
        return None
    rows = by_field.get("stockholdersEquity")
    concepts = rows[rows["value"].notna()]["source_concept"].dropna() if rows is not None else pd.Series(dtype=object)
    incl_nci = bool(len(concepts)) and _EQUITY_INCL_NCI in str(concepts.iloc[-1])
    if not incl_nci:
        nci = row.get("minorityInterest")
        if nci is None:
            nci = _deduced_nci(by_field)
        if nci is None:
            # Point-in-time: zero is assumed only while no visible fact has tagged an NCI value.
            if _has_valued_fact(by_field, "minorityInterest"):
                return None
        else:
            equity = equity + nci
    return float(assets) - float(equity)


def _as_datetime(column: pd.Series) -> pd.Series:
    """`column` as timestamps, converting only when it is not already datetime."""
    return column if is_datetime64_any_dtype(column) else pd.to_datetime(column, errors="coerce")


def _latest_period_known(visible: pd.DataFrame, as_of: pd.Timestamp) -> pd.Timestamp:
    """`fiscal_end`: the latest `period_of_report` (else `period_end`) visible by `as_of`, capped at `as_of`.

    The cap prevents look-ahead from a filing dated before its period closes. Monotone non-decreasing over events.
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


def _snapshot(ticker: str, visible: _Visible, event: pd.Series, catalogue: Catalogue) -> dict:
    """One complete history row from `visible` (every fact filed on or before `as_of`)."""
    # load: this event's fiscal end and regime (periods and instants come with `visible`)
    as_of = pd.Timestamp(event["as_of"])
    period = _latest_period_known(visible.facts, as_of)
    # `visible.facts` is a filing-date-sorted prefix, so the last non-null regime is the latest-filed one.
    regime = visible.facts["regime"].dropna()
    regime = str(regime.iloc[-1]) if not regime.empty else None
    row: dict = {
        "ticker": ticker,
        "as_of": event["as_of"],
        "fiscal_end": period,
        "fiscal_quarter": fiscal_quarter_of_end(period, visible.year_ends),
        HISTORY_REGIME: regime,
        **{c: event[c] for c in HISTORY_PROVENANCE},
    }

    # compute: every catalogue field, an instant aligned on `as_of` (the cover-page share count is dated after the
    # period end), a duration field from its newest TTM, which must be this row's own quarter
    for field in catalogue.history_fields:
        if field in _FORMULAS:
            continue  # computed once the inputs are in
        if catalogue.field(field).kind == INSTANT:
            value = _instant(visible.lookup, field, as_of)
        else:
            newest = visible.ttm.get(field)
            value = None if newest is None or _is_stale(newest, period) or pd.isna(newest["value"]) else float(newest["value"])
        row[field] = _gate(catalogue, regime, field, value)

    # compute: the identities, after the field loop so `minorityInterest` is available and before `_FORMULAS` so
    # `grossMargins` can use `grossProfit`; `operatingIncome`'s `derived_fallback` is deliberately unwired
    if row.get("totalLiabilities") is None:
        row["totalLiabilities"] = _total_liabilities_identity(row, visible.by_field)
    if row.get("grossProfit") is None:
        row["grossProfit"] = _gross_profit_identity(row, visible.facts)

    # compute: the ratios and the two discrete-quarter columns
    for column, (inputs, _) in _FORMULAS.items():
        row[column] = _ratio(column, *(row.get(name) for name in inputs))
    for column, source in _QUARTER_LABEL_COLUMNS.items():
        newest = visible.quarters.get(source)
        row[column] = None if newest is None or pd.isna(newest["value"]) else float(newest["value"])

    # check: last, so the guard also covers derived and computed columns
    _hard_guard(row)
    return row


def _hard_guard(row: dict) -> None:
    """Null every `HARD_GUARDS` violation in `row` in place, before the write."""
    for field, is_impossible in HARD_GUARDS.items():
        value = row.get(field)
        if value is not None and pd.notna(value) and is_impossible(float(value)):
            row[field] = None


def _gate(catalogue: Catalogue, regime: str | None, field: str, value: float | None) -> float | None:
    """Regime gating (history layer only): a `regime_gated` field `expected_absent` for the regime is dropped."""
    if regime and catalogue.expected_absent(regime, field) and catalogue.field(field).regime_gated:
        return None
    return value


# ---------------------------------------------------------------------- entry point ---


def build_ticker(
    ticker: str, facts, *, catalogue: Catalogue | None = None, guards: PeriodGuards | None = None, after: pd.Timestamp | None = None
) -> pd.DataFrame:
    """One ticker's `sec_fundamentals_history` frame: 66 columns, one row per publication event of an original filing.

    `facts` is the ticker's `sec_fundamentals_facts` rows (a `companyfacts` mapping is accepted for synthetic fixtures
    only); amendment facts are ignored. Every event's snapshot is built only from facts with `filing_date <= as_of`
    (`_Replay` reuses a field's periods from an earlier event only while that field's visible facts are unchanged), so
    no-leakage follows from the algorithm. `after` keeps only the events with `as_of > after`; each snapshot depends on
    its own prefix only, so those rows equal a full replay's. Asserts the 66-column contract and the grain.
    """
    catalogue = catalogue or load_catalogue()
    guards = guards or load_guards()
    frame = _normalise_facts(facts, catalogue)
    columns = catalogue.history_columns
    assert len(columns) == 66, f"the column contract is {len(columns)}, not 66"
    events = publication_events(frame)
    if after is not None and not events.empty:
        events = events[pd.to_datetime(events["as_of"]) > pd.Timestamp(after)].reset_index(drop=True)
    if events.empty:
        return pd.DataFrame(columns=columns)

    replay = _Replay(_plain_strings(frame), catalogue, guards)
    rows = [_snapshot(ticker, replay.at(event["as_of"]), event, catalogue) for _, event in events.iterrows()]

    # Pinned dtypes (all-null columns too) for `diff_against_stored` and cold-table type inference.
    history = pin_dtypes(
        pd.DataFrame(rows).reindex(columns=columns),
        dates=("as_of", "fiscal_end"),
        floats=[column for column in columns if column not in (*HISTORY_KEYS, HISTORY_REGIME, *HISTORY_PROVENANCE)],
        texts=(*HISTORY_PROVENANCE, HISTORY_REGIME),
    )
    # Nullable Int64: a ticker with no fiscal calendar yet keeps a NULL quarter, not 0.
    history["fiscal_quarter"] = history["fiscal_quarter"].astype("Int64")
    _assert_grain(ticker, history)
    return history


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
    """`frame` reduced to `PERIOD_COLUMNS` (every engine filter copies its frame, so fewer columns copy less)."""
    return frame[[c for c in PERIOD_COLUMNS if c in frame.columns]].copy()


def _plain_strings(frame: pd.DataFrame) -> pd.DataFrame:
    """`frame` with its string columns cast to object once, so per-event slices and groups never take Arrow strings."""
    strings = {c: object for c in frame.columns if is_string_dtype(frame[c]) and not is_object_dtype(frame[c])}
    return frame.astype(strings) if strings else frame


def _normalise_facts(facts, catalogue: Catalogue) -> pd.DataFrame:
    """The original filings' facts with timestamp dates, defaults for columns a fixture may lack, sorted by `filing_date`."""
    if not isinstance(facts, pd.DataFrame):
        facts = facts_frame_from_companyfacts(facts, catalogue)
    out = facts.copy()
    for column in ("filing_date", "period_of_report", "period_start", "period_end"):
        out[column] = pd.to_datetime(cast(Any, out.get(column)), errors="coerce")
    for column, default in (("regime", None), ("form", "10-Q"), ("accession_number", ""), ("source_concept", None)):
        if column not in out.columns:
            out[column] = default
    out = _originals(out)
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
    """One `sec_fundamentals_facts`-shaped row per companyfacts entry of `concept`, across all its units."""
    return [
        {
            "ticker": "FIXTURE",
            "accession_number": f"{concept}-{unit}-{i}",
            "field": field,
            "fiscal_year": pd.Timestamp(entry["end"]).year,
            "fiscal_period": entry.get("fp", "NA"),
            "form": entry.get("form", "10-Q"),
            "filing_date": entry.get("filed"),
            "period_of_report": entry["end"],
            "regime": None,
            "period_start": entry.get("start"),
            "period_end": entry["end"],
            "value": entry.get("val"),
            "unit": unit,
            "source_concept": concept,
        }
        for unit, entries in (payload.get("units") or {}).items()
        for i, entry in enumerate(entries)
    ]


def facts_frame_from_companyfacts(blob: dict, catalogue: Catalogue) -> pd.DataFrame:
    """A `sec_fundamentals_facts`-shaped frame from a raw `companyfacts` mapping, for synthetic fixtures ONLY.

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
    # One accession per (filing date, form), so the events are filings and not facts.
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


def _seam_rule_windows(context: Context, ticker: str, filers: int) -> tuple[CikWindow, ...] | None:
    """The windows the seam rule runs on, or None when it does not run: it runs for several filers, and for one filer
    only when a window has a dated start (a `fresh_start` CIK) -- a single filer the roster does not hold is left as is."""
    if filers == 0:
        return None
    identity = load_identity(context)
    if filers == 1 and ticker not in identity.roster_cik:
        return None
    windows = identity.filing_scope(ticker).windows
    return windows if filers > 1 or any(window.valid_from is not None for window in windows) else None


# ------------------------------------------------------------------- immutability ---

#: The `sec_fundamentals_facts` columns the replay reads (projected read, one ticker at a time).
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
    "period_of_report",
    "regime",
    "period_start",
    "period_end",
    "period_days",
    "value",
    "unit",
    "source_concept",
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
    for column in ("as_of", "fiscal_end"):
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    return out.set_index("as_of").sort_index()


def _unpublished_events(context, ticker: str, stored: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """The rebuilt history rows whose `as_of` is not stored yet.

    Raises ValueError, after logging the diff, if a stored row would change (history is append-only).
    """
    drift = diff_against_stored(stored, history)
    if not drift.empty:
        context.log.error(
            "history: %s would CHANGE %d already-published cell(s) "
            "across %d row(s) -- refusing to overwrite. Re-run with "
            "sec-fundamentals-history --full to accept:\n%s",
            ticker,
            len(drift),
            drift["as_of"].nunique(),
            drift.head(20).to_string(),
        )
        raise ValueError(
            f"{ticker}: {len(drift)} stored sec_fundamentals_history cell(s) would change; history is append-only "
            "(run sec-fundamentals-history --full to rebuild)"
        )
    known = set(pd.to_datetime(stored["as_of"]))
    return history[~pd.to_datetime(history["as_of"]).isin(known).values]


def _drop_history(context: Context, ticker: str) -> None:
    """Delete a rebuilt ticker's history when its facts now yield no event."""
    deleted = context.store.delete(Tables.sec_fundamentals_history, {"ticker": ticker})
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

#: The `sec_fundamentals_facts` columns the triage reads: one filing date per row, and its form (originals only count).
_TRIAGE_COLUMNS: tuple[str, ...] = ("ticker", "filing_date", "form")


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
        Tables.sec_fundamentals_history, columns=["ticker", "as_of", *HISTORY_PROVENANCE], where={"ticker": list(tickers)}, optional=True
    )
    if stored is None:
        return {}
    stored = stored.assign(as_of=pd.to_datetime(stored["as_of"]).dt.normalize())
    return {str(ticker): group.drop(columns="ticker").sort_values("as_of").reset_index(drop=True) for ticker, group in stored.groupby("ticker")}


def _filing_dates(context: Context, tickers: list[str]) -> dict[str, pd.Series]:
    """Each ticker's distinct original filing dates, streamed so no chunk outlives its reduction."""
    if not tickers:
        return {}
    parts = []
    for chunk in context.store.iter_load(Tables.sec_fundamentals_facts, columns=list(_TRIAGE_COLUMNS), where={"ticker": list(tickers)}):
        originals = _originals(chunk)
        filed = pd.to_datetime(originals["filing_date"], errors="coerce").dt.normalize()
        parts.append(pd.DataFrame({"ticker": originals["ticker"].astype(str), "filing_date": filed}).dropna().drop_duplicates())
    if not parts:
        return {}
    dates = pd.concat(parts, ignore_index=True).drop_duplicates()
    return {str(ticker): group["filing_date"] for ticker, group in dates.groupby("ticker")}


def _ticker_path(ticker: str, events: pd.DataFrame, filed: pd.Series | None, recent: frozenset[str]) -> str:
    """The path of a ticker with stored history (see `_history_work`)."""
    if filed is None or filed.empty:
        return FULL
    newest = pd.Timestamp(events["as_of"].max())
    # Every original publishes, so the stored dates equal the filing dates up to the newest stored row unless the past moved.
    moved = set(events["as_of"]) != set(filed[filed <= newest])
    if (filed > newest).any():
        return INCREMENTAL
    return CHECK if moved or ticker in recent else SKIP


def _history_work(context: Context, tickers: list[str], *, verify: bool = False, recent: frozenset[str] = frozenset()) -> dict[str, HistoryWork]:
    """Each ticker's history path, from two narrow reads and before any replay.

    `full`: no stored history, `verify`, or no facts. `incremental`: an original filed after the newest stored row.
    `check`: lineage-recent (`recent`), or the stored dates and the original filing dates disagree up to the newest
    stored row (a back-dated original). `skip`: otherwise. `incremental` and `check` compare the rebuilt event list
    first (`_incremental_rows`), so a back-dated original reaches the full replay.
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
        work[ticker] = HistoryWork(_ticker_path(ticker, events, dates.get(ticker), recent), pd.Timestamp(events["as_of"].max()), events)
    return work


def _event_keys(events: pd.DataFrame) -> list[tuple]:
    """`(as_of, publication_form)` per event, normalised so stored (DB round-trip) and rebuilt frames compare equal."""
    keys = [
        (pd.Timestamp(as_of).normalize(), None if form is None or pd.isna(form) else str(form))
        for as_of, form in zip(events["as_of"], events["publication_form"], strict=True)
    ]
    return sorted(keys, key=lambda key: key[0])


def _incremental_rows(ticker: str, df_facts: pd.DataFrame, work: HistoryWork, catalogue: Catalogue, guards: PeriodGuards) -> pd.DataFrame | None:
    """Rows of the events after `work.newest` only, or None when the rebuilt event list up to it differs from the stored one."""
    assert work.newest is not None and work.stored_events is not None
    events = publication_events(_normalise_facts(df_facts, catalogue))
    dates = pd.to_datetime(events["as_of"])
    if _event_keys(events[dates <= work.newest]) != _event_keys(work.stored_events):
        return None
    if not (dates > work.newest).any():
        return pd.DataFrame(columns=catalogue.history_columns)
    built = build_ticker(ticker, df_facts, catalogue=catalogue, guards=guards, after=work.newest)
    # Append-only: never upsert onto a stored row, whatever the build returned.
    return built[(pd.to_datetime(built["as_of"]) > work.newest).values]


# ---------------------------------------------------------------------- the build ---


def _ticker_facts(context: Context, ticker: str, seams: Mapping[str, pd.Timestamp]) -> pd.DataFrame | None:
    """The ticker's original-filing facts with the seam rule and, for a declared reverse acquisition (`seams`), the
    comparatives rule applied, or None when it has none stored."""
    df_facts = context.store.load(Tables.sec_fundamentals_facts, columns=list(FACT_COLUMNS), where={"ticker": ticker}, optional=True)
    if df_facts is None:
        return df_facts
    # Amendment facts are ignored before the seam rule, so an amendment never decides which filer owns a period.
    df_facts = _originals(df_facts)
    windows = _seam_rule_windows(context, ticker, _filer_count(df_facts))
    if windows is not None:
        df_kept = keep_window_owner_filings(df_facts, windows)
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


def _build_in_worker(ticker: str, df_facts: pd.DataFrame) -> pd.DataFrame:
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
    full: Future[pd.DataFrame] | pd.DataFrame | None = None
    rows: pd.DataFrame | None = None
    no_facts: bool = False


def _full_rows(
    context: Context,
    ticker: str,
    df_history: pd.DataFrame,
    *,
    rebuild_history: bool,
    rebuild: frozenset[str],
    catalogue: Catalogue,
) -> pd.DataFrame | None:
    """The rows of a full replay (`df_history`) to save after the drift guard, or None when the facts yield no event."""
    if df_history.empty:
        if ticker in rebuild:
            _drop_history(context, ticker)
        return None
    # Explicit projection so the read fails loudly if the table and the column contract diverge.
    df_stored = context.store.load(Tables.sec_fundamentals_history, columns=list(catalogue.history_columns), where={"ticker": ticker}, optional=True)
    drifted = ticker in rebuild and df_stored is not None and not diff_against_stored(df_stored, df_history).empty
    if rebuild_history or drifted:
        deleted = context.store.delete(Tables.sec_fundamentals_history, {"ticker": ticker})
        context.log.warning(
            "history: %s REBUILT -- %d row(s) deleted and recomputed. "
            "Log this in the phase report: a rebuild re-derives numbers "
            "under whatever model is already trained on them.",
            ticker,
            deleted,
        )
    elif df_stored is not None:
        df_history = _unpublished_events(context, ticker, df_stored, df_history)
    return df_history


def build_fundamentals_history(
    context: Context, tickers: list[str], *, rebuild_history: bool = False, verify_history: bool = False, as_of: pd.Timestamp | None = None
) -> None:
    """`sec_fundamentals_facts` -> `sec_fundamentals_history`, per ticker, from stored facts only (no network).

    Triage first (`_history_work`): a current ticker is skipped without reading its facts, a ticker with newer
    filings snapshots its new events only, and the rest (incl. a back-dated original) replay in full. Append-only: a
    full replay saves only new `as_of` events, and a stored row that would change raises ValueError after logging the
    diff. `verify_history=True` replays every ticker in full under that guard. `rebuild_history=True` (CLI
    `sec-fundamentals-history --full`) deletes the ticker's rows and rebuilds from stored facts; so does a ticker whose
    lineage scope changed recently (`resume.recently_changed` on `as_of`) when its recomputed history differs from the
    stored one. A ticker whose facts come from several CIKs, or whose window has a dated start (a `fresh_start` CIK),
    goes through the seam rule (`keep_window_owner_filings`) on its identity-layer windows; a declared reverse
    acquisition then drops the pre-seam comparatives of its survivor's post-seam filings.
    Full builds run in a process pool of `data_extract.fundamentals_workers` (in-process for 1 worker or a single
    ticker); the parent reads, guards, saves and logs, in ticker order.
    """
    catalogue = load_catalogue(str(context.config_dir))
    guards = load_guards(str(context.config_dir))
    seams = {entry.ticker: entry.seam_date for entry in load_security_manual(str(context.config_dir)).reverse_acquisitions}
    history_rows = 0
    rebuild = frozenset(tickers) if rebuild_history else _scope_changed_recently(context, tickers, as_of)
    if rebuild and not rebuild_history:
        context.log.info(
            "history: %d ticker lineage scope(s) changed recently -> rebuilt where the history moved: %s", len(rebuild), ", ".join(sorted(rebuild))
        )
    if rebuild_history:
        work = {ticker: HistoryWork(FULL) for ticker in tickers}
    else:
        work = _history_work(context, tickers, verify=verify_history, recent=rebuild)
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
        nonlocal history_rows
        ticker = entry.ticker
        if entry.no_facts:
            context.log.info("history: %s has no stored facts -- skipped", ticker)
            if ticker in rebuild:
                _drop_history(context, ticker)
            return
        df_history = entry.rows
        if entry.full is not None:
            full = entry.full.result() if isinstance(entry.full, Future) else entry.full
            df_history = _full_rows(context, ticker, full, rebuild_history=rebuild_history, rebuild=rebuild, catalogue=catalogue)
        if df_history is None:
            return
        if df_history.empty:
            context.log.info("history: %s already current (0 new events)", ticker)
            return
        context.store.save(Tables.sec_fundamentals_history, df_history)
        context.log.info("history: %s +%d event row(s)", ticker, len(df_history))
        history_rows += len(df_history)

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

    context.log.info("history: %d ticker(s), +%d event row(s)", len(tickers), history_rows)
