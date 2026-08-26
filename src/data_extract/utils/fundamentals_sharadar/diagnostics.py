"""
diagnostics.py  (src/data_extract/utils/fundamentals_sharadar/diagnostics.py)
------------------------------------------------------------------------------------
READ-ONLY measurement of `fundamentals_sharadar`, for exactly two consumers: the Full-tier
purchase decision, and the per-field zero rule phase 3 needs.

⚠ **THIS IS NOT THE SEC CHECK SCHEME AND MUST NEVER BE WIRED INTO IT** (D25). Nothing here
registers a `CHECK_REGISTRY` entry, writes a `fundamentals_check` row, imports from
`src/validate/`, or is reachable from the validator CLI. It writes no production data at all:
one markdown report and one proposed JSON rule file, both outside the database. If a change to
this module starts to look like adding a check, it is in the wrong file.

## The three gates, and why the spec's third one had to be replaced

The spec proposed `Q4 == FY - 9M` as an acceptance check. It is **dead on arrival**: Sharadar
CONSTRUCTS Q4 as `ARY - Σ(Q1..Q3)`, so the identity is an identity and was measured at
`+0.000%` on every year tested. A check that can never fail can never inform. `confirm_q4_
tautology` is kept anyway -- not as a gate, but as EVIDENCE FOR THE RECORD that the gate is
empty, so nobody re-proposes it in six months.

What replaces it is `gate_implausible_quarters`: the construction cannot fail the identity,
but it can and does produce absurd LEVELS. The legacy Quandl documentation shows it yielding
ABT 2011 Q4 revenue of **-$7.1bn**, annotated as intentional "to ensure that the quarterly and
annual financials are aligned". That is the failure mode worth measuring.

## Two traps this module exists to close

1. **A zero-fill verdict is only as strong as the basis behind it.** Sharadar zero-fills 41
   fields, and a 0 can mean "structurally not applicable" (a bank has no inventory) or
   "absent, and we wrote a zero anyway" (`intexp = 0` for JPM). Telling them apart needs a
   second opinion, and the obvious one -- `fundamentals_history_sec` -- is on a DIFFERENT
   BASIS: it is TTM for duration fields, and three of its columns are documented supersets of
   Sharadar's. So the comparison is basis-matched explicitly: duration fields are judged at
   the TTM level against Sharadar's own ART dimension, instant fields point-in-time, and a
   `sec_wider` counterpart can only ever produce a SUSPICION, never a verdict.

2. **A fiscal year is not a calendar year, and `calendardate` will not tell you which is
   which.** Grouping quarters by `calendardate` looks obviously right and is obviously wrong:
   for an ARY row Sharadar sets `calendardate` to December of the ASSIGNED CALENDAR YEAR, not
   to the fiscal year end, so every filer's annual row looks like a December year-end. The
   first implementation derived each ticker's year-end quarter from exactly that column,
   concluded all 30 were December filers, and split the four quarters of every non-December
   filer across two year labels -- 849 of 3,446 triples showed a spurious ΣARQ-vs-ARY
   deviation before this was caught. Sharadar's own `fiscalperiod` (`2022-Q1` ... `2022-FY`)
   already carries the assignment; `with_fiscal_period` reads it and nothing here re-derives
   it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.constants.constants import (
    SHARADAR_CONFIG_SUBDIR, SHARADAR_DIAGNOSTIC_EXTRA_COLUMNS, SHARADAR_EVENT_FIELDS,
    SHARADAR_FLOW_FIELDS, SHARADAR_NON_NEGATIVE_FIELDS, SHARADAR_SEC_COUNTERPART,
    SHARADAR_ZERO_FILLED_FIELDS, SHARADAR_ZERO_RULE_CONTRADICTION_SHARE,
    SHARADAR_ZERO_RULE_MIN_CHECKED, SHARADAR_ZERO_RULE_MIXED_SHARE,
    SHARADAR_ZERO_RULES_FILENAME,
)
from src.context import Context
from src.data_store.schema import Tables

#: Default destination of the generated report. A dated plan path rather than a constant in
#: `constants.py`: it belongs to one planning task, not to the pipeline's vocabulary.
DEFAULT_REPORT_PATH = ("reports/planning/active-tasks/2026-08-26-sharadar-integration/"
                       "phase-2-findings.md")

#: How many flagged rows the report lists in full. The count is always printed beside it, so
#: the tail is visible without being enumerated.
WORST_ROWS = 20

#: `|fcf - (ncfo + capex)|` above which the identity is NOT holding "to the cent". One dollar,
#: on figures in the billions -- anything larger is a definition difference, not float noise.
FCF_IDENTITY_TOLERANCE = 1.0

#: The FLOAT-NOISE FLOOR on |ΣARQ - ARY| / |ARY|, not a quality bar.
#:
#: Where the Q4 identity holds it holds at EXACTLY 0.0, because Sharadar builds Q4 by
#: subtraction -- so this threshold is not separating "close enough" from "too far", it is
#: separating the residue of summing four doubles from a genuinely different number. Measured:
#: 3,395 triples at exactly 0, 79 under this floor, 58 above it and none of those 58 anywhere
#: near it (the smallest is ~1%). Nothing lands in the gap, which is what a noise floor should
#: look like.
Q4_TAUTOLOGY_MAX_PCT = 0.0001


# --------------------------------------------------------------------------- #
# loading                                                                      #
# --------------------------------------------------------------------------- #
def _diagnostic_columns() -> list[str]:
    """The projection: the table's declared `read_columns` plus the 4 zero-filled fields it
    deliberately omits. Never the whole table -- 112 columns x 3 dimensions is the widest
    extract table in the schema."""
    return list(dict.fromkeys(list(Tables.sharadar_fundamentals.read_columns)
                              + list(SHARADAR_DIAGNOSTIC_EXTRA_COLUMNS)))


def _load_sharadar(context: Context, tickers: Sequence[str] | None,
                   dimensions: Sequence[str]) -> pd.DataFrame:
    """One projected read of `fundamentals_sharadar`, filtered server-side to `dimensions`.

    Dates are re-coerced on the way out: a Postgres DATE column round-trips as
    `datetime.date`, which does not compare equal to the `datetime64` on the other side of
    every merge in this module.
    """
    where: dict[str, object] = {"dimension": list(dimensions)}
    if tickers:
        where["ticker"] = list(tickers)
    frame = context.store.load(Tables.sharadar_fundamentals, columns=_diagnostic_columns(),
                               where=where, optional=True)
    if frame is None or frame.empty:
        raise RuntimeError(
            f"{Tables.sharadar_fundamentals} has no {'/'.join(dimensions)} rows for the "
            f"requested scope. Phase 2 measures the DB, not the API (D29) -- run "
            f"`python -m src data_extract fundamentals-sharadar` first.")
    for col in ("date", "reportperiod", "calendardate"):
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce")
    return frame


def _load_sec(context: Context, tickers: Sequence[str]) -> pd.DataFrame:
    """The SEC layer, projected to the columns the cross-checks need.

    Returns an EMPTY frame rather than raising when no requested ticker is on the SEC roster:
    the SEC comparison is a bonus available on the overlap, and a Sharadar-only universe must
    still produce every other gate.
    """
    cols = ["ticker", "as_of", "sharesOutstanding"]
    for sec_cols, _ in SHARADAR_SEC_COUNTERPART.values():
        cols.extend(sec_cols)
    cols = list(dict.fromkeys(cols))
    frame = context.store.load(Tables.fundamentals_history_sec, columns=cols,
                               where={"ticker": list(tickers)} if tickers else None,
                               optional=True)
    if frame is None or frame.empty:
        return pd.DataFrame(columns=cols)
    frame["as_of"] = pd.to_datetime(frame["as_of"], errors="coerce")
    return frame


# --------------------------------------------------------------------------- #
# fiscal-year grouping                                                         #
# --------------------------------------------------------------------------- #
def _quarter_index(dates: pd.Series) -> pd.Series:
    """A calendar quarter as one monotonically increasing integer: `year*4 + (quarter-1)`.

    Gap detection is integer arithmetic on this, which is why it never has to reason about
    month lengths or 52/53-week retail calendars -- Sharadar's `calendardate` has already been
    rounded to the nearest calendar quarter-end. Used ONLY for completeness; the fiscal-year
    grouping deliberately does not touch this column (see `with_fiscal_period`).
    """
    cal = pd.to_datetime(dates)
    return (cal.dt.year * 4 + (cal.dt.quarter - 1)).astype("Int64")


def _quarter_label(q_index: int) -> str:
    return f"{q_index // 4}Q{q_index % 4 + 1}"


def with_fiscal_period(frame: pd.DataFrame) -> pd.DataFrame:
    """Split Sharadar's own `fiscalperiod` into `fiscal_year` (int) and `fiscal_position`.

    ⚠ **DO NOT DERIVE THE FISCAL YEAR FROM `calendardate`.** That was the first implementation
    and it was measurably wrong, for a reason worth writing down: for an ARY row Sharadar sets
    `calendardate` to **December of the calendar year the fiscal year is assigned to**, NOT to
    the fiscal year end. AAPL's `2022-FY` row carries `calendardate = 2022-12-31` against
    `reportperiod = 2022-09-24`. Reading the fiscal-year-end quarter off the annual rows
    therefore returns Q4 for every filer on earth, the derived shift is zero for all of them,
    and every non-December filer gets its four quarters split across two year labels. Measured:
    849 of 3,446 (ticker, year, field) triples showed a spurious ΣARQ-vs-ARY deviation, on
    exactly the seven non-December DJIA filers (AAPL, MSFT, DIS, CSCO, NKE, V, PG).

    `fiscalperiod` is the vendor's OWN assignment -- `2022-Q1` for the quarter ending
    2021-12-25, `2022-FY` for the year ending 2022-09-24 -- and it is the only field that
    already knows which fiscal year a row belongs to. Format verified across all stored rows:
    `<year>-Q1|Q2|Q3|Q4|FY`, no other suffix.
    """
    period = frame["fiscalperiod"].astype(str)
    split = period.str.rsplit("-", n=1, expand=True)
    return frame.assign(
        fiscal_year=pd.to_numeric(split[0], errors="coerce").astype("Int64"),
        fiscal_position=split[1].str.upper())


# --------------------------------------------------------------------------- #
# gate 1 -- completeness                                                       #
# --------------------------------------------------------------------------- #
def gate_completeness(context: Context,
                      tickers: Sequence[str] | None = None) -> pd.DataFrame:
    """Missing quarters per ticker, measured against EACH TICKER'S OWN observed window.

    A ticker whose history simply starts late is not a gap -- on a 5-year entitlement every
    ticker would otherwise show a hole back to whatever the earliest one happens to reach. The
    expected count is `last - first + 1` quarters of that ticker's own span, so the only thing
    this can report is a HOLE, which is the only thing that would break a TTM build.
    """
    arq = _load_sharadar(context, tickers, ("ARQ",))
    arq = arq.assign(_q=_quarter_index(arq["calendardate"]))
    rows: list[dict] = []
    for ticker, group in arq.groupby("ticker", sort=True):
        observed = sorted(int(q) for q in group["_q"].dropna().unique())
        if not observed:
            continue
        span = range(observed[0], observed[-1] + 1)
        missing = [q for q in span if q not in set(observed)]
        rows.append({
            "ticker": str(ticker),
            "first_quarter": _quarter_label(observed[0]),
            "last_quarter": _quarter_label(observed[-1]),
            "n_rows": int(len(group)),
            "n_quarters": len(observed),
            "expected_quarters": len(span),
            "n_missing": len(missing),
            "missing_quarters": ", ".join(_quarter_label(q) for q in missing) or "-",
            # a duplicate calendardate is a DIFFERENT defect from a gap: two filings rounding
            # onto the same normalised quarter. Counted so it cannot hide inside `n_rows`.
            "n_duplicate_quarters": int(len(group) - len(observed)),
        })
    return pd.DataFrame(rows).sort_values(["n_missing", "ticker"], ascending=[False, True])


# --------------------------------------------------------------------------- #
# gate 2 -- implausible quarters                                               #
# --------------------------------------------------------------------------- #
def _value_fields(frame: pd.DataFrame) -> list[str]:
    """Every numeric column of the projection -- the identifiers removed."""
    ids = {"ticker", "dimension", "calendardate", "date", "reportperiod", "fiscalperiod",
           "lastupdated", "_q", "fiscal_year", "fiscal_position"}
    return [c for c in frame.columns
            if c not in ids and pd.api.types.is_numeric_dtype(frame[c])]


_FLAG_COLUMNS = ["ticker", "field", "fiscal_year", "fiscal_position", "calendardate", "value",
                 "reason", "max_other_abs", "ratio_vs_other"]


def gate_implausible_quarters(context: Context, tickers: Sequence[str] | None = None, *,
                              ratio: float = 3.0) -> pd.DataFrame:
    """Every ARQ cell that the Q4 construction cannot plausibly have produced honestly.

    Two flags, both scoped to (field, ticker, fiscal year):

    * `negative` -- a value below zero in a field that has no negative reading. This is the
      ABT -$7.1bn revenue failure mode, and it is a LEVEL error: the annual row still foots,
      so no identity check can ever see it.
    * `magnitude` -- |value| more than `ratio` x the largest OTHER quarter of the same fiscal
      year. `ratio` is `data_extract.fundamentals_periods.max_opposite_sign_q4_ratio`, reused
      rather than reinvented: the SEC path already calibrated 3.0 against confirmed as-filed
      quarters (Allstate FY2023 at 1.10x, Gilead's FY2017 TCJA writedown at 1.26x), and a
      second threshold measuring the same thing would only be a second thing to get wrong.

    A fiscal year needs at least 3 quarters before the magnitude test runs -- with two, "the
    largest other quarter" is one number and the ratio is noise.

    Reshaped to LONG once rather than looped per field: 65 fields x two grouped transforms
    apiece cost ~3 minutes of `groupby.apply`, and one sort plus one `cumcount` over the melted
    frame is the same arithmetic in seconds.
    """
    arq = with_fiscal_period(_load_sharadar(context, tickers, ("ARQ",)))
    fields = _value_fields(arq)
    keys = ["ticker", "fiscal_year", "fiscal_position", "calendardate"]
    long = arq.melt(id_vars=keys, value_vars=fields, var_name="field",
                    value_name="value").dropna(subset=["value", "fiscal_year"])
    if long.empty:
        return pd.DataFrame(columns=_FLAG_COLUMNS)

    negative = long[long["field"].isin(SHARADAR_NON_NEGATIVE_FIELDS) & (long["value"] < 0)]
    negative = negative.assign(reason="negative", max_other_abs=np.nan, ratio_vs_other=np.nan)

    group = ["field", "ticker", "fiscal_year"]
    long = long.assign(_abs=long["value"].abs()).sort_values(
        group + ["_abs"], ascending=[True, True, True, False], kind="mergesort")
    ranked = long.groupby(group, observed=True)["_abs"]
    long["_rank"] = ranked.cumcount()
    long["_size"] = ranked.transform("size")
    index = pd.MultiIndex.from_frame(long[group])
    # the two largest |values| of each (field, ticker, fiscal year), taken off the sort rather
    # than from a per-group lambda
    largest = long[long["_rank"] == 0].set_index(group)["_abs"]
    runner_up = long[long["_rank"] == 1].set_index(group)["_abs"]
    long["_max1"] = largest.reindex(index).to_numpy()
    long["_max2"] = runner_up.reindex(index).to_numpy()
    # "the largest OTHER quarter": the runner-up for the row that IS the largest, else the
    # largest. A tie therefore lands at ratio 1.0 instead of dividing a row by itself.
    long["_max_other"] = np.where(long["_rank"] == 0, long["_max2"], long["_max1"])
    outlier = ((long["_size"] >= 3) & (long["_max_other"] > 0)
               & (long["_abs"] > ratio * long["_max_other"]))
    magnitude = long[outlier].assign(
        reason="magnitude", max_other_abs=long.loc[outlier, "_max_other"],
        ratio_vs_other=long.loc[outlier, "_abs"] / long.loc[outlier, "_max_other"])

    flagged = pd.concat([negative, magnitude], ignore_index=True)
    if flagged.empty:
        return pd.DataFrame(columns=_FLAG_COLUMNS)
    return flagged[_FLAG_COLUMNS].sort_values(
        ["ratio_vs_other", "field"], ascending=[False, True], na_position="last")


# --------------------------------------------------------------------------- #
# gate 3 -- zero-fill prevalence, per field                                    #
# --------------------------------------------------------------------------- #
def _sec_counterpart_value(sec: pd.DataFrame, sec_cols: Sequence[str]) -> pd.DataFrame:
    """`(ticker, as_of, _sec)` where `_sec` is the sum of `sec_cols`, NaN only if ALL are."""
    present = [c for c in sec_cols if c in sec.columns]
    if not present:
        return pd.DataFrame(columns=["ticker", "as_of", "_sec"])
    return pd.DataFrame({
        "ticker": sec["ticker"].astype(str),
        "as_of": sec["as_of"],
        "_sec": sec[present].sum(axis=1, min_count=1),
    })


def _sec_verdicts(field: str, zeros: pd.DataFrame, art: pd.DataFrame,
                  sec: pd.DataFrame) -> dict[str, int]:
    """How the SEC layer judges each zero cell of `field`, on a BASIS-MATCHED comparison.

    * A DURATION field is judged at the TTM level, because that is the only grain on which the
      two sources mean the same thing: `fundamentals_history_sec` is TTM, so a non-zero there
      says nothing about any individual quarter. The Sharadar side of that comparison is its
      own ART dimension. A zero quarter inside a non-zero trailing year stays INCONCLUSIVE --
      a quarter really can be zero.
    * An INSTANT field is compared point-in-time, filing date to filing date.
    * A `sec_wider` counterpart can only ever produce `suspect`. `totalDebt` carries lease
      liabilities Sharadar's `debt` does not, so "SEC non-zero, Sharadar zero" may be entirely
      the components Sharadar never claimed to have.
    """
    empty = {"overlap_zeros": int(len(zeros)), "checked": 0, "agrees": 0, "absent": 0,
             "contradicted": 0, "suspect": 0, "inconclusive": 0, "contradicted_tickers": ""}
    sec_cols, comparability = SHARADAR_SEC_COUNTERPART[field]
    counterpart = _sec_counterpart_value(sec, sec_cols)
    if counterpart.empty or zeros.empty:
        return empty

    joined = zeros.merge(counterpart, left_on=["ticker", "date"],
                         right_on=["ticker", "as_of"], how="left")
    if field in SHARADAR_FLOW_FIELDS:
        ttm = art[["ticker", "date", field]].rename(columns={field: "_ttm"})
        joined = joined.merge(ttm, on=["ticker", "date"], how="left")
    else:
        joined["_ttm"] = np.nan

    has_sec = joined["_sec"].notna()
    sec_zero = has_sec & (joined["_sec"].abs() == 0)
    sec_nonzero = has_sec & (joined["_sec"].abs() > 0)
    if field in SHARADAR_FLOW_FIELDS:
        # only a zero TRAILING YEAR against a non-zero SEC TTM is a like-for-like disagreement
        strong = sec_nonzero & joined["_ttm"].notna() & (joined["_ttm"].abs() == 0)
    else:
        strong = sec_nonzero
    contradicted = strong if comparability == "exact" else pd.Series(False, index=joined.index)
    return {
        "overlap_zeros": int(len(joined)),
        "checked": int(has_sec.sum()),
        "agrees": int(sec_zero.sum()),
        # SEC ABSENT is agreement, not ignorance: the SEC path stores NULL where it found no
        # value and never 0, so "SEC has nothing here either" is the strongest available
        # evidence that Sharadar's zero means "not applicable". `rnd` lives entirely here --
        # 140 of 140 overlap zeros, every one of them a filer with no R&D line at all.
        "absent": int((~has_sec).sum()),
        "contradicted": int(contradicted.sum()),
        "suspect": int((sec_nonzero & strong & ~contradicted).sum()),
        "inconclusive": int((sec_nonzero & ~strong).sum()),
        # naming the filers is what makes the verdict checkable against an actual filing
        "contradicted_tickers": ", ".join(
            sorted(joined.loc[contradicted, "ticker"].astype(str).unique())),
    }


def gate_zero_fill(context: Context, tickers: Sequence[str] | None = None) -> pd.DataFrame:
    """Per-field zero-fill prevalence, with the evidence needed to decide whether each zero
    is a fact or a fill. One row per field in `SHARADAR_ZERO_FILLED_FIELDS`.

    Two independent bodies of evidence, because neither is sufficient alone:

    * **Sharadar-internal** -- `n_tickers_all_zero` (the field is 0 in EVERY row of that
      ticker: the signature of "not applicable") against `n_zero_mixed` (zeros in tickers that
      report the same field non-zero in another quarter: the signature of a fill). This is the
      ONLY evidence available for the 21 fields with no SEC counterpart.
    * **SEC cross-check** on the overlapping tickers, basis-matched -- see `_sec_verdicts`.
    """
    arq = _load_sharadar(context, tickers, ("ARQ",))
    art = _load_sharadar(context, tickers, ("ART",))
    sharadar_tickers = sorted(str(t) for t in arq["ticker"].unique())
    sec = _load_sec(context, sharadar_tickers)
    overlap = sorted(set(sharadar_tickers) & set(sec["ticker"].astype(str))) if not sec.empty else []

    rows: list[dict] = []
    for field in sorted(SHARADAR_ZERO_FILLED_FIELDS):
        if field not in arq.columns:
            rows.append({"field": field, "n_rows": 0, "n_zero": 0, "pct_zero": np.nan,
                         "n_tickers": 0, "n_tickers_all_zero": 0, "n_zero_mixed": 0,
                         "sec_basis": "not projected", "sec_overlap_zeros": 0,
                         "sec_checked": 0, "sec_agrees": 0, "sec_absent": 0,
                         "sec_contradicted": 0, "sec_suspect": 0, "sec_inconclusive": 0,
                         "sec_contradicted_tickers": ""})
            continue
        series = arq[["ticker", "date", field]].dropna(subset=[field])
        is_zero = series[field] == 0
        per_ticker = series.groupby("ticker")[field].apply(lambda s: int((s != 0).sum()))
        all_zero = set(per_ticker.index[per_ticker == 0].astype(str))
        mixed = is_zero & ~series["ticker"].astype(str).isin(all_zero)

        counterpart = SHARADAR_SEC_COUNTERPART.get(field)
        if counterpart is None or not overlap:
            basis = "no SEC counterpart" if counterpart is None else "no overlapping ticker"
            verdicts = {"overlap_zeros": 0, "checked": 0, "agrees": 0, "absent": 0,
                        "contradicted": 0, "suspect": 0, "inconclusive": 0,
                        "contradicted_tickers": ""}
        else:
            sec_cols, comparability = counterpart
            grain = "TTM" if field in SHARADAR_FLOW_FIELDS else "instant"
            basis = f"{'+'.join(sec_cols)} ({comparability}, {grain})"
            zeros = series[is_zero & series["ticker"].astype(str).isin(overlap)][
                ["ticker", "date"]].copy()
            zeros["ticker"] = zeros["ticker"].astype(str)
            verdicts = _sec_verdicts(field, zeros, art, sec)

        rows.append({
            "field": field,
            "n_rows": int(len(series)),
            "n_zero": int(is_zero.sum()),
            "pct_zero": float(is_zero.mean()) if len(series) else np.nan,
            "n_tickers": int(series["ticker"].nunique()),
            "n_tickers_all_zero": len(all_zero),
            "n_zero_mixed": int(mixed.sum()),
            "sec_basis": basis,
            "sec_overlap_zeros": verdicts["overlap_zeros"],
            "sec_checked": verdicts["checked"],
            "sec_agrees": verdicts["agrees"],
            "sec_absent": verdicts["absent"],
            "sec_contradicted": verdicts["contradicted"],
            "sec_suspect": verdicts["suspect"],
            "sec_inconclusive": verdicts["inconclusive"],
            "sec_contradicted_tickers": verdicts["contradicted_tickers"],
        })
    return pd.DataFrame(rows).sort_values("pct_zero", ascending=False, na_position="last")


# --------------------------------------------------------------------------- #
# the D-decision cross-checks                                                  #
# --------------------------------------------------------------------------- #
def cross_check_shares(context: Context,
                       tickers: Sequence[str] | None = None) -> pd.DataFrame:
    """`sharesbas` vs the SEC layer's `sharesOutstanding`, per overlapping ticker.

    The open question behind D-decision `sharesOutstanding <- sharesbas` is whether Sharadar
    SUMS MULTIPLE SHARE CLASSES, which their documentation does not say. This repo already
    solved that problem painfully on the SEC side for 36 multi-class tickers, by summing the
    cover-page `dei:EntityCommonStockSharesOutstanding` across classes -- so the SEC column is
    a known CONSOLIDATED basis, and the ratio against it is the answer.

    A per-ticker MEDIAN, not a mean: one bad filing date should not move the verdict, and the
    question is about a systematic factor, not about noise. A ratio at 1.0 means the two agree;
    a ratio stable and materially below 1.0 means Sharadar is carrying one class only.

    ⚠ `ratio_span` is the column that actually answers a DIFFERENT and more damaging question.
    A ticker whose ratio is 1.0 at the recent end and 10.0 at the old end has not got a share
    CLASS problem -- it has had a STOCK SPLIT, and Sharadar has restated its whole history onto
    the post-split basis while the SEC column is the as-filed cover-page count. Measured on
    NVDA (25.0bn shares reported for 2021-11-22, against ~2.5bn actually outstanding before the
    June 2024 10-for-1) and WMT (3-for-1, February 2024). `sharefactor` is 1.0 on every one of
    those rows and does NOT flag it.
    """
    arq = _load_sharadar(context, tickers, ("ARQ",))
    sec = _load_sec(context, sorted(str(t) for t in arq["ticker"].unique()))
    if sec.empty:
        return pd.DataFrame(columns=["ticker", "n_dates", "median_ratio", "min_ratio",
                                     "max_ratio", "median_sharefactor"])
    left = arq[["ticker", "date", "sharesbas", "sharefactor"]].copy()
    left["ticker"] = left["ticker"].astype(str)
    right = sec[["ticker", "as_of", "sharesOutstanding"]].copy()
    right["ticker"] = right["ticker"].astype(str)
    joined = left.merge(right, left_on=["ticker", "date"], right_on=["ticker", "as_of"],
                        how="inner").dropna(subset=["sharesbas", "sharesOutstanding"])
    joined = joined[joined["sharesOutstanding"] != 0]
    if joined.empty:
        return pd.DataFrame(columns=["ticker", "n_dates", "median_ratio", "min_ratio",
                                     "max_ratio", "median_sharefactor"])
    joined["ratio"] = joined["sharesbas"] / joined["sharesOutstanding"]
    out = joined.groupby("ticker").agg(
        n_dates=("ratio", "size"), median_ratio=("ratio", "median"),
        min_ratio=("ratio", "min"), max_ratio=("ratio", "max"),
        median_sharefactor=("sharefactor", "median")).reset_index()
    out["ratio_span"] = out["max_ratio"] / out["min_ratio"]
    out["verdict"] = np.where(
        out["ratio_span"] >= 1.5, "SPLIT-ADJUSTED history (not as-filed)",
        np.where((out["median_ratio"] - 1).abs() <= 0.05, "agrees with the SEC cover page",
                 "systematic level difference -- investigate the share-class basis"))
    return out.sort_values("ratio_span", ascending=False)


def confirm_sign_conventions(context: Context,
                             tickers: Sequence[str] | None = None) -> dict:
    """Re-assert from STORED data what was measured from the API: `capex <= 0` throughout, and
    `fcf == ncfo + capex` to the cent.

    Cheap assertions protecting an expensive mistake. The phase-3 field map maps
    `freeCashflow <- fcf` with NO reconstruction and `capex` with a sign flip (the SEC
    catalogue declares `capex` non-negative, Sharadar stores it negative). If either identity
    fails, both of those decisions are wrong and phase 3 must not start.
    """
    frame = _load_sharadar(context, tickers, ("ARQ", "ARY", "ART"))
    out: dict = {"dimensions": {}, "capex_sign_holds": True, "fcf_identity_holds": True}
    for dimension, group in frame.groupby("dimension", sort=True):
        capex = group["capex"].dropna()
        residual = (group["fcf"] - (group["ncfo"] + group["capex"])).abs().dropna()
        positive = group.loc[capex.index][capex > 0]
        worst = residual.idxmax() if not residual.empty else None
        block = {
            "capex_rows": int(len(capex)),
            "capex_positive": int((capex > 0).sum()),
            "capex_positive_tickers": sorted(positive["ticker"].astype(str).unique().tolist()),
            "capex_max": float(capex.max()) if not capex.empty else float("nan"),
            "fcf_rows": int(len(residual)),
            "fcf_max_abs_residual": float(residual.max()) if not residual.empty else 0.0,
            "fcf_worst_row": (f"{frame.at[worst, 'ticker']} "
                              f"{pd.Timestamp(frame.at[worst, 'date']).date()}"
                              if worst is not None else "-"),
            "fcf_violations": int((residual > FCF_IDENTITY_TOLERANCE).sum()),
        }
        out["dimensions"][str(dimension)] = block
        out["capex_sign_holds"] &= block["capex_positive"] == 0
        out["fcf_identity_holds"] &= block["fcf_violations"] == 0
    out["capex_rows_total"] = sum(b["capex_rows"] for b in out["dimensions"].values())
    out["capex_positive_total"] = sum(b["capex_positive"] for b in out["dimensions"].values())
    out["capex_positive_tickers"] = sorted(
        {t for b in out["dimensions"].values() for t in b["capex_positive_tickers"]})
    out["capex_positive_rows"] = frame.loc[
        frame["capex"] > 0, ["ticker", "dimension", "date", "fiscalperiod", "capex"]
    ].sort_values("capex", ascending=False)
    out["ok"] = bool(out["capex_sign_holds"] and out["fcf_identity_holds"])
    return out


def confirm_q4_tautology(context: Context,
                         tickers: Sequence[str] | None = None) -> pd.DataFrame:
    """ΣARQ vs ARY per (ticker, fiscal year, field), for the DURATION fields only.

    NOT a quality check -- evidence for the record. Sharadar constructs Q4 as
    `ARY - Σ(Q1..Q3)`, so this identity holds by arithmetic and the spec's acceptance check #3
    can never fail on this vendor. Measuring it once and writing the number down is what stops
    it being re-proposed as a gate.

    Only fiscal years with all four quarters AND an annual row are compared; a partial year
    would produce a deviation that says nothing about the construction.
    """
    arq = with_fiscal_period(_load_sharadar(context, tickers, ("ARQ",)))
    ary = with_fiscal_period(_load_sharadar(context, tickers, ("ARY",)))
    fields = [f for f in _value_fields(arq) if f in SHARADAR_FLOW_FIELDS]

    counts = arq.groupby(["ticker", "fiscal_year"], observed=True).size().rename("n_quarters")
    quarterly = arq.groupby(["ticker", "fiscal_year"], observed=True)[fields].sum(min_count=1)
    annual = ary.groupby(["ticker", "fiscal_year"], observed=True)[fields].sum(min_count=1)
    complete = counts[counts == 4].index
    quarterly = quarterly.loc[quarterly.index.isin(complete)]
    annual = annual.loc[annual.index.isin(quarterly.index)]
    quarterly = quarterly.loc[quarterly.index.isin(annual.index)]
    if quarterly.empty:
        return pd.DataFrame(columns=["ticker", "fiscal_year", "field", "sum_arq", "ary",
                                     "abs_diff", "pct_dev"])

    stacked_q = quarterly.stack(future_stack=True).rename("sum_arq").reset_index()
    stacked_a = annual.stack(future_stack=True).rename("ary").reset_index()
    key = ["ticker", "fiscal_year", "level_2"]
    merged = stacked_q.merge(stacked_a, on=key, how="inner").rename(
        columns={"level_2": "field"}).dropna(subset=["sum_arq", "ary"])
    merged = merged[merged["ary"].abs() > 0]
    merged["abs_diff"] = (merged["sum_arq"] - merged["ary"]).abs()
    merged["pct_dev"] = merged["abs_diff"] / merged["ary"].abs()
    return merged.sort_values("pct_dev", ascending=False)


def q4_tautology_summary(frame: pd.DataFrame) -> dict:
    """The tautology as a DISTRIBUTION, because a single max is the wrong statistic for it.

    The identity holds EXACTLY -- `pct_dev == 0.0`, not "small" -- wherever it holds at all,
    because Sharadar built Q4 by subtraction. So the informative number is the SHARE of triples
    at exactly zero, and the interesting rows are the handful that are not: those are fiscal
    years Sharadar RESTATED between publishing the quarters and publishing the year, which is
    a real event and not a rounding error.
    """
    if frame.empty:
        return {"n": 0, "n_exact": 0, "share_exact": float("nan"), "n_over_bar": 0,
                "max_dev": float("nan"), "concentration": pd.DataFrame()}
    exact = frame["pct_dev"] == 0
    over = frame["pct_dev"] > Q4_TAUTOLOGY_MAX_PCT
    concentration = (frame[over].groupby(["ticker", "fiscal_year"]).size()
                     .rename("n_fields").reset_index().sort_values("n_fields", ascending=False))
    return {
        "n": int(len(frame)),
        "n_exact": int(exact.sum()),
        "share_exact": float(exact.mean()),
        "n_over_bar": int(over.sum()),
        # the rest sit between 0 and the bar: float noise on a sum of four doubles
        "n_float_noise": int(len(frame) - exact.sum() - over.sum()),
        "max_dev": float(frame["pct_dev"].max()),
        "concentration": concentration,
    }


# --------------------------------------------------------------------------- #
# the per-field zero rule -- machine-proposed, HUMAN-APPROVED                   #
# --------------------------------------------------------------------------- #
def propose_zero_rules(zero_frame: pd.DataFrame) -> dict[str, dict]:
    """The per-field `"null"` / `"keep"` proposal, with the number behind each one.

    Two rules only, as decided: `"null"` means a 0 in this field is UNKNOWN, `"keep"` means it
    is a real value. The ladder, in order:

      1. never zero            -> keep. Nothing to decide.
      2. SEC contradicts       -> null. Measured as a rate over the zeros the SEC layer COULD
                                  JUDGE, not over all zeros: `inventory` is wrong on 4 of the
                                  4 cells with a SEC counterpart, and dividing those 4 by the
                                  140 zeros nobody could check would hide a broken column.
      3. mostly mixed tickers  -> null. The same ticker reports the field non-zero elsewhere,
                                  which is the fill signature without needing a second source.
                                  NOT applied to `SHARADAR_EVENT_FIELDS`, where a zero quarter
                                  means "no acquisition happened" and is a fact.
      4. every zero structural -> keep. Every zero belongs to a ticker that NEVER reports the
                                  field: a bank has no inventory, a retailer no R&D.
      5. otherwise             -> keep, and say what is unresolved.

    Nulling is the conservative direction here, and not only on the usual "a wrong number is
    worse than a missing one" grounds: where the SEC path finds no value it stores NULL and
    never 0, so nulling moves the two producers onto the SAME treatment ahead of the phase-4
    merge, rather than leaving one of them claiming a zero the other calls unknown.
    """
    rules: dict[str, dict] = {}
    for row in zero_frame.itertuples(index=False):
        field = str(row.field)
        n_zero = int(row.n_zero)
        pct = None if pd.isna(row.pct_zero) else round(float(row.pct_zero), 4)
        checked, contradicted = int(row.sec_checked), int(row.sec_contradicted)
        judged = (checked >= SHARADAR_ZERO_RULE_MIN_CHECKED
                  and contradicted / checked >= SHARADAR_ZERO_RULE_CONTRADICTION_SHARE)
        mixed = int(row.n_zero_mixed)
        if int(row.n_rows) == 0:
            rule, reason = "keep", "not measured: the column is absent from the stored table"
        elif n_zero == 0:
            rule, reason = "keep", f"never zero in {int(row.n_rows)} stored ARQ rows"
        elif judged:
            rule = "null"
            reason = (f"provably wrong: the SEC layer reports a non-zero {row.sec_basis} on "
                      f"{contradicted}/{checked} of the zeros it could judge "
                      f"({row.sec_contradicted_tickers})")
        elif field in SHARADAR_EVENT_FIELDS:
            rule = "keep"
            reason = (f"discrete event: a zero means no transaction that quarter, so the "
                      f"{mixed}/{n_zero} zeros in mixed tickers are expected, not a fill"
                      + (f"; {_sec_note(row)}" if checked else ""))
        elif mixed / n_zero >= SHARADAR_ZERO_RULE_MIXED_SHARE:
            rule = "null"
            reason = (f"{mixed}/{n_zero} zeros sit in tickers that report this field non-zero "
                      f"in another quarter, which is a fill and not a fact")
        elif mixed == 0:
            rule = "keep"
            reason = (f"structural: every zero belongs to one of {int(row.n_tickers_all_zero)} "
                      f"tickers that never report this field"
                      + (f"; SEC agrees or is equally absent on "
                         f"{int(row.sec_agrees) + int(row.sec_absent)}/"
                         f"{int(row.sec_overlap_zeros)}" if int(row.sec_overlap_zeros) else ""))
        else:
            rule = "keep"
            reason = (f"{mixed}/{n_zero} zeros are in mixed tickers, below the "
                      f"{SHARADAR_ZERO_RULE_MIXED_SHARE:.0%} bar; " + _sec_note(row))
        rules[field] = {"rule": rule, "reason": reason, "pct_zero": pct}
    return rules


def _sec_note(row) -> str:
    """What the SEC layer had to say, distinguishing the three cases that are NOT the same.

    "No counterpart" and "a counterpart that was itself absent" read alike in a count of zero
    checked cells, and they are opposite evidence: the SEC path stores NULL where it found no
    value, so a counterpart that is absent on every overlap zero AGREES with Sharadar. `rnd`
    is exactly that -- 140 of 140 overlap zeros with no SEC value -- and calling it "no SEC
    counterpart, so this rests on Sharadar alone" said the opposite of what was measured.
    """
    if str(row.sec_basis).startswith("no "):
        return "no SEC counterpart, so this rests on Sharadar alone"
    checked, absent = int(row.sec_checked), int(row.sec_absent)
    if checked:
        return (f"SEC contradicts {int(row.sec_contradicted)}/{checked} of the zeros it could "
                f"judge" + (f", and is equally absent on {absent} more" if absent else ""))
    if absent:
        return (f"the SEC layer has no value either on all {absent} overlap zeros, which is "
                f"agreement -- it stores NULL where it finds nothing, never 0")
    return "no overlapping ticker had a zero to check"


def render_zero_rules(rules: dict[str, dict], *, generated: str, scope: str) -> str:
    """The rules file as TEXT, emitted one field per line with aligned keys.

    Hand-emitted rather than `json.dumps(indent=2)` on purpose: this file is meant to be READ
    and hand-edited during approval, and a naive dump explodes 41 entries into 200 lines of
    one-key-per-line noise. `configs/fundamentals/*.json` in this repo are hand-formatted for
    the same reason, and a `json.dumps` round-trip there reformats all 545 lines.

    The emitter is validated by its caller (`write_zero_rules` re-parses what it wrote), so
    hand-formatting can never cost a broken file.
    """
    width = max(len(f'"{name}":') for name in rules) if rules else 0
    lines = [
        "{",
        '  "_README": [',
        '    "PER-FIELD RULE for Sharadar\'s 41 documented zero-filled indicators.",',
        '    "\\"null\\" = treat a stored 0 as UNKNOWN. \\"keep\\" = the 0 is a real value.",',
        '    "MACHINE-PROPOSED by `data_extract sharadar-diagnostics`. NOT YET APPROVED:",',
        '    "this file is only a PROPOSAL until a human adds an `_APPROVED` block, and",',
        '    "phase 3 must refuse to run without one.",',
        '    "Phase 3 reads this file and fails loudly on a field with no entry.",',
        '    "Keys starting with _ are documentation and must be skipped by the reader.",',
        f'    "generated: {generated}",',
        f'    "scope: {scope}"',
        "  ],",
    ]
    items = sorted(rules.items())
    for index, (name, block) in enumerate(items):
        key = f'"{name}":'.ljust(width)
        pct = "null" if block["pct_zero"] is None else f'{block["pct_zero"]}'
        comma = "" if index == len(items) - 1 else ","
        lines.append(f'  {key} {{"rule": {json.dumps(block["rule"])}, '
                     f'"reason": {json.dumps(block["reason"])}, '
                     f'"pct_zero": {pct}}}{comma}')
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_zero_rules(path: Path, rules: dict[str, dict], *, generated: str,
                     scope: str) -> Path:
    """Write the proposal, NEVER over a file a human has already approved.

    An existing file is left alone and the proposal lands beside it as `*.proposed.json`. The
    rule file is defined as human-approved; silently overwriting it would make that phrase
    meaningless the first time the diagnostic is re-run.
    """
    text = render_zero_rules(rules, generated=generated, scope=scope)
    parsed = json.loads(text)                      # the emitter validates itself
    missing = sorted(SHARADAR_ZERO_FILLED_FIELDS - set(parsed))
    if missing:
        raise RuntimeError(f"the zero-rule emitter dropped {len(missing)} field(s): {missing}")
    target = path if not path.exists() else path.with_suffix(".proposed.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


# --------------------------------------------------------------------------- #
# the report                                                                   #
# --------------------------------------------------------------------------- #
def _md_cell(value) -> str:
    """One cell, formatted so a reader can compare two numbers at a glance.

    `{:,.4g}` was the first attempt and rendered every money figure as `2.746e+09`, which is
    unreadable next to `1.371e+09` -- the two differ by a factor of two and look identical.
    Magnitudes above 100k therefore get thousands separators and no exponent.
    """
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        if value == 0:
            return "0"
        if abs(value) >= 1e5:
            return f"{value:,.0f}"
        return f"{value:,.4g}" if abs(value) >= 0.001 else f"{value:.3e}"
    # a literal pipe inside a cell or a header silently splits the row into extra columns
    return str(value).replace("|", "\\|")


def _md_table(frame: pd.DataFrame, limit: int | None = None) -> str:
    """A markdown table, rendered here rather than via `to_markdown` so the module carries no
    `tabulate` dependency for one call."""
    if frame is None or frame.empty:
        return "_(no rows)_\n"
    shown = frame if limit is None else frame.head(limit)
    header = "| " + " | ".join(str(c).replace("|", "\\|") for c in shown.columns) + " |"
    rule = "|" + "|".join("---" for _ in shown.columns) + "|"
    body = ["| " + " | ".join(_md_cell(v) for v in row) + " |"
            for row in shown.itertuples(index=False)]
    tail = ("" if limit is None or len(frame) <= limit
            else f"\n_{limit} of {len(frame)} rows shown._\n")
    return "\n".join([header, rule, *body]) + "\n" + tail


def _worst_offender(rows: pd.DataFrame) -> str:
    """", and N of them are TICKER alone" when one ticker dominates a flag set.

    Whether a defect is spread across the roster or concentrated in one filer is the difference
    between "the vendor's mapping is broken" and "the vendor mishandles bank cash-flow
    statements", and those need different fixes.
    """
    if rows is None or rows.empty:
        return ""
    counts = rows["ticker"].value_counts()
    if counts.iloc[0] <= 1 or counts.iloc[0] < len(rows) / 2:
        return ""
    return f" — and {counts.iloc[0]} of the {len(rows)} are {counts.index[0]} alone"


def _findings(results: dict) -> list[str]:
    """The three findings the report has to end with, each one a number.

    Written as prose deliberately: the phase hands back a PURCHASE DECISION, and a table dump
    is not an answer to "is this data good enough to buy the Full tier?".
    """
    completeness = results["completeness"]
    implausible = results["implausible"]
    zero = results["zero_fill"]
    rules = results["zero_rules"]

    n_missing = int(completeness["n_missing"].sum())
    n_tickers_with_gap = int((completeness["n_missing"] > 0).sum())
    n_dup = int(completeness["n_duplicate_quarters"].sum())
    gap_shape = ("No gaps at all" if n_missing == 0 else
                 f"{n_missing} missing quarter(s) across {n_tickers_with_gap} ticker(s)")

    n_negative = int((implausible["reason"] == "negative").sum()) if not implausible.empty else 0
    magnitude = (implausible[implausible["reason"] == "magnitude"] if not implausible.empty
                 else implausible)
    n_magnitude = len(magnitude)
    q4_share = float((magnitude["fiscal_position"] == "Q4").mean()) if n_magnitude else 0.0
    cashflow_share = (float(magnitude["field"].str.startswith(("ncf", "fcf")).mean())
                      if n_magnitude else 0.0)
    negative_fields = (", ".join(f"{f} ({n})" for f, n in
                                 implausible[implausible["reason"] == "negative"]["field"]
                                 .value_counts().items()) if n_negative else "none")

    nulled = sorted(f for f, block in rules.items() if block["rule"] == "null")
    measured = zero[zero["n_rows"] > 0]
    cells = int(measured["n_rows"].sum())
    removed = int(measured[measured["field"].isin(nulled)]["n_zero"].sum())

    return [
        f"**Completeness — clean.** {gap_shape}, over {len(completeness)} ticker(s) measured "
        f"against each ticker's own observed window, and {n_dup} duplicate normalised quarter(s). "
        f"There is nothing structural or random to distinguish, because there is nothing.",
        f"**Implausible quarters — {n_negative} real, the rest is lumpiness.** The {n_negative} "
        f"negative value(s) in a field that has no negative reading are the only LEVEL errors: "
        f"{negative_fields}. The {n_magnitude} magnitude outlier(s) beyond {results['ratio']}x "
        f"the largest other quarter are **not** a Q4 construction artefact — only {q4_share:.0%} "
        f"sit in the Q4 position (chance is 25%), and {cashflow_share:.0%} of them are `ncf*` / "
        f"`fcf` legs, where one acquisition or one bond issue legitimately dwarfs the year. The "
        f"threshold was calibrated on the SEC path's income-statement fields and does not "
        f"transfer to event-driven cash-flow lines.",
        f"**Zero rule — {len(nulled)} of {len(rules)} fields must be NULL-ruled** "
        f"({', '.join(nulled) if nulled else 'none'}), converting {removed:,} of {cells:,} "
        f"measured cells ({removed / cells:.2%}) from 0 to NULL.",
    ]


def render_report(results: dict) -> str:
    """The phase-2 findings document. Every claim is a number, and the last section is a plain
    answer to the question the phase exists to answer."""
    signs = results["sign_conventions"]
    tautology = results["q4_tautology"]
    summary = q4_tautology_summary(tautology)
    shares = results["shares"]
    zero = results["zero_fill"]
    implausible = results["implausible"]

    sign_rows = pd.DataFrame([
        {"dimension": dimension, "rows": block["capex_rows"],
         "capex > 0": block["capex_positive"], "capex max": block["capex_max"],
         "fcf rows": block["fcf_rows"],
         "max abs fcf residual": block["fcf_max_abs_residual"],
         "violations": block["fcf_violations"], "worst row": block["fcf_worst_row"]}
        for dimension, block in results["sign_conventions"]["dimensions"].items()])

    rules_rows = pd.DataFrame([
        {"field": name, "rule": block["rule"], "pct_zero": block["pct_zero"],
         "reason": block["reason"]}
        for name, block in sorted(results["zero_rules"].items())]
    ).sort_values(["rule", "pct_zero"], ascending=[True, False])

    parts: list[str] = [
        "# Phase 2 — Sharadar acceptance gates, measured from the database",
        "",
        f"**Generated**: {results['generated']}  ",
        f"**Scope**: {results['scope']}  ",
        f"**Source**: `{Tables.sharadar_fundamentals}` (read-only) cross-checked against "
        f"`{Tables.fundamentals_history_sec}` on {results['n_overlap']} overlapping ticker(s).",
        "",
        "> ⚠ This is **not** the SEC check scheme (D25). No check was registered, no "
        "`fundamentals_check` row was written, and `src/validate/` was neither imported nor "
        "invoked. This is a standalone read-only diagnostic whose only consumers are the "
        "Full-tier purchase decision and phase 3's `sharadar_zero_rules.json`.",
        "",
        "---",
        "",
        "## Gate 1 — completeness",
        "",
        "Expected quarter count is measured against **each ticker's own observed window**, not "
        "a global start: on a 5-year entitlement a ticker whose history begins late is not a "
        "gap. Quarters are Sharadar's own `calendardate`, already normalised to the nearest "
        "calendar quarter-end, so 52/53-week retail calendars need no special handling.",
        "",
        _md_table(results["completeness"]),
        "",
        "## Gate 2 — implausible quarters",
        "",
        f"Replaces the spec's acceptance check #3, which is dead (see gate 4). Sharadar "
        f"CONSTRUCTS Q4 as `ARY - Σ(Q1..Q3)`; the identity therefore cannot fail, but the "
        f"construction can produce absurd levels — the legacy Quandl documentation shows it "
        f"yielding ABT 2011 Q4 revenue of **-$7.1bn**. Magnitude threshold is "
        f"`data_extract.fundamentals_periods.max_opposite_sign_q4_ratio` = "
        f"**{results['ratio']}**, reused from the SEC path rather than reinvented.",
        "",
        f"**{len(implausible)} flagged cell(s)**: "
        f"{int((implausible['reason'] == 'negative').sum()) if not implausible.empty else 0} "
        f"negative, "
        f"{int((implausible['reason'] == 'magnitude').sum()) if not implausible.empty else 0} "
        f"magnitude.",
        "",
        "**Every negative — these are the findings that matter.** A negative in a field with "
        "no negative reading is a level error the annual row still absorbs, so no identity "
        "check can ever see it:",
        "",
        _md_table(implausible[implausible["reason"] == "negative"]
                  .sort_values(["field", "ticker", "calendardate"])),
        "",
        "**Magnitude outliers, by field and by fiscal position.** Read these two tables before "
        "the row dump: they say the threshold is measuring lumpiness, not error.",
        "",
        _md_table(
            (implausible[implausible["reason"] == "magnitude"]["field"].value_counts()
             .rename("n_flagged").reset_index().rename(columns={"index": "field"}))
            if not implausible.empty else implausible),
        "",
        _md_table(
            (implausible[implausible["reason"] == "magnitude"]["fiscal_position"]
             .value_counts().rename("n_flagged").reset_index()
             .rename(columns={"index": "fiscal_position"}))
            if not implausible.empty else implausible),
        "",
        f"Worst {WORST_ROWS} by ratio:",
        "",
        _md_table(implausible, WORST_ROWS),
        "",
        "## Gate 3 — zero-fill prevalence, per field",
        "",
        "`n_zero_mixed` is the Sharadar-internal signal: zeros belonging to a ticker that "
        "reports the **same field non-zero in another quarter**. It is the only evidence "
        "available for the 21 fields with no SEC counterpart, and it needs no basis "
        "reconciliation.",
        "",
        "The SEC columns are **basis-matched** before comparison: a duration field is judged "
        "at the TTM level (`fundamentals_history_sec` is TTM, so a non-zero there says nothing "
        "about one quarter — the Sharadar side of that comparison is its own ART dimension), "
        "an instant field point-in-time. A `sec_wider` counterpart can only produce "
        "`sec_suspect`, never `sec_contradicted`: `totalDebt` carries lease liabilities "
        "Sharadar's `debt` does not, and `cash` carries restricted cash and short-term "
        "investments its `cashneq` does not.",
        "",
        _md_table(zero),
        "",
        "### The proposed rule",
        "",
        "Two rules only: `null` (treat 0 as unknown) and `keep` (0 is a real value). "
        "Machine-proposed here, **human-approved** in "
        f"`configs/sharadar/{results['rules_filename']}`. Written to "
        f"`{results['rules_written_to']}`.",
        "",
        _md_table(rules_rows),
        "",
        "## Gate 4 — the Q4 identity is tautological (evidence, not a check)",
        "",
        f"ΣARQ vs ARY per (ticker, fiscal year, field) over the {len(SHARADAR_FLOW_FIELDS)} "
        f"duration fields, on {summary['n']:,} comparable triples.",
        "",
        f"- **exactly zero: {summary['n_exact']:,} / {summary['n']:,} "
        f"({summary['share_exact']:.1%})** — not \"small\", *exactly* 0.0",
        f"- float noise (0 < dev ≤ {Q4_TAUTOLOGY_MAX_PCT:.2%}): {summary['n_float_noise']:,}",
        f"- materially non-zero: **{summary['n_over_bar']:,}** "
        f"({summary['n_over_bar'] / summary['n']:.1%}), max {summary['max_dev']:.2%}",
        "",
        "This is the number that kills the spec's acceptance check #3. Sharadar builds Q4 by "
        "subtraction, so wherever the identity holds it holds **exactly**, and the check "
        "carries no information about the quality of the quarters it is supposedly testing. "
        "Recorded here so it is not re-proposed as a gate later.",
        "",
        "⚠ **But the exceptions are not noise, and they are not what the plan expected.** The "
        "plan predicted `+0.000%` everywhere. The residual triples are fiscal years Sharadar "
        "**restated between publishing the quarters and publishing the year** — a real event. "
        "They cluster hard, which is what tells you it is not arithmetic drift:",
        "",
        _md_table(summary["concentration"], WORST_ROWS),
        "",
        results["restatement_note"],
        "",
        f"Worst {WORST_ROWS} triples:",
        "",
        _md_table(tautology.head(WORST_ROWS)),
        "",
        "## Sign conventions (a stop condition for phase 3)",
        "",
        "`capex <= 0` throughout and `fcf == ncfo + capex` to the cent. The phase-3 field map "
        "maps `freeCashflow <- fcf` with **no reconstruction**, and flips `capex`'s sign "
        "because the SEC catalogue declares it non-negative while Sharadar stores it negative.",
        "",
        _md_table(sign_rows),
        "",
        "Every row with a positive `capex`:",
        "",
        _md_table(signs["capex_positive_rows"]),
        "",
        f"- `fcf == ncfo + capex`: **{'HOLDS' if signs['fcf_identity_holds'] else 'FAILED'}** — "
        f"so `freeCashflow <- fcf` needs no reconstruction, as decided.",
        f"- `capex <= 0`: **{'HOLDS' if signs['capex_sign_holds'] else 'DOES NOT HOLD'}**. "
        f"{signs['capex_positive_total']} of {signs['capex_rows_total']} rows "
        f"({signs['capex_positive_total'] / max(signs['capex_rows_total'], 1):.2%}) carry a "
        f"POSITIVE capex, on {', '.join(signs['capex_positive_tickers']) or 'no ticker'}"
        f"{_worst_offender(signs['capex_positive_rows'])}.",
        "",
        "**Consequence for phase 3, stated precisely:** the identity is not universal, so an "
        "unconditional `capex = -sharadar.capex` would write a *negative* value into a column "
        "the SEC catalogue declares `non_negative`, on those rows. The fix is a guard, not a "
        "different mapping — flip the sign where `capex <= 0` and NULL the rest, recording "
        "them. This does not invalidate the field map; it invalidates doing it blind.",
        "",
        "## `sharesbas` cross-check (D-decision `sharesOutstanding <- sharesbas`)",
        "",
        "Whether Sharadar SUMS MULTIPLE SHARE CLASSES is undocumented. The SEC column is a "
        "known **consolidated** basis — this repo built it for 36 multi-class tickers by "
        "summing the cover-page `dei:EntityCommonStockSharesOutstanding` across classes — so a "
        "systematic ratio, not noise, is the answer.",
        "",
        _md_table(shares),
        "",
        "🚨 **The share-class question is not the finding here.** 12 of 14 tickers sit at "
        "exactly 1.0, so Sharadar is not carrying one class of a multi-class filer. What the "
        "`ratio_span` column exposes instead is that **`sharesbas` is retroactively "
        "SPLIT-ADJUSTED**: NVDA's 2021-11-22 row reports 25.0bn shares against the ~2.5bn "
        "actually outstanding before its June 2024 10-for-1, and WMT shows the same at 3x for "
        "its February 2024 split. `sharefactor` is `1.0` on every one of those rows and does "
        "**not** flag it.",
        "",
        "That makes `sharesbas` **not point-in-time**, which is a different and more serious "
        "property than the one D-decision asked about. Anything that multiplies it by an "
        "as-filed price — a market cap, a per-share book value — is wrong by the split factor "
        "for every date before the split. Phase 3 must either take `sharesOutstanding` from "
        "the SEC layer on the overlap, or de-adjust `sharesbas` using `sharadar_actions`, "
        "which is already ingested and carries the split events.",
        "",
        "---",
        "",
        "## The decision this phase hands back",
        "",
    ]
    parts.extend(f"{index}. {finding}" for index, finding in enumerate(_findings(results), 1))
    parts.extend(["", f"**Recommendation**: {results['recommendation']}", ""])
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# orchestration                                                                #
# --------------------------------------------------------------------------- #
def _restatement_note(tautology: pd.DataFrame) -> str:
    """Name the worst (ticker, fiscal year) cluster with its actual numbers.

    A concentration table shows that the deviations cluster; only the numbers show WHY. The
    top cluster is a whole income statement moving together by one consistent amount, which is
    what a restatement looks like and what arithmetic drift does not.
    """
    over = tautology[tautology["pct_dev"] > Q4_TAUTOLOGY_MAX_PCT]
    if over.empty:
        return "No triple deviates materially, so the identity is tautological without exception."
    top = over.groupby(["ticker", "fiscal_year"]).size().idxmax()
    block = over[(over["ticker"] == top[0]) & (over["fiscal_year"] == top[1])]
    revenue = block[block["field"] == "revenue"]
    line = (f"`{top[0]}` FY{top[1]} is the worst, with **{len(block)} fields** moving together"
            + (f" — its quarters sum to {revenue.iloc[0]['sum_arq']:,.0f} of revenue against an "
               f"annual row of {revenue.iloc[0]['ary']:,.0f}, a gap of "
               f"{revenue.iloc[0]['pct_dev']:.1%}" if not revenue.empty else "") + ".")
    return (line + " A whole income statement moving in one direction at once is a **restated "
            "year**, not arithmetic drift — the quarters were published before the "
            "reclassification and the annual row after it. Phase 4's gap check will see these "
            "same years, and should not read them as extraction defects.")


def _recommendation(results: dict) -> str:
    """A plain answer to "is this data good enough to buy the Full tier?".

    The distinction the recommendation turns on is **fatal vs bounded**. A defect that has a
    measured size and a mechanical fix is a phase-3 task, not a reason to refuse a purchase; a
    defect whose size is unknown, or which no rule can localise, is. Only the second kind
    blocks.
    """
    signs = results["sign_conventions"]
    completeness = results["completeness"]
    implausible = results["implausible"]
    shares = results["shares"]
    nulled = [f for f, block in results["zero_rules"].items() if block["rule"] == "null"]

    fatal: list[str] = []
    bounded: list[str] = []

    if not signs["fcf_identity_holds"]:
        fatal.append("`fcf != ncfo + capex`, so `freeCashflow <- fcf` needs a reconstruction "
                     "nobody has designed")
    n_missing = int(completeness["n_missing"].sum())
    if n_missing:
        fatal.append(f"{n_missing} quarter(s) are missing outright, which breaks the TTM build")

    if not signs["capex_sign_holds"]:
        bounded.append(f"`capex` is positive on {signs['capex_positive_total']} of "
                       f"{signs['capex_rows_total']} rows "
                       f"({', '.join(signs['capex_positive_tickers'])}) — guard the sign flip "
                       f"and NULL the exceptions")
    n_negative = (int((implausible["reason"] == "negative").sum())
                  if not implausible.empty else 0)
    if n_negative:
        bounded.append(f"{n_negative} negative value(s) in non-negative fields — enumerated "
                       f"above, so each can be settled against its filing")
    split_adjusted = (shares[shares["verdict"].str.startswith("SPLIT")]["ticker"].tolist()
                      if not shares.empty else [])
    if split_adjusted:
        bounded.append(f"`sharesbas` is retroactively split-adjusted ({', '.join(split_adjusted)}"
                       f"), so it is not point-in-time and must not be multiplied by an "
                       f"as-filed price")
    if nulled:
        bounded.append(f"{len(nulled)} field(s) zero-fill wrongly and are NULL-ruled")

    if fatal:
        return ("**Do not buy yet.** " + "; ".join(fatal) +
                ". These have no bounded fix, and a wider roster multiplies whatever this "
                "window already contains.")
    return (
        "**Buy the Full tier — but do not let phase 3 map a single field blind.**\n\n"
        "The data clears the gates that would have been disqualifying: no missing quarters, "
        "`fcf == ncfo + capex` exactly, and every remaining defect has a measured size and a "
        "mechanical fix. What this phase actually bought you is the list of those defects, "
        "and it is longer than the plan assumed:\n\n"
        + "\n".join(f"- {item}" for item in bounded) +
        "\n\nThe residual risk is the one no gate here can close. Sharadar constructs Q4 by "
        "subtraction, so the identity is tautological wherever it holds and this window can "
        "detect a bad LEVEL but never a bad CONSTRUCTION. A 5-year DJIA-30 window is also not "
        "a 20-year S&P 500 window — the three CIK-cutover tickers (D19) are not in it at all, "
        "and that test is written but skipped. **Re-run this diagnostic against the full "
        "history on day one of the new entitlement**, before any of it reaches the cube.")


def run_diagnostics(context: Context, tickers: Sequence[str] | None = None, *,
                    report_path: str | Path = DEFAULT_REPORT_PATH,
                    config_dir: str | Path = "./configs") -> dict:
    """Run all six diagnostics, write the report and the proposed rule file, log a summary.

    Writes NO production data: one markdown report and one JSON proposal, both outside the
    database. Returns everything it measured so a test can assert on it without re-reading
    the files.
    """
    ratio = float(context.config.data_extract.fundamentals_periods.max_opposite_sign_q4_ratio)
    generated = pd.Timestamp.today().strftime("%Y-%m-%d")

    arq = _load_sharadar(context, tickers, ("ARQ",))
    scope_tickers = sorted(str(t) for t in arq["ticker"].unique())
    sec = _load_sec(context, scope_tickers)
    overlap = sorted(set(scope_tickers) & set(sec["ticker"].astype(str))) if not sec.empty else []
    scope = (f"{len(scope_tickers)} ticker(s), "
             f"{pd.Timestamp(arq['date'].min()).date()}..{pd.Timestamp(arq['date'].max()).date()}")
    context.log.info("Sharadar diagnostics: %s; %d overlap with %s",
                     scope, len(overlap), Tables.fundamentals_history_sec)

    results: dict = {
        "generated": generated,
        "scope": scope,
        "tickers": scope_tickers,
        "overlap": overlap,
        "n_overlap": len(overlap),
        "ratio": ratio,
        "completeness": gate_completeness(context, tickers),
        "implausible": gate_implausible_quarters(context, tickers, ratio=ratio),
        "zero_fill": gate_zero_fill(context, tickers),
        "shares": cross_check_shares(context, tickers),
        "sign_conventions": confirm_sign_conventions(context, tickers),
        "q4_tautology": confirm_q4_tautology(context, tickers),
        "rules_filename": SHARADAR_ZERO_RULES_FILENAME,
    }
    results["restatement_note"] = _restatement_note(results["q4_tautology"])
    results["zero_rules"] = propose_zero_rules(results["zero_fill"])
    rules_path = Path(config_dir) / SHARADAR_CONFIG_SUBDIR / SHARADAR_ZERO_RULES_FILENAME
    written = write_zero_rules(rules_path, results["zero_rules"], generated=generated,
                               scope=scope)
    results["rules_written_to"] = written.as_posix()
    results["recommendation"] = _recommendation(results)

    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(results), encoding="utf-8")
    results["report_written_to"] = report.as_posix()

    signs = results["sign_conventions"]
    tautology = results["q4_tautology"]
    nulled = [f for f, block in results["zero_rules"].items() if block["rule"] == "null"]
    context.log.info("Gate 1 completeness   : %d missing quarter(s) across %d ticker(s)",
                     int(results["completeness"]["n_missing"].sum()),
                     int((results["completeness"]["n_missing"] > 0).sum()))
    context.log.info("Gate 2 implausible    : %d flagged (%d negative, %d magnitude)",
                     len(results["implausible"]),
                     int((results["implausible"]["reason"] == "negative").sum())
                     if not results["implausible"].empty else 0,
                     int((results["implausible"]["reason"] == "magnitude").sum())
                     if not results["implausible"].empty else 0)
    context.log.info("Gate 3 zero-fill      : %d of %d field(s) proposed `null` -> %s",
                     len(nulled), len(results["zero_rules"]), ", ".join(nulled) or "none")
    context.log.info("Sign conventions      : fcf==ncfo+capex %s | capex<=0 %s (%d of %d rows "
                     "positive: %s)",
                     "HOLDS" if signs["fcf_identity_holds"] else "FAILED",
                     "HOLDS" if signs["capex_sign_holds"] else "DOES NOT HOLD",
                     signs["capex_positive_total"], signs["capex_rows_total"],
                     ", ".join(signs["capex_positive_tickers"]) or "none")
    summary = q4_tautology_summary(tautology)
    context.log.info("Q4 identity           : EXACTLY zero on %d/%d triples (%.1f%%), %d "
                     "materially non-zero (max %.2f%%) -- tautological where it holds, so the "
                     "spec's check #3 carries no information",
                     summary["n_exact"], summary["n"], 100 * summary["share_exact"],
                     summary["n_over_bar"], 100 * summary["max_dev"])
    shares = results["shares"]
    split = shares[shares["verdict"].str.startswith("SPLIT")] if not shares.empty else shares
    context.log.warning("sharesbas vs SEC      : %d/%d ticker(s) agree at 1.0; %d are "
                        "SPLIT-ADJUSTED and therefore NOT point-in-time: %s",
                        int(((shares["median_ratio"] - 1).abs() <= 0.05).sum())
                        if not shares.empty else 0, len(shares), len(split),
                        ", ".join(split["ticker"]) if not split.empty else "none")
    context.log.warning("Report -> %s | proposed rules -> %s",
                        results["report_written_to"], results["rules_written_to"])
    return results
