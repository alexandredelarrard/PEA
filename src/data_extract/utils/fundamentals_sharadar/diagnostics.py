"""
diagnostics.py  (src/data_extract/utils/fundamentals_sharadar/diagnostics.py)
------------------------------------------------------------------------------------
READ-ONLY measurement of `fundamentals_sharadar`: five gates, one markdown report of what
they measured, and nothing else.

⚠ **THIS IS NOT THE SEC CHECK SCHEME AND MUST NEVER BE WIRED INTO IT** (D25). Nothing here
registers a `CHECK_REGISTRY` entry, writes a `fundamentals_check` row, imports from
`src/validate/`, or is reachable from the validator CLI. It writes no production data at all:
one markdown report outside the database. If a change to this module starts to look like
adding a check, it is in the wrong file.

The report RENDERS WHAT THIS RUN MEASURED and states no conclusions. An earlier version
narrated findings in prose with the phase-2 numbers baked into the sentences; those numbers
went stale the moment the entitlement widened, and a report that confidently prints a stale
figure is worse than one that prints none. The phase-2 conclusions are recorded once, with
their scope, in `reports/planning/active-tasks/2026-08-26-sharadar-integration/`.

## Why the Q4 identity is not one of the gates

The spec proposed `Q4 == FY - 9M` as an acceptance check. It is **dead on arrival**: Sharadar
CONSTRUCTS Q4 as `ARY - Σ(Q1..Q3)`, so the identity is an identity -- measured at `+0.000%` on
every year tested. A check that can never fail can never inform. What replaces it is
`gate_implausible_quarters`: the construction cannot fail the identity, but it can and does
produce absurd LEVELS. The legacy Quandl documentation shows it yielding ABT 2011 Q4 revenue of
**-$7.1bn**, annotated as intentional "to ensure that the quarterly and annual financials are
aligned". That is the failure mode worth measuring.

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

## One read, not nine

Every gate is a PURE function of frames. `run_diagnostics` performs the single projected read
and slices it per dimension; an earlier version had each gate load for itself, which re-read
the widest extract table in the schema (112 columns x 3 dimensions) nine times per run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.constants.constants import (
    SHARADAR_DIAGNOSTIC_EXTRA_COLUMNS, SHARADAR_FLOW_FIELDS, SHARADAR_NON_NEGATIVE_FIELDS,
    SHARADAR_SEC_COUNTERPART, SHARADAR_ZERO_FILLED_FIELDS,
)
from src.context import Context
from src.data_store.schema import Tables
from src.utils.quarters import quarter_label, quarter_ordinal

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

#: Every dimension the gates need, read in one pass.
DIAGNOSTIC_DIMENSIONS: tuple[str, ...] = ("ARQ", "ARY", "ART")


# --------------------------------------------------------------------------- #
# loading                                                                      #
# --------------------------------------------------------------------------- #
def _diagnostic_columns() -> list[str]:
    """The projection: the table's declared `read_columns` plus the 4 zero-filled fields it
    deliberately omits. Never the whole table -- 112 columns x 3 dimensions is the widest
    extract table in the schema."""
    return list(dict.fromkeys(list(Tables.sharadar_fundamentals.read_columns)
                              + list(SHARADAR_DIAGNOSTIC_EXTRA_COLUMNS)))


def load_sharadar(context: Context, tickers: Sequence[str] | None,
                  dimensions: Sequence[str] = DIAGNOSTIC_DIMENSIONS) -> pd.DataFrame:
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


def load_sec(context: Context, tickers: Sequence[str]) -> pd.DataFrame:
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
def gate_completeness(arq: pd.DataFrame) -> pd.DataFrame:
    """Missing quarters per ticker, measured against EACH TICKER'S OWN observed window.

    A ticker whose history simply starts late is not a gap -- otherwise every ticker would
    show a hole back to whatever the earliest one happens to reach. The expected count is
    `last - first + 1` quarters of that ticker's own span, so the only thing this can report
    is a HOLE, which is the only thing that would break a TTM build.
    """
    arq = arq.assign(_q=quarter_ordinal(arq["calendardate"]))
    rows: list[dict] = []
    for ticker, group in arq.groupby("ticker", sort=True):
        observed = sorted(int(q) for q in group["_q"].dropna().unique())
        if not observed:
            continue
        seen = set(observed)
        span = range(observed[0], observed[-1] + 1)
        missing = [q for q in span if q not in seen]
        rows.append({
            "ticker": str(ticker),
            "first_quarter": quarter_label(observed[0]),
            "last_quarter": quarter_label(observed[-1]),
            "n_rows": int(len(group)),
            "n_quarters": len(observed),
            "expected_quarters": len(span),
            "n_missing": len(missing),
            "missing_quarters": ", ".join(quarter_label(q) for q in missing) or "-",
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


def gate_implausible_quarters(arq: pd.DataFrame, *, ratio: float = 3.0) -> pd.DataFrame:
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
    arq = with_fiscal_period(arq)
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
    if field in SHARADAR_FLOW_FIELDS and not art.empty and field in art.columns:
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


def gate_zero_fill(arq: pd.DataFrame, art: pd.DataFrame, sec: pd.DataFrame) -> pd.DataFrame:
    """Per-field zero-fill prevalence, with the evidence needed to decide whether each zero
    is a fact or a fill. One row per field in `SHARADAR_ZERO_FILLED_FIELDS`.

    Two independent bodies of evidence, because neither is sufficient alone:

    * **Sharadar-internal** -- `n_tickers_all_zero` (the field is 0 in EVERY row of that
      ticker: the signature of "not applicable") against `n_zero_mixed` (zeros in tickers that
      report the same field non-zero in another quarter: the signature of a fill). This is the
      ONLY evidence available for the 21 fields with no SEC counterpart.
    * **SEC cross-check** on the overlapping tickers, basis-matched -- see `_sec_verdicts`.
    """
    sharadar_tickers = set(arq["ticker"].astype(str))
    overlap = (sorted(sharadar_tickers & set(sec["ticker"].astype(str)))
               if not sec.empty else [])

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
        # the count of NON-zero rows per ticker, vectorised -- a ticker at 0 is all-zero
        non_zero_per_ticker = series.assign(_nz=series[field].ne(0)).groupby("ticker")["_nz"].sum()
        all_zero = set(non_zero_per_ticker.index[non_zero_per_ticker == 0].astype(str))
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
_SHARE_COLUMNS = ["ticker", "n_dates", "median_ratio", "min_ratio", "max_ratio",
                  "median_sharefactor"]


def cross_check_shares(arq: pd.DataFrame, sec: pd.DataFrame) -> pd.DataFrame:
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
    if sec.empty:
        return pd.DataFrame(columns=_SHARE_COLUMNS)
    left = arq[["ticker", "date", "sharesbas", "sharefactor"]].copy()
    left["ticker"] = left["ticker"].astype(str)
    right = sec[["ticker", "as_of", "sharesOutstanding"]].copy()
    right["ticker"] = right["ticker"].astype(str)
    joined = left.merge(right, left_on=["ticker", "date"], right_on=["ticker", "as_of"],
                        how="inner").dropna(subset=["sharesbas", "sharesOutstanding"])
    joined = joined[joined["sharesOutstanding"] != 0]
    if joined.empty:
        return pd.DataFrame(columns=_SHARE_COLUMNS)
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


def confirm_sign_conventions(frame: pd.DataFrame) -> dict:
    """Re-assert from STORED data what was measured from the API: `capex <= 0` throughout, and
    `fcf == ncfo + capex` to the cent.

    Cheap assertions protecting an expensive mistake. The phase-3 field map maps
    `freeCashflow <- fcf` with NO reconstruction and `capex` with a sign flip (the SEC
    catalogue declares `capex` non-negative, Sharadar stores it negative). If either identity
    fails, both of those decisions are wrong and the field map is built on sand.
    """
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


# --------------------------------------------------------------------------- #
# the report -- what THIS run measured, and nothing else                       #
# --------------------------------------------------------------------------- #
def md_cell(value: object) -> str:
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


def md_table(frame: pd.DataFrame, limit: int | None = None) -> str:
    """A markdown table, rendered here rather than via `to_markdown` so `src/` carries no
    `tabulate` dependency -- it is not in `pyproject.toml` and only resolves today because
    something else pulled it into the venv."""
    if frame is None or frame.empty:
        return "_(no rows)_\n"
    shown = frame if limit is None else frame.head(limit)
    header = "| " + " | ".join(str(c).replace("|", "\\|") for c in shown.columns) + " |"
    rule = "|" + "|".join("---" for _ in shown.columns) + "|"
    body = ["| " + " | ".join(md_cell(v) for v in row) + " |"
            for row in shown.itertuples(index=False)]
    tail = ("" if limit is None or len(frame) <= limit
            else f"\n_{limit} of {len(frame)} rows shown._\n")
    return "\n".join([header, rule, *body]) + "\n" + tail


def render_report(results: dict) -> str:
    """The five gate tables under a scope header. No prose verdicts.

    Every number here is measured by the run that writes it. Conclusions belong in a dated
    report where their scope is recorded beside them, not in a generator that will reprint
    last quarter's finding against next quarter's data.
    """
    signs = results["sign_conventions"]
    sign_rows = pd.DataFrame([
        {"dimension": dim, **{k: v for k, v in block.items()
                              if k != "capex_positive_tickers"}}
        for dim, block in signs["dimensions"].items()])
    parts = [
        f"# Sharadar diagnostics — {results['generated']}",
        "",
        f"Scope: {results['scope']}. "
        f"{results['n_overlap']} ticker(s) overlap `{Tables.fundamentals_history_sec}`. "
        f"Magnitude ratio: {results['ratio']}.",
        "",
        "## Gate 1 — completeness (missing quarters per ticker)",
        md_table(results["completeness"], WORST_ROWS),
        "## Gate 2 — implausible quarters",
        f"{len(results['implausible'])} flagged.",
        md_table(results["implausible"], WORST_ROWS),
        "## Gate 3 — zero-fill prevalence per field",
        md_table(results["zero_fill"]),
        "## Cross-check — `sharesbas` vs the SEC cover-page count",
        md_table(results["shares"], WORST_ROWS),
        "## Sign conventions — `capex <= 0` and `fcf == ncfo + capex`",
        f"capex sign holds: **{signs['capex_sign_holds']}**; "
        f"fcf identity holds: **{signs['fcf_identity_holds']}**.",
        md_table(sign_rows),
    ]
    return "\n".join(parts) + "\n"


def run_diagnostics(context: Context, tickers: Sequence[str] | None = None, *,
                    report_path: str | Path = DEFAULT_REPORT_PATH) -> dict:
    """Run the five gates off ONE read, write the report, log a summary.

    Writes NO production data: one markdown report outside the database. Returns everything it
    measured so a test can assert on it without re-reading the file.
    """
    ratio = float(context.config.data_extract.fundamentals_periods.max_opposite_sign_q4_ratio)

    frame = load_sharadar(context, tickers)
    by_dimension = {dim: group for dim, group in frame.groupby("dimension", sort=False)}
    arq = by_dimension.get("ARQ", frame.iloc[:0])
    art = by_dimension.get("ART", frame.iloc[:0])
    if arq.empty:
        raise RuntimeError(
            f"{Tables.sharadar_fundamentals} has no ARQ rows for the requested scope; every "
            f"gate is built on the as-reported quarters.")

    scope_tickers = sorted(str(t) for t in arq["ticker"].unique())
    sec = load_sec(context, scope_tickers)
    overlap = sorted(set(scope_tickers) & set(sec["ticker"].astype(str))) if not sec.empty else []
    scope = (f"{len(scope_tickers)} ticker(s), "
             f"{pd.Timestamp(arq['date'].min()).date()}..{pd.Timestamp(arq['date'].max()).date()}")
    context.log.info("Sharadar diagnostics: %s; %d overlap with %s",
                     scope, len(overlap), Tables.fundamentals_history_sec)

    results: dict = {
        "generated": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "scope": scope,
        "tickers": scope_tickers,
        "overlap": overlap,
        "n_overlap": len(overlap),
        "ratio": ratio,
        "completeness": gate_completeness(arq),
        "implausible": gate_implausible_quarters(arq, ratio=ratio),
        "zero_fill": gate_zero_fill(arq, art, sec),
        "shares": cross_check_shares(arq, sec),
        "sign_conventions": confirm_sign_conventions(frame),
    }

    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(results), encoding="utf-8")
    results["report_written_to"] = report.as_posix()

    signs = results["sign_conventions"]
    context.log.info("Gate 1 completeness   : %d missing quarter(s) across %d ticker(s)",
                     int(results["completeness"]["n_missing"].sum()),
                     int((results["completeness"]["n_missing"] > 0).sum()))
    context.log.info("Gate 2 implausible    : %d flagged (%d negative, %d magnitude)",
                     len(results["implausible"]),
                     int((results["implausible"]["reason"] == "negative").sum())
                     if not results["implausible"].empty else 0,
                     int((results["implausible"]["reason"] == "magnitude").sum())
                     if not results["implausible"].empty else 0)
    zero_fill = results["zero_fill"]
    context.log.info("Gate 3 zero-fill      : %d field(s) measured; %d with a contradicted zero",
                     len(zero_fill), int((zero_fill["sec_contradicted"] > 0).sum()))
    context.log.info("Sign conventions      : fcf==ncfo+capex %s | capex<=0 %s (%d of %d rows "
                     "positive: %s)",
                     "HOLDS" if signs["fcf_identity_holds"] else "FAILED",
                     "HOLDS" if signs["capex_sign_holds"] else "DOES NOT HOLD",
                     signs["capex_positive_total"], signs["capex_rows_total"],
                     ", ".join(signs["capex_positive_tickers"]) or "none")
    shares = results["shares"]
    split = shares[shares["verdict"].str.startswith("SPLIT")] if not shares.empty else shares
    context.log.warning("sharesbas vs SEC      : %d/%d ticker(s) agree at 1.0; %d are "
                        "SPLIT-ADJUSTED and therefore NOT point-in-time: %s",
                        int(((shares["median_ratio"] - 1).abs() <= 0.05).sum())
                        if not shares.empty else 0, len(shares), len(split),
                        ", ".join(split["ticker"]) if not split.empty else "none")
    context.log.warning("Report -> %s", results["report_written_to"])
    return results
