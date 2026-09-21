"""
coverage.py  (src/validate/checks/coverage.py)
--------------------------------------------------------------------------------------------
Which names are in the table, and for how much of the history they could have been in it.

THREE THINGS THIS CHECK REFUSES TO DO, each because doing it produced a false result before:

1. **It never compares against `sp500_tickers`.** That table is 500 rows with no date column
   -- a snapshot, not a point-in-time roster. The universe is `load_universe_tickers`, which
   is 491: the roster minus `INSUFFICIENT_HISTORY_TICKERS` and
   `config.data_extract.redundant_ticks`. The nine names in that difference
   (`FDXF GEHC GEV HONA KVUE Q SNDK SOLV VLTO`) are absent BY DECLARATION, and reporting them
   as missing data is the false positive `momentum/_scripts/08` recorded as D-08. The
   exclusion set is IMPORTED here, never copied, so it cannot go stale against the constant.

2. **It never measures a ticker against the table's own date range.** `KHC` did not exist
   before 2015 and `SW`, `MRVL` and `CBOE` are each legitimately absent from early quarters of
   every table in this database. Each ticker is measured against ITS OWN span in `prices`, so
   a late listing costs nothing and a name that stopped updating costs everything.

   ⚠ AND IT DOES NOT MEASURE DEPTH, BECAUSE DEPTH IS NOT A DEFECT. Measured on the same day
   across three parts, the median ticker's share of its own price span is 0.999 on
   `cube_part_momentum`, **0.870** on `cube_part_institutionals` and **0.975** on
   `cube_part_governance`, with minima of 0.99, 0.55 and **0.027**. Every one of those is
   correct: momentum is computed from prices, so it starts when the price history starts,
   while a 13F feature cannot exist before the fund first filed and a proxy feature cannot
   exist before the company's first DEF 14A. One threshold on that ratio fires on 339 of 491
   institutionals names and measures nothing but each source's start date. So the ratio is
   reported as a METRIC (`head_share`) and the two sub-checks that ARE defect-shaped without
   a per-table declaration carry the findings:
     - **interior** -- of the sessions the ticker traded INSIDE its own span in this table
       (first row -> last row), how many carry a row. A hole there is missing data on any
       reading, whatever the source's start date.
     - **trailing** -- the same over the last `recent_sessions`. A name that stopped updating
       is the live defect, and it is invisible to a depth ratio computed over thirty years.

3. **It never assumes the table is a daily session panel.** It measures that: the share of the
   table's own distinct dates that are `prices` sessions. Below `_SESSION_GRID_MIN` the whole
   check abstains, because "88% coverage" of a grid the table was never built on is a number
   with no meaning attached.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.constants.constants import INSUFFICIENT_HISTORY_TICKERS
from src.context import Context
from src.data_store.schema import Table, Tables, resolve
from src.utils.universe import load_universe_tickers
from src.validate.frame import as_ts
from src.validate.io import cache_used, read_columns
from src.validate.result import CheckResult, Finding, full_table_only
from src.validate.spec import load_spec

log = logging.getLogger(__name__)

CHECK = "coverage"

#: Findings filed per kind before the rest collapse into the count carried in `metrics`.
#: Sixty findings of one shape is how a report stops being read.
_MAX_FINDINGS = 20

#: The share of the table's distinct dates that must be `prices` sessions before a
#: per-session coverage ratio means anything. See point 3 of the module docstring.
_SESSION_GRID_MIN = 0.99


def _exclusions(context: Context) -> set[str]:
    """The declared universe exclusions, IMPORTED from the two places that own them.

    ⚠ Never inline this list. `load_universe_tickers` subtracts exactly this set, so a copy
    that drifted would make this check report names the universe had already dropped."""
    return INSUFFICIENT_HISTORY_TICKERS | {
        str(t).strip().upper() for t in context.config.data_extract.redundant_ticks}


def _panel(frame: pd.DataFrame, ticker_col: str, date_col: str) -> pd.DataFrame:
    """Per ticker: rows, first and last date."""
    grouped = frame.groupby(ticker_col)[date_col]
    return pd.DataFrame({"rows": grouped.size(), "first": grouped.min(), "last": grouped.max()})


def _expected_sessions(sessions: np.ndarray, start: Any, end: Any) -> int:
    """How many reference sessions fall in `[start, end]`."""
    if pd.isna(start) or pd.isna(end) or end < start:
        return 0
    lo = int(np.searchsorted(sessions, np.datetime64(pd.Timestamp(start)), side="left"))
    hi = int(np.searchsorted(sessions, np.datetime64(pd.Timestamp(end)), side="right"))
    return max(hi - lo, 0)


def check_coverage(context: Context, table: Table | str, *, config: DictConfig,
                   cache: Any = None, tickers: list[str] | None = None,
                   reference: Table | str = Tables.prices,
                   min_share: float | None = None, edge_days: int | None = None,
                   **kwargs: Any) -> CheckResult:
    """Tickers present vs the universe, each against its own `prices` span, plus the edge."""
    spec_t = resolve(table)
    if (declined := full_table_only(CHECK, spec_t.name, tickers)) is not None:
        return declined
    spec = load_spec(config, spec_t, coverage_min_share=min_share, edge_days=edge_days)
    date_col = spec_t.date_col
    live = context.store.columns(spec_t)
    ticker_col = next((c for c in ("ticker", "symbol") if c in live), None)
    if date_col is None or ticker_col is None:
        return CheckResult.abstained(
            CHECK, spec_t.name,
            f"coverage is a (ticker x session) question and this table has "
            f"date_col={date_col!r}, ticker column={ticker_col!r}")

    universe = load_universe_tickers(context)
    if not universe:
        return CheckResult.abstained(CHECK, spec_t.name,
                                     "load_universe_tickers returned nothing -- "
                                     "`sp500_tickers` is empty or absent")
    universe_set = set(universe)
    excluded = _exclusions(context)

    # -- the reference grid: one row per (ticker, session) that actually traded ---------- #
    ref_cols = context.store.columns(reference)
    ref_ticker = next((c for c in ("ticker", "symbol") if c in ref_cols), None)
    if ref_ticker is None or "date" not in ref_cols:
        return CheckResult.abstained(
            CHECK, spec_t.name,
            f"the reference table `{reference}` has no (ticker, date) grid to measure against")
    prices = context.store.load(reference, columns=[ref_ticker, "date"])
    prices = prices.rename(columns={ref_ticker: ticker_col})
    prices["date"] = as_ts(prices["date"])
    prices[ticker_col] = prices[ticker_col].astype(str).str.strip().str.upper()
    sessions = np.sort(prices["date"].unique())
    price_span = _panel(prices, ticker_col, "date")
    priced = set(price_span.index)

    # -- the table under test ------------------------------------------------------------ #
    frame = read_columns(context, spec_t, [ticker_col, date_col], cache=cache)
    if frame.empty:
        return CheckResult.abstained(CHECK, spec_t.name, "the table is empty -- nothing to measure")
    frame[date_col] = as_ts(frame[date_col])
    frame[ticker_col] = frame[ticker_col].astype(str).str.strip().str.upper()

    table_dates = pd.DatetimeIndex(frame[date_col].dropna().unique()).sort_values()
    on_grid = float(np.isin(table_dates.values, sessions).mean()) if len(table_dates) else 0.0
    if on_grid < _SESSION_GRID_MIN:
        return CheckResult.abstained(
            CHECK, spec_t.name,
            f"only {on_grid:.1%} of this table's {len(table_dates):,} distinct dates are "
            f"`{reference}` sessions, so it is not a daily session panel and a per-session "
            f"coverage share would be measuring the wrong denominator")

    panel = _panel(frame, ticker_col, date_col)
    present = set(panel.index)
    rows = len(frame)
    first_date, last_date = table_dates[0], table_dates[-1]

    findings: list[Finding] = []

    # -- 1. the universe itself ----------------------------------------------------------- #
    if len(universe) != spec.universe_expected:
        findings.append(Finding.at(
            5,
            observed=f"load_universe_tickers returned {len(universe)} names",
            expected=f"{spec.universe_expected} (configs/validate.yml universe_expected)",
            universe=len(universe), declared=spec.universe_expected))

    # -- 2. universe names with no row at all --------------------------------------------- #
    absent = sorted(universe_set - present)
    for ticker in absent[:_MAX_FINDINGS]:
        span = price_span.loc[ticker] if ticker in price_span.index else None
        findings.append(Finding.at(
            8, ticker=ticker,
            observed=f"{ticker} has 0 rows in {spec_t.name}",
            expected=(f"rows over its own price span "
                      f"{span['first'].date()} -> {span['last'].date()} ({int(span['rows']):,} "
                      f"sessions)" if span is not None else
                      "rows, or absence from the universe -- it has no `prices` history either"),
            in_prices=ticker in priced,
            price_rows=int(span["rows"]) if span is not None else 0))

    # -- 3. names in the table that the universe does not know ---------------------------- #
    #    Split by `prices`, because the two cases are different defects: a name with price
    #    history is a roster that moved on, a name with none is a row from nowhere.
    unknown = sorted(present - universe_set - excluded)
    for ticker in unknown[:_MAX_FINDINGS]:
        in_prices = ticker in priced
        findings.append(Finding.at(
            4 if in_prices else 7, ticker=ticker,
            observed=f"{ticker} holds {int(panel.loc[ticker, 'rows']):,} rows in "
                     f"{spec_t.name} but is not in the {len(universe)}-name universe"
                     + ("" if in_prices else " and has no `prices` rows either"),
            expected="every ticker in a cube part is a current universe member; a former "
                     "member is stale data, and a ticker in neither the roster nor `prices` "
                     "came from somewhere no other table knows about",
            in_prices=in_prices, rows=int(panel.loc[ticker, "rows"])))

    # -- 4. the declared exclusions, reported so a reader can see they were considered ---- #
    #    INFO by construction -- these are absent on purpose. See `_FAIL_FLOOR` in result.py.
    excluded_present = sorted(present & excluded)
    findings.append(Finding.at(
        1,
        observed=f"{len(excluded)} declared exclusions; {len(excluded_present)} of them "
                 f"carry rows here ({', '.join(excluded_present) if excluded_present else 'none'})",
        expected="informational: INSUFFICIENT_HISTORY_TICKERS u data_extract.redundant_ticks "
                 "are outside the universe by declaration, and their absence is not a defect",
        exclusions=sorted(excluded), present=excluded_present))

    # -- 5. interior holes, trailing staleness, and depth as a metric --------------------- #
    #    The denominator is always the sessions THAT TICKER TRADED, never a table-wide grid.
    members = sorted(present & universe_set & priced)
    traded = prices[prices[ticker_col].isin(members)][[ticker_col, "date"]]
    edges = panel.loc[members, ["first", "last"]]
    joined = traded.join(edges, on=ticker_col)
    inside = joined[(joined["date"] >= joined["first"]) & (joined["date"] <= joined["last"])]
    interior_expected = inside.groupby(ticker_col).size()
    head_expected = joined[joined["date"] < joined["first"]].groupby(ticker_col).size()

    cut = pd.Timestamp(sessions[-min(spec.recent_sessions, len(sessions))])
    trailing_expected = (traded[traded["date"] >= cut].groupby(ticker_col).size())
    trailing_seen = (frame[frame[date_col] >= cut].groupby(ticker_col).size())

    interior: dict[str, float] = {}
    trailing: dict[str, float] = {}
    head: dict[str, float] = {}
    for ticker in members:
        own = int(price_span.loc[ticker, "rows"])
        expected = int(interior_expected.get(ticker, 0))
        if expected > 0:
            interior[ticker] = int(panel.loc[ticker, "rows"]) / expected
        want = int(trailing_expected.get(ticker, 0))
        if want > 0:
            trailing[ticker] = int(trailing_seen.get(ticker, 0)) / want
        if own > 0:
            # Strictly the HEAD: sessions it traded BEFORE its first row here. The tail is
            # not folded in, because a missing tail is staleness and `trailing` files it.
            head[ticker] = int(head_expected.get(ticker, 0)) / own

    holed = sorted((t for t, s in interior.items() if s < spec.coverage_min_share),
                   key=lambda t: interior[t])
    for ticker in holed[:_MAX_FINDINGS]:
        expected = int(interior_expected[ticker])
        findings.append(Finding.at(
            5, ticker=ticker,
            observed=f"{int(panel.loc[ticker, 'rows']):,} rows over {expected:,} sessions it "
                     f"traded between its own first and last row here "
                     f"({interior[ticker]:.1%})",
            expected=f">= {spec.coverage_min_share:.0%} -- inside a span this table already "
                     f"claims for the ticker, a traded session with no row is a hole, not a "
                     f"late-starting source",
            share=round(interior[ticker], 4), rows=int(panel.loc[ticker, "rows"]),
            sessions_traded=expected,
            first=panel.loc[ticker, "first"], last=panel.loc[ticker, "last"]))

    stale = sorted((t for t, s in trailing.items() if s < spec.coverage_min_share),
                   key=lambda t: trailing[t])
    for ticker in stale[:_MAX_FINDINGS]:
        findings.append(Finding.at(
            7, ticker=ticker,
            observed=f"{int(trailing_seen.get(ticker, 0)):,} of the "
                     f"{int(trailing_expected[ticker]):,} sessions it traded in the last "
                     f"{spec.recent_sessions} ({trailing[ticker]:.1%}); last row "
                     f"{pd.Timestamp(panel.loc[ticker, 'last']).date()}",
            expected=f">= {spec.coverage_min_share:.0%} of its recent traded sessions -- a "
                     f"name that stopped updating still passes every row-count and every "
                     f"depth ratio computed over the whole history",
            share=round(trailing[ticker], 4), last=panel.loc[ticker, "last"]))

    # Depth is INFO: it is each source's start date, not a defect. See the module docstring.
    head_series = pd.Series(head, dtype=float)
    if len(head_series):
        deepest = head_series.idxmax()
        findings.append(Finding.at(
            1,
            observed=f"median ticker has no rows over {head_series.median():.1%} of its own "
                     f"price history before its first row here; worst is {deepest} at "
                     f"{head_series.max():.1%}",
            expected="informational: a feature cannot predate its own source's first "
                     "publication, so this is measured and reported, never asserted on",
            median_head_share=round(float(head_series.median()), 4),
            worst=deepest, worst_head_share=round(float(head_series.max()), 4),
            deepest_heads={t: round(float(v), 4)
                           for t, v in head_series.nlargest(10).items()}))

    # -- 6. the edge: a partial write shows up as a thin last day, not as an error -------- #
    edge = []
    recent = sessions[sessions >= np.datetime64(first_date)][-spec.edge_days:]
    by_date = frame.groupby(date_col)[ticker_col].agg(set)
    traded_by_date = prices.groupby("date")[ticker_col].agg(set)
    for day in recent:
        stamp = pd.Timestamp(day)
        seen = by_date.get(stamp, set())
        # ⚠ THE DENOMINATOR IS UNIVERSE MEMBERS THAT TRADED, not every ticker in `prices`.
        # `prices` carries the declared exclusions too, and counting them here re-creates
        # D-08 one session at a time: a complete table reads as short by exactly the number
        # of names it was never supposed to have.
        want_names = traded_by_date.get(stamp, set()) & universe_set
        got, want = len(seen & universe_set), len(want_names)
        # The names, not just the count: "490 of 491" is a number, "490 of 491, missing NVDA"
        # is somewhere to look.
        gap = sorted(want_names - seen)
        edge.append({"date": stamp, "tickers": got, "reference_tickers": want,
                     "missing": gap[:25]})
        if want and got < spec.coverage_min_share * want:
            findings.append(Finding.at(
                8,
                observed=f"{stamp.date()}: {got} tickers vs {want} trading in `{reference}`",
                expected=f">= {spec.coverage_min_share:.0%} of the session's traded names -- "
                         f"a thin edge day is a partial write, and it is invisible to a "
                         f"row-count check because the rows that ARE there are correct",
                date=stamp, tickers=got, reference_tickers=want,
                share=round(got / want, 4), missing=gap[:25]))

    def _distribution(values: dict[str, float], below: list[str]) -> dict[str, Any]:
        series = pd.Series(values, dtype=float)
        if not len(series):
            return {"n": 0, "threshold": spec.coverage_min_share}
        return {"n": int(len(series)), "min": float(series.min()),
                "p01": float(series.quantile(0.01)), "p50": float(series.median()),
                "mean": float(series.mean()), "max": float(series.max()),
                # A share ABOVE 1 is rows on sessions the ticker did not trade -- either a
                # duplicated key (which `grain` owns and scores 10) or a feature row on a
                # day with no price. Counted here rather than filed, so the two checks do
                # not report the same defect twice.
                "above_one": int((series > 1.0).sum()),
                "below_threshold": len(below), "threshold": spec.coverage_min_share}

    scope = {"rows": rows, "tickers": len(present), "first_date": first_date,
             "last_date": last_date, "universe": len(universe),
             "reference": str(reference), "sessions_on_grid": round(on_grid, 5),
             "recent_sessions": spec.recent_sessions, "edge_days": spec.edge_days,
             "source": "cache" if cache_used(cache, spec_t) else "db"}
    metrics = {
        "universe": len(universe), "tickers_present": len(present),
        "absent": absent, "unknown": unknown, "exclusions_present": excluded_present,
        "interior": _distribution(interior, holed),
        "trailing": _distribution(trailing, stale),
        # Depth, reported and never asserted on: the share of its own price history each
        # ticker predates its first row here.
        "head_share": {t: round(float(v), 4) for t, v in head_series.nlargest(25).items()},
        "holed_tickers": {t: round(interior[t], 4) for t in holed[:50]},
        "stale_tickers": {t: round(trailing[t], 4) for t in stale[:50]},
        "edge": edge,
    }
    return CheckResult.measured(CHECK, spec_t.name, findings, scope=scope, metrics=metrics)
