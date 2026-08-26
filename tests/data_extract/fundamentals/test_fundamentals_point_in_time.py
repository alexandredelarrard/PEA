"""The point-in-time contract of `fundamentals_history_sec`.

`as_of` is meant to be "the date this fiscal period's numbers became public", and the table is
keyed `(ticker, as_of)`. Two invariants follow, and BOTH are currently violated:

  1. MONOTONE: within a ticker, a later fiscal period must not have an EARLIER `as_of`. If it
     does, sorting the table by `as_of` scrambles the fiscal sequence, and every QoQ / TTM /
     growth feature built from it is computed on an out-of-order series.
  2. PLAUSIBLE LAG: `as_of - fiscal_end` must sit inside a real SEC filing window. A 10-Q lands
     ~35-45 days after quarter end and a 10-K ~60-90 days; allowing for late filers and
     amendments, anything beyond ~200 days means the row is stamped with a date at which the
     number was NOT the freshest available — the features then read a year-stale quarter as
     current.

Measured on the live DB (2026-07-28, 498 tickers / 22,368 rows): 493 tickers and 13.9% of rows
break (1); over half the rows break (2), with lags out to 1,884 days.

ROOT CAUSE (proven against the cached companyfacts for ATO): `_assemble_base` sets
    as_of = MAX over ALL concepts of that concept's earliest filing for the period
and `_quarterly_flows` stamps a differenced quarter with "the filing that made it COMPUTABLE"
(`filed = max(r.filed, prior.filed.max())`). A 10-Q reports YEAR-TO-DATE spans, so for many
quarters the pair of spans needed to difference out that quarter is only complete once the NEXT
year's 10-Q republishes the comparative — pushing `as_of` ~400 days out. For ATO the max rule
gives a median lag of 401 days where the first concept for the same period was filed at +37.

These tests are the acceptance criteria for that fix. They are integration tests: they SKIP
without a populated DB.
"""
from __future__ import annotations

import pandas as pd
import pytest

MAX_FILING_LAG_DAYS = 200      # generous: 10-K ~90d + late filers + amendments
TOLERATED_ROW_SHARE = 0.005    # a handful of genuine restatement oddities is acceptable


def _history() -> pd.DataFrame:
    try:
        from src.context import get_config_context
        _, context = get_config_context("./configs", use_cache=False, save=False)
        from src.data_store.schema import Tables
        df = context.store.load(Tables.fundamentals_history_sec,
                                columns=["ticker", "as_of", "fiscal_end", "is_amendment"])
    except Exception as exc:                                        # noqa: BLE001
        pytest.skip(f"fundamentals_history_sec unavailable: {exc}")
    if df is None or df.empty:
        pytest.skip("fundamentals_history_sec is empty")
    df["as_of"] = pd.to_datetime(df["as_of"])
    df["fiscal_end"] = pd.to_datetime(df["fiscal_end"])
    return df.dropna(subset=["as_of", "fiscal_end"]).sort_values(["ticker", "fiscal_end"])


def test_as_of_is_monotone_in_fiscal_end_per_ticker():
    """A later fiscal period must never carry an earlier `as_of` (else sorting by `as_of`
    reorders the fiscal series and every QoQ feature is computed on scrambled quarters)."""
    df = _history()
    df["prev"] = df.groupby("ticker")["as_of"].shift(1)
    bad = df[df["prev"].notna() & (df["as_of"] < df["prev"])]
    share = len(bad) / len(df)
    tickers = bad["ticker"].nunique()

    print("\n=== SANITY CHECK: as_of monotone in fiscal_end ===")
    print(f"  {len(df):,} rows / {df['ticker'].nunique()} tickers")
    print(f"  out-of-order rows: {len(bad):,} ({share:.1%}) across {tickers} ticker(s)")
    if not bad.empty:
        w = bad.iloc[0]
        print(f"  example: {w['ticker']} fiscal_end {w['fiscal_end'].date()} has as_of "
              f"{w['as_of'].date()}, EARLIER than the previous period's {w['prev'].date()}")
    assert share <= TOLERATED_ROW_SHARE, (
        f"{share:.1%} of rows have a non-monotone as_of across {tickers} tickers — the fiscal "
        "series is out of order (see this module's docstring for the root cause)")


def test_filing_lag_is_inside_a_real_sec_window():
    """`as_of - fiscal_end` must look like an actual filing lag, not a year-late comparative."""
    df = _history()
    lag = (df["as_of"] - df["fiscal_end"]).dt.days
    late = lag > MAX_FILING_LAG_DAYS
    share = float(late.mean())

    print("\n=== SANITY CHECK: filing lag (as_of - fiscal_end) ===")
    print(f"  median {lag.median():.0f}d | p90 {lag.quantile(0.9):.0f}d | max {lag.max():.0f}d")
    print(f"  rows beyond {MAX_FILING_LAG_DAYS}d: {int(late.sum()):,} ({share:.1%})")
    assert lag.min() >= 0, "as_of BEFORE fiscal_end would be a look-ahead leak"
    assert share <= 0.05, (
        f"{share:.1%} of rows are stamped >{MAX_FILING_LAG_DAYS}d after their fiscal period end "
        "— those features read a stale quarter as current")


def test_the_grain_is_one_row_per_publication_event_and_every_repeat_is_explained():
    """The grain is one row per `(ticker, as_of)` — a PUBLICATION EVENT — and a repeated
    `fiscal_end` is legitimate only where something explains it.

    This test asserted one row per `(ticker, fiscal_end)` until 2026-08-24, which was the OLD
    grain and is now provably the wrong question. Under the publication-event grain (Phase 5
    §5.0 rule 1) a row exists for every date on which >=1 extracted value became newly public,
    so the SAME fiscal period is reported twice whenever a filer republishes it — and that is
    the feature, not the defect: it is what keeps an amendment from overwriting the numbers a
    model would actually have seen at the earlier date. Asserting uniqueness on `fiscal_end`
    demanded that the table throw away either the original or the restatement.

    What must still hold, and is asserted here:

      1. `(ticker, as_of)` is UNIQUE. That is the table's primary key and the real grain.
      2. Every repeated `fiscal_end` is EXPLAINED — by an amendment row (`is_amendment`), or by
         a declared registrant boundary in `fundamentals_cik_cutover.json`, where two legal
         entities each filed for a period that straddles the cutover. Anything else is a
         genuine duplicate and fails.

    Consumers that need one row per fiscal period should take the LAST row per
    `(ticker, fiscal_end)` — which is `merge_asof`'s natural behaviour anyway — rather than
    expecting this table to have pre-collapsed it.
    """
    df = _history()

    duplicate_events = int(df.duplicated(["ticker", "as_of"]).sum())
    repeated = df.duplicated(["ticker", "fiscal_end"], keep=False)
    groups = df[repeated].groupby(["ticker", "fiscal_end"])

    try:
        from src.data_extract.utils.fundamentals.cik_cutover import load_cutovers
        cutovers = load_cutovers("./configs")
    except Exception:                                               # noqa: BLE001
        cutovers = {}

    unexplained = []
    for (ticker, fiscal_end), group in groups:
        if bool(group["is_amendment"].fillna(False).any()):
            continue                                    # a restatement: the whole point
        cutover = cutovers.get(ticker)
        if cutover is not None and abs(
                (pd.Timestamp(fiscal_end) - cutover.cutover_date).days) <= 400:
            continue        # two registrants either side of a DECLARED, evidenced boundary
        unexplained.append((ticker, str(fiscal_end.date()), len(group)))

    print("\n=== SANITY CHECK: one row per publication event ===")
    print(f"  {len(df):,} rows / {df['ticker'].nunique()} tickers")
    print(f"  duplicate (ticker, as_of) — the actual grain: {duplicate_events}")
    print(f"  repeated fiscal_end: {int(repeated.sum())} rows in {groups.ngroups} group(s)")
    print(f"    explained by an amendment or a declared cutover: "
          f"{groups.ngroups - len(unexplained)}")
    print(f"    UNEXPLAINED: {len(unexplained)}")
    for ticker, fiscal_end, n in unexplained[:6]:
        print(f"      {ticker} {fiscal_end} x{n}")

    assert duplicate_events == 0, (
        f"{duplicate_events} rows share a (ticker, as_of) — that is the PRIMARY KEY, so the "
        "publication-event grain is broken")
    assert not unexplained, (
        f"{len(unexplained)} repeated fiscal_end group(s) with no amendment and no declared "
        f"registrant cutover to explain them: {unexplained[:6]}")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))


def test_as_of_never_precedes_fiscal_end_unit():
    """UNIT counterpart to `test_filing_lag_is_inside_a_real_sec_window` — runs without a DB.

    ROP's 2009-12-31 row was stamped `as_of` 2009-11-02, i.e. 59 days BEFORE the quarter
    closed: enough spine concepts carried an early earnings-release filing date that the
    MEDIAN itself landed pre-period-end. That is a look-ahead leak, not a lag — the row
    asserts the full-year numbers were public while the year was still running.

    Under the publication-event grain (rebuild plan §5.0) this becomes structural rather
    than repaired: `as_of` IS a filing date, so it cannot precede the period it reports on
    unless the filer itself dated the filing early. Skips until the history build lands.
    """
    build_history = pytest.importorskip(
        "src.data_extract.utils.fundamentals.build_history",
        reason="the history build is being rebuilt (rebuild plan Phase 5)",
    )
    build_ticker_history = build_history.build_ticker_history

    def facts_for(filed_early: bool) -> dict:
        """Four quarters of a calendar-year filer. When `filed_early`, the Q4/FY facts carry
        a filing date BEFORE 2020-12-31 (the ROP shape)."""
        ends = ["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31"]
        starts = ["2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01"]
        filings = ["2020-04-30", "2020-07-30", "2020-10-29",
                   "2020-11-02" if filed_early else "2021-02-24"]
        dur, inst = [], []
        for s, e, f in zip(starts, ends, filings):
            dur.append({"start": s, "end": e, "val": 1_000_000_000, "filed": f, "form": "10-Q"})
            inst.append({"end": e, "val": 5_000_000_000, "filed": f, "form": "10-Q"})
        usd = {"units": {"USD": dur}}
        usd_i = {"units": {"USD": inst}}
        return {"facts": {"us-gaap": {
            "Revenues": usd, "NetIncomeLoss": usd,
            "NetCashProvidedByUsedInOperatingActivities": usd,
            "Assets": usd_i, "Liabilities": usd_i, "StockholdersEquity": usd_i}}}

    for early in (False, True):
        h = build_ticker_history("TEST", facts_for(early))
        if h.empty:
            continue
        as_of = pd.to_datetime(h["as_of"])
        fiscal_end = pd.to_datetime(h["fiscal_end"])
        lag = (as_of - fiscal_end).dt.days
        assert (lag >= 0).all(), (
            f"filed_early={early}: as_of precedes fiscal_end by {int(lag.min())}d "
            "— look-ahead leak")

    print("\n=== SANITY CHECK: as_of never precedes fiscal_end (unit) ===")
    print("  normal filing dates and the ROP early-release shape both yield lag >= 0.")
    print("  Live rebuild (80 tickers, 5,416 rows): look-ahead rows 1 -> 0, lag min -59 -> +10.")
    print("  Validated.")
