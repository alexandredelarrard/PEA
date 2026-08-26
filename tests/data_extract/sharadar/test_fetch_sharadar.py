"""Tests for the Sharadar SF1 extraction
(src/data_extract/utils/fundamentals_sharadar/).

Per the project testing rule, the feature / economic tests run against the REAL Sharadar API
and the REAL database; only the dtype test gets a synthetic known-truth fixture, because you
cannot verify "a column the first ticker never populates still becomes float64" without
constructing a first ticker that never populates it.

Every test prints a sanity-check conclusion.
"""
from __future__ import annotations

import time

import pandas as pd
import pandas.api.types as ptypes
import pytest

from src.constants.constants import (
    SHARADAR_DIMENSIONS, SHARADAR_ID_COLUMNS, SHARADAR_SF1_COLUMNS,
)
from src.data_store.schema import Tables
from src.data_extract.utils.fundamentals_sharadar import client as client_mod
from src.data_extract.utils.fundamentals_sharadar.client import (
    NotEntitled, cast_value_columns, sharadar_get,
)
from src.data_extract.utils.fundamentals_sharadar.fetch_sharadar import (
    fetch_sharadar_fundamentals, fetch_sharadar_tickers,
)

CONFIG_DIR = "./configs"
ENTITLED = "AAPL"          # measured entitled on the current key
NOT_ENTITLED = "ADBE"      # measured 403 "Exceeds free tier"


@pytest.fixture(scope="module")
def context():
    """A real Context (DB + .env). Skips rather than errors when either is unavailable --
    these are integration tests, and a machine without the Postgres container should report
    'skipped', not a wall of connection tracebacks."""
    from src.context import get_config_context
    try:
        _, ctx = get_config_context(CONFIG_DIR, use_cache=False, save=False)
        with ctx.store.engine.connect():
            pass
    except Exception as exc:                                            # noqa: BLE001
        pytest.skip(f"context/database unavailable ({type(exc).__name__}: {exc})")
    return ctx


# --------------------------------------------------------------------------- #
# 1. The regression test for the TEXT-column bug (synthetic, known-truth)      #
# --------------------------------------------------------------------------- #
def test_value_columns_are_float():
    """`ensure_table` types a table off the FIRST frame it ever sees, and `ddl.sql_type`
    falls through to TEXT for an object dtype. So a column the first ticker never populates
    must STILL be float64 before the first write -- otherwise it lands as TEXT and every
    later ticker's real number is stored as a string."""
    first_ticker = pd.DataFrame({
        "ticker": ["AAPL", "AAPL"],
        "dimension": ["ARQ", "ARQ"],
        "calendardate": ["2025-03-31", "2025-06-30"],
        "date": ["2025-05-02", "2025-08-01"],
        "reportperiod": ["2025-03-29", "2025-06-28"],
        "fiscalperiod": ["Q2", "Q3"],
        "lastupdated": ["2026-07-31", "2026-07-31"],
        "revenue": [95359000000.0, 94036000000.0],
        "deposits": [None, None],          # a bank column: never populated by AAPL
        "intexp": [None, None],            # ditto
        "eps": ["1.65", "1.57"],           # arrives as text if the CSV had a stray space
    })
    out = cast_value_columns(first_ticker)

    value_cols = [c for c in out.columns if c not in SHARADAR_ID_COLUMNS]
    non_float = {c: str(out[c].dtype) for c in value_cols if out[c].dtype != "float64"}
    id_cols_kept = {c: str(out[c].dtype) for c in out.columns if c in SHARADAR_ID_COLUMNS}

    print("\n=== SANITY CHECK: SF1 value columns are float64 before the first write ===")
    print(f"  value columns      : {len(value_cols)} -> "
          f"{ {c: str(out[c].dtype) for c in value_cols} }")
    print(f"  identifier columns : {id_cols_kept}")
    print(f"  all-None columns   : deposits={out['deposits'].dtype}, "
          f"intexp={out['intexp'].dtype}  <- these are the TEXT-column bug")
    print(f"  text-like numbers  : eps={out['eps'].dtype}, value={out['eps'].tolist()}")

    assert not non_float, f"these would be created as TEXT: {non_float}"
    assert out["eps"].tolist() == [1.65, 1.57], "a text number must become a real float"
    # NOT `dtype == object`: this pandas backs string columns with arrow (`str`), so the
    # property to pin is that an identifier stayed NON-NUMERIC, not which string dtype it got.
    numeric_ids = {c: str(out[c].dtype) for c in SHARADAR_ID_COLUMNS
                   if c in out.columns and ptypes.is_numeric_dtype(out[c])}
    assert not numeric_ids, f"identifier columns must NOT be cast to float: {numeric_ids}"
    print(f"  OK: {len(value_cols)}/{len(value_cols)} value columns are float64, including "
          f"the two this ticker never reports.")


# --------------------------------------------------------------------------- #
# 2. The response header IS the contract (real call)                           #
# --------------------------------------------------------------------------- #
def test_response_header_matches_contract(context):
    """`fields=` drops an unavailable field SILENTLY -- a typo yields a missing column and no
    warning. So the header is validated against `SHARADAR_SF1_COLUMNS` on every response, and
    this test pins that contract against the live feed."""
    frame = sharadar_get(context, "fundamentals", ticker=ENTITLED, dimension="ARQ",
                         sort="date.asc", limit=5, **{"date.gte": "2024-01-01"})
    if frame is None:
        pytest.skip("Sharadar request failed (network)")

    got = tuple(frame.columns)
    expected = tuple(SHARADAR_SF1_COLUMNS)
    missing = [c for c in expected if c not in got]
    unexpected = [c for c in got if c not in expected]

    print("\n=== SANITY CHECK: SF1 response header vs the stored contract ===")
    print(f"  received {len(got)} columns, contract has {len(expected)}")
    print(f"  in contract but NOT delivered : {missing or 'none'}")
    print(f"  delivered but NOT in contract : {unexpected or 'none'}")
    print(f"  order identical               : {got == expected}")

    assert not missing, f"the feed stopped delivering: {missing}"
    assert not unexpected, f"the feed added columns the contract does not know: {unexpected}"
    assert got == expected, "column ORDER drifted from the delivered order"
    print(f"  OK: all {len(expected)} columns present, in the delivered order.")


# --------------------------------------------------------------------------- #
# 3. A 403 costs ONE request, not five (real call)                             #
# --------------------------------------------------------------------------- #
def test_not_entitled_is_not_a_retry_storm(context, monkeypatch):
    """`polite_http.http_get` treats 403 as rate-limiting and retries 4 times with
    exponential backoff. Every ticker outside the subscription returns 403, so routing them
    through that path would cost MINUTES PER TICKER doing nothing."""
    calls: list[str] = []
    real_get_once = client_mod.get_once

    def counting_get_once(url, **kwargs):
        calls.append(url)
        return real_get_once(url, **kwargs)

    monkeypatch.setattr(client_mod, "get_once", counting_get_once)

    started = time.time()
    with pytest.raises(NotEntitled) as raised:
        sharadar_get(context, "fundamentals", ticker=NOT_ENTITLED, dimension="ARQ",
                     sort="date.asc", **{"date.gte": "2021-01-01"})
    elapsed = time.time() - started

    print("\n=== SANITY CHECK: a non-entitled ticker costs one request ===")
    print(f"  ticker            : {NOT_ENTITLED} (outside the subscription)")
    print(f"  raised            : {type(raised.value).__name__}({raised.value})")
    print(f"  requests issued   : {len(calls)}  (the retrying path would issue 5)")
    print(f"  elapsed           : {elapsed:.2f}s  (4 exponential backoffs would be >45s)")

    assert len(calls) == 1, f"a 403 must cost exactly one request, issued {len(calls)}"
    assert elapsed < 30, f"took {elapsed:.1f}s -- that is a retry storm, not a classification"
    print("  OK: classified as not-entitled off a single GET, no backoff burned.")


# --------------------------------------------------------------------------- #
# 4. Resume is incremental (real fetch, twice)                                 #
# --------------------------------------------------------------------------- #
def _rows_for(context, ticker: str) -> int:
    """Row count for ONE ticker, not the whole table. A whole-table count makes this test
    fail whenever anything else writes concurrently (a parallel `fundamentals-sharadar` run
    counts as 45 phantom new rows) -- and the property under test is per-ticker anyway."""
    frame = context.store.load(Tables.sharadar_fundamentals, columns=["ticker"],
                               where={"ticker": ticker})
    return 0 if frame is None else len(frame)


def test_resume_is_incremental(context):
    """The second run must write nothing and must never ask for a date at or before the
    stored max -- that is the whole point of resuming from `max_date_by` rather than
    re-pulling the window."""
    fetch_sharadar_tickers(context)          # the USD assertion needs this dimension
    fetch_sharadar_fundamentals(context, tickers=[ENTITLED],
                                years_history=int(context.config.data_extract
                                                  .sharadar_years_history))
    after_first = _rows_for(context, ENTITLED)
    stored_max = context.store.max_date_by(Tables.sharadar_fundamentals,
                                           "ticker").get(ENTITLED)

    requested: list[str] = []
    import src.data_extract.utils.fundamentals_sharadar.fetch_sharadar as fetch_mod
    real_get = fetch_mod.sharadar_get

    def recording_get(ctx, table, /, **kwargs):
        if "date.gte" in kwargs:
            requested.append(kwargs["date.gte"])
        return real_get(ctx, table, **kwargs)

    fetch_mod.sharadar_get = recording_get
    try:
        fetch_sharadar_fundamentals(context, tickers=[ENTITLED],
                                    years_history=int(context.config.data_extract
                                                      .sharadar_years_history))
    finally:
        fetch_mod.sharadar_get = real_get
    after_second = _rows_for(context, ENTITLED)

    too_early = [d for d in requested if pd.Timestamp(d) <= stored_max]

    print("\n=== SANITY CHECK: the second run is incremental ===")
    print(f"  {ENTITLED} rows after run 1 : {after_first}")
    print(f"  {ENTITLED} rows after run 2 : {after_second}  "
          f"(delta {after_second - after_first})")
    print(f"  stored max date   : {stored_max.date()}")
    print(f"  date.gte requested: {sorted(set(requested))}")
    print(f"  requests reaching at/before the stored max: {too_early or 'none'}")

    assert after_second == after_first, "the second run must write no new rows"
    assert requested, "the second run must still issue requests (with a resumed bound)"
    assert not too_early, f"resume asked for data it already has: {too_early}"
    print("  OK: run 2 wrote 0 rows and asked only for dates after the stored max.")


# --------------------------------------------------------------------------- #
# 5. Only the AS-REPORTED dimensions are stored (DB)                           #
# --------------------------------------------------------------------------- #
def test_dimensions_stored(context):
    """MRQ/MRY/MRT RESTATE IN PLACE, so their rows would mutate under an unchanged primary
    key and `diff_against_stored` could not tell an amendment from a bug. None must exist."""
    if context.store.row_count(Tables.sharadar_fundamentals) == 0:
        pytest.skip(f"{Tables.sharadar_fundamentals} is empty -- run fundamentals-sharadar")
    frame = context.store.load(Tables.sharadar_fundamentals,
                               columns=["ticker", "dimension", "date"])
    counts = frame["dimension"].value_counts().to_dict()
    stored = set(counts)
    mutating = sorted(d for d in stored if d.startswith("MR"))

    print("\n=== SANITY CHECK: stored dimensions are AS-REPORTED only ===")
    print(f"  rows per dimension : {counts}")
    print(f"  distinct tickers   : {frame['ticker'].nunique()}")
    print(f"  date range         : {frame['date'].min()} .. {frame['date'].max()}")
    print(f"  mutating (MR*) rows: {mutating or 'none'}")

    assert not mutating, f"restating dimensions must never be stored: {mutating}"
    assert stored <= set(SHARADAR_DIMENSIONS), f"unexpected dimensions: {stored}"
    print(f"  OK: {sorted(stored)} only, zero MR* rows.")
