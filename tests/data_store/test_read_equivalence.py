"""
The data-layer refactor guard: every read that MOVES from raw SQL onto `DataStore` must
return the same frame it returned before.

WHY THIS EXISTS. `tests/data_aggregate/test_aggregate_regression.py` hashes the output of the
aggregation BUILDERS over frozen inputs -- it never calls `store.load`, so it is structurally
blind to an I/O regression. And at HEAD it cannot cover the target/beta path at all (its
baseline predates commit 0053dc3, so all 6 labels and `panel.betas` are excluded there). This
file is the other half: it pins the 15 raw-SQL shapes the refactor absorbs into the store, by
keeping each ORIGINAL query as a literal and asserting the new store call is frame-equal.

Each case names the production site it came from. When a site is migrated, its case stops
being a claim about the future and becomes a regression test for the code that shipped.

EVERY CASE MUST BE BOUNDED. THIS IS A HARD RULE, not a style preference. The tables are
multi-GB: `sec13f_hr` 21.7M rows, `cube` 5.6M, `cube_part_prices` 1.85M, `prices` 1.8M. An
equivalence check materializes the frame TWICE (old query + new call), so an unbounded case
does not merely run slowly -- an earlier version of this file scoped two cases by `since=`
alone and exhausted the machine's memory.

So: every row-returning case carries a KEY predicate (one ticker) or a LIMIT, and the
unbounded shapes are only ever exercised against the small tables (`cube_part_market`, 15k
rows). That still tests the thing that actually breaks -- predicate COMPOSITION, column
PROJECTION and date/type BINDING. It deliberately proves nothing about volume; `iter_load`'s
memory behaviour is not something a frame-equality test can assert anyway.

Aggregate-only shapes (MAX / MIN / COUNT / DISTINCT) return a handful of rows regardless of
table size, so they need no extra scoping.

CAPABILITY-GATED. The store methods land in phase 2 of the refactor, so a case whose method
does not exist yet SKIPS with an explicit reason, and `test_every_case_is_live` reports how
many remain. That test is the phase-6 gate: it must report zero.
"""
from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import bindparam, text

from src.data_store.errors import TableEmptyError, TableMissingError
from src.data_store.store import DataStore
from src.utils.db import get_engine

# A ticker with deep history in every per-ticker table, and a short window that is cheap to scan.
TICKER = "AAPL"
SINCE = pd.Timestamp("2024-01-02")


@pytest.fixture(scope="module")
def store() -> DataStore:
    """The LIVE store. These are equivalence checks against real stored rows: a synthetic
    SQLite fixture cannot show that a rewritten predicate still selects the same data."""
    engine = get_engine()
    try:
        with engine.connect():
            pass
    except Exception as exc:                                        # noqa: BLE001
        pytest.skip(f"database unavailable: {type(exc).__name__}")
    return DataStore(engine)


def _sql(store: DataStore, sql: str, params: dict | None = None,
         expanding: str | None = None) -> pd.DataFrame:
    """Run an ORIGINAL query verbatim and return its frame."""
    stmt = text(sql)
    if expanding is not None:
        stmt = stmt.bindparams(bindparam(expanding, expanding=True))
    with store.engine.connect() as conn:
        return pd.read_sql(stmt, conn, params=params or {})


def _requires(store: DataStore, *methods: str) -> None:
    missing = [m for m in methods if not hasattr(store, m)]
    if missing:
        pytest.skip(f"DataStore.{'/'.join(missing)} not implemented yet (refactor phase 2)")


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """Column order and row order are NOT part of the contract -- content is."""
    return (df.reindex(sorted(df.columns), axis=1)
              .sort_values(by=sorted(df.columns), kind="mergesort")
              .reset_index(drop=True))


# --------------------------------------------------------------------------- #
# introspection                                                                #
# --------------------------------------------------------------------------- #
def test_max_date_matches_part_io_max_date(store):
    """`PartStore.max_date` -- `SELECT MAX(date) FROM "<part>"` (part_io.py:52)."""
    _requires(store, "max_date")
    old = _sql(store, 'SELECT MAX(date) AS m FROM "cube_part_prices"')["m"].iloc[0]
    assert store.max_date("cube_part_prices") == pd.Timestamp(old).normalize()


def test_columns_matches_part_io_columns(store):
    """`PartStore.columns` -- `SELECT * FROM "<part>" LIMIT 0` (part_io.py:64)."""
    _requires(store, "columns")
    old = list(_sql(store, 'SELECT * FROM "cube_part_market" LIMIT 0').columns)
    assert list(store.columns("cube_part_market")) == old


def test_columns_matches_information_schema_for_cube(store):
    """`step_train._cube_columns` / `ls_model._cube_columns` -- information_schema
    (step_train.py:156, ls_model.py:71). `inspect()` resolves through `search_path` and is
    stricter than an unqualified `table_name = 'cube'`, so this pins that they agree HERE."""
    _requires(store, "columns")
    old = set(_sql(store, "SELECT column_name FROM information_schema.columns "
                          "WHERE table_name = 'cube'")["column_name"])
    assert set(store.columns("cube")) == old


def test_row_count_matches_part_io_row_count(store):
    """`PartStore.row_count` -- `SELECT COUNT(*) FROM "<part>"` (part_io.py:72)."""
    _requires(store, "row_count")
    old = int(_sql(store, 'SELECT COUNT(*) AS n FROM "cube_part_market"')["n"].iloc[0])
    assert store.row_count("cube_part_market") == old


def test_bounds_matches_hf_transcripts_min_max(store):
    """`fetch_hf_transcripts` -- `SELECT MIN(quarter), MAX(quarter) FROM
    "earnings_call_sections"` (fetch_hf_transcripts.py:164). NOT dates: string quarters."""
    _requires(store, "bounds")
    old = _sql(store, 'SELECT MIN(quarter) AS lo, MAX(quarter) AS hi '
                      'FROM "earnings_call_sections"')
    assert store.bounds("earnings_call_sections", "quarter") == (
        old["lo"].iloc[0], old["hi"].iloc[0])


def test_max_date_matches_freshness_max_date(store):
    """`freshness._max_date` -- `SELECT MAX("<col>") FROM "<table>"` (freshness.py:50), on a
    table whose freshness column is the FILING date, not its period end."""
    _requires(store, "max_date")
    old = _sql(store, 'SELECT MAX("filed") AS m FROM "notes_num"')["m"].iloc[0]
    assert store.max_date("notes_num", "filed") == pd.Timestamp(old).normalize()


# --------------------------------------------------------------------------- #
# distinct                                                                     #
# --------------------------------------------------------------------------- #
def test_distinct_matches_sec_utils_ingested_quarters(store):
    """`sec_utils.bulk_ingested_quarters` -- `SELECT DISTINCT quarter FROM "<table>"`
    (sec_utils.py:107)."""
    _requires(store, "distinct")
    old = set(_sql(store, 'SELECT DISTINCT quarter FROM "earnings_call_sections"')["quarter"]
              .dropna())
    assert set(store.distinct("earnings_call_sections", "quarter")) == old


def test_distinct_with_notnull_matches_step_train_horizons(store):
    """`step_train._distinct_horizons` -- `SELECT DISTINCT target_horizon FROM cube
    WHERE "<target>" IS NOT NULL ORDER BY target_horizon` (step_train.py:114). This is the
    case that needs an IS NOT NULL predicate composed with DISTINCT."""
    _requires(store, "distinct", "NOT_NULL")
    old = _sql(store, 'SELECT DISTINCT target_horizon FROM cube '
                      'WHERE "target_rank" IS NOT NULL ORDER BY target_horizon'
               )["target_horizon"].tolist()
    new = store.distinct("cube", "target_horizon",
                         where={"target_rank": store.NOT_NULL}, order="asc")
    assert new == old


def test_distinct_ordered_and_limited_matches_step_train_recent_dates(store):
    """`step_train` latest dates -- `SELECT DISTINCT date FROM cube ORDER BY date DESC
    LIMIT :n` (step_train.py:809)."""
    _requires(store, "distinct")
    old = _sql(store, "SELECT DISTINCT date FROM cube ORDER BY date DESC LIMIT :n",
               {"n": 5})["date"].tolist()
    new = store.distinct("cube", "date", order="desc", limit=5)
    assert [pd.Timestamp(d) for d in new] == [pd.Timestamp(d) for d in old]


# --------------------------------------------------------------------------- #
# reads: range predicate, projection, IN, IS NOT NULL                          #
# --------------------------------------------------------------------------- #
def test_since_matches_part_io_read_since(store):
    """`PartStore.read(since=)` -- `SELECT <cols> FROM "<part>" WHERE date >= :since`
    (part_io.py:86). The date is BOUND, not formatted into the string.

    Run against `cube_part_market` (15k rows), the only part small enough to compare
    unscoped. `cube_part_prices` is 1.85M rows and is covered by the ticker-scoped case
    below."""
    _requires(store, "load")
    cols = ["date", "ticker", "close"]
    old = _sql(store, 'SELECT "date", "ticker", "close" FROM "cube_part_market" '
                      "WHERE date >= :since", {"since": SINCE.strftime("%Y-%m-%d")})
    new = store.load("cube_part_market", columns=cols, since=SINCE)
    pd.testing.assert_frame_equal(_norm(new), _norm(old), check_dtype=False)


def test_since_composes_with_a_key_predicate_on_a_large_part(store):
    """The same `date >= :since` on the 1.85M-row `cube_part_prices`, scoped to ONE ticker so
    the comparison stays bounded -- this is the shape every cube sub-step actually issues."""
    _requires(store, "load")
    cols = ["date", "ticker", "close"]
    old = _sql(store, 'SELECT "date", "ticker", "close" FROM "cube_part_prices" '
                      "WHERE date >= :since AND ticker = :t",
               {"since": SINCE.strftime("%Y-%m-%d"), "t": TICKER})
    new = store.load("cube_part_prices", columns=cols, since=SINCE, where={"ticker": TICKER})
    pd.testing.assert_frame_equal(_norm(new), _norm(old), check_dtype=False)


def test_since_matches_ls_model_prices_window(store):
    """`ls_model` -- `SELECT * FROM prices WHERE date >= :cut` (ls_model.py:101), scoped to one
    ticker (the table is 1.8M rows).

    The sibling cube query at ls_model.py:88-90 INTERPOLATES its dates straight into the SQL
    string; this pins that replacing both with one BOUND predicate selects the same rows."""
    _requires(store, "load")
    old = _sql(store, 'SELECT "date", "ticker", "close" FROM prices '
                      "WHERE date >= :cut AND ticker = :t",
               {"cut": SINCE.strftime("%Y-%m-%d"), "t": TICKER})
    new = store.load("prices", columns=["date", "ticker", "close"], since=SINCE,
                     where={"ticker": TICKER})
    pd.testing.assert_frame_equal(_norm(new), _norm(old), check_dtype=False)


def test_notnull_and_equality_compose_like_step_train_panel(store):
    """`step_train._load_horizon_panel` -- `SELECT <proj> FROM cube WHERE "<target>" IS NOT
    NULL AND target_horizon = :h` (step_train.py:132): a projection, an IS NOT NULL and an
    equality, ANDed. Scoped to one ticker so it does not scan 5.6M rows."""
    _requires(store, "load", "NOT_NULL")
    cols = ["date", "ticker", "target_horizon", "target_rank"]
    old = _sql(store, 'SELECT "date", "ticker", "target_horizon", "target_rank" FROM cube '
                      'WHERE "target_rank" IS NOT NULL AND target_horizon = :h '
                      "AND ticker = :t", {"h": 30, "t": TICKER})
    new = store.load("cube", columns=cols,
                     where={"target_rank": store.NOT_NULL, "target_horizon": 30,
                            "ticker": TICKER})
    pd.testing.assert_frame_equal(_norm(new), _norm(old), check_dtype=False)


def test_in_predicate_matches_fundamental_features_tag_pushdown(store):
    """`fundamental_features.load_tagged_facts` -- `WHERE tag IN :tags` via
    `bindparam(expanding=True)` (fundamental_features.py:417)."""
    _requires(store, "load")
    tags = ["DefinedBenefitPlanBenefitObligation",
            "DefinedBenefitPlanFairValueOfPlanAssets"]
    cols = ["adsh", "tag", "ddate", "qtrs", "value"]
    # the two-tag filter IS the bound here: it selects a few thousand of the 40k rows
    old = _sql(store, 'SELECT "adsh", "tag", "ddate", "qtrs", "value" FROM "notes_num" '
                      "WHERE tag IN :tags", {"tags": tags}, expanding="tags")
    new = store.load("notes_num", columns=cols, where={"tag": tags})
    assert len(old) < 50_000, f"case is not bounded any more ({len(old)} rows)"
    pd.testing.assert_frame_equal(_norm(new), _norm(old), check_dtype=False)


def test_iter_load_concatenates_to_the_same_frame(store):
    """`iter_load` must be a pure chunking of `load` -- same rows, same columns, no
    duplication or loss at the chunk boundaries (the keyset/stream_results path)."""
    _requires(store, "iter_load", "load")
    cols = ["date", "ticker", "close"]
    whole = store.load("cube_part_market", columns=cols)
    chunks = list(store.iter_load("cube_part_market", columns=cols, chunksize=1_000))
    assert len(chunks) > 1, "chunksize did not actually split the read"
    pd.testing.assert_frame_equal(_norm(pd.concat(chunks, ignore_index=True)),
                                  _norm(whole), check_dtype=False)


# --------------------------------------------------------------------------- #
# the load contract itself                                                     #
# --------------------------------------------------------------------------- #
def test_missing_table_raises_a_typed_error(store):
    """`load` fails loud on a missing table, but with a TYPED error -- so a caller can catch
    "not created yet" without also swallowing a mistyped column or a dead connection, which
    is what the eight bare `except Exception` blocks did."""
    with pytest.raises(TableMissingError):
        store.load("no_such_table_xyz")


def test_empty_result_raises_rather_than_returning_a_frame(store):
    """An empty read is a fault, not a value: no fabricated frame."""
    with pytest.raises(TableEmptyError):
        store.load("cube_part_market", where={"ticker": "__no_such_ticker__"})


def test_optional_returns_none_for_the_resume_reads(store):
    """The one documented exception: a fetcher asking "what do we already have?" against a
    cold database. None, not an empty frame -- that is what those callers branch on."""
    assert store.load("no_such_table_xyz", optional=True) is None


# --------------------------------------------------------------------------- #
# the phase-6 gate                                                             #
# --------------------------------------------------------------------------- #
def test_every_case_is_live(store):
    """The gate: by the end of the refactor no case may still be waiting on a store method.

    Skips are invisible in a green run, so this states the remaining work as an assertion
    instead. It is expected to FAIL until phase 2 lands, and the failure message is the
    to-do list."""
    needed = ("exists", "columns", "row_count", "max_date", "bounds", "distinct",
              "load", "iter_load", "append_tail", "drop", "NOT_NULL")
    missing = [m for m in needed if not hasattr(store, m)]
    print(f"\n[read-equivalence] {len(needed) - len(missing)}/{len(needed)} store "
          f"capabilities live")
    if missing:
        print(f"    still to implement: {', '.join(missing)}")
    else:
        print("    SANITY CHECK: every raw-SQL shape the refactor absorbs has a store "
              "equivalent, and each is pinned frame-equal to its original query.")
    assert not missing, f"DataStore is missing: {missing}"
