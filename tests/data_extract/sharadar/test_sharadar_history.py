"""Tests for the Sharadar history build (src/data_extract/utils/fundamentals_sharadar/sharadar_history.py).

The SEC-independence and grain rules get offline known-truth fixtures (`FakeStore`); the column contract and the
coverage claims run on real data from Postgres and skip when it is unavailable. Each test prints its conclusion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import inspect

from src.data_extract.utils.fundamentals.kpi_catalogue import HISTORY_PROVENANCE
from src.data_extract.utils.fundamentals_sharadar.build_ttm import ARQ
from src.data_extract.utils.fundamentals_sharadar.field_map import load_field_map
from src.data_extract.utils.fundamentals_sharadar.sharadar_history import (
    HISTORY_KEYS,
    build_frame,
    build_sharadar_history,
    collapse_same_date,
    history_columns,
)
from src.data_store.schema import Tables, name_of
from tests.conftest import FakeStore

CONFIG_DIR = Path("./configs")
REPO_CONFIGS = Path(__file__).resolve().parents[3] / "configs"

#: The three CIK-cutover tickers D19 names; the continuity test SKIPS when the entitled roster lacks one.
CIK_CUTOVER = ("APA", "GOOGL", "ETN")


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def field_map():
    return load_field_map(str(CONFIG_DIR))


@pytest.fixture(scope="module")
def context():
    """A real Context (DB + .env), skipping rather than erroring when either is missing."""
    from src.context import get_config_context

    try:
        _, ctx = get_config_context(str(CONFIG_DIR), use_cache=False, save=False)
        with ctx.store.engine.connect():
            pass
        if ctx.store.row_count(Tables.sharadar_fundamentals_facts) == 0:
            pytest.skip(f"{Tables.sharadar_fundamentals_facts} is empty -- run fundamentals-sharadar")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"context/database unavailable ({type(exc).__name__}: {exc})")
    return ctx


@pytest.fixture(scope="module")
def sources(context):
    """The real inputs, loaded once: Sharadar ARQ and actions."""
    vendor = context.store.load(Tables.sharadar_fundamentals_facts, project=True, where={"dimension": ARQ})
    actions = context.store.load(Tables.sharadar_actions, project=True, optional=True)
    return vendor, actions


@pytest.fixture(scope="module")
def built(sources, field_map):
    vendor, actions = sources
    return build_frame(vendor, actions, field_map)


class _RecordingStore(FakeStore):
    """`FakeStore` that records every table it is asked to load."""

    def __init__(self, tables: dict[Any, pd.DataFrame]) -> None:
        super().__init__(tables)
        self.read: list[str] = []

    def load(self, table: Any, *args: Any, **kwargs: Any) -> pd.DataFrame | None:
        self.read.append(name_of(table))
        return super().load(table, *args, **kwargs)


def _vendor_arq(field_map) -> pd.DataFrame:
    """Six consecutive ARQ quarters of one ticker, every mapped SF1 column filled."""
    ends = [pd.Period(q, freq="Q").end_time.normalize() for q in ("2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1", "2024Q2")]
    frame = pd.DataFrame(
        {
            "ticker": "AAA",
            "dimension": ARQ,
            "calendardate": [e.date() for e in ends],
            "reportperiod": [e.date() for e in ends],
            "date": [(e + pd.Timedelta(days=40)).date() for e in ends],
            "fiscalperiod": [f"{e.year}-Q{e.quarter}" for e in ends],
        }
    )
    return frame.assign(**{str(s.source): 100.0 for s in field_map.direct.values()})


# --------------------------------------------------------------------------- #
# SEC independence (AC-008)                                                    #
# --------------------------------------------------------------------------- #
def test_the_build_reads_no_sec_table() -> None:
    """With no SEC table in the store the build runs, reads no `sec_*` table and writes exactly the declared columns."""
    field_map = load_field_map(str(REPO_CONFIGS))
    store = _RecordingStore(
        {
            Tables.sharadar_fundamentals_facts: _vendor_arq(field_map),
            Tables.sharadar_actions: pd.DataFrame(columns=["ticker", "date", "action", "value"]),
            Tables.prices_splits: pd.DataFrame(columns=["ticker", "date", "ratio"]),
        }
    )
    context = SimpleNamespace(store=store, log=logging.getLogger("test.sharadar_history"), config_dir=str(REPO_CONFIGS))
    build_sharadar_history(context, ["AAA"], config_dir=str(REPO_CONFIGS))
    saved = store.saved_frames(Tables.sharadar_fundamentals_history)
    print("\n=== SANITY CHECK: the Sharadar history build without any SEC table ===")
    print(f"  tables read: {sorted(set(store.read))}")
    print(f"  rows written: {len(saved[0]) if saved else 0}; columns {len(saved[0].columns) if saved else 0}")
    assert not [t for t in store.read if t.startswith("sec_")], f"the build read an SEC table: {sorted(set(store.read))}"
    assert len(saved) == 1 and len(saved[0]) == 6
    assert tuple(saved[0].columns) == tuple(Tables.sharadar_fundamentals_history.read_columns)
    assert not [c for c in saved[0].columns if c.endswith("_sec")]
    assert saved[0]["totalAssets"].notna().all() and saved[0]["totalRevenue"].notna().sum() == 3
    print("  OK: no sec_* read, 6 rows on the 77-column contract, instants on every row and TTM from the 4th quarter.")


# --------------------------------------------------------------------------- #
# the column contract                                                          #
# --------------------------------------------------------------------------- #
def test_column_contract(built, field_map):
    """The built frame's columns EQUAL the declared list, in order.

    List equality, not a subset check, and in ORDER: a silent column drift is invisible downstream because
    `pit.fundamentals_to_daily` returns an EMPTY FRAME for a column it cannot find rather than raising.
    """
    declared = history_columns(field_map)
    columns = tuple(built.columns)
    print(f"\ndeclared {len(declared)} | built {len(columns)}")
    print(f"built but not declared: {[c for c in columns if c not in set(declared)]}")
    print(f"declared but not built: {[c for c in declared if c not in set(columns)]}")
    assert columns == declared
    # Deliberately redundant: bump it WITH a contract change, never to make the test pass.
    # 92 -> 77 is the 14 `*_sec` columns and `stockholdersEquityInclNci` leaving (D8).
    assert len(columns) == 77, f"the Sharadar history contract is {len(columns)}, not 77"
    assert tuple(Tables.sharadar_fundamentals_history.read_columns) == declared, (
        "schema.py's read_columns and the field map state the same contract twice; they differ"
    )


def test_no_amendment_columns(built):
    """The SEC provenance column `publication_form` is ABSENT (D15): Sharadar ships no form."""
    present = [c for c in HISTORY_PROVENANCE if c in built.columns]
    print(f"\nSEC provenance columns: {list(HISTORY_PROVENANCE)}")
    print(f"present in the Sharadar history: {present or 'none -- as decided (D15)'}")
    assert not present
    assert "source" not in built.columns, "there is one source per table, so a per-row `source` would be a lie"


def test_value_columns_are_float(context, built):
    """Every value column is float64 in the frame and not TEXT in the database (the `ensure_table` regression)."""
    wrong = [c for c in built.columns if c not in HISTORY_KEYS and built[c].dtype != np.float64]
    print(f"\nin-frame: {len(built.columns) - len(HISTORY_KEYS)} value column(s); non-float64: {wrong or 'none'}")
    assert not wrong

    name = name_of(Tables.sharadar_fundamentals_history)
    if not context.store.exists(name):
        pytest.skip(f"{name} is not built yet -- run `fundamentals-sharadar`")
    types = {c["name"]: str(c["type"]) for c in inspect(context.store.engine).get_columns(name)}
    texty = {c: t for c, t in types.items() if c not in HISTORY_KEYS and any(mark in t.upper() for mark in ("TEXT", "CHAR", "STRING"))}
    print(f"in-DB: {len(types)} column(s); TEXT among the value columns: {texty or 'none'}")
    assert not texty


# --------------------------------------------------------------------------- #
# grain and coverage                                                           #
# --------------------------------------------------------------------------- #
def test_same_date_collapse_keeps_the_greatest_period():
    """Sharadar ships no form column, so the SEC 10-K-first rule has no analogue: the vendor's own
    rule is the greatest `reportperiod` on a duplicate `(ticker, date)`."""
    frame = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "as_of": pd.to_datetime(["2024-02-15", "2024-02-15", "2024-02-15"]),
            "fiscal_end": pd.to_datetime(["2023-09-30", "2023-12-31", "2023-12-31"]),
            "totalRevenue": [1.0, 2.0, 3.0],
        }
    )
    kept, dropped = collapse_same_date(frame)
    print(f"\nkept:\n{kept.to_string(index=False)}\ndropped:\n{dropped.to_string(index=False)}")
    assert len(kept) == 2 and len(dropped) == 1
    assert kept.loc[kept["ticker"] == "AAA", "totalRevenue"].iloc[0] == 2.0
    assert not kept.duplicated(["ticker", "as_of"]).any()


def test_history_grain_is_one_row_per_publication(built):
    """PK (ticker, as_of), asserted on the built frame rather than trusted to the upsert."""
    duplicated = built[built.duplicated(["ticker", "as_of"], keep=False)]
    print(
        f"\n{len(built)} row(s), {built['ticker'].nunique()} ticker(s), "
        f"{built['as_of'].min().date()}..{built['as_of'].max().date()}; "
        f"duplicate keys: {len(duplicated)}"
    )
    assert duplicated.empty


def test_headcount_is_not_in_the_contract(field_map):
    """Headcount is not a Sharadar history column: the cube reads `fundamentals_employees` directly."""
    declared = Tables.sharadar_fundamentals_history.read_columns
    print(f"\nSharadar history contract: {len(declared)} column(s)")
    assert "employees" not in field_map.outputs
    assert not [c for c in declared if c.startswith("employees")]
    assert len(history_columns(field_map)) == 77
    print("OK: no headcount column in the field map or the contract (77 columns)")


@pytest.mark.parametrize("ticker", CIK_CUTOVER)
def test_cik_cutover_continuity(sources, built, ticker):
    """D19: the series is keyed on `ticker`, so the 3 CIK-cutover names need explicit continuity.

    SKIPPED rather than quietly passing when the entitled roster does not carry the ticker.
    """
    vendor, _ = sources
    if ticker not in set(vendor["ticker"]):
        pytest.skip(f"{ticker} is not in the entitled Sharadar roster -- D19's CIK-cutover continuity is UNVERIFIABLE here, not verified")
    rows = built[built["ticker"] == ticker].sort_values("as_of")
    print(f"\n{ticker}: {len(rows)} row(s), {rows['as_of'].min()}..{rows['as_of'].max()}")
    assert rows["totalAssets"].notna().sum() > 0
    gaps = rows["as_of"].diff().dt.days.dropna()
    print(f"largest gap between publications: {gaps.max()} day(s)")
    assert gaps.max() < 200, "a CIK cutover left a hole in the series"
