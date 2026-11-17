"""StepManagerProfile: the nightly enrichment of `sec_13f_manager_details` with the profile columns.

The store is an in-memory SQLite database seeded with a small hand-built panel of four managers over eight report quarters (known
holdings, so the plumbing is checked, not the economics) and, when the dev database is reachable, with a real bounded sample of
2024 books checked against the pure per-quarter pipeline. No DAG, cube build or PostgreSQL write is made.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from omegaconf import DictConfig

from src.data_aggregate.transformers.step_manager_profile import StepManagerProfile
from src.data_aggregate.utils.manager_profile.lines import LINE_COLUMNS
from src.data_aggregate.utils.manager_profile.profile import profile_rows
from src.data_aggregate.utils.manager_profile.refresh import PROFILE_COLUMNS
from src.data_extract.utils.institutionals.manager_rows import EXTRACT_COLUMNS
from src.data_store.schema import Tables
from src.utils.window import WindowRun
from tests.canonical_schema import create_canonical
from tests.data_aggregate.test_manager_profile import live_prefixes, live_reference, live_store, load_raw, run_quarters, sample_ciks
from tests.support.config import fixture_config
from tests.support.windows import run_cube

ROOT = Path(__file__).resolve().parents[2]
PERIODS = list(pd.date_range("2022-03-31", periods=8, freq="QE"))
MANAGERS = {
    "0000000001": "ACME PENSION FUND",
    "0000000002": "FIRST NATIONAL BANK",
    "0000000003": "ZETA CAPITAL MANAGEMENT",
    "0000000004": "J SMITH",
}
TICKERS = [f"T{n}" for n in range(1, 7)]
CUSIPS = {ticker: f"{n:06d}10{n}" for n, ticker in enumerate(TICKERS, start=1)}
SECTORS = ["Technology", "Energy", "Healthcare", "Utilities", "Industrials", "Technology"]


def _quarter_lines(index: int, period: pd.Timestamp) -> list[dict[str, Any]]:
    """Each manager holds two thirds of the tickers (a different third per quarter) at a smooth price path."""
    rows: list[dict[str, Any]] = []
    for m, cik in enumerate(MANAGERS):
        accession = f"{cik}-{period:%y%m%d}"
        filed = period + pd.Timedelta(days=40)
        marker = {
            "cusip": "_empty",
            "ticker": "_empty",
            "issuer_name": None,
            "position_type": None,
            "shares": None,
            "value_usd": None,
            "form": "13F-HR",
        }
        rows.append({"cik": cik, "period": period, "filing_date": filed, "accession_number": accession, "parse_status": "complete", **marker})
        for j, ticker in enumerate(TICKERS):
            if (m + j + index) % 3 == 0:
                continue
            shares = 1000.0 * (j + 1) * (1 + 0.2 * ((m + index) % 3))
            price = 10.0 * (j + 1) * (1 + 0.02 * index)
            line = {"cusip": CUSIPS[ticker], "ticker": ticker, "issuer_name": f"{ticker} CORP", "position_type": "common"}
            rows.append(
                {"cik": cik, "period": period, "filing_date": filed, "accession_number": accession, "parse_status": "complete", "form": None}
                | line
                | {"shares": shares, "value_usd": shares * price}
            )
    return rows


def _extract_rows(index: int, period: pd.Timestamp) -> pd.DataFrame:
    """The extract writer's columns for one quarter (what `thirteen-f-managers` stores)."""
    rows = [
        {
            "cik": cik,
            "period": period,
            "manager_name": name,
            "state_or_country": None,
            "report_type": "holdings",
            "sec_file_number": None,
            "accession_number": f"{cik}-{period:%y%m%d}",
            "first_filed_date": period + pd.Timedelta(days=40),
            "source": "edgar_index",
            "source_period": period,
        }
        for cik, name in MANAGERS.items()
    ]
    return pd.DataFrame(rows, columns=list(EXTRACT_COLUMNS))


def _add_quarter(store: Any, index: int) -> None:
    """Lines, then the extract row, of report quarter `index` (the nightly order: fetch, extract, profile); the extract creates the table whole."""
    store.create_declared(Tables.sec_13f_manager_details)
    store.save(Tables.sec13f_hr, pd.DataFrame(_quarter_lines(index, PERIODS[index])))
    store.save(Tables.sec_13f_manager_details, _extract_rows(index, PERIODS[index]))


@pytest.fixture
def cfg() -> DictConfig:
    return fixture_config()


@pytest.fixture
def store(sqlite_store: Any) -> Any:
    """Reference tables and `sec13f_hr` from the canonical DDL; the manager table is created whole by the extract writer (`create_declared`)."""
    create_canonical(sqlite_store.engine, [Tables.sec13f_hr, Tables.cusip_ticker_map, Tables.sharadar_tickers])
    sqlite_store.save(Tables.cusip_ticker_map, pd.DataFrame({"cusip": list(CUSIPS.values()), "ticker": list(CUSIPS)}))
    tickers = pd.DataFrame(
        {
            "table": "SEP",
            "permaticker": range(1, 7),
            "ticker": TICKERS,
            "category": "Domestic Common Stock",
            "sector": SECTORS,
            "lastpricedate": pd.Timestamp("2026-01-02"),
        }
    )
    sqlite_store.save(Tables.sharadar_tickers, tickers)
    return sqlite_store


class Reads:
    """Records each `store.load` of `sec13f_hr` (the report quarters its cik/period filter names)."""

    def __init__(self, store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tables: list[str] = []
        self.periods: list[str] = []
        self.manager_columns: list[Any] = []
        load = store.load

        def _spy(table: Any, *args: Any, **kwargs: Any) -> Any:
            name = getattr(table, "name", table)
            self.tables.append(name)
            if name == Tables.sec_13f_manager_details.name:
                self.manager_columns.append(kwargs.get("columns"))
            if name == Tables.sec13f_hr.name:
                period = (kwargs.get("where") or {})["period"]
                self.periods.extend([period] if isinstance(period, str) else period)
            return load(table, *args, **kwargs)

        monkeypatch.setattr(store, "load", _spy)

    def quarters(self) -> list[str]:
        return sorted(set(self.periods))


def _step(store: Any, cfg: DictConfig) -> StepManagerProfile:
    context = types.SimpleNamespace(
        store=store, config=cfg, config_dir=ROOT / "configs", run=WindowRun("incremental", "profile source snapshot", pd.Timestamp("2026-11-16"))
    )
    return StepManagerProfile(context=context, config=cfg)  # type: ignore[arg-type]


def _table(store: Any) -> pd.DataFrame:
    df = store.load(Tables.sec_13f_manager_details)
    df["period"] = pd.to_datetime(df["period"])
    return df.sort_values(["cik", "period"], ignore_index=True)


def _stamp(period: pd.Timestamp) -> str:
    return f"{period:%Y-%m-%d}"


def test_the_table_is_declared_whole_so_the_profile_step_adds_no_column(store: Any, cfg: DictConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    _add_quarter(store, 0)  # a single quarter: no turnover, cap or active-share evidence, so those float columns are all null
    declared = list(EXTRACT_COLUMNS) + list(PROFILE_COLUMNS)
    assert store.columns(Tables.sec_13f_manager_details) == declared, "the extract's table already holds all 40 columns, profile ones empty"
    calls: list[tuple[str, pd.DataFrame]] = []
    save = store.save
    monkeypatch.setattr(store, "ensure_columns", lambda *a, **k: pytest.fail("the profile step alters no table"))
    monkeypatch.setattr(store, "save", lambda table, df, *a, **k: calls.append(("save", df)) or save(table, df, *a, **k))

    changed = run_cube(_step(store, cfg), full=False)

    assert [name for name, _ in calls] == ["save"], "one upsert, no ALTER first"
    assert set(calls[0][1].columns) == {"cik", "period", *PROFILE_COLUMNS}, "only the profile columns (and the key) are sent"
    assert store.columns(Tables.sec_13f_manager_details) == declared
    df = _table(store)
    assert changed == len(MANAGERS) and df["turnover"].isna().all() and df["active_share"].isna().all()
    assert df["category"].notna().all() and df["n_quarters_used"].eq(1).all()
    print("\n=== SANITY CHECK: table declared whole ===")
    print(f"{len(df)} rows; 40 columns from the extract's creation, profile upserted with no ALTER. Validated.")


def test_profile_step_first_on_an_empty_database_writes_nothing_and_the_extract_then_creates_the_table(store: Any, cfg: DictConfig) -> None:
    assert run_cube(_step(store, cfg), full=False) == 0 and not store.exists(Tables.sec_13f_manager_details), "no table, no rows, no error"
    _add_quarter(store, 0)
    assert store.columns(Tables.sec_13f_manager_details) == list(EXTRACT_COLUMNS) + list(PROFILE_COLUMNS)
    assert run_cube(_step(store, cfg), full=False) == len(MANAGERS)
    print("\n=== SANITY CHECK: profile step first ===")
    print("an empty database ends the profile run quietly; after the extract's write the next profile run fills 4 rows. Validated.")


def test_second_run_changes_nothing_reads_no_lines_and_leaves_the_extract_columns(
    store: Any, cfg: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    for index in range(len(PERIODS)):
        _add_quarter(store, index)
    before = _table(store)[list(EXTRACT_COLUMNS)]

    first = run_cube(_step(store, cfg), full=False)
    table = _table(store)
    reads = Reads(store, monkeypatch)
    second = run_cube(_step(store, cfg), full=False)

    assert first == len(table) == len(PERIODS) * len(MANAGERS) and second == 0
    assert reads.periods == [] and Tables.sec13f_hr.name not in reads.tables, "a night without new or changed rows reads no line"
    assert len(reads.manager_columns) == 1 and "state_or_country" not in reads.manager_columns[0], "the manager table is read once, projected"
    assert {"cik", "period", "source_period", "manager_name", "first_filed_date", *PROFILE_COLUMNS} == set(reads.manager_columns[0])
    pd.testing.assert_frame_equal(_table(store), table)
    pd.testing.assert_frame_equal(table[list(EXTRACT_COLUMNS)], before, check_dtype=False)
    assert table["n_quarters_used"].max() == 4 and table["turnover"].notna().sum() > 0
    print("\n=== SANITY CHECK: idempotent no-op ===")
    print(
        f"first run wrote {first} rows, second wrote {second}; reads of sec13f_hr on the second night: {len(reads.periods)}; extract columns identical."
    )


def test_a_snapped_row_is_profiled_from_the_lines_stored_under_its_source_period(
    store: Any, cfg: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A filing the extract placed in a quarter end is read under its own (mistyped) period and profiled as that quarter's row."""
    for index in range(len(PERIODS)):
        _add_quarter(store, index)
    run_cube(_step(store, cfg), full=False)
    reference = _table(store)

    quarter, cik, typo = PERIODS[3], "0000000002", pd.Timestamp("2022-11-15")
    store.update(Tables.sec13f_hr, {"period": typo}, {"cik": cik, "period": _stamp(quarter)})  # the filer typed a mid-quarter period
    store.update(Tables.sec_13f_manager_details, {"source_period": typo}, {"cik": cik, "period": _stamp(quarter)})
    reads = Reads(store, monkeypatch)
    changed = run_cube(_step(store, cfg), full=True)
    full_reads = list(reads.periods)
    table = _table(store)
    reads = Reads(store, monkeypatch)
    second = run_cube(_step(store, cfg), full=False)

    assert changed == 0, "the snapped row profiles exactly as when its lines were stored under the quarter end"
    pd.testing.assert_frame_equal(table.drop(columns="source_period"), reference.drop(columns="source_period"))
    assert _stamp(typo) in full_reads and table["period"].dt.is_quarter_end.all()
    assert second == 0 and reads.periods == [], "the next night reads no line with a snapped row present"
    row = table[(table["cik"] == cik) & (table["period"] == quarter)].iloc[0]
    print("\n=== SANITY CHECK: snapped row ===")
    print(
        f"row {cik} {_stamp(quarter)} (lines under {_stamp(typo)}): category {row['category']}, hhi {row['hhi']:.4f}; full rebuild changed {changed}, next night read {len(reads.periods)}."
    )


def test_new_quarter_recomputes_only_the_affected_quarters_and_equals_a_full_rebuild(
    store: Any, cfg: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    for index in range(len(PERIODS) - 1):
        _add_quarter(store, index)
    run_cube(_step(store, cfg), full=False)
    _add_quarter(store, len(PERIODS) - 1)  # the nightly extract adds the new quarter's rows
    reads = Reads(store, monkeypatch)

    changed = run_cube(_step(store, cfg), full=False)

    window = int(cfg.data_aggregate.manager_profile.window_quarters)
    expected = [_stamp(p) for p in PERIODS[-window - 1 :]]
    read = reads.quarters()
    assert read == expected, "the new quarter and the window of quarters before it"
    new_rows = len(MANAGERS)
    assert new_rows <= changed <= new_rows * 2, "the new quarter's rows and, at most, the preceding quarter's rewritten rows"
    incremental = _table(store)
    assert run_cube(_step(store, cfg), full=True) == 0, "a full rebuild reproduces the incremental table"
    pd.testing.assert_frame_equal(_table(store), incremental)
    print("\n=== SANITY CHECK: incremental equals full ===")
    print(f"new quarter {_stamp(PERIODS[-1])}: read quarters {read}; {changed} rows changed; full rebuild changes 0 rows.")


def test_a_changed_extract_row_triggers_a_recompute_of_its_quarters(store: Any, cfg: DictConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    for index in range(len(PERIODS)):
        _add_quarter(store, index)
    run_cube(_step(store, cfg), full=False)
    stored = _table(store)

    renamed = _extract_rows(2, PERIODS[2]).iloc[[3]].assign(manager_name="J SMITH PENSION PLAN")  # the extract rewrote a name
    store.save(Tables.sec_13f_manager_details, renamed)
    reads = Reads(store, monkeypatch)
    changed = run_cube(_step(store, cfg), full=False)

    after = _table(store)
    row = after[(after["cik"] == "0000000004") & (after["period"] == PERIODS[2])].iloc[0]
    unchanged = stored[(stored["cik"] == "0000000004") & (stored["period"] == PERIODS[2])].iloc[0]
    assert changed >= 1 and row["category"] == "pension" and unchanged["category"] == "fund"
    read = reads.quarters()
    assert _stamp(PERIODS[2]) in read and _stamp(PERIODS[-1]) not in read, "only the quarter's window is recomputed"
    assert (after["category"][after["cik"] != "0000000004"] == stored["category"][stored["cik"] != "0000000004"]).all()
    assert run_cube(_step(store, cfg), full=True) == 0
    print("\n=== SANITY CHECK: changed extract row ===")
    print(f"name change at {_stamp(PERIODS[2])}: category -> {row['category']}, {changed} rows rewritten, read quarters {read}.")


def test_real_sample_books_match_the_pure_pipeline(store: Any, cfg: DictConfig) -> None:
    """A real bounded sample (three 2024 quarters, about 60 managers) written into SQLite and profiled by the step."""
    live = live_store()
    periods = ["2024-03-31", "2024-06-30", "2024-09-30"]
    ciks = sample_ciks(live, periods[-1], 50)
    raw = load_raw(live, ciks, periods)
    details = live.load(Tables.sec_13f_manager_details, where={"cik": ciks, "period": periods})
    assert details is not None
    details = details[[column for column in EXTRACT_COLUMNS if column in details.columns]]
    store.save(Tables.sec13f_hr, pd.concat(raw.values(), ignore_index=True)[list(LINE_COLUMNS)].assign(parse_status="complete"))
    store.create_declared(Tables.sec_13f_manager_details)
    store.save(Tables.sec_13f_manager_details, details)
    store.save(Tables.cusip_ticker_map, live.load(Tables.cusip_ticker_map))
    store.save(Tables.sharadar_tickers, live.load(Tables.sharadar_tickers))

    changed = run_cube(_step(store, cfg), full=True)

    table = _table(store)
    reference, prefixes = live_reference(live), live_prefixes(live)
    expected = profile_rows(run_quarters(raw, reference, prefixes, cfg.data_aggregate.manager_profile), cfg.data_aggregate.manager_profile)
    merged = table.merge(expected, on=["cik", "period"], suffixes=("", "_expected"))
    last = merged[merged["period"] == pd.Timestamp(periods[-1])]
    assert changed == len(details) and len(last) >= 40
    for column in ("hhi", "turnover", "active_share", "pct_equity_us", "book_value_usd"):
        pd.testing.assert_series_equal(last[column], last[f"{column}_expected"], check_names=False, rtol=1e-9)
    assert table["category"].notna().all()
    print("\n=== SANITY CHECK: real sample through the step ===")
    print(
        f"{len(table)} rows ({len(last)} managers at {periods[-1]}); the step's profile equals the pure pipeline's on hhi, turnover, active share, mix and book value."
    )
