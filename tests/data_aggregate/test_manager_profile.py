"""Manager profile logic: line cleaning, single-quarter statistics and the 4-quarter profile.

Exact math uses hand-computed known-truth books; the real-data checks read a bounded sample of `sec13f_hr` managers (CIK list plus
periods on the primary key) for 2000Q4, 2008Q4 and 2024Q4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig

from src.data_aggregate.utils.manager_profile.lines import LINE_COLUMNS, clean_book
from src.data_aggregate.utils.manager_profile.profile import profile_rows
from src.data_aggregate.utils.manager_profile.quarter_stats import STAT_COLUMNS, compact_book, quarter_stats
from src.data_aggregate.utils.manager_profile.reference import classify_securities, prefix_tickers, quarter_aggregate, sharadar_reference
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.config import read_config
from src.utils.db import get_engine

ROOT = Path(__file__).resolve().parents[2]
Q0, Q1 = pd.Timestamp("2024-03-31"), pd.Timestamp("2024-06-30")
MIX = ["pct_equity_us", "pct_equity_foreign", "pct_etf", "pct_bond", "pct_options", "pct_other"]
CAPS = ["pct_cap_small", "pct_cap_mid", "pct_cap_large"]
SECTORS = [
    "pct_sector_basic_materials",
    "pct_sector_communication_services",
    "pct_sector_consumer_cyclical",
    "pct_sector_consumer_defensive",
    "pct_sector_energy",
    "pct_sector_financial_services",
    "pct_sector_healthcare",
    "pct_sector_industrials",
    "pct_sector_real_estate",
    "pct_sector_technology",
    "pct_sector_utilities",
    "pct_sector_unmapped",
]
# Real managers present in every sample quarter: Berkshire Hathaway, Vanguard Group, FMR (Fidelity), State Street.
KNOWN_CIKS = ["0001067983", "0000102909", "0000315066", "0000093751"]
NO_PREFIX = pd.Series(dtype=object)
REAL_QUARTERS = ["2000-12-31", "2008-12-31", "2024-12-31"]
CFG = read_config(str(ROOT / "configs")).build_cube.manager_profile


@pytest.fixture(scope="module")
def cfg() -> DictConfig:
    return read_config(str(ROOT / "configs")).build_cube.manager_profile


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "permaticker": [1, 2, 3, 4, 5, 6, 7],
            "ticker": ["T1", "T2", "T3", "T4", "T5", "DUP", "DUP"],
            "category": [
                "Domestic Common Stock",
                "Domestic Common Stock Primary Class",
                "ADR Common Stock",
                "Domestic Common Stock",
                "Canadian Common Stock",
                "Domestic Common Stock",
                "ADR Common Stock",
            ],
            "sector": ["Technology", "Energy", "Healthcare", "Utilities", "Basic Materials", "", "Real Estate"],
            "lastpricedate": pd.to_datetime(["2024-06-01"] * 5 + ["2024-06-01", "2020-01-01"]),
        }
    )
    return sharadar_reference(frame)


def _raw(cik: str, period: str, accession: str, lines: list[tuple], form: str = "13F-HR", filed: str = "2024-08-01") -> pd.DataFrame:
    """A marker row plus one row per line `(ticker, cusip, name, position_type, shares, value, extra)`."""
    base = {"cik": cik, "period": period, "accession_number": accession, "filing_date": filed, "form": form}
    rows = [{**base, "ticker": "_empty", "cusip": "_empty", "position_type": None}]
    for ticker, cusip, name, ptype, shares, value, extra in lines:
        row = {**base, "ticker": ticker, "cusip": cusip, "issuer_name": name, "position_type": ptype, "shares": shares, "form": None}
        row.update({"value_usd": value, "call_value": 0.0, "put_value": 0.0, "debt_value": 0.0, **extra})
        rows.append(row)
    df = pd.DataFrame(rows)
    for column in LINE_COLUMNS:
        if column not in df:
            df[column] = np.nan
    return df[list(LINE_COLUMNS)]


def _cleaned(cik: str, period: pd.Timestamp, items: list[tuple], reference: pd.DataFrame) -> pd.DataFrame:
    """Cleaned and classified lines of one fixture book; items are `(ticker, kind, shares, value, name[, title_of_class])`."""
    rows = [
        {
            "cik": cik,
            "period": period,
            "kind": kind,
            "key": ticker,
            "ticker": ticker if ticker != "_none" else np.nan,
            "cusip": ticker,
            "issuer_name": name,
            "title_of_class": title[0] if title else "COM",
            "filing_date": period + pd.Timedelta(days=40),
            "shares": shares,
            "value": value,
        }
        for ticker, kind, shares, value, name, *title in items
    ]
    df = pd.DataFrame(rows)
    df["weight"] = df["value"] / df.groupby("cik")["value"].transform("sum")
    return classify_securities(df, reference, CFG)


def test_sector_names_are_the_sharadar_spellings(reference: pd.DataFrame) -> None:
    assert reference.loc["T1", "sector_col"] == "pct_sector_technology"
    assert reference.loc["T5", "sector_col"] == "pct_sector_basic_materials"
    assert reference.loc["T2", "security_class"] == "us" and reference.loc["T3", "security_class"] == "foreign"
    assert reference.loc["T5", "security_class"] == "foreign"
    # The duplicated ticker keeps the row that has a sector.
    assert reference.loc["DUP", "sector_col"] == "pct_sector_real_estate"
    assert reference.index.is_unique
    print("\n=== SANITY CHECK: sharadar reference ===")
    print("Sector spellings map to the schema columns and the duplicated ticker keeps its sector row; reference is one row per ticker.")


def test_clean_book_keeps_the_original_accession_and_sums_duplicates(cfg: DictConfig) -> None:
    original = _raw(
        "0000000001",
        "2024-03-31",
        "A1",
        [
            ("T1", "C1", "ONE INC", "common", 100.0, 1000.0, {"call_value": 9e9}),
            ("T1", "C1", "ONE INC", "common", 50.0, 500.0, {}),
            ("T2", "C2", "TWO INC", "common", 10.0, -5.0, {}),
            ("T3", "C3", "THREE INC", "common", 10.0, 0.0, {}),
            ("T4", "C4", "FOUR INC", "common", 10.0, np.nan, {}),
            ("_empty", "C9", "OLD CO", "common", 20.0, 200.0, {}),
            ("_empty", "D1", "A BOND", "debt", 0.0, 0.0, {"debt_value": 700.0}),
            ("T1", "C1", "ONE INC", "call", 0.0, 0.0, {"call_value": 300.0}),
        ],
        filed="2024-05-10",
    )
    amendment = _raw(
        "0000000001",
        "2024-03-31",
        "A2",
        [("T1", "C1", "ONE INC", "common", 999.0, 99999.0, {})],
        form="13F-HR/A",
        filed="2024-06-01",
    )
    amendment_only = _raw("0000000002", "2024-03-31", "B1", [("T1", "C1", "ONE INC", "common", 10.0, 100.0, {})], form="13F-HR/A")
    before_period = _raw("0000000003", "2024-03-31", "C1", [("T1", "C1", "ONE INC", "common", 10.0, 100.0, {})], filed="2024-03-01")

    df = clean_book(pd.concat([original, amendment, amendment_only, before_period]), cfg, NO_PREFIX)

    one = df[df["cik"] == "0000000001"].set_index(["kind", "key"])
    assert set(df["cik"]) == {"0000000001", "0000000002"}
    assert one.loc[("common", "T1"), "value"] == 1500.0 and one.loc[("common", "T1"), "shares"] == 150.0
    assert one.loc[("common", "C9"), "value"] == 200.0 and pd.isna(one.loc[("common", "C9"), "ticker"])
    assert one.loc[("debt", "D1"), "value"] == 700.0 and one.loc[("call", "T1"), "value"] == 300.0
    assert len(one) == 4, "negative, zero and null values and the amendment are dropped"
    assert one["value"].sum() == 2700.0, "common rows' call_value / put_value are never added"
    assert np.isclose(one["weight"].sum(), 1.0)
    assert df.loc[df["cik"] == "0000000002", "value"].tolist() == [100.0], "amendment-only filer falls back to its earliest accession"
    print("\n=== SANITY CHECK: clean_book ===")
    print(
        "Original accession chosen, duplicates summed, bad values dropped, `_empty` ticker is no ticker, option columns not added; weights sum to 1."
    )


def test_clean_book_repairs_a_book_reported_in_the_wrong_unit(cfg: DictConfig) -> None:
    prices = {f"S{i}": 10.0 * i for i in range(1, 7)}
    frames = []
    for n, cik in enumerate(["0000000011", "0000000012", "0000000013", "0000000014"]):
        scale = 1000.0 if cik == "0000000014" else 1.0
        lines = [(f"S{i}", f"CS{i}", f"S{i} INC", "common", 100.0 * (n + 1), prices[f"S{i}"] * 100.0 * (n + 1) * scale, {}) for i in range(1, 7)]
        frames.append(_raw(cik, "2024-03-31", f"U{n}", lines))

    df = clean_book(pd.concat(frames), cfg, NO_PREFIX)

    totals = df.groupby("cik")["value"].sum()
    truth = {cik: sum(prices.values()) * 100.0 * (n + 1) for n, cik in enumerate(totals.index)}
    assert np.allclose(totals.to_numpy(), [truth[c] for c in totals.index]), "the x1000 book is rescaled, not dropped"
    print("\n=== SANITY CHECK: unit repair ===")
    print(f"Book totals {totals.round(0).to_dict()} equal the true-unit totals; the x1000 filer is back on the common unit.")


def test_turnover_new_rate_concentration_mix_sector_cap_active_share(reference: pd.DataFrame, cfg: DictConfig) -> None:
    q0 = _cleaned("A", Q0, [("T1", "common", 100.0, 1000.0, ""), ("T2", "common", 100.0, 2000.0, ""), ("T4", "common", 40.0, 2000.0, "")], reference)
    q1 = _cleaned("A", Q1, [("T1", "common", 100.0, 1000.0, ""), ("T2", "common", 50.0, 1000.0, ""), ("T3", "common", 50.0, 2000.0, "")], reference)

    stats = quarter_stats(q1, compact_book(q0), quarter_aggregate(q0, cfg.cap_breakpoints), cfg).iloc[0]

    assert stats["n_positions"] == 3 and np.isclose(stats["hhi"], 0.375)
    assert np.isclose(stats["turnover"], 0.8), "T2 changed by -50% (0.4) and T4 exited (0.4); T1 unchanged"
    assert np.isclose(stats["new_position_rate"], 1 / 3), "T3 is the only new position of three"
    assert np.isclose(stats["pct_equity_us"], 0.5) and np.isclose(stats["pct_equity_foreign"], 0.5)
    assert stats[["pct_bond", "pct_options", "pct_etf", "pct_other"]].eq(0).all()
    assert np.isclose(stats["pct_sector_technology"], 0.25) and np.isclose(stats["pct_sector_energy"], 0.25)
    assert np.isclose(stats["pct_sector_healthcare"], 0.5) and np.isclose(stats[SECTORS].sum(), 1.0)
    assert np.isclose(stats["active_share"], 0.55), "0.5 x (.05 + .15 + .5 + .4) against the previous aggregate"
    assert np.isclose(stats["pct_cap_mid"], 0.25) and np.isclose(stats["pct_cap_large"], 0.25) and stats["pct_cap_small"] == 0.0
    assert np.isclose(stats["mapped_share"], 0.5), "T3 has no previous-quarter holder: no size"
    assert stats["book_value_usd"] == 4000.0
    print("\n=== SANITY CHECK: quarter statistics ===")
    print("hhi .375, turnover .8, new rate 1/3, active share .55, caps and sectors equal the hand computation.")


def test_asset_mix_sums_to_one_with_bonds_and_options(reference: pd.DataFrame, cfg: DictConfig) -> None:
    items = [
        ("T1", "common", 10.0, 2000.0, "ONE INC"),
        ("T3", "common", 10.0, 1000.0, "THREE ADR"),
        ("_none", "common", 10.0, 500.0, "ISHARES CORE S&P 500 ETF"),
        ("U9", "common", 10.0, 500.0, "ACME CORP", "ORD"),
        ("B1", "debt", 0.0, 1000.0, "A BOND"),
        ("T1", "call", 0.0, 400.0, "ONE INC"),
        ("T1", "put", 0.0, 100.0, "ONE INC"),
    ]
    lines = _cleaned("A", Q1, items, reference)

    stats = quarter_stats(lines, None, None, cfg).iloc[0]

    expected = {"pct_equity_us": 2000, "pct_equity_foreign": 1000, "pct_etf": 500, "pct_bond": 1000, "pct_options": 500, "pct_other": 500}
    for column, value in expected.items():
        assert np.isclose(stats[column], value / 5500), column
    assert np.isclose(stats[MIX].sum(), 1.0)
    assert stats["n_positions"] == 4 and np.isclose(stats["hhi"], (0.5**2) + 0.25**2 + 0.125**2 + 0.125**2), "equity positions only"
    print("\n=== SANITY CHECK: asset mix ===")
    print("Six mix shares equal value / 5,500 and sum to 1 including the bond and the option rows.")


def test_missing_evidence_reads_null_never_zero(reference: pd.DataFrame, cfg: DictConfig) -> None:
    lines = _cleaned("A", Q1, [("T1", "common", 10.0, 1000.0, "ONE INC")], reference)
    first = quarter_stats(lines, None, None, cfg).iloc[0]
    assert first[["turnover", "new_position_rate", "active_share", *CAPS, "mapped_share"]].isna().all(), "no preceding quarter, no aggregate"
    assert first["n_positions"] == 1 and np.isclose(first["hhi"], 1.0)

    bonds = _cleaned("B", Q1, [("B1", "debt", 0.0, 900.0, "A BOND")], reference)
    only_bonds = quarter_stats(bonds, None, None, cfg).iloc[0]
    assert only_bonds["n_positions"] == 0 and np.isclose(only_bonds["pct_bond"], 1.0)
    assert only_bonds[["hhi", *SECTORS, *CAPS]].isna().all(), "no equity: concentration, sector and cap are unknown, not zero"
    print("\n=== SANITY CHECK: nulls ===")
    print("A first quarter has null turnover/new rate/active share/caps and a bond-only book has null hhi/sector/cap; nothing is coerced to 0.")


def test_cap_buckets_follow_cumulative_value_of_the_previous_quarter(reference: pd.DataFrame, cfg: DictConfig) -> None:
    previous = _cleaned(
        "X",
        Q0,
        [("T1", "common", 1.0, 60.0, ""), ("T2", "common", 1.0, 25.0, ""), ("T3", "common", 1.0, 10.0, ""), ("T4", "common", 1.0, 5.0, "")],
        reference,
    )
    aggregate = quarter_aggregate(previous, cfg.cap_breakpoints)
    assert aggregate["cap"].to_dict() == {"T1": "large", "T2": "large", "T3": "mid", "T4": "small"}

    book = _cleaned("Y", Q1, [("T1", "common", 1.0, 50.0, ""), ("T3", "common", 1.0, 30.0, ""), ("T4", "common", 1.0, 20.0, "")], reference)
    stats = quarter_stats(book, None, aggregate, cfg).iloc[0]
    assert np.isclose(stats["pct_cap_large"], 0.5) and np.isclose(stats["pct_cap_mid"], 0.3) and np.isclose(stats["pct_cap_small"], 0.2)
    assert np.isclose(stats["mapped_share"], 1.0)
    print("\n=== SANITY CHECK: cap buckets ===")
    print("Top 70% of cumulative previous-quarter value is large, the next 20% mid, the rest small; the book's shares match.")


def test_active_share_is_zero_for_the_aggregate_book_and_one_for_a_disjoint_one(reference: pd.DataFrame, cfg: DictConfig) -> None:
    index_book = _cleaned("I", Q1, [("T1", "common", 1.0, 600.0, ""), ("T2", "common", 1.0, 300.0, ""), ("T4", "common", 1.0, 100.0, "")], reference)
    aggregate = quarter_aggregate(index_book, cfg.cap_breakpoints)

    same = quarter_stats(index_book, None, aggregate, cfg).iloc[0]
    other = _cleaned("J", Q1, [("T3", "common", 1.0, 100.0, "")], reference)
    disjoint = quarter_stats(other, None, aggregate, cfg).iloc[0]

    assert abs(same["active_share"]) < 1e-12 and np.isclose(disjoint["active_share"], 1.0)
    print("\n=== SANITY CHECK: active share ===")
    print("The aggregate book has active share 0 and a book with no overlap has 1.")


def test_profile_is_the_calendar_window_mean_with_minimum_quarters(cfg: DictConfig) -> None:
    periods = pd.to_datetime(["2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30", "2025-03-31"])
    frame = pd.DataFrame({column: np.nan for column in STAT_COLUMNS}, index=range(5))
    frame["cik"], frame["period"] = "X", periods
    frame["filed_date"] = periods + pd.Timedelta(days=40)
    frame["n_positions"] = [10.0, 20.0, 30.0, 40.0, 60.0]
    frame["turnover"] = [np.nan, 0.2, 0.4, 0.6, 0.8]
    frame["book_value_usd"] = [1.0, 2.0, 3.0, 4.0, 100.0]
    lone = frame.iloc[[0]].assign(cik="Z")

    profile = profile_rows(pd.concat([frame, lone], ignore_index=True), cfg)

    def pick(cik: str, period: pd.Timestamp) -> pd.Series:
        return profile[profile["cik"].eq(cik) & profile["period"].eq(period)].iloc[0]

    first = pick("X", periods[0])
    assert first["n_quarters_used"] == 1 and pd.isna(first["n_positions"]), "one quarter is below the minimum of 2"
    second = pick("X", periods[1])
    assert second["n_quarters_used"] == 2 and np.isclose(second["n_positions"], 15.0) and pd.isna(second["turnover"]), "one turnover value < minimum"
    last = pick("X", periods[4])
    assert last["n_quarters_used"] == 3 and np.isclose(last["n_positions"], (30 + 40 + 60) / 3), "window is the 4 quarter ends, 2024-12-31 absent"
    assert np.isclose(last["turnover"], 0.6) and last["book_value_usd"] == 4.0, "mean over non-null values; median book value"
    assert pick("Z", periods[0])["n_quarters_used"] == 1
    print("\n=== SANITY CHECK: profile window ===")
    print("Mean over the 4 calendar quarter ends ending at the period; fewer than 2 usable quarters read null; a missing quarter is not filled.")


def test_a_window_quarter_filed_after_the_row_is_left_out_of_its_mean(cfg: DictConfig) -> None:
    """A late filer's earlier quarter (first filed after the row's own book) does not enter the row's mean; the row's own quarter always does."""
    periods = pd.to_datetime(["2024-03-31", "2024-06-30", "2024-09-30"])
    frame = pd.DataFrame({column: np.nan for column in STAT_COLUMNS}, index=range(3))
    frame["cik"], frame["period"] = "X", periods
    frame["n_positions"] = [10.0, 20.0, 40.0]
    frame["book_value_usd"] = [1.0, 2.0, 3.0]
    # 2024-03-31 is filed in January 2025, after the 2024-06-30 and 2024-09-30 books (filed in August and November 2024).
    frame["filed_date"] = pd.to_datetime(["2025-01-15", "2024-08-10", "2024-11-10"])

    profile = profile_rows(frame, cfg).set_index("period")

    assert profile.loc[periods[1], "n_quarters_used"] == 1 and pd.isna(profile.loc[periods[1], "n_positions"]), "below min_quarters after the drop"
    last = profile.loc[periods[2]]
    assert profile["n_quarters_used"].loc[periods[2]] == 2 and np.isclose(profile["n_positions"].loc[periods[2]], 30.0), (
        "mean of 2024-06-30 and 2024-09-30 only"
    )
    assert np.isclose(last["book_value_usd"], 2.5), "median of the two kept quarters"
    print("\n=== SANITY CHECK: late window quarter ===")
    print(
        "The quarter filed after the row's own book is dropped from the 4-quarter mean; the current quarter stays; min_quarters applies after the drop."
    )


# --------------------------------------------------------------------------- real data


def _frame(df: pd.DataFrame | None) -> pd.DataFrame:
    assert df is not None
    return df


def live_store() -> DataStore:
    load_dotenv(find_dotenv(usecwd=True))  # a worktree has no .env of its own: use the repository one
    engine = get_engine()
    try:
        with engine.connect():
            pass
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"database unavailable: {type(exc).__name__}")
    return DataStore(engine)


def sample_ciks(store: DataStore, period: str, size: int = 100) -> list[str]:
    """The known large managers plus a deterministic sample of the quarter's filers (CIK list read from the manager table)."""
    df = _frame(store.load(Tables.sec_13f_manager_details, columns=["cik"], where={"period": period}))
    others = sorted(set(df["cik"]) - set(KNOWN_CIKS))
    step = max(len(others) // size, 1)
    return sorted({*KNOWN_CIKS, *others[::step]} & (set(df["cik"])))


def load_raw(store: DataStore, ciks: list[str], periods: list[str]) -> dict[str, pd.DataFrame]:
    """Markers and lines of the sample CIKs by period, read on the primary key (cik list plus period list)."""
    df = _frame(store.load(Tables.sec13f_hr, columns=list(LINE_COLUMNS), where={"cik": ciks, "period": periods}, markers=True))
    df["period"] = pd.to_datetime(df["period"])
    return {pd.Timestamp(str(period)).strftime("%Y-%m-%d"): frame for period, frame in df.groupby("period")}


def run_quarters(raw: dict[str, pd.DataFrame], reference: pd.DataFrame, prefixes: pd.Series, cfg: DictConfig) -> pd.DataFrame:
    """Single-quarter statistics over consecutive periods; the previous quarter's book and aggregate feed only the next quarter end."""
    out, state = [], None
    for period in sorted(raw):
        lines = classify_securities(clean_book(raw[period], cfg, prefixes), reference, cfg)
        stamp = pd.Timestamp(period)
        adjacent = state is not None and state[0] == stamp - pd.offsets.QuarterEnd(1)
        previous = state if state is not None and adjacent else None
        out.append(quarter_stats(lines, previous[1] if previous else None, previous[2] if previous else None, cfg))
        state = (stamp, compact_book(lines), quarter_aggregate(lines, cfg.cap_breakpoints))
    return pd.concat(out, ignore_index=True)


def live_prefixes(store: DataStore) -> pd.Series:
    return prefix_tickers(_frame(store.load(Tables.cusip_ticker_map)))


def live_reference(store: DataStore) -> pd.DataFrame:
    return sharadar_reference(_frame(store.load(Tables.sharadar_tickers, columns=["permaticker", "ticker", "category", "sector", "lastpricedate"])))


@pytest.mark.parametrize("period", REAL_QUARTERS)
def test_real_sample_quarter_is_within_definitions(period: str, cfg: DictConfig) -> None:
    store = live_store()
    previous = (pd.Timestamp(period) - pd.offsets.QuarterEnd(1)).strftime("%Y-%m-%d")
    raw = load_raw(store, sample_ciks(store, period), [previous, period])

    stats = run_quarters(raw, live_reference(store), live_prefixes(store), cfg)
    now = stats[stats["period"] == pd.Timestamp(period)].set_index("cik")

    assert len(now) >= 50 and now["book_value_usd"].gt(0).all()
    assert now["hhi"].dropna().between(0, 1).all() and now["turnover"].dropna().between(0, 1).all()
    assert now["new_position_rate"].dropna().between(0, 1).all() and now["active_share"].dropna().between(0, 1 + 1e-9).all()
    assert np.allclose(now[MIX].sum(axis=1), 1.0), "asset mix sums to 1 for every book"
    equity = now["n_positions"].gt(0)
    assert np.allclose(now.loc[equity, SECTORS].sum(axis=1), 1.0)
    assert (now.loc[equity, CAPS].sum(axis=1) <= 1 + 1e-9).all() and now["mapped_share"].dropna().between(0, 1).all()
    known = now.reindex([c for c in KNOWN_CIKS if c in now.index])
    print(f"\n=== SANITY CHECK: real sample {period} ({len(now)} managers) ===")
    print(
        known[["n_positions", "hhi", "turnover", "new_position_rate", "pct_equity_us", "active_share", "mapped_share", "book_value_usd"]]
        .round(4)
        .to_string()
    )
    print(f"Median hhi {now['hhi'].median():.4f}, median mapped share {now['mapped_share'].median():.2f}; all values inside their definitions.")
    if period == "2024-12-31":
        assert now["hhi"].to_dict()["0001067983"] > 5 * now["hhi"].to_dict()["0000102909"], "Berkshire is far more concentrated than Vanguard"
        assert now["active_share"].to_dict()["0000102909"] < 0.5, "Vanguard's 13F book sits close to the aggregate book"
        print("Berkshire reads a concentrated book, Vanguard an index-like one.")


def test_eight_character_cusips_map_through_the_issuer_prefix(cfg: DictConfig) -> None:
    store = live_store()
    raw = load_raw(store, ["0000093751"], ["2008-12-31"])["2008-12-31"]
    reference = live_reference(store)

    before = classify_securities(clean_book(raw, cfg, NO_PREFIX), reference, cfg)
    after = classify_securities(clean_book(raw, cfg, live_prefixes(store)), reference, cfg)
    mix = quarter_stats(after, None, None, cfg).iloc[0]

    equity = after[after["kind"].eq("common")]
    assert before["ticker"].notna().sum() == 0, "State Street's 2008 CUSIPs have 8 characters and never match the 9-character map"
    assert equity["ticker"].notna().mean() > 0.2 and mix["pct_equity_us"] > 0.2
    assert after["key"].is_unique
    print("\n=== SANITY CHECK: CUSIP prefix-8 mapping, State Street 2008Q4 ===")
    print(f"Tickers mapped for {equity['ticker'].notna().mean():.0%} of equity lines, pct_equity_us 0 -> {mix['pct_equity_us']:.2f}.")


@pytest.mark.parametrize(
    ("cik", "period", "name"),
    [("0000728100", "2012-09-30", "Lord Abbett"), ("0000102909", "2008-03-31", "Vanguard")],
)
def test_a_garbage_share_count_cannot_make_a_line_larger_than_its_book(cik: str, period: str, name: str, cfg: DictConfig) -> None:
    store = live_store()
    ciks = _frame(store.load(Tables.sec_13f_manager_details, columns=["cik"], where={"period": period}))["cik"].tolist()
    raw = pd.concat([load_raw(store, ciks[i : i + 1000], [period])[period] for i in range(0, len(ciks), 1000)], ignore_index=True)

    quarter = clean_book(raw, cfg, NO_PREFIX)
    alone = clean_book(raw[raw["cik"] == cik], cfg, NO_PREFIX)

    mine = quarter[quarter["cik"].eq(cik) & quarter["kind"].eq("common")]
    equity_weight = mine["value"] / mine["value"].sum()
    hhi = float((equity_weight**2).sum())
    assert len(mine) >= 500
    assert equity_weight.max() < 0.2 and hhi < 0.05, "no single line is a fifth of a 500-name book"
    assert quarter.loc[quarter["cik"].eq(cik), "value"].sum() <= 2 * alone["value"].sum(), "the cleaned book stays near its filed total"
    print(f"\n=== SANITY CHECK: oversized re-valuation, {name} {period} ===")
    print(
        f"{len(mine)} positions, max weight {equity_weight.max():.3f}, hhi {hhi:.4f}, cleaned {quarter.loc[quarter['cik'].eq(cik), 'value'].sum():.3e} vs filed {alone['value'].sum():.3e}."
    )
