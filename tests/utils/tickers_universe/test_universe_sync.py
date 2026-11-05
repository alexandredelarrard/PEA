"""`sync` builds `universe_tickers` from scratch on live-exported fixtures and keeps it current.

Fixtures (read-only exports): `roster.csv` is the pre-refactor roster table (500 rows, 2026-11-02),
`sharadar_tickers.csv` the Sharadar rows of the 491 universe tickers and their alias symbols (2026-11-02),
`sec_company_tickers.csv` the SEC snapshot rows of the same symbols (2026-10-23).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data_store.schema import Tables
from src.utils.config import read_config
from src.utils.tickers_universe.sync import sync
from src.utils.tickers_universe.universe import Universe, UniverseError
from tests.support.universe import universe_config

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "universe"
AS_OF = pd.Timestamp("2026-11-02")


@pytest.fixture
def roster() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "roster.csv", dtype=str)


@pytest.fixture
def sec() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "sec_company_tickers.csv", dtype=str, parse_dates=["fetched_at"])


@pytest.fixture
def config():
    return read_config(path=str(Path(__file__).resolve().parents[3] / "configs"))


@pytest.fixture
def store(sqlite_store, roster, sec):
    sharadar = pd.read_csv(FIXTURES / "sharadar_tickers.csv", dtype=str).assign(table="fundamentals")
    sharadar["permaticker"] = sharadar["permaticker"].astype("int64")
    sharadar["lastupdated"] = pd.to_datetime(sharadar["lastupdated"])
    sqlite_store.save(Tables.sharadar_tickers, sharadar)
    sqlite_store.save(Tables.sec_company_tickers, sec)
    sqlite_store.save(Tables.prices, pd.DataFrame({"ticker": roster["ticker"], "date": pd.Timestamp("2026-10-30"), "close": 1.0}))
    return sqlite_store


def _entries(config, tickers: list[str]) -> dict[str, dict[str, str]]:
    return {t: dict(config.universe.tickers[t]) for t in tickers}


def test_first_build_takes_sec_identity_and_the_yml_gics(store, roster, sec, config) -> None:
    written = sync(store, config, AS_OF)
    universe = Universe(store, config)
    built = universe.frame.set_index("ticker")
    expected = roster.set_index("ticker").loc[universe.tickers]
    gics = ["sector", "industry_group", "sub_industry"]
    diff = (built[gics] != expected[gics]).any(axis=1)

    assert written == len(universe.tickers) == 491
    assert not diff.any(), built[gics][diff]
    assert built["cik"].isna().all(), "U-49: the CIK is an identity column, the sync never writes it"
    assert built["name"].tolist() == sec.set_index("ticker").loc[universe.tickers, "name"].tolist()
    assert set(built["universe_added_on"].astype(str)) == {"2000-01-01"}
    assert built.loc["GOOGL", "cusips"] == "02079K305 38259P508"
    assert built.loc["BRK-B", "cusips"] == "084670207 084670702"
    assert built["cusips"].notna().all()
    print("\n=== SANITY CHECK: first build ===")
    print(f"  {written} rows; the 3 GICS columns (yml) equal the pre-refactor roster for all 491; names are SEC's; CIK left NULL (U-49);")
    print(f"  every row has CUSIPs; {built['sector'].nunique()} sectors / {built['industry_group'].nunique()} groups.")


def test_gics_codes_equal_the_pre_change_codes(store, config) -> None:
    sync(store, config, AS_OF)
    universe = Universe(store, config)
    baseline = json.loads((FIXTURES / "baseline_gics.json").read_text())
    coded = universe.gics_codes(pd.DataFrame({"ticker": universe.tickers}))

    for column in ("sector", "industry_group"):
        expected = [baseline[f"{column}_codes"][baseline[column][t]] for t in universe.tickers]
        assert coded[column].tolist() == expected
        assert getattr(universe, f"{column}_map") == baseline[column]
    print("\n=== SANITY CHECK: GICS equivalence ===")
    print(f"  sector and industry_group maps and int codes equal the pre-change baseline on {len(universe.tickers)} names.")


def test_second_sync_writes_nothing_and_keeps_the_table(store, config) -> None:
    sync(store, config, AS_OF)
    first = Universe(store, config).frame
    written = sync(store, config, AS_OF + pd.Timedelta(days=1))
    second = Universe(store, config).frame

    assert written == 0
    pd.testing.assert_frame_equal(first, second)
    print("\n=== SANITY CHECK: idempotent ===")
    print("  a second sync the next day writes 0 rows; the frame (incl. universe_added_on) is identical.")


def test_a_sharadar_change_rewrites_only_that_row_and_never_the_carried_columns(store, sec, config) -> None:
    sync(store, config, AS_OF)
    store.save(
        Tables.sharadar_tickers,
        pd.DataFrame({"table": ["fundamentals"], "permaticker": [199059], "ticker": ["AAPL"], "category": ["X"], "name": ["NEW"]}),
    )
    written = sync(store, config, AS_OF)
    row = Universe(store, config).frame.set_index("ticker").loc["AAPL"]

    assert written == 1
    assert row["category"] == "X" and row["sharadar_name"] == "NEW" and row["name"] == sec.set_index("ticker").loc["AAPL", "name"]
    print("\n=== SANITY CHECK: nightly refresh ===")
    print("  one Sharadar change -> 1 row written; the carried name stays the SEC one.")


def test_a_gics_edit_in_the_yml_rewrites_only_that_row(store, config) -> None:
    sync(store, config, AS_OF)
    entries = _entries(config, list(config.universe.tickers))
    entries["AAPL"] = {"sector": "Information Technology", "industry": "Semiconductors"}
    written = sync(store, universe_config(entries, config.universe.aliases), AS_OF)
    row = Universe(store, config).frame.set_index("ticker").loc["AAPL"]

    assert written == 1
    assert (row["sub_industry"], row["industry_group"]) == ("Semiconductors", "Semiconductors & Semiconductor Equipment")
    print("\n=== SANITY CHECK: yml GICS ===")
    print("  AAPL's yml industry edited -> 1 row written, its industry group derived from the vocabulary.")


def test_an_invalid_gics_entry_aborts_naming_every_ticker(store, config) -> None:
    entries = _entries(config, ["AAPL", "MSFT", "JPM"])
    entries["AAPL"] = {"sector": "Financials", "industry": "Semiconductors"}
    entries["MSFT"] = {"sector": "Information Technology", "industry": "Fintech"}
    with pytest.raises(UniverseError, match=r"AAPL .*MSFT"):
        sync(store, universe_config(entries), AS_OF)
    assert not store.exists(Tables.universe_tickers)
    print("\n=== SANITY CHECK: invalid GICS ===")
    print("  a sub-industry outside its sector (AAPL) and an unknown one (MSFT) -> one UniverseError naming both, nothing written.")


def test_entering_and_leaving_tickers(store, config) -> None:
    sync(store, config, AS_OF)
    tickers = [t for t in config.universe.tickers if t != "ZTS"]
    sync(store, universe_config(_entries(config, tickers), config.universe.aliases), AS_OF)
    assert store.row_count(Tables.universe_tickers) == 491
    store.delete(Tables.prices, {"ticker": ["ZTS"]})
    sync(store, config, AS_OF + pd.Timedelta(days=3))
    universe = Universe(store, config)
    added = universe.frame.set_index("ticker").loc["ZTS", "universe_added_on"]

    assert pd.Timestamp(added) == AS_OF + pd.Timedelta(days=3)
    assert universe.new_since(7, AS_OF + pd.Timedelta(days=4)) == {"ZTS"}
    assert universe.new_since(7, AS_OF + pd.Timedelta(days=20)) == set()
    print("\n=== SANITY CHECK: entering ticker ===")
    print("  ZTS removed, then re-added with no prices: stamped with the sync date and the only new ticker in the overlap.")


def test_a_ticker_removed_from_the_yml_stays_with_in_universe_false_and_flips_back(store, config) -> None:
    sync(store, config, AS_OF)
    before = store.load(Tables.universe_tickers).set_index("ticker").loc["ZTS"]
    without = universe_config(_entries(config, [t for t in config.universe.tickers if t != "ZTS"]), config.universe.aliases)
    written = sync(store, without, AS_OF + pd.Timedelta(days=1))
    left = store.load(Tables.universe_tickers).set_index("ticker")
    universe = Universe(store, without)

    assert written == 1 and len(left) == 491 and not bool(left.loc["ZTS", "in_universe"])
    assert left.loc["ZTS", ["name", "cik", "sector", "sub_industry"]].tolist() == before[["name", "cik", "sector", "sub_industry"]].tolist()
    assert left.drop(index="ZTS")["in_universe"].astype(bool).all()
    assert "ZTS" not in universe.tickers and "ZTS" not in set(universe.frame["ticker"]) and len(universe.frame) == 490
    assert sync(store, without, AS_OF + pd.Timedelta(days=2)) == 0

    assert sync(store, config, AS_OF + pd.Timedelta(days=3)) == 1
    back = Universe(store, config)
    assert "ZTS" in back.tickers and len(back.frame) == 491
    print("\n=== SANITY CHECK: removed ticker kept ===")
    print("  ZTS left the yml -> its row stays (name, CIK, GICS kept) with in_universe=false, 1 row written, the reader lists 490;")
    print("  a second night writes 0; ZTS re-added -> in_universe=true again, 1 row written, the reader lists 491.")


def test_identity_owned_columns_survive_a_sync_and_the_reader_returns_bounds(store, config) -> None:
    sync(store, config, AS_OF)
    identity = {
        "ticker": ["WBD", "PSKY"],
        "cik": ["0001437107", "0002041610"],
        "company_id": ["E0001437107", "E0000813828"],
        "trading_symbol": ["WBD", "SKYD"],
        "first_price_date": [pd.Timestamp("2005-07-08"), pd.Timestamp("1986-01-02")],
        "ended_on": [pd.Timestamp("2026-10-05"), None],
        "status": ["ended", "renamed"],
    }
    store.save(Tables.universe_tickers, pd.DataFrame(identity))
    store.save(Tables.sharadar_tickers, pd.DataFrame({"table": ["fundamentals"], "permaticker": [193488], "ticker": ["WBD"], "category": ["X"]}))
    written = sync(store, config, AS_OF + pd.Timedelta(days=1))
    universe = Universe(store, config)
    rows = universe.frame.set_index("ticker")

    assert written == 1 and rows.loc["WBD", "category"] == "X"
    for column in ("cik", "company_id", "trading_symbol", "status"):
        assert rows.loc[["WBD", "PSKY"], column].tolist() == identity[column], column
    assert universe.bounds["WBD"] == (pd.Timestamp("2005-07-08"), pd.Timestamp("2026-10-05"))
    assert universe.bounds["PSKY"] == (pd.Timestamp("1986-01-02"), None)
    assert universe.bounds["AAPL"] == (None, None) and set(universe.bounds) == set(universe.tickers)
    print("\n=== SANITY CHECK: two writers ===")
    print("  identity wrote cik, company_id, trading_symbol, bounds and status on WBD/PSKY; a sync rewriting WBD's Sharadar row keeps them all;")
    print("  bounds: WBD 2005-07-08..2026-10-05, PSKY open, AAPL (no identity yet) unbounded; one entry per universe ticker.")


def test_a_ticker_without_a_sharadar_row_aborts(store) -> None:
    gics = {"sector": "Information Technology", "industry": "Application Software"}
    with pytest.raises(UniverseError, match="FOO"):
        sync(store, universe_config({"AAPL": gics, "FOO": gics}), AS_OF)
    assert not store.exists(Tables.universe_tickers)
    print("\n=== SANITY CHECK: unmatched ticker ===")
    print("  FOO has no Sharadar row -> UniverseError naming it, nothing written.")


def test_alias_cusips_join_the_union_and_a_listed_row_beats_a_delisted_one(store, config) -> None:
    extra = pd.DataFrame(
        {
            "table": ["fundamentals", "fundamentals"],
            "permaticker": [900001, 900002],
            "ticker": ["GOOG", "AAPL"],
            "name": ["ALPHABET INC CLASS C", "OLD APPLE"],
            "isdelisted": ["N", "Y"],
            "cusips": ["99999X999", "11111X111"],
            "lastupdated": pd.to_datetime(["2026-08-01", "2026-10-01"]),
        }
    )
    store.save(Tables.sharadar_tickers, extra)
    sync(store, config, AS_OF)
    built = Universe(store, config).frame.set_index("ticker")

    assert built.loc["GOOGL", "cusips"] == "02079K305 38259P508 99999X999"
    assert built.loc["AAPL", "sharadar_name"] == "APPLE INC" and built.loc["AAPL", "cusips"] == "037833100"
    print("\n=== SANITY CHECK: Sharadar row choice ===")
    print("  GOOG's own CUSIP joins GOOGL's union; AAPL keeps its listed row over a newer delisted one.")


def test_a_ticker_the_sec_no_longer_lists_takes_sharadars_name(store, config) -> None:
    store.delete(Tables.sec_company_tickers, {"ticker": ["AAPL"]})
    sync(store, config, AS_OF)
    built = Universe(store, config).frame.set_index("ticker")

    assert built.loc["AAPL", "name"] == "APPLE INC" and pd.isna(built.loc["AAPL", "cik"])
    print("\n=== SANITY CHECK: SEC-unlisted ticker ===")
    print("  AAPL missing from the SEC snapshot takes Sharadar's name; its CIK is left to identity (U-49).")
