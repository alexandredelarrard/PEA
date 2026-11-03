"""`sync` builds `tickers_universe` from scratch on live-exported fixtures and keeps it current.

Fixtures (read-only exports of 2026-11-02): `roster.csv` is `sp500_tickers` (500 rows),
`sharadar_tickers.csv` the Sharadar rows of the 491 universe tickers and their alias symbols.
The Wikipedia page is rendered from the roster, so the parser, the `.`->`-` spelling, the CIK
padding and the sub-industry dictionary are checked against the stored values.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data_store.schema import Tables
from src.utils.config import read_config
from src.utils.tickers_universe import sync as sync_module
from src.utils.tickers_universe.sync import sync
from src.utils.tickers_universe.universe import Universe, UniverseError
from src.utils.tickers_universe.vocabulary import GICS_INDUSTRY_GROUPS, GICS_SECTORS
from src.utils.tickers_universe.wikipedia_bootstrap import parse_sp500_table
from tests.support.universe import universe_config

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "universe"
AS_OF = pd.Timestamp("2026-11-02")


@pytest.fixture
def roster() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "roster.csv", dtype=str)


@pytest.fixture
def config():
    return read_config(path=str(Path(__file__).resolve().parents[3] / "configs"))


@pytest.fixture
def store(sqlite_store, roster, monkeypatch):
    sharadar = pd.read_csv(FIXTURES / "sharadar_tickers.csv", dtype=str).assign(table="fundamentals")
    sharadar["permaticker"] = sharadar["permaticker"].astype("int64")
    sharadar["lastupdated"] = pd.to_datetime(sharadar["lastupdated"])
    sqlite_store.save(Tables.sharadar_tickers, sharadar)
    sqlite_store.save(Tables.prices, pd.DataFrame({"ticker": roster["ticker"], "date": pd.Timestamp("2026-10-30"), "close": 1.0}))
    page = roster.assign(Symbol=roster["ticker"].str.replace("-", "."), CIK=roster["cik"].astype(int)).rename(
        columns={"name": "Security", "sector": "GICS Sector", "sub_industry": "GICS Sub-Industry"}
    )
    html = page[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry", "CIK"]].to_html(index=False)
    monkeypatch.setattr(sync_module, "wikipedia_sp500", lambda: parse_sp500_table(html))
    return sqlite_store


def test_first_build_matches_the_pre_change_roster(store, roster, config) -> None:
    written = sync(store, config, AS_OF)
    universe = Universe(store, config)
    built = universe.frame.set_index("ticker")
    expected = roster.set_index("ticker").loc[universe.tickers]
    columns = ["name", "cik", "sector", "industry_group", "sub_industry"]
    diff = (built[columns] != expected[columns]).any(axis=1)

    assert written == len(universe.tickers) == 491
    assert not diff.any(), built[columns][diff]
    assert set(built["universe_added_on"].astype(str)) == {"2000-01-01"}
    assert built.loc["GOOGL", "cusips"] == "02079K305 38259P508"
    assert built.loc["BRK-B", "cusips"] == "084670207 084670702"
    assert built["cusips"].notna().all()
    assert set(built["sector"]) <= GICS_SECTORS and set(built["industry_group"]) <= GICS_INDUSTRY_GROUPS
    print("\n=== SANITY CHECK: first build ===")
    print(f"  {written} rows; name/cik/3 GICS columns equal the roster for all 491; every row has CUSIPs;")
    print(f"  {built['sector'].nunique()} sectors / {built['industry_group'].nunique()} groups, all in the vocabulary.")


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


def test_a_sharadar_change_rewrites_only_that_row_and_never_the_carried_columns(store, config, monkeypatch) -> None:
    sync(store, config, AS_OF)
    store.save(Tables.sharadar_tickers, pd.DataFrame({"table": ["fundamentals"], "permaticker": [199059], "ticker": ["AAPL"], "category": ["X"], "name": ["NEW"]}))
    written = sync(store, config, AS_OF)
    row = Universe(store, config).frame.set_index("ticker").loc["AAPL"]

    assert written == 1
    assert row["category"] == "X" and row["sharadar_name"] == "NEW" and row["name"] == "Apple Inc."
    print("\n=== SANITY CHECK: nightly refresh ===")
    print("  one Sharadar change -> 1 row written; the carried name stays the roster's.")


def test_entering_and_leaving_tickers(store, config) -> None:
    sync(store, config, AS_OF)
    tickers = [t for t in config.universe.tickers if t != "ZTS"]
    small = universe_config(tickers, config.universe.aliases)
    sync(store, small, AS_OF)
    store.delete(Tables.prices, {"ticker": ["ZTS"]})
    sync(store, config, AS_OF + pd.Timedelta(days=3))
    universe = Universe(store, config)
    added = universe.frame.set_index("ticker").loc["ZTS", "universe_added_on"]

    assert pd.Timestamp(added) == AS_OF + pd.Timedelta(days=3)
    assert universe.new_since(7, AS_OF + pd.Timedelta(days=4)) == {"ZTS"}
    assert universe.new_since(7, AS_OF + pd.Timedelta(days=20)) == set()
    print("\n=== SANITY CHECK: entering ticker ===")
    print("  ZTS removed, then re-added with no prices: stamped with the sync date and the only new ticker in the overlap.")


def test_a_ticker_without_a_sharadar_row_aborts(store) -> None:
    with pytest.raises(UniverseError, match="FOO"):
        sync(store, universe_config(["AAPL", "FOO"]), AS_OF)
    assert not store.exists(Tables.tickers_universe)
    print("\n=== SANITY CHECK: unmatched ticker ===")
    print("  FOO has no Sharadar row -> UniverseError naming it, nothing written.")
