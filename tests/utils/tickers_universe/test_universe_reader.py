"""`Universe` validates `universe.yml`, refuses a table out of step with it and checks CLI subsets."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_store.schema import Tables
from src.utils.tickers_universe.universe import Universe, UniverseError, universe_config
from tests.support.universe import seed_universe
from tests.support.universe import universe_config as make_config


@pytest.mark.parametrize(
    ("tickers", "aliases", "message"),
    [
        ([], {}, "lists no ticker"),
        (["AAPL", "aapl ", "MSFT"], {}, r"twice: \['AAPL'\]"),
        (["AAPL"], {"GOOGL": ["GOOG"]}, r"not in its tickers: \['GOOGL'\]"),
    ],
)
def test_invalid_yml_is_refused(tickers, aliases, message) -> None:
    with pytest.raises(UniverseError, match=message):
        universe_config(make_config(tickers, aliases))
    print(f"\n=== SANITY CHECK: yml validation === {message!r} raised")


def test_missing_or_stale_table_raises_and_writes_nothing(sqlite_store) -> None:
    config = make_config(["AAPL", "MSFT"])
    with pytest.raises(UniverseError, match="data_extract universe"):
        Universe(sqlite_store, config)
    seed_universe(sqlite_store, ["AAPL"])
    with pytest.raises(UniverseError, match=r"\['MSFT'\]"):
        Universe(sqlite_store, config)
    assert sqlite_store.row_count(Tables.tickers_universe) == 1
    print("\n=== SANITY CHECK: reader guards === missing and stale tables raise with the sync command; nothing written.")


def test_subset_normalises_dedupes_and_lists_every_unknown(sqlite_store) -> None:
    universe = seed_universe(sqlite_store, ["AAPL", "GOOGL", "MSFT"], aliases={"GOOGL": ["GOOG"]})

    assert universe.subset([" msft", "AAPL", "msft"]) == ["MSFT", "AAPL"]
    with pytest.raises(UniverseError, match=r"GOOG \(alias of GOOGL\), FOO"):
        universe.subset(["AAPL", "GOOG", "FOO"])
    assert universe.alias_symbols == frozenset({"GOOG"})
    assert universe.outside(["AAPL"]) == frozenset({"GOOGL", "MSFT"}) and universe.is_full(["MSFT", "GOOGL", "AAPL"])
    print("\n=== SANITY CHECK: subset === order kept, duplicates dropped, unknowns listed once with the alias owner.")


def test_gics_accessors_and_added_on(sqlite_store) -> None:
    universe = seed_universe(
        sqlite_store,
        ["AAPL", "XOM", "NEW"],
        rows={
            "AAPL": {"sector": "Information Technology", "industry_group": "Technology Hardware & Equipment"},
            "XOM": {"sector": "Energy", "industry_group": "Energy"},
            "NEW": {"universe_added_on": pd.Timestamp("2026-11-01")},
        },
    )
    frame = pd.DataFrame({"ticker": ["XOM", "AAPL", "NEW", "ZZZ"]})

    assert universe.attach_gics(frame)["sector"].tolist()[:2] == ["Energy", "Information Technology"]
    assert universe.gics_codes(frame)["sector"].tolist() == [0, 1, -1, -1] and list(frame.columns) == ["ticker"]
    assert universe.new_since(7, pd.Timestamp("2026-11-02")) == {"NEW"}
    assert set(universe.added_on_dates(["NEW", "AAPL"], pd.Timestamp("2026-10-01"))) == {"AAPL"}
    print("\n=== SANITY CHECK: GICS and dates === names attached, sorted codes with -1 for unknown (input untouched), NEW is the only new ticker.")


def test_class_separators_resolve_to_the_yfinance_spelling(sqlite_store) -> None:
    assert universe_config(make_config(["BRK.B", "aapl"], {"BRK.B": ["BRK.A"]})) == (["AAPL", "BRK-B"], {"BRK-B": ["BRK-A"]})
    universe = seed_universe(sqlite_store, ["AAPL", "BRK-B"], aliases={"BRK-B": ["BRK-A"]})

    assert universe.subset(["brk.b", "BRK/B", "AAPL"]) == ["BRK-B", "AAPL"]
    assert universe.roster["ticker"].tolist() == ["AAPL", "BRK-B"]
    with pytest.raises(UniverseError, match=r"BRK-A \(alias of BRK-B\)"):
        universe.subset(["BRK.A"])
    print("\n=== SANITY CHECK: class separators === BRK.B and BRK/B resolve to BRK-B; -t BRK.A is named as the alias of BRK-B.")
