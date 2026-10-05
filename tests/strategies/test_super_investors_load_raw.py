"""
`SuperInvestorsStrategy.load_raw` (src/strategies/step_super_investors.py).

The replication sleeve reads the roster managers' raw `sec13f_hr` rows. A manager whose 13F filer
CIK changed (Appaloosa: Management LP to 2015-12-31, Appaloosa LP from 2016-03-31) must come back
as ONE manager: both filers are read, each only inside its window, under the manager ID.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd
from omegaconf import OmegaConf

from src.data_store.schema import Tables
from src.strategies.step_super_investors import SuperInvestorsStrategy
from tests.fixtures.superinvestor_config import APPALOOSA_CHAIN, APPALOOSA_NEW, APPALOOSA_OLD, write_roster_config

_AM_OLD, _AM_NEW, _BRK = APPALOOSA_OLD, APPALOOSA_NEW, "0001067983"


def _hr(cik: str, period: date, ticker: str, value: float) -> dict:
    return {"cik": cik, "period": period, "ticker": ticker, "cusip": f"C{ticker}", "shares": 10.0, "value_usd": value, "filing_date": period}


def test_load_raw_relabels_chain(sqlite_store, tmp_path):
    roster = [
        {"snapshot_date": date(2016, 8, 1), "dataroma_code": code, "manager_name": name, "cik": cik, "resolution": "override", "source_url": "x"}
        for code, name, cik in [("AM", "David Tepper - Appaloosa", _AM_NEW), ("BRK", "Berkshire Hathaway", _BRK)]
    ]
    sqlite_store.replace(Tables.superinvestor_roster.name, pd.DataFrame(roster))
    hr = [
        _hr(_AM_OLD, date(2015, 12, 31), "AAA", 1.0),
        _hr(_AM_OLD, date(2016, 3, 31), "AAA", 99.0),  # the predecessor outside its window
        _hr(_AM_NEW, date(2016, 3, 31), "AAA", 2.0),
        _hr(_AM_NEW, date(2016, 6, 30), "AAA", 3.0),
        _hr(_BRK, date(2016, 6, 30), "BBB", 4.0),
        _hr("0000000009", date(2016, 6, 30), "AAA", 5.0),  # not on the roster
    ]
    sqlite_store.save(Tables.sec13f_hr, pd.DataFrame(hr))
    sqlite_store.save(Tables.prices, pd.DataFrame({"date": [date(2016, 7, 1)], "ticker": ["AAA"], "close_split": [10.0]}))
    ctx = SimpleNamespace(store=sqlite_store, config_dir=write_roster_config(tmp_path, {"manager_ciks": APPALOOSA_CHAIN}))

    funds, _prices, end = SuperInvestorsStrategy(context=ctx, config=OmegaConf.create({})).load_raw()  # type: ignore[arg-type]

    assert set(funds["cik"]) == {_AM_OLD, _BRK}
    am = funds[funds["cik"] == _AM_OLD].assign(period=lambda d: pd.to_datetime(d["period"])).sort_values("period")
    assert am["value_usd"].tolist() == [1.0, 2.0, 3.0]
    assert not am["period"].duplicated().any()
    assert end == pd.Timestamp("2016-07-01")
    print("\n=== SANITY: replication sleeve reads a succession as one manager ===")
    print(
        f"  roster stores Appaloosa LP ({_AM_NEW}); load_raw read both filers and returned {len(am)} Appaloosa quarters "
        f"under {_AM_OLD} (predecessor to 2015-12-31, successor from 2016-03-31, the overlapping predecessor row dropped); "
        "the non-roster filer is not read. Validated on the real store."
    )
