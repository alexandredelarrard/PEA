"""
Ticker -> CIK resolution. `sp500_tickers` is the single source of truth (it carries
`cik` + `name` + GICS), so `load_cik_mapping` reads it directly; the redundant,
SEC-sourced `cik_mapping` table (whose company_tickers.json feed mismapped active
tickers like XOM to a non-filing shell) was retired.

test_load_cik_mapping_reads_sp500_tickers — reads sp500_tickers, zero-pads the CIK,
    and preserves the name and GICS columns callers rely on.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd

from src.data_extract.utils.common import sec_utils


def test_load_cik_mapping_reads_sp500_tickers(sqlite_store):
    sp500 = pd.DataFrame(
        [
            {
                "ticker": "XOM",
                "name": "ExxonMobil",
                "cik": "0000034088",
                "sector": "Energy",
                "industry_group": "Energy",
                "sub_industry": "Integrated Oil & Gas",
            },
            {
                "ticker": "AAPL",
                "name": "Apple",
                "cik": "320193",  # deliberately not zero-padded
                "sector": "Information Technology",
                "industry_group": "Technology Hardware & Equipment",
                "sub_industry": "Technology Hardware, Storage & Peripherals",
            },
        ]
    )

    sqlite_store.replace("sp500_tickers", sp500)  # the ONLY table it may read (no cik_mapping)
    context: Any = SimpleNamespace(store=sqlite_store)
    m = sec_utils.load_cik_mapping(context)
    d = {r["ticker"]: r for _, r in m.iterrows()}

    # CIK zero-padded to 10 digits for SEC URLs (even when stored short)
    assert d["AAPL"]["cik"] == "0000320193"
    assert d["XOM"]["cik"] == "0000034088"
    # name and GICS preserved (callers do r.get("name"))
    assert {"name", "sector", "industry_group", "sub_industry"} <= set(m.columns)
    assert d["XOM"]["name"] == "ExxonMobil"
    assert d["XOM"]["sector"] == "Energy"

    print("\n=== SANITY CHECK: load_cik_mapping from sp500_tickers ===")
    print(
        f"  reads sp500_tickers only; CIK zero-padded (AAPL {d['AAPL']['cik']}); name and GICS preserved. Redundant cik_mapping retired. Validated."
    )
