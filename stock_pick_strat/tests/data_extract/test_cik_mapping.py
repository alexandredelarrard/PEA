"""
Ticker -> CIK resolution. `sp500_tickers` is the single source of truth (it carries
`cik` + `name` + GICS), so `load_cik_mapping` reads it directly; the redundant,
SEC-sourced `cik_mapping` table (whose company_tickers.json feed mismapped active
tickers like XOM to a non-filing shell) was retired.

test_load_cik_mapping_reads_sp500_tickers — reads sp500_tickers, zero-pads the CIK,
    aliases company_name from name, and preserves the GICS columns callers rely on.
"""
from __future__ import annotations

import pandas as pd

from src.data_extract.utils.common import sec_utils


def test_load_cik_mapping_reads_sp500_tickers():
    sp500 = pd.DataFrame([
        {"ticker": "XOM", "name": "ExxonMobil", "cik": "0000034088",
         "sector": "Energy", "industry_group": "Energy", "sub_industry": "Integrated Oil & Gas"},
        {"ticker": "AAPL", "name": "Apple", "cik": "320193",        # deliberately not zero-padded
         "sector": "Information Technology", "industry_group": "Technology Hardware & Equipment",
         "sub_industry": "Technology Hardware, Storage & Peripherals"},
    ])

    class _Store:
        def load(self, name, columns=None, limit=None):
            assert name == "sp500_tickers"          # no longer reads a cik_mapping table
            return sp500.copy()

    class _Ctx:
        store = _Store()

    m = sec_utils.load_cik_mapping(_Ctx())
    d = {r["ticker"]: r for _, r in m.iterrows()}

    # CIK zero-padded to 10 digits for SEC URLs (even when stored short)
    assert d["AAPL"]["cik"] == "0000320193"
    assert d["XOM"]["cik"] == "0000034088"
    # company_name aliased from name (callers do r.get("company_name")); GICS preserved
    assert {"company_name", "sector", "industry_group", "sub_industry"} <= set(m.columns)
    assert d["XOM"]["company_name"] == "ExxonMobil"
    assert d["XOM"]["sector"] == "Energy"

    print("\n=== SANITY CHECK: load_cik_mapping from sp500_tickers ===")
    print(f"  reads sp500_tickers only; CIK zero-padded (AAPL {d['AAPL']['cik']}); "
          f"company_name aliased from name; GICS preserved. Redundant cik_mapping retired. Validated.")
