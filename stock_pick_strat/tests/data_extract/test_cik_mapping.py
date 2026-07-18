"""
CIK resolution corrections. SEC's company_tickers.json sometimes maps an ACTIVE,
currently-trading ticker to a newly-registered non-filing holding shell (empty
companyfacts) instead of the operating filer — build_cik_mapping pins those to the
operating CIK. Pre-spin-off shells (FedEx Freight, Honeywell Aerospace) are LEFT in:
their mapping is correct, they simply have no filings yet, and the builder skips
empty companyfacts gracefully so they auto-populate the moment they file.

test_cik_mapping_pins_active_ticker_keeps_shells — XOM pinned to 34088; FDXF/HONA
    kept with their real CIKs; other tickers untouched.
"""
from __future__ import annotations

from src.data_extract.utils.common import sec_utils


def test_cik_mapping_pins_active_ticker_keeps_shells(monkeypatch):
    # SEC file as it currently is: XOM -> holdco shell (active ticker, wrong),
    # FDXF/HONA -> pre-spin shells (correct entity, no filings yet), HON/NVDA fine.
    fake = {
        "0": {"cik_str": 2115436, "ticker": "XOM", "title": "ExxonMobil Holdings Corp"},
        "1": {"cik_str": 2082247, "ticker": "FDXF", "title": "FedEx Freight Holding Company, Inc."},
        "2": {"cik_str": 2089271, "ticker": "HONA", "title": "Honeywell Aerospace Inc."},
        "3": {"cik_str": 773840, "ticker": "HON", "title": "HONEYWELL INTERNATIONAL INC"},
        "4": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
    }

    class _Resp:
        def json(self):
            return fake

    saved = {}

    class _Store:
        def save(self, name, df, pk=None):
            saved["name"], saved["df"] = name, df
            return len(df)

    class _Ctx:
        store = _Store()

    monkeypatch.setattr(sec_utils, "sec_get", lambda url: _Resp())
    out = sec_utils.build_cik_mapping(_Ctx(), ["XOM", "FDXF", "HONA", "HON", "NVDA"])
    d = dict(zip(out["ticker"], out["cik"]))

    # active ticker mismapped to a non-filing shell -> pinned to the operating filer
    assert d["XOM"] == "0000034088", "XOM must be pinned to the operating filer 34088"
    # correctly-mapped pre-spin shells are KEPT (no hardcoded drop-list landmine)
    assert d["FDXF"] == "0002082247" and d["HONA"] == "0002089271", \
        "pre-spin shells must be kept so they auto-populate once they file"
    # everything else passes through, zero-padded to 10 digits
    assert d["HON"] == "0000773840" and d["NVDA"] == "0001045810"
    assert saved["name"] == "cik_mapping"

    print("\n=== SANITY CHECK: cik_mapping active-ticker override ===")
    print(f"  XOM pinned -> {d['XOM']} (was 0002115436 shell); pre-spin FDXF/HONA kept "
          f"({d['FDXF']}/{d['HONA']}) to auto-populate on first filing; HON/NVDA unchanged.")
    print("  -> only the ACTIVE mismap is corrected; empty-but-correct shells are not dropped. Validated.")
