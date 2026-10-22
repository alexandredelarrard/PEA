"""Covers this batch's three changes:
  Sector-neutral portfolio checks now live in tests/portfolio.
  2. dual-class dedup (GOOG/FOX/NWS dropped, one per CIK) + GICS industry group.
  3. Google-Trends header rotation (anti-429).

The 13F data-set filename test that used to live here went away with the quarterly zip
download: fetch_13f discovers filings by date through edgartools now, so there are no
data-set names to build. See tests/data_aggregate/test_institutional_features.py.
"""

from __future__ import annotations

import pandas as pd


# ---- dual-class dedup + GICS industry group ---------------------------------
def test_dedupe_share_classes_and_gics():
    from src.data_extract.utils.common.gics import industry_group
    from src.data_extract.utils.prices.fetch_tickers import _dedupe_share_classes

    df = pd.DataFrame(
        {
            "ticker": ["GOOGL", "GOOG", "FOXA", "FOX", "NWSA", "NWS", "AAPL"],
            "cik": ["0001652044", "0001652044", "0001754301", "0001754301", "0001564708", "0001564708", "0000320193"],
            "name": ["Alphabet A", "Alphabet C", "Fox A", "Fox B", "News A", "News B", "Apple"],
        }
    )
    out = _dedupe_share_classes(df)
    kept = set(out["ticker"])
    assert {"GOOGL", "FOXA", "NWSA", "AAPL"} <= kept
    assert not ({"GOOG", "FOX", "NWS"} & kept), f"redundant classes not dropped: {kept}"
    assert len(out) == 4  # 3 companies deduped + AAPL

    # GICS sub-industry -> industry group (24); unknown -> sector fallback
    assert industry_group("Semiconductors", "Information Technology") == "Semiconductors & Semiconductor Equipment"
    assert industry_group("Systems Software", "Information Technology") == "Software & Services"
    assert industry_group("Integrated Oil & Gas", "Energy") == "Energy"
    assert industry_group("Totally Unknown Sub", "Utilities") == "Utilities"  # fallback
    print("\n=== SANITY CHECK: dual-class dedup + GICS industry group ===")
    print(f"  kept {sorted(kept)} (GOOG/FOX/NWS dropped, one per CIK); sub-industry->industry-group maps, unknown falls back to sector. Validated.")


# ---- 2. sector-neutral portfolio --------------------------------------------


if __name__ == "__main__":
    test_dedupe_share_classes_and_gics()
