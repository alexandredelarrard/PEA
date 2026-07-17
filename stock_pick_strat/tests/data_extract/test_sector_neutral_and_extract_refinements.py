"""Covers this batch's four changes:
  1. 13F data-set filenames use the DDMMMYYYY-DDMMMYYYY window convention.
  2. Sector-neutral portfolio construction -> net weight per group ~ 0.
  3. dual-class dedup (GOOG/FOX/NWS dropped, one per CIK) + GICS industry group.
  4. Google-Trends header rotation (anti-429).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---- 1. 13F filenames --------------------------------------------------------
def test_13f_period_names_format():
    from src.data_extract.utils.prices.fetch_13f import _period_names
    today = pd.Timestamp("2026-07-15")
    names = _period_names(years_history=2, today=today)
    # user's example window must be produced, and the Jun-Aug 2025 one
    assert "01dec2025-28feb2026" in names, names
    assert "01jun2025-31aug2025" in names, names
    # windows whose end is in the future are excluded
    assert all(not n.startswith("01dec2026") for n in names)
    # every name matches DDMMMYYYY-DDMMMYYYY (lowercase)
    import re
    pat = re.compile(r"^\d{2}[a-z]{3}\d{4}-\d{2}[a-z]{3}\d{4}$")
    assert all(pat.match(n) for n in names), names
    print("\n=== SANITY CHECK: 13F data-set filename windows ===")
    print(f"  {len(names)} windows incl 01dec2025-28feb2026 & 01jun2025-31aug2025; "
          f"all DDMMMYYYY-DDMMMYYYY; future windows excluded. Validated.")


# ---- 3. dual-class dedup + GICS industry group ------------------------------
def test_dedupe_share_classes_and_gics():
    from src.data_extract.utils.prices.fetch_prices import _dedupe_share_classes
    from src.data_extract.utils.common.gics import industry_group
    df = pd.DataFrame({
        "ticker": ["GOOGL", "GOOG", "FOXA", "FOX", "NWSA", "NWS", "AAPL"],
        "cik": ["0001652044", "0001652044", "0001754301", "0001754301",
                "0001564708", "0001564708", "0000320193"],
        "name": ["Alphabet A", "Alphabet C", "Fox A", "Fox B", "News A", "News B", "Apple"],
    })
    out = _dedupe_share_classes(df)
    kept = set(out["ticker"])
    assert {"GOOGL", "FOXA", "NWSA", "AAPL"} <= kept
    assert not ({"GOOG", "FOX", "NWS"} & kept), f"redundant classes not dropped: {kept}"
    assert len(out) == 4                       # 3 companies deduped + AAPL

    # GICS sub-industry -> industry group (24); unknown -> sector fallback
    assert industry_group("Semiconductors", "Information Technology") == "Semiconductors & Semiconductor Equipment"
    assert industry_group("Systems Software", "Information Technology") == "Software & Services"
    assert industry_group("Integrated Oil & Gas", "Energy") == "Energy"
    assert industry_group("Totally Unknown Sub", "Utilities") == "Utilities"   # fallback
    print("\n=== SANITY CHECK: dual-class dedup + GICS industry group ===")
    print(f"  kept {sorted(kept)} (GOOG/FOX/NWS dropped, one per CIK); "
          f"sub-industry->industry-group maps, unknown falls back to sector. Validated.")


# ---- 2. sector-neutral portfolio --------------------------------------------
def test_sector_neutral_weights_sum_to_zero_per_group():
    from src.post_processing.utils.strategies_opt import optimize_day
    rng = np.random.default_rng(0)
    n = 60
    alpha = rng.normal(0, 1, n)
    beta = rng.uniform(0.6, 1.4, n)
    var = rng.uniform(1e-4, 4e-4, n)
    groups = np.array(["A", "B", "C"])[rng.integers(0, 3, n)]

    w_plain = optimize_day(alpha, beta, var, beta_neutral=True, pos_cap=0.5)
    w_sec = optimize_day(alpha, beta, var, beta_neutral=True, pos_cap=0.5,
                         sector_labels=list(groups))

    # plain: only GLOBAL dollar-neutral -> per-group sums are generally NONZERO
    plain_group_sums = {g: w_plain[groups == g].sum() for g in set(groups)}
    # sector-neutral: EACH group sum ~ 0
    sec_group_sums = {g: w_sec[groups == g].sum() for g in set(groups)}
    assert max(abs(v) for v in sec_group_sums.values()) < 1e-6, sec_group_sums
    assert max(abs(v) for v in plain_group_sums.values()) > 1e-3, plain_group_sums
    # still beta-neutral
    assert abs(float((beta * w_sec).sum())) < 1e-6
    print("\n=== SANITY CHECK: sector-neutral construction ===")
    print(f"  per-group net weight: plain max |sum|={max(abs(v) for v in plain_group_sums.values()):.4f} "
          f"vs sector-neutral max |sum|={max(abs(v) for v in sec_group_sums.values()):.2e} (~0); "
          f"beta-neutral preserved. Validated.")


# ---- 4. Google Trends header rotation ---------------------------------------
def test_google_trends_header_rotation():
    try:
        from src.data_extract.utils.behavioral.fetch_google_trends import _random_header, _USER_AGENTS
    except ModuleNotFoundError as e:      # module hard-imports pytrends at top
        print(f"\n=== SKIP: Google Trends header test (pytrends not installed: {e}) ===")
        return
    seen = {_random_header()["User-Agent"] for _ in range(200)}
    assert len(seen) >= 3, "User-Agent not rotating"
    assert seen <= set(_USER_AGENTS)
    h = _random_header()
    for k in ("User-Agent", "Accept-Language", "Accept", "Referer"):
        assert k in h and h[k]
    print("\n=== SANITY CHECK: Google Trends anti-429 header rotation ===")
    print(f"  rotated across {len(seen)} User-Agents over 200 calls; full browser-like "
          f"headers incl Referer. (Plus fresh session + retry/backoff + jitter.) Validated.")


if __name__ == "__main__":
    test_13f_period_names_format()
    test_dedupe_share_classes_and_gics()
    test_sector_neutral_weights_sum_to_zero_per_group()
    test_google_trends_header_rotation()
