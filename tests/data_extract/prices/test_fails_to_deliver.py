"""SEC Fails-to-Deliver: parse + semi-monthly period logic + the FTD feature
(fails/volume, publication-lagged so it's leak-free)."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.data_extract.utils.prices import fetch_fails_to_deliver as ftd
from src.data_aggregate.utils.extras.short_interest_features import (
    build_short_interest_feature_panel, _FTD_PUB_LAG)


def test_periods_semimonthly_bounded():
    ps = ftd._periods(1, today=pd.Timestamp("2024-03-10"))
    assert ps[:2] == ["202301a", "202301b"]                 # a=1-15, b=16-end
    assert ps[-2:] == ["202403a", "202403b"]                # up to the current month only
    assert "202312b" in ps
    assert all(int(p[:4]) >= ftd.SEC_FTD_FIRST_YEAR for p in ftd._periods(50, today=pd.Timestamp("2024-03-10")))
    # 15y reaches into the legacy era (FIRST_YEAR=2009) -> full history, not just 2017+
    assert "201001a" in ftd._periods(15, today=pd.Timestamp("2024-03-10"))
    print("\n=== SANITY: FTD semi-monthly periods ===")
    print(f"  years_history=1 @2024-03 -> {len(ps)} files 202301a..202403b; 15y reaches back to 2010 "
          f"(legacy era, FIRST_YEAR={ftd.SEC_FTD_FIRST_YEAR}). Validated.")


def test_period_urls_legacy_vs_modern_boundary():
    """<= 2017-06a -> FOIA legacy path (first); >= 2017-06b -> current path; the other
    path is the fallback. The switch is the 2nd half of June 2017."""
    _FOIA = "frequently-requested-foia"
    leg = ftd._period_urls("201301a")
    assert _FOIA in leg[0] and "cnsfails201301a.zip" in leg[0] and _FOIA not in leg[1]
    assert _FOIA in ftd._period_urls("201706a")[0]      # boundary: last legacy period
    assert _FOIA not in ftd._period_urls("201706b")[0]  # boundary: first modern period
    mod = ftd._period_urls("202401a")
    assert _FOIA not in mod[0] and "fails-deliver-data/cnsfails202401a" in mod[0] and _FOIA in mod[1]
    print("\n=== SANITY: FTD legacy/modern URL selection ===")
    print("  <=201706a -> FOIA legacy path first (modern fallback); >=201706b -> current path "
          "(legacy fallback). Boundary at 2017-06b. Validated.")


def test_parse_ftd_math_and_na_price():
    raw = ("SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
           "20240102|X|AAPL|1000|APPLE INC|180.50\n"
           "20240102|Y|MSFT|500|MICROSOFT|.\n"              # PRICE '.' = N/A
           "20240103|Z|AAPL|200|APPLE INC|181.00\n")
    df = ftd._parse_ftd(raw)
    a = df[(df["ticker"] == "AAPL") & (df["date"] == pd.Timestamp("2024-01-02"))].iloc[0]
    assert a["fails_quantity"] == 1000.0 and abs(a["fails_value"] - 180_500.0) < 1e-6
    m = df[df["ticker"] == "MSFT"].iloc[0]
    assert m["fails_quantity"] == 500.0 and pd.isna(m["fails_value"])   # '.' price -> value NaN
    assert set(df["ticker"]) == {"AAPL", "MSFT"} and len(df) == 3       # 2 AAPL dates + 1 MSFT
    print("\n=== SANITY: FTD parse ===")
    print("  AAPL 1000@180.5 -> fails_value $180.5k; MSFT price '.' -> fails_value NaN. Validated.")


def test_parse_ftd_matches_real_legacy_and_modern_samples():
    """Real rows pulled from SEC's actual cnsfails200907a.zip (legacy path) and
    cnsfails202401a.zip (current path): both eras share the identical column layout
    and units (verified live during this refactor), so one parser handles both --
    unlike 13F, FTD has no $thousands-vs-$ones split to guard against."""
    raw = ("SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
           "20090701|037833100|AAPL|32975|APPLE INC;COM NPV|142.43\n"    # legacy era
           "20240102|037833100|AAPL|516|APPLE INC;COM NPV|192.53\n")    # modern era
    df = ftd._parse_ftd(raw)
    legacy = df[df["date"] == pd.Timestamp("2009-07-01")].iloc[0]
    modern = df[df["date"] == pd.Timestamp("2024-01-02")].iloc[0]
    assert legacy["fails_quantity"] == 32975.0 and abs(legacy["fails_value"] - 4_696_629.25) < 1e-2
    assert modern["fails_quantity"] == 516.0 and abs(modern["fails_value"] - 99_345.48) < 1e-2
    print("\n=== SANITY CHECK: FTD real legacy vs. modern sample ===")
    print(f"  2009-07-01 AAPL 32975@142.43 -> ${legacy['fails_value']:,.2f}; "
          f"2024-01-02 AAPL 516@192.53 -> ${modern['fails_value']:,.2f}. Same units both eras. Validated.")


def test_fetch_skips_done_periods_and_upserts_without_duplicating(sqlite_store, monkeypatch, tmp_path):
    """Resume contract: an already-ingested period is never re-fetched while the universe
    is stable; a universe change re-parses cached periods, but the upsert on (ticker, date)
    must not duplicate rows already stored."""
    ctx = SimpleNamespace(store=sqlite_store, paths={"DATA_STORE": tmp_path})

    def _raw_for(path) -> str:
        period = path.stem.removeprefix("cnsfails")
        yyyymm, day = period[:6], ("01" if period.endswith("a") else "16")
        return ("SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
                f"{yyyymm}{day}|037833100|AAPL|100|APPLE INC|190.00\n"
                f"{yyyymm}{day}|055555555|MSFT|50|MICROSOFT|300.00\n")

    requested: list[str] = []
    def _fake_ensure_zip(path, urls, *, label, timeout, log):
        requested.append(label)
        return path
    monkeypatch.setattr(ftd, "ensure_zip", _fake_ensure_zip)
    monkeypatch.setattr(ftd, "read_zip_text", lambda path, log=None: _raw_for(path))
    monkeypatch.setattr(ftd, "_periods", lambda years_history, today=None: ["202401a", "202401b"])
    monkeypatch.setattr(ftd, "record_run", lambda *a, **k: None)

    # period 202401a already ingested, universe already converged on AAPL
    sqlite_store.replace("sec_fails_to_deliver", pd.DataFrame({
        "ticker": ["AAPL"], "date": pd.to_datetime(["2024-01-01"]),
        "fails_quantity": [100.0], "fails_value": [19000.0], "period": ["202401a"],
    }))
    cache = ftd.cache_dir(ctx, "sec_fails_to_deliver")
    ftd.save_processed_universe(cache, ftd.Tables.sec_fails_to_deliver, {"AAPL"})

    # 1) stable universe -> the already-done period is skipped, only the new one is fetched
    saved = ftd.fetch_fails_to_deliver(ctx, tickers=["AAPL"], years_history=1)
    assert requested == ["FTD 202401b"], f"already-done period was re-fetched: {requested}"
    assert saved == 1
    stored = sqlite_store.load("sec_fails_to_deliver")
    assert len(stored) == 2                                # seeded 202401a row + new 202401b row

    # 2) universe grows (MSFT) -> both cached periods are re-parsed, but the upsert on
    #    (ticker, date) must not duplicate the 202401a/202401b AAPL rows already stored
    requested.clear()
    saved2 = ftd.fetch_fails_to_deliver(ctx, tickers=["AAPL", "MSFT"], years_history=1)
    assert requested == ["FTD 202401a", "FTD 202401b"]
    stored2 = sqlite_store.load("sec_fails_to_deliver")
    assert len(stored2) == 4                               # AAPL+MSFT x 2 periods, no duplicates
    assert saved2 == 4

    print("\n=== SANITY CHECK: FTD resume + universe-growth reparse ===")
    print(f"  stable universe -> 202401a skipped (no fetch), only 202401b fetched; "
          f"universe growth -> both reparsed, {len(stored2)} distinct rows stored "
          "(upsert, no duplicates). Validated.")


def test_ftd_feature_ranks_high_fails_and_is_leak_free():
    idx = pd.bdate_range("2024-01-01", periods=120)
    days = idx[:30]
    fails = pd.concat([
        pd.DataFrame({"date": days, "ticker": "HI", "fails_quantity": 1e5}),
        pd.DataFrame({"date": days, "ticker": "MID", "fails_quantity": 1e4}),
        pd.DataFrame({"date": days, "ticker": "LO", "fails_quantity": 1e2}),
    ], ignore_index=True)
    volume = pd.DataFrame({t: 1e6 for t in ("HI", "MID", "LO")}, index=idx)
    peers = {"HI": {"MID": 1.0, "LO": 1.0}, "MID": {"HI": 1.0, "LO": 1.0}, "LO": {"HI": 1.0, "MID": 1.0}}

    panel = build_short_interest_feature_panel(None, peers, idx, fails_history=fails, volume=volume)
    assert "f_fails_to_deliver_ratio_xs" in panel.columns

    # after the publication lag, HI (0.1 fails/vol) ranks above LO (0.0001)
    d = idx[_FTD_PUB_LAG + 25]
    row = panel[panel["date"] == d].set_index("ticker")
    assert row["f_fails_to_deliver_ratio_xs"]["HI"] > row["f_fails_to_deliver_ratio_xs"]["LO"]

    # leak-free: before the publication lag the fails signal is not yet visible
    early = panel[panel["date"] == idx[5]]
    assert early.empty or early["f_fails_to_deliver_ratio_xs"].isna().all()

    print("\n=== SANITY: FTD feature (fails/volume, publication-lagged) ===")
    print(f"  HI fails/vol 0.10 ranks above LO 0.0001 after the {_FTD_PUB_LAG}d lag; "
          f"pre-lag signal absent (leak-free). Validated.")
