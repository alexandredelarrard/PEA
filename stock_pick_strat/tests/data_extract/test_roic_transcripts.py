"""
Roic AI earnings-call fallback (src/data_extract/utils/behavioral/fetch_roic_transcripts.py).

Priority: HF backbone -> Roic AI (recent gap) -> Motley Fool (last resort). These tests prove:
  * Roic fills ONLY the missing quarters it actually covers, parses to sections, saves per ticker;
  * a ticker Roic doesn't cover is left for the fool fallback (nothing saved);
  * once a quarter is in the DB, the SHARED gap logic excludes it -> fool skips it;
  * a best-effort LIVE check that FRT / PM / MTD all have their recent quarters on Roic.
Transport is mocked; the live test skips without a ROIC API key.
"""
from __future__ import annotations

import logging
import time
import types
from pathlib import Path

import pandas as pd
import pytest

from src.data_extract.utils.behavioral import fetch_roic_transcripts as roic
from src.data_extract.utils.behavioral.fetch_earnings_calls import _missing_for, _quarter_index


def _ctx(saved: list):
    store = types.SimpleNamespace(save=lambda t, df: (saved.append(df), len(df))[1])
    return types.SimpleNamespace(store=store, log=logging.getLogger("roic-test"))


def test_roic_fills_missing_covered_only(monkeypatch):
    monkeypatch.setenv("ROIC_API_KEY", "test-key")
    # FRT is missing 2025Q2/Q3 and Roic HAS both; ZZZ is missing 2025Q2 but Roic has NOTHING
    monkeypatch.setattr(roic, "missing_quarters_by_ticker",
                        lambda ctx, tickers=None, since="2025-01-01": {
                            "FRT": ["2025Q2", "2025Q3"], "ZZZ": ["2025Q2"]})

    def _fake_list(ticker, apikey):
        return {"2025Q2": "2025-05-01", "2025Q3": "2025-08-01"} if ticker == "FRT" else {}

    def _fake_tx(ticker, quarter, apikey):
        secs = {"full": "operator " * 80, "prepared_remarks": "ceo remarks " * 80,
                "qa": "analyst question ceo answer " * 40}
        return secs, "2025-05-01"

    monkeypatch.setattr(roic, "roic_list_quarters", _fake_list)
    monkeypatch.setattr(roic, "roic_transcript_sections", _fake_tx)

    saved: list[pd.DataFrame] = []
    n = roic.fetch_roic_transcripts(_ctx(saved), tickers=["FRT", "ZZZ"], pause=0.0)

    assert n > 0 and saved, "Roic should have saved FRT's covered quarters"
    rows = pd.concat(saved, ignore_index=True)
    got = set(map(tuple, rows[["ticker", "quarter"]].drop_duplicates().to_numpy()))
    assert got == {("FRT", "2025Q2"), ("FRT", "2025Q3")}, got     # ZZZ left for fool
    assert set(rows["tag"]).issubset({"full", "prepared_remarks", "qa", "participants"})

    print("\n=== SANITY CHECK: Roic fills missing (covered) quarters ===")
    print(f"  FRT missing 2025Q2/Q3 & on Roic -> saved {sorted(got)}; "
          "ZZZ missing but NOT on Roic -> nothing saved (falls through to fool). Validated.")


def test_no_key_is_noop(monkeypatch):
    for k in roic.ROIC_API_KEY_ENV:
        monkeypatch.delenv(k, raising=False)
    saved: list = []
    assert roic.fetch_roic_transcripts(_ctx(saved), tickers=["FRT"], pause=0.0) == 0
    assert saved == []
    print("\n=== SANITY CHECK: no Roic key -> no-op ===")
    print("  missing key -> returns 0, saves nothing, pipeline falls back to fool. Validated.")


def test_db_covered_quarter_excluded_from_fool_gap():
    """Once Roic writes a quarter to the DB, the SHARED gap logic drops it -> fool won't refetch it."""
    end = _quarter_index(2026, 2)              # latest expected
    floor = _quarter_index(2025, 1)            # gap floor (no HF for this name)
    have_db = {"FRT": {"2025Q2"}}              # Roic already saved 2025Q2
    miss = _missing_for("FRT", hf_latest={}, floor_idx=floor, end_idx=end,
                        cache=Path("/does/not/exist"), have_db=have_db, have_json={})
    assert "2025Q2" not in miss, "a DB-covered (Roic) quarter must NOT be in the fool gap"
    assert "2025Q1" in miss, "an uncovered quarter stays in the gap"
    print("\n=== SANITY CHECK: Roic-covered quarter excluded from fool gap ===")
    print(f"  DB has FRT 2025Q2 -> fool gap = {sorted(miss)} (2025Q2 dropped, 2025Q1 kept). Validated.")


@pytest.mark.skipif(roic._api_key() is None, reason="no ROIC_API_KEY -> live Roic check skipped")
def test_roic_live_frt_pm_mtd():
    """Best-effort LIVE check: FRT, PM, MTD all have recent (2025+) quarters on Roic AI, and a
    transcript fetch parses into sections. Paced for the 5 req/min free tier."""
    key = roic._api_key()
    coverage = {}
    for t in ["FRT", "PM", "MTD"]:
        avail = roic.roic_list_quarters(t, key)
        recent = sorted(q for q in avail if q >= "2025Q1")
        coverage[t] = recent
        assert recent, f"{t}: Roic returned no 2025+ quarters"
        time.sleep(13)                                     # respect 5 req/min
    # fetch + parse one recent transcript end-to-end
    q = coverage["FRT"][-1]
    sections, as_of = roic.roic_transcript_sections("FRT", q, key)
    assert sections.get("full"), f"FRT {q}: no transcript content parsed"

    print("\n=== SANITY CHECK: Roic LIVE coverage FRT/PM/MTD ===")
    for t, qs in coverage.items():
        print(f"  {t}: {qs}")
    print(f"  FRT {q} parsed sections: {sorted(sections)} (as_of {as_of}). "
          "Roic covers these names' recent quarters (fool has none). Validated.")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
