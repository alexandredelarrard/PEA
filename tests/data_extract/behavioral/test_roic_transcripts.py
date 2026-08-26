"""
Roic AI earnings-call fallback (src/data_extract/utils/behavioral/fetch_roic_transcripts.py).

Priority: HF backbone -> Roic AI (recent gap) -> Motley Fool (last resort). These tests prove:
  * Roic fills ONLY the missing quarters it actually covers, parses to sections, saves per ticker;
  * a ticker Roic doesn't cover is left for the fool fallback (nothing saved);
  * the gap is computed ONCE and handed down: Roic reports what it filled, `remaining_after`
    subtracts it, and only the remainder reaches the fool quote-page discovery;
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
from src.data_extract.utils.behavioral.utils_missing_quarters import (
    _missing_for,
    _quarter_index,
    remaining_after,
)


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
    result = roic.fetch_roic_transcripts(_ctx(saved), tickers=["FRT", "ZZZ"], pause=0.0)

    assert result.saved > 0 and saved, "Roic should have saved FRT's covered quarters"
    rows = pd.concat(saved, ignore_index=True)
    got = set(map(tuple, rows[["ticker", "quarter"]].drop_duplicates().to_numpy()))
    assert got == {("FRT", "2025Q2"), ("FRT", "2025Q3")}, got     # ZZZ left for fool
    assert set(rows["tag"]).issubset({"full", "prepared_remarks", "qa", "participants"})
    # what it FILLED is reported back, so the caller can subtract it from the shared gap
    assert result.filled == {"FRT": {"2025Q2", "2025Q3"}}, result.filled

    print("\n=== SANITY CHECK: Roic fills missing (covered) quarters ===")
    print(f"  FRT missing 2025Q2/Q3 & on Roic -> saved {sorted(got)}; "
          "ZZZ missing but NOT on Roic -> nothing saved (falls through to fool). Validated.")


def test_shared_gap_is_computed_once_and_handed_down(monkeypatch):
    """The user-facing contract of the refactor: ONE gap computation, then roic, then fool.

    `missing` is INJECTED into Roic (it must not re-derive it -- that scan reads the 1.8 GB HF
    parquet), Roic reports `filled`, and `remaining_after` leaves the fool step exactly the
    quarters Roic could not supply."""
    monkeypatch.setenv("ROIC_API_KEY", "test-key")

    # a sentinel that FAILS the test if Roic recomputes the gap instead of using the injected one
    def _must_not_be_called(*a, **k):
        raise AssertionError("fetch_roic_transcripts re-derived the gap instead of using `missing`")

    monkeypatch.setattr(roic, "missing_quarters_by_ticker", _must_not_be_called)
    # Roic covers FRT's Q2 only; ZZZ not at all
    monkeypatch.setattr(roic, "roic_list_quarters",
                        lambda t, k: {"2025Q2": "2025-05-01"} if t == "FRT" else {})
    monkeypatch.setattr(roic, "roic_transcript_sections",
                        lambda t, q, k: ({"full": "operator " * 80}, "2025-05-01"))

    missing = {"FRT": ["2025Q2", "2025Q3"], "ZZZ": ["2025Q2"]}
    result = roic.fetch_roic_transcripts(_ctx([]), missing=missing, pause=0.0)
    left = remaining_after(missing, result.filled)

    assert result.filled == {"FRT": {"2025Q2"}}, result.filled
    assert left == {"FRT": ["2025Q3"], "ZZZ": ["2025Q2"]}, left
    assert missing == {"FRT": ["2025Q2", "2025Q3"], "ZZZ": ["2025Q2"]}, "input must not be mutated"

    print("\n=== SANITY CHECK: one gap, computed once, handed down ===")
    print(f"  gap in            : {missing}")
    print(f"  Roic filled       : {result.filled}  (its own gap derivation was never called)")
    print(f"  -> left for fool  : {left}")
    print("  FRT 2025Q2 dropped (Roic got it), FRT 2025Q3 + ZZZ 2025Q2 remain. Validated.")


def test_no_key_is_noop(monkeypatch):
    monkeypatch.delenv("ROIC_API_KEY", raising=False)
    saved: list = []
    result = roic.fetch_roic_transcripts(_ctx(saved), tickers=["FRT"], pause=0.0)
    assert result.saved == 0 and result.filled == {}
    assert saved == []
    print("\n=== SANITY CHECK: no Roic key -> no-op ===")
    print("  missing key -> saved=0, filled={}, pipeline falls back to fool. Validated.")


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
        # SKIP, don't fail, on an empty first response: `roic_list_quarters` returns {} for a
        # transport error as well as for genuinely-absent coverage, and the corporate TLS proxy
        # drops a connection often enough that a hard assert reads as "Roic lost these names"
        # when it is really a network hiccup (the same convention as the live MF test).
        if not recent and t == "FRT":
            pytest.skip("Roic unreachable (empty response for FRT) -> live check skipped")
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
