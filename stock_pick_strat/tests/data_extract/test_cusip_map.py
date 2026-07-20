"""CUSIP->ticker map (src/data_extract/utils/prices/fetch_cusip_map.py).

A CUSIP is always 9 chars, but filers drop the leading zero on all-digit ones, so
the same security appears as '037833100' and '37833100'. The map must canonicalize
both the stored and the queried CUSIP (zfill 9) so the incremental 'already mapped?'
check actually SKIPS instead of re-running the rate-limited OpenFIGI lookup each time.
"""
from __future__ import annotations

import types

import pandas as pd

import src.data_extract.utils.prices.fetch_cusip_map as cm
from src.data_extract.utils.prices.fetch_cusip_map import (
    normalize_cusip, build_cusip_ticker_map,
)


class _FakeStore:
    """Minimal in-memory stand-in for context.store (dedupes on cusip like the DB)."""
    def __init__(self):
        self._t: dict[str, pd.DataFrame] = {}

    def load(self, name, columns=None, limit=None):
        return self._t.get(name, pd.DataFrame(columns=["cusip", "ticker"])).copy()

    def save(self, name, df, pk=None):
        prev = self._t.get(name, pd.DataFrame(columns=df.columns))
        self._t[name] = (pd.concat([prev, df], ignore_index=True)
                         .drop_duplicates("cusip", keep="last").reset_index(drop=True))
        return len(df)


def _ctx():
    return types.SimpleNamespace(store=_FakeStore())


def test_normalize_cusip():
    assert normalize_cusip("37833100") == "037833100"      # dropped leading zero restored
    assert normalize_cusip("037833100") == "037833100"     # already 9 -> unchanged
    assert normalize_cusip(" 037833100 ") == "037833100"   # stripped
    assert normalize_cusip("aapl0001x") == "AAPL0001X"     # uppercased (9-char no-op)
    for bad in (None, "", "nan", "NaN", "<NA>", float("nan")):
        assert normalize_cusip(bad) is None
    print("\n=== SANITY: normalize_cusip ===")
    print("  '37833100' -> '037833100' (zfill 9); strip/upper applied; blank/NaN -> None. Validated.")


def test_build_cusip_map_skips_across_zfill(monkeypatch):
    """The bug: a mapped CUSIP re-queried in a different zero-padding was re-looked-up
    every run. After canonicalization it is skipped."""
    calls = {"n": 0}

    def fake_req(cusips, api_key):
        calls["n"] += 1
        return [{"data": [{"ticker": "AAPL"}]} for _ in cusips]
    monkeypatch.setattr(cm, "_openfigi_request", fake_req)

    ctx = _ctx()
    # run 1: unmapped -> ONE OpenFIGI call; stores the canonical '037833100'
    out1 = build_cusip_ticker_map(ctx, ["037833100"], pause=0.0)
    assert calls["n"] == 1
    assert set(out1["cusip"]) == {"037833100"} and out1.iloc[0]["ticker"] == "AAPL"

    # run 2: SAME security, but the filer dropped the leading zero -> must SKIP (no call)
    out2 = build_cusip_ticker_map(ctx, ["37833100"], pause=0.0)
    assert calls["n"] == 1, "re-looked-up an already-mapped CUSIP (zfill mismatch -> redo every run)"
    assert set(out2["cusip"]) == {"037833100"}

    # run 3: a genuinely new CUSIP still gets looked up (skip isn't over-eager)
    build_cusip_ticker_map(ctx, ["37833100", "594918104"], pause=0.0)
    assert calls["n"] == 2

    print("\n=== SANITY: cusip map incremental skip across zfill ===")
    print("  run1 mapped '037833100' (1 call); run2 '37833100' (same security) -> 0 new calls "
          "(skipped); run3 new CUSIP -> 1 call. No more re-doing every run. Validated.")


def test_unmapped_cusip_recorded_not_requeried(monkeypatch):
    """A CUSIP OpenFIGI can't map (bond / option / delisted) is RECORDED as attempted
    (ticker NULL) so it isn't re-queried against the rate-limited API every run -- the
    'takes ages' bug. Only real mappings come back for the holdings<->ticker merge."""
    calls = {"n": 0}

    def fake_req(cusips, api_key):
        calls["n"] += 1
        return [({"data": [{"ticker": "AAPL"}]} if c == "037833100" else {"data": []})
                for c in cusips]                       # 999999999 = genuine no-match
    monkeypatch.setattr(cm, "_openfigi_request", fake_req)

    ctx = _ctx()
    out1 = build_cusip_ticker_map(ctx, ["037833100", "999999999"], pause=0.0)
    assert calls["n"] == 1
    assert set(out1["cusip"]) == {"037833100"}         # only the real mapping is returned
    # the unmappable cusip was recorded -> NOT re-queried on the next run
    out2 = build_cusip_ticker_map(ctx, ["037833100", "999999999"], pause=0.0)
    assert calls["n"] == 1, "unmapped CUSIP was re-queried (the 'takes ages' bug is not fixed)"
    assert set(out2["cusip"]) == {"037833100"}

    print("\n=== SANITY: unmapped CUSIP recorded, not re-queried ===")
    print("  '999999999' (no OpenFIGI match) recorded as attempted -> run2 makes 0 new calls; "
          "only real mappings feed the merge. The rate-limited tail no longer re-runs. Validated.")
