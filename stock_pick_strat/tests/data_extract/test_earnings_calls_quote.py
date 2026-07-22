"""
Per-ticker QUOTE-PAGE transcript discovery
(src/data_extract/utils/behavioral/fetch_earnings_calls.py::build_transcript_index_by_ticker).

MF requires the exact date + slug in a transcript URL (404s otherwise), so URLs can't be
built from (ticker, quarter). Instead we hit each ticker's quote page
`fool.com/quote/{exchange}/{ticker}/`, which LISTS its recent transcript URLs (exact date +
slug baked in) — one request/ticker, not capped at MF's ~500-page global feed. This gives
COMPLETE coverage since a cutoff (default 2025-01-01). Tests: exchange fallback, since-filter,
foreign-ticker filter, JSON merge — plus a best-effort live check on real quote pages.
"""
from __future__ import annotations

import json
import types

import pandas as pd
import pytest

from src.data_extract.utils.behavioral import fetch_earnings_calls as fe


def _ctx(tickers, tmp_path):
    store = types.SimpleNamespace(load=lambda table, columns=None: pd.DataFrame({"ticker": list(tickers)}))
    return types.SimpleNamespace(store=store, paths={"DATA_STORE": tmp_path})


def _t(date, slug):   # a transcript path as it appears in quote-page JSON
    return f'"url":"/earnings/call-transcripts/{date}/{slug}-earnings-call-transcript/"'


def test_quote_discovery_filters_and_merges(tmp_path, monkeypatch):
    # AAA lives on nasdaq; its page has a POST-cutoff call, a PRE-cutoff call, and a FOREIGN
    # (BBB) link. BBB lives on NYSE only (nasdaq 404s -> exchange fallback must find it).
    pages = {
        "nasdaq/aaa": "junk " + _t("2025/07/17", "aaa-q2-2025") + " " +
                      _t("2024/07/17", "aaa-q2-2024") + " " + _t("2025/06/01", "bbb-q4-2024"),
        "nyse/bbb": "x " + _t("2025/04/30", "beta-corp-bbb-q1-2025"),
    }

    def fake_get(url, *a, **k):
        for key, html in pages.items():
            if f"/quote/{key}/" in url:
                return html
        return None                                   # wrong exchange / unknown -> 404

    monkeypatch.setattr(fe, "_get", fake_get)
    ctx = _ctx(["AAA", "BBB"], tmp_path)

    idx = fe.build_transcript_index_by_ticker(ctx, since="2025-01-01",
                                              exchanges=("nasdaq", "nyse"), pause=0.0)
    got = {(r["ticker"], r["quarter"]) for r in idx.values()}

    assert ("AAA", "2025Q2") in got                    # post-cutoff, own page -> kept
    assert ("BBB", "2025Q1") in got                    # found via NYSE fallback
    assert ("AAA", "2024Q2") not in got                # pre-cutoff -> filtered by `since`
    assert ("BBB", "2024Q4") not in got                # foreign link on AAA's page -> ticker-filtered
    # persisted to the same big JSON the downloader reads
    saved = json.loads((tmp_path / "call_transcripts" / "transcript_index.json").read_text())
    assert len(saved) == 2

    # _quote_page exchange fallback returns the resolving exchange
    assert fe._quote_page("BBB", ("nasdaq", "nyse")) == (pages["nyse/bbb"], "nyse")
    assert fe._quote_page("ZZZ", ("nasdaq", "nyse")) == (None, None)

    print("\n=== SANITY CHECK: quote-page transcript discovery ===")
    print(f"  kept {sorted(got)} from 2 quote pages")
    print("  since-filter drops pre-2025 (AAA 2024Q2); ticker-filter drops the foreign BBB link on "
          "AAA's page; NYSE fallback found BBB after nasdaq 404. Merged to transcript_index.json.")
    print("  CONCLUSION: (ticker,quarter) -> exact MF URL via the quote page, uncapped, since a "
          "cutoff. Validated.")


def test_quote_discovery_live(tmp_path):
    """Best-effort live check: real MF quote pages yield post-2025 transcript links."""
    ctx = _ctx(["AAPL", "NVDA", "JPM"], tmp_path)
    try:
        idx = fe.build_transcript_index_by_ticker(ctx, since="2025-01-01", pause=0.8)
    except Exception as e:                             # noqa: BLE001
        pytest.skip(f"MF unreachable: {e}")
    if not idx:
        pytest.skip("no links returned (MF blocked)")
    by_tkr = {}
    for r in idx.values():
        by_tkr.setdefault(r["ticker"], []).append(r["quarter"])
    assert all(d["call_date"] >= "2025-01-01" for d in idx.values()), "since-filter leaked older calls"
    print("\n=== SANITY CHECK: quote-page discovery on REAL MF pages ===")
    for t, qs in sorted(by_tkr.items()):
        print(f"  {t}: {sorted(set(qs), reverse=True)}")
    print(f"  {len(idx)} post-2025 transcript URLs across {len(by_tkr)} tickers, uncapped. Validated.")


def test_quote_discovery_skips_covered_tickers(tmp_path, monkeypatch):
    # with `_expected_quarter_count` forced to 1, a single post-cutoff link marks a ticker
    # COMPLETE -> the SECOND run must skip it (no request), so re-runs don't redo work / hit 429s.
    monkeypatch.setattr(fe, "_expected_quarter_count", lambda *a, **k: 1)
    page = "x " + _t("2025/07/17", "aaa-q2-2025")
    calls = {"n": 0}

    def fake_get(url, *a, **k):
        calls["n"] += 1
        return page if "/quote/nasdaq/aaa/" in url else None      # AAA on nasdaq only

    monkeypatch.setattr(fe, "_get", fake_get)
    ctx = _ctx(["AAA"], tmp_path)

    fe.build_transcript_index_by_ticker(ctx, since="2025-01-01", pause=0.0)
    first = calls["n"]
    fe.build_transcript_index_by_ticker(ctx, since="2025-01-01", pause=0.0)     # re-run
    second = calls["n"] - first

    assert first >= 1, "first run should have fetched AAA"
    assert second == 0, f"re-run should SKIP already-complete AAA (made {second} requests)"
    print("\n=== SANITY CHECK: skip already-complete tickers ===")
    print(f"  run1 made {first} request(s); run2 made {second} (AAA complete in the JSON index -> "
          "skipped). Re-runs don't redo extraction or re-hit the site.")


if __name__ == "__main__":
    import tempfile, pathlib
    test_quote_discovery_live(pathlib.Path(tempfile.mkdtemp()))
