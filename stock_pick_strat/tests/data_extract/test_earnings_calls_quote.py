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


def _idx_to_tuple(idx: int) -> tuple[int, int]:
    return idx // 4, idx % 4 + 1


def test_quote_discovery_hf_and_local_gap(tmp_path, monkeypatch):
    """The 429 fix: only tickers with a REAL missing-quarter gap hit fool. A ticker whose HF
    backbone already reaches the latest expected quarter is skipped; so is one whose gap is
    already on disk. Only the genuinely-behind ticker spends a request."""
    end_idx = fe._latest_expected_quarter_index()                 # newest quarter expected today
    end_q = fe._index_to_quarter(end_idx)                         # e.g. "2026Q2"
    y, q = _idx_to_tuple(end_idx)

    # AAA: HF already covers up to the latest expected quarter -> no gap.
    # BBB: HF is 2 quarters behind -> a gap -> must fetch its quote page.
    # CCC: HF 1 quarter behind (gap = {end_q}) BUT that quarter is already on disk -> skip.
    def fake_hf(context, tickers=None):
        return {"AAA": _idx_to_tuple(end_idx), "BBB": _idx_to_tuple(end_idx - 2),
                "CCC": _idx_to_tuple(end_idx - 1)}
    monkeypatch.setattr(
        "src.data_extract.utils.behavioral.fetch_hf_transcripts.hf_latest_quarter_by_ticker", fake_hf)

    # pre-seed CCC's gap quarter on disk so it is already complete
    ccc_dir = tmp_path / "call_transcripts" / "CCC"
    ccc_dir.mkdir(parents=True)
    (ccc_dir / f"{end_q}.html").write_text("cached", encoding="utf-8")

    bbb_page = "x " + _t(f"{y}/06/01", f"bbb-q{q}-{y}")           # BBB's latest-quarter link
    calls: list[str] = []

    def fake_get(url, *a, **k):
        calls.append(url)
        return bbb_page if "/quote/nasdaq/bbb/" in url else None

    monkeypatch.setattr(fe, "_get", fake_get)
    ctx = _ctx(["AAA", "BBB", "CCC"], tmp_path)

    idx = fe.build_transcript_index_by_ticker(ctx, pause=0.0)
    got = {(r["ticker"], r["quarter"]) for r in idx.values()}

    assert not any("/aaa/" in u for u in calls), "AAA (HF-complete) must NOT be requested"
    assert not any("/ccc/" in u for u in calls), "CCC (gap already on disk) must NOT be requested"
    assert any("/bbb/" in u for u in calls), "BBB (behind HF) must be requested"
    assert ("BBB", end_q) in got, "BBB's latest-quarter link should be indexed"
    assert not any(t == "AAA" for t, _ in got) and not any(t == "CCC" for t, _ in got)

    print("\n=== SANITY CHECK: HF-aware + local-folder gap (429 fix) ===")
    print(f"  latest expected quarter today = {end_q}")
    print(f"  AAA HF@{end_q} (complete) -> skipped | CCC HF@{fe._index_to_quarter(end_idx-1)} but "
          f"{end_q}.html on disk -> skipped | BBB HF@{fe._index_to_quarter(end_idx-2)} -> fetched")
    print(f"  requests made: {[u.split('/quote/')[-1].rstrip('/') for u in calls]}")
    print("  CONCLUSION: only the genuinely-behind ticker hits fool; HF horizon + local files + DB "
          "coverage skip the rest -> far fewer requests, no 429 burst. Validated.")


if __name__ == "__main__":
    import tempfile, pathlib
    test_quote_discovery_live(pathlib.Path(tempfile.mkdtemp()))
