"""
Motley Fool transcript-index crawl stop logic
(src/data_extract/utils/behavioral/fetch_earnings_calls.py::build_transcript_index).

Regression test for the "index stops after ~16 links" bug: the MF index is a global,
all-companies, newest-first feed, so most pages carry NO S&P 500 name. The old
`stop_after_empty` counted those universe-empty pages as "empty" and aborted the crawl
after 3 of them — truncating the index to a handful of links. The fix only counts a page
toward the stop when it has universe transcripts that are ALL already indexed (genuine
re-run convergence); a page with no universe names must NOT stop the deep crawl.
"""
from __future__ import annotations

import json
import types

import pandas as pd

from src.data_extract.utils.behavioral import fetch_earnings_calls as fe


def _href(date: str, slug: str, q: int, fy: int) -> str:
    y, m, d = date.split("-")
    return (f'<a href="/earnings/call-transcripts/{y}/{m}/{d}/{slug}-q{q}-{fy}-'
            f'earnings-call-transcript/">x</a>')


def _page(items) -> str:
    return "<html><body>" + "".join(_href(*it) for it in items) + "</body></html>"


def _fake_ctx(tickers, tmp_path):
    store = types.SimpleNamespace(
        load=lambda table, columns=None: pd.DataFrame({"ticker": list(tickers)}))
    return types.SimpleNamespace(store=store, paths={"DATA_STORE": tmp_path})


def test_crawl_does_not_stop_on_universe_empty_pages(tmp_path, monkeypatch):
    # AAA, BBB on p1 (new); p2-p4 have NO universe name (small-caps) -> the OLD logic would
    # abort here; p5 carries a NEW universe name (CCC) only reachable if the crawl continued;
    # p6-p9 repeat already-indexed universe links -> genuine convergence -> stop.
    pages = {
        1: _page([("2026-07-20", "alpha-aaa", 1, 2026), ("2026-07-19", "beta-bbb", 1, 2026),
                  ("2026-07-19", "junkco-zzz", 1, 2026)]),
        2: _page([("2026-07-18", "smallcap-yyy", 2, 2026)]),
        3: _page([("2026-07-17", "micro-xxx", 2, 2026)]),
        4: _page([("2026-07-16", "tiny-www", 2, 2026)]),
        5: _page([("2026-07-15", "gamma-ccc", 1, 2026)]),
        6: _page([("2026-07-20", "alpha-aaa", 1, 2026)]),
        7: _page([("2026-07-20", "alpha-aaa", 1, 2026)]),
        8: _page([("2026-07-20", "alpha-aaa", 1, 2026)]),
        9: _page([("2026-07-20", "alpha-aaa", 1, 2026)]),
    }
    fetched = []

    def fake_get(url, *a, **k):
        page = 1 if url == fe._INDEX else int(url.rstrip("/").split("/")[-1])
        fetched.append(page)
        return pages.get(page)                      # None past the defined pages = end of feed

    monkeypatch.setattr(fe, "_get", fake_get)
    ctx = _fake_ctx(["AAA", "BBB", "CCC"], tmp_path)

    index = fe.build_transcript_index(ctx, max_pages=50, stop_after_empty=4,
                                      pause=0.0, history_years=100)

    tickers = {r["ticker"] for r in index.values()}
    assert tickers == {"AAA", "BBB", "CCC"}, f"missing universe links: {tickers}"
    # the crawl reached page 5 (CCC) despite 3 universe-empty pages before it — the bug fix
    assert max(fetched) >= 5, f"crawl stopped early at page {max(fetched)} (universe-empty abort)"
    # convergence still stops it (didn't run to max_pages)
    assert max(fetched) <= 9, f"crawl did not converge, ran to page {max(fetched)}"
    # persisted to the big JSON
    saved = json.loads((tmp_path / "call_transcripts" / "transcript_index.json").read_text())
    assert len(saved) == 3

    # unit-level: the convergence predicate itself
    assert fe._page_converged([{"url": "u"}], 0) is True        # universe links, all seen
    assert fe._page_converged([], 0) is False                   # no universe names -> keep going
    assert fe._page_converged([{"url": "u"}], 1) is False       # a new link

    print("\n=== SANITY CHECK: MF transcript-index crawl stop logic ===")
    print(f"  pages fetched: {fetched}")
    print(f"  indexed tickers: {sorted(tickers)} ({len(saved)} links)")
    print("  3 universe-EMPTY pages (p2-p4) no longer abort the crawl -> CCC on p5 is reached;")
    print("  genuine convergence (p6-p9 all already-indexed) stops it at "
          f"page {max(fetched)}. Old logic stopped at ~p3-4 with a truncated index.")


def test_crawl_stops_at_history_horizon(tmp_path, monkeypatch):
    # every page is OLD (2016) and carries a universe name; with a 1-year horizon the crawl
    # must stop as soon as it sees the feed is older than the cutoff (not run forever).
    old = _page([("2016-01-15", "alpha-aaa", 1, 2016)])
    fetched = []

    def fake_get(url, *a, **k):
        page = 1 if url == fe._INDEX else int(url.rstrip("/").split("/")[-1])
        fetched.append(page)
        return old

    monkeypatch.setattr(fe, "_get", fake_get)
    ctx = _fake_ctx(["AAA"], tmp_path)
    fe.build_transcript_index(ctx, max_pages=50, stop_after_empty=4, pause=0.0, history_years=1)
    assert max(fetched) <= 2, f"history-horizon stop failed; ran to page {max(fetched)}"
    print("  history horizon: a 2016 feed with a 1y horizon stops at page "
          f"{max(fetched)} instead of crawling max_pages. Validated.")


if __name__ == "__main__":
    import tempfile, pathlib
    for t in (test_crawl_does_not_stop_on_universe_empty_pages, test_crawl_stops_at_history_horizon):
        import pytest
        pytest.main(["-x", "-s", f"{__file__}::{t.__name__}"])
