"""
Incremental / point-in-time behaviour of the Wikipedia pageviews fetcher
(src/data_extract/utils/behavioral/fetch_wiki_pageviews.py::fetch_wiki_pageviews).

The last-extracted day is read PER TICKER from the stored `wiki_pageviews` table (max date per
ticker) and only days AFTER it are requested; a ticker current within the publication lag makes
NO request. Pace defaults to <=1 request/second. Network + article resolution are mocked.
"""
from __future__ import annotations

import inspect
import types

import pandas as pd

from src.constants.constants import DATE_FORMAT_COMPACT
from src.data_extract.utils.behavioral import fetch_wiki_pageviews as wp


def test_wiki_incremental_reads_last_date_per_ticker(tmp_path, monkeypatch):
    today = pd.Timestamp.today().normalize()
    aaa_last = today - pd.Timedelta(days=10)     # STALE -> re-extract from aaa_last + 1
    bbb_last = today - pd.Timedelta(days=1)      # CURRENT (within refetch window) -> skip, no call

    names = pd.DataFrame({"ticker": ["AAA", "BBB"], "name": ["Aaa Inc", "Bbb Corp"]})
    existing = pd.DataFrame({"date": [aaa_last, bbb_last], "ticker": ["AAA", "BBB"],
                             "pageviews": [100.0, 200.0]})

    saved: dict[str, pd.DataFrame] = {}
    store = types.SimpleNamespace(
        load=lambda t, columns=None: names if t == "sp500_tickers" else existing,
        save=lambda t, df: saved.__setitem__(t, df))
    ctx = types.SimpleNamespace(store=store, paths={"DATA_STORE": tmp_path})

    calls: list[tuple[str, str, str]] = []

    def fake_fetch(article, start, end):
        calls.append((article, start, end))
        return [{"timestamp": start + "00", "views": 5.0}]     # one day at `start`

    monkeypatch.setattr(wp, "_fetch_article", fake_fetch)
    monkeypatch.setattr(wp, "_resolve_wiki_article", lambda name, **k: name.replace(" ", "_"))

    wp.fetch_wiki_pageviews(ctx, pause=0.0, refetch_window_days=2)

    # only the stale ticker was requested, and from the day AFTER its stored max
    assert len(calls) == 1, f"expected 1 request (AAA only), got {len(calls)}"
    assert calls[0][0] == "Aaa_Inc"
    expected_start = (aaa_last + pd.Timedelta(days=1)).strftime(DATE_FORMAT_COMPACT)
    assert calls[0][1] == expected_start, f"start {calls[0][1]} != last+1 {expected_start}"
    # the default pace is <= 1 request / second
    assert inspect.signature(wp.fetch_wiki_pageviews).parameters["pause"].default == 1.0

    print("\n=== SANITY CHECK: Wikipedia incremental per-ticker ===")
    print(f"  stored max: AAA={aaa_last.date()} (stale), BBB={bbb_last.date()} (current)")
    print(f"  requests: {[(a, s) for a, s, _ in calls]} -> AAA re-extracted from {expected_start} "
          f"(last+1); BBB skipped (no call)")
    print(f"  default pause = {inspect.signature(wp.fetch_wiki_pageviews).parameters['pause'].default}s "
          "-> <=1 req/s. Validated.")


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    test_wiki_incremental_reads_last_date_per_ticker(Path(tempfile.mkdtemp()), None)
