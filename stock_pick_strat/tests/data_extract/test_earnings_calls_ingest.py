"""
Incremental MF ingest (src/data_extract/utils/behavioral/fetch_earnings_calls.py::ingest_earnings_calls).

The DAG ingest step re-runs daily on a full transcript cache; it must SKIP (ticker, quarter) already
in earnings_call_sections instead of re-reading + re-parsing every cached HTML (the "warning then
nothing happens" stall). Verified with a tiny on-disk cache + a fake store.
"""
from __future__ import annotations

import types

import pandas as pd

from src.data_extract.utils.behavioral import fetch_earnings_calls as fe

_PREP = ("Good morning and welcome to the call. Revenue grew and margins expanded across every "
         "region this quarter, with strong momentum into the next period. " * 4)
_QA = ("Analyst: How is demand trending next quarter? CEO: Demand is strong and pricing held up "
       "well across the whole portfolio and we expect that to continue. " * 4)
_HTML = (f'<html><body><div class="transcript-content">\nCALL PARTICIPANTS\n'
         f'Jane Doe -- Chief Executive Officer\nOperator\n{_PREP}\nQuestions and Answers\n{_QA}\n'
         f'</div></body></html>')


class _FakeStore:
    """Records save() calls and serves load(earnings_call_sections) from an in-memory frame."""
    def __init__(self, existing: pd.DataFrame):
        self._existing = existing
        self.saved: list[pd.DataFrame] = []

    def load(self, table, columns=None):
        df = self._existing
        return df[columns] if (columns and not df.empty) else df

    def save(self, table, df):
        self.saved.append(df)
        return len(df)


def _seed_cache(tmp_path, pairs):
    cache = tmp_path / "call_transcripts"
    for tkr, q in pairs:
        d = cache / tkr
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{q}.html").write_text(_HTML, encoding="utf-8")
    return cache


def _ctx(tmp_path, existing_keys):
    existing = pd.DataFrame(existing_keys, columns=["ticker", "quarter"]) if existing_keys \
        else pd.DataFrame(columns=["ticker", "quarter"])
    store = _FakeStore(existing)
    return types.SimpleNamespace(store=store, paths={"DATA_STORE": tmp_path})


def test_ingest_skips_already_ingested(tmp_path):
    # cache has 3 transcripts; 2 already in the DB -> only the 1 NEW one is parsed + saved
    _seed_cache(tmp_path, [("AAA", "2025Q1"), ("AAA", "2025Q2"), ("BBB", "2025Q1")])
    ctx = _ctx(tmp_path, existing_keys=[("AAA", "2025Q1"), ("AAA", "2025Q2")])

    saved = fe.ingest_earnings_calls(ctx)

    assert saved > 0, "the one new transcript should have produced sections"
    assert len(ctx.store.saved) == 1
    got = set(map(tuple, ctx.store.saved[0][["ticker", "quarter"]].drop_duplicates().to_numpy()))
    assert got == {("BBB", "2025Q1")}, f"only the NEW (ticker,quarter) should be ingested, got {got}"

    # re-run: everything now already ingested -> NO parse, NO save, returns 0 (the stall is gone)
    ctx2 = _ctx(tmp_path, existing_keys=[("AAA", "2025Q1"), ("AAA", "2025Q2"), ("BBB", "2025Q1")])
    assert fe.ingest_earnings_calls(ctx2) == 0
    assert ctx2.store.saved == [], "a fully-ingested cache must not re-parse or re-save"

    # force=True re-parses everything even when present
    ctx3 = _ctx(tmp_path, existing_keys=[("AAA", "2025Q1"), ("AAA", "2025Q2"), ("BBB", "2025Q1")])
    assert fe.ingest_earnings_calls(ctx3, force=True) > 0
    forced = set(map(tuple, ctx3.store.saved[0][["ticker", "quarter"]].drop_duplicates().to_numpy()))
    assert forced == {("AAA", "2025Q1"), ("AAA", "2025Q2"), ("BBB", "2025Q1")}

    print("\n=== SANITY CHECK: incremental MF ingest ===")
    print(f"  3 cached, 2 already in DB -> ingested only {sorted(got)} (1 new)")
    print("  re-run with all present -> 0 saved, no re-parse (no more 'nothing happens' stall)")
    print(f"  force=True -> re-ingests all {len(forced)}. Validated.")


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    test_ingest_skips_already_ingested(Path(tempfile.mkdtemp()))
