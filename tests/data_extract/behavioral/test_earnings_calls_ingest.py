"""
Incremental MF ingest (src/data_extract/utils/behavioral/fetch_earnings_calls.py::ingest_earnings_calls).

The DAG ingest step re-runs daily on a full transcript cache; it must SKIP (ticker, quarter) already
in earnings_call_sections instead of re-reading + re-parsing every cached HTML (the "warning then
nothing happens" stall). Verified with a tiny on-disk cache + a fake store.
"""

from __future__ import annotations

import types
from typing import Any

import pandas as pd

from src.constants.constants import EARNINGS_CALL_SENTIMENT_INVALID_PENDING_MODEL
from src.data_extract.utils.behavioral import fetch_earnings_calls as fe
from src.data_extract.utils.behavioral.utils_earnings_call_cache import save_earnings_call_sections
from src.data_store.schema import Tables
from src.utils.text_metrics import assess_earnings_call_sections
from tests.conftest import FakeStore  # the ONE shared store double -- ABSOLUTE, see its docstring

_PREP = (
    "Good morning and welcome to the call. Revenue grew and margins expanded across every "
    "region this quarter, with strong momentum into the next period. " * 4
)
_QA = (
    "Analyst: How is demand trending next quarter? CEO: Demand is strong and pricing held up "
    "well across the whole portfolio and we expect that to continue. " * 4
)
_HTML = (
    f'<html><body><div class="transcript-content">\nCALL PARTICIPANTS\n'
    f"Jane Doe -- Chief Executive Officer\nOperator\n{_PREP}\nQuestions and Answers\n{_QA}\n"
    f"</div></body></html>"
)


def _seed_cache(tmp_path, pairs):
    cache = tmp_path / "call_transcripts"
    for tkr, q in pairs:
        d = cache / tkr
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{q}.html").write_text(_HTML, encoding="utf-8")
    return cache


def _ctx(tmp_path, existing_keys) -> Any:
    existing = pd.DataFrame(
        [
            {"ticker": ticker, "quarter": quarter, "tag": tag, "as_of": "2025-05-01", "text": text}
            for ticker, quarter in existing_keys
            for tag, text in (("prepared_remarks", _PREP), ("qa", _QA))
        ]
    )
    store = FakeStore({"earnings_call_sections": existing} if existing_keys else {})
    # `run_manifest._manifest_path` reads `config.local.filename.extraction`
    # (value from configs/paths.yml), so the double has to carry it.
    return types.SimpleNamespace(
        store=store,
        paths={"DATA_STORE": tmp_path},
        config=types.SimpleNamespace(
            local=types.SimpleNamespace(
                filename=types.SimpleNamespace(extraction="extraction_manifest.json"),
                paths=types.SimpleNamespace(call_transcripts="call_transcripts"),
            )
        ),
    )


def test_ingest_skips_already_ingested(tmp_path):
    # cache has 3 transcripts; 2 already in the DB -> only the 1 NEW one is parsed + saved
    _seed_cache(tmp_path, [("AAA", "2025Q1"), ("AAA", "2025Q2"), ("BBB", "2025Q1")])
    ctx = _ctx(tmp_path, existing_keys=[("AAA", "2025Q1"), ("AAA", "2025Q2")])

    saved = fe.ingest_earnings_calls(ctx)

    assert saved > 0, "the one new transcript should have produced sections"
    assert len(ctx.store.saved_frames()) == 1
    got = set(map(tuple, ctx.store.saved_frames()[0][["ticker", "quarter"]].drop_duplicates().to_numpy()))
    assert got == {("BBB", "2025Q1")}, f"only the NEW (ticker,quarter) should be ingested, got {got}"

    # re-run: everything now already ingested -> NO parse, NO save, returns 0 (the stall is gone)
    ctx2 = _ctx(tmp_path, existing_keys=[("AAA", "2025Q1"), ("AAA", "2025Q2"), ("BBB", "2025Q1")])
    assert fe.ingest_earnings_calls(ctx2) == 0
    assert ctx2.store.saved_frames() == [], "a fully-ingested cache must not re-parse or re-save"

    # force=True re-parses everything even when present
    ctx3 = _ctx(tmp_path, existing_keys=[("AAA", "2025Q1"), ("AAA", "2025Q2"), ("BBB", "2025Q1")])
    assert fe.ingest_earnings_calls(ctx3, force=True) > 0
    forced = set(map(tuple, ctx3.store.saved_frames()[0][["ticker", "quarter"]].drop_duplicates().to_numpy()))
    assert forced == {("AAA", "2025Q1"), ("AAA", "2025Q2"), ("BBB", "2025Q1")}

    print("\n=== SANITY CHECK: incremental MF ingest ===")
    print(f"  3 cached, 2 already in DB -> ingested only {sorted(got)} (1 new)")
    print("  re-run with all present -> 0 saved, no re-parse (no more 'nothing happens' stall)")
    print(f"  force=True -> re-ingests all {len(forced)}. Validated.")


def test_source_replacement_invalidates_sentiment_and_embedding_caches() -> None:
    cached = pd.DataFrame({"ticker": ["AAA", "AAA"], "quarter": ["2025Q1", "2025Q2"], "tag": ["qa", "qa"]})
    embeddings = pd.DataFrame({"ticker": ["AAA", "AAA"], "quarter": ["2025Q1", "2025Q2"], "section": ["qa", "qa"], "turn_index": [0, 0]})
    store = FakeStore({Tables.earnings_call_sentiment: cached, Tables.earning_calls_embedding: embeddings})
    context = types.SimpleNamespace(store=store)
    replacement = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "quarter": ["2025Q1", "2025Q1"],
            "tag": ["prepared_remarks", "qa"],
            "text": [_PREP, _QA],
        }
    )

    save_earnings_call_sections(context, replacement)

    assert set(store.t[Tables.earnings_call_sentiment.name]["quarter"]) == {"2025Q2"}
    assert set(store.t[Tables.earning_calls_embedding.name]["quarter"]) == {"2025Q2"}
    print("\n=== SANITY CHECK: transcript replacement invalidates derivatives ===")
    print("  replacing AAA 2025Q1 deletes only that call's sentiment and embedding rows. Validated.")


def test_forced_malformed_refresh_replaces_old_signal_with_null_marker(tmp_path) -> None:
    cache = _seed_cache(tmp_path, [("AAA", "2025Q1")])
    (cache / "AAA" / "2025Q1.html").write_text(
        '<html><body><div class="transcript-content">Thanks.</div></body></html>',
        encoding="utf-8",
    )
    context = _ctx(tmp_path, existing_keys=[("AAA", "2025Q1")])
    stale = pd.DataFrame({"ticker": ["AAA"], "quarter": ["2025Q1"], "tag": ["qa"]})
    context.store.t[Tables.earnings_call_sentiment.name] = stale.copy()
    context.store.t[Tables.earning_calls_embedding.name] = stale.assign(section="qa", turn_index=0)

    saved = fe.ingest_earnings_calls(context, force=True)

    current = context.store.t[Tables.earnings_call_sections.name]
    sections = dict(zip(current["tag"], current["text"], strict=False))
    assert saved == 2
    assert not assess_earnings_call_sections(sections).valid
    marker = context.store.t[Tables.earnings_call_sentiment.name]
    assert len(marker) == 2
    assert set(marker["tag"]) == {"prepared_remarks", "qa"}
    assert set(marker["model"]) == {EARNINGS_CALL_SENTIMENT_INVALID_PENDING_MODEL}
    assert set(pd.to_datetime(marker["as_of"])) == {pd.Timestamp("2025-05-01")}
    assert marker[["sent_pos", "sent_neg", "sent_neu"]].isna().all().all()
    assert context.store.t[Tables.earning_calls_embedding.name].empty
    print("\n=== SANITY CHECK: malformed forced refresh ===")
    print("  refreshed malformed HTML clears embeddings and leaves a pending null marker that forces historical cube repair. Validated.")


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    test_ingest_skips_already_ingested(Path(tempfile.mkdtemp()))
