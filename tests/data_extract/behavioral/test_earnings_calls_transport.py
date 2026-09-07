"""
Transport resilience of the earnings-call DOWNLOAD stage
(fetch_hf_transcripts._stream_download + fetch_earnings_calls.download_earnings_calls).

Two failure modes that cost the whole recent gap, both reproduced here without a network:
  1. curl_cffi's streamed `Response` has NO context-manager protocol, so streaming it through a
     `with` block raises TypeError before a single byte lands -> the 1.8 GB HF backbone download
     dies on its fallback path.
  2. The HF backbone is deep HISTORY, but it is fetched FIRST; any exception there used to
     propagate and abort the whole step, so ROIC and Motley Fool -- the only sources for the
     RECENT quarters the live features need -- never ran at all.
"""
from __future__ import annotations

import types

import pytest

from src.data_extract.utils.behavioral import fetch_hf_transcripts as hf
from src.data_extract.utils.behavioral import fetch_earnings_calls as fec


class _NoCtxStreamResponse:
    """curl_cffi 0.15.0's streamed response: has .iter_content/.raise_for_status/.close,
    but deliberately NO __enter__/__exit__ (that is the bug under test)."""

    def __init__(self, chunks):
        self._chunks, self.closed, self.status_code = chunks, False, 200

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield from self._chunks

    def close(self):
        self.closed = True


def test_stream_download_survives_curl_response_without_context_manager(tmp_path, monkeypatch):
    """requests fails -> the curl_cffi fallback must still write the file and close the
    response, WITHOUT relying on `with`."""
    payload = [b"PAR1-header", b"row-group-bytes", b"PAR1-footer"]
    resp = _NoCtxStreamResponse(payload)

    def _boom(*a, **k):
        raise OSError("requests path unavailable in this test")

    monkeypatch.setattr(hf, "corporate_session",
                        lambda: types.SimpleNamespace(get=_boom))
    monkeypatch.setattr(hf.cr, "get", lambda *a, **k: resp)

    dest = tmp_path / "hf.parquet"
    hf._stream_download("https://huggingface.co/whatever.parquet", dest)   # must NOT raise

    assert dest.read_bytes() == b"".join(payload), "fallback must stream every chunk to disk"
    assert resp.closed, "the streamed response must be closed explicitly (no context manager)"
    assert not hasattr(resp, "__enter__"), "guard: the stub reproduces the real curl_cffi shape"

    print("\n=== SANITY CHECK: _stream_download curl_cffi fallback ===")
    print(f"  requests raised -> curl_cffi fallback wrote {dest.stat().st_size} bytes and closed "
          "the response; no context-manager TypeError. Validated.")


def test_stream_download_prefers_requests_and_skips_the_fallback(tmp_path, monkeypatch):
    """The happy path stays on plain requests -- the fallback is not touched."""

    class _CtxResp(_NoCtxStreamResponse):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    called = {"curl": 0}
    monkeypatch.setattr(hf, "corporate_session",
                        lambda: types.SimpleNamespace(
                            get=lambda *a, **k: _CtxResp([b"abc", b"def"])))
    monkeypatch.setattr(hf.cr, "get",
                        lambda *a, **k: called.__setitem__("curl", called["curl"] + 1))

    dest = tmp_path / "hf.parquet"
    hf._stream_download("https://huggingface.co/whatever.parquet", dest)

    assert dest.read_bytes() == b"abcdef"
    assert called["curl"] == 0, "curl_cffi must not be called when requests succeeds"
    print("  happy path: streamed via requests, curl_cffi untouched. Validated.")


def test_hf_failure_does_not_stop_roic_and_fool(monkeypatch):
    """An HF download failure is logged and stepped over: ROIC and the Fool index still run,
    so the RECENT quarters are still collected."""
    ran: list[str] = []

    def _hf_boom(context, *a, **k):
        ran.append("hf")
        raise RuntimeError("HF parquet unreachable")

    monkeypatch.setattr(fec, "download_hf_parquet", _hf_boom)
    monkeypatch.setattr(fec, "missing_quarters_by_ticker",
                        lambda *a, **k: {"AIG": ["2026Q1", "2026Q2"]})

    def _roic(context, tickers=None, missing=None, since=None):
        ran.append("roic")
        assert missing == {"AIG": ["2026Q1", "2026Q2"]}, "ROIC must still receive the real gap"
        return types.SimpleNamespace(filled={"AIG": ["2026Q1"]})

    def _fool_index(context, tickers=None, missing=None, since=None):
        ran.append("fool")
        assert missing == {"AIG": ["2026Q2"]}, "fool must get only what ROIC left"

    monkeypatch.setattr(fec, "fetch_roic_transcripts", _roic)
    monkeypatch.setattr(fec, "build_transcript_index_by_ticker", _fool_index)
    monkeypatch.setattr(fec, "download_transcripts", lambda *a, **k: ran.append("download"))

    fec.download_earnings_calls(context=None, tickers=["AIG"])       # must NOT raise

    assert ran == ["hf", "roic", "fool", "download"], f"every stage must still run, got {ran}"

    print("\n=== SANITY CHECK: HF failure is non-fatal ===")
    print(f"  HF raised RuntimeError -> stages still executed in order: {ran}. The recent-gap "
          "sources (ROIC, fool) survive a dead HF backbone. Validated.")


def test_hf_ingest_failure_still_ingests_the_fool_html(monkeypatch):
    """Same isolation on the INGEST stage: `ingest_hf_transcripts` can itself hit the network
    (it downloads the parquet when the table doesn't span the backbone), and the recent
    quarters live in the fool HTML the second leg reads."""
    ran: list[str] = []

    def _hf_ingest_boom(context, tickers=None, force=False):
        ran.append("hf")
        raise RuntimeError("HF parquet unreachable")

    monkeypatch.setattr(fec, "ingest_hf_transcripts", _hf_ingest_boom)
    monkeypatch.setattr(fec, "ingest_earnings_calls",
                        lambda context, tickers=None, force=False: (ran.append("fool"), 42)[1])
    monkeypatch.setattr(fec, "record_run", lambda *a, **k: ran.append("record"))

    saved = fec.ingest_all_earnings_calls(context=None, tickers=["AIG"])

    assert ran == ["hf", "fool", "record"], f"the fool ingest must still run, got {ran}"
    assert saved == 42, "the fool leg's row count must still be returned/recorded"

    print("  ingest stage: HF ingest raised -> fool HTML still ingested (+42 rows) and the run "
          "still recorded. Validated.")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
