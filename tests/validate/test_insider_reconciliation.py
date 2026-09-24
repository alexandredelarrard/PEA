"""Completed-quarter EDGAR replay orchestration."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.validate import insider_reconciliation as module


def test_replay_initializes_edgar_and_returns_a_typed_empty_frame(monkeypatch):
    state = {"identity_initialized": False}
    context = SimpleNamespace(ensure_edgar_identity=lambda: state.__setitem__("identity_initialized", True))
    bulk = pd.DataFrame(
        {
            "accession_number": ["0000000001-26-000001"],
            "ticker": ["AAA"],
        }
    )
    monkeypatch.setattr(
        module,
        "load_cik_mapping",
        lambda context, tickers: pd.DataFrame({"ticker": ["AAA"], "cik": ["1"]}),
    )
    monkeypatch.setattr(module, "load_identity", lambda context: object())
    monkeypatch.setattr(module, "insider_filings", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        module,
        "run_per_ticker",
        lambda cik_map, worker, **kwargs: [worker("AAA", "1")],
    )

    live, diagnostics = module.replay_completed_quarter(
        context,
        pd.Period("2026Q2", freq="Q"),
        bulk,
    )

    assert state["identity_initialized"]
    assert live.empty
    assert "accession_number" in live.columns
    assert diagnostics["listing_coverage"] == 0.0
    print(
        "SANITY: the replay initializes the SEC identity before listing and represents a "
        "zero-row result with the full typed contract, so the gate fails visibly instead "
        "of crashing on a missing accession column."
    )
