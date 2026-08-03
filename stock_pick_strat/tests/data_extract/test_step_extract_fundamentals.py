"""
StepExtractFundamentals wiring test: `run()` calls its sources directly, in
order, with no per-source error isolation -- a failure in any one of them
propagates immediately (no _run_source wrapper). `fetch_financial_statements`
is no longer imported by this module, so it is never called (not exercised
here); the active sequence is fundamentals -> fundamentals_derive ->
earnings_surprises -> insider_transactions -> financial_notes.
"""
from __future__ import annotations

from src.data_extract.transformers.step_extract_fundamentals import StepExtractFundamentals


def test_run_calls_its_active_sources_directly_in_order(monkeypatch):
    calls: list[str] = []

    def _ok(name):
        def _fn(*args, **kwargs):
            calls.append(name)
        return _fn

    import src.data_extract.transformers.step_extract_fundamentals as mod
    monkeypatch.setattr(mod, "fetch_fundamentals_edgartools", _ok("fundamentals"))
    monkeypatch.setattr(mod, "rebuild_fundamentals_history", _ok("fundamentals_derive"))
    monkeypatch.setattr(mod, "fetch_earnings_surprises", _ok("earnings_surprises"))
    monkeypatch.setattr(mod, "fetch_insider_transactions", _ok("insider_transactions"))
    monkeypatch.setattr(mod, "fetch_financial_notes", _ok("financial_notes"))

    context = object.__new__(object)   # not touched: every dependency is monkeypatched
    step = object.__new__(StepExtractFundamentals)
    step._context = context

    step.run(tickers=["AAPL"])

    # "financial_statements" is NOT in this list -- fetch_financial_statements
    # is no longer imported/called by run().
    assert calls == ["fundamentals", "fundamentals_derive",
                     "earnings_surprises", "insider_transactions", "financial_notes"]


def test_a_failing_source_aborts_the_rest(monkeypatch):
    """No per-source try/except anymore -- a raised exception propagates and the
    remaining sources never run."""
    calls: list[str] = []

    def _ok(name):
        def _fn(*args, **kwargs):
            calls.append(name)
        return _fn

    def _boom(*args, **kwargs):
        calls.append("fundamentals")
        raise RuntimeError("simulated fundamentals fetch failure")

    import src.data_extract.transformers.step_extract_fundamentals as mod
    monkeypatch.setattr(mod, "fetch_fundamentals_edgartools", _boom)
    monkeypatch.setattr(mod, "rebuild_fundamentals_history", _ok("fundamentals_derive"))
    monkeypatch.setattr(mod, "fetch_earnings_surprises", _ok("earnings_surprises"))
    monkeypatch.setattr(mod, "fetch_insider_transactions", _ok("insider_transactions"))
    monkeypatch.setattr(mod, "fetch_financial_notes", _ok("financial_notes"))

    context = object.__new__(object)
    step = object.__new__(StepExtractFundamentals)
    step._context = context

    import pytest
    with pytest.raises(RuntimeError, match="simulated fundamentals fetch failure"):
        step.run(tickers=["AAPL"])

    assert calls == ["fundamentals"]

    print("\n=== SANITY CHECK: StepExtractFundamentals direct calls, no isolation ===")
    print("  run() calls its active sources directly in order (no _run_source wrapper); a")
    print("  failure in one source propagates immediately and the remaining sources never run.")
    print("  Validated.")
