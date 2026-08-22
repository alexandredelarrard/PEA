"""
StepExtractFundamentals wiring test: `run()` calls its sources directly, in
order, with no per-source error isolation -- a failure in any one of them
propagates immediately (no _run_source wrapper). `fetch_financial_statements`
is no longer imported by this module, so it is never called (not exercised
here); the active sequence is earnings_surprises -> insider_transactions ->
financial_notes.

The fundamentals fetch and history build are absent from that sequence while the
fundamentals stack is rebuilt (reports/planning/active-tasks/
2026-08-21-fundamentals-rebuild-plan.md); Phase 5 re-adds them at the head of the
list, and `EXPECTED_SOURCES` below is the one place this test records the order.
"""
from __future__ import annotations

import pytest

from src.data_extract.transformers.step_extract_fundamentals import StepExtractFundamentals

#: (module attribute, label) for every source `run()` is expected to call, in call order.
EXPECTED_SOURCES: tuple[tuple[str, str], ...] = (
    ("fetch_earnings_surprises", "earnings_surprises"),
    ("fetch_insider_transactions", "insider_transactions"),
    ("fetch_financial_notes", "financial_notes"),
)


def _patched_step(monkeypatch, calls: list[str], *, boom: str | None = None):
    """A `StepExtractFundamentals` with every source replaced by a recorder.

    `boom` names the label whose recorder raises after recording, for the
    no-isolation test."""
    import src.data_extract.transformers.step_extract_fundamentals as mod

    def _recorder(label: str):
        def _fn(*args, **kwargs):
            calls.append(label)
            if label == boom:
                raise RuntimeError("simulated fundamentals fetch failure")
        return _fn

    for attr, label in EXPECTED_SOURCES:
        monkeypatch.setattr(mod, attr, _recorder(label))

    context = object.__new__(object)   # not touched: every dependency is monkeypatched
    step = object.__new__(StepExtractFundamentals)
    step._context = context
    return step


def test_run_calls_its_active_sources_directly_in_order(monkeypatch):
    calls: list[str] = []
    _patched_step(monkeypatch, calls).run(tickers=["AAPL"])

    # "financial_statements" is NOT in this list -- fetch_financial_statements
    # is no longer imported/called by run().
    assert calls == [label for _, label in EXPECTED_SOURCES]


def test_a_failing_source_aborts_the_rest(monkeypatch):
    """No per-source try/except anymore -- a raised exception propagates and the
    remaining sources never run."""
    calls: list[str] = []
    first = EXPECTED_SOURCES[0][1]
    step = _patched_step(monkeypatch, calls, boom=first)

    with pytest.raises(RuntimeError, match="simulated fundamentals fetch failure"):
        step.run(tickers=["AAPL"])

    assert calls == [first]

    print("\n=== SANITY CHECK: StepExtractFundamentals direct calls, no isolation ===")
    print(f"  run() calls {len(EXPECTED_SOURCES)} active sources directly in order "
          f"({', '.join(label for _, label in EXPECTED_SOURCES)}); no _run_source wrapper.")
    print(f"  A failure in the first source ({first}) propagates and the rest never run.")
    print("  Validated.")
