"""
StepExtractFundamentals wiring test: `run()` calls its sources directly, in
order, with no per-source error isolation -- a failure in any one of them
propagates immediately (no _run_source wrapper). `fetch_financial_statements`
is no longer imported by this module, so it is never called (not exercised
here); the active sequence is earnings_surprises -> insider_transactions ->
financial_notes.

Phase 3 of the fundamentals rebuild (reports/planning/active-tasks/
2026-08-21-fundamentals-rebuild-plan.md) put `fetch_fundamentals_sec` back at the
HEAD of that sequence -- the facts layer must land before anything derived from it.
The `fundamentals_history` build follows at Phase 5, immediately after it.
`EXPECTED_SOURCES` below is the one place this test records the order.
"""
from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from src.data_extract.transformers.step_extract_fundamentals import StepExtractFundamentals

#: (module attribute, label) for every source `run()` is expected to call, in call order.
EXPECTED_SOURCES: tuple[tuple[str, str], ...] = (
    ("fetch_fundamentals_sec", "fundamentals_sec"),
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
    # `run()` reads years_history off the config to size the EDGAR listing window.
    step._config = OmegaConf.create({"data_extract": {"years_history": 15}})
    return step


def test_run_calls_its_active_sources_directly_in_order(monkeypatch):
    calls: list[str] = []
    _patched_step(monkeypatch, calls).run(tickers=["AAPL"])

    # "financial_statements" is NOT in this list -- fetch_financial_statements
    # is no longer imported/called by run().
    assert calls == [label for _, label in EXPECTED_SOURCES]
    assert calls[0] == "fundamentals_sec", (
        "the facts layer must run FIRST -- everything else in this step is either "
        "independent of it or, from Phase 5, derived from it")
    print("\n=== SANITY CHECK: StepExtractFundamentals call order ===")
    print(f"  {' -> '.join(calls)}")
    print("  OK: facts layer first, no per-source error isolation.")


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
