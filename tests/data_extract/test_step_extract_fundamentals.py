"""
StepExtractFundamentals wiring test: `run()` calls its sources directly, in order, with no
per-source error isolation -- a failure in any one of them propagates immediately (no
`_run_source` wrapper).

The ACTIVE sequence is `earnings_surprises -> financial_notes -> financial_statements`.

⚠ TWO SOURCES ARE IMPORTED BUT CURRENTLY DORMANT. `fetch_fundamentals_sec` (the per-filing
XBRL facts walk) and `build_fundamentals_history` (the publication-event replay over exactly
those facts) are commented out in `run()` while `fundamentals_history` is served Sharadar-first
and `fundamentals_history_sec` is being rebuilt separately. They are still module attributes,
so this test patches them and asserts they are NOT called -- and `test_the_facts_walk_and_its_replay_stay_adjacent`
re-arms the ordering invariant automatically the moment either is re-enabled, rather than
leaving the rule unguarded until someone remembers it.

`fetch_insider_transactions` is NOT part of this step: it moved to `StepExtractPrices`
(`src/data_extract/utils/prices/fetch_insider_transactions.py`), alongside the other
price/settlement signals.
"""
from __future__ import annotations

import logging

import pytest
from omegaconf import OmegaConf

from src.data_extract.transformers.step_extract_fundamentals import StepExtractFundamentals

#: (module attribute, label) for every source `run()` actually calls, in call order.
EXPECTED_SOURCES: tuple[tuple[str, str], ...] = (
    ("fetch_earnings_surprises", "earnings_surprises"),
    ("fetch_financial_notes", "financial_notes"),
    ("fetch_financial_statements", "financial_statements"),
)

#: Imported and patchable, but not called by `run()` today. Kept in the test so that
#: re-enabling one is a VISIBLE change here rather than a silent one.
DORMANT_SOURCES: tuple[tuple[str, str], ...] = (
    ("fetch_fundamentals_sec", "fundamentals_sec"),
    ("build_fundamentals_history", "fundamentals_history"),
)

ALL_SOURCES = EXPECTED_SOURCES + DORMANT_SOURCES


def _patched_step(monkeypatch, calls: list[str], *, boom: str | None = None):
    """A `StepExtractFundamentals` with every source replaced by a recorder.

    `boom` names the label whose recorder raises after recording, for the no-isolation test."""
    import src.data_extract.transformers.step_extract_fundamentals as mod

    def _recorder(label: str):
        def _fn(*args, **kwargs):
            calls.append(label)
            if label == boom:
                raise RuntimeError("simulated fundamentals fetch failure")
        return _fn

    for attr, label in ALL_SOURCES:
        monkeypatch.setattr(mod, attr, _recorder(label))

    context = object.__new__(object)   # not touched: every dependency is monkeypatched
    step = object.__new__(StepExtractFundamentals)
    step._context = context
    # `run()` reads years_history off the config to size the EDGAR listing window.
    # `_config`, not `config`: that is the attribute `Step.__init__` sets, and reading
    # `self.config` here was a real AttributeError on every live run of this step.
    step._config = OmegaConf.create({"data_extract": {"years_history": 15}})
    # `__init__` is bypassed, so the logger `_stage` writes its per-fetcher timings to has to
    # be supplied by hand.
    step._log = logging.getLogger("test")
    return step


def test_run_calls_its_active_sources_directly_in_order(monkeypatch):
    calls: list[str] = []
    _patched_step(monkeypatch, calls).run(tickers=["AAPL"])

    assert calls == [label for _, label in EXPECTED_SOURCES]
    for _, dormant in DORMANT_SOURCES:
        assert dormant not in calls, (
            f"{dormant} is commented out in run() -- if it was deliberately re-enabled, move "
            f"it from DORMANT_SOURCES to EXPECTED_SOURCES so the order is pinned again")

    print("\n=== SANITY CHECK: StepExtractFundamentals call order ===")
    print(f"  active : {' -> '.join(calls)}")
    print(f"  dormant: {', '.join(label for _, label in DORMANT_SOURCES)} "
          f"(imported, commented out in run())")
    print("  OK: direct calls, no per-source error isolation.")


def test_the_facts_walk_and_its_replay_stay_adjacent(monkeypatch):
    """The invariant that outlives the current wiring, so it cannot be lost while dormant.

    `build_fundamentals_history` replays exactly the facts `fetch_fundamentals_sec` just
    stored, so it must run IMMEDIATELY after it -- never later in the sequence and never on
    its own schedule. A run that fetched a 10-K without replaying it would leave the newest
    filing invisible to every consumer of `fundamentals_history`.

    While both are commented out this asserts the vacuous case; the moment either is
    re-enabled it becomes a real ordering check with no edit required."""
    calls: list[str] = []
    _patched_step(monkeypatch, calls).run(tickers=["AAPL"])

    if "fundamentals_sec" not in calls:
        assert "fundamentals_history" not in calls, (
            "the history replay must never run without the facts walk that feeds it")
        print("\n  facts walk dormant -> replay correctly absent too (invariant vacuous)")
        return

    assert calls[0] == "fundamentals_sec", (
        "the facts layer must run FIRST -- everything else in this step is either "
        "independent of it or derived from it")
    assert calls[1] == "fundamentals_history", (
        "the history replay must run IMMEDIATELY after the facts walk, not later in the "
        "sequence: it reads exactly what that walk stored")
    print(f"\n  facts walk active -> adjacency held: {calls[0]} -> {calls[1]}")


def test_a_failing_source_aborts_the_rest(monkeypatch):
    """No per-source try/except -- a raised exception propagates and the remaining sources
    never run."""
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
