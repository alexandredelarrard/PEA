"""
StepExtractFundamentals wiring test. Both fundamentals layers (`fetch_fundamentals_sec`, the per-filing
XBRL facts walk, and `build_fundamentals_history`, its publication-event replay) are commented out in
`run()`; the `fundamentals` command runs them. Earnings surprises moved to
`StepExtractPrices.run_earnings_surprises` (`test_step_extract_prices.py`), the notes and statements zips
to `StepExtractZips`. `fetch_insider_transactions` is `StepExtractZips.run_ingest_insider`.
"""

from __future__ import annotations

import logging
from typing import Any

from omegaconf import OmegaConf

import src.data_extract.transformers.step_extract_fundamentals as mod
from src.data_extract.transformers.step_extract_fundamentals import StepExtractFundamentals


def _step() -> Any:
    """A `StepExtractFundamentals` without `Step.__init__`: `run()` reads only `_log`."""
    step: Any = object.__new__(StepExtractFundamentals)
    step._context = object()
    step._config = OmegaConf.create({"data_extract": {"years_history": 15}})
    step._log = logging.getLogger("test")
    return step


def test_earnings_surprises_left_this_step() -> None:
    """Earnings surprises are the prices step's now; `run()` imports no fetcher and only logs."""
    _step().run(tickers=["AAPL"])

    assert not hasattr(mod, "fetch_earnings_surprises"), "earnings surprises moved to StepExtractPrices"
    assert not hasattr(mod, "fetch_fundamentals_sec") and not hasattr(mod, "build_fundamentals_history")
    print("\n=== SANITY CHECK: StepExtractFundamentals after the move ===")
    print("  no fetcher is imported here; run() logs that both layers are dormant and returns. OK.")
