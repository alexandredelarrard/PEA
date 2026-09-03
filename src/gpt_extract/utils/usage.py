"""
usage.py  (src/gpt_extract/utils/usage.py)
------------------------------------------
Token accounting for an extraction run: what was sent, what came back, and what it cost.

The API reports token counts per response and nothing captured them for a long time, so a
backfill's real bill could only be estimated after the fact.
"""
from __future__ import annotations

import threading
from typing import Mapping

#: gpt-5-mini list price per 1M tokens: fresh input / cached input / output.
PRICE_IN, PRICE_CACHED_IN, PRICE_OUT = 0.25, 0.025, 2.00

DEFAULT_PRICES: dict[str, float] = {
    "input": PRICE_IN, "cached_input": PRICE_CACHED_IN, "output": PRICE_OUT,
}


class UsageTracker:
    """Token counts for one run, safe to share across a thread pool.

    ONE tracker is deliberately shared by every worker: `prompt_cache_key` only pays off
    while every call keeps hitting the same cached prefix, so the pool's calls are one
    accounting unit. `totals` is a read-modify-write from every worker, and without the
    lock the call count silently under-reports under concurrency.

    `cached_input_tokens` is what makes `prompt_cache_key` measurable rather than assumed.
    Cached input is 10x cheaper, so a run whose cached share is zero costs roughly 10x what
    its budget assumed -- which is only visible if the number is recorded.
    """

    def __init__(self) -> None:
        self.last: dict[str, int] | None = None
        self.totals: dict[str, int] = {
            "calls": 0, "input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0,
        }
        self._lock = threading.Lock()

    def record(self, usage: Mapping[str, int] | None) -> None:
        """Fold one call's counts into `last` and `totals`."""
        if usage is None:
            self.last = None
            return
        counts = {
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "cached_input_tokens": int(usage.get("cached_input_tokens", 0) or 0),
        }
        with self._lock:
            self.last = counts
            self.totals["calls"] += 1
            for key, value in counts.items():
                self.totals[key] += value

    def spend_estimate(self, prices: Mapping[str, float] | None = None) -> float:
        """Dollars for the run so far, so it reports its own bill instead of being
        estimated afterwards.

        `cached_input_tokens` is a SUBSET of `input_tokens`, not a separate bucket, so the
        fresh-input count is the difference.
        """
        p = dict(DEFAULT_PRICES) | dict(prices or {})
        t = self.totals
        fresh = max(t["input_tokens"] - t["cached_input_tokens"], 0)
        return (fresh * p["input"]
                + t["cached_input_tokens"] * p["cached_input"]
                + t["output_tokens"] * p["output"]) / 1e6

    @property
    def cached_share(self) -> float:
        """Fraction of input tokens served from the cached prefix. Zero after the first
        call means `prompt_cache_key` stopped being sent."""
        total_in = self.totals["input_tokens"]
        return self.totals["cached_input_tokens"] / total_in if total_in else 0.0
