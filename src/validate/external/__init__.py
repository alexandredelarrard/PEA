"""
Adapters for EXTERNAL fundamentals sources -- Tier 4 of the validator's tier stack.

TIER 4 IS DEFERRED (plan-5b decision 51), and `yahoo_comparison.py` is here because decision
62 says validation code lives under `src/validate/`, not because Tier 4 is being built. It
has ZERO Python importers outside its own test; its only entry point is its `__main__`.

Why deferred, in one number: Boritz & No measure aggregators disagreeing with the filed 10-K
at **6.5-7.7%**, roughly 10x the effect sizes Tier 3 measures off the filer's own disjoint
evidence. A tier whose noise floor sits above the signal it is meant to check cannot settle a
finding, and this is the only tier with a network cost.

REVISIT TRIGGER: Phase 9 finds wrong values Tiers 1-3 cannot explain, **on a field where the
aggregator's basis is known to match ours** -- which is what `classify_bucket`'s a/b/c
partition already records.

The plan also asks for the adapter to be cut back to fetch-only, with its ranking / bucketing
/ verdict logic moved into the validator. NOT DONE, deliberately: there is no Tier 4 to move
it into while Tier 4 is deferred, so the cut-back would delete the only implementation of
logic a future Tier 4 would have to re-derive. It is recorded as pending on Tier 4's adoption
rather than performed now.

⚠ WHAT THIS PACKAGE CANNOT DO. `yfinance` exposes only ~4-5 trailing, CURRENT-RESTATED
quarters, so `yahoo_comparison` can check the most recent quarters and nothing else -- it can
never validate a historical AS-FILED value, which is the property `fundamentals_history_sec`
exists to guarantee. If Tier 4 is ever adopted for as-filed history, build it on
`fundamentals_sharadar`, which has real point-in-time depth, rather than on anything here.
"""
