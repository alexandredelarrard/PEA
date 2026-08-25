"""
Adapters for EXTERNAL fundamentals sources -- Tier 4 of the validator's tier stack.

TIER 4 IS DEFERRED (plan-5b decision 51), and these two modules are here because decision 62
says validation code lives under `src/validate/`, not because Tier 4 is being built. They
moved from `src/utils/`, where they had ZERO Python importers and were referenced only by
comment blocks in `constants.py`.

Why deferred, in one number: Boritz & No measure aggregators disagreeing with the filed 10-K
at **6.5-7.7%**, roughly 10x the effect sizes Tier 3 measures off the filer's own disjoint
evidence. A tier whose noise floor sits above the signal it is meant to check cannot settle a
finding, and this is the only tier with a network cost and a paid dependency.

REVISIT TRIGGER: Phase 9 finds wrong values Tiers 1-3 cannot explain, **on a field where the
aggregator's basis is known to match ours** -- which is what `classify_bucket`'s a/b/c
partition in each module already records.

The plan also asks for both modules to be cut back to fetch-only adapters, with their
ranking / bucketing / verdict logic moved into the validator. NOT DONE, deliberately: there
is no Tier 4 to move it into while Tier 4 is deferred, so the cut-back would delete the only
implementation of logic a future Tier 4 would have to re-derive. It is recorded as pending on
Tier 4's adoption rather than performed now.
"""
