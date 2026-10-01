# Independent correctness review

## Scope and finding

The reviewer inspected the FTD extraction, feature, step, schema, and test diff. One P2 issue was found: for a partial-universe cube build, selecting the latest ZIP from ticker-filtered rows could treat a freshly cached older ZIP as the globally latest ZIP. This could move historical FTD availability to today.

## Resolution and re-review

`StepCubeInstitutionals` now reads the global distinct `period` values from `sec_fails_to_deliver` through `context.store` and passes them into the date derivation. The cache override applies only if the globally latest eligible period is present in the scoped rows. A known-truth test covers an MSFT-like `202608b` subset when another ticker has `202609a` and the older ZIP has a fresh mtime. It requires the September 15 historical date. The focused suite passed 41 tests after the fix.

The independent reviewer re-read the updated diff and found no remaining correctness issues. The only production caller of `build_short_flow_feature_panel` supplies the global-period handoff.

## Limits

The review establishes source-level and fixture-level behavior. The read-only live-data comparison and persisted cube checks are recorded separately in `04-validation-01.md`; existing cube rows were not rebuilt.
