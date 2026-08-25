"""
The fundamentals validator: `fundamentals_facts` / `fundamentals_history` -> a ranked list of
FIXABLE ISSUES, written to `fundamentals_check` and read back out of it.

    validator.py   FundamentalsValidator -- the ONE implementation that judges a value
    scope.py       RunScope -- what a run covered, hashed, so two runs can be differenced
    substrate.py   Substrates -- every frame a CHECK reads, loaded once, projected
    finding.py     the investigation packet, the cross-run `finding_id`, the `cluster_id`
    ledger.py      the only read-back of the three validator tables; comparable runs
    clusters.py    findings -> clusters -> field families, scored and routed
    report.py      the check-health gate, the delta, the ranked clusters, the wontfix footer
    checks/        CHECK_REGISTRY + the three tier modules

## The shape of the loop, in one paragraph

A run writes every finding it makes -- nothing is subtracted -- plus one row per check saying
what it examined and at what scope. A later run of the SAME scope can therefore be differenced
against it, and the row-count drop is what proves a fix worked. Findings are grouped into
`(ticker, field)` CLUSTERS for ranking, because 11,926 findings were never 11,926 bugs; a
cluster is one thing to fix and the checks that fired on it are the evidence that it is real.

The JSON register that used to subtract settled findings is gone. A human's only remaining
assertion is a `wontfix` in `fundamentals_check_status`, which is applied when the report is
RENDERED rather than when a row is written, and which reopens by itself if the cluster grows.

Read `../README.md` first. Domain-scoped from day one so a future prices or insider validator
has an obvious home and this does not become a fundamentals package with a generic name.
"""
