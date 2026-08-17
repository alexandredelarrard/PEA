---
name: auditing-fundamentals
description: Audits SEC XBRL fundamentals extraction for coverage and consistency
  across tickers, quarters and KPIs. Use when checking whether fundamentals are
  correctly extracted, investigating a missing or implausible financial value,
  or triaging fundamentals_facts / fundamentals_history data quality.
---

# Auditing fundamentals

## Workflow
1. Scan: `"$PY" scripts/audit_coverage.py --tickers JPM,MAA`
2. Triage: read the TOP 20 rows of data/audit/findings_ranked.csv. Never the full matrix.
3. Verify: `"$PY" scripts/audit_verify_cell.py JPM 2023Q4 totalRevenue`
...

Tag resolution rules: see [references/tag_resolution.md](references/tag_resolution.md)
