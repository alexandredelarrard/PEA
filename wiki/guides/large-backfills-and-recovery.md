---
title: Large backfills and recovery operations
description: Source-grounded procedures for fundamentals, registrant, ownership, insider, and validation recovery work.
type: guide
tags:
  - wiki
  - guide
  - operations
  - backfill
  - recovery
---

# Large backfills and recovery operations

## Goal

Run expensive or destructive historical repairs without defeating resume logic, mixing identity regimes, exhausting memory, or declaring success from row counts alone.

> [!CAUTION]
> These procedures can rewrite large tables or spend hours on SEC traffic. Inspect command help, take a recoverable table-level backup, record a before snapshot, and run one SEC network walk at a time.

## Shared operating rules

- Run from the repository root with the explicit Python 3.13 virtual-environment executable described in [run the pipeline](./run-the-pipeline.md).
- Prefix every shell command with `rtk`.
- Chunk edgartools workloads into separate processes. Process exit is the reliable way to release its per-filing caches.
- Use `-F`, `--rebuild`, `--rebuild-history`, or `--reparse` only for the behavior named by that command. They are not interchangeable.
- Stop a long-running process by PID. Never terminate every `python.exe` process.
- Capture a before/after artifact and run validation. A larger table can still contain a coverage cliff, duplicated identity history, or stale cube rows.

## Fundamentals resolution sweep

The committed [resolution sweep](../../scripts/sweep_fundamentals_resolution.py) pays the EDGAR/XBRL cost once per filing and writes two resolution variants into one Parquet ledger per ticker. The offline [sweep report](../../scripts/report_fundamentals_sweep.py) then measures route mix, before/after agreement, duplicate facts, regime coverage, debt basis, form coverage, and annual footing.

~~~bash
rtk "$PY" scripts/sweep_fundamentals_resolution.py --roster both --workers 4 --limit 4
rtk "$PY" scripts/report_fundamentals_sweep.py --roster both
~~~

Repeat the first command until every roster ticker has a ledger. Existing ticker files are skipped; use `--refresh` only when deliberately replacing one. Keep `--limit 4` or another small batch: a single 52-ticker process has previously reached roughly 14.7 GB RSS. Run one driver only because concurrent `to_parquet` writes to the same ticker are not atomic.

The sweep honors the registrant chain and resolves each filing both with and without the statement-role guard. Do not compare runs that used different roster, catalogue, GICS, or registrant inputs.

## Registrant-boundary recovery

A legal-registrant change can leave prices continuous while filing history silently starts too late. Before reparsing, freeze the affected tables and cube parts with [registrant impact](../../scripts/registrant_impact.py):

~~~bash
rtk "$PY" scripts/registrant_impact.py --snapshot baseline_YYYY-MM-DD
~~~

Then apply the approved register change and recover each source according to its storage model:

~~~bash
rtk "$PY" -m src data_extract insider-transactions --reparse
rtk "$PY" -m src data_extract financial-notes --reparse
rtk "$PY" -m src data_extract financial-statements --reparse
rtk "$PY" -m src data_extract sec-8k-items -t TICKER1,TICKER2
rtk "$PY" -m src data_extract def14a -t TICKER1,TICKER2 -F
~~~

The bulk reparses are local because archives are cached. They must cover the full cache: a partial reparse produces an artificial historical boundary that downstream code cannot distinguish from real source availability. Run network listing walks serially because SEC rate limiting is process-local.

After rebuilding affected cube parts, compare the frozen state:

~~~bash
rtk "$PY" scripts/registrant_impact.py --compare baseline_YYYY-MM-DD --tickers TICKER1,TICKER2
~~~

Use the comparison to prove that earlier filing dates and recovered rows reached model inputs, not merely extraction tables.

## Fundamentals backfill

The extraction CLI separates network resolution from local history replay:

~~~bash
rtk "$PY" -m src data_extract fundamentals-facts -F -t AAPL,CSCO,KR,XOM,APA,EOG
rtk "$PY" -m src data_extract fundamentals-history-sec -t AAPL,CSCO,KR,XOM,APA,EOG
~~~

Use multiple bounded ticker chunks. `-F` is required for a from-scratch chunk because the run manifest otherwise treats an equal-sized next chunk as a repeated scope and may resume from the prior run date.

Choose the narrowest repair:

| Defect | Command |
| --- | --- |
| Stored facts are correct; replay/history logic changed | `fundamentals-history-sec --rebuild-history -t ...` |
| Fact resolution itself is wrong | `fundamentals --rebuild -t ...` |
| Vendor inputs changed; rebuild merged consumer history | `fundamentals-history-merged -F -t ...` |

A ticker with no stored accessions may not self-heal through an ordinary incremental run. Identify genuinely empty tickers explicitly, then force only that scope.

## Applying a fundamentals schema change

PostgreSQL initialization DDL runs only for a new volume. On a live volume, inferred creation from an all-null first frame can produce the wrong physical type. The inspected [recreation script](../../scripts/recreate_fundamentals_tables.py) deliberately drops and recreates its declared table set from committed DDL:

~~~bash
rtk "$PY" scripts/recreate_fundamentals_tables.py --dry-run
rtk "$PY" scripts/recreate_fundamentals_tables.py --yes
~~~

This is destructive. Request approval, verify the exact table set printed by the dry run, take backups, and be prepared to rerun facts and history. Do not substitute an ad-hoc `DELETE`.

## Schedule 13D and 13G

Both fetchers understand the pre- and post-mandate form-name eras. Use a full walk after parser/discovery changes or a from-scratch rebuild:

~~~bash
rtk "$PY" -m src data_extract sec-13d -F
rtk "$PY" -m src data_extract sec-13g -F --years 15 -t TICKER1,TICKER2
~~~

Chunk 13G more aggressively than 13D. An institutional filer’s listing contains filings against many unrelated issuers; the issuer guard drops them only after discovery and parsing, so cost follows listing size rather than retained rows.

Do not restore pre-mandate ownership percentages to the cube merely because the backfill completes. The acceptance criteria remain in [TODO](../TODO.md).

## Insider reparse and bulk/live cutover

After a parser or column-contract change, back up the canonical bulk table and reparse cached quarters:

~~~bash
rtk "$PY" -m src data_extract insider-transactions --reparse
~~~

Use `--live-full` when open-quarter discovery or parsing changed. For a completed quarter, the validation command replays EDGAR and compares it with the ZIP representation:

~~~bash
rtk "$PY" -m src validate insider-parity --quarter YYYYQn -o reports/validate/YYYY-MM-DD-insider-parity
~~~

Add `--refresh-replay` only when the retained replay cache must be replaced. Promote `source_freshness.insider_bulk_authoritative_through` only after the report passes. Keep live staging rows: the canonical reader selects one complete representation per accession, and the staging copy makes reconciliation reproducible.

## Source-coverage evidence

Run [source coverage report](../../scripts/source_coverage_report.py) after an ownership backfill:

~~~bash
rtk "$PY" scripts/source_coverage_report.py --since-year 1995
~~~

It streams bounded projections and reports distinct source coverage by year plus the 13F managers-per-ticker time series. This detects a fetch-regime break that a total row count hides.

## Table validation

The current validation CLI is table-oriented. Pull a reusable snapshot, then run the checks relevant to the contract:

~~~bash
rtk "$PY" -m src validate pull -T fundamentals_history_sec -o reports/validate/YYYY-MM-DD-fundamentals
rtk "$PY" -m src validate grain -T fundamentals_history_sec -o reports/validate/YYYY-MM-DD-fundamentals
rtk "$PY" -m src validate coverage -T fundamentals_history_sec -o reports/validate/YYYY-MM-DD-fundamentals
rtk "$PY" -m src validate profile -T fundamentals_history_sec -o reports/validate/YYYY-MM-DD-fundamentals
rtk "$PY" -m src validate leakage -T cube_part_fundamentals -o reports/validate/YYYY-MM-DD-cube-fundamentals
rtk "$PY" -m src validate timeseries -T cube_part_fundamentals -o reports/validate/YYYY-MM-DD-cube-fundamentals
~~~

Commands and exit semantics are owned by [validate CLI](../../src/validate/cli.py): `0` means pass, `1` means a finding stands, and `3` means the check abstained. An abstention is not a pass. Available checks include `grain`, `coverage`, `profile`, `redundancy`, `leakage`, `clip`, `timeseries`, `bounds`, and `catalogue`.

Finish with the project’s required sanity conclusion and the appropriate [validation guide](./validate-a-change.md).

## Related

- [Run the pipeline](./run-the-pipeline.md)
- [Data sources](../reference/data-sources.md)
- [Live database](../reference/live-database.md)
- [Testing and validation](./testing.md)
- [TODO](../TODO.md)
