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
rtk "$PY" -m src data_extract fundamentals-facts -t TICKER1,TICKER2
rtk "$PY" -m src data_extract fundamentals-employees -t TICKER1,TICKER2
rtk "$PY" -m src data_extract sec-8k-items -t TICKER1,TICKER2
rtk "$PY" -m src data_extract def14a -t TICKER1,TICKER2 -F
~~~

The bulk reparses are local because archives are cached. They must cover the full cache: a partial reparse produces an artificial historical boundary that downstream code cannot distinguish from real source availability. Identity-aware network fetchers list each ticker's filings from the local filing index over its current registrant lineage on every run, so a newly declared CIK's filings enter the work list with no flag. Run those walks serially because SEC rate limiting is process-local; use `-F` only to re-read filings already stored.

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

Use multiple bounded ticker chunks. A from-scratch chunk needs no `-F`: its work list is every indexed filing not yet stored. Use `-F` only to re-read stored filings. A chunk larger than `max_documents_per_run` drains over several runs unless `--no-cap` is given.

Choose the narrowest repair:

| Defect | Command |
| --- | --- |
| Stored facts are correct; replay/history logic changed | `fundamentals-history-sec --rebuild-history -t ...` |
| Fact resolution itself is wrong | `fundamentals --rebuild -t ...` |
| Employee parser missed or rejected headcount | `fundamentals-employees -F -t ...` |
| Vendor inputs changed; rebuild merged consumer history | `fundamentals-history-merged -F -t ...` |

A ticker with no stored accession lists its whole window on the next ordinary run. A filing that was read and has no XBRL facts is stored as a marker and is not listed again.

Employee repair is independent of facts and SEC-history replay. To re-decide specific filings, delete their `fundamentals_employees` rows (count or NULL) and let the next run read them; to pin a value by hand, add the accession to `configs/sec/employees_manual_roster.json`. Full employee mode re-decides every filing in the window. Retrieval or body-read failures write no row, so they retry on the next run.

## Applying a fundamentals schema change

PostgreSQL initialization DDL runs only for a new volume. On a live volume, inferred creation from an all-null first frame can produce the wrong physical type. The inspected [recreation script](../../scripts/recreate_fundamentals_tables.py) deliberately drops and recreates its declared table set from committed DDL:

~~~bash
rtk "$PY" scripts/recreate_fundamentals_tables.py --dry-run
rtk "$PY" scripts/recreate_fundamentals_tables.py --yes
~~~

This is destructive. Request approval, verify the exact table set printed by the dry run, take backups, and be prepared to rerun facts and history. Do not substitute an ad-hoc `DELETE`.

## Schedule 13D and 13G

Both contracts list the pre- and post-mandate form names, so a from-scratch rebuild needs no flag. Use `-F` after a parser change, to re-read stored filings:

~~~bash
rtk "$PY" -m src data_extract sec-13d -F
rtk "$PY" -m src data_extract sec-13g -F --years 15 -t TICKER1,TICKER2
~~~

Chunk 13G more aggressively than 13D. The filing index lists a filing under every party, so a universe asset manager or bank lists every 13G it filed on other issuers. The first read of each resolves the key's role from the filing header and stores a filer-role marker, so that cost falls on the first run only.

Do not restore pre-mandate ownership percentages to the cube merely because the backfill completes. The acceptance criteria remain in [TODO](../TODO.md).

## Document backlogs and markers

A first run over a table's full history (a cleared table, the one-time backlog) can exceed `max_documents_per_run`. Run it once outside 01:00 with `--no-cap`, one EDGAR walk at a time. Before it, read the work list without touching SEC:

~~~bash
rtk "$PY" -m src data_extract edgar-index
rtk "$PY" -m src data_extract resume-plan --table sec_13g --timings
~~~

`edgar-index --build` downloads every quarter of the index again. A filing that was read and holds no data for its key is stored once as an empty-filing marker. `markers --count` counts them per table; `markers --delete` removes every marker by its sentinel and is a rollback tool only, because the next run lists those filings again.

## 13F holes and new tickers

The nightly `thirteen-f` walk starts at the newest stored `filing_date` minus the 7-day overlap and keeps a low watermark: a filing inside the overlap that still fails after the retry rounds holds back every filing after it, so the next run starts at or before it; an older failure is skipped with an ERROR naming its accession. A hole older than the overlap needs an explicit window:

~~~bash
rtk "$PY" -m src data_extract thirteen-f --filing-window 2024-01-01:2024-03-01
~~~

`thirteen-f-backfill` fills the history of a ticker added within the last 7 days from the SEC 13F data sets, then one EDGAR walk over the filings after the newest data set, once per ticker. `-t X` narrows the new tickers (an established X does nothing); `-t X -F` re-reads every data set for X. Run a first manual backfill outside 01:00 and read its unit-check log lines. The value unit of each filing comes from edgartools' private `_detect_value_in_thousands`: an edgartools upgrade that renames it breaks the `data_extract` CLI import, which `test_13f_backfill_zip.py` catches.

## Insider reparse and full rebuild

`insider_transactions` is one table for both sources (see [data sources](../reference/data-sources.md)). A run ingests pending zip quarters first, then reads from EDGAR every indexed Form 3/4/5 of each ticker's lineage filed from the earlier of the day after the last stored zip quarter and the run date minus 7 days, except the accessions already stored from EDGAR.

- **Zip parse change** (a new column, an identity change): `--reparse` re-reads every cached quarter. Filings EDGAR already holds only get their `quarter` stamped; every other filing is re-saved from the zip. Nothing is re-downloaded. Run it without `-t`: only a full-universe run sweeps stored rows that the identity screen now rejects.

  ~~~bash
  rtk "$PY" -m src data_extract insider-transactions --reparse
  ~~~

- **Full rebuild** (an EDGAR parser change, a key or encoding change): drop the table, then run with `-F`. `-F` implies `--reparse`: every cached zip is re-parsed into the empty table, then EDGAR re-reads its whole window, including filings already stored from EDGAR.

  ~~~bash
  rtk docker exec pea_db psql -U alexandre -d pea -c "DROP TABLE insider_transactions"
  rtk "$PY" -m src data_extract insider-transactions -F
  ~~~

  Dropping the table is a live-DB change: ask first. `store.save` recreates it from the registry on the first zip save.

Check the run log: per zip quarter, the `X / N filings missing from EDGAR` WARNING and its mismatch INFO line; one identity-exclusion WARNING per run; and the EDGAR coverage line. The cube's insider frontier is read from the table itself (see [source availability](../concepts/source-availability.md)). There is no parity command or promotion step: EDGAR always wins on overlap.

After a refill, check that every quarter was ingested before trusting the frontier. A quarter that no SEC path serves is skipped with only per-URL WARNINGs and stays pending, while the EDGAR window still starts after the last stored quarter, so the hole shows only in the stored quarters. Compare the stored quarters with the expected ones (2006q1 to the last published quarter):

~~~bash
rtk docker exec pea_db psql -U alexandre -d pea -c "SELECT count(DISTINCT quarter), min(quarter), max(quarter) FROM insider_transactions"
~~~

If a quarter is missing, one normal run (no `-F`, no `-t`) ingests it: it is still pending because no row carries its `quarter`. Its first `missing from EDGAR` WARNING then reads near 100 %, an artefact of the hole, not an EDGAR loss.

Run from a worktree with `ROOT_PATH` set to the main repo root, so the run uses the cached zips and filing index instead of downloading them again into the worktree:

~~~powershell
$env:ROOT_PATH = "<main repo root>"; rtk $PY -m src data_extract insider-transactions -F
~~~

~~~bash
ROOT_PATH="<main repo root>" rtk "$PY" -m src data_extract insider-transactions -F
~~~

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
