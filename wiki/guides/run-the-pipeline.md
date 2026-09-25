---
title: Run the pipeline
description: Interpreter, database, CLI, Airflow, cold-start, backfill, and validation operations.
type: guide
tags:
  - wiki
  - guide
  - runbook
  - operations
---
# Run the pipeline

## Goal

Execute a source, cube, model, portfolio, or validation stage from the repository root with the correct interpreter, database route, configuration, and operational safeguards.

> [!IMPORTANT]
> Prefix every shell command with `rtk`. The examples below show the underlying command shape; retain the prefix in actual use.

## Python environment

The project requires Python 3.13. `python`, `python3`, Poetry, and Conda are not usable from the host `PATH`; call the Poetry virtual-environment executable directly:

```bash
PY="$HOME/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe"
rtk "$PY" -m pytest tests/path/test_file.py::test_case -v -s
rtk "$PY" -m src data_extract --help
```

Run from the repository root so `./configs`, `.env`, package imports, and artifact paths resolve. If the Poetry environment hash changes, locate the `stock-pick-strat-*` directory under Poetry's cache.

## Database access

The normal local container is `pea_db`, PostgreSQL 16, database `pea`. The transferred volume's owner is `alexandre`, which differs from compose defaults.

The safest read-only or SQL-administration route is the trusted Unix socket inside the container:

```bash
rtk docker exec pea_db psql -U alexandre -d pea -c "SELECT count(*) FROM prices;"
```

On Git Bash, keep `MSYS_NO_PATHCONV=1` before Docker commands whose container paths could be rewritten. Host connections through port 5432 require the volume's password; do not alter roles or `pg_hba.conf` to avoid asking for it.

Lifecycle:

```bash
rtk docker compose up -d db
rtk docker ps -a
rtk docker exec -it pea_db vacuumdb -U alexandre -d pea --full
```

PostgreSQL init scripts run only on an empty volume. Before any rebuild of the volume, inspect the running container's mounts because an older container may retain a stale pre-restructure bind even though [docker-compose.yml](../../docker-compose.yml) is correct.

Never kill all `python.exe` processes. Stop long source jobs by PID.

## Environment variables

The ignored root `.env` is loaded by [Context](../../src/context.py). Common keys:

| Variable | Purpose |
| --- | --- |
| `SEC_USER_AGENT` | Required real contact identity for SEC EDGAR. |
| `FRED_API_KEY` | FRED macro series. |
| `OPENAI_API_KEY` | Structured proxy/vote extraction and embeddings. |
| `OPENFIGI_API_KEY` | Optional acceleration for CUSIP resolution. |
| PostgreSQL variables or `DATABASE_URL` | Host database override. |
| `ROOT_PATH` | Alternate runtime root, notably Airflow. |

Never reveal, copy, or commit `.env`.

## CLI shape

[src/__main__.py](../../src/__main__.py) dispatches to the dynamic Click root in [src/cli.py](../../src/cli.py):

```bash
rtk "$PY" -m src <package> <command> [-c ./configs] [-t AAPL,MSFT] [-F]
```

Each invocation builds a fresh `Context`. `-F` means full/rebuild according to the owning command; read the command help before using it on a large source.

## Extraction commands

Run `seed-universe` before stages that resolve the default ticker set.

| Area | Commands | Notes |
| --- | --- | --- |
| Universe/prices | `seed-universe`, `price-history`, `dividends`, `splits`, `macro` | Price history is heavy; split adjustment is retroactive, so a full price refresh can be required after a split. |
| Institutionals | `thirteen-f`, `superinvestors`, `thirteen-f-managers`, `insider-transactions`, `short-interest`, `fails-to-deliver`, `sec-8k-items`, `sec-13d`, `sec-13g` | 13G and full manager books are heavy. Use the dedicated vote command only after 8-K narratives exist. |
| Identity | `identity-tables` | Rebuilds `symbol_tenure` and `entity_lineage` together from cached ownership files and database evidence. |
| Fundamentals | `fundamentals`, `fundamentals-facts`, `fundamentals-history-sec`, `fundamentals-sharadar`, `fundamentals-history-merged`, `sharadar-tickers`, `sharadar-actions`, `sharadar-sp500`, `sharadar-gap-check`, `earnings-surprises`, `financial-statements`, `financial-notes` | Facts are network-heavy; SEC and merged history rebuilds are local once inputs exist. |
| Structure/text | `def14a`, `def14a-edgar`, `sec-8k-votes`, `filing-text` | LLM-backed DEF 14A and vote extraction spend API calls; deterministic DEF 14A XBRL is separate. |
| Behavioral | `wiki-pageviews`, `google-trends` | Google Trends is deliberately rate-limited and slow. |
| Calls | `download-earnings-calls`, `ingest-earnings-calls` | First caches source files, then parses them into the database. |

Headcount is produced during the fundamentals filing walk; there is no separate employee command. Exact current names and options are defined in [data_extract/cli.py](../../src/data_extract/cli.py).

## Peers and cube

```bash
rtk "$PY" -m src data_peers deduce-peers
rtk "$PY" -m src data_aggregate build-prices
rtk "$PY" -m src data_aggregate build-target
rtk "$PY" -m src data_aggregate build-fundamentals
rtk "$PY" -m src data_aggregate build-momentum
rtk "$PY" -m src data_aggregate build-text
rtk "$PY" -m src data_aggregate build-institutionals
rtk "$PY" -m src data_aggregate build-governance
rtk "$PY" -m src data_aggregate assemble-cube
rtk "$PY" -m src data_aggregate cube-status
```

The same eight part objects are driven by the composite build, individual CLI commands, and the Airflow chain. The command list and warm-ups come from [parts.py](../../src/data_aggregate/utils/common/parts.py).

Parts are incremental by default. `-F` forces a rebuild. A changed target label/horizon set changes physical columns; the target step detects this and rebuilds, after which the cube must be assembled again.

Institutional incremental computation intentionally reads full history for expanding/event-age families and writes only the tail. Budget it like a full calculation. Governance carries a long daily self-history despite annual filing inputs; do not infer cheapness from source cadence.

## Modelling and portfolio

```bash
rtk "$PY" -m src modelling train --train-start YYYY-MM-DD --train-end YYYY-MM-DD
rtk "$PY" -m src portfolio backtest
rtk "$PY" -m src modelling full-train
rtk "$PY" -m src modelling predict --n-dates 1
rtk "$PY" -m src portfolio strategy-moves
```

The production order matters: evaluate holdout artifacts before replacing them with the full-history fit. Daily prediction loads saved production models; it does not retrain.

See [modelling and portfolio](../reference/modelling-and-portfolio.md).

## Cold start

```text
1. Start the database.
2. Create the root .env with SEC_USER_AGENT and required provider keys.
3. Seed the universe.
4. Run required extraction sources, starting with price history.
5. Deduce peers.
6. Build all cube parts and assemble the cube with a full run.
7. Train the holdout model, backtest it, then full-train production artifacts.
8. Run portfolio analysis and strategy moves as needed.
9. Validate the populated domains.
```

A cold database is allowed to expose missing tables as errors. Do not “fix” that by turning all reads optional.

## Airflow

Airflow runs on Python 3.12, while the pipeline dependencies are installed into an isolated virtual environment at `/opt/pipeline`. DAG tasks invoke:

```text
/opt/pipeline/bin/python -m src <package> <command>
```

with working directory `/opt/airflow/project`.

Start the metadata database and initializer before scheduler/webserver:

```bash
rtk docker compose up -d airflow-db airflow-init
rtk docker compose up -d airflow-scheduler airflow-webserver
```

The compose stack mounts the repository, persistent `data/`, and DAG directory separately. Inside containers, use the service hostname `db`, not localhost. Operational pools throttle SEC bulk, SEC API, scraping, and aggregate tasks.

For schedules and triggers, see [DAGs and infrastructure](../modules/dags-and-infrastructure.md) and [nightly refresh](../flows/nightly-data-refresh.md).

## Heavy backfills

This page keeps the decision summary. Exact recovery sequences, rollback evidence, coverage reports, and current validation commands live in [large backfills and recovery](./large-backfills-and-recovery.md).

### General rules

- Back up affected tables before destructive or broad writes.
- Chunk edgartools walks into separate processes; its per-filing caches can retain memory.
- Run only one SEC network walk at a time because the rate limiter is per process.
- Use full/reparse flags when the manifest watermark would otherwise skip historical rows.
- Stop jobs by PID.
- Do not infer success from row-count growth alone; run source coverage and domain validation.

### Fundamentals

Run facts in bounded ticker chunks with `-F`, then replay SEC history locally. A table-schema change requires the approved recreation script and a dry run before applying committed DDL. Validate fundamentals by tier/roster and render retained reports rather than re-running when only presentation changes.

### Registrant boundaries

Bulk datasets must be reparsed from cache after expanding a registrant chain; the ticker universe size has not changed, so ordinary incremental logic cannot discover the new CIK history. Network listing fetchers then walk the affected register tickers serially. Take before/after impact snapshots and per-table dumps.

### 13D/13G

Form-string eras and manifests make `-F` important for complete rebuilds. 13G listing cost is driven by all filings a financial institution submitted against other issuers, not only rows retained for that ticker, so chunk it.

### Insider bulk/live cutover

Run canonical insider extraction, use `--live-full` after discovery/parser changes, replay a completed quarter through the parity validator, and advance the authoritative bulk quarter only after a retained PASS. Keep staged live rows as the reproducible audit copy.

## Validation commands

The validation CLI is table-oriented. Pull a reusable snapshot, then run the checks that match the table contract:

```bash
rtk "$PY" -m src validate pull -T cube_part_institutionals -o reports/validate/YYYY-MM-DD-institutionals
rtk "$PY" -m src validate grain -T cube_part_institutionals -o reports/validate/YYYY-MM-DD-institutionals
rtk "$PY" -m src validate coverage -T cube_part_institutionals -o reports/validate/YYYY-MM-DD-institutionals
rtk "$PY" -m src validate profile -T cube_part_institutionals -o reports/validate/YYYY-MM-DD-institutionals
rtk "$PY" -m src validate leakage -T cube_part_institutionals -o reports/validate/YYYY-MM-DD-institutionals
rtk "$PY" -m src validate timeseries -T cube_part_institutionals -o reports/validate/YYYY-MM-DD-institutionals
```

The inspected [validation CLI](../../src/validate/cli.py) also exposes `redundancy`, `clip`, `bounds`, and `catalogue`. Exit codes are contractual: `0` pass, `1` finding, `3` abstained. An abstention is not a pass.

Completed-quarter insider promotion uses its dedicated command:

```bash
rtk "$PY" -m src validate insider-parity --quarter YYYYQn -o reports/validate/YYYY-MM-DD-insider-parity
```

For expensive historical repair, rollback, coverage, and retained-evidence procedures, use [large backfills and recovery](./large-backfills-and-recovery.md). Reports remain excluded artifacts; the wiki records the procedure and contracts, not generated results.

## Streamlit and scripts

Run the dashboard with its environment's Streamlit executable and [app.py](../../app/app.py). It assumes model artifacts already exist.

Regenerate managed DDL through [generate_schema_sql.py](../../scripts/generate_schema_sql.py) after an approved table-registry change. Treat one-off scripts as operational instruments: read their arguments, mutation behavior, and rollback notes before execution.

## Gotchas

- PostgreSQL `DATE` returns `datetime.date`; a Parquet-only test can hide this.
- `load` raises on missing/empty tables by design.
- `iter_load` holds a connection until exhausted or closed.
- Bulk SEC caches can be tens of gigabytes; check before downloading again.
- A full fundamentals quality pass can take hours; scope iterations.
- The running container may not reflect the latest compose bind configuration.
- A shell pipeline can mask the underlying command's exit status; capture logs without losing the real code.

## Related

- [Orchestration and runtime](../architecture/orchestration-and-runtime.md)
- [Data sources](../reference/data-sources.md)
- [Live database](../reference/live-database.md)
- [Testing and validation](./testing.md)
- [Validate a change](./validate-a-change.md)
- [Large backfills and recovery](./large-backfills-and-recovery.md)
