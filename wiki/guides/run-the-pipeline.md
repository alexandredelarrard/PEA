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
| `FINBERT_DEVICE` | `cuda` or `cpu` for earnings-call FinBERT scoring; unset picks CUDA when available. |
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
| Universe/prices | `seed-universe`, `price-history`, `macro` | One `price-history` download writes `prices`, `prices_dividends` and `prices_splits`; a new split re-pulls its ticker's full history. Price history is heavy. |
| Institutionals | `thirteen-f`, `thirteen-f-backfill`, `superinvestors`, `thirteen-f-managers`, `insider-transactions`, `short-interest`, `fails-to-deliver`, `sec-8k-items`, `sec-13d`, `sec-13g` | 13G is heavy. `thirteen-f` writes the S&P 500 slice and the roster managers' complete books in one oldest-first walk; `thirteen-f-backfill` fills a new ticker's 13F history; `thirteen-f-managers` only catches each roster CIK up from its stored `sec13f_manager_holdings` frontier. `short-interest --repair-gaps` re-reads once the days a key misses inside its stored span. Use the dedicated vote command only after 8-K narratives exist. |
| Identity | `identity-tables` | Rebuilds `symbol_tenure` and `entity_lineage` together from one pass over the cached Form 3/4/5 zips plus database evidence; a table whose rebuilt frame matches the stored one is not rewritten. |
| Fundamentals | `fundamentals`, `fundamentals-facts`, `fundamentals-employees`, `fundamentals-history-sec`, `fundamentals-sharadar`, `fundamentals-history-merged`, `sharadar-tickers`, `sharadar-actions`, `sharadar-sp500`, `sharadar-gap-check`, `earnings-surprises`, `financial-statements`, `financial-notes` | Facts and employees are independent SEC network walks; SEC and merged history rebuilds are local once inputs exist. |
| Structure/text | `def14a`, `def14a-edgar`, `sec-8k-votes`, `filing-text` | LLM-backed DEF 14A and vote extraction spend API calls; deterministic DEF 14A XBRL is separate. |
| Earnings calls | `extract-earnings-calls [-F] [-t]` | Reads the defeatbeta HuggingFace parquet into `earnings_call_sections`: only the row groups whose latest call reaches the stored frontier minus the overlap; `-F` compares every scoped call and deletes calls the source no longer has. |
| Resume tooling | `extraction-status`, `resume-plan`, `edgar-index`, `markers` | `extraction-status` prints the per-table freshness report and exits 0. `resume-plan [--table T] [-t X] [--timings]` prints each EDGAR document table's work list by key class, read-only. `edgar-index [--build]` refreshes the local filing index (`--build` downloads every quarter again). `markers --count` counts empty-filing markers per table; `markers --delete` removes them, for a rollback only. |

### What a run fetches

Every fetcher derives its work list from its own table's rows, its `Resume` contract in [schema.py](../../src/data_store/schema.py) and the run date. Missing filings, holes and new tickers are found on every run; to repair stored rows, delete them and the next run fetches them again.

- **New ticker.** A ticker whose `sp500_tickers.added_on` lies inside the table's overlap (7 days for the archives and the 13F backfill) gets its full history. A ticker taken out of `INSUFFICIENT_HISTORY_TICKERS` keeps its old `added_on`, so it is not new: run `-t <ticker> -F` once for each source.
- **`-t X`** plans X alone. An EDGAR document command reads X's missing filings. An archive command (`financial-statements`, `financial-notes`, `fails-to-deliver`, the zip half of `insider-transactions`) does nothing for an established X; `-t X -F` re-reads X's stored periods. A `-t` 13F run never reads past the stored frontier.
- **`-F`** re-reads the whole listed history, stored filings and markers included. `--no-cap` lifts `max_documents_per_run`.
- **Archives.** A period is done only when every table of its fetcher holds it. A period with no universe row is listed again every night. A change of the FTD symbol policy needs an explicit `-F`.
- **Short interest.** A night reads only the days on which no key has a row; `--repair-gaps` re-reads each key's own gap days once.
- **DEF 14A LLM.** A stored row, an evidence-free answer included, counts as done; only `-F` sends the proxy again.
- **13F.** A hole older than the 7-day overlap needs `thirteen-f --filing-window START:END`.

### Source-specific notes

`fundamentals-employees` owns the 10-K/10-K/A/10-K405 headcount walk and uses GPT-6 Luna on filing text. It stores source-supported values at the SEC filing date, and a NULL row for a filing with no usable count, and can run without replaying SEC XBRL facts or `fundamentals_history_sec`. A normal run lists the whole history window and skips filing dates that already have a row; a filing in `configs/sec/employees_manual_roster.json` takes its value from there instead of the LLM. Delete a row to have that filing re-decided. `--full` re-decides every filing in the window and upserts counts or NULLs. A full run uses SEC and OpenAI APIs and writes the employee table. Exact current names and options are defined in [data_extract/cli.py](../../src/data_extract/cli.py).

For a notes availability-date correction, first obtain the approved `available_at` column migration for both notes tables and back up the affected data. Then run `rtk "$PY" -m src data_extract financial-notes --repair-availability`: this updates existing clocks only and does not download or reparse ZIPs. Check the resulting period dates (for example, `2021_08` → 2021-09-13), then fully rebuild `cube_part_fundamentals` and reassemble `cube`. The ordinary 45-trading-session refresh cannot rewrite the older historical feature rows whose dates changed. See [data sources](../reference/data-sources.md) for the estimated-versus-observed date policy.

For a `pension_facts` ZIP-vintage/clock correction, first verify every cached quarterly ZIP has readable `sub.txt` and `num.txt`, take a restorable table dump, then recreate only `public.pension_facts` and run `rtk "$PY" -m src data_extract financial-statements -c ./configs --reparse`. A normal incremental run cannot replace the old four-column primary key or recover earlier ZIP vintages. Verify the new five-column key, `available_at DATE`, one date per quarter, and representative revisions before using the data. Rebuild `cube_part_fundamentals` with `-F` and reassemble `cube` separately: the 45-session refresh does not repair old feature dates. Do not treat the +12 historical estimate as a verified SEC posting date. See [data sources](../reference/data-sources.md) and [table catalog](../reference/table-catalog.md).

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

Parts are incremental by default: each rewrites at least its last 7 sessions (45 for fundamentals), so a source correction older than that needs `-F`, which forces a rebuild. A changed target label/horizon set changes physical columns; the target step detects this and rebuilds, after which the cube must be assembled again.

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
4. Build `identity-tables` before any identity-consuming SEC extraction.
5. Run required extraction sources, starting with price history.
6. Deduce peers.
7. Build all cube parts and assemble the cube with a full run.
8. Train the holdout model, backtest it, then full-train production artifacts.
9. Run portfolio analysis and strategy moves as needed.
10. Validate the populated domains.
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

On a managed Windows laptop whose HTTPS traffic is re-signed by a corporate proxy, export the Windows trust store once and recreate the Airflow services so the writable yfinance cache setting takes effect:

```bash
rtk "$PY" -m src.utils.ssl_setup
rtk docker compose up -d --force-recreate airflow-scheduler airflow-webserver
```

The generated `.cache/corporate_ca_bundle.pem` is ignored by Git and visible inside Airflow through the repository bind mount. Pipeline startup reuses it on Linux for `SSL_CERT_FILE`, `CURL_CA_BUNDLE`, and `REQUESTS_CA_BUNDLE`; certificate and hostname verification remain enabled. Compose also directs yfinance's timezone and cookie caches to writable `/tmp/pea-cache`.

The compose stack mounts the repository, persistent `data/`, and DAG directory separately. Inside containers, use the service hostname `db`, not localhost. Operational pools throttle SEC bulk, SEC API, and aggregate tasks; light sources, including earnings calls, run in the default pool.

In the extraction DAG, `identity-tables` runs after `insider-transactions` and before every identity-consuming SEC task, including facts, standalone employees, deterministic and LLM proxy extraction, 8-K/13D/13G, and filing text. The independent 13F manager chain is not an issuer-identity consumer. Sharadar merge waits for both SEC facts/history and employee extraction.

For schedules and triggers, see [DAGs and infrastructure](../modules/dags-and-infrastructure.md) and [nightly refresh](../flows/nightly-data-refresh.md).

## Heavy backfills

This page keeps the decision summary. Exact recovery sequences, rollback evidence, coverage reports, and current validation commands live in [large backfills and recovery](./large-backfills-and-recovery.md).

### General rules

- Back up affected tables before destructive or broad writes.
- Chunk edgartools walks into separate processes; its per-filing caches can retain memory.
- Run only one SEC network walk at a time because the rate limiter is per process.
- Use full/reparse flags only to re-read history that is already stored; missing filings and holes need no flag.
- Stop jobs by PID.
- Do not infer success from row-count growth alone; run source coverage and domain validation.

### Fundamentals

Run facts in bounded ticker chunks with `-F`, then replay SEC history locally. A table-schema change requires the approved recreation script and a dry run before applying committed DDL. Validate fundamentals by tier/roster and render retained reports rather than re-running when only presentation changes.

### Registrant boundaries

Bulk datasets must be reparsed from cache after expanding a registrant chain; the ticker universe size has not changed, so ordinary incremental logic cannot discover the new CIK history. Network listing fetchers then walk the affected register tickers serially. Take before/after impact snapshots and per-table dumps.

### 13D/13G

Both form-string eras are in the contract forms, so a normal run lists every missing filing; `-F` re-reads stored ones too. The filing index lists a 13G under every party, so a universe bank or asset manager also lists the thousands of 13Gs it filed on other issuers; each is read once and stored as a filer-role marker, so the first backlog run pays that cost. Chunk a 13G `-F`.

### Insider transactions

`insider-transactions` loads one table: pending zip quarters first (only the filings EDGAR lacks), then every indexed Form 3/4/5 of each ticker's lineage filed from the earlier of the day after the last stored zip quarter and the run date minus 7 days that is not yet stored from EDGAR. The stored-row identity sweep runs only on a full-universe run; a `-t` run skips it. `--reparse` re-reads every cached zip quarter after a zip parse change; `-F` implies `--reparse` and also re-reads the EDGAR window, including filings already stored from EDGAR. A full rebuild drops the table first, then runs `-F` (see [large backfills and recovery](./large-backfills-and-recovery.md)). Read the per-quarter `filings missing from EDGAR` WARNING and the one identity-exclusion WARNING per run; there is no parity or promotion step.

### Earnings-call rebuild

A full text rebuild is dominated by FinBERT. Measured on 2026-10-02 (`reports/validate/2026-10-01-earnings-call-extraction-gaps/03-implementation.md`):

| Stage | Cost | Resume |
| --- | --- | --- |
| `extract-earnings-calls -F` | 18.6 min for 33,591 calls, mostly DB writes | Writes one row group per batch, so a crash loses at most one batch. |
| FinBERT sentiment | 27–37 s per call on CPU, about 280 h for every call | Scores are upserted per ticker under the `speaker-clean-v2` cache version. |
| OpenAI embeddings | about 9,150 tokens per call, about $6 for every call; about 100 calls/min, bound by `float8[]` inserts | Complete calls are skipped on (ticker, quarter, model tag). |

Run FinBERT on a GPU with `FINBERT_DEVICE=cuda`. `build-text` scores sentiment before it embeds and has no `-t` scope; `-F` rebuilds `cube_part_text` for the universe, after which the cube must be assembled again. Then validate the part:

```bash
rtk "$PY" -m src validate earnings-calls -T cube_part_text -o reports/validate/YYYY-MM-DD-earnings-calls
```

The check reads `earnings_call_sections` itself; `-T earnings_call_sections` is the wrong target. It reports per-ticker and per-calendar-quarter coverage, paragraph-grain checks, split status (`ok` rate at least 0.985, prepared word share q05 at least 0.15, from `configs/validate.yml`), `as_of` against the nearest `earnings_surprises` date, and the 12-column feature schema, bounds, redundancy and drift.

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

For expensive historical repair, rollback, coverage, and retained-evidence procedures, use [large backfills and recovery](./large-backfills-and-recovery.md). Reports remain excluded artifacts; the wiki records the procedure and contracts, not generated results.

## Streamlit and scripts

Run the dashboard with its environment's Streamlit executable and [app.py](../../app/app.py). It assumes model artifacts already exist.

Regenerate managed DDL through [generate_schema_sql.py](../../scripts/generate_schema_sql.py) after an approved table-registry change. Treat one-off scripts as operational instruments: read their arguments, mutation behavior, and rollback notes before execution.

## Gotchas

- PostgreSQL `DATE` returns `datetime.date`; a Parquet-only test can hide this.
- `load` raises on missing/empty tables by design.
- `iter_load` holds a connection until exhausted or closed.
- Bulk SEC caches can be tens of gigabytes; check before downloading again. The SEC 13F data sets (`sec_13f_datasets`) download lazily, only when a new ticker needs them.
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
