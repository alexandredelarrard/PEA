---
title: Run the pipeline
description: Interpreter, database, CLI, Airflow, cold-start, backfill, and validation operations.
type: guide
tags:
  - wiki
  - guide
  - runbook
  - operations
source_commit: aa9ac35ff94b6992ac43106230f63e3d50082cab
---
# Run the pipeline

## Goal

Execute a source, cube, model, portfolio, or validation stage from the repository root with the correct interpreter, database route, configuration, and operational safeguards.

> [!IMPORTANT]
> Prefix every shell command with `rtk`. The examples below show the underlying command shape; retain the prefix in actual use.

## Codebase navigation

For source questions, follow [Query the codebase with Graphify](./query-the-codebase.md): read the relevant wiki page, query the local graph, then inspect returned source locations. The guide includes the local AST-only refresh command and preserves OpenKnowledge Markdown handling.

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

PostgreSQL init scripts run only on an empty volume. The four fundamentals tables (`sec_fundamentals_facts`, `sec_fundamentals_history`, `sharadar_fundamentals_facts`, `sharadar_fundamentals_history`) are not in `sql/schema.sql`: the nightly run creates each on its first write, so no schema step comes before their first run (see [table catalog](../reference/table-catalog.md#fundamentals)). [docker-compose.yml](../../docker-compose.yml) mounts only `sql/schema.sql`, read-only, as `docker-entrypoint-initdb.d/01-schema.sql`; it preserves the external volume declaration. Canonical SQL already contains the 13F accession keys and all five DEF14A retry counters. Existing databases are assumed current: ingestion performs no 13F/DEF14A adoption, backfill or legacy-default synthesis. Do not recreate an existing volume to apply this code change. Before any separately approved volume rebuild, inspect the running container's mounts because an older container may retain a stale pre-restructure bind.

Never kill all `python.exe` processes. Stop long source jobs by PID.

## Environment variables

The ignored root `.env` is loaded by [Context](../../src/context.py). Common keys:

| Variable | Purpose |
| --- | --- |
| `SEC_USER_AGENT` | Required real contact identity for SEC EDGAR. |
| `FRED_API_KEY` | FRED macro series. |
| `OPENAI_API_KEY` | Structured proxy/vote extraction and embeddings. |
| `OPENFIGI_API_KEY` | Optional acceleration for CUSIP resolution. |
| `FINBERT_DEVICE` | `cuda` requires working CUDA and propagates model/device failures; `cpu` forces CPU; unset selects CUDA when available. See [Airflow GPU setup](#airflow-gpu-setup). |
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

The universe is [configs/universe.yml](../../configs/universe.yml): each ticker maps to its GICS `sector` and sub-industry (`industry`); the industry group is derived. To add a ticker, add one line there. After editing it, or on a fresh database, run `universe [--as-of] [--only sec-tickers|sharadar-tickers|sharadar-actions|sec-submissions|sync]` (all five parts by default): it refreshes `sec_company_tickers`, `sharadar_tickers`, the Sharadar actions and the SEC bulk submissions, then the sync writes `universe_tickers` (only what changed; a new ticker takes its CIK and name from the SEC list) and every stage reads it through `context.universe`, which raises `UniverseError` naming this command when the table is missing, empty or differs from the yml. `-t` takes universe tickers only: an unknown name or an alias symbol (GOOG, FOX, NWS, BRK-A) is refused before any fetch.

| Area | Commands | Notes |
| --- | --- | --- |
| Universe/prices | `universe`, `price-history`, `macro`, `earnings-surprises` | One `price-history` run shares one yfinance session across `fetch_price_actions` (`prices_splits`), `fetch_dividends` (`prices_dividends`) and `fetch_prices` (`prices`), in that order, so a ticker window is downloaded once; splits are planned from the stored `prices`, which `fetch_prices` advances last. A new split or cash dividend, or a first-calendar-year `close_split` that no longer matches Yahoo's (the anchor rule of the [price-basis contract](../reference/data-sources.md#price-basis-contract)), re-pulls its ticker's full history. Price history is heavy. |
| Institutionals | `thirteen-f`, `thirteen-f-backfill`, `superinvestors`, `thirteen-f-managers`, `insider-edgar`, `short-interest`, `sec-8k-items`, `sec-13d`, `sec-13g` | 13G is heavy. `insider-edgar` is the EDGAR half of `insider_transactions`; its zip half is `zips-ingest --only insider` (the DAG runs that in `sec_bulk`, then `insider-edgar` in `sec_api`). `short-interest` and `zips-ingest --only ftd` stamp the raw per-security rows from `universe_cusip` and rebuild the ticker tables. `thirteen-f` writes the S&P 500 slice and the roster managers' complete books in one oldest-first walk; `thirteen-f-backfill` fills a new ticker's 13F history; `thirteen-f-managers` only catches each roster CIK up on the accessions not yet complete in `sec13f_hr`. `short-interest --repair-gaps` re-reads once the days a key misses inside its stored span. Use the dedicated vote command only after 8-K narratives exist. |
| SEC bulk zips and identity | `zips-download [--only SOURCE] [-F]`, `identity [--only STAGE] [-t] [-F] [--dry-run]`, `zips-ingest [--only SOURCE] [-t] [-F] [--repair-availability]` | Run `zips-download`, then `identity`, then `zips-ingest`, before any identity-consuming SEC command; `--only` is repeatable on the two zip commands. `zips-download` (sources `insider`, `notes`, `ftd`, `sec-tickers`) only caches the Form 3/4/5, Notes and FTD zips and snapshots SEC's current tickers, its one table write; with no lineage stored it caches every listed FTD period. `identity` runs offline in eight stages: `capture-evidence` stores the cover-page symbols and the Form 3/4/5 tenure (`-F` re-captures every cached zip), while `cusip-history` reads the in-scope lines of every cached FTD zip and stores none (`zips-ingest --only ftd` is the only writer of the two FTD tables); `company-anchor`, `cik-history`, `symbol-history`, `cusip-history` and `lifecycle` rebuild each company's frozen `company_id`, CIK windows, symbol intervals, CUSIP lines and `first_price_date` / `ended_on` from stored inputs; `publish`, the only stage that saves, writes `universe_cik_symbol`, `universe_cusip` and the identity columns of `universe_tickers` after its completeness gate; and `propagate` purges rows whose filer CIK left the company or its window, re-parses the bulk families, re-stamps the FTD, short-volume, insider and 13F rows, and rebuilds purged tickers' history, for tickers changed in the last 7 days (`--full`: all of them, for a config-only change; a first build checks every ticker). `--only propagate --dry-run` only lists the pending changes; `--dry-run` with any other `--only` (or none) is refused (usage error, nothing runs). `zips-ingest` (sources `insider`, `notes`, `ftd`) parses the cached archives into `insider_transactions` (zip quarters), `notes_num` (footnote pension facts and the recognized net pension liability)/`notes_text` and the two FTD tables; `--repair-availability` is refused unless `--only notes` is the only source. `validate identity` is the read-only check. |
| Fundamentals | `sec-fundamentals-facts`, `sec-fundamentals-history`, `fundamentals-check`, `fundamentals-employees`, `fundamentals-sharadar`, `sharadar-sp500` | `sec-fundamentals-facts` (the seventh EDGAR walk, original 10-K and 10-Q only) and `fundamentals-employees` are independent SEC network walks. `sec-fundamentals-history` replays stored facts offline (`--full` deletes and replays the tickers, `--verify-history` replays and refuses a changed published row); after a `sec-fundamentals-facts -F` re-read, run it with `--full` or `--verify-history`. `fundamentals-sharadar` fetches SF1 and builds `sharadar_fundamentals_history` from Sharadar only; after a lineage change run `fundamentals-sharadar -F -t <tickers>`. `fundamentals-check` is read-only and logs the tickers held by one history only, Sharadar CIKs outside their company, the exact-key match share and the fields where enough rows differ. Notes numerics, including every pension fact, are `zips-ingest --only notes`. Adding a tag to `_NOTES_NUM_TAGS` reaches history only through `zips-ingest --only notes -F` (all cached Notes zips, about 1 to 1.5 h, one process, an estimate), then rebuild `cube_part_fundamentals -F`. |
| Structure/text | `def14a`, `def14a-edgar`, `sec-8k-votes`, `filing-text` | LLM-backed DEF 14A and vote extraction spend API calls; deterministic DEF 14A XBRL is separate. |
| Earnings calls | `extract-earnings-calls [-F] [-t]` | Uses dated identity to expand current/predecessor symbols, reads complete scoped metadata with footer symbol pruning, then fetches only missing/reissued payloads under the current ticker. An ordinary run can recover old alias gaps regardless of the stored frontier. `-F` additionally reconciles source-absent calls for tickers present in the source, retaining vanished-ticker/failed-read safeguards. |
| Resume tooling | `extraction-status`, `resume-plan`, `edgar-index`, `markers` | `extraction-status` prints the per-table freshness report and exits 0. `resume-plan [--table T] [-t X] [--timings]` prints each EDGAR document table's work list by key class, read-only. `edgar-index [-F]` refreshes the local filing index (`-F` downloads every quarter again); the DAG runs it once a night before the seven EDGAR walks, and a walk run alone skips its own refresh when today's (`refreshed_today`) already covered its window, so the index is refreshed once per night. `markers --count` counts empty-filing markers per table; `markers --delete` removes them, for a rollback only. |

### Replacements for the per-source zip and identity commands

| Removed command | Replacement |
| --- | --- |
| `insider-download` | `zips-download --only insider` |
| `notes-download [-F]` | `zips-download --only notes`, then `identity --only capture-evidence [-F]` |
| `ftd-download [-F] [--as-of]` | `zips-download --only ftd [--as-of]` (every listed period is cached; `identity` reads them all) |
| `sec-tickers` | `universe --only sec-tickers` |
| `sharadar-tickers` | `universe --only sharadar-tickers` |
| `identity-tables` | `identity` (all eight stages), or `identity --only publish` (rebuilds every stage from the stored inputs and saves) |
| `identity-propagate [-t] [-F] [--dry-run] [--as-of]` | `identity --only propagate [-t] [-F] [--dry-run] [--as-of]` |
| `insider-zip [-t] [-F]` | `zips-ingest --only insider [-t] [-F]` |
| `financial-notes [-t] [-F] [--repair-availability]` | `zips-ingest --only notes [-t] [-F] [--repair-availability]` |
| `fails-to-deliver [-t] [-F]` | `zips-ingest --only ftd [-t] [-F]` |
| `insider-transactions [-t] [-F] [--no-cap]` | `zips-ingest --only insider [-t] [-F]`, then `insider-edgar [-t] [-F] [--no-cap]` |

`identity-tables` ran tenure, lineage and master in one process on one Form 3/4/5 scan; three `--only` runs scan twice, `identity` once.

### What a run fetches

Every fetcher derives its work list from its own table's rows, its `Resume` contract in [schema.py](../../src/data_store/schema.py) and the run date. Missing filings, holes and new tickers are found on every run. DEF 14A LLM repairs stored gaps automatically under its per-section retry rule below; other sources retain their own recovery contracts. No code reads the retired run manifest or the bulk `*_universe.json` sidecars; they are kept in `data/_retired/2026-10/`.

- **New ticker.** A ticker whose `universe_tickers.universe_added_on` lies inside the table's overlap (7 days for the archives and the 13F backfill) gets its full history. A dated series (prices, short interest, Sharadar) pulls it once: when the key's first stored date is within the overlap of the history floor or source start, a rerun resumes forward. For short interest that start is the later of the FINRA start and the key's first `prices` date, so a joiner listed late (a spin-off) is also read in full once; a key with no price row keeps the FINRA start. A ticker removed from `universe.yml` loses its row; added back, it gets a new `universe_added_on` and is new again.
- **`-t X`** plans X alone. An EDGAR document command reads X's missing filings. An archive ingest (`zips-ingest`, any source) does nothing for an established X; `-t X -F` re-reads X's stored periods. A `-t` 13F run never reads past the stored frontier.
- **`-F`** is command-specific. EDGAR document commands that support it re-read their listed history, stored filings and markers included; `--no-cap` lifts `max_documents_per_run` where offered. Neither `thirteen-f` nor `thirteen-f-managers` accepts `-F`; use their explicit filing-window or pending-book rules below. Check [extraction CLI](../../src/data_extract/cli.py) for the owning command's options.
- **Archives.** A period is done only when every table of its fetcher holds it. A period with no universe row is listed again every night. A change of the FTD symbol policy needs an explicit `-F`.
- **Short interest.** A night reads each ticker's window plus the days on which no key has a row; `--repair-gaps` re-reads each key's own gap days once. `-t X -F` re-reads X's days and upserts only X's rows.
- **Tapes after a lineage change.** Run `short-interest`, `zips-ingest --only ftd` and `identity --only propagate` unscoped while tickers sit in the 7-day window: a `-t` re-stamp counts a line that moved between two universe companies under both until the next unscoped run.
- **DEF 14A LLM.** `def14a` lists the full rolling history and schedules five independently owned sections. A section with a missing key field and `retry < 2` gets one call and `retry+1`; two is final and costs no source read. Only filings with an open section are downloaded. Stored-JSON replay is off by default (`replay_stored_json=True` in code). Optional nulls and unreadable filings do not trigger repeated paid work. There is no separate repair command or call-cap parameter. [The extraction flow](../flows/sec-llm-extraction.md) defines the table ownership and empty markers.
- **13F.** A hole older than the 7-day overlap needs `thirteen-f --filing-window START:END`.

### DEF 14A retry adoption

The five retry counters are already part of canonical database initialization. Existing schemas are assumed current. The completed one-time adoption/backfill and `prepare_def14a_retries` hook are removed; generic writes persist the section owner's attempts and facts without issuing ALTERs or synthesizing legacy metadata.

An ordinary `rtk "$PY" -m src data_extract def14a -c ./configs` run then processes only its derived domain worklist (open sections). The two-attempt limit always holds. Keep the configured rolling history and dated registrant guards. Do not use the legacy `scripts/def14a_reextract.py --write` replacement path after adoption; it now refuses paid writes. Do not delete counters to restart exhausted sections as routine recovery.

### 13F version adoption before a historical cube rebuild

The approved code stores original/amendment accessions in the two existing 13F tables, preserving legacy rows. A cube rerun cannot recreate overwritten source versions. The 11 October live check found no completed accessions, so first replay the required original source scope and pass the read-only validator. Use cached archives where available; broad recovery is a separate bounded operational task.

For the existing historical universe, run the source reloads sequentially from merged dev:

```powershell
rtk "$PY" -m src data_extract thirteen-f-backfill -F -c ./configs
rtk "$PY" -m src data_extract thirteen-f-managers --years 31 -c ./configs
```

The first rereads all published ZIP datasets from the configured source start (2013Q2), preserves each complete original/amendment accession for relevant managers and also fills roster-manager books. The second catches up missing EDGAR accessions for roster managers within the explicit 31-year report-period window; it has no `-F`. Existing-table migration and metadata columns are installed by the writer, preserving old rows as unverified `_legacy` accessions. These commands do not delete either table. Pre-2013 broad history and unavailable source files are not automatically certified by a successful run.

For bounded verification or targeted recovery, use `data_extract thirteen-f-managers --vintages` with both repeated `--cik` and `--accession`, plus required `--period-start`, `--period-end`, `--filed-through`, `--max-filings`. Optional repeated `--archive` selects cached source transport; it does not replace CIK/accession scope. Review `--dry-run` first. The [CLI](../../src/data_extract/cli.py) writes both existing destinations; it creates no auxiliary business table. Validate the same named scope with `validate institutionals --cik ... --period-start ... --period-end ... --filed-through ...`.

After source adoption and the user's price refresh, run `data_aggregate build-institutionals -F`, then `data_aggregate build-sec-thirteen -F`, then `data_aggregate assemble-cube`. With nonempty validated output, these full rebuilds remove transferred/retired columns from the old part and populate the new part; deleting all cube tables is unnecessary. A builder returning no rows retains its previous table even on a full run. Verify physical column sets, unique date/ticker keys and readiness for both parts and the assembled cube before use. These recovery/rebuild operations were not run by closure validation. Code merge/check status lives in the local audit `06-final.md`. Feature-science adoption also requires these two full rebuilds: seven insider cap names retain their names while changing to action-date per-trade ratios, and the elite part changes its cohort/feature contract. A routine tail cannot migrate older same-named rows. Verify the effective ranked-list digest/K/90-day policy and train fresh compatible models after reassembly. The implementation audit does not execute these production commands.

### Source-specific notes

`fundamentals-employees` owns the 10-K/10-K/A/10-K405 headcount walk and uses GPT-6 Luna on filing text. It stores one row per filing date at the SEC filing date (the guarded components with their basis, status and quotes, or a status-only row for a filing with no usable count) and can run without replaying SEC XBRL facts or `sec_fundamentals_history`. A normal run lists the whole history window and skips filing dates that already have a row. Delete a row to have that filing re-decided. `--full` re-decides every filing in the window and upserts the rows. A full run uses SEC and OpenAI APIs and writes the employee table. One process grows with every ticker it reads, so run large sets in sequential batches ([runbook](./large-backfills-and-recovery.md#employee-headcount-universe-run)). Exact current names and options are defined in [data_extract/cli.py](../../src/data_extract/cli.py).

For a notes availability-date correction, first obtain the approved `available_at` column migration for both notes tables and back up the affected data. Then run `rtk "$PY" -m src data_extract zips-ingest --only notes --repair-availability`: this updates existing clocks only and does not download or reparse ZIPs. Check the resulting period dates (for example, `2021_08` → 2021-09-13), then fully rebuild `cube_part_fundamentals` and reassemble `cube`. The ordinary 45-trading-session refresh cannot rewrite the older historical feature rows whose dates changed. See [data sources](../reference/data-sources.md) for the estimated-versus-observed date policy.

For a `notes_num` tag-set change, no incremental run reaches history: run `rtk "$PY" -m src data_extract zips-ingest --only notes -F -c ./configs` on every cached Notes zip (a partial re-parse gives an artificial boundary; about 1 to 1.5 h), then rebuild `cube_part_fundamentals` with `-F` and reassemble `cube` separately. See [data sources](../reference/data-sources.md) and [table catalog](../reference/table-catalog.md).

### Superinvestor roster

For an ordinary refresh, update the roster and then catch up its managers' missing books:

```bash
rtk proxy "$PY" -m src data_extract superinvestors -c ./configs
rtk proxy "$PY" -m src data_extract thirteen-f-managers --years 31 -c ./configs
```

`--years 31` sets the report-period history depth for this run; it does not force a reread. `thirteen-f-managers` has no `-F` option. Its [pending-book planner](../../src/data_extract/utils/institutionals/fetch_13f_managers.py) skips only verified completed accession packets. Legacy latest-only rows do not mark an original or amendment complete. `superinvestors -F` rebuilds roster dates and CIK resolutions; it does not repair holdings.

After a parser fix, repair existing holdings with a backed-up, source-verified whole-book proposal and an atomic scoped application, rather than relying on catch-up upserts to remove stale CUSIPs. The parser fix is merged into `dev`, and the 2026-10-07 historical repair was applied on 2026-10-08 (4 replaced books, 62 NULL-amount books); priorities and exact remaining gaps are in [TODO](../TODO.md#superinvestor-roster-follow-ups). After that repair, build `cube_part_prices` if absent, run `data_aggregate build-sec-thirteen -F`, then `data_aggregate assemble-cube` and the relevant read-only [validation](./validate-a-change.md). The latter assembly command has no `-F` option; see [aggregate CLI](../../src/data_aggregate/cli.py).

The daily `superinvestors` command writes a snapshot only when Dataroma's roster changed. Rebuild the whole `superinvestor_roster` table after a change to its history/overrides registers or roster extraction code, and on a cold database. The feature-only `ranked_managers.json` ordering does not change that source table: changing it or K requires a full SEC-thirteen feature rebuild, cube reassembly and compatible fresh models instead. The default list is provisional; see [configuration](../reference/configuration.md).

```bash
rtk "$PY" -m src data_extract superinvestors -F -c ./configs
rtk "$PY" -m src data_extract thirteen-f-managers -c ./configs
```

`-F` reads the committed quarterly history plus every stored live snapshot, resolves every code fresh (overrides, then EDGAR company search), stores each chained manager's member valid at the snapshot, checks primary-key uniqueness and runs the resolution and 13F activity gates, and only then writes: it upserts the rebuilt rows in one transaction, then deletes the stored keys they lack with targeted per-date deletes; it then runs the ordinary on-change refresh, so a broken Dataroma page fails `-F` after the rebuild is already written (the rebuild itself is unaffected). A resolution, primary-key or gate failure, an `EdgarListingError` (the EDGAR 13F-HR listing could not be read: rerun, do not edit the overrides) or a refused scrape (the page lists no manager or under 80% of the latest snapshot's) raises before its write touches the table. The table is never emptied, so its live snapshots stay readable for the next rebuild: a crash during the upsert leaves the old table, a crash during the deletes leaves a superset, and a rerun converges to the same table. A gate failure names the code, CIK, nearest known 13F period and window: research the filer on EDGAR and add an evidence-backed `cik_overrides`, `manager_ciks` window or dated `inactive` range to [overrides.json](../../configs/superinvestors/overrides.json), then rerun. `thirteen-f-managers` then fetches the books of any new filer CIK (one EDGAR walk; run nothing else against SEC meanwhile).

If the `data_extraction` DAG was paused so older code could not append a snapshot while a roster change was on a branch, rerun `-F` from the merged code first, then unpause the DAG.

To regenerate the history file (a parser fix, or new Wayback captures of past quarters), keep the end before the first live snapshot so history and live dates never collide, review the diff, commit it, then rebuild as above:

```bash
rtk "$PY" scripts/build_dataroma_roster_history.py --cache-dir <wayback-cache-dir> --until 2026-09-07
```

The cache directory keeps the CDX and every capture, so a rerun is offline and byte-identical. See [Superinvestor roster (Dataroma)](../reference/data-sources.md#superinvestor-roster-dataroma).

## Peers and cube

```bash
rtk "$PY" -m src data_peers deduce-peers
rtk "$PY" -m src data_aggregate build-prices
rtk "$PY" -m src data_aggregate build-target
rtk "$PY" -m src data_aggregate build-fundamentals
rtk "$PY" -m src data_aggregate build-momentum
rtk "$PY" -m src data_aggregate build-text
rtk "$PY" -m src data_aggregate build-institutionals
rtk "$PY" -m src data_aggregate build-sec-thirteen
rtk "$PY" -m src data_aggregate build-governance
rtk "$PY" -m src data_aggregate assemble-cube
rtk "$PY" -m src data_aggregate cube-status
```

`build-prices -F` rebuilds `cube_part_prices` in full; `build-prices -t X,Y -F` deletes every saved row of X and Y and rebuilds those tickers from scratch, other tickers untouched (`-t` without `-F` is refused). A normal run takes the same path by itself for any ticker whose history older than the refresh window changed; no other part needs a rerun.

The same nine part objects are driven by eight builders in the composite build, individual CLI commands, and the sequential Airflow chain. The command list and warm-ups come from [parts.py](../../src/data_aggregate/utils/common/parts.py).

### Assemble selected inputs and check readiness

Edit `build_cube.assembly` in [build_cube.yml](../../configs/build_cube.yml) using the [column selection contract](../reference/configuration.md#cube-assembly-selection), then run `rtk "$PY" -m src data_aggregate assemble-cube -c ./configs` followed by `rtk "$PY" -m src data_aggregate cube-status -c ./configs`. Assembly has no `-F` option. This rebuilds the derived cube from persisted parts; it does not rebuild feature parts.

The [assembler](../../src/data_aggregate/transformers/step_assemble_cube.py) requires prices, targets, betas, fundamentals, momentum and governance. Text/institutionals/SEC thirteen can be absent or empty; populated broken inputs stop the build. Every price date/ticker remains, including the newest session with null future labels. Selected names are projected before payload reads. Removing selected columns or skipping a former optional source physically removes its old cube columns; confirm model-input compatibility before using narrower output.

The [status gate](../../src/data_aggregate/utils/common/part_status.py) reports `reference_price_date`, each participating part's expected/actual edge and `refresh_command`. Use the named command to refresh stale/missing parts, then reassemble. At 03:00, the latest stored preceding price session is sufficient. This gate compares table edges, not each ticker/feature's coverage or source prices against a market calendar; existing source/prediction freshness checks remain responsible for those source-age checks.

Input preflight finishes before output changes. The streamed replacement remains non-atomic: a later interruption can leave a partial cube. Resolve the reported failure and rerun assembly from preserved parts. No source part is removed.

Parts are incremental by default: each rewrites at least its last 7 sessions (45 for fundamentals), so a source correction older than that needs `-F`, which forces a rebuild. A changed target label/horizon set changes physical columns; the target step detects this and rebuilds, after which the cube must be assembled again.

`build-institutionals` retains its full-history calculation contract for expanding/event-age families. `build-sec-thirteen` also reads complete sparse dependencies, but streams the broad raw source, caches prepared manager-book statistics and bounds normal daily projection to the planned output window. Current defaults compute 398 existing trading dates and rewrite eight inclusively when the saved frontier is current; a full/schema-change run projects the full calendar. After merging the sec13f-cube-speed change (merged-class `sec13f_hr` books, implied-price whole-book values, superinvestor start at the `sec13f_hr` availability date), pause the DAG and run `data_aggregate build-sec-thirteen -F`, then `data_aggregate assemble-cube`, before resuming it: an incremental run rewrites only the tail, so older saved rows would keep the previous definitions. This reduces raw residency and daily work without proving capacity for the complete source. Governance carries a long daily self-history despite annual filing inputs; do not infer cheapness from source cadence.

## Modelling and portfolio

From a linked checkout, set `ROOT_PATH` to the original repository's absolute path in the parent shell before this sequence (Bash: `export ROOT_PATH="/absolute/path/to/original/PEA"`; PowerShell: `$env:ROOT_PATH = "C:\absolute\path\to\original\PEA"`). Keep it set for all four weekly commands and validation so they share the same data root.

~~~bash
rtk "$PY" -m src modelling train -c ./configs
rtk "$PY" -m src modelling assess -c ./configs
rtk "$PY" -m src portfolio backtest -c ./configs
rtk "$PY" -m src modelling full-train -c ./configs
rtk "$PY" -m src modelling predict --n-dates 1
~~~

Train generates a fresh ID. Standalone assess selects latest complete training; portfolio/full-train select latest assessed training. Optional downstream --run-id pins an exact run; the DAG does this automatically through XCom. Train saves folds/cutoff and basic IC/IR; assess recomputes every score and explanation; full-train restores saved fitting controls/schema and fits all currently mature labels. A failed task blocks successors. See [dated assessment](#dated-modelling-assessment) and [the lifecycle](../reference/modelling-and-portfolio.md).

## Cold start

```text
1. Start the database.
2. Create the root .env with SEC_USER_AGENT and required provider keys.
3. Run `universe` (SEC and Sharadar ticker lists, then `universe_tickers` synced from configs/universe.yml).
4. Run `zips-download`, the identity build order (see the identity cutover below) and `zips-ingest` before any identity-consuming SEC extraction.
5. Run required extraction sources, starting with price history.
6. Deduce peers.
7. Build all cube parts and assemble the cube with a full run.
8. Train the holdout model, backtest it, then full-train production artifacts.
9. Run portfolio backtest on the assessed run and open its backtest_portfolio/report.html.
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

[CA setup](../../src/utils/ssl_setup.py) preserves explicit CA overrides. If the terminal inherits `SSL_CERT_FILE` pointing at a Conda CA file without the corporate proxy root, propagating that file to curl does not fix its missing trust anchor. After generating the repository bundle, select it in that same PowerShell terminal before rerunning the extraction:

```powershell
$env:SSL_CERT_FILE = (Resolve-Path '.cache/corporate_ca_bundle.pem').Path
rtk poetry run python -m src data_extract earnings-surprises
```

This changes the current terminal's environment, not Conda's certificate file. A later Conda activation may select its own CA file again. Separately set `CURL_CA_BUNDLE` or `REQUESTS_CA_BUNDLE` values still retain their explicit paths; inspect and correct those too if they point at the wrong bundle.

The compose stack mounts the repository, persistent `data/`, and DAG directory separately. Inside containers, use the service hostname `db`, not localhost. Operational pools throttle SEC bulk, SEC API, and aggregate tasks; light sources, including earnings calls, run in the default pool.

In the extraction DAG the identity stage comes first: `zips_download >> identity >>` the identity consumers (`zips_ingest_insider`, `zips_ingest_datasets`, short interest, facts, standalone employees, LLM and deterministic DEF 14A, 8-K, 13D, 13G, filing text, `insider_edgar`), and `identity >> price_history` (the fetch list adds the master's current secondary classes). `identity >> edgar_index >>` the seven EDGAR walks (`insider_edgar`, `sec_13d`, `sec_13g`, `sec_8k_items`, `filing_text`, `def14a_edgar`, `sec_fundamentals_facts`), and `zips_ingest_insider >> insider_edgar`. `zips_ingest_insider` (`zips-ingest --only insider`) and `zips_ingest_datasets` (`zips-ingest --only notes --only ftd`) run in `sec_bulk`, `edgar_index` and the walks in `sec_api`. `identity` and `price_history` run on ALL_DONE: a failed download leaves the build on its cached evidence, a failed build leaves prices on the stored master. A failed `edgar_index` holds the seven walks. The derived `sec_8k_votes` follows its parent and `sec_fundamentals_history` follows `sec_fundamentals_facts`; `fundamentals_sharadar` waits only for `identity`, with no path to the SEC tasks; `fundamentals_check` runs after both histories on ALL_DONE; the employee task feeds `extraction_status`. `sharadar_actions` (`sharadar-actions`) is an independent market-wide fetcher; its edges and the DAG schedule are on [DAGs and infrastructure](../modules/dags-and-infrastructure.md). The independent 13F manager chain and the other identity-independent sources run beside the stage. **Hard identity gate:** fetchers → `extraction_status` → `identity_check` → `trigger_data_aggregation`. The first two run on ALL_DONE once every fetcher has finished; `identity_check` (`validate identity`, report under `reports/validate/identity-nightly`) fails on foreign rows or a broken lineage invariant and abstains on an empty lineage, and the trigger runs only on its success. Extraction itself never waits for it. `StepExtractAllData` runs the same stage (zip downloads, `StepExtractIdentity`, zip ingests) before its domain steps. A standalone consumer command reads the stored lineage and master, so run `identity` first.

For schedules and triggers, see [DAGs and infrastructure](../modules/dags-and-infrastructure.md) and [nightly refresh](../flows/nightly-data-refresh.md).

### Airflow GPU setup

The [Airflow Dockerfile](../../airflow/Dockerfile.airflow) explicitly installs `torch==2.14.0+cu126` from the CUDA 12.6 wheel index and checks `torch.version.cuda == "12.6"` at build time. The same image can run on CPU. [SentimentEngine](../../src/data_aggregate/utils/text/nlp_sentiment.py) selects the device with `torch.cuda.is_available()`, moves both the classifier and input tensors there, and logs the requested device, PyTorch version, compiled CUDA version, availability and GPU name. `FINBERT_DEVICE=cpu` forces CPU. Unset selects CUDA when available; an explicit `cuda` propagates initialization failures rather than silently scoring on CPU or skipping sentiment.

On the Windows NVIDIA machine, use Docker Desktop's WSL2 Linux backend with a compatible NVIDIA driver and GPU passthrough. An installed host CUDA toolkit alone does not expose a GPU to a container. The base [Compose file](../../docker-compose.yml) reserves one NVIDIA GPU and sets `FINBERT_DEVICE=cuda` in the shared Airflow configuration. LocalExecutor runs the pipeline on the scheduler. Use the base Compose file for the commands below.

Build and recreate Airflow on the GPU host:

~~~powershell
rtk docker compose -f docker-compose.yml up -d --build airflow-scheduler airflow-webserver
~~~

On a corporate network, the optional `pip_ca` build secret lets pip use the existing ignored trust bundle for the PyTorch download without including the certificate in the image:

~~~powershell
rtk proxy docker build --secret id=pip_ca,src=.cache/corporate_ca_bundle.pem -f airflow/Dockerfile.airflow -t pea-airflow:latest airflow
rtk docker compose -f docker-compose.yml up -d airflow-scheduler airflow-webserver
~~~

Before running the DAG, check the **pipeline** interpreter and execute a real CUDA kernel inside its scheduler:

~~~powershell
rtk docker compose -f docker-compose.yml exec airflow-scheduler /opt/pipeline/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available(), 'GPU is not visible to pipeline PyTorch'; print(torch.cuda.get_device_name()); x=torch.ones(2, device='cuda'); assert (x+x).cpu().tolist()==[2.0,2.0]; print('SANITY: CUDA allocation and kernel passed')"
~~~

Then check real FinBERT inference through the same engine used by `build-text` (the first invocation may cache model weights):

~~~powershell
rtk docker compose -f docker-compose.yml exec -w /opt/airflow/project airflow-scheduler /opt/pipeline/bin/python -c "import logging, math; from src.data_aggregate.utils.text.nlp_sentiment import get_sentiment_engine; from omegaconf import OmegaConf; logging.basicConfig(level=logging.INFO); c=OmegaConf.load('configs/build_cube.yml'); e=get_sentiment_engine(c.build_cube.earnings_calls.sentiment_model); assert e is not None and e.device=='cuda'; p=e.score_texts(['Revenue and operating income increased.'])[0]; assert p and all(math.isfinite(v) and 0<=v<=1 for v in p.values()) and abs(sum(p.values())-1)<1e-5; print('SANITY: FinBERT CUDA inference passed', p)"
~~~

These checks do not build the cube or write application tables. A successful image build proves CUDA support was installed; only the GPU-host kernel and inference checks prove the driver's runtime and GPU architecture work. The existing score cache is reused; changing device does not require a full text rebuild. The routing/failure checks live in [test_nlp_sentiment.py](../../tests/utils/test_nlp_sentiment.py).

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

A register or manual-lineage edit needs no hand-run reparse. After the edit, which must cite an SEC filing accession, rebuild and save the identity tables (`identity --only publish`, which rebuilds every stage from the stored inputs), read `identity --only propagate --dry-run`, then run `identity --only propagate` (unscoped) within 7 days: bulk families re-parse the changed tickers from cache, foreign rows are purged with one WARNING per table, tape and insider rows are re-stamped, and each EDGAR fetcher lists a new CIK's filings on its next run. Finish with `validate identity`.

**Config-only changes move no stamp.** A changed co-registrant, form policy, `reverse_acquisitions` or `vendor_series_overrides` entry reaches stored rows only through `identity --only propagate --full`. Tickers named in `merger_metadata` or `co_registrants` of [security_master_manual.json](../../configs/sec/security_master_manual.json) are re-stamped on every propagation, so an added or changed entry needs nothing more; a **removed** entry drops its ticker from that set, so run `identity --only propagate --full`, then `zips-ingest --only insider -F` and `insider-edgar -F` to restore purged co-registrant rows. After a manual issuer override for a conflict CUSIP, run `zips-download --only ftd -F`, `identity --only capture-evidence -F`, `identity --only publish`, `zips-ingest --only ftd -F`.

**Rollback.** Purged rows are not archived. A wrong purge is undone by fixing the lineage (`configs/sec/registrant_cutover.json`, `entity_lineage_manual.json` or `symbol_tenure_manual.json`), rerunning `identity`, and re-fetching: the changed tickers' new stamp makes the EDGAR fetchers relist them and the bulk families re-parse them. Re-fetching `def14a_*` LLM rows costs API calls. Raw FTD and RegSHO rows are re-stamped in place, and deleted only when their security leaves scope; a RegSHO day older than FINRA's rolling window cannot be re-fetched.

### Identity first build (one-off, user-run)

The identity tables are `universe_cik_symbol`, `universe_cusip` and the identity columns of `universe_tickers`. No DDL step: `DataStore.save` and `replace` create a missing table, so the first run recreates them. Run once, by hand, with the extraction DAG paused, one heavy process at a time:

0. **Back up** the tables you drop: `pg_dump -U alexandre -d pea -t tickers_universe -t universe_tickers -t entity_lineage -t security_master -t identity_history -t sec_fails_to_deliver -t sec_fails_to_deliver_security -Fc -f identity_pre_redesign.dump`.
1. **Drop** the old identity tables (`tickers_universe`, `entity_lineage`, `security_master`, `identity_history`, any partial `universe_tickers`, `universe_cik_symbol`, `universe_cusip`) and both FTD tables. Keep `symbol_tenure`, `sec_cik_profile`, `sec_short_volume_security`, the `sharadar_*` tables, `sec_company_tickers`, every fact table and the FTD zip cache (the only source of the FTD rebuild).
2. **Universe:** `universe` (sec-tickers, sharadar-tickers, sharadar-actions, sec-submissions, then the sync: 491 rows, identity columns still NULL). Then `zips-download` (every listed FTD period while no lineage is stored).
3. **Identity cold start**, in order: `identity` (the CUSIP history reads the FTD zips after this run's symbol history, so `universe_cusip` is published in the same run; propagation checks every ticker); `zips-ingest --only ftd -F` (both FTD tables from the cache under the new lines); `identity --only propagate --dry-run`, then `identity --only propagate`.
4. **Gate:** `validate identity`. It applies the JSON `resolutions` as the build does; a FAIL must name only rows the propagation still has to purge.
5. **Resume the DAG.** The first nightly run rewrites every stored 13F accession holding a second share class (one re-read each from the local edgartools cache, U-67).

**Rollback.** `pg_restore -U alexandre -d pea --clean --if-exists identity_pre_redesign.dump` with the DAG still paused.

### 13D/13G

Both form-string eras are in the contract forms, so a normal run lists every missing filing; `-F` re-reads stored ones too. The filing index lists a 13G under every party, so a universe bank or asset manager also lists the thousands of 13Gs it filed on other issuers; each is read once and stored as a filer-role marker, so the first backlog run pays that cost. Chunk a 13G `-F`.

### Insider transactions

`insider_transactions` is filled by two commands: `zips-ingest --only insider` loads the pending zip quarters (only the filings EDGAR lacks), then `insider-edgar` reads every indexed Form 3/4/5 of each ticker's lineage filed from the earlier of the day after the last stored zip quarter and the run date minus 7 days that is not yet stored from EDGAR. The stored-row identity sweep runs only on a full-universe run; a `-t` run skips it. `zips-ingest --only insider -F` re-reads every cached zip quarter after a zip parse change without an EDGAR walk; `insider-edgar -F` re-reads the EDGAR window, including filings already stored from EDGAR. A full rebuild drops the table first, then runs both with `-F` (see [large backfills and recovery](./large-backfills-and-recovery.md)). Read the per-quarter `filings missing from EDGAR` WARNING and the one identity-exclusion WARNING per run; there is no parity or promotion step.

### Earnings-call rebuild

Normal extraction checks the entire scoped provider index and can recover introduction-verified raw gaps without a separate backfill script or date cutoff. The recovery summary separates rejected source rows, selected raw calls, new/reissued/already-stored calls and paragraph writes. Canonical occupancy comes from normal provider calls rather than DB presence, so reruns retain recoveries consistently. Required intro/null-ID comparison reads finish before full-reconcile deletion. Raw admission preserves the dated feature identity checks and the configured availability clock; source recovery counts do not establish model-feature coverage.

After separate production approval, adopt a lineage/availability change with current identity tables, lineage-aware extraction, a full `data_aggregate build-text -F`, then `data_aggregate assemble-cube`. `build-text` has no ticker scope. Ordinary extraction recovers missing historical aliases but preserves old raw keys; reviewed raw cleanup requires the scoped full reconcile. Runtime issuer checks exclude unsupported legacy calls from features in the meantime. New/reissued calls follow normal sentiment/embedding scoring; valid current-version caches can be reused for a clock-only change. No raw/cache `as_of` is shifted.

The ten-column role/embedding contract uses the same adoption sequence: `data_aggregate build-text -F`, then `data_aggregate assemble-cube`, then read-only validation and fresh training for both tree families. Existing compatible speaker-clean-v3 prepared/legacy-management scores and vectors are reused; only missing question sentiment needs new inference. Saved v2 outputs still require the prior cleaning-version recomputation. The feature-schema change alone needs no extra embeddings or source-date shift. Existing nullable unused sentiment uncertainty storage can remain until database recreation.

The feature clock is owned by `build_cube.earnings_calls.availability_lag_days` in [build_cube.yml](../../configs/build_cube.yml): two calendar days, followed by the first eligible trading session. A normal 30-calendar-day rewrite does not rewrite older feature clocks; a full part rebuild and cube reassembly are necessary for historical adoption. The provider exposes call dates without a first-available timestamp, so two days is an estimated buffer. The 2026-10-19 scoped validation recovered ECHO2→62 and BNY3→78 into an isolated sink, but current-model feature generation abstained because the bounded cache contained only older model versions. Source-format and old-feature drift flags remain documented in `reports/validate/2026-10-19-earnings-call-lineage/`; no production backfill or paid scoring ran in that validation.

A full text run reuses valid model results and scores only missing or invalid outputs. Cold scoring can be dominated by FinBERT. The following measurements are historical, from 2026-10-02 (`reports/validate/2026-10-01-earnings-call-extraction-gaps/03-implementation.md`); they are not a throughput guarantee for the current flow. The later [text-cube acceptance](./testing.md#text-cube-acceptance-2026-10-22) records actual configured-model/PostgreSQL tests on 253 calls, a 10.3-second warm median and zero repeated inference. Whole-universe cold capacity remains unmeasured.

| Stage | Cost | Resume |
| --- | --- | --- |
| `extract-earnings-calls -F` | 18.6 min for 33,591 calls, mostly DB writes | Writes one row group per batch, so a crash loses at most one batch. |
| FinBERT sentiment | 27–37 s per call on CPU, about 280 h for every call | Historical CPU timing; the current flow saves one completed call at a time under the configured model plus the speaker-clean-v3 cleaning version. Existing v2 scores/vectors need recomputing; use build-text -F and subsequent cube assembly to adopt the cleaning across historical feature rows. |
| OpenAI embeddings | about 9,150 tokens per call, about $6 for every call; about 100 calls/min, bound by `float8[]` inserts | Complete calls are skipped on (ticker, quarter, model tag). |

Run FinBERT on a GPU with `FINBERT_DEVICE=cuda`. `build-text` scores sentiment before it embeds and has no `-t` scope; `-F` rebuilds `cube_part_text` for the universe, after which the cube must be assembled again. Then validate the part:

```bash
rtk "$PY" -m src validate earnings-calls -T cube_part_text -o reports/validate/YYYY-MM-DD-earnings-calls
```

The check reads `earnings_call_sections` itself; `-T earnings_call_sections` is the wrong target. It reports per-ticker and per-calendar-quarter coverage, paragraph-grain checks, raw split status, usable-call rate (at least 0.985 under `usable_call_rate_min`) and prepared word share among eligible both-section calls (q05 at least 0.15, from `configs/validate.yml`), `as_of` against the nearest `earnings_surprises` date, and the ten-column feature schema, bounds, redundancy and drift.

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

## Offline portfolio report and scripts

Run portfolio backtest with the exact assessed --run-id and open <run>/backtest_portfolio/report.html locally. Configure fixed LS/S&P/cash percentages and cash remuneration in [portfolio.yml](../../configs/portfolio.yml); changing them requires a rerun. The report itself filters N/cadence without a server. No Streamlit application or daily ledger command remains.

Regenerate managed DDL with `rtk "$PY" sql/generate_schema_sql.py` ([generator](../../sql/generate_schema_sql.py)) after an approved table-registry change. Treat one-off scripts as operational instruments: read their arguments, mutation behavior, and rollback notes before execution.

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

## Dated modelling assessment

Run the [four weekly commands](../../src/modelling/cli.py). In a linked checkout, first [set ROOT_PATH in the parent shell](#modelling-and-portfolio) and retain it through validation:

~~~bash
rtk "$PY" -m src modelling train -c ./configs
rtk "$PY" -m src modelling assess -c ./configs
rtk "$PY" -m src portfolio backtest -c ./configs
rtk "$PY" -m src modelling full-train -c ./configs
~~~

Train creates a fresh run ID and emits it as the final successful stdout line. It saves original fold/cutoff models and basic IC/IR summaries. Assess selects the latest completed training and independently recomputes all scores, metrics, explanations and HTML. It logs the original training UTC/ID and warns strictly after24hours. Portfolio/full-train default to the latest successfully assessed training. Do not continue after a failure.

For concurrent/manual work, pass the exact returned ID to downstream --run-id options. The DAG already pins that exact ID through native XCom, including downstream retries. Train has no --run-id or partial resume; assess/full-train have no --resume, and assess has no --report-only. Every assessment recalculates everything. Every full-train invocation creates a fresh refit attempt using saved fitting settings, resolved schema, seed and mean-last-two-fold rounds.

Dated output is model.assessment.output_dir/<run_id> (normally data/output/modelling_backtest/<run_id>). Fresh assessment/ and report.html remain inside that run; portfolio/ and full_history/<attempt_id>/ share its identity. Assessment builds in temporary space, preserves the previous successful result during computation and uses bounded backup/rollback during replacement. Inspect retained backup/lock evidence after interruption; the next assess still starts its calculations anew. latest.json indicates assessment success only.

Production uses model.models_dir. For a rehearsal, copy configs and isolate both assessment.output_dir and models_dir before invoking full-train; a normal full-train publishes production. Training no longer writes historical prediction/signal/CV tables. Daily predict still writes predictions_latest.

~~~bash
rtk "$PY" -m src modelling assess -c ./configs --run-id <returned-id>
rtk "$PY" -m src validate modelling -c ./configs -o data/output/modelling_backtest/<returned-id>
~~~

The [read-only validator](../../src/validate/checks/modelling.py) checks source chronology, predictions/IC/IR/effects, complete publication and report content. Incomplete evidence is ABSTAINED. Retain accepted historical-vintage limitations and missing configured inputs. Old assessments are not migrated.

Text model stages compare all compact source keys with their own saved model tables, so old missing/failed calls remain eligible for scoring independently of the feature rewrite. Normal text features rewrite 30 calendar days from their stored frontier and read only active emitter calls and their actual adjacent fiscal predecessors. Changes older than that output range require build-text -F followed by assemble-cube. Full feature mode reuses valid model results; it does not force paid embedding or FinBERT work to repeat. Prepared-only/answer-only calls retain supported features, with unavailable sentiment roles recorded as NULL scores and zero words.
