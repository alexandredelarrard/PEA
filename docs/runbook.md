# Runbook

Scope: how to actually execute anything in this repo. Read this before running a command.

## The Python interpreter

**`python`, `poetry` and `conda` are NOT on PATH** in this environment (`python`/`python3` resolve to
the Windows Store stub, which fails). Call the Poetry venv executable directly:

```bash
PY="$HOME/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe"
"$PY" -m pytest tests/... -v -s
"$PY" -m src data_extract --help
```

Run from the **repo root**, so `./configs`, `.env` and `src` resolve.

If the venv hash (`lkf53h9P`) changed, re-find it:
`ls "$HOME/AppData/Local/pypoetry/Cache/virtualenvs/"` → the `stock-pick-strat-*` entry.

Project requires **Python ≥3.13**. Shell here is Git Bash (POSIX); PowerShell is the user's primary.

## The database

Container **`pea_db`** (postgres 16.14) → `localhost:5432`, database **`pea`**.

> The data **volume was transferred from another machine**, so the superuser role is **`alexandre`**,
> not the compose default `pea`. Postgres ignores `POSTGRES_USER` when the volume already exists, so
> `role "pea" does not exist`. There is no `postgres` role either, so `docker exec -u postgres` fails.

### Best route — no password at all

`pg_hba.conf` in that volume has `local all all trust`, so a connection over the container's **Unix
socket** is trusted:

```bash
MSYS_NO_PATHCONV=1 docker exec pea_db psql -U alexandre -d pea -c "SELECT count(*) FROM prices;"

# for a DataFrame: wrap the query and read stdout
MSYS_NO_PATHCONV=1 docker exec pea_db psql -U alexandre -d pea \
  -c "COPY (SELECT … ) TO STDOUT WITH CSV HEADER"

# multi-statement: pipe a heredoc to stdin (docker cp of a Windows path does not work)
MSYS_NO_PATHCONV=1 docker exec -i pea_db psql -U alexandre -d pea <<'EOF'
SELECT …;
EOF
```

`MSYS_NO_PATHCONV=1` is required — Git Bash otherwise mangles the container-side paths.

Prefer this over asking for the password: it needs no secret and changes nothing about the DB's auth.
**Never `ALTER ROLE` or edit `pg_hba.conf`** — those are security settings.

### Host connection (needs the password)

A host connection via the mapped port arrives from the Docker gateway IP, so it misses the
`127.0.0.1` trust lines and hits `host all all all scram-sha-256`. `.env` does not carry the password
for this transferred volume, so ask the user for it.

`src/utils/db.py::database_url` uses `URL.create`, which URL-escapes `!` correctly — the old
string-mangling problem is fixed. If you ever hit it again in an ad-hoc script, connect via a
psycopg2 creator instead of a URL string:

```python
eng = create_engine("postgresql+psycopg2://", creator=lambda: psycopg2.connect(
    host="localhost", port=5432, user="alexandre", password=PW, dbname="pea"))
```

`get_config_context()` builds `ctx.store.engine` from `database_url()`, which defaults to
`pea/pea@5432` (wrong for this volume) unless `.env` / `DATABASE_URL` overrides it. For a host
script, build your own engine and inject the loaded frame.

### Lifecycle

```bash
docker compose up -d db          # recreates on the EXISTING volume — no data risk:
                                 # sql/ initdb scripts run only on an EMPTY data dir
docker ps -a                     # pea_db + the four Airflow containers (currently exited)
docker exec -it pea_db vacuumdb -U alexandre -d pea --full
```

Volume backup (from `main.py`'s notes):
`docker run --rm -v stock_pick_strat_pgdata:/volume alpine tar czf - -C /volume . > backup.tar.gz`

**Never blanket-kill `python.exe` by image name** — that has already killed a multi-hour SEC download.
Kill by PID only.

## Environment

A git-ignored `.env` at the repo root (template: `.env.example`):

```dotenv
SEC_USER_AGENT="Your Name your.email@example.com"   # REQUIRED by SEC EDGAR
FRED_API_KEY=...
OPENAI_API_KEY=sk-...                                # DEF 14A extraction + embeddings
OPENFIGI_API_KEY=...                                 # optional: speeds 13F CUSIP mapping
# DB: override with POSTGRES_* or DATABASE_URL
```

Loaded by `Context._load_env` via `find_dotenv(usecwd=True)`, and by `tests/conftest.py` via an
explicit path so it works from any cwd.

## The CLI

`"$PY" -m src <package> <command> [-c ./configs] [-t AAPL,MSFT] [-F/--force|--full]`

Shared options: `-c/--config-path` (default `./configs`), `-t/--tickers` (default = the full
`sp500_tickers` universe), `-f/--force` or `-F/--full`. Every command builds a fresh `Context`.

### `data_extract` — one command per source

```
seed-universe              # MUST run first; everything resolves the universe from sp500_tickers
price-history              # prices + dividends (HEAVY)
short-interest  fails-to-deliver  macro
thirteen-f                 # 13F bulk + OpenFIGI cusip map (HEAVY)
superinvestors             # needs 13F
fundamentals               # both layers: facts (network, HEAVY) then history (replay)
fundamentals-facts         # SEC XBRL per-filing -> fundamentals_facts (+ headcount). HEAVY
fundamentals-history-sec   # fundamentals_facts -> fundamentals_history_sec + _reason_codes. No network
fundamentals-sharadar      # the WHOLE Sharadar producer: tickers -> SF1 -> actions -> sp500
                           #   -> the MERGED fundamentals_history. Needs SHARADAR_API_KEY
sharadar-tickers  sharadar-actions  sharadar-sp500   # one table each, for a targeted refresh
fundamentals-history-merged # rebuild ONLY the merged table. No network; -F deletes first
sharadar-gap-check         # READ-ONLY: where Sharadar and the SEC layer disagree. --propose
sharadar-diagnostics       # READ-ONLY acceptance gates -> a markdown report. Writes no data
earnings-surprises  financial-statements  insider-transactions
financial-notes            # VERY HEAVY
def14a                     # LLM-parsed governance (costs OpenAI calls)
sec-8k-items  sec-13d  filing-text
wiki-pageviews  google-trends
download-earnings-calls    # to disk, no DB (HEAVY)
ingest-earnings-calls      # cached transcripts -> earnings_call_sections; -F re-parses all
```

There is **no `employees` command** — headcount comes from the same 10-K as `fundamentals`.

### `data_peers`

```
deduce-peers               # -> paths["SECTOR_PEERS_PATH"] (data/output/sector_peers.json)
```

### `data_aggregate` — the seven-step cube build

```
build-prices          # -> cube_part_prices                      (the ONLY reader of raw prices)
build-target          # -> cube_part_targets + cube_part_betas
build-fundamentals    # -> cube_part_fundamentals
build-momentum        # -> cube_part_momentum
build-text            # -> cube_part_text
build-extras          # -> cube_part_extras
assemble-cube         # read the parts -> composites -> the `cube` table
build-cube            # all seven in ONE process (what main.py does)
cube-status           # JSON status of every part; exit 2 if any part is behind
```

Each build command is **incremental by default** (reads its part's latest date, recomputes a
warm-up-padded trailing window, appends). `-F/--full` forces a rebuild.

### `modelling` / `portfolio`

```
modelling train [--train-start YYYY-MM-DD] [--train-end YYYY-MM-DD]
modelling full-train                  # production refit on ALL history
modelling predict [--n-dates N]        # -> predictions_latest
portfolio backtest                     # -> data/output/portfolio/
portfolio strategy-moves               # -> the `strategy` ledger
```

### Other

```bash
"$PY" -m scripts.generate_schema_sql          # regenerate sql/schema.sql from registry + reflection
streamlit run app/app.py                       # the dashboard (assumes pre-trained models)
```

## First-run / cold-start order

```
1. docker compose up -d db
2. .env in place (SEC_USER_AGENT at minimum)
3. python -m src data_extract seed-universe
4. the extraction commands you need (price-history first — everything downstream needs prices)
5. python -m src data_peers deduce-peers
6. python -m src data_aggregate build-cube --full
7. python -m src modelling train  then  full-train
8. python -m src portfolio backtest
```

## Airflow

The four Airflow containers are **currently exited**. Airflow runs in its own image
(`Dockerfile.airflow`, apache/airflow on Python 3.12) with the pipeline in an **isolated venv at
`/opt/pipeline`** — its deps must never clash with Airflow's (Airflow pins SQLAlchemy <2.0; the
pipeline needs 2.0). DAGs therefore call
`/opt/pipeline/bin/python -m src <package> <command>` with `cwd=/opt/airflow/project`.

```bash
docker compose up -d airflow-db airflow-init
docker compose up -d airflow-scheduler airflow-webserver     # UI on :8080, admin/admin
```

Container path mapping: repo → `/opt/airflow/project`, `./data` → `/opt/airflow/project/data`
(host-persistent download cache), `./src/dags` → `/opt/airflow/dags`. `ROOT_PATH=/opt/airflow/project`
and `POSTGRES_HOST=db` (the compose service name, not `localhost`).

Pools created by `airflow-init`: `sec_bulk` 2, `sec_api` 2, `scrape` 2, `aggregate` 3.

## Auditing fundamentals resolution — the 52-ticker sweep

Two committed instruments, because every acceptance number in the rebuild's Phases 3b, 4 and 4b
came from scratchpad scripts that no longer exist. The sweep pays the network cost once; the
report is offline and re-derivable in seconds.

```bash
"$PY" scripts/sweep_fundamentals_resolution.py --roster both --workers 4 --limit 4
"$PY" scripts/report_fundamentals_sweep.py [--roster in_sample|out_of_sample|both]
```

`--roster` names a list in `configs/fundamentals/fundamentals_rosters.json`. One network pass
resolves each filing **twice** — with and without 4c.1's `prefer_structure` guard — off a single
`filing.xbrl()`, which is what makes a before/after acceptance one sweep rather than two. Ledgers
land as one parquet per ticker under `data/fundamentals_sweep/` (gitignored), so a run resumes;
`--refresh` re-sweeps tickers that are already cached.

- **`--limit N` in a shell loop is mandatory, not a convenience.** edgartools never releases its
  per-filing caches inside a process: an all-52 single run reached **14.7 GB RSS**. One process
  per 4 tickers keeps it under ~4 GB, and a killed batch costs at most 4 tickers.
- **Run one driver at a time.** `to_parquet` is not atomic, so two drivers writing the same
  ticker corrupt the ledger rather than merely wasting CPU. Stopping a shell task leaves the
  loop's python children alive — kill them **by PID** (never by image name; that has destroyed a
  multi-hour SEC download here before) and confirm none remain before restarting.
- Wall clock: ~5-9 min per batch of 4 at `--workers 4`, so ~60-90 min for all 52. It is
  CPU-bound in XBRL parsing, not network-bound, so more workers than cores does not help.
- The sweep honours `fundamentals_cik_cutover.json`. Without that it would walk only the current
  registrant and measure APA at 22 filings instead of 65 — a pipeline nobody runs.

`measure_total_liabilities_legs.py` used to sit here as a third instrument. It was DELETED in
plan-5b: its finding is now a standing constraint in `cross_identity`'s docstring -- 0 of 44
10-Ks declare a `Liabilities` total, leg-sets vary by filer *and* by year, and an unlisted
us-gaap sibling is dropped silently, so `totalLiabilities` stays an identity and never a
leg-sum. Note that the CONSTRAINT is preserved and the MEASUREMENT is not: the script read the
calculation linkbase over the network, which no check does. Recover it from git history if the
leg-set question is ever reopened.

## Backfilling fundamentals from scratch — chunk it, and pass `-F`

```bash
for CHUNK in "AAPL,CSCO,KR,XOM,APA,EOG" "VLO,JPM,BAC,MTB,USB,MET" ...; do
  "$PY" -m src data_extract fundamentals-facts -F -t "$CHUNK"
done
"$PY" -m src data_extract fundamentals-history-sec -t "<all of them>"
```

- **Chunk into separate PROCESSES** for the same reason the sweep does: edgartools never releases
  its per-filing caches inside one.
- **`-F/--full` is not optional.** The run manifest's incremental test is "did the ticker universe
  change size since the last run?", so two consecutive 6-ticker chunks look like a repeat of one
  run and the second gets `since = last run`, i.e. nothing. Measured: chunk 1 wrote 31,540 rows
  and chunks 2-9 wrote **0**.
- The history build is a separate command because it costs no network: a bug in the history layer
  is fixed with `fundamentals-history-sec --rebuild-history -t APA`, and only a bug in the RESOLUTION
  layer needs `fundamentals --rebuild -t APA`, which deletes all four tables and refetches.
- Wall clock: ~7 min per 6-ticker chunk of facts, ~2.5 min per ticker for the history replay.
- **A SCHEMA change to any of the four tables needs `scripts/recreate_fundamentals_tables.py`
  first.** `sql/schema.sql` runs only when Postgres initialises a volume; on a live one
  `store.save` creates a missing table by inferring dtypes from the first frame it is handed, so
  an all-None column becomes TEXT and every later ticker's number is stored as a string. The
  script drops and re-creates the four tables from the committed DDL -- destructive, `--dry-run`
  first, `--yes` to apply -- and is what a PK or column-contract change is applied with.

Then check what you built -- with the VALIDATOR, which absorbed the eight gates that used to
live in `scripts/verify_fundamentals_history.py`:

```bash
"$PY" -m src validate fundamentals --tier 1 [-t AAPL,JPM]
"$PY" -m src validate fundamentals --roster in_sample --roster out_of_sample  # --roster REPEATS
"$PY" -m src validate report --run-id 3df52ae9af75   # re-render a run; NO re-run, NO writes
"$PY" -m src validate status set <cluster_id> --note "..."   # a wontfix; a NUMBER is enforced
"$PY" -m src validate status clear <cluster_id>
"$PY" -m src validate checks                      # what does this tool actually test?
```

Reports default to `reports/validate/YYYY-MM-DD/<scope>.md` with a `.json` beside it -- markdown
for a human, JSON for an agent. Findings are ranked as **clusters**: one `(ticker, field)`
defect, with every check that fired on it as corroborating evidence rather than as separate
work. 11,926 findings on the calibration roster are 2,323 clusters in 50 field families.

Two runs can only be differenced when their **scope hashes match** (same tickers, fields and
tiers). The report omits the delta, with a reason, when no comparable prior run exists -- a
first run must never render as a trend.

Read-only against every table but the three the validator owns, and **it gates nothing** -- the nightly
build runs to completion whatever it finds (plan-5b decision 45). The eight §5.8 gates are now
Tier-1 checks: `grain` (no duplicate `(ticker, as_of)`, `fiscal_end` monotone, no look-ahead),
`column_contract` (the 69 columns, IN ORDER), `unexplained_null` (no NULL cell without a
`fundamentals_reason_codes` row for its own `(ticker, as_of, field)`), `filing_lag`,
`amendment_ledger` (how much the 365-day cutoff refuses), `same_day_collapse`, `coverage_field`
(per-regime coverage, with the absence oracle absorbed from `audit_absence_evidence.py`) and
`code_vocabulary`.

**Read the check-health gate before the rankings.** It renders above them for a reason: a
cluster list drawn from a mis-calibrated run reads as authoritative regardless. A check marked
ABSTAINED examined nothing, which is not a pass; a check marked THRESHOLD BUG is above its own
declared ceiling and is burying real findings under itself. When either is present the report
banners that the rankings may be inflated. `src/validate/README.md` is the operating manual, and its §4 --
"when it does not work" -- is the part worth reading twice.

## Finishing a task — the definition-of-done report

Contract and rationale: [definition_of_done.md](definition_of_done.md). Pick the generator that
matches the work; each writes `reports/<YYYY-MM-DD>/<slug>__<TYPE>.md` (one folder per day) and
prints what still needs a human sentence.

```bash
"$PY" scripts/dod/refactor_metrics.py --slug <slug> --tests tests/path/test_x.py
"$PY" scripts/dod/data_profile.py     --slug <slug> --tables fundamentals_facts [--tickers AAPL,JPM]
"$PY" scripts/dod/modelling_report.py --slug <slug> [--compare-run <run_stamp>]
```

Then write §1, §5 and §6 yourself. **Never edit the ` ```json dod-metrics ` block** — it carries a
`content_hash` the `Stop` hook recomputes.

The hook is **warn-only** by default: it records a verdict and never blocks. To enforce it, set
`PEA_DOD_MODE=enforce` or create `.claude/dod-enforce`. To turn it off entirely, create
`.claude/dod-disabled` or set `PEA_DOD=off`. Verdicts accumulate at
`%LOCALAPPDATA%\pea-dod\<repo-hash>\<session_id>\verdicts.jsonl` — read that file before flipping
to enforce, to see what it *would* have blocked.

## Gotchas

- **Postgres `DATE` columns round-trip as `datetime.date`, not `Timestamp`.** A parquet-cached test
  harness hides this entire bug class — verify against the DB.
- `load` raises on a missing/empty table. On a cold DB that is correct behaviour, not a bug.
- `iter_load` holds a pooled connection — exhaust or close it.
- The **running** `pea_db` container predates the move to the repo root and still binds the
  nonexistent `./stock_pick_strat/sql` as its initdb dir (`docker-compose.yml` itself is correct).
  Harmless while the volume has data; recreate the container before rebuilding the volume.
- The 13F / notes / insider bulk downloads are multi-GB and cached under `data/`. Check for the cached
  zip before re-running a fetcher.
- A full fundamentals quality-gate pass is **~60s/ticker ≈ 8 hours** for the universe, not "minutes".
  Scope it with `-t` when iterating.
