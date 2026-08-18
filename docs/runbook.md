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
short-interest  fails-to-deliver  market-prices  macro  macro-assets
thirteen-f                 # 13F bulk + OpenFIGI cusip map (HEAVY)
superinvestors             # needs 13F
fundamentals               # SEC XBRL per-filing + rebuild fundamentals_history (HEAVY)
earnings-surprises  financial-statements  insider-transactions
financial-notes            # VERY HEAVY
def14a                     # LLM-parsed governance (costs OpenAI calls)
sec-8k-items  sec-13d  filing-text
wiki-pageviews  google-trends
download-earnings-calls    # to disk, no DB (HEAVY)
ingest-earnings-calls      # cached transcripts -> earnings_call_sections; -F re-parses all
check-freshness            # the gate; JSON on the last stdout line, exit 2 when stale
```

There is **no `employees` command** — headcount comes from the same 10-K as `fundamentals`.

### `data_peers`

```
deduce-peers               # -> paths["SECTOR_PEERS_PATH"] (data/output/sector_peers.json)
```

### `data_aggregate` — the seven-step cube build

```
build-prices          # -> cube_part_prices + cube_part_market   (the ONLY reader of raw prices)
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
