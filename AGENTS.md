`stock_pick_strat` — quant long/short S&P 500 pipeline: extract market, fundamental,
governance, ownership, and alternative data into PostgreSQL; build a point-in-time,
peer-relative feature cube; train models; backtest long/short, S&P 500 and cash budgets.
Package at the repo root. Python 3.13, OmegaConf, pandas, SQLAlchemy 2 + PostgreSQL 16,
LightGBM/SHAP, OpenAI, pytest; Airflow uses an isolated Python 3.12 environment.

## Read the wiki, query Graphify, then inspect code

| Task | Canonical page |
| --- | --- |
| find code or understand stage wiring | [System overview](wiki/architecture/system-overview.md) |
| touch a table, grain, PK, date, projection, or freshness rule | [Table catalog](wiki/reference/table-catalog.md) |
| know what is populated in the local DB | [Live database](wiki/reference/live-database.md) |
| add/debug a fetcher or source quirk | [Data sources](wiki/reference/data-sources.md) |
| read or write tabular data | [Data access](wiki/guides/data-access.md) |
| find a knob or owning YAML | [Configuration](wiki/reference/configuration.md) |
| change models, strategies, portfolio, or ledger | [Modelling and portfolio](wiki/reference/modelling-and-portfolio.md) |
| write Python under `src/` | [Coding standards](wiki/guides/coding-standards.md) |
| write or run a test | [Testing and validation](wiki/guides/testing.md) |
| execute CLI, DB, Docker, Airflow, or routine validation | [Run the pipeline](wiki/guides/run-the-pipeline.md) |
| run high-cost backfills or historical recovery | [Recovery guide](wiki/guides/large-backfills-and-recovery.md) |
| review accepted deferred work | [TODO](wiki/TODO.md) |

## Hard rules

- Prefix every shell command with `rtk`.
- After the wiki, run `rtk proxy graphify query "<question>" --budget 1500` before raw code searches; refresh missing/stale graphs per the [Graphify guide](wiki/guides/query-the-codebase.md).
- Read/write in-scope Markdown only through OpenKnowledge MCP; source code uses native tools.
- All tabular I/O uses `context.store`; SQL only in `src/data_store/`; project/scope reads; `iter_load` for cube-sized data.
- Put table reads at the top of each function; read each table once and reuse the result.
- Never add test-only, unused or speculative parameters; tests use the existing production interface.
- Table names live only in `src/data_store/schema.py`; use `Tables.<name>`.
- Consumers read merged `fundamentals_history`; validation reads `fundamentals_history_sec`.
- Keep single-function values local and domain-only values with their owner; `src/constants/` is for shared contracts; tuning goes in `configs/`.
- Never pass logger arguments; use `self._log`, `context.log`, or a module logger; never `print()` in app code.
- Fully annotate signatures; imports stay at module top; sibling `src/` packages do not cross-import.
- Fetchers plan through their `schema.Resume` contract and `resume.py`, never a full read or a file.
- Feature/economic tests use real data; exact parser/math tests use known-truth fixtures.
- A test is not complete until it prints a sanity-check conclusion; report targeted output only.
- Ask before editing `context.py`, `utils/step.py`, constants, data store/DDL, configs,
  `data/`, the PostgreSQL volume, or the aggregate fingerprint baseline.
- Keep this file at 70 lines or fewer and synchronize durable guidance with the wiki.
- Finish important data/model/output work with the relevant read-only validator and report.

## Code map

Every major `src/` package owns a `step_*.py` orchestrator that inherits `Step`, calls
`super().__init__(context=context, config=config)`, and exposes `run()` as its public method.
`StepLongShort` adds `run_train`/`run_predict`; portfolio orchestration lives in `steps/`.

```text
data_store/    only SQL; registry, facade, DDL
data_extract/  ordered source fetchers and six domain sub-steps
data_peers/    business/return peer baskets
data_aggregate/ eight cube parts plus streamed assembly
modelling/     steps/ (train, assess), transformers/ (models, diagnostics), utils/
portfolio/     steps/, transformers/, utils/: long/short, S&P 500 and cash backtests
validate/      all read-only validation code
utils/         shared cross-package code
repo root: configs/ wiki/ tests/ scripts/ sql/ main.py
```

## Running

```bash
PY="$HOME/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe"
rtk "$PY" -m pytest tests/path/test.py::test_fn -v -s
rtk "$PY" -m src <package> <command> [-c ./configs] [-t AAPL] [-F]
rtk docker exec pea_db psql -U alexandre -d pea -c "…"
```
