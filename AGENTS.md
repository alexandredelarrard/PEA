`stock_pick_strat` — quant long/short S&P 500 pipeline: extract market / fundamental / governance /
alt-data into **PostgreSQL**, build point-in-time peer-relative features (the "cube"), train
cross-sectional models, blend strategy sleeves into one book. Package lives at the **repo root**.
Python 3.13, OmegaConf, pandas, SQLAlchemy 2.0 + Postgres 16 (Docker), LightGBM/SHAP, OpenAI, click,
pytest; Airflow in its own 3.12 container.

## Read the doc for your task FIRST, then the code

| Task | Doc |
|---|---|
| find where code lives / how a stage is wired | [docs/architecture.md](docs/architecture.md) |
| touch a table (PK, grain, date col, projection, freshness) | [docs/data_schema.md](docs/data_schema.md) |
| know what's **actually populated** in the live DB | [docs/database.md](docs/database.md) |
| add/debug a fetcher, or a source's quirks | [docs/data_sources.md](docs/data_sources.md) |
| read or write tabular data | [docs/data_conventions.md](docs/data_conventions.md) |
| find a knob / which YAML owns it | [docs/config.md](docs/config.md) |
| model, strategy or portfolio layer | [docs/modelling.md](docs/modelling.md) |
| write Python in `src/` (naming, logging, risk zones) | [docs/coding_standard.md](docs/coding_standard.md) |
| write or run a test | [docs/testing.md](docs/testing.md) |
| **execute** anything (interpreter, DB, CLI) | [docs/runbook.md](docs/runbook.md) |

## Hard rules (always apply)

- All tabular I/O via `self._context.store`. No `sqlalchemy` / `pd.read_sql` / `to_sql` /
  `store.engine` outside `src/data_store/` — a test enforces this.
- Never read a large table unprojected: `columns=`/`project=True` **and** `where=`/`since=`;
  `iter_load` for cube-sized reads.
- `load` RAISES on missing/empty. `optional=True` only where nothing is legitimate → branch on
  `is None`, not `.empty`.
- Table names live only in `src/data_store/schema.py` → `Tables.<name>`, never a string literal.
  Likewise never hand-list what a registry drives (`CUBE_PARTS`, `STRATEGY_REGISTRY`, `FORM_REGISTRY`).
- Literals (URLs, formats, thresholds) → `src/constants/constants.py`. Tunable numbers → `configs/`.
- Logging: `self._log` in a Step/Strategy, `context.log` in a helper taking `context`. Never
  `print()`. `Context` has `.log` — there is **no** `.logger`.
- Full type annotations; imports at top of file; no cross-imports between `src/` subfolders
  (shared code → `src/utils/`).
- Feature/economic tests use **real** data; only parsing math gets synthetic known-truth fixtures.
  Every model: TimeSeriesSplit CV + SHAP + printed OOF metrics before it's "done."
- A test isn't done until it **prints a sanity-check conclusion**. Report only the new test's output.
- Ask before editing risk zones: `context.py`, `utils/step.py`, `constants/`, `data_store/`,
  `sql/schema.sql`, `configs/`, `data/` + the Postgres volume, the aggregate fingerprint baseline.

## Code map

Every `src/` subfolder owns a `step_*.py` orchestrator: inherit `Step`, call
`super().__init__(context=context, config=config)`, expose `run()` as the only public method.
(`src/strategies/` sleeves implement `base.Strategy.run(PortfolioInputs) -> StrategyResult` instead.)

```
data_store/   the ONLY SQL — schema.py (table registry), store.py (DataStore), ddl.py
data_extract/ StepExtractAllData + 4 sub-steps + fetchers   -> raw tables
data_peers/   StepDeducePeers                               -> sector_peers.json
data_aggregate/ StepBuildCube: 7 sub-steps -> cube_part_*   -> cube
modelling/    long_short/ (trained ensemble), trend/, long_book/
strategies/   sleeves: ls_equity, eq_long_only, long_book, trend_cta
portfolio/    StepPortfolio (ERC blend), StepStrategyMoves (`strategy` ledger)
utils/ context.py constants/ dags/ cli.py   |   repo: configs/ docs/ tests/ app/ scripts/ sql/ main.py
```

## Running — `python`/`poetry` are NOT on PATH; run from the repo root

```bash
PY="$HOME/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe"
"$PY" -m pytest tests/path/test.py::test_fn -v -s      # -s shows the sanity print
"$PY" -m src <package> <command> [-c ./configs] [-t AAPL] [-F]
MSYS_NO_PATHCONV=1 docker exec pea_db psql -U alexandre -d pea -c "…"   # DB, no password
```

## Todo, each agent run 

Keep `AGENTS.md`, `README.md`, `src/CLAUDE.md`, `docs/*.md`, `.cursor/rules/*.mdc` in sync in the
same change. Propose new shared conventions before editing this file.