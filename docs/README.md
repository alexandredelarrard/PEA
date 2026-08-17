# docs/ — agent context index

**Read only the docs your task needs.** Each file below is self-contained and states its own
scope. `AGENTS.md` (repo root) is the always-loaded summary; these are the details it defers to.

| Doc | Read it when… |
|---|---|
| [architecture.md](architecture.md) | you need to find *where* code lives, or how a stage is wired (Step pattern, entry points, DAGs) |
| [data_schema.md](data_schema.md) | you touch a table: its PK, grain, date column, projection, freshness |
| [database.md](database.md) | you need to know what is **actually populated** in the live DB right now (row counts, coverage, missing tables) |
| [data_sources.md](data_sources.md) | you add/debug a fetcher, or need a source's quirks, keys and lag |
| [data_conventions.md](data_conventions.md) | you read or write tabular data (store facade, projection, incremental, point-in-time) |
| [config.md](config.md) | you need a knob: which YAML owns it and how the merge works |
| [modelling.md](modelling.md) | you touch the model / strategy / portfolio layer |
| [coding_standard.md](coding_standard.md) | you write any Python in `src/` (naming, logging, typing, risk zones) |
| [testing.md](testing.md) | you write or run a test |
| [runbook.md](runbook.md) | you need to *execute* something (interpreter, DB access, CLI commands) |

## The five rules that override everything

1. **All tabular I/O goes through `self._context.store`.** No `sqlalchemy`, `pd.read_sql`,
   `to_sql` or `store.engine` outside `src/data_store/`. Enforced by
   [tests/data_store/test_store_boundary.py](../tests/data_store/test_store_boundary.py).
2. **Never read a large table unprojected.** Always `columns=` / `project=True` plus `where=` /
   `since=`; `iter_load` for cube-sized reads. `sec13f_hr` is 21.7M rows; `cube` is ~570 columns.
3. **Table names live only in [src/data_store/schema.py](../src/data_store/schema.py).** Reference
   as `Tables.<name>`.
4. **Log via `self._log`** (in a `Step`) or **`context.log`** (in a helper taking `context`).
   Never `print()`.
5. **A test is not done until it prints a sanity-check conclusion.** See [testing.md](testing.md).

## Known documentation drift (verified 2026-08-17)

Three instruction files predate the repo restructure and contain claims that are **false against
the current code**. Trust `docs/` over them where they conflict:

| Claim | Where | Reality |
|---|---|---|
| `self._context.logger` | [src/CLAUDE.md:150](../src/CLAUDE.md), [AGENTS.md:52](../AGENTS.md) | `Context` has **no** `logger` attribute — it has `.log`. `Step` exposes `self._log`. `_context.logger` would raise `AttributeError`; zero call sites use it. |
| `store.existing_dates(...)` | [.cursor/rules/base.mdc](../.cursor/rules/base.mdc), [utils-config.mdc](../.cursor/rules/utils-config.mdc) | No such method. Use `max_date` / `bounds` / `distinct`. |
| `schema_registry.py`, `schema_sql.py` | [.cursor/rules/data-layer.mdc](../.cursor/rules/data-layer.mdc), [utils-config.mdc](../.cursor/rules/utils-config.mdc) | Renamed to `schema.py` / `ddl.py`. |
| `backtest.yml` | [.cursor/rules/utils-config.mdc](../.cursor/rules/utils-config.mdc) | Does not exist; the backtest is configured by `configs/portfolio.yml` + `configs/strategy/*.yml`. |
| Paths under `stock_pick_strat/` | README.md, AGENTS.md, app/app.py docstring | The package moved to the **repo root**. There is no `stock_pick_strat/` directory. |
| `sql/database.md` | [src/CLAUDE.md:33](../src/CLAUDE.md) | Moved to `docs/database.md`. |
