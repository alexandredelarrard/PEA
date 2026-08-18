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
| [definition_of_done.md](definition_of_done.md) | you are about to call a task finished |

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

## Finishing a task

An "important" task is not done when the code runs — it is done when a report exists. See
[definition_of_done.md](definition_of_done.md) for the three report types (REFACTOR / DATA /
MODELLING), the mandatory sections, and the `Stop` hook that enforces them.
