# Coding standard

Scope: how to write Python in `src/`. Short on purpose — the data rules are in
[data_conventions.md](data_conventions.md), the structural rules in
[architecture.md](architecture.md).


## Functions and variables 

- Write class, methods, functions extremely cleanly: the less code the better, edge cases are specific functions.
- Write explicit variable names as convention. `df_xx` for dataframes, transparent variable names to improve readability. 
- Split large function into unitary functions doing one unique goal. 
- Write small functions to be reusable. If used through different modules, avoid circular import and move the function to a `/utils/` folder or subfolder.

## Constants first

Before naming any column, key, URL, threshold, or date format: **grep
[src/constants/constants.py](../src/constants/constants.py)** (927 lines). Reuse if present; add it
there *before* referencing it if not. Never hardcode a global literal inline.

What belongs there: date formats (`DATE_FORMAT`, `DATE_FORMAT_COMPACT`), every SEC/FINRA/Trends/Fool/
Roic/Dataroma URL template, form lists (`FUNDAMENTALS_FORMS`, `SEC_8K_FORMS`), section keys,
model names (`FINBERT_TONE_MODEL`, `EARNINGS_CALL_EMBED_MODEL`), GICS sector/group names,
`SECTOR_KPI_SCOPE`, plausibility bounds (`SHARES_OUTSTANDING_MIN/MAX`, `EPS_ABS_MAX`, …),
ticker override/exclusion sets, `PANEL_KEYS`, freshness cadence thresholds.

What does **not**: table names (those live only in `schema.py` — no `*_TABLE` constants), and tunable
numbers (those live in `configs/` — see [config.md](config.md)).

## Logging

- **`self._log`** in a `Step` (or a `Strategy` — both set it in `__init__`). It is named
  after the **subclass's** module (`logging.getLogger(type(self).__module__)`), so a log
  line names the step that emitted it rather than `src.utils.step`.
- **`context.log`** in a helper that receives `context`.
- **Never `print()`** in `src/`. Fix any remaining print if still exists.
- `logging.getLogger(__name__)` at module level is accepted in the leaf util/builder modules that
  take no `context`; the logging config routes them the same way.
- **There is no `self._context.logger`.** `Context` exposes `.log`; `_context.logger` would raise
  `AttributeError`. Some instruction files still say otherwise — see
  [docs/README.md](README.md#known-documentation-drift-verified-2026-08-17).

Prefer `f{}` f-strings in log calls over %s.

## Swallowing exceptions

- The per-ticker / per-filing convention stays: **one bad ticker or filing must not abort
  a 490-ticker walk**.
- But a defect in THIS repo is not a bad ticker. `edgar_driver.PROGRAMMING_ERRORS`
  (`NameError`, `AttributeError`, `TypeError`, `KeyError`, `ImportError`) is **re-raised**
  by `edgar_driver._worker` and by `filing_rows`, which aborts the pool and fails the run.
  A `NameError` logged as a per-ticker warning cost NEM, MO and AIZ every fact they had
  while a 10.6 h run reported success.
- The exception is a **library** boundary: the `except` around `filing.xbrl()` still
  swallows every class, because absorbing malformed XBRL is what it is for. Narrow the
  `except` around OUR code, never around theirs.

## Typing & imports

- **Full type annotations on every signature**, parameters and return.
- `from __future__ import annotations` at the top of new modules (most of `src/` has it).
- **All imports at the top of the file.** No function-local imports.
- Fix cycle breaks, circular import. No import inside the function
- No cross-imports between `src/` subfolders — shared logic goes in `src/utils/`.

## Module docstrings carry the reasoning

This codebase's docstrings are unusually load-bearing: they record *why* a value is what it is, and
what broke last time. 

Be as clear and direct as possible. Keep to the strict minimum and give key ideas, not whole details. 

**When you change one of these, update its docstring in the same edit.** When you are tempted to
"simplify" one of them, read the docstring first — several explicitly say the duplication is
deliberate.


## What not to do

- Do **not** replace OmegaConf.
- Do **not** alter the `Step` inheritance architecture.
- Do **not** cross-import between `src/` subfolders.
- Do **not** hardcode inline strings, file paths, or table names.
- Do **not** stash a heavy frame on `self` in a cube sub-step (breaks the memory invariant).
- Do **not** reformat unrelated code, and do not widen a diff beyond the task.
- Do **not** mark work complete without the printed sanity-check conclusion in the test output.

## Risk zones — propose and get approval before editing

| File / directory | Why |
|---|---|
| `src/context.py` | global pipeline context; changes cascade everywhere |
| `src/utils/step.py` | base class for every step |
| `src/constants/constants.py` | global literals; a rename cascades downstream |
| `src/data_store/*`, `sql/schema.sql` | the DB layer & DDL; a schema rename affects all reads/writes |
| `configs/*.yml` | a schema change must be mirrored in consuming code |
| `data/` and the Postgres volume | non-recoverable model artifacts, caches, and data |
| `tests/data_aggregate/aggregate_fingerprint_baseline.json` | see [testing.md](testing.md) — regeneration is tightly gated |

## Prefer the smallest correct change

The explicit goal of this codebase's conventions is that an agent can make a **minimal, local**
change. Before writing new code:

1. Check `src/constants/constants.py` for the literal.
2. Check `src/utils/` and the relevant `<package>/utils/common/` for the helper.
3. Check whether a registry already drives what you are about to hand-list (`schema.py::Tables`,
   `parts.py::CUBE_PARTS`, `STRATEGY_REGISTRY`, `FORM_REGISTRY`). Adding a row to a registry is
   usually the whole change.
4. Write the test alongside the implementation.
5. Run the targeted test with `-v -s`.

## Communicating results

- Output **only** the new test's results plus its printed sanity-check conclusion. Do not paste full
  suite summaries unless asked.
- On a refactor, state which tests are affected and why, concisely.
- For a multi-step task, confirm each step before proceeding.

## Documentation synchronisation

`AGENTS.md`, `README.md` and `docs/*.md` must move together with the code. **Propose a new shared
convention and get approval before editing `AGENTS.md`**, which is capped at **70 lines** — to add a
line there, remove one or push the detail into `docs/`.
