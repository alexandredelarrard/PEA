---
title: Coding standards
description: Python structure, typing, logging, data boundaries, risk zones, and documentation rules.
type: guide
tags:
  - wiki
  - guide
  - coding-standards
---
# Coding standards

## Goal

Make small, local, typed Python changes that preserve pipeline architecture, data contracts, and operational evidence.

## Structure

Every major `src/` package owns a `step_*.py` orchestrator. A step:

1. inherits [Step](../../src/utils/step.py);
2. calls `super().__init__(context=context, config=config)`;
3. exposes `run()` as its only public operation; and
4. keeps implementation in package-local private helpers or composed sub-steps.

Class names use `StepPascalCase`; files use `step_snake_case.py`. Strategies are the deliberate exception and implement [Strategy.run](../../src/strategies/base.py). [StepLongShort](../../src/modelling/steps/step_long_short.py) is the second exception: it lives in `src/modelling/steps/` and exposes `run_train()` and `run_predict()` besides `run()` (which calls `run_train()`), because training and prediction share one step's configuration.

Do not cross-import between sibling `src/` packages. Shared logic belongs in the [runtime and shared utilities](../modules/runtime-and-shared-utils.md) package. The sanctioned service exception is [src/gpt_extract](../../src/gpt_extract/), which centralizes LLM clients, prompts, and embeddings.

## Functions and types

- Fully annotate every function and method signature.
- Put `from __future__ import annotations` at the top of new modules.
- Keep all imports at module scope; solve cycles rather than hiding imports inside functions.
- Variable names are lowercase (`N806`) and say what they hold: DataFrames carry a `df_` prefix (`df_prices`); scalars and collections use descriptive names (`n_filings`, `missing_tickers`).
- Functions and methods use lower snake case (`N802`); arguments use lower snake case (`N803`); local variables inside functions use lower snake case (`N806`); exception classes end in `Error` (`N818`). Capitals are reserved for module-level constants and class names.
- Prefer Python 3.13 syntax. Write unions as `A | B`, including runtime checks such as `isinstance(value, A | B)`. The runtime-check form is a project convention even though Ruff 0.16.9 removed `UP038`.

## Function design and data flow

- One function does one thing in at most 150 lines. A top-level orchestrator (a Step's `run()` or a `build_fundamentals_<x>` function) only calls the small step functions in order.
- Nesting is at most 3 levels; 2 is the standard. Flatten deeper blocks with guard clauses, early returns, or a helper.
- No pass-through functions: a function whose body only calls another function is removed, and callers call the target directly.
- Functions doing the same thing are declared once. Logic repeated in two or more places, including across fetchers, is extracted to the closest shared utils module (package `utils/` for one package, `src/utils/` across packages) and called from every dependent; speculative one-use abstractions remain discouraged.
- Pass a value computed in one function to the next; never re-declare or recompute it elsewhere.
- Data flows in the golden-standard order: load → clean → compute features or steps → post-process → save → log, report, and sanity checks.

A static scan of function length, nesting depth, and pass-throughs is part of refactor validation.

## Ruff and Pyright contract

[pyproject.toml](../../pyproject.toml) is the Ruff source of truth. Ruff selects `E`, `W`, `F`, `I`, `B`, `N`, and `UP`, uses a 150-character formatter width, and ignores only `E501` so long URLs and prose comments do not fail lint. The formatter owns layout and import sorting; do not hand-format around it.

The enabled rules mean, in particular:

- imports stay at module scope and in Ruff order (`E402`, `I`);
- `zip()` calls state their length contract with `strict=True` or `strict=False` (`B905`);
- names follow the `N802`, `N803`, `N806`, and `N818` rules above; and
- obsolete syntax is upgraded for the repository's Python 3.13 runtime (`UP`).

[pyrightconfig.json](../../pyrightconfig.json) resolves the project-local `.venv`, which points to the Poetry environment. Keep that link valid and keep third-party typing packages, including `pandas-stubs`, synchronized with the runtime packages. Fix typing at the narrowest truthful boundary with annotations, explicit `None` checks, protocols, or targeted `cast()` calls. Do not add repository-wide suppressions or blanket missing-import ignores.

Airflow DAGs are the narrow exception: they run in the separate Python 3.12 Airflow environment, so a DAG may carry a file-scoped Pyright suppression only for imports and DSL attributes supplied by that environment. Application and test modules must remain clean without that exception.

## Docstrings and comments

Module and callable docstrings are a few lines saying what the code does: logic, inputs, outputs, and load-bearing invariants in concise English. They carry no history, no rationale chronology ("why we chose this"), and no measurements; those belong in reports or the [wiki log](../log.md).

Keep comments sparse and explain why a non-obvious choice exists. When changing a behavior whose docstring carries an invariant, update that prose in the same change. Preserve deliberate duplication when the existing explanation says independent policies must remain independent.

## Constants and knobs

Before adding a URL, field name, format, threshold, taxonomy value, model identifier, or other literal, search [constants.py](../../src/constants/constants.py).

- World facts and stable literals belong in constants.
- Tunable numerical choices belong in [configuration](../reference/configuration.md).
- Table names and grains belong only in [schema.py](../../src/data_store/schema.py).
- Module-level constants sit at the top of the module, directly after the imports.
- A module global holds only a parameter, URL, or value a user may tune; a value local to one function stays a local variable, even when used twice.

Do not introduce `*_TABLE` constants or duplicate registries.

## Logging and error handling

Use:

- `self._log` inside a Step or Strategy;
- `context.log` inside a helper that accepts the context; or
- a module logger in a leaf utility that intentionally has no context.

Never call `print()` in application code. `Context` exposes `.log`, not `.logger`.

Provider walks catch source failures per ticker or filing so one malformed external record does not abort the universe. Programming errors in repository code—such as `NameError`, `AttributeError`, `TypeError`, `KeyError`, and `ImportError`—must escape and fail the run. Broad catches are acceptable only at a library boundary whose contract is to absorb malformed external data.

## Data rules

All tabular I/O goes through `context.store`; the full contract is in [data access and storage](./data-access.md).

Never:

- import SQLAlchemy outside the store package;
- read a large table without projection and row scope;
- use a string table name;
- store a heavy cube frame on `self`;
- resume a fetcher by loading its table; or
- treat unavailable data as zero.

Cube builders keep heavy frames local to `run()`, load only their declared price fields, and let the part registry own incremental policy.

### Performance

Tables are large and pipelines run daily.

- Pre-filter a frame to the rows that need computing before computing anything.
- Vectorise with column operations; do not use `DataFrame.apply` or Python loops over rows on large frames.
- Fetchers plan through their `schema.Resume` contract and `resume.py`, never a full read or a file.

## Prefer the smallest correct change

Before adding code:

1. search constants for the literal;
2. search shared and package-local utilities for the helper;
3. look for a registry that already owns the set being changed;
4. modify the closest existing behavior;
5. add the targeted test with the implementation; and
6. execute that test with `-v -s`.

Registries such as `Tables`, the cube-part registry, strategy registry, and SEC form policy often reduce a multi-file change to one declaration plus its consumer test.

## Testing expectations

Feature and economic tests use small real-data samples. Synthetic known-truth fixtures are reserved for parser and mathematical identities, normally paired with a real-data coverage check. Every completed test prints a concise sanity conclusion that a reviewer can interpret.

See [testing and validation](./testing.md).

## Risk zones

Ask before editing:

| Area | Why |
| --- | --- |
| [context.py](../../src/context.py) | Shared construction for configuration, logging, paths, environment, and storage. |
| [step.py](../../src/utils/step.py) | Base class for nearly every pipeline stage. |
| `src/constants/` | Global literal changes cascade across sources and features. |
| `src/data_store/`, [sql/schema.sql](../../sql/schema.sql) | Table, DDL, and dialect boundary. |
| `configs/` | Contract changes must stay synchronized with consumers and tests. |
| `data/` and PostgreSQL volume | Non-recoverable artifacts and data. |
| aggregate fingerprint baseline | Numeric-regression evidence with tightly gated regeneration. |

Avoid unrelated reformatting and do not widen a diff beyond the task.

## Documentation synchronization

The canonical documentation surface is now [the wiki](../OVERVIEW.md). Keep `AGENTS.md`, `CLAUDE.md`, and affected wiki pages synchronized with code. Add durable details to the relevant architecture, module, flow, concept, guide, or reference page; keep `AGENTS.md` below its 70-line budget as the always-loaded summary.

Propose a new shared convention before editing the instruction contract. Historical implementation details belong in reports or the append-only [wiki log](../log.md), not source docstrings.

## Communicating results

Report the result first. For tests, show only the targeted test's relevant output and printed sanity conclusion unless a broader suite was requested. For a refactor, name the affected contracts and evidence concisely.

Important data, model, or output work ends with the appropriate read-only validator and a report where the repository workflow requires one.

## Checklist

- [ ] Step/Strategy boundary preserved.
- [ ] Full annotations and top-level imports.
- [ ] Each function does one thing in at most 150 lines and 3 nesting levels.
- [ ] No pass-through functions.
- [ ] Flow ordered load → clean → compute → post-process → save → log.
- [ ] Repeated logic extracted to the closest utils module.
- [ ] Large frames pre-filtered and vectorised; no row loops or `apply`.
- [ ] Docstrings a few lines, without history or measurements.
- [ ] Constants, config, and table registry used correctly.
- [ ] Store-only tabular I/O.
- [ ] Point-in-time and availability semantics preserved.
- [ ] Targeted test includes a sanity conclusion.
- [ ] Risk-zone approval obtained.
- [ ] Wiki and agent instructions remain synchronized.

## Related

- [Step pattern](../concepts/step-pattern.md)
- [Runtime and shared utilities](../modules/runtime-and-shared-utils.md)
- [Data access and storage](./data-access.md)
- [Testing and validation](./testing.md)
