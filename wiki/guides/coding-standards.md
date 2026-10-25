---
title: Coding standards
description: Python structure, typing, logging, data boundaries, risk zones, and documentation rules.
type: guide
tags:
  - wiki
  - guide
  - coding-standards
source_commit: c2a09cf41e68d0f9addca8c1f02c2b7ce0423c25
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

Class names use `StepPascalCase`; files use `step_snake_case.py`. [StepLongShort](../../src/modelling/steps/step_long_short.py) is the exception: it lives in `src/modelling/steps/` and exposes `run_train()` and `run_predict()` besides `run()` (which calls `run_train()`), because training and prediction share one step's configuration.

Do not cross-import between sibling `src/` packages. Shared logic belongs in the [runtime and shared utilities](../modules/runtime-and-shared-utils.md) package. The sanctioned service exception is [src/gpt_extract](../../src/gpt_extract/), which centralizes LLM clients, prompts, and embeddings.

## Functions and types

- Fully annotate every function and method signature.
- Put `from __future__ import annotations` at the top of new modules.
- Keep all imports at module scope; solve cycles rather than hiding imports inside functions.
- Variable names are lowercase (`N806`) and say what they hold: DataFrames carry a `df_` prefix (`df_prices`); scalars and collections use descriptive names (`n_filings`, `missing_tickers`).
- Functions and methods use lower snake case (`N802`); arguments use lower snake case (`N803`); local variables inside functions use lower snake case (`N806`); exception classes end in `Error` (`N818`). Capitals are reserved for module-level constants and class names.
- Prefer Python 3.13 syntax. Write unions as `A | B`, including runtime checks such as `isinstance(value, A | B)`. The runtime-check form is a project convention even though Ruff 0.16.9 removed `UP038`.

## Mandatory rules for every coding agent

These rules are compulsory for every future coding change:

- Never add a function parameter whose only callers or non-default values are in tests. Tests must exercise the existing production interface; seed a test store or patch the external dependency instead of adding test-only inputs.
- Put a function's `store.load`, `store.iter_load`, and other table reads in a visible loading block at the top, before cleaning and computations. Read each table only once per function, including reads hidden behind helpers: combine the required projection and row scope into one read and reuse the result. Stream large results once with `iter_load`; repeated table calls are slow.
- Never pass a logger as a function or constructor input. Use the logger already available through `self._log` or the existing context (`self._context.log` in this repository); otherwise declare `logger = logging.getLogger(__name__)` once at module scope and use it directly.
- Do not add unused inputs or speculative options. Every parameter must serve a current production caller and be consumed by the function or its downstream calls. Keep the interface simple and add complexity only when a concrete use case requires it.

## Function design and data flow

- One function does one thing in at most 90 lines. A top-level orchestrator (a Step's `run()` or a `build_fundamentals_<x>` function) only calls the small step functions in order.
- Nesting is at most 3 levels; 2 is the standard. Flatten deeper blocks with guard clauses, early returns, or a helper.
- No pass-through functions: a function whose body only calls another function is removed, and callers call the target directly.
- Functions doing the same thing are declared once. Logic repeated in two or more places, including across fetchers, is extracted to the closest shared utils module (package `utils/` for one package, `src/utils/` across packages) and called from every dependent; speculative one-use abstractions remain discouraged.
- Pass a value computed in one function to the next; never re-declare or recompute it elsewhere.
- Default assessed ticker lists come from [load_universe_tickers](../../src/utils/universe.py) and are already normalized. CLI `-t` input is a comma-separated string: split it first and call [normalise_ticker](../../src/utils/string.py) once per symbol. Downstream lists need no repeated normalization; reuse the complete universe snapshot when guarding subset writes. Normalize raw source/config symbols with the shared helper, preserving source-specific security-marker semantics.
- Data flows in the golden-standard order: load → clean → compute features or steps → post-process → save → log, report, and sanity checks.
- No function input only used by tests + no logger passed as an input of any function.
- No more than 6 inputs in a function or class method
- self.store = context.store at the start of a class and pass through its methods.
- No global variable imported from constants.py when only used by its sub package functions. At the top of the function instead or in utils of the sub package part of the codebase.
- Flow ordered load / check remaining to do → clean → compute → post-process → save → log / report / sanity checks.

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

- Keep only genuinely shared cross-package contracts in `src/constants/`. Values used by one function stay local to that function; domain-only shared values live with their existing domain owner. Do not add earnings-call-only aliases to the global constants file.
- Tunable numerical choices belong in [configuration](../reference/configuration.md).
- Table names and grains belong only in [schema.py](../../src/data_store/schema.py).
- Module-level constants sit at the top of the module, directly after the imports.
- A module global holds only a parameter, URL, or value a user may tune; a value local to one function stays a local variable, even when used twice.

Do not introduce `*_TABLE` constants or duplicate registries.

## Logging and error handling

Never pass a logger as a function or constructor argument. Standalone modules declare `logger = logging.getLogger(__name__)` once and use it directly. Where an object already has context, use its existing `self._log` or context logger; do not add a context parameter just for logging.

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

Registries such as `Tables`, the cube-part registry and SEC form policy often reduce a multi-file change to one declaration plus its consumer test.

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
