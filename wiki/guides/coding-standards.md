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

Class names use `StepPascalCase`; files use `step_snake_case.py`. Strategies are the deliberate exception and implement [Strategy.run](../../src/strategies/base.py).

Do not cross-import between sibling `src/` packages. Shared logic belongs in [src/utils](../../src/utils/). The sanctioned service exception is [src/gpt_extract](../../src/gpt_extract/), which centralizes LLM clients, prompts, and embeddings.

## Functions and types

- Fully annotate every function and method signature.
- Put `from __future__ import annotations` at the top of new modules.
- Keep all imports at module scope; solve cycles rather than hiding imports inside functions.
- Use explicit variable names. DataFrame variables may use a clear `df_...` prefix.
- Split large routines into helpers with one purpose.
- Reuse a helper only when the abstraction is already real; package-local helpers are preferable to speculative global utilities.
- Function and method names are lower snake case.

## Docstrings and comments

Module and callable docstrings explain logic, inputs, outputs, and load-bearing invariants in concise English. They do not record commit history, bug chronology, or a full design report.

Keep comments sparse and explain why a non-obvious choice exists. When changing a behavior whose docstring carries an invariant, update that prose in the same change. Preserve deliberate duplication when the existing explanation says independent policies must remain independent.

## Constants and knobs

Before adding a URL, field name, format, threshold, taxonomy value, model identifier, or other literal, search [constants.py](../../src/constants/constants.py).

- World facts and stable literals belong in constants.
- Tunable numerical choices belong in [configuration](../reference/configuration.md).
- Table names and grains belong only in [schema.py](../../src/data_store/schema.py).

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
