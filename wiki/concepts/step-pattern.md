---
title: Step pattern
description: The common orchestration contract used by pipeline stages.
type: concept
tags:
  - wiki
  - concept
source_commit: 1b5f33a2a210336c14758ce4897008c72da1d42c
---
# Step pattern

## Definition

A `Step` is a package-level orchestrator whose public entry point is `run()`. A step whose parts also run alone exposes them as public `run_*` sub-runs that `run()` calls in order: `StepLongShort` adds `run_train()` / `run_predict()`, and each extraction step exposes one per source or stage (see [coding standards](../guides/coding-standards.md)). Subclasses call the base constructor with a `Context` and `DictConfig`, then keep implementation work in private methods, helpers, or composed sub-steps. The base class in [utils/step.py](../../src/utils/step.py) supplies configuration, context, a module-specific logger, a derived name, and the current date.

## Why it matters

The pattern gives the CLI, Airflow, scratch driver, and tests a consistent unit of execution. Super-steps such as `StepExtractAllData` and `StepBuildCube` can compose package-local stages without exposing every helper as public API. `StepExtractAllData` composes `StepExtractZips` (downloads, then ingests, around the identity build), `StepExtractIdentity`, `StepExtractEdgar` and `StepExtractLlm` with the prices, institutionals, fundamentals and behavioral steps; each extraction CLI command builds one step and calls one of its `run_*` sub-runs. It also keeps logging and configuration conventions uniform.

Portfolio follows the same Step contract: [StepPortfolio](../../src/portfolio/steps/step_portfolio.py) composes package-local scoring, construction, simulation and report transformations.

## Where it lives

- Base class: [src/utils/step.py](../../src/utils/step.py)
- Extraction composition: [step_extract_all_data.py](../../src/data_extract/step_extract_all_data.py); functional steps under [transformers](../../src/data_extract/transformers/)
- Cube composition: [step_build_cube.py](../../src/data_aggregate/step_build_cube.py)
- Runtime construction: [context.py](../../src/context.py)
- Naming and package rules: [system overview](../architecture/system-overview.md)

## Related

- [Runtime and shared utilities](../modules/runtime-and-shared-utils.md)
- [Orchestration and runtime](../architecture/orchestration-and-runtime.md)
- [Strategy sleeves](./strategy-sleeves.md)
