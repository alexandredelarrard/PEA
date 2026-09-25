---
title: Step pattern
description: The common orchestration contract used by pipeline stages.
type: concept
tags:
  - wiki
  - concept
---
# Step pattern

## Definition

A `Step` is a package-level orchestrator with a single public `run()` method. Subclasses call the base constructor with a `Context` and `DictConfig`, then keep implementation work in private methods, helpers, or composed sub-steps. The base class in [utils/step.py](../../src/utils/step.py) supplies configuration, context, a module-specific logger, a derived name, and the current date.

## Why it matters

The pattern gives the CLI, Airflow, scratch driver, and tests a consistent unit of execution. Super-steps such as `StepExtractAllData` and `StepBuildCube` can compose package-local stages without exposing every helper as public API. It also keeps logging and configuration conventions uniform.

Strategy sleeves deliberately use a different contract: `Strategy.run(PortfolioInputs) -> StrategyResult`, defined in [strategies/base.py](../../src/strategies/base.py).

## Where it lives

- Base class: [src/utils/step.py](../../src/utils/step.py)
- Extraction composition: [step_extract_all_data.py](../../src/data_extract/step_extract_all_data.py)
- Cube composition: [step_build_cube.py](../../src/data_aggregate/step_build_cube.py)
- Runtime construction: [context.py](../../src/context.py)
- Naming and package rules: [system overview](../architecture/system-overview.md)

## Related

- [Runtime and shared utilities](../modules/runtime-and-shared-utils.md)
- [Orchestration and runtime](../architecture/orchestration-and-runtime.md)
- [Strategy sleeves](./strategy-sleeves.md)
