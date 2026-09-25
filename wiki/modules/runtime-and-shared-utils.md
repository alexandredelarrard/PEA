---
title: Runtime and shared utilities
description: Context construction, Step lifecycle, root CLI discovery, and cross-package utilities.
type: module
tags:
  - wiki
  - module
---
# Runtime and shared utilities

## Summary

This module supplies the process-wide runtime used by every extraction, aggregation, modelling, and portfolio command. `Context` owns configuration, paths, logging, random state, environment loading, and the shared data store; `Step` gives orchestrators a consistent lifecycle and logger.

## Responsibilities

- Build the effective OmegaConf configuration and application context.
- Configure logging, random seeds, corporate TLS trust, environment variables, and database access.
- Discover package CLIs and expose them through `python -m src`.
- Host utilities shared across package boundaries, including universe resolution, macro-series loading, HTTP pacing, text metrics, risk parity, and plotting.

## Public API / entry points

- `get_config_context()` and `Context` in [context.py](../../src/context.py).
- `Step` in [utils/step.py](../../src/utils/step.py).
- `read_config()` in [utils/config.py](../../src/utils/config.py).
- `CLI` and `main()` in [cli.py](../../src/cli.py), reached through [__main__.py](../../src/__main__.py).

## Key files

- [src/context.py](../../src/context.py) assembles the runtime and exposes `context.store`, `context.log`, and `context.paths`.
- [src/utils/step.py](../../src/utils/step.py) implements the [Step pattern](../concepts/step-pattern.md).
- [src/utils/db.py](../../src/utils/db.py) creates the SQLAlchemy engine handed to [DataStore](./data-store.md).
- [src/utils/universe.py](../../src/utils/universe.py) resolves the current equity universe.
- [src/utils/macro.py](../../src/utils/macro.py) reads named market and macro series.

## Dependencies

The runtime depends on [constants and configuration](./constants-and-configuration.md), [the data store](./data-store.md), OmegaConf, python-dotenv, stdlib logging, and SQLAlchemy only at the engine-factory boundary.

## Participates in

Every CLI command and `Step` uses this module. It is the common entry surface for [nightly refresh](../flows/nightly-data-refresh.md), [cube construction](../flows/cube-build.md), training, prediction, and portfolio runs.

## Related

- [Orchestration and runtime architecture](../architecture/orchestration-and-runtime.md)
- [Run the pipeline](../guides/run-the-pipeline.md)
- [Coding standards](../guides/coding-standards.md)
