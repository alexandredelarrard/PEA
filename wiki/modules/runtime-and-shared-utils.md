---
title: Runtime and shared utilities
description: Context construction, Step lifecycle, root CLI discovery, and cross-package utilities.
type: module
tags:
  - wiki
  - module
source_commit: 15df4358b2a4ea81c34f02c71076256aed23544a
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
- [src/utils/universe.py](../../src/utils/universe.py) resolves the current equity universe. `load_universe_tickers(context)` returns the normalized analysis list after configured exclusions; callers reuse it unchanged. CLI CSV text is split before [normalise_ticker](../../src/utils/string.py) is applied to individual symbols. Scope guards share the loaded complete list across nested source calls; CIK, GICS and entry-date metadata reads keep their projected roster columns.
- [src/utils/macro.py](../../src/utils/macro.py) reads named market and macro series.

## Dependencies

The runtime depends on [constants and configuration](./constants-and-configuration.md), [the data store](./data-store.md), OmegaConf, python-dotenv, stdlib logging, and SQLAlchemy only at the engine-factory boundary.

## Participates in

Every CLI command and `Step` uses this module. It is the common entry surface for [nightly refresh](../flows/nightly-data-refresh.md), [cube construction](../flows/cube-build.md), training, prediction, and portfolio runs.

## Related

- [Orchestration and runtime architecture](../architecture/orchestration-and-runtime.md)
- [Run the pipeline](../guides/run-the-pipeline.md)
- [Coding standards](../guides/coding-standards.md)

## Text utility ownership

Shared introduction/speaker parsing lives in [earnings_call_split.py](../../src/utils/text/earnings_call_split.py); shared tokenization, cleaning and partial-call quality live in [text_metrics.py](../../src/utils/text/text_metrics.py). Extraction, aggregation and validation use these canonical functions. FinBERT inference and the uncertainty vocabulary have only aggregate production consumers and live in [data_aggregate/utils/text](../../src/data_aggregate/utils/text/nlp_sentiment.py). Model choices come from YAML, with one local engine per sentiment stage.

## Institutional utility ownership

The aggregation-only `roster_as_of` and `first_snapshot_date` functions live in [sec_thirteen/utils/superinvestor_roster.py](../../src/data_aggregate/sec_thirteen/utils/superinvestor_roster.py); `iter_public_books` lives in [sec_thirteen/utils/institutional_vintages.py](../../src/data_aggregate/sec_thirteen/utils/institutional_vintages.py). The remaining [shared roster](../../src/utils/superinvestor_roster.py) owns configuration, manager chains and snapshot/display policy used by extraction and aggregation. The remaining [shared vintage utilities](../../src/utils/institutional_vintages.py) own source reconciliation, completion and public-book clocks used across packages. Aggregate-only availability/decay/share-basis helpers are under `data_aggregate/utils/common`; source-specific utilities stay with their step. All aggregation steps bind `context.store` in initialization.
