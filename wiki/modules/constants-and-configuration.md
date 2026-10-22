---
title: Constants and configuration
description: Ownership of literals, tunable settings, model families, portfolio and validation contracts.
type: module
tags:
  - wiki
  - module
source_commit: aeb931206d78eceb72ae1536988839056a630d1d
---
# Constants and configuration

## Summary

Configuration is an OmegaConf tree assembled from every YAML file under `configs/`. Tunable values belong in YAML. Genuinely shared contracts belong in `src/constants/`; values used by one function stay local, and domain-only shared values stay with their existing owner. Extracted SEC form lists belong only in each table's `Resume.forms` in [schema.py](../../src/data_store/schema.py); fetchers, derived form subsets, the EDGAR index, registrant policy and validation read those contracts. The ownership map is documented in [configuration reference](../reference/configuration.md).

## Responsibilities

- Load `configs/configs.yml` first and merge the remaining YAML files recursively.
- Separate stage-specific knobs by top-level key.
- Centralize genuinely shared source contracts and CLI option declarations. Keep single-function values local; keep domain-only shared values with their existing domain owner. Earnings-call scoring/turn tags, quality cutoff and report-to-quarter lag are local. NLP and embedding cache identities live with those utilities; the feature schema and signal lifetime remain shared because the builder and validator both enforce them.
- Keep structured overrides and evidence in versioned JSON files under `configs/`.

## Public API / entry points

- `read_config(path)` in [utils/config.py](../../src/utils/config.py).
- `get_config_context()` in [context.py](../../src/context.py).
- Shared Click option specifications in [command_line_interface.py](../../src/constants/command_line_interface.py). Every extraction and aggregation option uses `@click.option(*NAME_ARGS, **NAME_KWARGS)` directly; heterogeneous keyword dictionaries are typed there, with no local copies or casts in those CLIs. `-F` / `--full` selects complete input processing; `fundamentals --rebuild` separately deletes stored facts before refetching. Gap-check reports use `--report`.
- Named market-series registries such as `MACRO_MARKET_SERIES` in [constants_price.py](../../src/constants/constants_price.py).

## Key files

- [configs/configs.yml](../../configs/configs.yml) owns the seed and extraction window.
- [configs/build_cube.yml](../../configs/build_cube.yml) owns factor, target, feature, and incremental parameters.
- [configs/modellling.yml](../../configs/modellling.yml) plus [configs/models](../../configs/models/lgbm_modelling.yml) own training and model-family settings.
- [configs/portfolio.yml](../../configs/portfolio.yml) owns fixed long/short, S&P 500 and cash budgets, schedules, construction windows and simulated costs.
- [configs/validate.yml](../../configs/validate.yml) declares validation contracts.
- [configs/gpt.yml](../../configs/gpt.yml) owns model/provider selection, concurrency, budgets, and embedding settings.
- [constants.py](../../src/constants/constants.py) and [constants_price.py](../../src/constants/constants_price.py) own stable literals.

## Dependencies

OmegaConf performs recursive merges. Consumers access the final `DictConfig` through [Context](./runtime-and-shared-utils.md); no package maintains a second configuration loader.

## Participates in

Configuration controls every stage, including [data extraction](./data-extract.md), [cube aggregation](./data-aggregate.md), [modelling](./modelling.md), [portfolio construction](./strategies.md), and [validation](./validation.md).

## Related

- [Change a model or backtest](../guides/add-a-model-or-sleeve.md)
- [Add a cube feature](../guides/add-a-cube-feature.md)
- [Configuration reference](../reference/configuration.md)

## Institutional split defaults

The split removes eight aggregation-only globals from [constants.py](../../src/constants/constants.py): `FTD_HISTORICAL_LAG_DAYS`, `FTD_LATEST_PERIOD_MAX_AGE_DAYS`, `FTD_RECENT_CACHE_DAYS`, `F13_MAX_EARLY_DAYS`, `F13_MAX_LATE_DAYS`, `F13_REVISION_MIN_MOVE`, `SEC_13F_VALUE_FACTORS` and `PANEL_KEYS`. FTD/filing-band/monetary values remain local to their consumers; the shared domain revision floor lives in [institutional_features.py](../../src/data_aggregate/sec_thirteen/utils/institutional_features.py), and panel keys in [panel_merge.py](../../src/data_aggregate/utils/common/panel_merge.py). Cross-package filing and settlement clocks remain shared. Existing `build_cube.institutionals` and `data_availability.institutionals` namespaces are preserved; the two steps read their owned settings.
