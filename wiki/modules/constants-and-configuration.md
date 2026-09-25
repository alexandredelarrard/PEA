---
title: Constants and configuration
description: Ownership of literals, tunable settings, paths, model families, strategies, and validation contracts.
type: module
tags:
  - wiki
  - module
---
# Constants and configuration

## Summary

Configuration is an OmegaConf tree assembled from every YAML file under `configs/`. Tunable values belong in YAML; stable source facts such as URLs, form vocabularies, date formats, market-series names, and plausibility bounds belong in `src/constants/`. The ownership map is documented in [configuration reference](../reference/configuration.md).

## Responsibilities

- Load `configs/configs.yml` first and merge the remaining YAML files recursively.
- Separate stage-specific knobs by top-level key.
- Centralize literal source contracts and shared CLI option declarations.
- Keep structured overrides and evidence in versioned JSON files under `configs/`.

## Public API / entry points

- `read_config(path)` in [utils/config.py](../../src/utils/config.py).
- `get_config_context()` in [context.py](../../src/context.py).
- Shared Click option tuples in [command_line_interface.py](../../src/constants/command_line_interface.py).
- Named market-series registries such as `MACRO_MARKET_SERIES` in [constants_price.py](../../src/constants/constants_price.py).

## Key files

- [configs/configs.yml](../../configs/configs.yml) owns the seed and extraction window.
- [configs/build_cube.yml](../../configs/build_cube.yml) owns factor, target, feature, and incremental parameters.
- [configs/modellling.yml](../../configs/modellling.yml) plus [configs/models](../../configs/models/) own training and model-family settings.
- [configs/strategy](../../configs/strategy/) and [configs/portfolio.yml](../../configs/portfolio.yml) own sleeve construction and portfolio-wide allocation.
- [configs/validate.yml](../../configs/validate.yml) declares validation contracts.
- [configs/gpt.yml](../../configs/gpt.yml) owns model/provider selection, concurrency, budgets, and embedding settings.
- [constants.py](../../src/constants/constants.py) and [constants_price.py](../../src/constants/constants_price.py) own stable literals.

## Dependencies

OmegaConf performs recursive merges. Consumers access the final `DictConfig` through [Context](./runtime-and-shared-utils.md); no package maintains a second configuration loader.

## Participates in

Configuration controls every stage, including [data extraction](./data-extract.md), [cube aggregation](./data-aggregate.md), [modelling](./modelling.md), [strategy sleeves](./strategies.md), and [validation](./validation.md).

## Related

- [Add a model or sleeve](../guides/add-a-model-or-sleeve.md)
- [Add a cube feature](../guides/add-a-cube-feature.md)
- [Configuration reference](../reference/configuration.md)
