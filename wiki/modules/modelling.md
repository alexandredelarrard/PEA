---
title: Modelling
description: Cross-sectional ensemble training, latest-date prediction, diagnostics, and macro signal engines.
type: module
tags:
  - wiki
  - module
---
# Modelling

## Summary

`src/modelling` contains signal engines rather than portfolio construction. The long/short engine trains one cross-sectional ensemble per forecast horizon from the cube; the trend and long-book engines transform long-history macro prices into strategy inputs.

## Responsibilities

- Select configured feature families and target columns from the cube schema.
- Load and train one horizon at a time to bound memory.
- Use time-series cross-validation with an embargo.
- Train ElasticNet, LightGBM, and LightGBM random-forest members.
- Persist model artifacts, metadata, SHAP/PDP diagnostics, predictions, and blended signals.
- Score the newest cube dates without requiring matured targets.
- Provide multi-asset trend and allocation primitives.

## Public API / entry points

- `StepModelling.run(full_history=False)` and `predict_latest()` in [step_train.py](../../src/modelling/long_short/step_train.py).
- The `train`, `full-train`, and `predict` commands in [modelling/cli.py](../../src/modelling/cli.py).
- `trend_book()` in [trend/signal.py](../../src/modelling/trend/signal.py).
- `allocation_backtest()` in [long_book/allocation.py](../../src/modelling/long_book/allocation.py).

## Key files

- [src/modelling/long_short/step_train.py](../../src/modelling/long_short/step_train.py) owns per-horizon training and latest prediction.
- [src/modelling/long_short/utils/model.py](../../src/modelling/long_short/utils/model.py) contains model-family helpers.
- [src/modelling/long_short/utils/diagnostics.py](../../src/modelling/long_short/utils/diagnostics.py) and [shap_analysis.py](../../src/modelling/long_short/utils/shap_analysis.py) produce diagnostics.
- [configs/modellling.yml](../../configs/modellling.yml) owns shared model and training settings; [configs/models](../../configs/models/) owns family-specific features and hyperparameters.

## Dependencies

The long/short engine consumes [cube aggregation](./data-aggregate.md) output through [DataStore](./data-store.md). Trend and long-book engines consume named macro series. [Strategy sleeves](./strategies.md) wrap these engines into a common portfolio contract.

## Participates in

- [Model training and daily prediction](../flows/model-training-and-prediction.md)
- [Feature-to-portfolio architecture](../architecture/feature-to-portfolio.md)
- [Add a model or sleeve](../guides/add-a-model-or-sleeve.md)

## Related

- [Modelling, strategies, and portfolio](../reference/modelling-and-portfolio.md)
- [Strategy sleeves](../concepts/strategy-sleeves.md)
- [Tests module](./tests.md)
