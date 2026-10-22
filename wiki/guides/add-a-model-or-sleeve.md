---
source_commit: 55d6e2d199fa9a83aca7d5442549eeb66eedd189
title: Change a model or backtest
description: Extend models and portfolio transformations while preserving temporal contracts.
type: guide
tags:
  - wiki
  - guide
---
# Add a model or change the backtest

## Goal

Extend the saved cross-sectional ensemble or change the independent portfolio backtest while preserving model/data clocks.

## Steps

### Model family

1. Add one top-level model-family YAML under [configs/models](../../configs/models/lgbm_modelling.yml) with hyperparameters and its own feature columns.
2. Add a `BaseModel` subclass under `src/modelling/transformers/` (see [base.py](../../src/modelling/transformers/base.py)): set `config_key` (its YAML block), implement `_fit`, `_predict` and `importance`, and set `supports_shap` / `supports_classification`. Persistence is inherited: every member pickles itself.
3. Register its `model.ensemble` name in `MODEL_FAMILIES` ([transformers/__init__.py](../../src/modelling/transformers/__init__.py)) and add the file to `EXPECTED_TRANSFORMERS` in [test_architecture_guard.py](../../tests/modelling/test_architecture_guard.py). [StepLongShort](../../src/modelling/steps/step_long_short.py) then trains, diagnoses, saves, and loads it per horizon with no step change.
4. Preserve time-series cross-validation, embargo handling, validation-only diagnostics, and per-day prediction standardization.
5. Add persistence, reproducibility, diagnostics, and ensemble-member tests under [tests/modelling](../../tests/modelling/test_assessment_cli.py).

### Portfolio backtest

1. Keep feature-only ensemble scoring in [signals.py](../../src/portfolio/transformers/signals.py), through modelling.utils.
2. Change sector quotas/risk weights in [construction.py](../../src/portfolio/transformers/construction.py) and capital/quantity accounting in [simulation.py](../../src/portfolio/transformers/simulation.py).
3. Add tunable assumptions only to [portfolio.yml](../../configs/portfolio.yml), with the required configuration approval.
4. Preserve cutoff guards, J+1 close execution, independent allocation/selection schedules and self-financing costs.
5. Run known-truth [portfolio tests](../../tests/portfolio/test_simulation.py) and a bounded independent replay with existing data before interpreting Sharpe.

## Gotchas

Do not use random time-series folds, immature training outcomes, future-return availability for selection, or future-period correlations for horizon weights. A dollar-neutral sector book does not guarantee beta neutrality. Changes to allocations require a new Python simulation. There is no sleeve registry, Streamlit application or daily ledger command.

## Related

- [Modelling](../modules/modelling.md)
- [Portfolio](../modules/portfolio.md)
- [Model training and daily prediction](../flows/model-training-and-prediction.md)
