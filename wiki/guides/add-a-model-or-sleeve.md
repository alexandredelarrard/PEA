---
title: Add a model or sleeve
description: Extend model families or strategy sleeves while preserving training, artifact, and portfolio contracts.
type: guide
tags:
  - wiki
  - guide
---
# Add a model or sleeve

## Goal

Add a model family to the cross-sectional ensemble or add a self-contained strategy sleeve without coupling signal generation, construction, and portfolio allocation.

## Steps

### Model family

1. Add one top-level model-family YAML under [configs/models](../../configs/models/) with hyperparameters and its own feature columns.
2. Add a `BaseModel` subclass under `src/modelling/transformers/` (see [base.py](../../src/modelling/transformers/base.py)): set `config_key` (its YAML block), implement `_fit`, `_predict` and `importance`, and set `supports_shap` / `supports_classification`. Persistence is inherited: every member pickles itself.
3. Register its `model.ensemble` name in `MODEL_FAMILIES` ([transformers/__init__.py](../../src/modelling/transformers/__init__.py)) and add the file to `EXPECTED_TRANSFORMERS` in [test_architecture_guard.py](../../tests/modelling/test_architecture_guard.py). [StepLongShort](../../src/modelling/steps/step_long_short.py) then trains, diagnoses, saves, and loads it per horizon with no step change.
4. Preserve time-series cross-validation, embargo handling, validation-only diagnostics, and per-day prediction standardization.
5. Add persistence, reproducibility, diagnostics, and ensemble-member tests under [tests/modelling](../../tests/modelling/).

### Strategy sleeve

1. Add a `Strategy` subclass under [src/strategies](../../src/strategies/) and declare its `name` and `config_key`.
2. Accept `PortfolioInputs` and return a complete `StrategyResult`, including traded weight and price panels when the ledger must reproduce trades.
3. Add a dedicated top-level strategy YAML under [configs/strategy](../../configs/strategy/).
4. Register the class in [STRATEGY_REGISTRY](../../src/strategies/__init__.py).
5. Add the sleeve name to [configs/portfolio.yml](../../configs/portfolio.yml) only when it should run by default.
6. Add construction, cost, position, and blend tests under [tests/strategies](../../tests/strategies/) and [tests/portfolio](../../tests/portfolio/).

## Relevant code

- Training: [step_long_short.py](../../src/modelling/steps/step_long_short.py); model families: [transformers/base.py](../../src/modelling/transformers/base.py)
- Strategy contract: [strategies/base.py](../../src/strategies/base.py)
- Strategy registry: [strategies/__init__.py](../../src/strategies/__init__.py)
- Portfolio consumer: [step_portfolio.py](../../src/portfolio/step_portfolio.py)
- Reference behavior: [modelling, strategies, and portfolio](../reference/modelling-and-portfolio.md)

## Gotchas

Do not use random folds for cross-sectional time-series data. Never fit the linear family with `task: classification` (it is regression-only and refuses it); a classification task needs a {0, 1} label, not a rank target. Old artifacts without `artifact_format` must be retrained. Do not duplicate portfolio-wide capital, window, target-volatility, or risk-free settings in a sleeve config.

## Related

- [Modelling](../modules/modelling.md)
- [Strategies](../modules/strategies.md)
- [Strategy sleeves](../concepts/strategy-sleeves.md)
- [Model training and daily prediction](../flows/model-training-and-prediction.md)
