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
2. Extend model construction, fitting, persistence, and reload logic in [modelling/long_short/utils/model.py](../../src/modelling/long_short/utils/model.py).
3. Ensure [StepModelling](../../src/modelling/long_short/step_train.py) can train, diagnose, save, and load the family per horizon.
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

- Training: [step_train.py](../../src/modelling/long_short/step_train.py)
- Strategy contract: [strategies/base.py](../../src/strategies/base.py)
- Strategy registry: [strategies/__init__.py](../../src/strategies/__init__.py)
- Portfolio consumer: [step_portfolio.py](../../src/portfolio/step_portfolio.py)
- Reference behavior: [modelling, strategies, and portfolio](../reference/modelling-and-portfolio.md)

## Gotchas

Do not use random folds for cross-sectional time-series data. A LightGBM random forest is still serialized as a booster, while linear models use a different artifact format. Do not duplicate portfolio-wide capital, window, target-volatility, or risk-free settings in a sleeve config.

## Related

- [Modelling](../modules/modelling.md)
- [Strategies](../modules/strategies.md)
- [Strategy sleeves](../concepts/strategy-sleeves.md)
- [Model training and daily prediction](../flows/model-training-and-prediction.md)
