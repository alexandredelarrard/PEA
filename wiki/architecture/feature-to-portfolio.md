---
title: Feature-to-portfolio pipeline
description: How cube parts become models, strategy sleeves, portfolio weights, and trade-ledger rows.
type: architecture
tags:
  - wiki
  - architecture
---
# Feature-to-portfolio pipeline

## Summary

Downstream processing is split into four layers: cube construction, signal modelling, self-contained strategy sleeves, and portfolio allocation/execution. Each layer persists or returns an explicit contract rather than reaching into another layer's internals. The modelling and portfolio contracts are documented in [modelling, strategies, and portfolio](../reference/modelling-and-portfolio.md).

## Diagram

~~~mermaid
flowchart LR
  Sources[(Extract tables)] --> Parts[Feature and target parts]
  Parts --> Cube[(Wide cube)]
  Cube --> Horizon[One panel per horizon]
  Horizon --> Ensemble[ElasticNet LightGBM RF]
  Ensemble --> Predictions[(Predictions)]
  Predictions --> LSEquity[LS equity sleeve]
  Macro[(Macro prices)] --> LongBook[Long book sleeve]
  Macro --> Trend[Trend CTA sleeve]
  LSEquity --> Blend[ERC sleeve blend]
  LongBook --> Blend
  Trend --> Blend
  Blend --> Vol[Global volatility target]
  Vol --> Ledger[(Strategy ledger)]
~~~

## Key components

- [step_build_cube.py](../../src/data_aggregate/step_build_cube.py) coordinates the cube-part builders; [parts.py](../../src/data_aggregate/utils/common/parts.py) is their registry.
- `StepModelling.run` and `predict_latest` in [step_train.py](../../src/modelling/long_short/step_train.py) train and score one horizon at a time.
- `PortfolioInputs`, `StrategyResult`, and the abstract `Strategy` contract live in [strategies/base.py](../../src/strategies/base.py).
- `STRATEGY_REGISTRY` in [strategies/__init__.py](../../src/strategies/__init__.py) maps configured sleeve names to implementations.
- `StepPortfolio` blends sleeve returns in [step_portfolio.py](../../src/portfolio/step_portfolio.py); `StepStrategyMoves` rebuilds each sleeve's book at its time-varying allocation in [step_strategy_moves.py](../../src/portfolio/step_strategy_moves.py).

## Design decisions

The cube stores targets wide, one row per date and ticker. Training still processes one horizon at a time, and the latest prediction path loads no target columns because forward labels have not matured.

Sleeves are independent and receive portfolio-level inputs rather than reading another sleeve. The portfolio owns ERC or inverse-volatility blending, the global volatility target, leverage, and capital. The ledger recomputes share quantities from scaled weight panels because time-varying capital allocation cannot be reproduced by multiplying an already-rounded blotter.

## Related

- [Cube build](../flows/cube-build.md)
- [Model training and daily prediction](../flows/model-training-and-prediction.md)
- [Portfolio to trade ledger](../flows/portfolio-to-trade-ledger.md)
- [Strategy sleeves](../concepts/strategy-sleeves.md)
