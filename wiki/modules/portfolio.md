---
title: Portfolio
description: Sleeve blending, global volatility targeting, analysis, and persistence of executable strategy moves.
type: module
tags:
  - wiki
  - module
---
# Portfolio

## Summary

`src/portfolio` owns cross-sleeve allocation and execution reporting. `StepPortfolio` runs configured sleeves, aligns their returns, computes ERC or inverse-volatility weights, applies a portfolio-level volatility target and leverage cap, and reports the blended book. `StepStrategyMoves` turns the same allocation into persisted trade-ledger rows.

## Responsibilities

- Construct common `PortfolioInputs` from portfolio configuration.
- Execute registered sleeves and drop those without usable data.
- Align sleeve histories and compute dynamic blend weights.
- Apply global volatility scaling and leverage limits.
- Produce metrics, plots, and workbook blotters.
- Re-size each sleeve's traded panel by time-varying allocation and rebuild share-level trades.
- FIFO-match round trips and upsert the `strategy` ledger.

## Public API / entry points

- `StepPortfolio.run()` in [step_portfolio.py](../../src/portfolio/step_portfolio.py).
- `StepStrategyMoves.run()` in [step_strategy_moves.py](../../src/portfolio/step_strategy_moves.py).
- Portfolio CLI commands in [portfolio/cli.py](../../src/portfolio/cli.py).
- Blend helpers in [portfolio/utils/blend.py](../../src/portfolio/utils/blend.py).

## Key files

- [src/portfolio/step_portfolio.py](../../src/portfolio/step_portfolio.py) runs sleeves, blends returns, and reports.
- [src/portfolio/step_strategy_moves.py](../../src/portfolio/step_strategy_moves.py) creates the persisted ledger.
- [src/portfolio/analysis.py](../../src/portfolio/analysis.py) produces portfolio analysis.
- [configs/portfolio.yml](../../configs/portfolio.yml) owns sleeve selection, capital, costs, covariance mode, target volatility, and maximum leverage.

## Dependencies

The module depends on [strategy sleeves](./strategies.md), macro benchmark data from [DataStore](./data-store.md), and common risk-parity utilities. It persists through the same store facade as the rest of the pipeline.

## Participates in

- [Portfolio to trade ledger](../flows/portfolio-to-trade-ledger.md)
- [Model training and daily prediction](../flows/model-training-and-prediction.md)
- [Feature-to-portfolio architecture](../architecture/feature-to-portfolio.md)

## Related

- [Strategy sleeves](../concepts/strategy-sleeves.md)
- [Add a model or sleeve](../guides/add-a-model-or-sleeve.md)
- [Modelling, strategies, and portfolio](../reference/modelling-and-portfolio.md)
