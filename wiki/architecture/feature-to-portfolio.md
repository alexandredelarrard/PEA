---
source_commit: 55d6e2d199fa9a83aca7d5442549eeb66eedd189
title: Feature-to-portfolio pipeline
description: Point-in-time features, saved models and an independent portfolio backtest.
type: architecture
tags:
  - wiki
  - architecture
---
# Feature-to-portfolio pipeline

## Summary

Existing source tables feed a point-in-time cube, saved cross-sectional ensembles, and an independent quantity-based portfolio backtest. Training and assessment remain separate from portfolio construction. [Modelling and portfolio contracts](../reference/modelling-and-portfolio.md) describe the data clocks.

~~~mermaid
flowchart LR
  Sources[(Extract tables)] --> Parts[Feature and target parts]
  Parts --> Cube[(Point in time cube)]
  Cube --> Train[Chronological model training]
  Train --> Models[(Saved run cutoff members)]
  Models --> Assess[Fresh saved-model assessment]
  Cube --> Assess
  Assess --> ModelReport[Modelling report]
  Models --> Scores[Feature-only ensemble scoring]
  Cube --> Scores
  Scores --> Sector[Sector neutral LS construction]
  Sources --> Sector
  Sector --> Engine[LS S and P and cash quantity engine]
  Engine --> Report[Offline portfolio report]
~~~

## Key files and boundaries

- [step_build_cube.py](../../src/data_aggregate/step_build_cube.py) coordinates registered feature/target builders.
- [StepLongShort](../../src/modelling/steps/step_long_short.py) saves original fold/cutoff members and basic IC/IR; production full-history fitting is a separate command.
- [StepAssessLongShort](../../src/modelling/steps/step_assess_long_short.py) reloads models and current data to recompute diagnostics and its report.
- [signals.py](../../src/portfolio/transformers/signals.py) consumes only public modelling utilities, never its steps or transformers.
- [StepPortfolio](../../src/portfolio/steps/step_portfolio.py) owns the nine simulations and run-local report.
- [DataStore](../modules/data-store.md) remains the boundary for every tabular read; portfolio writes no source or ledger table.

## Design decisions

Labels do not determine scoring membership. Horizon weights use training information only. Signal close J precedes execution close J+1. Fixed capital allocation and LS selection have independent observed-session schedules.

The long/short book is dollar-neutral per sector when selection is rebalanced; the S&P pocket remains directional. The separate free cash pocket earns the fixed rate. Actual held quantities, corporate actions, implicit distributions and executed deltas determine P&L and fees. No ERC blend or application adapter remains.

## Related

- [Cube build](../flows/cube-build.md)
- [Modelling lifecycle](../flows/model-training-and-prediction.md)
- [Portfolio backtest](../flows/portfolio-to-trade-ledger.md)
- [Capital budgets](../concepts/strategy-sleeves.md)
