---
title: Strategies
description: Self-contained portfolio sleeves sharing one input and result contract.
type: module
tags:
  - wiki
  - module
---
# Strategies

## Summary

`src/strategies` turns signal engines and market data into self-contained sleeve return streams, holdings, traded weight panels, and metrics. Sleeves do not call one another; the portfolio layer discovers them through a registry and treats them uniformly.

## Responsibilities

- Define immutable portfolio inputs and the common strategy-result shape.
- Implement long/short equity, equity long-only, long-book, and trend-CTA sleeves.
- Convert forecasts or macro signals into constrained weights.
- Apply trading costs, turnover controls, position sizing, and optional integer-share projection.
- Produce the exact weight and price panels needed to reconstruct trades.
- Provide sleeve-specific analysis and plots.

## Public API / entry points

- `PortfolioInputs`, `StrategyResult`, and abstract `Strategy.run()` in [base.py](../../src/strategies/base.py).
- `STRATEGY_REGISTRY` in [strategies/__init__.py](../../src/strategies/__init__.py).
- `LongShortStrategy`, `EqLongOnlyStrategy`, `LongBookStrategy`, and `TrendCTAStrategy` in their respective [step modules](../../src/strategies/).

## Key files

- [step_ls.py](../../src/strategies/step_ls.py) implements market-neutral equity long/short.
- [step_eq_long_only.py](../../src/strategies/step_eq_long_only.py) selects a buffered top-N long book.
- [step_long_book.py](../../src/strategies/step_long_book.py) wraps multi-asset ERC, trend, and regime tilts.
- [step_trend.py](../../src/strategies/step_trend.py) implements multi-speed time-series momentum.
- [utils/strategies_opt.py](../../src/strategies/utils/strategies_opt.py), [integer_shares.py](../../src/strategies/utils/integer_shares.py), and [blotter.py](../../src/strategies/utils/blotter.py) support construction and execution.

## Dependencies

Sleeves consume outputs from [modelling](./modelling.md), macro prices from [DataStore](./data-store.md), and per-sleeve YAML under [configs/strategy](../../configs/strategy/). They depend on the shared [strategy-sleeves contract](../concepts/strategy-sleeves.md).

## Participates in

All sleeves are executed and blended by [portfolio](./portfolio.md), including the [portfolio-to-trade-ledger flow](../flows/portfolio-to-trade-ledger.md).

## Related

- [Feature-to-portfolio architecture](../architecture/feature-to-portfolio.md)
- [Add a model or sleeve](../guides/add-a-model-or-sleeve.md)
- [Modelling, strategies, and portfolio](../reference/modelling-and-portfolio.md)
