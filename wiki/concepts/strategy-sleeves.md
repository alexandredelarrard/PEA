---
title: Strategy sleeves
description: Independent strategy implementations blended only at the portfolio layer.
type: concept
tags:
  - wiki
  - concept
---
# Strategy sleeves

## Definition

A sleeve is a self-contained strategy implementing `Strategy.run(PortfolioInputs) -> StrategyResult`. It owns signal-to-weight construction and trading assumptions, while receiving portfolio-wide capital, dates, target volatility, costs, risk-free rate, and analysis flags from above.

## Why it matters

The contract prevents one strategy from depending on another and lets the portfolio blend heterogeneous return streams uniformly. `StrategyResult` includes both summary positions and the exact `book_weights` and `book_prices` used for trading so the execution layer can reconstruct share moves after applying dynamic portfolio allocation.

A sleeve name does not have to equal its configuration key; each strategy class declares `config_key`, and [STRATEGY_REGISTRY](../../src/strategies/__init__.py) is the discovery surface.

## Where it lives

- Contracts: [src/strategies/base.py](../../src/strategies/base.py)
- Registry: [src/strategies/__init__.py](../../src/strategies/__init__.py)
- Implementations: [src/strategies](../../src/strategies/)
- Portfolio consumer: [step_portfolio.py](../../src/portfolio/step_portfolio.py)
- Configuration: [configs/strategy](../../configs/strategy/) and [configs/portfolio.yml](../../configs/portfolio.yml)

## Related

- [Strategies](../modules/strategies.md)
- [Portfolio](../modules/portfolio.md)
- [Portfolio to trade ledger](../flows/portfolio-to-trade-ledger.md)
- [Add a model or sleeve](../guides/add-a-model-or-sleeve.md)
