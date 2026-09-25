---
title: Portfolio to trade ledger
description: From sleeve results and dynamic ERC allocation to share-level moves and FIFO round trips.
type: flow
tags:
  - wiki
  - flow
---
# Portfolio to trade ledger

## Summary

The trade-ledger path reuses the same sleeve execution and blend as the portfolio backtest. It scales each sleeve's original traded weight panel by its time-varying ERC weight and portfolio leverage, recomputes share-level trades, matches round trips, and upserts the strategy ledger.

## Trigger

The `strategy-moves` command in [portfolio/cli.py](../../src/portfolio/cli.py) invokes `StepStrategyMoves`. The daily prediction DAG runs it after the newest model scores have been persisted.

## Sequence diagram

~~~mermaid
sequenceDiagram
  participant Moves as StepStrategyMoves
  participant Portfolio as StepPortfolio
  participant Sleeve as Strategy sleeve
  participant Blend as ERC blend
  participant Blotter as Trade blotter
  participant Store as DataStore
  Moves->>Portfolio: load sleeves and blend
  Portfolio->>Sleeve: run PortfolioInputs
  Sleeve-->>Portfolio: StrategyResult
  Portfolio->>Blend: align returns and size weights
  Blend-->>Moves: sleeve weights and leverage
  Moves->>Blotter: rescale book and compute shares
  Blotter-->>Moves: FIFO round trip ledger
  Moves->>Store: upsert strategy rows
~~~

## Steps

1. [StepPortfolio](../../src/portfolio/step_portfolio.py) constructs shared inputs and runs every configured [strategy sleeve](../concepts/strategy-sleeves.md).
2. Portfolio blending returns aligned sleeve streams, dynamic weights, and global leverage.
3. [StepStrategyMoves](../../src/portfolio/step_strategy_moves.py) multiplies each sleeve's `book_weights` by its allocation factor over time.
4. [blotter.py](../../src/strategies/utils/blotter.py) computes share moves using the sleeve's own prices and cost settings.
5. [positions.py](../../src/strategies/utils/positions.py) FIFO-matches entries and exits.
6. The ledger is upserted so an existing opening row can receive exit price and realized P&L later.

## Failure modes

Scaling an already-created dollar blotter is wrong when capital and integer shares vary through time. Missing price panels prevent reconstruction, a zero allocation produces no ledger rows, and unknown or empty sleeves are dropped before blending. The portfolio requires at least two sleeves with usable returns.

## Related

- [Portfolio](../modules/portfolio.md)
- [Strategies](../modules/strategies.md)
- [Feature-to-portfolio architecture](../architecture/feature-to-portfolio.md)
- [Model training and daily prediction](./model-training-and-prediction.md)
