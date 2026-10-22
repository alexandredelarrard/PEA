---
source_commit: 55d6e2d199fa9a83aca7d5442549eeb66eedd189
title: Model portfolio backtest
description: Saved cutoff models to quantity accounting and an offline portfolio report.
type: flow
tags:
  - wiki
  - flow
---
# Model portfolio backtest

## Trigger

The weekly modelling DAG calls [portfolio backtest](../../src/portfolio/cli.py) with the exact training ID after assessment. A manual run uses the same command. The daily prediction DAG continues to publish forecasts; its old ledger task and command have been removed. Existing historical strategy-table data is untouched.

## Sequence

~~~mermaid
sequenceDiagram
  participant CLI as Portfolio CLI
  participant Run as Assessed run
  participant Store as DataStore
  participant Step as StepPortfolio
  participant Engine as Quantity simulator
  participant HTML as Offline report
  CLI->>Run: pin cutoff identity and outcome frontier
  CLI->>Step: validated run and output routing
  Step->>Store: bounded price, sector and feature reads
  Step->>Run: verify and load saved member bytes
  Step->>Engine: nine N/cadence books plus S&P references
  Engine-->>Step: daily NAV, orders, positions and status
  Step->>HTML: atomically publish backtest_portfolio/report.html
~~~

## Accounting clocks

1. Start each book at 100 before setup costs.
2. Freeze selection and target weights after close J using only past model/risk/momentum information.
3. Let previous quantities earn total returns and eligible cash interest until J+1 close.
4. Apply actual share-basis changes, then execute the net changes for coincident capital/selection schedules.
5. Deduct all buy/sell fees once and solve after-fee budgets without debt.
6. Preserve quantities until the next required change; report both allocation and sector drift.

[Construction](../../src/portfolio/transformers/construction.py) and [simulation](../../src/portfolio/transformers/simulation.py) share the typed [contracts](../../src/portfolio/utils/contracts.py). The renderer uses saved results rather than a separate JavaScript financial engine.

## Failure modes

Missing held prices, total returns, execution prices, risk history or capacity explicitly marks an incomplete scenario. A decision is never reselected using J+1 availability. Incomplete periods receive matching-period S&P metrics and visible dates. If the benchmark or all scenarios fail, the prior complete report is preserved. Rendering/publication failure also preserves prior bytes.

## Related

- [Portfolio](../modules/portfolio.md)
- [Construction](../modules/strategies.md)
- [Model training and daily prediction](./model-training-and-prediction.md)
