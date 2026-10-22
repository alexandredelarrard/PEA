---
source_commit: 55d6e2d199fa9a83aca7d5442549eeb66eedd189
title: Portfolio capital budgets
description: Self-financing long/short, S&P 500 and free cash budgets.
type: concept
tags:
  - wiki
  - concept
---
# Portfolio capital budgets

## Definition

The backtest assigns fixed target percentages to long/short, S&P500 and free cash. Default50/25/25 is a user-settable research choice in [portfolio.yml](../../configs/portfolio.yml), not an automatic risk-allocation result.

At an allocation date, targets apply to current NAV after all net transaction costs. A value 110 with50/25/25 targets allocates 55/27.5/27.5 before any applicable costs. Between dates the values drift with holdings; there is no daily implicit weight reset.

## Long/short capital

LS capital C funds a gross book C: C/2 long and C/2 short at a dedicated LS rebalance. Its reserved collateral and short-sale proceeds remain in that budget; they cannot fund extra S&P or remunerated cash. A stock dividend/split changes actual quantities without creating an order; total-return distributions credit longs and debit shorts once.

Allocation cadence and LS selection cadence are independent. A budget-only change scales the old signed quantities proportionally. Selection dates recompute names, sector budgets and risk weights using information from the preceding close.

## Interest and expenses

Only the free cash pocket compounds at the chosen annual effective rate over elapsed calendar days. LS collateral earns no extra interest. Actual buy/sell deltas pay the configured proportional costs, including short opening and cover. The financing equation preserves nonnegative cash without introducing a loan.

See [simulation.py](../../src/portfolio/transformers/simulation.py) and the [contracts](../../src/portfolio/utils/contracts.py).

## Related

- [Portfolio](../modules/portfolio.md)
- [Construction](../modules/strategies.md)
- [Backtest flow](../flows/portfolio-to-trade-ledger.md)
