---
source_commit: 55d6e2d199fa9a83aca7d5442549eeb66eedd189
title: Portfolio
description: Independent model long/short, S&P 500 and cash backtest and offline HTML.
type: module
tags:
  - wiki
  - module
---
# Portfolio

## Summary

The portfolio is a standalone model backtest with three capital budgets: long/short equities, an S&P 500 total-return proxy, and interest-bearing free cash. There is no application server, strategy registry, ERC blend or trade-ledger writer.

## Entry point and saved models

Run `python -m src portfolio backtest -c ./configs --run-id <assessed-run>`. Omitting the ID selects the latest successfully assessed run. The [CLI](../../src/portfolio/cli.py) refuses incompatible identities, full-history models, absent inventories and training/selection outcomes after the configured cutoff before consuming model members. The [signal transformation](../../src/portfolio/transformers/signals.py) verifies the saved ensemble and uses only training IC/IR for nonnegative horizon weights.

[StepPortfolio.run](../../src/portfolio/steps/step_portfolio.py) reads existing data, scores projected feature-only monthly batches preserving full daily cross-sections, simulates all nine books, and publishes `<model-run>/backtest_portfolio/report.html`. It never retrains or writes source/strategy tables.

## Construction and timing

N = 10/50/100 means N longs and N shorts in total. [Construction](../../src/portfolio/transformers/construction.py) allocates deterministic quotas across sectors, ranks names inside each sector, tilts gross sector budgets toward higher trailing momentum, and weights each leg by inverse trailing variance. Each sector has equal long/short dollars when selection is rebalanced.

The calendar uses sessions witnessed by stock prices, total-return/share-basis panels or feature rows; macro-only carried holiday/weekend rows do not create an execution session. A date absent from every such source cannot be independently recovered from carried macro prices.

The signal is fixed at close J. Transactions occur at close of the next observed market session J+1. Old quantities earn the intervening return. Daily, weekly and monthly LS selection are separate from the weekly default allocation of capital. A budget-only reallocation scales the existing book; it does not restore sector weights before the next LS selection date.

## Accounting

The [simulation](../../src/portfolio/transformers/simulation.py) holds signed fractional quantities between schedules. It adjusts quantities for actual corporate actions, books distributions implied by total returns once, and charges proportional costs on actual net quantity changes. The stock quote reconstruction in [inputs](../../src/portfolio/utils/inputs.py) reuses the shared vendor/actual share-basis resolver and stored level factors; missing factors are not replaced with one.

A budget of 50 buys 25 longs and sells 25 shorts; reserved LS cash/collateral and blocked short proceeds never become additional free capital. Only the separate cash pocket earns the effective annual rate over calendar days. Targets use current capital after financed costs. Negative capital or missing required held/execution observations explicitly interrupts a scenario.

## Configuration and report

[portfolio.yml](../../configs/portfolio.yml) owns allocation, dates, cadence, risk/momentum windows, cash rate and cost assumptions. Defaults are 50/25/25, 3.3% effective annual cash, 126-session sector momentum, 63-session variance, 5 bps friction on each side and 0.206 bps additional sell costs. These are research assumptions, not an optimized allocation or exact historical broker tariff.

The offline report compares net/gross base100 curves and drawdowns against S&P, records capital by pocket, fees, monthly position changes, orders, sector drift and model provenance. N/cadence filters select separately simulated books. Incomplete scenarios show their reason and effective period; benchmark metrics use that same period. Changing allocation requires a new Python run. Failed publication preserves the prior complete HTML.

## Validation and limits

Known-truth tests under [tests/portfolio](../../tests/portfolio/test_simulation.py) cover timing, financing, cash interest, splits/dividends, sector neutrality, missing data, training-outcome guards and report recovery. A bounded independent replay also reconciled existing NVDA/MSFT prices through the 2024 NVDA split. No compatible assessed cutoff run was available during implementation, so full economic performance of the nine model strategies was not verified.

The current universe/sector classification is retrospective. Shorts are assumed available and fractional, without borrow charges. Dollar neutrality does not imply beta neutrality. Automatic volatility allocation, macro regime rules, Alpaca orders and additional asset classes are outside this version.

## Related

- [Portfolio construction](./strategies.md)
- [Capital budgets](../concepts/strategy-sleeves.md)
- [Backtest flow](../flows/portfolio-to-trade-ledger.md)
- [Modelling and portfolio contracts](../reference/modelling-and-portfolio.md)
