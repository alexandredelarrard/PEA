---
source_commit: 55d6e2d199fa9a83aca7d5442549eeb66eedd189
title: Portfolio construction
description: Sector quotas, model rankings and signed risk weights inside portfolio.
type: module
tags:
  - wiki
  - module
---
# Portfolio construction

## Summary

Signal-to-position construction lives inside [portfolio](./portfolio.md), under `transformers/`. The previous `src/strategies` package, registry, long-only sleeve and per-sleeve YAMLs have been removed. There is one accounting engine for the requested long/short, S&P and cash budgets.

## Interfaces and files

- [contracts.py](../../src/portfolio/utils/contracts.py) defines validated settings, market inputs and simulation results.
- [signals.py](../../src/portfolio/transformers/signals.py) scores the saved cutoff ensemble through public modelling utilities.
- [construction.py](../../src/portfolio/transformers/construction.py) builds sector quotas and signed gross weights.
- [simulation.py](../../src/portfolio/transformers/simulation.py) executes real quantity deltas and follows capital.
- [portfolio.yml](../../configs/portfolio.yml) owns the shared assumptions.

## Selection and risk

A scenario chooses 10,50 or 100 names on each side across all sectors, with no overlap. Sectors need enough eligible names and complete past risk/momentum windows. For N smaller than eligible sector count, higher-momentum sectors receive the available slots; otherwise sectors receive initial representation before remaining capacity-aware deterministic allocation.

Gross sector budgets are proportional to `1 + percentile_rank(momentum)`, bounded to a 1–2 multiplier. Each sector allocates half its budget to longs and half to shorts. Within each sector/leg, inverse variance weights minimize the diagonal risk objective. A lack of capacity interrupts the scenario instead of quietly changing N.

Sector and global dollar neutrality hold at LS selection/reweighting dates. Quantities drift between dates. Weekly capital-only changes scale that drift until the next LS selection.

## Related

- [Capital budgets](../concepts/strategy-sleeves.md)
- [Backtest flow](../flows/portfolio-to-trade-ledger.md)
- [Changing models or the backtest](../guides/add-a-model-or-sleeve.md)
