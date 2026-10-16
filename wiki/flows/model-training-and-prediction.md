---
title: Model training and daily prediction
description: Weekly holdout training and production refit, followed by daily artifact-backed scoring.
type: flow
tags:
  - wiki
  - flow
---
# Model training and daily prediction

## Summary

Training and prediction have different cadences and data requirements. The weekly DAG first measures a holdout model and portfolio, then refits on all available history for production. The daily DAG loads that full-history artifact and scores the newest cube rows without retraining.

## Trigger

[dag_modelling.py](../../src/dags/dag_modelling.py) schedules weekly training. [dag_strat_prediction.py](../../src/dags/dag_strat_prediction.py) runs daily and is also triggered after cube aggregation.

## Sequence diagram

~~~mermaid
sequenceDiagram
  participant Weekly as Weekly DAG
  participant Model as StepLongShort
  participant Backtest as StepPortfolio
  participant Artifacts as Model artifacts
  participant Daily as Daily prediction DAG
  participant Cube as Cube table
  participant Pred as Predictions table
  Weekly->>Model: train holdout ensembles
  Model->>Artifacts: save models metadata diagnostics
  Weekly->>Backtest: run out of sample portfolio
  Weekly->>Model: full train on all history
  Model->>Artifacts: replace production artifacts
  Daily->>Artifacts: load production ensemble
  Daily->>Cube: load latest feature rows
  Daily->>Pred: write predictions latest
~~~

## Steps

1. `StepLongShort.run_train()` in [step_long_short.py](../../src/modelling/steps/step_long_short.py) resolves cube columns and horizons without loading the full cube.
2. It streams one projected, labelled horizon panel at a time. Five outer chronological folds use one early-stopped LightGBM each; a purged latest-10% tail inside the training period selects its rounds. That same model scores the untouched outer test. Linear/RF use all outer-training rows. The median selected count drives a fresh all-eligible-row cutoff fit; no second fit occurs inside a fold.
3. It saves one pickle per member, `metadata.json`, predictions, signals, the `Monitor` diagnostics (IC and drawdown, SHAP, PDPs), a label-only quantile backtest per horizon, and run KPIs.
4. The portfolio backtest uses the holdout artifacts before production refitting.
5. `run_train(full_history=True)` consumes the saved fold selection and fits all label-matured rows through the latest complete cube/price session, without repeating cross-validation. Missing or incompatible selection metadata requires `modelling train` first.
6. `run_predict()` first raises `StaleInputsError` and scores nothing when the share of universe tickers with a fresh own latest date in `prices`, `fundamentals_sharadar` or `fundamentals_history` is below `prediction_fresh_share`, or when the cube ends before `prices` ([freshness.py](../../src/utils/freshness.py)). It then loads only latest feature rows and writes member, horizon-ensemble, and blended predictions in long form.

## Failure modes

Random cross-validation leaks future information, so the code uses time-series folds and an embargo. Loading all horizons or all cube columns at once can exceed memory. Scoring through a labelled-panel helper would drop the newest dates because their forward targets are still null; the dedicated latest-prediction path avoids target columns entirely.

## Related

- [Modelling](../modules/modelling.md)
- [Portfolio](../modules/portfolio.md)
- [Portfolio to trade ledger](./portfolio-to-trade-ledger.md)
- [Add a model or sleeve](../guides/add-a-model-or-sleeve.md)
