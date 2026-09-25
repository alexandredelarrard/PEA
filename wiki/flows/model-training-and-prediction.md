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
  participant Model as StepModelling
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

1. `StepModelling.run()` in [step_train.py](../../src/modelling/long_short/step_train.py) resolves cube columns and horizons without loading the full cube.
2. It loads one labelled horizon panel at a time, cross-validates, trains each configured family, and frees the panel.
3. It saves member artifacts, `metadata.json`, predictions, signals, SHAP data, and run KPIs.
4. The portfolio backtest uses the holdout artifacts before production refitting.
5. `run(full_history=True)` trains the production ensemble on all history.
6. `predict_latest()` loads only latest feature rows and writes member, horizon-ensemble, and blended predictions in long form.

## Failure modes

Random cross-validation leaks future information, so the code uses time-series folds and an embargo. Loading all horizons or all cube columns at once can exceed memory. Scoring through a labelled-panel helper would drop the newest dates because their forward targets are still null; the dedicated latest-prediction path avoids target columns entirely.

## Related

- [Modelling](../modules/modelling.md)
- [Portfolio](../modules/portfolio.md)
- [Portfolio to trade ledger](./portfolio-to-trade-ledger.md)
- [Add a model or sleeve](../guides/add-a-model-or-sleeve.md)
