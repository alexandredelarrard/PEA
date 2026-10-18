---
title: Modelling
description: "Cross-sectional long/short ensemble: model transformers, the StepLongShort train/predict step, diagnostics and the label-only backtest."
type: module
tags:
  - wiki
  - module
source_commit: 800be024be975942df7dbf542f807a09af103933
---
# Modelling

## Summary

`src/modelling` trains and scores the long/short signal; it does not build portfolios. It has three layers:
`steps/` (one Step per strategy), `transformers/` (reusable model classes, the Monitor and the Backtest) and
`utils/` (functions shared by both). The long/short step trains one cross-sectional ensemble per forecast
horizon from the cube, then blends the horizons into one signal.

## Responsibilities

- Select configured feature families and target columns from the cube schema.
- Load and train one horizon at a time to bound memory (one projected load per horizon, float32).
- Use purged walk-forward cross-validation; training label endpoints must precede the validation/test feature dates, in addition to the configured embargo.
- Fit one early-stopped LightGBM per outer fold; use the median selected count for final all-eligible-row fits, and reuse saved fold selection in production without repeating CV.
- Train ElasticNet / ridge, LightGBM and LightGBM random-forest members (`model.ensemble`).
- Report CV and OOS IC (with cumulative-IC drawdown), SHAP, dependence plots, PDPs and importance.
- Run a label-only quantile backtest of each horizon's OOS predictions as the last training stage.
- Persist member pickles, metadata, predictions and the blended signal; score the newest cube dates
  without requiring matured targets (float64 reads).

## Layers

| Layer | Files | Rule |
| --- | --- | --- |
| `steps/` | `step_long_short.py` | orchestrates; inherits `Step` |
| `transformers/` | `base.py`, `lightgbm_model.py`, `random_forest.py`, `linear_regression.py`, `monitor.py`, `backtest.py` | never imports `steps` |
| `utils/` | `cv.py`, `metrics.py`, `ensemble.py`, `features.py`, `panel.py`, `artifacts.py` | imports neither `transformers` nor `steps` |

The layout and the layering are pinned by
[test_architecture_guard.py](../../tests/modelling/test_architecture_guard.py). Consumers that only score
(`ls_model`, the app) reach modelling through `utils` only: a pickled member unpickles to its own class.

## Public API / entry points

- `StepLongShort.run_train(full_history=False)` and `run_predict(n_dates=1)` in
  [step_long_short.py](../../src/modelling/steps/step_long_short.py); `run()` calls `run_train()`. This is
  the one Step with two public methods besides `run()` (training and prediction share their state).
- The `train`, `full-train`, and `predict` commands in [modelling/cli.py](../../src/modelling/cli.py).
- Model families (`model_class(name)` in [transformers/__init__.py](../../src/modelling/transformers/__init__.py)):
  `BaseModel(context, config, family, horizon)` with `load_data`, `fit`, `predict`, `evaluate`, `importance`,
  `save` and `load`. Each family reads its own YAML block, including `task: regression | classification`
  (classification needs a {0, 1} label; the linear family is regression-only).
- `Monitor` ([monitor.py](../../src/modelling/transformers/monitor.py)) and `Backtest`
  ([backtest.py](../../src/modelling/transformers/backtest.py)).

## Key files

- [steps/step_long_short.py](../../src/modelling/steps/step_long_short.py) owns per-horizon training, the
  horizon blend and latest prediction; it returns a `TrainResult` instead of keeping frames on `self`.
- [transformers/lightgbm_model.py](../../src/modelling/transformers/lightgbm_model.py) holds `train_booster`
  and the LightGBM family; [random_forest.py](../../src/modelling/transformers/random_forest.py) borrows
  its categoricals, monotone map and column fallback from the `lgbm` block.
- [transformers/linear_regression.py](../../src/modelling/transformers/linear_regression.py) is a pure-numpy
  ridge / elastic net on standardised features.
- [utils/artifacts.py](../../src/modelling/utils/artifacts.py) owns `models_dir`, member paths and
  `metadata.json`.
- [configs/modellling.yml](../../configs/modellling.yml) owns shared model and training settings;
  [LightGBM settings](../../configs/models/lgbm_modelling.yml) and the other family YAML files own features, hyperparameters and `task`.

## Artifacts

`model.models_dir` (default `output/models`, relative to the data store) holds one pickle per member,
`model_h<h>_<family>.pkl` (the fitted transformer without its context, config or logger), and
`metadata.json`, stamped `artifact_format: transformer-pickle-v1`. Metadata without that stamp (models
trained before the 2026-10 refactor) raises a "retrain" error. The additional training contract is `matured-fold-es-v1`; saved per-horizon fold counts, dates, metrics and resolved input/parameter signatures must match. Run `modelling train` first, then `full-train` can reuse that selection on all matured rows. See [the lifecycle and schema migration](../reference/modelling-and-portfolio.md).

## Dependencies

The step consumes [cube aggregation](./data-aggregate.md) output through [DataStore](./data-store.md).
[Strategy sleeves](./strategies.md) score the saved ensemble through `utils` to build their books.

## Participates in

- [Model training and daily prediction](../flows/model-training-and-prediction.md)
- [Feature-to-portfolio architecture](../architecture/feature-to-portfolio.md)
- [Add a model or sleeve](../guides/add-a-model-or-sleeve.md)

## Related

- [Modelling, strategies, and portfolio](../reference/modelling-and-portfolio.md)
- [Strategy sleeves](../concepts/strategy-sleeves.md)
- [Tests module](./tests.md)

## Isolated horizon-60 assessment

`StepAssessLongShort` in [step_assess_long_short.py](../../src/modelling/steps/step_assess_long_short.py) runs the explicit `modelling assess` command. It reuses `StepLongShort.run()` for the current-config baseline and selected cutoff fit, and the original cross-validation path for each fixed-parameter candidate. Numeric strings use the shared strict preparation in [features.py](../../src/modelling/utils/features.py); only declared categoricals receive missing code -1.

Development selects separate LGBM, LightGBM RF and elastic-net pools using five purged forward folds and pooled daily h60 IC-IR. Candidate, input, fixed-control, selection-source and library identities gate explicit resume. Frozen pools and cutoff artifacts precede sealed outcomes. The run owns all models, OOF/member predictions, SHAP/PDP and report evidence; its store wrapper refuses application mutations. Normal weekly training does not implicitly run feature search.

[assessment_diagnostics.py](../../src/modelling/transformers/assessment_diagnostics.py) supplies original-fold explanations and retrospective cohort/sector checks. [assessment_report.py](../../src/modelling/transformers/assessment_report.py) renders one offline HTML file from artifacts. A report rerender performs no fit or application read. See [assessment operations](../guides/run-the-pipeline.md) and [validation evidence](../guides/testing.md). Historical classification, peer-vintage and universe limitations must accompany claims; a temporal split alone does not certify point-in-time inputs.
