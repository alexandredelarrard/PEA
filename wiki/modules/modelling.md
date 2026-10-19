---
title: Modelling
description: Dated five-fold and cutoff training, saved-model assessment, production generations and daily scoring.
type: module
tags:
  - wiki
  - module
source_commit: 48a92a6f4103e57ef4c4b0cfa8f018d13ffca97e
---
# Modelling

## Summary

`src/modelling` trains and scores the cross-sectional long/short signal. [StepLongShort](../../src/modelling/steps/step_long_short.py) fits models; [StepAssessLongShort](../../src/modelling/steps/step_assess_long_short.py) reads the saved models and produces diagnostics. [Portfolio](./portfolio.md) converts sleeve books into portfolio returns separately.

The weekly sequence is training → model assessment → portfolio backtest → production refit. All four tasks share one explicit safe run ID. Assessment failure prevents both downstream tasks. The [training flow](../flows/model-training-and-prediction.md) describes the clocks and handoffs.

## Responsibilities

- Resolve configured family/horizon pools against actual cube columns, recording unavailable configured names.
- Stream projected data for one labelled horizon at a time, using numeric-text validation and declared categorical preparation from [features.py](../../src/modelling/utils/features.py).
- Fit five original expanding chronological folds, purging actual forward-label endpoints before inner validation and outer test dates.
- Use each fold's early-stopped LightGBM directly on its outer period; fit fresh cutoff members after CV using median selected LightGBM rounds. Linear and RF use all eligible outer-training rows.
- Save original member pickles, OOF predictions, counts and provenance before releasing each fold. Keep the configured feature pools, parameters, monotonic directions and blending rules.
- Explain saved original fold/cutoff members without fitting or feature search; report all three families and the ensemble across h30/h60/h90.
- Refit all latest mature rows for production using the named run's saved counts, without repeating CV. Daily scoring loads a complete production generation and needs features only.

## Layers and public API

| Layer | Owning files | Contract |
| --- | --- | --- |
| Steps | [step_long_short.py](../../src/modelling/steps/step_long_short.py), [step_assess_long_short.py](../../src/modelling/steps/step_assess_long_short.py) | Training/prediction and saved-model analysis orchestration; inherit `Step`. |
| Transformers | [base.py](../../src/modelling/transformers/base.py), [lightgbm_model.py](../../src/modelling/transformers/lightgbm_model.py), [random_forest.py](../../src/modelling/transformers/random_forest.py), [linear_regression.py](../../src/modelling/transformers/linear_regression.py) | Family fit/predict/evaluate/importance and compatible transformer pickles. |
| Diagnostics | [assessment_diagnostics.py](../../src/modelling/transformers/assessment_diagnostics.py), [assessment_report.py](../../src/modelling/transformers/assessment_report.py) | Sampled effects and the offline artifact-only HTML renderer; never fit. |
| Utilities | [artifacts.py](../../src/modelling/utils/artifacts.py), [cv.py](../../src/modelling/utils/cv.py), [ensemble.py](../../src/modelling/utils/ensemble.py), [assessment_metrics.py](../../src/modelling/utils/assessment_metrics.py) | Artifact routing/identity, temporal splits, blending and metric arithmetic. No transformer/step imports. |

`StepLongShort.run()` delegates to `run_train()`. `run_train(full_history=True)` is the same trainer's final production path; `run_predict(n_dates=1)` performs inference only. `StepAssessLongShort.run()` reads an explicitly named completed training run. The [CLI](../../src/modelling/cli.py) exposes `train`, `assess`, `full-train` and `predict`; `assess --report-only` loads completed analysis without database reads or model unpickling.

[Monitor](../../src/modelling/transformers/monitor.py) retains reusable standalone helpers; operational training no longer writes its legacy report tree or duplicates assessment SHAP/PDP. The optional [label-only backtest](../../src/modelling/transformers/backtest.py) remains separate from the portfolio return backtest.

## Dated artifacts and production models

[modellling.yml](../../configs/modellling.yml) owns shared target/horizon/CV/diagnostic controls; YAML files under `configs/models/` own each family's pool, hyperparameters and regression/classification task. Runtime output paths resolve under `DATA_STORE` unless absolute.

`model.assessment.output_dir` defaults to `output/modelling_backtest`. Each `<run_id>` folder holds:

- `folds/h<h>/fold<n>/`: three original fitted transformers and metadata for each of five folds.
- `training/`: calendar, original OOF/member scores, fold counts, row/label endpoints, source/library identities and file hashes. Immutable `configuration.json` captures the resolved seed, training/model controls, three family blocks and application-output switches; provider credentials, environment and logging are excluded. Readers verify its bytes and reconstructed modelling signature before consuming the run.
- `cutoff/`: nine fresh cutoff members and metadata.
- `assessment/h<h>/fold<n>/`: feature-only scores, deterministic explanation samples, SHAP or linear contributions, importance and curves. Fold 0 is the cutoff backtest.
- `report.html`: self-contained model diagnostics; `portfolio/` contains that run's portfolio outputs.
- `full_history/<attempt_id>/`: fresh final members and their effective mature endpoints.

`latest.json` at the assessment root references the latest completed assessment, not whole-DAG success. Failed/completed dated runs are retained. Resume verifies configuration, source, library, calendar, prepared input and artifact identities; changed historical inputs require a new run.

Production `model.models_dir` defaults to `output/models`. Full-history publication copies all required members into immutable `generations/<run_id>/<attempt_id>/` and atomically replaces one root `metadata.json` only after verifying the complete inventory. A failed attempt leaves the prior generation available; retry allocates a new attempt. Readers pin one metadata snapshot for all member loads.

Each member remains `model_h<h>_<family>.pkl`, the fitted transformer without context/config/logger. New generation routing is `transformer-pickle-generation-v2`; new readers also accept legacy flat `transformer-pickle-v1`. Old readers reject the new routing. Rollback to old code requires restoring compatible earlier metadata first. The training contract remains `matured-fold-es-v1`.

## Report plots

The HTML has two sections and three plot areas:

1. IC/IC-IR through time for linear, LGBM, RF and ensemble, with horizon, Validation/Backtest and Daily IC/Smoothed IC/Smoothed IC-IR controls.
2. All retained-feature importance and the chosen variable's effect, with independent horizon/family/feature/Signed effect/PDP controls.

Signed tree mode retains a cutoff sample scatter (feature value versus signed SHAP) and six binned-mean curves: the five original held-out folds plus cutoff. Original sampled values are saved for every fold. Tree importance is date-balanced mean absolute sampled SHAP. Linear importance is absolute saved standardized beta, with its sign in hover. The signed plot's x-axis uses each member's saved training-standardized input; y is `beta * ((x - mean_train) / std_train)`. Hover retains raw inputs and signed beta. Missing inputs contribute 0 and have no observed x/bin; the intercept is explicit. The view is labelled Signed linear contribution. Zero/all-missing retained features stay selectable. PDP uses only the cutoff member and holds other sampled inputs fixed; correlated inputs limit causal interpretation.

Daily Spearman IC uses finite prediction/label pairs after complete cross-sections are scored. Rolling statistics require all configured 252 observed sessions; undefined/constant values stay missing. IC-IR scales daily IC dispersion by `sqrt(252 / horizon)` and is not portfolio-return IR.

Historical current-roster/GICS, static-peer and vendor-vintage limitations remain. Forward folds and outcome cutoffs prevent model/label overlap; they do not certify upstream point-in-time inputs. The previously inspected 2022+ backtest is not a new sealed evaluation.

## Dependencies and validation

All application table reads/writes use [DataStore](./data-store.md). Strategy scoring reaches saved models through utilities. The [independent validator](../../src/validate/checks/modelling.py) reconciles saved predictions, chronology, actual retained inputs, IC/IR, native effects/PDP, publication and report identities; incomplete evidence abstains. See [testing](../guides/testing.md#assessment-contracts-and-financial-replay) and [run commands](../guides/run-the-pipeline.md#dated-modelling-assessment).

The [architecture guard](../../tests/modelling/test_architecture_guard.py) enforces dependency boundaries. [Artifact tests](../../tests/modelling/test_run_artifacts.py) cover complete publication, failure/retry and pinned readers; [assessment integration](../../tests/modelling/test_assessment_evaluation.py) forbids fitting and checks artifact-only rerender.
