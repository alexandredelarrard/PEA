---
title: Modelling
description: Dated five-fold and cutoff training, saved-model assessment, production generations and daily scoring.
type: module
tags:
  - wiki
  - module
source_commit: 2c9504c51e3c1d230d3d3623ff03eb8cffc4d11a
---
# Modelling

## Summary

`src/modelling` trains and scores the cross-sectional long/short signal. [StepLongShort](../../src/modelling/steps/step_long_short.py) fits models and prints basic fold IC/IR; [StepAssessLongShort](../../src/modelling/steps/step_assess_long_short.py) reloads saved models and current data to recompute every assessment. [Portfolio](./portfolio.md) converts sleeve books into returns separately.

The weekly sequence is training → model assessment → portfolio backtest → production refit. Train generates a new run ID; the [DAG](../../src/dags/dag_modelling.py) captures its successful final stdout value through XCom and pins that ID in every downstream task. A failed task blocks its successors. See the [training flow](../flows/model-training-and-prediction.md).

## Responsibilities

- Resolve configured family/horizon pools against actual cube columns, recording unavailable configured names.
- Stream projected data for one horizon at a time. Keep scoring rows with unknown targets; fitting uses finite, mature labels only. [features.py](../../src/modelling/utils/features.py) owns numeric-text validation and categorical preparation.
- Fit five expanding chronological folds by default. Observed-session embargo and actual forward-label endpoints exclude overlapping training targets.
- Fit LightGBM on all eligible fold-training rows and use labelled external-test rows for RMSE early stopping. There is no inner validation tail. This deliberately makes fold IC/IR influenced by round selection.
- Score all test titles, normalize all finite predictions, and calculate IC on finite prediction/observed-label pairs.
- Save original models and minimal fold/aggregate JSON, without SHAP/PDP, importance, OOF prediction files, post-2022 metrics, HTML or historical prediction/signal table writes.
- Fit fresh cutoff models on all eligible development rows. LightGBM uses the upward-rounded mean of best rounds from the last two configured chronological folds, separately by horizon.
- Let assessment own all explanations, rich metrics and reports. Refit all currently mature rows for production using the selected run's saved fitting settings, resolved schema, seed and final rounds, without repeating CV.

## Layers and public API

| Layer | Owning files | Contract |
| --- | --- | --- |
| Steps | [step_long_short.py](../../src/modelling/steps/step_long_short.py), [step_assess_long_short.py](../../src/modelling/steps/step_assess_long_short.py) | Training/prediction and fresh saved-model assessment; inherit Step. |
| Transformers | [base.py](../../src/modelling/transformers/base.py), [lightgbm_model.py](../../src/modelling/transformers/lightgbm_model.py), [random_forest.py](../../src/modelling/transformers/random_forest.py), [linear_regression.py](../../src/modelling/transformers/linear_regression.py) | Family fit/predict/evaluate/importance and transformer pickles. |
| Diagnostics | [assessment_diagnostics.py](../../src/modelling/transformers/assessment_diagnostics.py), [assessment_report.py](../../src/modelling/transformers/assessment_report.py) | Sampled effects and self-contained HTML rendering, without fitting. |
| Utilities | [artifacts.py](../../src/modelling/utils/artifacts.py), [cv.py](../../src/modelling/utils/cv.py), [ensemble.py](../../src/modelling/utils/ensemble.py), [assessment_metrics.py](../../src/modelling/utils/assessment_metrics.py), [modelling_runs.py](../../src/utils/modelling_runs.py) | Model integrity, temporal splits, scoring, metrics and shared run selection. |

`StepLongShort.run()` delegates to `run_train()`. `run_train(full_history=True)` freshly fits production models; `run_predict(n_dates=1)` performs inference only. `StepAssessLongShort(context, config, run_id=None).run()` always recomputes from saved models and current data. Standalone assess selects the latest successfully completed training by completion UTC; optional CLI `--run-id` or constructor `run_id` selects exactly that run. It logs training UTC and ID and warns if age is strictly greater than 24 hours. An invalid explicit selection refuses without fallback.

The [CLI](../../src/modelling/cli.py) exposes train, assess, full-train and predict. Train has no supplied ID or partial resume; assess has no resume/report-only mode. Standalone portfolio/full-train default to the latest successfully assessed training and accept an exact ID. Monitor and the unused label-only Backtest have been removed; their live diagnostic helpers belong to assessment diagnostics.

## Dated artifacts and production models

[modellling.yml](../../configs/modellling.yml) owns target/horizon/CV/diagnostic controls; `configs/models/` owns family features and fitting parameters. Paths resolve under DATA_STORE unless absolute. Tree categorical inputs remain selected in ordinary columns lists; lgbm.categoricals only assigns types. The shared monotonic map applies to retained inputs.

Under `model.assessment.output_dir` (normally output/modelling_backtest), each fresh run contains:

- metadata.json: run identity, immutable training completion UTC, resolved schema, declared folds, summaries and horizon weights.
- training/configuration.json: the allowlisted seed, train/model fitting controls and three family blocks; training/calendar.parquet records the training calendar. No provider secrets or application-output switches are captured.
- folds/h<h>/fold<n>/: original models and minimal fold audit; cutoff/: fresh development models.
- assessment/h<h>/fold<n>/: newly computed complete scores, explanation samples, SHAP or linear contributions, importance and curves. Fold 0 means cutoff/post-2022.
- assessment/metadata.json and report.html: completed assessment provenance and self-contained report.
- portfolio/: portfolio outputs; full_history/<attempt_id>/: fresh final refit attempts.

There are no replay/source/configuration hashes, analysis caches or historical-run migration requirements. Saved member byte inventories remain verified at their owning load boundary. Dated assessment and the [dated portfolio CLI](../../src/portfolio/cli.py) refuse a missing original inventory before any member is consumed; generic daily legacy loading retains its existing optional-member behavior. Current fitting YAML drift does not invalidate saved-model assessment. Full-train restores the exact saved resolved feature schema, so newly appeared cube columns cannot silently enter that refit.

Each assessment builds in fresh run-local scratch space. An exclusive same-run lock prevents competing publication. Successful publication replaces assessment/ and report.html with bounded backup/rollback; completion is written last. A failed calculation preserves the prior complete result. Filesystem replacement is not one atomic transaction; retained backup evidence permits recovery after an interrupted swap. The next assess still recalculates everything. latest.json denotes assessment success only; selectors inspect complete manifests rather than relying on that pointer.

Production `model.models_dir` defaults to output/models. Full-history publication copies a complete verified inventory into immutable generations/<run_id>/<attempt_id>/, then atomically switches root metadata.json. A failure preserves the prior generation. Readers pin one metadata snapshot.

Each member is model_h<h>_<family>.pkl without context/config/logger. Generation routing remains transformer-pickle-generation-v2; daily production readers still accept legacy flat transformer-pickle-v1. The dated fresh-assessment-v1 workflow requires new training; old assessments are not migrated. The existing production training contract remains matured-fold-es-v1.

## Report plots

The HTML contains:

1. IC/IC-IR through time by model, horizon and Fold tests / Post-2022 population.
2. All retained-feature importance and signed effect/PDP by horizon, family and feature.

Fold-test curves concatenate only each original fold's configured test dates. They do not fill gaps or average fold models over the entire history. Effective maturity endpoints are shown per horizon. Default five-fold windows cover 2000–2021; a reduced three-fold rehearsal covers 2007–2021.

Signed tree plots retain sampled cutoff scatter and binned-mean curves for every declared original fold plus cutoff. Tree importance is date-balanced mean absolute sampled SHAP. Linear importance is absolute saved standardized beta. Signed linear contributions use each member's saved training scaler; hover retains raw input and signed beta. Missing inputs contribute zero and have no observed x/bin; the intercept is explicit. Zero/all-missing retained features remain selectable. PDP is cutoff-only and varies one input while holding the sampled background fixed; it is not a causal estimate.

Diagnostics settings control assessment only: shap_sample limits explanation rows, dependence_sample limits scatter, pdp_grid is the maximum continuous quantile-grid size, pdp_background limits rows averaged per PDP point, and rolling_window controls smoothing. Defaults remain 12000/1000/30/2000/252. These budgets never limit IC scoring.

Daily Spearman IC uses finite prediction/label pairs after normalization over all finite predictions. IC-IR is mean daily IC / population standard deviation × sqrt(252 / horizon), distinct from portfolio-return IR. Rolling statistics require the full configured observed-session support; unsupported and constant values stay missing.

Current-roster/GICS, static-peer and vendor-history limitations remain. Purges prevent model/label overlap but do not certify upstream point-in-time history. Fold round selection uses the external test; post-2022 never selects rounds.

## Dependencies and validation

All application tabular access uses [DataStore](./data-store.md). Daily prediction alone retains predictions_latest writes. Strategy/backtest scores complete feature-only populations before return availability is considered.

The [independent validator](../../src/validate/checks/modelling.py) reconciles saved scores and metric inputs, source labels and purges, resolved member inputs, native predictions/contributions/PDP, production publication and report content. Incomplete evidence abstains. See [testing](../guides/testing.md#assessment-contracts-and-financial-replay) and [run commands](../guides/run-the-pipeline.md#dated-modelling-assessment).

[Architecture guards](../../tests/modelling/test_architecture_guard.py) enforce package boundaries. [Artifact tests](../../tests/modelling/test_run_artifacts.py) cover publication and pinned readers. [Assessment integration](../../tests/modelling/test_assessment_evaluation.py) forbids fitting, independently validates outputs and exercises a fresh second assessment after old scores are corrupted.

The [2026-10-21 acceptance record](../guides/testing.md#modelling-workflow-acceptance-2026-10-21) covers this workflow on the user-approved isolated synthetic dataset. The [2026-10-19 record](../guides/testing.md#modelling-dag-acceptance-2026-10-19) describes the prior workflow.
