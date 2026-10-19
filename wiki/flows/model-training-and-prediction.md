---
title: Model training and daily prediction
description: Five-fold/cutoff training, saved-model diagnostics, portfolio backtest and final production refit.
type: flow
tags:
  - wiki
  - flow
source_commit: 8e4bd26a0e837b01f0df16e29703dd8918458b77
---
# Model training and daily prediction

## Summary

The weekly [modelling DAG](../../src/dags/dag_modelling.py) runs four sequential tasks under one dated run ID. The daily [prediction DAG](../../src/dags/dag_strat_prediction.py) reads only a complete production generation and scores current feature rows without retraining.

~~~mermaid
sequenceDiagram
  participant Weekly as Weekly DAG
  participant Train as StepLongShort
  participant Files as Dated run artifacts
  participant Assess as StepAssessLongShort
  participant Portfolio as StepPortfolio
  participant Production as Production generation
  participant Daily as Daily prediction
  Weekly->>Train: train_model
  Train->>Files: original five folds, OOF, cutoff members
  Weekly->>Assess: backtest_modelling
  Files->>Assess: saved models and provenance
  Assess->>Files: scores, IC/IR, SHAP/linear/PDP, report.html
  Weekly->>Portfolio: backtest_portfolio
  Files->>Portfolio: same run's cutoff members
  Portfolio->>Files: portfolio outputs
  Weekly->>Train: full_train, reuse saved fold counts
  Train->>Files: full-history attempt
  Train->>Production: complete immutable generation, atomic metadata switch
  Production->>Daily: pin one metadata snapshot and load members
~~~

## Steps and clocks

1. [StepLongShort.run_train()](../../src/modelling/steps/step_long_short.py) resolves configured family/horizon inputs against the cube schema and streams one projected labelled horizon at a time. Configured unavailable names are recorded.
2. Five expanding forward folds validate 2000–02,2003–06,2007–12,2013–18 and2019–21, training from 1999. Actual observed price sessions define label ends. Outer training labels end strictly before test feature dates; the early-stop training labels end strictly before the inner validation dates. All development/test/early-stop outcomes must be known by `train.outcome_cutoff` (31 December 2021), in addition to the feature ceiling.
3. Each fold fits three original members. LightGBM uses the purged latest 10% inner tail for early stopping and that same model scores the outer test. Linear/RF use all eligible outer-training rows. Models, row endpoints, selected counts and original OOF predictions are saved before releasing each fold. There is no automatic feature selection or hyperparameter search.
4. After all five folds, fresh cutoff members fit every eligible development row, using median fold-selected LightGBM rounds. Cutoff files remain in that run, separate from production.
5. [StepAssessLongShort.run()](../../src/modelling/steps/step_assess_long_short.py) loads those exact saved members. It scores complete cross-sections, including unknown-label rows, before joining future labels. Original folds are evaluated only in their declared outer periods; cutoff members score from 3 January 2022 through each horizon's latest mature feature date. No fitting occurs in assessment.
6. Assessment saves per-model/ensemble IC and IC-IR, original-fold/cutoff SHAP or linear contributions, importance, cutoff PDP and the offline [two-section report](../modules/modelling.md#report-plots). Sampled tree scatter remains visible alongside five held-out fold curves and cutoff. Completed analysis can rerender with `assess --report-only` without DB reads or unpickling. Only a complete assessment advances `latest.json`.
7. [Portfolio backtest](../../src/portfolio/cli.py) requires the same completed run ID and uses its cutoff models. Portfolio dates (June 2022 onward), costs and construction remain configured as before. Outputs stay under that run's `portfolio/`.
8. The final task calls the existing `StepLongShort.run_train(full_history=True)` through `modelling full-train`. It reuses that run's saved CV counts, ignores historical development ceilings, and freshly fits all finite labels mature by the latest complete cube/price session. It never repeats CV. Each attempt and complete production generation is immutable; one atomic metadata replacement publishes the complete set.
9. Daily `run_predict()` checks source/cube freshness through [freshness.py](../../src/utils/freshness.py), pins one production metadata snapshot, then scores newest feature-only rows and writes member/horizon/blended `predictions_latest`. It does not require current targets.

## Failure and recovery

Any failed task blocks later tasks. Assessment cannot repair missing models by fitting. Explicit `--resume` accepts only verified matching checkpoints; changed historical inputs or configuration require a new run. Failed full-history attempts are retained, and retry uses a fresh attempt ID. The prior production metadata/generation remains readable until a complete new set is published; readers that pinned it finish on that generation.

Forward splits and horizon purging address label overlap, not current-universe/GICS, static peer or vendor-history look-ahead. Historical claims remain conditional on those accepted upstream limitations. Poor IC is measured rather than used as a reason to alter features or parameters in this DAG.

## Related

- [Modelling and artifacts](../modules/modelling.md)
- [Model and portfolio contracts](../reference/modelling-and-portfolio.md)
- [Dated run commands](../guides/run-the-pipeline.md#dated-modelling-assessment)
- [Independent validation](../guides/testing.md#assessment-contracts-and-financial-replay)
- [Portfolio to trade ledger](./portfolio-to-trade-ledger.md)
