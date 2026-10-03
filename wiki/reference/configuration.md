---
title: Configuration
description: OmegaConf assembly, ownership of every configuration domain, and change rules.
type: reference
tags:
  - wiki
  - reference
  - configuration
---
# Configuration

## Purpose

This is the canonical map from a pipeline concern to its owning OmegaConf file. Tunable numbers belong in YAML; stable facts about external systems belong in [constants.py](../../src/constants/constants.py); table identity belongs in [schema.py](../../src/data_store/schema.py).

## Assembly

[read_config](../../src/utils/config.py) loads `configs/configs.yml`, recursively discovers the remaining `configs/**/*.yml` files, and merges them into one OmegaConf tree. [get_config_context](../../src/context.py) then configures logging, seeds random generators, resolves paths and environment values, and constructs the shared context.

> [!WARNING]
> Discovery order is filesystem order. A top-level key must have exactly one owning file; duplicate owners create order-dependent merges.

Callers use attribute access such as `self._config.build_cube.targets.horizons`. Do not replace OmegaConf or introduce hidden Python defaults for a required knob.

## Ownership map

| Top-level key | Owner | Responsibility |
| --- | --- | --- |
| `seed`, `data_extract` | [configs.yml](../../configs/configs.yml) | Global random seed, history windows, source refresh switches, redundant tickers, and manifest self-healing cadence. |
| `local.paths` | [paths.yml](../../configs/paths.yml) | Repository root, artifact store, output, model, log, and peer-dictionary locations. |
| `logging` | [logging.yml](../../configs/logging.yml) | Standard-library logging tree and the in-memory log format. |
| `peers` | [peers.yml](../../configs/peers.yml) | Business-similarity and return-correlation peer construction. |
| `build_cube` | [build_cube.yml](../../configs/build_cube.yml) | Betas, targets, feature transforms, intrinsic value, historical comparisons, institutional policies, and output switches. |
| `model`, `train` | [modellling.yml](../../configs/modellling.yml) | Ensemble composition, target choice, CV, diagnostics, decay, `models_dir` (artifact folder under the data store), `backtest.n_quantiles` (label-only backtest buckets), and train/holdout boundaries. Each family YAML under `configs/models/` (e.g. [lgbm_modelling.yml](../../configs/models/lgbm_modelling.yml)) also sets `task: regression \| classification`. The filename intentionally contains three “l” characters. |
| `linear`, `lgbm`, `random_forest` | [configs/models](../../configs/models/) | Family hyperparameters and family-specific feature columns. |
| `strategy_ls`, `strategy_eq_long_only` | [configs/strategy](../../configs/strategy/) | Sleeve construction only. |
| `portfolio` | [portfolio.yml](../../configs/portfolio.yml) | Sleeve selection, dates, global costs, risk targeting, leverage, capital, blend, and analysis output. |
| `data_availability`, `source_freshness`, `earnings_calls` | [data.yml](../../configs/data.yml) | Institutional availability boundaries, field/derived overrides, live-insider lag, the parity-approved bulk quarter, and the earnings-call extractor knobs. |
| validation keys | [validate.yml](../../configs/validate.yml) | Point-in-time publication clocks, observed-zero exceptions, and validation policy. |

Curated evidence registers under [configs/sec](../../configs/sec/) are versioned data contracts rather than tuning knobs:

- `registrant_cutover.json`: dated legal-filer chains;
- `entity_lineage_manual.json`: CIK-to-economic-entity adjudication;
- `symbol_tenure_manual.json`: evidenced half-open market-symbol intervals;
- `superinvestor_overrides.json`: Dataroma manager-code to CIK overrides and the codes recorded as unresolvable, each with its reason, read by `load_superinvestor_overrides` in [fetch_superinvestors.py](../../src/data_extract/utils/institutionals/fetch_superinvestors.py) and cached per resolved config directory;
- `employees_manual_roster.json`: per-accession employee headcount decisions (count or null) that replace the LLM for a filing with no table row.

Runtime readers consume their validated/materialized representation where available; do not merge these concepts into one register.

## Extraction settings

At the inspected revision, equity, macro, and Sharadar history windows are each 31 years, while the manifest forces a full EDGAR relist every 30 days. The source series registry itself is not configurable: symbol-to-series mappings are world facts and remain in constants.

Key distinctions:

- `years_history` scopes ordinary equity/filing walks;
- `macro_years_history` owns the long macro window;
- `sharadar_years_history` is separate because entitlement and response size differ;
- `refresh_universe` controls replacement of the current roster;
- redundant class tickers prevent double-counting after the retained class is active;
- `earnings_calls` in [data.yml](../../configs/data.yml) tunes the defeatbeta transcript extractor: `lookback_days: 45` (an incremental run re-checks calls dated within this many days of the stored frontier), `read_workers: 4` (parallel HuggingFace row-group reads; writes stay on one thread) and `reconcile_days: 7` (a full comparison of every scoped call at least this often). The dataset repo and file path are module constants of the extractor;
- LLM model, concurrency, prompt cache, and action-specific character budgets are owned by [gpt.yml](../../configs/gpt.yml). `llm_model.open_ai` remains the GPT-6 Sol default; employee extraction selects `llm_model.open_ai_cheap` (GPT-6 Luna) with `reasoning_effort.employees: none`. `gpt.threads` (12) sizes the LLM worker pool for DEF 14A and Item 5.07 vote extraction; employee extraction pins one thread per ticker in code;
- regulatory dates are code constants, not knobs: the DEF 14A ECD listing floor 2022-12-16 (Item 402(v) effective date) is `_PVP_EFFECTIVE` in [fetch_def14a_edgar.py](../../src/data_extract/utils/structure/fetch_def14a_edgar.py).

## Data availability and freshness

[configs/data.yml](../../configs/data.yml) declares outer availability boundaries, not filled values. A table-level `__all__` date applies unless a field override exists; derived-feature boundaries capture extra publication lags or minimum history.

Runtime eligibility is the intersection of:

1. the configured source boundary;
2. actual source publication/coverage;
3. ticker or security eligibility;
4. required price, denominator, or related-source cells; and
5. family-specific completeness rules.

`source_freshness.insider_bulk_authoritative_through` is a reviewed cutover, not a date that advances automatically. Promote it only after the retained bulk/live parity validation passes.

## Cube settings

The current target contract uses horizons 30, 60, and 90 sessions, with 60 as the primary horizon. Labels are persisted wide as `target_<label>_h<horizon>`; changing either list changes the targets table's column set and forces a full target rebuild plus cube assembly.

Important blocks:

| Block | Contract |
| --- | --- |
| `betas` | Rolling window, minimum observations, ridge ratios, market prior, step, and forward-fill limit. Market beta shrinks toward one; other loadings shrink toward zero. |
| `targets` | Horizon/label set, minimum cross-section, volatility standardization, and joint neutralization against fitted loadings, momentum, size, and industry-group dummies. |
| `features` | Cross-sectional transform policy. |
| `intrinsic` | Two-stage DCF parameters; terminal growth must remain below discount rate. |
| `hist` | Five-year-style self-history comparison window and minimum observations. |
| `institutionals` | Decay, manager selection/staleness, coverage-break policy, conditioning windows, and source-family behavior. |
| `output` | Persistence switches for cube, signals, CV results, predictions, diagnostics, and artifacts. |

Every numerical value in [build_cube.yml](../../configs/build_cube.yml) carries an economic or measured justification. Read the adjacent comment before editing. If a feature introduces a longer daily look-back, update the [part registry](../../src/data_aggregate/utils/common/parts.py) as well.

## Model settings

[modellling.yml](../../configs/modellling.yml) currently selects rank targets and an ensemble of ElasticNet, LightGBM, and LightGBM random-forest mode. Time-series CV uses an embargo that defaults to the primary horizon when null. Weight decay is disabled because the measured fold-stability trade-off favored uniform history.

Each family owns its feature list. LightGBM additionally owns categorical features and monotonic constraints. The earnings-call contract in `configs/models/lgbm_modelling.yml` is exactly 12 raw/issuer-history columns; `_xs` and `_vs_peers` earnings-call variants are intentionally excluded. Directional constraints apply only where the economic sign is unambiguous: positive call tone and negative uncertainty. Q&A gap (management-answer tone minus prepared-remarks tone), coherence, embedding distance, and disclosure-length change are unconstrained. `configs/validate.yml` owns their quarterly cadence, raw plausibility bounds, 66-session lifetime, Spearman redundancy thresholds, PSI and missingness-delta limits, latest-20-session audit window, and the split-quality floors (`split_ok_rate_min: 0.985`, `prepared_share_q05_min: 0.15`). Train-versus-recent checks use the canonical `train.end_date` from `configs/modellling.yml`; the validation config does not duplicate that boundary. The dedicated validator reads these settings rather than duplicating them in Python. When a new feature has an economically unambiguous direction, add the constraint alongside the column and run the monotonic-contract tests.

Training boundaries describe the holdout evaluation run. Production `full-train` intentionally ignores the configured end date and fits through the latest eligible cube row.

## Strategies and portfolio

The portfolio selects `ls_equity` and `eq_long_only`, the only registered sleeves. The portfolio owns capital, global target volatility, costs, risk-free rate, covariance/blending policy, rebalance frequency, leverage, and output switches.

Sleeve YAML owns only sleeve construction. [PortfolioInputs](../../src/strategies/base.py) passes portfolio-level values down. A sleeve can override trading costs where explicitly supported, but must not duplicate capital or global risk settings.

The long/short sleeve owns neutrality, position and gross caps, turnover controls, covariance shrinkage, horizon blending, and integer-share behavior. Whole-share optimization requires realistic capital; otherwise positions can collapse below one share.

## Validation configuration

[validate.yml](../../configs/validate.yml) maps feature prefixes to publication clocks. Its default is strict and per ticker. Observed-zero declarations are narrow exceptions for builders that intentionally encode a known absence as zero; they alter the validator's clock, not stored data.

A multi-source feature becomes eligible only when all required source clocks have begun. Configuration cannot convert a missing dependency into an observation.

## Adding or changing a knob

1. Identify the stage that owns the behavior.
2. Add the key to that stage's existing YAML and explain why the default is appropriate.
3. Read it through the stage's config block; do not add a second default in Python.
4. Update tests for schema/contract changes.
5. If the value changes columns, look-backs, availability, or portfolio artifacts, update the relevant [cube](../modules/data-aggregate.md), [source](./data-sources.md), or [model](./modelling-and-portfolio.md) reference.
6. Request approval before changing `configs/`, which is a repository risk zone.

## Constants versus configuration

Use configuration for research choices and operational tuning. Use [constants.py](../../src/constants/constants.py) for URLs, date formats, form vocabularies, source series, taxonomy facts, model identifiers, GICS names, plausibility bounds, and other facts that should not vary between runs. Use the [table registry](./table-catalog.md) for table names and grain.

## Related

- [Constants and configuration module](../modules/constants-and-configuration.md)
- [Modelling and portfolio contracts](./modelling-and-portfolio.md)
- [Add a model or sleeve](../guides/add-a-model-or-sleeve.md)
- [Run the pipeline](../guides/run-the-pipeline.md)
