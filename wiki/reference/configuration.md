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
| `seed`, `data_extract` | [configs.yml](../../configs/configs.yml) | Global random seed, history windows, source refresh switches, redundant tickers, the SEC retry policy, document retry rounds, the per-run document cap, and the prediction fresh-share threshold. |
| `local.paths` | [paths.yml](../../configs/paths.yml) | Repository root, artifact store, output, model, log, and peer-dictionary locations, and the source caches, including the local EDGAR filing index (`sec_edgar_index`) and the SEC 13F data sets (`sec_13f_datasets`). |
| `logging` | [logging.yml](../../configs/logging.yml) | Standard-library logging tree and the in-memory log format. |
| `peers` | [peers.yml](../../configs/peers.yml) | Business-similarity and return-correlation peer construction. |
| `build_cube` | [build_cube.yml](../../configs/build_cube.yml) | Betas, targets, feature transforms, intrinsic value, historical comparisons, institutional policies, and output switches. |
| `model`, `train` | [modellling.yml](../../configs/modellling.yml) | Ensemble composition, target choice, CV, diagnostics, decay, `models_dir` (artifact folder under the data store), `backtest.n_quantiles` (label-only backtest buckets), and train/holdout boundaries. Each family YAML under `configs/models/` (e.g. [lgbm_modelling.yml](../../configs/models/lgbm_modelling.yml)) also sets `task: regression \| classification`. The filename intentionally contains three “l” characters. |
| `linear`, `lgbm`, `random_forest` | [configs/models](../../configs/models/) | Family hyperparameters and family-specific feature columns. |
| `strategy_ls`, `strategy_eq_long_only` | [configs/strategy](../../configs/strategy/) | Sleeve construction only. |
| `portfolio` | [portfolio.yml](../../configs/portfolio.yml) | Sleeve selection, dates, global costs, risk targeting, leverage, capital, blend, and analysis output. |
| `data_availability`, `source_freshness`, `earnings_calls` | [data.yml](../../configs/data.yml) | Institutional availability boundaries, field/derived overrides, the insider frontier lag, and the earnings-call extractor knobs. |
| validation keys | [validate.yml](../../configs/validate.yml) | Point-in-time publication clocks, observed-zero exceptions, and validation policy. |

Curated evidence registers under [configs/sec](../../configs/sec/) are versioned data contracts rather than tuning knobs:

- [registrant_cutover.json](../../configs/sec/registrant_cutover.json): the register, dated legal-filer chains. One key per ticker with `kind` (`reorganisation`, `domestication`), optional `basis` (e.g. `reverse_merger_accounting_predecessor` for MRK, PLD) and `segments` (`cik`, `valid_to`, `evidence` citing the accession); the open last segment is the roster CIK. The highest-priority membership oracle; an entry always overrides an automatic window.
- [entity_lineage_manual.json](../../configs/sec/entity_lineage_manual.json): CIK-to-economic-entity adjudications, one key per `<ticker>/<slug>` with `same_entity` or `own_entity` CIKs, `evidence` and `decided`; `_d19_allowlist` holds the roster-CIK cross-check's exceptions.
- [symbol_tenure_manual.json](../../configs/sec/symbol_tenure_manual.json): `tickers` maps a ticker to evidenced half-open symbol intervals (`symbol`, `issuer_cik`, `valid_from`, `valid_to`, `evidence`); `rejected` lists derived Form 3/4/5 tenures that are false (mis-tagged filings: F/0000712537, AXP/0001387156, and one- or two-day tenures on F, NTRS, PM, AME, SPG), each with its accessions and evidence, dropped at build.
- [security_master_manual.json](../../configs/sec/security_master_manual.json): curated security-master rows, each citing a URL or accession in `source` (refused at load otherwise). `conversion_ratios` (BRK-A 30, then 1,500); `market_boundaries` (CUSIP-qualified role and issuer over half-open trade dates: APTV/DLPH, NWSA, XOM, FOXA, GOOGL, EXE, WBD); `class_overrides`; `merger_metadata` (insider boundary-day facts, e.g. MRK 2009-11-03 after the close, acquired symbol SGP); `co_registrants` (subsidiary CIKs whose rows are not stored); `exchange_ratios` (one per register predecessor window whose vendor series replaces the canonical one: PLD, DD, LIN, EVRG, BKR, STE); `deferred_to_traded_security` (below).
- [vendor_coverage_exceptions.json](../../configs/sec/vendor_coverage_exceptions.json): `exceptions`, one accession-exact row per Sharadar quarter missing at a CIK cutover while the SEC filing exists (`ticker`, `quarter`, `period_end`, `cik`, `accession`, `filed`, `label`). An unrecorded missing quarter fails the continuity check, and so does a row whose quarter is present again (healed): delete it.
- [expected_lineage_changes.json](../../configs/sec/expected_lineage_changes.json): `hypotheses` the identity regression gate checks against the rows it computes at a cutover.

Runtime readers consume their validated/materialized representation where available; do not merge these concepts into one register. Every identity edit needs SEC evidence with an accession, then `identity-tables` and `identity-propagate` (`--every-ticker` when the edit moves no lineage stamp; see the [run guide](../guides/run-the-pipeline.md#registrant-boundaries)).

Identity rules that live in code, not YAML:

- **Automatic CIK windows** are on (`AUTO_WINDOWS_ENABLED = True` in [entity_lineage.py](../../src/data_extract/utils/common/entity_lineage.py)): a window is created only when cover pages (`dei`) and Forms 3/4/5 both place the CIK switch within 31 days (`SWITCH_TOLERANCE_DAYS`). Each is listed as an `automatic_cik_window` item by `validate identity`; a register entry wins.
- **Predecessor vendor series.** `fundamentals-sharadar` also fetches the `sharadar_tickers` rows whose SEC CIK is a register predecessor CIK, stored under their vendor ticker: BKR ← BHI, STE ← STE1, LIN ← PX1, EVRG ← WR1 (`predecessor_vendor_tickers` in [fetch_sharadar.py](../../src/data_extract/utils/fundamentals_sharadar/fetch_sharadar.py)). The merged history replaces the canonical vendor rows inside the predecessor's register window with that series, converted by the cited `exchange_ratios`. PLD1 and DD1 are excluded by their deferral.
- **`deferred_to_traded_security`** (in `security_master_manual.json`, read by [traded_security_deferrals.py](../../src/utils/traded_security_deferrals.py)): tickers whose history waits for the traded-security realignment, each with `defers` and a `source` citing the decision. `predecessor_series`: no vendor replacement and no exchange-ratio conversion inside the predecessor window (PLD, DD); `event_filing_windows`: 8-K and 13D/13G stay undated over every CIK of the entity (PLD, JCI, DD). A malformed entry raises; a change needs `identity-propagate --every-ticker`.

The Dataroma superinvestor roster keeps its two registers under `configs/superinvestors/`, read from `<config_dir>/superinvestors/` by [superinvestor_roster.py](../../src/utils/superinvestor_roster.py):

- [dataroma_roster_history.json](../../configs/superinvestors/dataroma_roster_history.json): one Wayback capture of the Dataroma home page per calendar quarter from 2012Q1, each with `captured_at`, `source_url` and `managers` (code to name). It is generated, never hand-edited: [build_dataroma_roster_history.py](../../scripts/build_dataroma_roster_history.py) regenerates it (see [data sources](./data-sources.md#superinvestor-roster-dataroma)).
- [overrides.json](../../configs/superinvestors/overrides.json): hand resolutions, each with its 13F evidence. `cik_overrides` maps a code to its filer CIK; `manager_ciks` chains a manager's successive filer CIKs in dated period windows under a manager ID (the oldest CIK); `unresolvable` lists codes allowed a NULL CIK; `inactive` lists dated ranges in which a resolved filer legitimately has no 13F. `load_superinvestor_overrides` validates the chains (no overlapping windows, manager ID = oldest member, a CIK in one chain only) and caches per resolved directory. Rebuild the table after any change (see [run the pipeline](../guides/run-the-pipeline.md#superinvestor-roster)).

## Extraction settings

Equity, macro, and Sharadar history windows are each 31 years. The source series registry itself is not configurable: symbol-to-series mappings are world facts and remain in constants. A table's resume overlap, EDGAR forms and source start are part of its `Resume` contract in [schema.py](../../src/data_store/schema.py), not YAML knobs.

Key distinctions:

- `years_history` scopes ordinary equity/filing walks;
- `macro_years_history` owns the long macro window;
- `sharadar_years_history` is separate because entitlement and response size differ;
- `refresh_universe` controls replacement of the current roster;
- redundant class tickers prevent double-counting after the retained class is active;
- `earnings_calls` in [data.yml](../../configs/data.yml) tunes the defeatbeta transcript extractor: `read_workers: 4` (parallel HuggingFace row-group reads; writes stay on one thread). The dataset repo and file path are module constants of the extractor;
- `sec_retry` (3 attempts, waits of 5 s then 20 s, Retry-After capped at 120 s) is the one retry policy of every SEC request, applied by [sec_io.py](../../src/data_extract/utils/common/sec_io.py);
- `retry_rounds` (3 rounds, waits of 60, 120 and 240 s, threshold 0.985, `min_failed` 2): a ticker whose EDGAR read rate in a run is below the threshold, with at least `min_failed` failed documents, re-reads only those documents in-task; the rest is listed again on the next run;
- `max_documents_per_run` (5,000) caps the documents one EDGAR fetch reads per run, newest first; a larger work list logs an ERROR and drains over later runs, and `--no-cap` lifts the cap for a one-time backlog;
- `prediction_fresh_share` (0.95): `modelling predict` refuses to score, and `extraction-status` marks the table RED, when the share of universe tickers whose own latest `prices`, `fundamentals_sharadar` or `fundamentals_history` date is within the table's cadence tolerance falls below it;
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

`source_freshness.insider_max_lag_days` is the only insider freshness knob: `cube-status` names the insider source behind when its completeness frontier (the last price session while the newest `insider_transactions` filing, markers included, lies within the table's 7-day overlap of it, else that filing date) is unknown or lags the institutionals part by more than this many days. There is no reviewed cutover date and no insider parity threshold; the zip and EDGAR sources share one table and EDGAR always wins.

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
| `workforce` | `part_time_weight`: α of the workforce headcount proxy FT + α·PT (0.65; must lie in [0.5, 1]). See [workforce features](./modelling-and-portfolio.md#workforce-features). |
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
