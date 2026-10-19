---
title: Configuration
description: OmegaConf assembly, ownership of every configuration domain, and change rules.
type: reference
tags:
  - wiki
  - reference
  - configuration
source_commit: 8e4bd26a0e837b01f0df16e29703dd8918458b77
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
| `linear`, `lgbm`, `random_forest` | [lgbm_modelling.yml](../../configs/models/lgbm_modelling.yml) and sibling family files | Family hyperparameters and family-specific feature columns. |
| `strategy_ls`, `strategy_eq_long_only` | `configs/strategy/` | Sleeve construction only. |
| `portfolio` | [portfolio.yml](../../configs/portfolio.yml) | Sleeve selection, dates, global costs, risk targeting, leverage, capital, blend, and analysis output. |
| `data_availability`, `source_freshness`, `earnings_calls` | [data.yml](../../configs/data.yml) | Institutional availability boundaries, field/derived overrides, the insider frontier lag, and the earnings-call extractor knobs. |
| validation keys | [validate.yml](../../configs/validate.yml) | Point-in-time publication clocks, observed-zero exceptions, and validation policy. |

`local.paths.peers_dir` in [paths.yml](../../configs/paths.yml) defaults to `output/peers` beneath `DATA_STORE`; an absolute path remains absolute. [Context](../../src/context.py) keeps `SECTOR_PEERS_PATH` as the complete active `sector_peers.json` filename. The active cache and named `.BEFORE.json.bak` backup relocate unchanged; peer construction and PostgreSQL description/embedding caches are unaffected. Activate path changes only with verified cache bytes at the destination; see [peer artifact paths](../modules/data-peers.md#artifact-paths).

[yf_price_bugfix.json](../../configs/prices/yf_price_bugfix.json) is the approved register for both price defects and vendor share-basis exceptions. `share_events` declares exact source expectations, sourced original-share/quote anchors, actual ordinary-share events, independently incorporated vendor restatements, optional represented price factors and explicit quoted-security units. `level_factor` retains hidden price-only wedges; `split_vintage`, `return_seams`, `null_ret` and `volume_scale` retain their repair roles. `seam_reviewed` (`ticker` to `[{date, why}]`) declares a reviewed genuine move to the validator and changes no data. See the [price-basis contract](data-sources.md#price-basis-contract).

For future supported missing/moved events or differing vendor treatment, edit this existing price register. An event's `actual` block has `date`/`ratio`; omitted `vendor` defaults to actual, while vendor ratio1 explicitly means no retrospective restatement. Omitted `price` preserves Yahoo's represented date/factor; a price override requires an independent `quoted_price` anchor in the affected interval. Source selectors describe observed rows, not replacement dates. `quote_units` carries the conversion date, before/after ordinary shares per quoted security and explicit title aliases. Generic common-stock labels do not prove ADS units.

The shared resolver in [price_adjustments.py](../../src/utils/price_adjustments.py) validates metadata and source/anchor evidence before a write. The scoped loader in [price_basis.py](../../src/utils/price_basis.py) verifies old anchors even during tail builds, using the same pure quote repairs as the price builder. Changed or unavailable evidence refuses an applicable new policy; verified empty sources differ from unavailable sources. A price anchor overlapping a nonunit hidden-level segment is explicitly refused until that combined basis can be independently verified.

A `return_seams` entry with `fields: ["close_total"]` must supply `expect_step`, `target_step`, `expect_quote_step` and evidence. Only the strict pre-event total-return prefix changes; a precise verified target is idempotent. Changed or incomplete applicable input refuses the build. Legacy entries without `fields` retain the all-price, target-one behavior. Supported total-only repairs recover total-return windows while residual quote seams keep OHLC windows masked. Historical JSON edits require the affected historical derived tables and consumers to be rebuilt; a default tail build cannot adopt an old prefix correction.

Curated evidence registers under `configs/sec/` are versioned data contracts rather than tuning knobs:

- [registrant_cutover.json](../../configs/sec/registrant_cutover.json): the register, dated legal-filer chains. One key per ticker with `kind` (`reorganisation`, `domestication`), optional `basis` (e.g. `reverse_merger_accounting_predecessor` for MRK) and `segments` (`cik`, `valid_to`, `evidence` citing the accession); the open last segment is the roster CIK. The highest-priority membership oracle; an entry always overrides an automatic window. The register follows the **traded security**: a ticker's pre-merger history is the security whose prices it carries. DD is Dow Chemical (0000029915) before 2017-08-31; PLD (AMB before 2011-06-03) and JCI (Tyco before 2016-09-02) have no entry, because the legal survivor's own CIK carries those prices; DOW has no entry and starts at 2019-04-01. A `fresh_start` entry (one segment with `valid_from`, no `valid_to`) dates a ticker's only CIK: its filings and tape count from that day, earlier ones belong to a cancelled predecessor security. EXE (Chesapeake, CIK 0000895126) starts at its 2021-02-10 emergence; the fundamentals history then keeps the comparatives of its post-emergence filings, like Sharadar.
- [entity_lineage_manual.json](../../configs/sec/entity_lineage_manual.json): CIK-to-economic-entity adjudications, one key per `<ticker>/<slug>` with `same_entity` or `own_entity` CIKs, `evidence` and `decided`. A `same_entity` entry also declares a merger's acquired target as an event-only CIK of the survivor (old ProLogis 0000899881 under PLD, old Johnson Controls 0000053669 under JCI, old DuPont 0000030554 under DD): its Forms 3/4/5 and FTD lines are kept as `acquired_constituent`, and it contributes no consolidating or event-filing row. `_d19_allowlist` holds the roster-CIK cross-check's exceptions.
- [symbol_tenure_manual.json](../../configs/sec/symbol_tenure_manual.json): `tickers` maps a ticker to evidenced half-open symbol intervals (`symbol`, `issuer_cik`, `valid_from`, `valid_to`, `evidence`); `rejected` lists derived Form 3/4/5 tenures that are false (mis-tagged filings: F/0000712537, AXP/0001387156, and one- or two-day tenures on F, NTRS, PM, AME, SPG), each with its accessions and evidence, dropped at build.
- [security_master_manual.json](../../configs/sec/security_master_manual.json): curated security-master rows, each citing a URL or accession in `source` (refused at load otherwise). `conversion_ratios` (BRK-A 30, then 1,500); `market_boundaries` (CUSIP-qualified role and issuer over half-open trade dates: APTV/DLPH, NWSA, XOM, FOXA, GOOGL, EXE, WBD); `class_overrides`; `merger_metadata` (insider boundary-day facts, e.g. MRK 2009-11-03 after the close, acquired symbol SGP); `co_registrants` (subsidiary CIKs whose rows are not stored); `exchange_ratios` (one per predecessor window whose vendor series replaces the canonical one: LIN, EVRG, BKR, STE, JCI); `vendor_series_overrides` (`ticker`, `vendor_ticker`, `cik`, `valid_from`, `valid_to`: a ticker whose own vendor series is the wrong company before a seam takes the named vendor ticker's series instead, JCI ← TYC before 2016-09-02); `reverse_acquisitions` (`ticker`, `seam_date`, `accounting_acquirer_cik`, `legal_acquirer_cik`: PLD 2011-06-03, JCI 2016-09-02; the SEC history drops a fact whose period ends before the seam when its filing's `period_of_report` is on or after it, so pre-seam periods keep the traded security's as-filed values).
- [vendor_coverage_exceptions.json](../../configs/sec/vendor_coverage_exceptions.json): `exceptions`, one accession-exact row per Sharadar quarter missing at a CIK cutover while the SEC filing exists (`ticker`, `quarter`, `period_end`, `cik`, `accession`, `filed`, `label`). An unrecorded missing quarter fails the continuity check, and so does a row whose quarter is present again (healed): delete it.
- [expected_lineage_changes.json](../../configs/sec/expected_lineage_changes.json): `hypotheses` the identity regression gate checks against the rows it computes at a cutover.

Runtime readers consume their validated/materialized representation where available; do not merge these concepts into one register. Every identity edit needs SEC evidence with an accession, then `identity-tables` and `identity-propagate` (`--full` when the edit moves no lineage stamp; see the [run guide](../guides/run-the-pipeline.md#registrant-boundaries)).

Identity rules that live in code, not YAML:

- **Automatic CIK windows** are on (`AUTO_WINDOWS_ENABLED = True` in [entity_lineage.py](../../src/data_extract/utils/common/entity_lineage.py)): a window is created only when cover pages (`dei`) and Forms 3/4/5 both place the CIK switch within 31 days (`SWITCH_TOLERANCE_DAYS`). Each is listed as an `automatic_cik_window` item by `validate identity`; a register entry wins.
- **Predecessor vendor series.** `fundamentals-sharadar` also fetches the `sharadar_tickers` rows whose SEC CIK is a register predecessor CIK, stored under their vendor ticker: BKR ← BHI, STE ← STE1, LIN ← PX1, EVRG ← WR1, plus the declared `vendor_series_overrides` (JCI ← TYC) (`load_predecessor_series` in [predecessor_series.py](../../src/utils/predecessor_series.py)). The merged history replaces the canonical vendor rows inside the predecessor window with that series, converted by the cited `exchange_ratios`. PLD and DD keep Sharadar's own rows, which are already AMB's and Dow Chemical's.
- **Share basis inside a predecessor window.** Shares carry only the genuine split events `split_events` accepts; price-only factors Yahoo books as splits (JCI's 2012 ADT spin and 2016 0.955 consolidation) live in the cube's level factor S(d), so JCI's exchange ratio is 1. Inside a predecessor window, S(d) reads the window owner's `sharadar_actions` relabelled to the ticker (`level_actions` in [level_basis.py](../../src/data_aggregate/utils/common/level_basis.py)), the same swap the merge makes.


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

## Cube assembly selection

[build_cube.yml](../../configs/build_cube.yml) supplies `build_cube.assembly` to the existing [assembler](../../src/data_aggregate/transformers/step_assemble_cube.py) and shared [policy](../../src/data_aggregate/utils/common/assembly.py). All registered parts participate. Prices, targets, betas, fundamentals, momentum and governance are compulsory; the committed optional list contains text and institutionals. Optional skips apply only to absent/physically empty tables; a populated broken input fails.

```yaml
build_cube:
  assembly:
    optional_parts:
      - cube_part_text
      - cube_part_institutionals
    columns:
      cube_part_fundamentals: null
      cube_part_governance: null
      cube_part_institutionals: null
```

A missing `columns` mapping, omitted part entry, null or empty list includes all actual columns. Nonempty lists select exact payload names, one YAML bullet per line; date and ticker are included automatically. For example, replace a null value with:

```yaml
      cube_part_fundamentals:
        - f_earnings_yield
        - f_ocf_yield
```

Unknown parts, malformed or duplicate lists, missing selected columns and selected-name collisions fail before changing output. Selection controls both the store read and the physical output schema; reassembly removes old deselected columns. It does not change producer tasks. Later removal of model inputs requires compatible model configuration/artifacts and retraining; the existing prediction input guard remains active.

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
| `fundamentals_features` | Fixed eight earnings-lag slots, eight-quarter persistence and three-session report response; prior SUE scale uses eight quarters. Changing fixed schema windows requires a feature-contract update. |
| `workforce` | `part_time_weight`: α of the workforce headcount proxy FT + α·PT (0.65; must lie in [0.5, 1]). See [workforce features](./modelling-and-portfolio.md#workforce-features). |
| `output` | Persistence switches for cube, signals, CV results, predictions, diagnostics, and artifacts. |

Every numerical value in [build_cube.yml](../../configs/build_cube.yml) carries an economic or measured justification. Read the adjacent comment before editing. If a feature introduces a longer daily look-back, update the [part registry](../../src/data_aggregate/utils/common/parts.py) as well.

## Model settings

[modellling.yml](../../configs/modellling.yml) selects zscore targets, h30/h60/h90 and ElasticNet, LightGBM and LightGBM random-forest mode. Five expanding crisis-aware windows cover 2000–02,2003–06,2007–12,2013–18 and2019–21; null embargo resolves to the current label horizon. Existing weight decay, feature pools, hyperparameters and monotonic directions remain fixed.

Each family owns its feature list. LightGBM additionally owns categorical features and monotonic constraints. The earnings-call contract in `configs/models/lgbm_modelling.yml` is exactly 12 raw/issuer-history columns; `_xs` and `_vs_peers` earnings-call variants are intentionally excluded. Directional constraints apply only where the economic sign is unambiguous: positive call tone and negative uncertainty. Q&A gap (management-answer tone minus prepared-remarks tone), coherence, embedding distance, and disclosure-length change are unconstrained. `configs/validate.yml` owns their quarterly cadence, raw plausibility bounds, 66-session lifetime, Spearman redundancy thresholds, PSI and missingness-delta limits, latest-20-session audit window, and the split-quality floors (`split_ok_rate_min: 0.985`, `prepared_share_q05_min: 0.15`). Train-versus-recent checks use the canonical `train.end_date` from `configs/modellling.yml`; the validation config does not duplicate that boundary. The dedicated validator reads these settings rather than duplicating them in Python. When a new feature has an economically unambiguous direction, add the constraint alongside the column and run the monotonic-contract tests.

train.end_date remains a feature ceiling, while train.outcome_cutoff=2021-12-31 limits every development outcome. Actual observed price-calendar label ends purge inner training before validation and outer training before test. Fixed-cutoff assessment starts 3 January 2022; portfolio start remains 1 June 2022. Final full-train reuses that run's saved fold-selected counts without CV and fits all finite labels mature by the latest complete cube/price session. See [the lifecycle](./modelling-and-portfolio.md#longshort-training-lifecycle).

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

Use configuration for research choices and operational tuning. Use [constants.py](../../src/constants/constants.py) for URLs, date formats, source series, taxonomy facts, model identifiers, GICS names, plausibility bounds, and other facts that should not vary between runs. Use the [table registry](./table-catalog.md) for table names, grain, and the extracted SEC forms in each table's `Resume.forms`.

## Related

- [Constants and configuration module](../modules/constants-and-configuration.md)
- [Modelling and portfolio contracts](./modelling-and-portfolio.md)
- [Add a model or sleeve](../guides/add-a-model-or-sleeve.md)
- [Run the pipeline](../guides/run-the-pipeline.md)

## Institutional value-unit candidate — 10 October 2026

The tested P1 candidate in .worktrees/institutional-13f-pit adds build_cube.institutionals.value_basis: min_securities=15, min_agreement=0.8, price_tolerance=0.1 and max_price_age_days=7. These knobs govern a common filing-wide monetary factor against reconstructed report-quarter-end as-traded prices, independently per quarter/filing. They do not represent the separate P2 minimum of 15 managers holding the same security. Shares are unchanged; explicit shares==value reconstruction and source-only/abstained statuses are distinguishable.

The candidate is not activated. The later user-approved fallback estimates uncertain common dollars as valid reported shares × the eligible restored quarter-end price, after the raw unit decision; estimated rows cannot vote for a factor. Source zeros remain zero. The subsequent typed-option extension uses call_shares/put_shares × the same eligible reference for uncertain underlying exposure; options never vote for the common filing factor. Nonzero unresolved debt values still require a debt-instrument quote and remain unknown. Common-dollar completeness is now 457/461, 472/476 and 436/482 ticker totals in the three tested first-publication cohorts; option/value coverage and immutable filing versions remain pending. Local candidate evidence: `reports/validate/2026-10-09-institutional-13f-audit/06-final.md`.

## Saved-model assessment controls

[modellling.yml](../../configs/modellling.yml) owns model.assessment.output_dir (output/modelling_backtest), optional configured run_id, evaluation_start (3January2022) and read_chunksize (I/O batching only). model.horizons selects30/60/90; model.cv owns the five crisis-aware windows; train.outcome_cutoff limits all development outcomes to31December2021. The CLI run ID overrides a configured value; every downstream task requires the same named run, without a latest fallback.

model.diagnostics owns shap_sample=12000 per development population/horizon (allocated across five original folds) and another12000 cutoff rows, dependence_sample=1000 displayed points, pdp_grid=30, pdp_background=2000 and rolling_window=252. Deterministic sampling affects explanations only, not fitting rows or feature pools. Tree importance uses sampled absolute SHAP; linear uses saved absolute standardized coefficients. RF retains its inherited LightGBM monotonic directions; a constrained response can reflect an imposed prior.

Family columns_by_horizon selects the complete input list, including any categorical variables. A missing override uses columns; an explicit empty list selects no inputs for that horizon. The shared lgbm.categoricals registry assigns types only to selected inputs, and the shared lgbm.monotonic map applies to each tree family's fitted inputs at every horizon. RF keeps its own feature lists; categoricals_by_horizon is retired. No operational feature-pruning/search/YAML-edit parameters remain. Assessment and its report-only mode never fit; explicit resume verifies model, input, source, calendar and settings identities. Production model.models_dir stays separate from dated cutoff artifacts. See [artifact paths and report controls](../modules/modelling.md).
