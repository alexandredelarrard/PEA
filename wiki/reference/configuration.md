---
title: Configuration
description: OmegaConf assembly, ownership of every configuration domain, and change rules.
type: reference
tags:
  - wiki
  - reference
  - configuration
source_commit: ea60c7d9612832515767f9b71338717e887b24b7
integration_parents:
  - 1195d18d6dcd3e380cacc0e8cb4a6cd5f5392090
  - ad346fa6d7c31c57046942e966727edb8414916f
---
# Configuration

## Purpose

This is the canonical map from a pipeline concern to its owning OmegaConf file. Tunable numbers belong in YAML. [data.yml](../../configs/data.yml) owns table timing; fixed domain values stay with their repair/calculation module, shared contracts belong in [constants.py](../../src/constants/constants.py), and table identity belongs in [schema.py](../../src/data_store/schema.py).

## Assembly

[read_config](../../src/utils/config.py) requires `configs/config.yml`, loads it first, then merges the remaining `configs/**/*.yml` files in sorted relative-path order. Every file must be a mapping; each top-level namespace has exactly one owner. Duplicate namespaces are rejected even when their values match. [get_config_context](../../src/context.py) then configures logging, seeds random generators, resolves paths and environment values, and constructs the shared context.

Callers use attribute access such as `self._config.data_aggregate.targets.horizons`. Do not replace OmegaConf or introduce hidden Python defaults for a required knob.

## Ownership map

| Top-level key | Owner | Responsibility |
| --- | --- | --- |
| `seed`, `local.paths` | [config.yml](../../configs/config.yml) | Random seed, existing path keys and source-cache locations. |
| `data_extract` | [data_extract.yml](../../configs/data_extract.yml) | Extraction controls, concurrency, retries and text budgets. |
| `universe` | [universe.yml](../../configs/universe.yml) | The traded universe and the only hand-edited ticker list: `tickers` (491 quoted yfinance symbols, each mapped to its GICS `sector` and sub-industry `industry`; the industry group is derived from `vocabulary.py`) and `aliases` (other share-class symbols of a universe ticker: GOOGL→GOOG, FOXA→FOX, NWSA→NWS, BRK-B→BRK-A). Validated on load (non-empty, no duplicates, every alias key in `tickers`; at sync, every sub-industry known and inside its sector); `data_extract universe` syncs `universe_tickers` from it. |
| `logging` | [logging.yml](../../configs/logging.yml) | Standard-library logging tree and the in-memory log format. |
| `peers` | [data_peers.yml](../../configs/data_peers.yml) | Business-similarity and return-correlation peer construction. |
| `data_aggregate` | [data_aggregate.yml](../../configs/data_aggregate.yml) | Betas, targets, feature transforms, intrinsic value, historical comparisons, institutional policies, and output switches. Two 13F blocks serve the manager profile and, through its book cleaning, the cluster cube part: `bond_lines` (regexes that find share lines which are bond principal: `cusip_pattern`, `cusip_excluded_prefixes`, `name_exclusions`, `title_markers`, `title_exclusions`; the shared book cleaning drops them from the share lines like `debt` rows and the profile weighs them by filed value, `manager_profile.bond_lines` is an interpolation of it) and `manager_profile` (`window_quarters`, `min_quarters`, `amendment_only_min_lines` = 4, `turnover_threshold`, `return_price_band`, `split_share_move`, `split_value_tolerance`, `cap_breakpoints`, `price_check_min_filers`, `security_markers`). |
| `manager_categories` | [manager_categories.yml](../../configs/manager_categories.yml) | Manager category rules for `sec_13f_manager_details.category`: name cleaning (`clean.drop_tokens`, `clean.trailing_tokens`) and ordered rules (`name`, `category`, one-line `description`, `patterns`, optional `unless`); the first matching rule wins, no match gets `default_category` (`fund`; `categories` lists the legal values, `unknown` is retired). A CIK listed in [manager_categories.json](../../configs/superinvestors/manager_categories.json) (`overrides`: ten-digit CIK to `{category, name}`, validated on load) takes that category over every rule. The profile tunables sit in `data_aggregate.manager_profile` (`window_quarters` 4, `min_quarters` 2, `turnover_threshold` 0.10 share-count change, `return_price_band` [0.10, 10.0] with `split_share_move` 0.20 and `split_value_tolerance` 0.25 (the `book_return` price-outlier and split rules), `cap_breakpoints` [0.70, 0.20] of cumulative 13F value for large and mid, `price_check_min_filers` 3, the distinct-filer count behind a consensus price; there is no price-check ratio, and `security_markers`: the word lists of the title and issuer name rules for common lines Sharadar does not class as a domestic, ADR or Canadian stock (unlisted, or listed under another category), `etf`, `real_estate`, `foreign` (regex markers on the upper-case `title_of_class | issuer_name`, whole words) and `us_titles` (`anchors` and `fillers` of a generic common-stock title)). |
| `cluster` | [cluster.yml](../../configs/cluster.yml) | Manager clustering and the per-cluster 13F cube features (see below). |
| `model`, `train` | [modelling.yml](../../configs/modelling.yml) | Ensemble composition, target choice, CV, diagnostics, decay, `models_dir` (artifact folder under the data store), fresh assessment controls and train/test boundaries. Each family YAML under `configs/models/` (e.g. [lgbm_modelling.yml](../../configs/models/lgbm_modelling.yml)) also sets `task: regression \| classification`. |
| `linear`, `lgbm`, `random_forest` | [lgbm_modelling.yml](../../configs/models/lgbm_modelling.yml) and sibling family files | Family hyperparameters and family-specific feature columns. |
| `portfolio` | [portfolio.yml](../../configs/portfolio.yml) | Fixed LS/S&P/cash budgets, optional end date (cutoff and start come from train.end_date), allocation cadence, momentum/risk windows, cash remuneration, proportional costs and read batch size. |
| `data` | [data.yml](../../configs/data.yml) | Source extraction starts, overlap reference columns, publication lags and explicit per-input cube history. |
| `validate` | [validate.yml](../../configs/validate.yml) | Read-only checks, comparison thresholds and independent scientific expectations. |

`local.paths.peers_dir` in [config.yml](../../configs/config.yml) defaults to `output/peers` beneath `DATA_STORE`; an absolute path remains absolute. [Context](../../src/context.py) keeps `SECTOR_PEERS_PATH` as the complete active `sector_peers.json` filename. The active cache and named `.BEFORE.json.bak` backup relocate unchanged; peer construction and PostgreSQL description/embedding caches are unaffected. Activate path changes only with verified cache bytes at the destination; see [peer artifact paths](../modules/data-peers.md#artifact-paths).

[yf_price_bugfix.json](../../configs/prices/yf_price_bugfix.json) is the approved register for both price defects and vendor share-basis exceptions. `share_events` declares exact source expectations, sourced original-share/quote anchors, actual ordinary-share events, independently incorporated vendor restatements, optional represented price factors and explicit quoted-security units. `level_factor` retains hidden price-only wedges; `split_vintage`, `return_seams`, `null_ret` and `volume_scale` retain their repair roles. `seam_reviewed` (`ticker` to `[{date, why}]`) declares a reviewed genuine move to the validator and changes no data. See the [price-basis contract](data-sources.md#price-basis-contract).

For future supported missing/moved events or differing vendor treatment, edit this existing price register. An event's `actual` block has `date`/`ratio`; omitted `vendor` defaults to actual, while vendor ratio1 explicitly means no retrospective restatement. Omitted `price` preserves Yahoo's represented date/factor; a price override requires an independent `quoted_price` anchor in the affected interval. Source selectors describe observed rows, not replacement dates. `quote_units` carries the conversion date, before/after ordinary shares per quoted security and explicit title aliases. Generic common-stock labels do not prove ADS units.

The shared resolver in [adjustments.py](../../src/utils/price/adjustments.py) validates metadata and source/anchor evidence before a write. The scoped loader in [basis.py](../../src/utils/price/basis.py) verifies old anchors even during tail builds, using the same pure quote repairs as the price builder. Changed or unavailable evidence refuses an applicable new policy; verified empty sources differ from unavailable sources. A price anchor overlapping a nonunit hidden-level segment is explicitly refused until that combined basis can be independently verified.

A `return_seams` entry with `fields: ["close_total"]` must supply `expect_step`, `target_step`, `expect_quote_step` and evidence. Only the strict pre-event total-return prefix changes; a precise verified target is idempotent. Changed or incomplete applicable input refuses the build. Legacy entries without `fields` retain the all-price, target-one behavior. Supported total-only repairs recover total-return windows while residual quote seams keep OHLC windows masked. Historical JSON edits require the affected historical derived tables and consumers to be rebuilt; a default tail build cannot adopt an old prefix correction.

Curated evidence registers under `configs/sec/` are versioned data contracts rather than tuning knobs:

- [registrant_cutover.json](../../configs/sec/registrant_cutover.json): the register, dated legal-filer chains. One key per ticker with `kind` (`reorganisation`, `domestication`), optional `basis` (e.g. `reverse_merger_accounting_predecessor` for MRK) and `segments` (`cik`, `valid_to`, `evidence` citing the accession); the open last segment is the roster CIK. The highest-priority membership oracle; an entry always overrides an automatic window. The register follows the **traded security**: a ticker's pre-merger history is the security whose prices it carries. DD is Dow Chemical (0000029915) before 2017-08-31; PLD (AMB before 2011-06-03) and JCI (Tyco before 2016-09-02) have no entry, because the legal survivor's own CIK carries those prices; DOW has no entry and starts at 2019-04-01. A `fresh_start` entry (one segment with `valid_from`, no `valid_to`) dates a ticker's only CIK: its filings and tape count from that day, earlier ones belong to a cancelled predecessor security. EXE (Chesapeake, CIK 0000895126) starts at its 2021-02-10 emergence; the fundamentals history then keeps the comparatives of its post-emergence filings, like Sharadar.
- [entity_lineage_manual.json](../../configs/sec/entity_lineage_manual.json): CIK-to-economic-entity adjudications, one key per `<ticker>/<slug>` with `same_entity` or `own_entity` CIKs, `evidence` and `decided`. A `same_entity` entry also declares a merger's acquired target as an event-only CIK of the survivor (old ProLogis 0000899881 under PLD, old Johnson Controls 0000053669 under JCI, old DuPont 0000030554 under DD): its Forms 3/4/5 and FTD lines are kept as `acquired_constituent`, and it contributes no consolidating or event-filing row. `_d19_allowlist` holds the roster-CIK cross-check's exceptions.
- [symbol_tenure_manual.json](../../configs/sec/symbol_tenure_manual.json): `tickers` maps a ticker to evidenced half-open symbol intervals (`symbol`, `issuer_cik`, `valid_from`, `valid_to`, `evidence`); `rejected` lists derived Form 3/4/5 tenures that are false (mis-tagged filings: F/0000712537, AXP/0001387156, and one- or two-day tenures on F, NTRS, PM, AME, SPG), each with its accessions and evidence, dropped at build.
- [security_master_manual.json](../../configs/sec/security_master_manual.json): curated security-master rows, each citing a URL or accession in `source` (refused at load otherwise). `conversion_ratios` (BRK-A 30, then 1,500); `market_boundaries` (CUSIP-qualified role and issuer over half-open trade dates: APTV/DLPH, NWSA, XOM, FOXA, GOOGL, EXE, WBD); `class_overrides`; `merger_metadata` (insider boundary-day facts, e.g. MRK 2009-11-03 after the close, acquired symbol SGP); `co_registrants` (subsidiary CIKs whose rows are not stored); `exchange_ratios` (one per predecessor window whose vendor series replaces the canonical one: LIN, EVRG, BKR, STE, JCI); `vendor_series_overrides` (`ticker`, `vendor_ticker`, `cik`, `valid_from`, `valid_to`: a ticker whose own vendor series is the wrong company before a seam takes the named vendor ticker's series instead, JCI ← TYC before 2016-09-02); `reverse_acquisitions` (`ticker`, `seam_date`, `accounting_acquirer_cik`, `legal_acquirer_cik`: PLD 2011-06-03, JCI 2016-09-02; the SEC history drops a fact whose period ends before the seam when its filing's `period_of_report` is on or after it, so pre-seam periods keep the traded security's as-filed values).
- [vendor_coverage_exceptions.json](../../configs/sec/vendor_coverage_exceptions.json): `exceptions`, one accession-exact row per Sharadar quarter missing at a CIK cutover while the SEC filing exists (`ticker`, `quarter`, `period_end`, `cik`, `accession`, `filed`, `label`). An unrecorded missing quarter fails the continuity check, and so does a row whose quarter is present again (healed): delete it.
- [expected_lineage_changes.json](../../configs/sec/expected_lineage_changes.json): `hypotheses` the identity regression gate checks against the rows it computes at a cutover.

Runtime readers consume their validated/materialized representation where available; do not merge these concepts into one register. Every identity edit needs SEC evidence with an accession, then `identity` (and `identity --only sync --full` when the edit moves no lineage stamp; see the [run guide](../guides/run-the-pipeline.md#registrant-boundaries)).

Identity rules that live in code, not YAML:

- **Automatic CIK windows** are on (`AUTO_WINDOWS_ENABLED = True` in [cik_history.py](../../src/data_extract/utils/identity/cik_history.py)): a window is created only when cover pages (`dei`) and Forms 3/4/5 both place the CIK switch within 31 days (`SWITCH_TOLERANCE_DAYS`). Each is listed as an `automatic_cik_window` item by `validate identity`; a register entry wins.
- **Predecessor vendor series.** `fundamentals-sharadar` also fetches the `sharadar_tickers` rows whose SEC CIK is a register predecessor CIK, stored under their vendor ticker: BKR ← BHI, STE ← STE1, LIN ← PX1, EVRG ← WR1, plus the declared `vendor_series_overrides` (JCI ← TYC) (`load_predecessor_series` in [predecessor_series.py](../../src/utils/predecessor_series.py)). The Sharadar history replaces the canonical vendor rows inside the predecessor window with that series, converted by the cited `exchange_ratios`. PLD and DD keep Sharadar's own rows, which are already AMB's and Dow Chemical's.
- **Share basis inside a predecessor window.** Shares carry only the genuine split events `split_events` accepts; price-only factors Yahoo books as splits (JCI's 2012 ADT spin and 2016 0.955 consolidation) live in the cube's level factor S(d), so JCI's exchange ratio is 1. Inside a predecessor window, the price build's S(d) reads the window owner's `sharadar_actions` relabelled to the ticker, the owner coming from the `price_owner_windows` table (a ticker with no row is its own owner); the Sharadar history build makes the same swap from the register (`load_level_actions` in [predecessor_series.py](../../src/utils/predecessor_series.py)).


The Dataroma superinvestor roster keeps its two registers under `configs/superinvestors/`, read from `<config_dir>/superinvestors/` by [superinvestor_roster.py](../../src/utils/superinvestor_roster.py):

- [dataroma_roster_history.json](../../configs/superinvestors/dataroma_roster_history.json): one Wayback capture of the Dataroma home page per calendar quarter from 2012Q1, each with `captured_at`, `source_url` and `managers` (code to name). It is generated, never hand-edited: [build_dataroma_roster_history.py](../../scripts/build_dataroma_roster_history.py) regenerates it (see [data sources](./data-sources.md#superinvestor-roster-dataroma)).
- [overrides.json](../../configs/superinvestors/overrides.json): hand resolutions, each with its 13F evidence. `cik_overrides` maps a code to its filer CIK; `manager_ciks` chains a manager's successive filer CIKs in dated period windows under a manager ID (the oldest CIK); `unresolvable` lists codes allowed a NULL CIK; `inactive` lists dated ranges in which a resolved filer legitimately has no 13F. `load_superinvestor_overrides` validates the chains (no overlapping windows, manager ID = oldest member, a CIK in one chain only) and caches per resolved directory. Rebuild the table after any change (see [run the pipeline](../guides/run-the-pipeline.md#superinvestor-roster)).

`cluster.yml` (top key `cluster`) configures [manager clustering and the per-cluster 13F features](../modules/data-aggregate.md#sec-13f-cluster-features). `fit` is the one switch with a side effect: `true` makes `build-cluster` refit k-means on the full history and overwrite [cluster_centroids.json](../../configs/cluster_centroids.json) (the saved fit: features with their `optional` flags, standardisation, centroids, `kept`, `number` and `tracked` per fitted cluster, a `version` hash that `load` verifies, so a hand edit is refused); `false`, the committed value, only reads it. Fit settings: `k`, `keep` (must equal the fitted clusters left after `exclude`), `seed`, `n_init`, `max_rows_per_quarter`, `fit_filters` (`category`, `min_positions: 4`, `min_book_usd: 20,000,000`: the size floor, applied to every category, below which a row is `-1`), `features` (a name, or `{name, transform: rank}` for a within-quarter percentile rank, plus `optional: true` for a feature that may be null, only `turnover`: a null leaves it out of that row's distance; no return column is allowed, and a block of only optional features is refused), `exclude` (the `pct_etf` and `pct_equity_foreign` shares at or above which a fitted cluster is dropped from the kept fund ids and takes a tracked id), `labels` (the registry of every stored `cluster` id: `id`, `name`, `features` true exactly for 0 to 10, `dropped_by` naming the `exclude` key of a tracked dropped fund cluster, `category` for a tracked category; `-2` not_computable, `-1` below_size_floor, 0 to 6 the kept fund clusters named from the saved centroids, 7 to 10 bank, insurance, broker_dealer, pension, 11 fund_etf_heavy, 12 fund_etf_mixed, 13 fund_foreign_heavy, 14 sovereign, 15 other; `cluster_config` validates it and the code reads ids and rules, the names are for people) and `stability` (acceptance thresholds of the stability check against the research partition; no `src` code reads them, so they do not gate a run, and the era check currently misses its 0.70 threshold, see [TODO](../TODO.md#cluster-13f-follow-ups-2026-11-18)). The fit is one full-history fit: the centroid positions carry an accepted look-ahead (composition only, no return or target) while the labels are point-in-time through the `t-1` row; `-1` now means only a row below the size floor (19.4% of the due rows of 2000, 7.4% of 2026; the dropped ETF and foreign fund clusters are the tracked ids 11 to 13, and a manager's first row, whose `turnover` is null, is clustered on the other features), `-2` a fund missing a required feature (0.19% of eligible rows), and clusters 9 (broker-dealer) and 10 (pension) have as few as 9 to 10 members in their thinnest quarters against `min_holders` 5 and 4. `features` deliberately omits `pct_bond` (heavy-tailed, it captured a 1.1% convertible-arbitrage blob). Labelling: `cluster_label_delay_days`, and `top_x` (null keeps every member). Cube: `clusters` (`fund` for each kept cluster, and `bank`, `insurance`, `broker_dealer`, `pension` with the fixed ids 7 to 10; each has its `features` list and `min_holders`), `max_age_calendar_days` (the carry of a quarter's values after its publication), `consistency_tolerance`, and `max_clusters` / `max_features_per_cluster` (a WARNING about the column count). Timing is not in `cluster.yml`: [data.yml](../../configs/data.yml) owns it. A quarter is published on the first session on or after its 45-day filing deadline plus `data.tables.sec13f_hr.pit.lag_days` (7 days; the lag the source declares for the table and the one the user kept for the part), the rewrite window is `data.tables.cube_part_sec_thirteen.overlap` (100 calendar days) and every input of its `warmup` map (prices, `sec13f_hr`, manager details, CUSIP identity, Sharadar tickers) is read in full (null). The 80 part columns follow from `clusters`, never from a list, and the three model configs list them by name. A change to `clusters`, `features` or the fit changes labels or columns and needs the sequence in the [run guide](../guides/run-the-pipeline.md#cluster-labels-and-the-13f-cube-part).

Unused decay and coverage-break settings are removed. Filing-wide value-unit repair thresholds are owned by [value_basis.py](../../src/data_aggregate/utils/sec_thirteen/value_basis.py): 15 distinct securities, 0.8 agreement, 0.1 price tolerance and seven-day maximum quote age. They have no per-call configuration override. [ranked_managers.json](../../configs/superinvestors/ranked_managers.json) and the `data_aggregate.sec_thirteen.superinvestor` block (selection `k`, `max_age_calendar_days`) are dead config: no step reads them, and they await a documented cleanup.

## Extraction settings

`data.tables.<registered table name>` in [data.yml](../../configs/data.yml) has one supported format: source `start_date`, optional `overlap: {method, column, days}` and `pit: {column, lag_days}`. Only `days`, `missing_docs` and `missing_zips` overlap methods are accepted; only `days` has a day count. Undated lookups have no temporal entry. Dated actions and roster snapshots have start dates only. Co-produced children retain parent work selection and their own source clocks where independently consumed. Validation rejects obsolete fields, unknown table/column names and invalid values before source or store construction. The [table registry](../../src/data_store/schema.py) owns physical names, keys, forms and structural archive units; the [Window resolver](../../src/utils/window.py) owns inclusive intervals.

Every extraction/cube command accepts `--as-of` as the execution cutoff. [Context](../../src/context.py) captures that instant once and derives its New York market date. The extraction DAG relays the same validated timestamp to every source task and the triggered aggregation DAG; retries reuse it. A cutoff caps configured dates and does not widen them.

Key distinctions:

- prices, macro and splits declare seven calendar days; Sharadar SF1 declares100 calendar days per source key;
- EDGAR walks compare configured-range index accessions with completed filings; they have no numeric day-tail control;
- ZIP downloads compare published units with valid cached files; normal ingestion reuses completed units. Explicit bounded recovery may replay them;
- the universe holds one ticker per issuer; a second share class is an `aliases` entry in [universe.yml](../../configs/universe.yml), never a `tickers` entry, so no class is counted twice;
- `data_extract.earnings_calls.read_workers` in [data_extract.yml](../../configs/data_extract.yml) controls parallel HuggingFace row-group reads; writes stay on one thread. Each source key resumes from its latest saved reference minus 180 calendar days in [data.yml](../../configs/data.yml), clamped to its extraction start. The dataset repository and file path remain source facts;
- `data_aggregate.earnings_calls` in [data_aggregate.yml](../../configs/data_aggregate.yml) owns the sentiment service and66-session feature signal. Embedding service/batch/text budgets remain in [gpt.yml](../../configs/gpt.yml). [data.yml](../../configs/data.yml) declares the call publication buffer of three calendar days, market-session roll, and seven-calendar-day text-part rewrite. Older feature corrections require an explicit historical rebuild using valid cached sentiment/embeddings;
- `sec_retry` (3 attempts, waits of 5 s then 20 s, Retry-After capped at 120 s) is the one retry policy of every SEC request, applied by [sec_io.py](../../src/data_extract/utils/common/sec_io.py);
- `retry_rounds` (3 rounds, waits of 60, 120 and 240 s, threshold 0.985, `min_failed` 2): a ticker whose EDGAR read rate in a run is below the threshold, with at least `min_failed` failed documents, re-reads only those documents in-task; the rest is listed again on the next run;
- `max_documents_per_run` (5,000) caps the documents one EDGAR fetch reads per run, newest first; a larger work list logs an ERROR and drains over later runs, and `--no-cap` lifts the cap for a one-time backlog;
- `llm_text_budget` holds the characters each LLM task sends from the stored text tables: `employees` (60,000, the 10-K excerpt with its prefix), `def14a` (130,000, the cut of each stored proxy section payload) and `votes` (40,000, the cut of each stored Item 5.07 narrative). A change takes effect on the next LLM run with no re-walk;
- `validate.fundamentals_comparison` in [validate.yml](../../configs/validate.yml) (`gap_pct` 0.03, `field_share` 0.05, `top_tickers` 3): `fundamentals-check` lists a shared field when at least `field_share` of the exactly matched `(ticker, as_of)` rows with both values present differ by more than `gap_pct` of the Sharadar value, with its `top_tickers` largest-gap tickers. The check writes nothing and a finding never fails the task;
- LLM model, concurrency and prompt cache are owned by [gpt.yml](../../configs/gpt.yml). `llm_model.open_ai` selects GPT-6 Luna; employee extraction selects `llm_model.open_ai_cheap` (GPT-6 Luna) with `reasoning_effort.employees: none`. `gpt.threads` (12) sizes the LLM worker pool for DEF 14A and Item 5.07 vote extraction; employee extraction builds `data_extract.fundamentals_workers` extractors of one thread each. Every text budget is `data_extract.llm_text_budget`; the extracter sends the payload it is given whole.

## Publication and input history

[point_in_time](../../src/utils/publication.py) is the shared dataframe entry point. It uses exactly the source's `pit.column`, adds `pit.lag_days` calendar days, then rolls forward to the first actual market session. Prepare independently consumed dated sources before joining them and retain their original clocks. Invalid row dates are reported and excluded; a missing configured column is a contract error. Future availability never rolls backward.

The macro policy sets `data.tables.prices_macro.pit.lag_days=2`. The shared beta/target loader locally uses zero added lag for SPY (`equity_tr`), gold and oil; every other consumed macro series follows the configured value. Both groups use the same `point_in_time` function. Raw macro dates/values, extraction and forward-fill remain unchanged. See [macro timing for betas and targets](./modelling-and-portfolio.md#macro-timing-for-betas-and-targets).

Both notes tables use period end +12 calendar days; FTD uses half-month end +21; both 13F book families use filing date +2; calls use call date +3. Raw acceptance and capture timestamps remain evidence and do not override the declared clock. Cube and assembly dates add no lag.

Each cube's `warmup` maps physical input tables to additional calendar days before its actual earliest rewrite date. Integers read from rewrite start minus that amount through the run ceiling. Zero retains the inclusive first write date; `null` reads full stored history without an artificial lower bound. Missing dependencies fail. Entity/column projections remain. Finite budgets cannot silently widen; insufficient budgets refuse the calculation, while naturally short stored history retains ordinary unavailable/minimum-observation behavior. Statistical windows stay in [data_aggregate.yml](../../configs/data_aggregate.yml).

Institutional Schedule 13D/13G inputs explicitly use `null`: historical empty-filing markers classify submitters, so a finite cutoff can change current counts. Finite budgets are refused before either ownership payload read. FTD share-basis inputs also explicitly retain full split, action and fundamental-count evidence; their output rewrite remains 30 calendar days. Macro beta/target inputs apply the selective publication rule above before factor construction.

Operational freshness settings, `extraction-status`, source-age status checks and the prediction freshness guard are removed. Economic age rules, completeness, listing/security eligibility and denominator requirements remain. Listing grace of 10 market sessions and identity recheck of 7 calendar days are `data_extract.listing_end_grace_sessions` and `identity_change_recheck_days`; neither widens source bounds.

## Cube assembly selection

[data_aggregate.yml](../../configs/data_aggregate.yml) supplies `data_aggregate.assembly` to the existing [assembler](../../src/data_aggregate/transformers/step_assemble_cube.py) and the [readiness report](../../src/data_aggregate/utils/common/part_status.py). `compulsory_parts` plus `optional_parts` are the parts merged into the cube (prices is the root; betas are not assembled); a registered part missing from both lists is silently excluded, which `test_committed_config` guards. The committed compulsory list is prices, targets, fundamentals, momentum and governance; the optional list contains text, institutionals and SEC thirteen. Optional skips apply only to absent/physically empty tables; a populated broken input fails.

```yaml
data_aggregate:
  assembly:
    optional_parts:
      - cube_part_text
      - cube_part_institutionals
      - cube_part_sec_thirteen
    compulsory_parts:
      - cube_part_prices
      - cube_part_targets
      - cube_part_fundamentals
      - cube_part_momentum
      - cube_part_governance
    columns:
      cube_part_fundamentals: null
      cube_part_governance: null
      cube_part_institutionals: null
      cube_part_sec_thirteen: null
```

A missing `columns` mapping, omitted part entry, null or empty list includes all actual columns. Nonempty lists select exact payload names, one YAML bullet per line; date and ticker are included automatically. For example, replace a null value with:

```yaml
      cube_part_fundamentals:
        - f_earnings_yield
        - f_ocf_yield
```

Unknown parts, malformed or duplicate lists, missing selected columns and selected-name collisions fail before changing output. Selection controls both the store read and the physical output schema; reassembly removes old deselected columns. It does not change producer tasks. Later removal of model inputs requires compatible model configuration/artifacts and retraining; the prediction feature/schema guard remains active.

## Cube settings

The current target contract uses horizons 30, 60, and 90 sessions, with each horizon explicitly selected by its consumer. Labels are persisted wide as `target_<label>_h<horizon>`; changing either list changes the targets table's column set and forces a full target rebuild plus cube assembly.

Important blocks:

| Block | Contract |
| --- | --- |
| `betas` | Rolling window, minimum observations, ridge ratios, market prior, step, and forward-fill limit. Market beta shrinks toward one; other loadings shrink toward zero. |
| `targets` | Horizon/label set, minimum cross-section, volatility standardization, and joint neutralization against fitted loadings, momentum, size, and industry-group dummies. |
| `features` | Cross-sectional transform policy. |
| `intrinsic` | Two-stage DCF parameters; terminal growth must remain below discount rate. |
| `hist` | Five-year-style self-history comparison window and minimum observations. |
| `sec_thirteen` | Only the `superinvestor` block (top-K 20, 90-day book age) is dead config awaiting cleanup (the 13F cube part reads `cluster.yml`). Publication policy is in `data.yml`. |
| `fundamentals_features` | `earnings_scale_quarters` controls prior SUE scale (2–8). Eight lag slots, eight-quarter persistence and three-session response are fixed with their mathematical owners; changing them requires a feature-contract migration. |
| `workforce` | `part_time_weight`: α of the workforce headcount proxy FT + α·PT (0.65; must lie in [0.5, 1]). See [workforce features](./modelling-and-portfolio.md#workforce-features). |

Every numerical value in [data_aggregate.yml](../../configs/data_aggregate.yml) carries an economic or measured justification. Read the adjacent comment before editing. If a feature introduces a longer look-back, check its explicit input budget in [data.yml](../../configs/data.yml) and its binding requirement in the [part registry](../../src/data_aggregate/utils/common/parts.py).

## Institutional correction settings

The fails-to-deliver policy in [data.yml](../../configs/data.yml) declares 21 calendar days from the persisted ZIP half-period end, rolled to the first eligible market session. This is a release estimate, independent of cache timestamps. The policy does not certify historical SEC publication dates.

The [linear](../../configs/models/linear_modelling.yml), [LightGBM](../../configs/models/lgbm_modelling.yml) and [random-forest](../../configs/models/random_forest_modelling.yml) pools each include all29 existing institutional fields exactly once. Their respective pool sizes are73,98 and100. LightGBM excludes the retired `f_y_earnings_growth` input and retains the already selected `f_y_earnings_growth_signed`. Targets, h30/h60/h90 and fitting settings are unchanged. No fit was run.

On 2026-11-12 every list of the three family files (`columns` and each `columns_by_horizon[h]`) gained 36 SEC 13F cube features once, under `# SEC 13F cube`; the block now holds the 80 `f_ic_cluster_<id>_<feature>` names of [cluster.yml](../../configs/cluster.yml) instead, and the 36 former names (`f_ic_inst_*`, `f_ic_super_*`) are gone from the pools, the part and the code. No monotonic direction was assigned and no fit was run. Before trusting 2025-2026 predictions or assessment, run the 13F ZIP backlog (`zips-ingest --only 13f`) and rebuild `cube_part_sec_thirteen`: on that date `sec13f_hr` still missed most filers from December 2024 ([live database](./live-database.md#13f-accessions-missing-versus-the-sec-data-sets--12-november-2026)).

## Model settings

[modelling.yml](../../configs/modelling.yml) selects zscore targets, h30/h60/h90 and ElasticNet, LightGBM and LightGBM random-forest mode. Five expanding crisis-aware windows cover 2000–02,2003–06,2007–12,2013–18 and2019–21; null embargo resolves to the current label horizon. Time weighting is configured by `model.weight_decay` in `modelling.yml` and is currently disabled. Feature pools, family hyperparameters and monotonic directions are owned by the YAML files under `configs/models/`; inspect the current files before fitting. These are configurable inputs rather than invariants of the training workflow.

Each family selects all inputs, numeric and categorical, through `columns` and `columns_by_horizon[h]`. For both tree families, `lgbm.categoricals` is the sole type registry: selected names in it receive native categorical handling; a registry entry does not add an input. `lgbm.monotonic` supplies the same directions to LightGBM and RF at every horizon, aligned to the fitted feature order; unselected directions stay inactive. The text cube exposes the [exact ten role/embedding features](../modules/data-aggregate.md#exact-ten-earnings-call-features). Both [LightGBM](../../configs/models/lgbm_modelling.yml) and [random forest](../../configs/models/random_forest_modelling.yml) select all ten once; [linear](../../configs/models/linear_modelling.yml) retains its existing selection. No new earnings-call monotonic directions are assigned. Retired tone/gap/uncertainty/history/length and `_xs`/`_vs_peers` variants are not emitted. Rebuild/reassemble the cube and train fresh models before adoption; old fitted artifacts requiring removed inputs remain incompatible.

[validate.yml](../../configs/validate.yml) declares tone/coherence bounds [-1,1], role-delta bounds [-2,2] and cosine-distance bounds [0,2]. It also owns quarterly cadence, the independent `expected_signal_sessions: 66` check against the operative aggregate setting, Spearman redundancy thresholds, PSI and missingness limits, the latest-20-session audit window and unchanged quality floors (`usable_call_rate_min: 0.985`, `prepared_share_q05_min: 0.15`). Prepared share is measured only on eligible calls containing both sections; raw split status is diagnostic. Train-versus-recent checks use the canonical train.end_date in [modelling.yml](../../configs/modelling.yml). The dedicated validator reads these settings instead of duplicating them in Python.

train.end_date is a feature ceiling; train.outcome_cutoff=2021-12-31 limits every development outcome. Observed price sessions and actual label ends purge training strictly before each external test. LightGBM fits all eligible training rows and uses external-test RMSE for early stopping; there is no inner validation tail. Cutoff/full-train rounds use the upward-rounded mean of best rounds in the last two configured chronological folds. Post-2022 never selects rounds. Fixed-cutoff assessment starts 3 January 2022; the portfolio cutoff is train.end_date and its start is derived as three days after it. Full-train restores saved fitting settings/schema and fits all currently mature finite labels without CV. See [the lifecycle](./modelling-and-portfolio.md#longshort-training-lifecycle).

## Portfolio backtest

[portfolio.yml](../../configs/portfolio.yml) is the sole owner. The defaults allocate50% long/short,25% S&P and25% free cash. The three nonnegative shares must sum to one; invalid values are refused rather than renormalized. All paths start at100.

allocation_frequency defaults to weekly; LS filters independently select daily/weekly/monthly. The portfolio cutoff is train.end_date from modelling.yml and the CLI sets start to three days after it, so neither is configured here. cash_rate=0.033 compounds only free cash over calendar days. momentum_window=126 and risk_window=63 require full past sessions. commission_bps=0, spread_bps=5 and sell_fee_bps=0.206 are a documented proportional proxy, not exact historical broker fees. read_chunksize controls bounded monthly feature loading.

There are no per-sleeve configs, leverage/target-vol/ERC knobs, Streamlit settings or ledger command. Allocation changes require a fresh Python backtest. See [portfolio](../modules/portfolio.md) for financing, missing-data and output contracts.

## Validation configuration

[validate.yml](../../configs/validate.yml) maps feature prefixes to publication clocks and owns read-only fundamentals comparisons. The coverage check in [coverage.py](../../src/validate/checks/coverage.py) reads expected membership from [universe.yml](../../configs/universe.yml) and independently compares current persisted `universe_tickers` membership; there is no second expected-universe-count setting. Its default is strict and per ticker. Observed-zero declarations are narrow exceptions for builders that intentionally encode a known absence as zero; they alter the validator's clock, not stored data.

A multi-source feature becomes eligible only when all required source clocks have begun. Configuration cannot convert a missing dependency into an observation.

## Adding or changing a knob

1. Identify the stage that owns the behavior.
2. Add the key to that stage's existing YAML and explain why the default is appropriate.
3. Read it through the stage's config block; do not add a second default in Python.
4. Update tests for schema/contract changes.
5. If the value changes columns, look-backs, availability, or portfolio artifacts, update the relevant [cube](../modules/data-aggregate.md), [source](./data-sources.md), or [model](./modelling-and-portfolio.md) reference.
6. Request approval before changing `configs/`, which is a repository risk zone.

## Constants versus configuration

Use configuration for research choices and operational tuning. Use [constants.py](../../src/constants/constants.py) for URLs, date formats, source series, taxonomy facts, model identifiers, plausibility bounds, and other facts that should not vary between runs. The GICS sector and industry-group vocabulary lives with its owner in [vocabulary.py](../../src/utils/tickers_universe/vocabulary.py). Use the [table registry](./table-catalog.md) for table names, grain, and the extracted SEC forms in each table's `Resume.forms`.

## Related

- [Constants and configuration module](../modules/constants-and-configuration.md)
- [Modelling and portfolio contracts](./modelling-and-portfolio.md)
- [Add a model or sleeve](../guides/add-a-model-or-sleeve.md)
- [Run the pipeline](../guides/run-the-pipeline.md)

## Institutional value-unit candidate — 10 October 2026

The historical P1 candidate introduced min_securities=15, min_agreement=0.8, price_tolerance=0.1 and max_price_age_days=7; these values now live with [value_basis.py](../../src/data_aggregate/utils/sec_thirteen/value_basis.py), without YAML knobs. They govern a common filing-wide monetary factor against reconstructed report-quarter-end as-traded prices, independently per quarter/filing. They do not represent the separate P2 minimum of 15 managers holding the same security. Shares are unchanged; explicit shares==value reconstruction and source-only/abstained statuses are distinguishable.

The candidate is not activated. The later user-approved fallback estimates uncertain common dollars as valid reported shares × the eligible restored quarter-end price, after the raw unit decision; estimated rows cannot vote for a factor. Source zeros remain zero. The subsequent typed-option extension uses call_shares/put_shares × the same eligible reference for uncertain underlying exposure; options never vote for the common filing factor. Nonzero unresolved debt values still require a debt-instrument quote and remain unknown. Common-dollar completeness is now 457/461, 472/476 and 436/482 ticker totals in the three tested first-publication cohorts; option/value coverage and immutable filing versions remain pending. Local candidate evidence: `reports/validate/2026-10-09-institutional-13f-audit/06-final.md`.

## Saved-model assessment controls

[modelling.yml](../../configs/modelling.yml) owns model.assessment.output_dir (output/modelling_backtest), evaluation_start (2022-01-03) and read_chunksize (I/O batching only). Train generates a fresh ID each time. Assess defaults to the latest completed training; optional CLI --run-id or constructor run_id selects exactly one run. Portfolio/full-train default to the latest assessed training. The DAG pins train's returned ID through XCom.

model.diagnostics controls explanations/display only: shap_sample=12000 per fold-test population/horizon allocated across declared folds plus another12000 cutoff rows; dependence_sample=1000 displayed points; pdp_grid=30 maximum continuous quantile points; pdp_background=2000 averaged rows per PDP point; rolling_window=252 supported observed sessions. IC always uses the complete scoring population. Tree importance is sampled absolute SHAP; linear importance is saved absolute standardized beta. Constrained responses can reflect an imposed prior.

Family columns_by_horizon selects the complete numeric/categorical input list. Missing overrides use columns; an explicit empty list selects none. lgbm.categoricals assigns types to selected inputs, and lgbm.monotonic supplies shared tree-family directions. No pruning/search/YAML-edit parameters remain.

Assess always recomputes scores, metrics, SHAP/PDP and HTML from saved models and current data. No report-only/resume mode or cache/source/current-YAML equality gate remains. Full-train restores the selected run's saved fitting settings, seed, exact resolved schema and final rounds. Model byte integrity remains checked on load. Production model.models_dir stays separate from dated cutoff artifacts. See [artifact paths and report controls](../modules/modelling.md).

## Cube timing and recovery

Each cube resolves its policy through [plan_window](../../src/data_aggregate/utils/common/incremental.py) and immutable [Window types](../../src/utils/window.py). `Context.run` owns the execution instant and mode. Raw observed prices define root output dates; saved price-part dates and keys define downstream output. Cubes have no independent start/end/PIT block. Explicit per-input Windows expand reads while writes retain their own bounds.

Normal rewrite tails are governance 380 calendar days, fundamentals/13F 100, prices/betas/momentum/institutionals 30, text 7 and targets 150. The target planner checks actual market sessions against every configured horizon and refuses an insufficient overlap. Joint targets/betas use the union of input requirements and retain separate write intervals. [Assembly](../../src/data_aggregate/transformers/step_assemble_cube.py) takes the earliest actual selected-part write boundary plus its catch-up boundary and reads its zero-history inputs over that interval.

Separate CLI/DAG tasks exchange nothing: every cube task receives the same `--as-of` cutoff (the trigger's `conf.as_of`, else the DAG run's `logical_date`, which clearing a task cannot change) and the same `-F` mode, and `assemble-cube` plans each selected part's window from that part's stored frontier. A part that wrote further back than its normal overlap (a cold start, a promotion to a full rebuild, a backfill) is not detected by the assembler; run `assemble-cube -F` after it. The in-process `build-cube` is the exception: it passes each part's exact write window to the assembler in memory.

Both 13F book families use exactly the declared filing-date clock plus 2 calendar days; the 90-calendar-day superinvestor book lifetime remains a separate economic rule. The approved bounded-memory path retains its broad stream and two whole-roster passes for consensus and final repair, all with identical source bounds. Stale or invalid sources do not become known zero.

The existing governance fills intentionally use later filings for CEO pay, stint constants, age and director/person attributes. The existing pooled13F unit/value repair also retains its donor evidence calculation. These user-approved retrospective calculations are explicit exceptions to strict evidence-level PIT; applying a destination filing buffer does not certify those imputed/repaired values as strictly causal. Their algorithms are preserved by this refactor.

Older governance/text corrections require an explicit historical rebuild. A price correction that changes the saved prefix outside its normal write window is refused before writes; use an aligned explicit full rebuild and reassembly. The refactor performs no production historical rebuild and does not validate predictive models or their artifacts.
