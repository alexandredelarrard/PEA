# CLAUDE.md

## Risk Zones (Ask Before Editing)

| File / Directory | Risk Reason |
|---|---|
| `src/context.py` | Global pipeline context — changes cascade everywhere |
| `src/utils/step.py` | Base class for all steps — breaks all inheritors |
| `src/constants/*.py` | Global literals — breaking renames cascade downstream |
| `configs/*.yaml` | Schema changes must be mirrored in consuming code |
| `src/data_store/*`, `sql/schema.sql` | DB layer & DDL — schema renames affect all reads/writes |
| `data/` & Postgres Volume | Non-recoverable model artifacts, caches, and DB volume |

> **Rule:** Always propose changes and obtain approval before editing risk zone files.

---

## Architecture & Code Structure (`stock_pick_strat/`)

Pipeline executes sequential `Step` classes (base: `src/utils/step.py`) wired by `main.py`. **All tabular data lives in PostgreSQL** accessed via `context.store` (`DataStore`).

### Core Infrastructure
- `src/context.py` — `Context`: configs, logging, `.store` (DB access), `.paths` (artifact paths).
- `src/constants/constants.py` — Global literals (dates, SEC URLs, sector scope thresholds).
- `src/data_store/` — the ONLY code in the repo that issues SQL (enforced by `tests/data_store/test_store_boundary.py`). Three files:
  - `schema.py` — **THE table registry.** One `Table` per DB table, declared once, carrying `name`, `pk`, `date_col`, `ticker_col`, `date_type_cols`, `vector_col`, `managed`, freshness cadence and read projection. Reference tables as `Tables.<name>`; **never** as a string literal, and never re-declare a table name elsewhere. Derived views (`ALL`, `BY_NAME`, `MANAGED`, `PARTS`, `freshness_tables()`, `projection()`) are comprehensions, not hand-lists.
  - `store.py` — `DataStore`, the single access layer. Reads: `load` / `iter_load` (+ `exists`, `columns`, `row_count`, `bounds`, `max_date`, `distinct`). Writes: `save` (upsert on the registry PK), `replace`, `append_tail`, `bulk_seed`, `delete`, `drop`, `ensure_columns`. Filters compose server-side: `where=` (equality / `IN` / `IS NULL` / `store.NOT_NULL`), `since=` / `until=`, `columns=` or `project=True`. All bound parameters — never string interpolation.
  - `ddl.py` — generates `sql/schema.sql` from the registry + live reflection (`python -m scripts.generate_schema_sql`); regeneration carries over any table it cannot reflect, so a table absent from the live DB does not lose its DDL.
- **`load` raises by default.** A missing or empty table is nearly always a real fault, so `load` raises `TableMissingError` / `TableEmptyError` (`errors.py`) rather than returning a fabricated empty frame. Pass `optional=True` **only** where finding nothing is legitimate (a fetcher's resume check on a cold DB, an optional feature source) — and then branch on `is None`, not `.empty`.
- `src/sql/` - `database.md` and `schemas.sql` to understand data sources, fields, refreshing pace and structure
- `.env` has database and api keys saved + SEC profile

### Universe & Macro Assets
- **Equity Universe (`sp500_tickers`):** Resolved via `src/utils/universe.py::load_universe_tickers(context)`. Seeded by S&P 500 scraper if empty or `refresh_universe: true`. All peer/cube/modeling steps consume this universe. Index changes only require updating rows in `sp500_tickers`.
- **Market/Macro Series (`other_tickers`):** SPY, VIX, oil, gold, FX fetched as OHLCV into `prices` via `fetch_market_prices`. Never added to the equity universe.

### Pipeline Steps (`src/`)

#### 1. Data Extraction (`src/data_extract/` via `StepExtractAllData`)
- **Prices (`utils/prices/`):** Prices, dividends, short interest, SEC FTD (`fails_to_deliver`), 13F, superinvestors (`fetch_superinvestors` → `data/superinvestors/superinvestors.json`), multi-asset macro series (`fetch_macro_assets` → `macro_asset_prices`).
- **Fundamentals (`utils/fundamentals/`):** `fetch_fundamentals_edgar.py` → `fundamentals_facts` (accession-grain raw facts) → `fundamentals_derive.py::rebuild_fundamentals_history` → `fundamentals_history` (PK: `ticker, as_of`). Period resolution in `fundamentals_periods.py`; diagnostics in `fundamentals_validation.py`.
  - **Tag resolution (`fundamentals_tags.py`):** Each logical field maps to a global, priority-ordered candidate tag list; the coalesce is era-agnostic by design, so a US-GAAP taxonomy migration (ASC 842, CECL, `X`→`XNet`) needs no dates. Two narrow overrides sit on top, both applied in `build_tag_frames` after the tag_map merge: `NON_NEGATIVE_STOCK_FIELDS` (a negative debt/asset/share-count is a filer defect → fact inadmissible, coalesce falls through) and `FIELD_TAG_DENYLIST` (per-issuer escape hatch, **deny never pin**, so an unlisted ticker keeps global resolution). **Rule:** a deny-list entry is the conclusion of a diagnosis — add one only with the evidence written beside it, after `fundamentals_tag_ledger` has ranked the case.
  - **Multi-class share counts (`CLASS_OF_STOCK_AXIS_SUFFIX`, `SHARE_CLASS_COMPONENT_FIELDS`):** A filer with more than one class of common stock tags NO undimensioned share count anywhere — every fact sits on `StatementClassOfStockAxis`, the classes disagree, so the dimension rules refuse them all and `shares_outstanding` (hence market cap) came out NULL for the whole multi-class cohort. `build_tag_frames` therefore (a) never admits a class-dimensioned fact as the company total and (b) rebuilds the total by summing the **cover-page** classes only (`dei:EntityCommonStockSharesOutstanding`), which the SEC cover page requires to be an exhaustive per-class enumeration. The balance-sheet parenthetical is **never** summed — measured incomplete or overlapping on 6 of 36 filers. `_cover_page_shares_fallback` then re-stamps the total onto the period as it already does for single-class filers. Where classes do **not** convert 1:1, the sum is put into the traded class's units using factors the filers tag themselves — `CommonStockConversionRatio` per class (ERIE ×2400 → exact), an `EconomicEquivalentPercentage` on the senior class (BRK-B 6.67e-4 → 0.02%), or a filer's own `SharesOutstandingAsConvertedBasis` (V, priority-0, needs no arithmetic). All **fill-only**: absent the hook the plain sum stands.
  - **Consolidated basis (`PARENT_OWNERSHIP_PERCENTAGE_TAG`):** market cap, `netIncome` (`ProfitLoss` first) and `stockholdersEquity` (incl-NCI first) are all on the **whole consolidated group**, matching `totalRevenue`/`totalAssets` — which have no parent-only concept in US-GAAP — and matching what every vendor publishes (verified: Yahoo's `marketCap` ÷ price ÷ `impliedSharesOutstanding` = 1.000). Previously income was the parent's slice while revenue was the group's, so a high-NCI filer's ratios were built from two different companies (IBKR's parent takes 22.6% of income; `sales_yield` was ~3.8x too high). For an **Up-C**, the share count is grossed up by the tagged parent ownership % — but only when the class sum demonstrably does not already cover the non-controlling holders, which the filing itself decides: IBKR's Class A is 99.99998% of its class sum against a tagged 26.6% (gross up), CVNA's is 65.4% against 65% (already whole — leave alone, its Class B is paired 1:1 with the LLC units). **Known artifact:** an Up-C's NCI income escapes the parent's corporate-tax layer, so a consolidated P/E reads cheaper than a buyer of the traded class gets (IBKR 33.6x vs 39.4x) — shared with every vendor; the alternative needs a parent-level revenue that does not exist.
  - **Tag ledger (`src/utils/fundamentals_tag_ledger.py`):** Collapses `fundamentals_facts` into `source_tag` eras and flags boundaries where the *level* jumps across a concept switch — i.e. two measures spliced into one column. Complements `analyze_history.py::detect_source_tag_misalignment` (which compares period-end vs interim tags *within* a fiscal year and deliberately ignores cross-year cutovers). Flag-only; writes `data/gaps/fundamentals_tag_{ledger,breaks}.csv`. `n_boundaries` separates a one-time cutover from a systematic per-filing swap; `n_tickers_same_switch` separates a taxonomy migration (fix the candidate list) from one filer's mis-tagging (deny-list entry). Employee headcount is a field of this table (`fundamentals_employees.py`, parsed from 10-K body text — no XBRL concept exists), surfacing as the `employees` column; the old `employees_history` table is retired.
- **Structure (`utils/structure/`):** DEF 14A governance/executive pay (`def14a_llm`), 8-K events (`fetch_8k_edgar.py` → `sec_8k`), 13D filings (`fetch_13d_edgar.py` → `sec_13d`, PK: `ticker, accession_number, rp_seq`). Deterministic proxy extraction: `fetch_def14a_edgar.py` → `def14a_edgar` (PK: `ticker, accession_number`) + 4 child tables (`_executive_comp`, `_director_comp`, `_ownership`, `_votes`).
  - **DEF 14A repair layer (`def14a_validate.py`):** edgartools' proxy HTML parser emits values that are silently WRONG rather than absent, so every row passes through this module before it is saved. It rescales unit-broken fee blocks, NULLs edgartools' fabricated `0.5` placeholder for the "*" (= "less than 1%") ownership footnote, undoes Total-column duplication, strips glued titles/footnote indices/addresses off name primary keys, drops subtotal pseudo-rows, and completes the CEO pay-ratio identity. **Rule: never fabricate** — a value is written only when deterministically recoverable, otherwise it is set to NaN. Only `def14a_edgar`'s XBRL-backed block is trusted unconditionally; the HTML-parsed child tables remain best-effort and are complemented by the LLM path.
- **Behavioral (`utils/behavioral/`):** Wikipedia, Google Trends, earnings call transcripts (`earnings_call_sections`). Deep history backbone: HuggingFace `kurry/sp500_earnings_transcripts`. Recent gap filler: `utils_missing_quarters.py` → Roic AI → Motley Fool quote pages. HTTP transport: `src/utils/polite_http.py` (`curl_cffi` rotation, rate limiting).
- **Shared Plumbing & Bulk SEC (`utils/common/`):** `bulk_cache.py` (zip caching & self-healing), `sec_utils.py` (rate limiting, state), `form_registry.py` (`FORM_REGISTRY`). Bulk datasets: insider (`insider_transactions`), financial statements (`pension_facts`), notes (`notes_num`, `notes_text`).

#### 2. Peer Deduction (`src/data_peers/`)
- `StepDeducePeers`: Return correlation and OpenAI embedding-based peer groups.

#### 3. Feature Aggregation (`src/data_aggregate/`)
`StepBuildCube` is a SUPER STEP over seven sub-steps in `transformers/` (mirroring
`StepExtractAllData`). Each persists one `cube_part_*` table and is ALSO a standalone CLI
command / DAG task; `run()` drives the same seven objects in-process. One code path, two drivers.

| sub-step | writes | contents |
|---|---|---|
| `StepCubePrices` | `cube_part_prices`, `cube_part_market` | the ONLY reader of raw `prices`: pivot, trading calendar, returns, universe restriction, peer sector returns |
| `StepCubeTarget` | `cube_part_targets`, `cube_part_betas` | factor panel → rolling betas → multi-horizon factor-neutral labels |
| `StepCubeFundamentals` | `cube_part_fundamentals` | fundamental, sector-KPI, earnings, workforce, dividend (one `fundamentals_history` read shared via `PitFrames`) |
| `StepCubeMomentum` | `cube_part_momentum` | everything from price variation: momentum, vol, trend, lottery, liquidity, seasonality, MACD/RSI/ATR |
| `StepCubeText` | `cube_part_text` | earnings-call FinBERT sentiment + OpenAI-embedding KPIs (both STREAM their sources) |
| `StepCubeExtras` | `cube_part_extras` | governance, 13F, elite 13F, insider, short interest, attention |
| `StepAssembleCube` | `cube` | read the parts → composites → per-horizon streamed write |

- **Memory invariant:** each sub-step keeps its heavy frames LOCAL to `run()` and reads the price
  grid back from `cube_part_prices` PROJECTED to the fields it declares (`_FIELDS`). Peak memory
  is the largest single sub-step, not the sum. Never stash a frame on `self`.
- **Part registry (`utils/common/parts.py`):** `CUBE_PARTS` is the single source of truth for part
  names, CLI commands, incremental warm-ups and per-group binding look-backs. The CLI, the DAG
  chain and `cube-status` all derive from it — do not hand-list parts anywhere.
- **Shared layer (`utils/common/`):** `price_frames.py` (the `PriceFrames` contract),
  `pit.py` (point-in-time accessors + the memoizing `PitFrames`), `panel.py` (peer-relative panel),
  `frames.py` (`ratio`/`safe_div`/`sanitize`), `xs.py` (ONE cross-sectional z + rank — the 3.0/4.0/8.0
  clips are three deliberate policies, `clip` is a required argument), `prices.py` (momentum,
  trailing vol, forward windows, `price_column_returns`), `incremental.py` (part lifecycle +
  the full-vs-incremental decision, straight on `context.store`), `panel_merge.py`,
  `capital.py`, `sector_gates.py`, `gics.py`, `peers_io.py`, `sources.py`, `data_utils.py`.
- **Collision Protection:** `PanelMerger.add` raises `FeatureCollisionError` on a duplicate feature
  name, naming the panel that already owns it. Applied to BOTH the per-step merge and the
  cross-part merge in assemble.
- **Domain utils:** `utils/{target,fundamentals,momentum,text,extras,assemble}/` hold the builders
  each sub-step calls. `fundamental_features._derived_fields` is a thin composition over ~30
  per-block `_*_fields(daily, …) -> dict` builders.
- **Guard:** `tests/data_aggregate/test_aggregate_regression.py` hashes 35 aggregation outputs
  (15 panels, 13 deduplicated primitives, 6 labels, the frozen input) against
  `aggregate_fingerprint_baseline.json`. **The baseline may be regenerated only in a commit that
  touches no `src/` file, or in a PR that is exclusively a declared numeric change.**

#### 4. Modelling (`src/modelling/`)
- **Long/Short (`long_short/`):** `step_train.py` (`StepModelling`) trains cross-sectional ensembles (ElasticNet + LightGBM + RandomForest). Output diagnostics to `data/output/diagnostics/<run_stamp>/`. Tree boosters resolved via `isinstance(m, lgb.Booster)`.
- **Trend (`trend/`):** `signal.py` (`trend_book`) computes directional time-series momentum on `macro_asset_prices`.
- **Long Book (`long_book/`):** `allocation.py` executes multi-asset ERC/risk-parity allocation with trend & VIX overlays.
- **Shared Utilities:** Reusable signal/math modules in `src/utils/trend.py` and `src/utils/risk_parity.py`.

#### 5. Strategies (`src/strategies/`)
Self-contained steps implementing `base.Strategy` interface (`run(inputs: PortfolioInputs) -> StrategyResult`):
- `step_ls.py` (`LongShortStrategy`, `"ls_equity"`): Out-of-sample prediction + dollar/beta/sector-neutral optimization (`utils/strategies_opt.py`).
- `step_long_book.py` (`LongBookStrategy`, `"long_book"`): Long-only allocation strategy.
- `step_trend.py` (`TrendCTAStrategy`, `"trend_cta"`): Vol-scaled trend strategy.
- **Analytics (`analysis/`):** Strategy-specific performance plots in `data/output/<sleeve>/analysis/`.

#### 6. Portfolio & Execution (`src/portfolio/`)
- `StepPortfolio` (`step_portfolio.py`): Blends sleeve return streams using risk-parity (ERC) weights (`utils/blend.py`) with a global volatility target.
- `StepStrategyMoves` (`step_strategy_moves.py`): Generates the live trading ledger (`strategy` table). Re-sizes the traded weight panel (`erc_weight × leverage`) before calculating share quantities. FIFO lot matching in `src/strategies/utils/positions.py::round_trip_ledger`.

### Config Hierarchy
- Strategy configs: `configs/strategy/{strategy_ls,strategy_trend,strategy_long_book}.yml`
- Portfolio config: `configs/portfolio.yml`
- Model hyperparams: `configs/models/` & `modellling.yml`

---

## Data & DB Conventions

- **Database I/O:** Read/write tabular data exclusively via `context.store`. Never import `sqlalchemy`, call `pd.read_sql`/`to_sql`, or touch `store.engine` outside `src/data_store/` — if the facade can't express a query, ADD the capability to `DataStore` rather than bypassing it. Schema auto-expands via `ensure_columns`. Keep non-tabular artifacts (models, plots, raw JSONs) in `context.paths`.
- **Never read a full large table.** Always project (`columns=` / `project=True`) and scope (`where=` / `since=` / `until=`); use `iter_load` for anything cube-sized. `cube` is ~26 GB and `sec13f_hr` ~21.7M rows — an unprojected read of either is an OOM, not a slow query.
- **Table names:** one source of truth, `schema.py`. Register a new table there and reference it as `Tables.<name>`; do not add a `*_TABLE` constant.
- **Point-In-Time & Incremental:** Save work per entity to prevent data loss. Lag features by filing date to prevent forward-looking bias.
- **SEC & XBRL Handling:** Coalesce alternative XBRL tags across candidate lists per period instead of picking the first present. Convert boolean flags to float indicators (`1.0`/`0.0`).

---

## Development Workflow & Rules

### Workflow for New Features
1. Check `src/constants/*.py` for existing literal definitions before introducing new keys/columns.
2. Check `src/utils/` for existing helpers.
3. Implement feature and place cross-module utilities in `src/utils/`.
4. Write unit test alongside the implementation.
5. Sanity-check edge cases against real data (NaNs, sparse feeds, TTM warmups, sector exclusions).
6. Run targeted test: `pytest tests/path/to/test.py::test_func -v -s`.

### How to Communicate Results
- Output ONLY the new test results along with the explicit printed sanity check conclusion.
- Do NOT output full test suite summaries unless asked.
- On refactorings, state affected tests and reasons concisely.
- For multi-step tasks, confirm each step prior to proceeding.

### Guidelines & Directives

#### MUST DO
- Put all global literals (date formats, URLs, thresholds) in `src/constants/constants.py`.
- Log strictly using `self._context.logger` (never `print()`).
- Keep import statements at the top of Python files.
- Place reusable functions in `src/utils/`.
- Maintain synchronization between `CLAUDE.md`, `AGENTS.md`, and `README.md`.
- Propose new conventions before updating `CLAUDE.md`.

#### MUST NOT DO
- Do NOT replace OmegaConf.
- Do NOT alter the `Step` inheritance architecture.
- Do NOT cross-import between `src/` subfolders (e.g., `data_extract` importing from `modelling`).
- Do NOT hardcode inline strings or file paths.
- Do NOT reformat unrelated code.
- Do NOT mark work as complete without including the printed sanity check conclusion in the test output.