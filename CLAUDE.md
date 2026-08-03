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
- `src/data_store/` — `store.py` (`DataStore`: `load`, `save`, `replace`, `ensure_columns`, `existing_dates`), `schema_registry.py` (PKs & date cols), `schema_sql.py` (`sql/schema.sql`), `io.py`.
- `src/sql/` - `database.md` and `schemas.sql` to understand data sources, fields, refreshing pace and structure

### Universe & Macro Assets
- **Equity Universe (`sp500_tickers`):** Resolved via `src/utils/universe.py::load_universe_tickers(context)`. Seeded by S&P 500 scraper if empty or `refresh_universe: true`. All peer/cube/modeling steps consume this universe. Index changes only require updating rows in `sp500_tickers`.
- **Market/Macro Series (`other_tickers`):** SPY, VIX, oil, gold, FX fetched as OHLCV into `prices` via `fetch_market_prices`. Never added to the equity universe.

### Pipeline Steps (`src/`)

#### 1. Data Extraction (`src/data_extract/` via `StepExtractAllData`)
- **Prices (`utils/prices/`):** Prices, dividends, short interest, SEC FTD (`fails_to_deliver`), 13F, superinvestors (`fetch_superinvestors` → `data/superinvestors/superinvestors.json`), multi-asset macro series (`fetch_macro_assets` → `macro_asset_prices`).
- **Fundamentals (`utils/fundamentals/`):** `fetch_fundamentals_edgar.py` → `fundamentals_facts` (accession-grain raw facts) → `fundamentals_derive.py::rebuild_fundamentals_history` → `fundamentals_history` (PK: `ticker, as_of`). Period resolution in `fundamentals_periods.py`; diagnostics in `fundamentals_validation.py`. Employee headcount is a field of this table (`fundamentals_employees.py`, parsed from 10-K body text — no XBRL concept exists), surfacing as the `employees` column; the old `employees_history` table is retired.
- **Structure (`utils/structure/`):** DEF 14A governance/executive pay (`def14a_llm`), 8-K events (`fetch_8k_edgar.py` → `sec_8k`), 13D filings (`fetch_13d_edgar.py` → `sec_13d`, PK: `ticker, accession_number, rp_seq`).
- **Behavioral (`utils/behavioral/`):** Wikipedia, Google Trends, earnings call transcripts (`earnings_call_sections`). Deep history backbone: HuggingFace `kurry/sp500_earnings_transcripts`. Recent gap filler: `utils_missing_quarters.py` → Roic AI → Motley Fool quote pages. HTTP transport: `src/utils/polite_http.py` (`curl_cffi` rotation, rate limiting).
- **Shared Plumbing & Bulk SEC (`utils/common/`):** `bulk_cache.py` (zip caching & self-healing), `sec_utils.py` (rate limiting, state), `form_registry.py` (`FORM_REGISTRY`). Bulk datasets: insider (`insider_transactions`), financial statements (`pension_facts`), notes (`notes_num`, `notes_text`).

#### 2. Peer Deduction (`src/data_peers/`)
- `StepDeducePeers`: Return correlation and OpenAI embedding-based peer groups.

#### 3. Feature Aggregation (`src/data_aggregate/`)
- `StepBuildCube`: Merges feature panels into the `cube` table.
- **Primitives (`utils/panel.py`):** `_ratio`, `_winsorize_xs`, `_peer_relative`, `build_peer_relative_panel`.
- **Collision Protection:** `_merge_panel` raises `FeatureCollisionError` on duplicate feature names across panels.
- **Sector Gating:** Sector KPIs scoped by GICS via `utils/sector_gates.py` (`SECTOR_KPI_SCOPE` in `constants.py`).
- **Capital Standardization:** Debt/net-debt/invested-capital calculations centralized in `utils/capital.py`.
- **Panels:** Fundamental, sector, forward valuation, earnings, governance (`def14a_impute.py`), employee, dividend, attention, institutional 13F, superinvestors (`superinvestor_features.py`), short interest, earnings call sentiment & embeddings (`earnings_call_features.py`, `nlp_sentiment.py`, `openai_embeddings.py`).

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

- **Database I/O:** Read/write tabular data exclusively via `context.store` (`load`/`save`/`replace`/`existing_dates`). Schema auto-expands via `ensure_columns`. Keep non-tabular artifacts (models, plots, raw JSONs) in `context.paths`.
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