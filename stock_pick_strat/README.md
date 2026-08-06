# S&P 500 Long/Short Stock-Picking Pipeline

A factor pipeline for a long/short S&P 500 equity strategy. It extracts price,
fundamental, governance, ownership, and alt-data signals from free sources into a
**PostgreSQL** database, builds a point-in-time, peer-relative feature **cube**,
trains cross-sectional models, and backtests the resulting signal.

Everything tabular lives in Postgres (Docker + a persistent volume) and is
accessed through a single `DataStore` facade — there are no parquet data files.

## Architecture

A chain of `Step` classes (`src/utils/step.py`), each with a `run()`, wired from
`main.py`:

```
StepExtractAllData          # 1. extract  (super-step → 4 sub-steps)
  ├─ StepExtractPrices        prices (+dividends), short interest, 13F holdings
  ├─ StepExtractFundamentals  SEC per-filing XBRL history (+ 10-K employee headcount), forward-P/E snapshot, earnings, macro
  ├─ StepExtractStructure     management, DEF 14A governance (LLM), SEC filings index
  └─ StepExtractBehavioral    Wikipedia pageviews, Google Trends, earnings-call transcripts
StepDeducePeers             # 2. peer baskets (return-corr + OpenAI embeddings)
StepBuildCube               # 3. super-step → 7 sub-steps, one `cube_part_*` table each
  ├─ StepCubePrices           normalized OHLCV + returns + peer sector returns (the only
  │                           reader of raw `prices`; every later step reads the part back)
  ├─ StepCubeTarget           factor panel → rolling betas → multi-horizon neutral labels
  ├─ StepCubeFundamentals     fundamental, sector KPI, earnings, workforce, dividend
  ├─ StepCubeMomentum         price-variation features (momentum, vol, MACD, liquidity)
  ├─ StepCubeText             earnings-call sentiment + embedding KPIs
  ├─ StepCubeExtras           governance, 13F, elite 13F, insider, short interest, attention
  └─ StepAssembleCube         read the parts → composites → the `cube` table
StepModelling               # 4. train the L/S ensemble (src/modelling/long_short) → `predictions`, `cube_signal`
StepPortfolio               # 5. run + blend the strategy sleeves into one book vs SP-hold
StepStrategyMoves           # 6. the trades to actually place → `strategy` (entry/exit price + PNL per position)
```

### Airflow schedule
| DAG | When | Tasks |
|---|---|---|
| `data_extraction` | nightly | per-source fetchers → freshness gate → triggers `data_aggregation` |
| `data_aggregation` | after extraction | peers → 6 sequential build steps → `assemble_cube` → `cube_status` → triggers `strat_prediction` |
| `strat_prediction` | **daily** 06:00 | `predict` (long-format `predictions_latest`) → `strategy_moves` (the `strategy` ledger) |
| `modelling` | **weekly**, Sat 02:00 | `train_model` (holdout) → `backtest_portfolio` (OOS) → `full_train` (ALL history, no holdout) |

Training is weekly and prediction is daily, so a freshly rebuilt cube is scored every night
without waiting for a retrain; `strat_prediction` reads back the artifacts from the last weekly
`full_train`.

### Strategies & portfolio
The backtest layer is a set of **self-contained strategy sleeves** (`src/strategies/`, each a
`Strategy.run(PortfolioInputs) → StrategyResult`) blended by a portfolio step (`src/portfolio/`):

| Sleeve | Model (`src/modelling/…`) | What it is |
|---|---|---|
| `ls_equity` | `long_short/` (trained ensemble) | market-neutral equity long/short alpha (OOS from `train_end`) |
| `long_book` | `long_book/allocation.py` | long-only multi-asset ERC allocation + trend overlay + VIX regime tilt |
| `trend_cta` | `trend/signal.py` | long/short multi-asset time-series-momentum (crisis-alpha diversifier) |

`StepPortfolio` reads `configs/portfolio.yml` (sleeve set + global vol/leverage/capital), runs each
sleeve on `configs/strategy/strategy_*.yml`, and blends their return streams by risk-parity/ERC
(= dynamic $-allocation) + a global vol target — reporting **per-strategy Sharpe vs the global
portfolio**. Each sleeve + the portfolio save analysis plots (IC, neutrality, correlation) under
`data/output/*/analysis/`. The long-book/trend sleeves run off the long-history `macro_asset_prices`
table (FRED rates/cash/FX + yfinance equity/gold/energy, since ~1995).

### Data sources (all free / freemium)
| Domain | Source | Notes |
|---|---|---|
| Prices, dividends | yfinance | full daily OHLCV, no key |
| Fundamentals (history) | SEC EDGAR `companyfacts` (XBRL) | ~10-15y point-in-time, keyed on filing date |
| Forward P/E (historical) | derived from `earnings_surprises` (consensus EPS) ÷ price | NTM forward-earnings yield, backtestable; market cap = shares × daily close |
| Governance / comp / ownership | SEC **DEF 14A** proxy, parsed by **OpenAI** (structured output) | directors, CEO age/pay, board, say-on-pay, ownership |
| Employees, mgmt, filings | SEC EDGAR (10-K text, submissions) | |
| Institutional holdings | SEC **Form 13F** data sets | split by stock / call / put / debt; CUSIP→ticker via OpenFIGI |
| Macro | FRED | yields, curve, VIX, credit spreads, breakeven |
| Short interest | FINRA RegSHO | daily short-volume |
| Attention | Wikipedia pageviews, Google Trends | retail-attention alt-data |
| Earnings-call tone | Motley Fool transcripts → local **FinBERT-tone** (GPU) + LM-uncertainty | `f_ec_*` tone / Q&A-gap / uncertainty / vocab-novelty (MF scraper needs a rework — JS/anti-bot site) |
| Peers / descriptions | OpenAI embeddings | business-similarity peers |

## Quick start

**1. Start the database** (Postgres 16 + persistent volume):
```bash
cd stock_pick_strat
docker compose up -d db          # creates DB pea/pea/pea on localhost:5432
```

**2. Configure `.env`** (git-ignored) in `stock_pick_strat/`:
```dotenv
SEC_USER_AGENT="Your Name your.email@example.com"   # required by SEC EDGAR
FRED_API_KEY=...                                     # free: fredapi
OPENAI_API_KEY=sk-...                                # DEF 14A extraction + embeddings
OPENFIGI_API_KEY=...                                 # optional: speeds 13F CUSIP mapping
# DB defaults to pea/pea/pea@localhost:5432; override with POSTGRES_* / DATABASE_URL
```

**3. Install deps & run** (Poetry):
```bash
poetry install
poetry run python main.py        # instantiates the steps; uncomment the .run() you want
```

The schema (`sql/schema.sql`) is applied automatically on first DB init. To seed
an existing DB from legacy parquet (one-off): `python -m scripts.migrate_parquet_to_db --create --bulk`.

## Database

Key tables (PK): `prices` (ticker,date) · `dividends` · `short_interest` ·
`fundamentals_history` (ticker,as_of) ·
`earnings_surprises` · `macro` (date) ·
`def14a_llm` (ticker,accession_number) · `sec13f_hr` · `cusip_ticker_map` ·
`sp500_tickers` (ticker, + name/cik/sector/industry — also the ticker→CIK source) · `google_trends` ·
`wiki_pageviews` · `ticker_embeddings` · `cube` (ticker,date,target_horizon) ·
`predictions` · `cube_signal` · `macro_asset_prices` (date — long-history multi-asset
allocation series: equity/gold/energy/bond-TR/cash/FX/VIX).

Access is always via `context.store` (`DataStore.load/save/replace/existing_dates`);
new columns auto-add via `ensure_columns`. Non-tabular artifacts (models, plots,
`sec_bulk_cache/` JSON + 13F zips, filing text) stay on disk under `data/`.

## Data reality (free-source limits)
- **SEC XBRL is the fundamentals backbone** — genuine point-in-time history keyed on filing date. Concepts are *coalesced* across candidate tags (filers split e.g. `Revenues`↔`RevenueFromContractWithCustomer` at ASC-606, `NetIncomeLoss`↔`ProfitLoss`); operating income is derived (gross − SG&A − R&D) when a filer doesn't tag it.
- **Sector-specific line items** (bank NII, insurance premiums/claims, REIT rental income, energy DD&A…) are extracted and turned into sector KPIs (NIM, combined ratio, FFO, …), gated by availability and normalized at the GICS industry-group level.
- **Forward P/E and 13F** accrue point-in-time going forward (yfinance/13F have no clean back-history for those); features build up over successive runs.
- **13F** is a long-only quarterly snapshot with a 45-day filing lag — separated into stock/call/put/debt; institutional "moves" come from quarter-over-quarter share deltas, not value deltas.

## Testing
```bash
poetry run pytest tests/ -v -s
```
Tests mix synthetic known-truth (for math) with real cached data (for coverage);
each prints a sanity-check conclusion. See `CLAUDE.md` for conventions.
