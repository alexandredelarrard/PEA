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
  ├─ StepExtractFundamentals  SEC companyfacts history, forward-P/E snapshot, earnings, macro
  ├─ StepExtractStructure     employees, management, DEF 14A governance (LLM), SEC filings index
  └─ StepExtractBehavioral    Wikipedia pageviews, Google Trends
StepDeducePeers             # 2. peer baskets (return-corr + OpenAI embeddings)
StepBuildCube               # 3. peer-relative feature panels → `cube` table
StepModelling               # 4. cross-sectional models → `predictions`, `cube_signal`
StepBacktest                # 5. long/short backtest
```

### Data sources (all free / freemium)
| Domain | Source | Notes |
|---|---|---|
| Prices, dividends | yfinance | full daily OHLCV, no key |
| Fundamentals (history) | SEC EDGAR `companyfacts` (XBRL) | ~10-15y point-in-time, keyed on filing date |
| Forward P/E, market cap | yfinance `.info` | accrues daily into `fundamentals_snapshot` |
| Governance / comp / ownership | SEC **DEF 14A** proxy, parsed by **OpenAI** (structured output) | directors, CEO age/pay, board, say-on-pay, ownership |
| Employees, mgmt, filings | SEC EDGAR (10-K text, submissions) | |
| Institutional holdings | SEC **Form 13F** data sets | split by stock / call / put / debt; CUSIP→ticker via OpenFIGI |
| Macro | FRED | yields, curve, VIX, credit spreads, breakeven |
| Short interest | FINRA RegSHO | daily short-volume |
| Attention | Wikipedia pageviews, Google Trends | retail-attention alt-data |
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
`fundamentals_history` (ticker,as_of) · `fundamentals_snapshot` (ticker,as_of, accrues) ·
`earnings_surprises` · `macro` (date) · `employees_history` · `management_history` ·
`def14a_llm` (ticker,accession_number) · `institutional_holdings` · `cusip_ticker_map` ·
`sp500_tickers` / `cik_mapping` (ticker, + sector/industry) · `google_trends` ·
`wiki_pageviews` · `ticker_embeddings` · `cube` (ticker,date,target_horizon) ·
`predictions` · `cube_signal`.

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
