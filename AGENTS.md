
## Project overview
`stock_pick_strat` — Python quant long/short stock-picking pipeline: extract market /
fundamental / governance / alt-data into **PostgreSQL**, engineer point-in-time peer-relative
features (the "cube"), train ML models with time-series CV, run portfolio backtests, expose
results via `app/`.

Stack: Python, OmegaConf (`DictConfig`), pandas, **SQLAlchemy + PostgreSQL (Docker)**,
OpenAI (DEF 14A extraction + embeddings), pytest.

> Keep this file and `CLAUDE.md` in sync with the code — see "Keep the docs current" below.

---

## Directory layout
```
stock_pick_strat/
├── configs/*.yml           # OmegaConf configs (merged by top-level key): configs.yml, build_cube.yml,
│                           #   modellling.yml (model:/train:), models/{lgbm,linear,random_forest},
│                           #   strategy/{strategy_ls,strategy_trend,strategy_long_book}.yml, portfolio.yml
├── data/                   # NON-tabular artifacts ONLY: sec_bulk_cache/ (companyfacts JSON + 13F zips),
│                           #   sec_{financial_statements,financial_notes,fails_to_deliver}/ (bulk-set zips;
│                           #   notes sets ~26GB @ 15y), output/{models,diagnostics}, sector_peers.json
├── sql/schema.sql          # generated DDL (CREATE TABLE IF NOT EXISTS + PKs), run on DB init
├── docker-compose.yml      # Postgres 16 + persistent volume   (+ Dockerfile)
├── scripts/                # schema generator, parquet→DB migrator
├── src/
│   ├── constants/constants.py  # global literals (date formats, SEC URLs) — SINGLE source of truth
│   ├── context.py              # Context: config, logging, env, `.store` (DB), `.paths` (artifacts only)
│   ├── data_store/             # DB layer: DataStore (store.py), schema_registry, schema_sql, io
│   ├── data_extract/           # step_extract_* : super-step + prices/fundamentals/structure/behavioral
│   │   └── utils/{prices,fundamentals,structure,behavioral,common}/   # fetchers
│   ├── data_peers/             # step_deduce_peers  (return-corr + OpenAI-embedding peers)
│   ├── data_aggregate/         # step_build_cube — peer-relative feature panels → `cube`
│   ├── modelling/              # per-STRATEGY model/signal: long_short/ (step_train=StepModelling + utils/),
│   │                           #   trend/ (signal.py::trend_book + utils/), long_book/ (allocation.py)
│   ├── strategies/             # self-contained per-strategy STEPS (base.Strategy.run(PortfolioInputs)):
│   │                           #   step_ls / step_long_book / step_trend + utils/ (strategies_opt, metrics,
│   │                           #   plot_analysis, accuracy) + analysis/ (per-sleeve IC/neutrality/corr plots)
│   ├── portfolio/              # step_portfolio — runs the configured sleeves, blends by risk-parity/ERC
│   │                           #   (dynamic $-allocation) + global vol/leverage; analysis.py (sleeve corr)
│   ├── utils/                  # shared cross-folder helpers: db, step, config, trend.py, risk_parity.py, …
│   └── cli.py
├── tests/                  # mirrors src/ ; conftest.py holds shared real-data fixtures
├── app/                    # Streamlit app
├── main.py, README.md, pyproject.toml
```

---

## The Step pattern

Every `src/` subfolder owns a `step_*.py` orchestrator (a super-step may compose sub-steps
in the same folder, as `data_extract/` does). This is the project's most important structural
convention — never deviate from it.

**Rules:**
- Inherits from `Step` (`src.utils.step`) to get `self._context` and `self._config`
- `__init__` always calls `super().__init__(context=context, config=config)`
- `run()` is the only public entry point — it calls helpers from the same folder
- Never call functions from another `src/` subfolder inside a step — use `src/utils/` instead
- Fetchers **save to the DB** via `self._context.store` and return the frame (they no longer write parquet)
- A super-step composes sub-steps and resolves shared inputs once (e.g. the ticker universe)

```python
from omegaconf import DictConfig

from src.context import Context
from src.utils.step import Step
from src.data_extract.utils.prices.fetch_prices import get_sp500_tickers
from src.data_extract.step_extract_prices import StepExtractPrices
from src.data_extract.step_extract_fundamentals import StepExtractFundamentals
# … structure, behavioral

class StepExtractAllData(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._prices = StepExtractPrices(context=context, config=config)
        self._fundamentals = StepExtractFundamentals(context=context, config=config)
        # … structure, behavioral

    def run(self) -> None:
        tickers = get_sp500_tickers(self._context) + self._config.data_extract.other_tickers
        self._prices.run(tickers=tickers)
        self._fundamentals.run(tickers=tickers)
        # … structure, behavioral
```

---

## Naming — constants before anything else

Before naming a column, DataFrame key, file path constant, or config key:
1. Check `src/constants/*.py`
2. If it exists → import and use it, never hardcode
3. If it doesn't exist → add it to the right constants file first, then import

---

## Code conventions

- `self._context.logger` for all logging — never `print()` or `logging.getLogger()` directly
- `self._config.<section>.<key>` for all config values — never hardcode
- Env is loaded by `Context`; business logic reads it through the context (infra like `utils/db.py` / the LLM extractor may read `os.getenv` for secret keys)
- Full type annotations on every function signature
- No cross-folder imports between `src/` subfolders except via `src/utils/`
- Global literals (date formats, SEC/API URLs, env-var keys, thresholds) live in `src/constants/constants.py` — never hardcode inline

---

## Data / DB conventions

- **DB-only I/O.** All tabular data reads/writes go through `self._context.store` (`DataStore`):
  `load` / `save` (upsert on PK) / `replace` (full rebuild) / `existing_dates`. Never read/write
  parquet for tables. New DataFrame columns auto-add to the table via `ensure_columns`.
- `context.paths` holds ONLY non-tabular artifacts (models, plots, `sec_bulk_cache/*.json`, filing text).
- **Point-in-time + incremental.** Fetchers resume from the DB's max date per entity and save
  **per entity**, so an interrupted run never loses expensive work (LLM / 13F / API calls); lag
  by filing date so features are leak-free.
- **Cache large downloads** to disk (companyfacts JSON, 13F zips); re-download only when missing.
- **Coalesce alternative XBRL tags** — union candidate tags per period, don't take the first present
  (`Revenues`↔`RevenueFromContractWithCustomer`, `NetIncomeLoss`↔`ProfitLoss`, equity ±NCI).
- **Booleans → numeric 1.0/0.0 flags** so they're usable as model features.
- New logical tables are declared in `src/data_store/schema_registry.py` (name → PK + incremental date col).

---

## Testing conventions

### Use real data, small sample
```python
df = load_real_data(context).head(100)   # time-series
df = load_real_data(context).sample(100, random_state=42)   # non-temporal
```
Never mock DataFrames with synthetic data for **feature / economic** tests — real data catches
NaNs, delisted tickers, corporate actions, and weekend gaps that mocks never will.

**Exception — parsing / derivation math** (XBRL concept extraction, TTM, ratio formulas, KPI
math): use synthetic **known-truth** fixtures, because you can only assert a value is correct if
you know the true inputs. Pair these with a real-data coverage check against the cached source
(e.g. build a real ticker's history and confirm the previously-missing field now populates).

### Write sanity checks, not just assertions
Every new feature test must include checks that validate economic or logical validity,
not just structure:
```python
def test_momentum_feature(sample_prices):
    result = compute_momentum(sample_prices, window=20)

    # Structural
    assert "momentum_20d" in result.columns
    assert result["momentum_20d"].isna().mean() < 0.1

    # Sanity — does the value make financial sense?
    winners = result.nlargest(5, "momentum_20d")
    losers  = result.nsmallest(5, "momentum_20d")
    assert (winners["momentum_20d"] > 0).all(), "Top momentum stocks must have positive returns"
    assert (losers["momentum_20d"]  < 0).all(), "Bottom momentum stocks must have negative returns"

    # Conclusion — printed at test end, always
    print("\n=== SANITY CHECK: momentum_20d ===")
    print(f"  Range : [{result['momentum_20d'].min():.4f}, {result['momentum_20d'].max():.4f}]")
    print(f"  NaN % : {result['momentum_20d'].isna().mean():.1%}  (expected ~{20/len(result):.1%} for window=20)")
    print(f"  ✓ Winners positive, losers negative — direction is correct")
```

The printed conclusion is mandatory. It states what was checked and why the result is valid.
Work is not done until this conclusion is written and passes.

### Run commands
```bash
pytest tests/path/to/test_file.py                        # whole file
pytest tests/path/to/test_file.py::test_function -v -s   # single test, -s to see print output
```

### Fixtures
Shared real-data loaders live in `tests/conftest.py`, scoped to `session`, and read from the DB
via the store:
```python
@pytest.fixture(scope="session")
def sample_prices():
    prices = _store().load("prices")
    if prices.empty:
        pytest.skip("prices table is empty")
    return prices
```

### File structure
`tests/` mirrors `src/`: `tests/data_extraction/test_fetch_prices.py`

---

## Modelling requirements

Every model must have all three before it is considered complete:
1. **TimeSeriesSplit CV** — never a single train/test split; `n_splits` and `gap` from config
2. **SHAP analysis** — feature importance on the validation set, top-N logged
3. **Printed metric summary** — OOF metrics per fold + aggregated before declaring done

---

## Config conventions

`read_config` merges every `configs/**/*.yml` by its top-level key. Data/model stages:
`configs.yml` (`data_extract`), `build_cube.yml`, `modellling.yml` (`model:`/`train:`),
`configs/models/*` (per-family hyperparams). Strategy/portfolio stack: `configs/strategy/*.yml`
(per-sleeve run params) + `configs/portfolio.yml` (sleeve set + global vol/leverage/capital/fees).
All numeric hyperparameters, window sizes, and thresholds live in config — never hardcoded.

---

## Keep the docs current

`AGENTS.md` and `CLAUDE.md` are the source of truth for how this repo is built — they must
track the code, not drift from it.

- **Review both regularly** and whenever the structure or conventions change (a new step / table /
  package, a moved or renamed module, a new data source, a changed data-flow), update them in the
  **same change** so they never go stale. `README.md` too when the pipeline stages change.
- When a **generic, reusable convention** emerges from a request, **propose it and ask for
  confirmation before writing it** into `AGENTS.md` / `CLAUDE.md` — don't edit these docs unprompted.
- Keep the two consistent: `CLAUDE.md` is the concise rulebook, `AGENTS.md` the fuller guide with
  examples; a convention added to one should be reflected in the other.