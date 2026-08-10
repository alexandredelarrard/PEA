
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
├── configs/                # OmegaConf configs: configs.yml, build_cube.yml, modellling.yml, models/, strategy/, portfolio.yml
├── data/                   # NON-tabular artifacts ONLY: sec_bulk_cache/, sec_*/ zips, models/, diagnostics/, sector_peers.json
├── sql/schema.sql          # Auto-generated DDL executed on DB initialization
├── docker-compose.yml      # PostgreSQL 16 service & persistent storage
├── scripts/                # Schema generator & migration utilities
├── src/
│   ├── constants/constants.py  # Global literals (date formats, URLs, thresholds) — SINGLE source of truth
│   ├── context.py              # Context initialization (configs, logger, DB .store, artifact .paths)
│   ├── data_store/             # THE data layer, and the only SQL in the repo:
│   │                           #   schema.py (the table registry: Tables.<name> + pk /
│   │                           #   date_col / projection), store.py (DataStore),
│   │                           #   ddl.py (sql/schema.sql), errors.py
│   ├── data_extract/           # StepExtractAllData & fetchers (utils/{prices,fundamentals,structure,behavioral,common}/)
│   ├── data_peers/             # StepDeducePeers (return correlation & OpenAI embedding-based peers)
│   ├── data_aggregate/         # StepBuildCube super-step -> 7 sub-steps in transformers/,
│   │                           #   one cube_part_* table each; shared layer in utils/common/
│   │                           #   (price_frames, pit, panel, xs, frames, prices, parts, incremental)
│   ├── modelling/              # Model/signal engines per strategy: long_short/, trend/, long_book/
│   ├── strategies/             # Self-contained strategy steps (step_ls, step_long_book, step_trend) & analytics
│   ├── portfolio/              # StepPortfolio (ERC blending & global vol scaling), StepStrategyMoves (live trading ledger)
│   ├── dags/                   # Airflow DAGs (data_extraction -> data_aggregation -> strat_prediction)
│   └── utils/                  # Cross-package shared utilities (db, step, config, trend, risk_parity, polite_http, string)
├── tests/                  # Mirrors src/ layout; conftest.py contains shared DB fixtures
├── app/                    # Streamlit visual dashboard
├── main.py, README.md, pyproject.toml
```

---

## The Step Pattern

Every `src/` subfolder owns a `step_*.py` orchestrator. This is the project's primary structural convention.

### Execution Rules
1. Inherit from `Step` (`src.utils.step`) to access `self._context` and `self._config`.
2. Always call `super().__init__(context=context, config=config)` inside `__init__`.
3. Expose `run()` as the sole public entry point.
4. **No cross-folder imports between `src/` subfolders** except through `src/utils/`.
5. Fetchers read/write tabular data via `self._context.store` (never write parquet files).
6. Super-steps compose child steps and resolve shared inputs once (e.g., universe tickers).

```python
from omegaconf import DictConfig
from src.context import Context
from src.utils.step import Step

class StepExample(Step):
    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._child_step = ChildStep(context=context, config=config)

    def run(self, tickers: list[str]) -> None:
        self._child_step.run(tickers=tickers)

---

## Code & Naming Conventions
Constants First: Check src/constants/constants.py before defining new column names, keys, or URLs. Add missing literals to constants.py before referencing them in code.

Logging: Use self._context.logger for all output (never use print() or raw logging.getLogger()).

Configuration: Access configuration values strictly via self._config.<section>.<key>.

Typing: Provide complete type annotations for every function signature.

Imports: Place all import statements at the top of the Python file.

---

## Data & Database Conventions
Database I/O: Read and write tabular data exclusively via self._context.store. DataStore reads with load / iter_load (+ exists, columns, distinct, bounds, max_date, row_count) and writes with save / replace / append_tail / bulk_seed / delete / drop / ensure_columns. Filter server-side: where= (equality / IN / IS NULL / store.NOT_NULL), since=, until=, columns= or project=True.

Never bypass the facade: no sqlalchemy import, no pd.read_sql / to_sql, no store.engine outside src/data_store/ (tests/data_store/test_store_boundary.py enforces this). If a query cannot be expressed, add the capability to DataStore instead.

Never read a full large table: always project and scope (cube is ~26 GB, sec13f_hr ~21.7M rows); use iter_load for cube-sized reads.

load raises TableMissingError / TableEmptyError on a missing or empty table. Pass optional=True only where finding nothing is legitimate (a resume check on a cold DB), then branch on `is None`.

Artifacts: Store non-tabular data (models, plots, raw JSONs) in paths defined by context.paths.

Point-in-Time Integrity: Resume extraction using the maximum entity date stored in the DB. Save progress per entity. Lag features by filing date to prevent forward-looking bias.

XBRL Processing: Union candidate XBRL tags per period rather than relying on the first match. Cast boolean values to numeric flags (1.0/0.0).

Schema Registration: Register new logical tables in src/data_store/schema.py as a Table (name, pk, date_col, ...) and reference them as Tables.<name>. Table names live ONLY there — never as a string literal and never as a *_TABLE constant in src/constants/constants.py.

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

### Mandatory Sanity Checks
Unit tests must validate logical/financial sense in addition to structural integrity, ending with a printed summary:

```python
def test_momentum_feature(sample_prices):
    result = compute_momentum(sample_prices, window=20)

    # Structural assertion
    assert "momentum_20d" in result.columns
    assert result["momentum_20d"].isna().mean() < 0.1

    # Financial logic sanity check
    winners = result.nlargest(5, "momentum_20d")
    losers = result.nsmallest(5, "momentum_20d")
    assert (winners["momentum_20d"] > 0).all()
    assert (losers["momentum_20d"] < 0).all()

    # Mandatory printed conclusion
    print("\n=== SANITY CHECK: momentum_20d ===")
    print(f"  Range : [{result['momentum_20d'].min():.4f}, {result['momentum_20d'].max():.4f}]")
    print(f"  NaN % : {result['momentum_20d'].isna().mean():.1%}")
    print("  ✓ Winners positive, losers negative — direction is correct")
```

### Test Commands
```bash
pytest tests/path/to/test_file.py                       # whole file
pytest tests/path/to/test_file.py::test_function -v -s   
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

## Documentation Synchronization
Maintain strict consistency across documentation files:

- Keep AGENTS.md, CLAUDE.md, and README.md updated in the same change whenever system architecture, data models, or conventions evolve.

- Propose any new shared convention and obtain approval before editing AGENTS.md or CLAUDE.md.