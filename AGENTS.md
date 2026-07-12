
## Project overview
`stock_pick_strat` — Python quant stock-picking pipeline: extract market data, engineer features,
train ML models with time-series CV, run portfolio backtests, expose results via `app/`.

Stack: Python, OmegaConf (`DictConfig`), pandas, pytest.

---

## Directory layout
```
stock_pick_strat/
├── configs/*.yaml          # OmegaConf configs — one per pipeline stage
├── data/                   # All artifacts: raw data, processed files, saved models
├── src/
│   ├── constants/*.py      # Column names, file keys, categorical values — SINGLE source of truth
│   ├── data_aggregation/   # Feature engineering: loads from data/, saves enriched files back
│   ├── data_extraction/    # Raw data fetching: fetches, saves to data/, returns None
│   ├── modelling/          # Model training, TimeSeriesSplit CV, SHAP analysis
│   ├── post_processing/    # Portfolio construction, backtesting
│   ├── utils/              # Shared helpers — only cross-folder logic lives here
│   ├── cli.py              # CLI definitions for future Airflow DAGs
│   └── context.py          # Runtime object: logger, env, paths — imported by every step
├── tests
│   └── fixture/            # Shared real-data fixtures, conftest.py
├── app/                    # Coming: Streamlit or React app
├── README.md
└── pyproject.toml
```

---

## The Step pattern

Every `src/` subfolder owns exactly one `step_*.py` orchestrator. This is the project's
most important structural convention — never deviate from it.

**Rules:**
- Inherits from `Step` (`src.utils.step`) to get `self._context` and `self._config`
- `__init__` always calls `super().__init__(context=context, config=config)`
- `run()` is the only public entry point — it calls helpers from the same folder
- Never call functions from another `src/` subfolder inside a step — use `src/utils/` instead

```python
from omegaconf import DictConfig

from src.context import Context
from src.utils.step import Step
from src.data_extraction.fetch_prices import fetch_price_history
from src.data_extraction.fetch_fundamentals import fetch_fundamentals

class StepExtractAllData(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    def run(self):
        tickers = self._config.data_extract.tickers
        fetch_price_history(self._context, tickers=tickers)
        fetch_fundamentals(self._context, tickers=tickers)
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
- Never access `os.environ` directly — use the context object
- All paths from config — never build paths with f-strings or `os.path.join` in business logic
- Full type annotations on every function signature
- No cross-folder imports between `src/` subfolders except via `src/utils/`

---

## Testing conventions

### Use real data, small sample
```python
df = load_real_data(context).head(100)   # time-series
df = load_real_data(context).sample(100, random_state=42)   # non-temporal
```
Never mock DataFrames with synthetic data for feature tests — real data catches NaNs,
delisted tickers, corporate actions, and weekend gaps that mocks never will.

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
    print(f"  → Feature validated.")
```

The printed conclusion is mandatory. It states what was checked and why the result is valid.
Work is not done until this conclusion is written and passes.

### Run commands
```bash
pytest tests/path/to/test_file.py                        # whole file
pytest tests/path/to/test_file.py::test_function -v -s   # single test, -s to see print output
```

### Fixtures
Shared real-data loaders live in `tests/fixture/conftest.py`, scoped to `session`:
```python
@pytest.fixture(scope="session")
def sample_prices(context):
    return load_prices(context).head(100)
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

One yaml per pipeline stage: `data_extract.yaml`, `modelling.yaml`, `backtest.yaml`.
All numeric hyperparameters, window sizes, and thresholds live in config — never hardcoded.