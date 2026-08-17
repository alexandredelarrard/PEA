# Modelling, strategies & portfolio

Scope: the layers downstream of the cube. For the knobs, see [config.md](config.md).

## The three-layer split

```
src/modelling/     signal ENGINES      long_short/ (trained ensemble) · trend/ · long_book/
src/strategies/    SLEEVES             Strategy.run(PortfolioInputs) -> StrategyResult
src/portfolio/     BLEND + EXECUTION   StepPortfolio (ERC) · StepStrategyMoves (ledger)
```

A **sleeve is self-contained**: it does not depend on another sleeve or on a backtest step. It reads
its own config block, the `PortfolioInputs` handed down, and its own data.

| Sleeve (`name`) | `config_key` | Engine | What it is |
|---|---|---|---|
| `ls_equity` | `strategy_ls` | `long_short/` | market-neutral equity long/short alpha (OOS from the model's `train_end`) |
| `eq_long_only` | `strategy_eq_long_only` | `long_short/` | long-only top-N, no shorts — retail-viable |
| `long_book` | `strategy_long_book` | `long_book/allocation.py` | long-only multi-asset ERC + trend overlay + VIX regime tilt |
| `trend_cta` | `strategy_trend` | `trend/signal.py` | long/short multi-asset time-series momentum (crisis-alpha diversifier) |

Registry: [src/strategies/__init__.py](../src/strategies/__init__.py)`::STRATEGY_REGISTRY`.
`name` does **not** map mechanically to `config_key` (`"ls_equity"` → `"strategy_ls"`), which is why
`config_key` is declared on the class — a caller needing a sleeve's fees without instantiating it
reads it from there.

The `long_book` / `trend_cta` sleeves run off the long-history `macro_asset_prices` table (~1995→),
not off `prices`.

## Contracts

```python
@dataclass(frozen=True)
class PortfolioInputs:           # what the portfolio hands DOWN — never duplicated in a sleeve config
    capital, target_vol, start, end, fee_bps, spread_bps, risk_free_rate, analysis

@dataclass
class StrategyResult:
    name, returns, metrics          # daily NET returns (fractional, date-indexed); ann/sharpe/maxdd
    positions, trades, extra
    book_weights, book_prices       # the EXACT panels `trades` was built from
```

`book_weights` / `book_prices` are not redundant with `positions`. The daily ledger must re-size each
sleeve's book by the portfolio's **time-varying** per-sleeve ERC weight × leverage, and share counts
are a non-linear function of a time-varying capital — so it re-runs the blotter on
`book_weights * factor(t)` rather than scaling the resulting dollar figures. `long_book` reports its
*pre-leverage* allocation in `positions` while trading the levered panel; `ls_equity` reports `None`.

## Training (`StepModelling`)

[src/modelling/long_short/step_train.py](../src/modelling/long_short/step_train.py) (927 lines).
Three entry points, all via `python -m src modelling …`:

| Command | Method | Purpose |
|---|---|---|
| `train` | `run()` | holdout train on `train.start_date` → `train.end_date`; CV + diagnostics. `--train-start`/`--train-end` override the config for one run |
| `full-train` | `run(full_history=True)` | **production** refit on ALL history to the latest cube date, no holdout. This is the artifact prediction reads back |
| `predict` | `predict_latest(n_dates=1)` | score the latest cube date(s) → `predictions_latest`. Loads artifacts from disk; **no retraining** |

The weekly DAG runs `train` → `backtest` → `full-train` in that order deliberately: the per-horizon
IR blend weights and the backtest are only meaningful measured out-of-sample. Fitting on everything
first would leave both in-sample, inflating the reported Sharpe and mis-weighting the horizon blend
that production then uses.

### Every model needs all three before it is "done"

1. **`TimeSeriesSplit` CV** — never `KFold` (leaks the future), never a single split. `n_splits` and
   the embargo come from `model.cv`; the embargo defaults to `primary_horizon` when null, which is
   what stops a label's forward window overlapping the next fold's train set.
2. **SHAP** — computed on the **validation** rows, not train. Per booster member, per horizon.
3. **Printed OOF metrics** — per fold *and* aggregated, before you call it finished.

Linear members have no SHAP/PDP; boosters are resolved via `isinstance(m, lgb.Booster)` — note
**random_forest is also a `lgb.Booster`** (LightGBM `boosting='rf'`), so it round-trips as `.txt`
while `elasticnet` pickles to `.pkl`.

### Artifacts

`paths["MODELS_DIR"]` = `data/output/models/`:

- one file per (horizon, member) via `member_model_path` — the shared naming rule the backtest and
  the Streamlit app read back.
- **`metadata.json`** — the contract between training and everything downstream: `horizons`,
  `feature_cols`, the per-family and **per-horizon** resolved column lists, `categorical_cols`,
  `label_column`, `target_type`, `train_start`, `train_end` (the *actual* latest trained date on a
  full-history run, the config cutoff otherwise), `full_history`, and `train_ic_ir` (the per-horizon
  IC_IR blend weights, floored at 0).

Per-run diagnostics land in `data/output/diagnostics/<run_stamp>/h<H>/`: `pdp/`,
`shap_values.parquet` (the raw per-row matrix keyed by date+ticker, `shap_sample` rows),
`shap_importance.png/.csv`, `feature_importance.xlsx`, `kpis.json` per horizon, and a flat `kpis.csv`
for the run.

Tabular output goes to the DB, not files: `predictions` and `cube_signal` via `replace`,
`predictions_latest` via `replace`.

### Reading the cube without OOM

`step_train` never loads the cube whole. It resolves the union of the families' feature columns
against `store.columns(Tables.cube)`, then loads **one horizon at a time**, projected, filtered to
labelled rows, and downcast to float32. Keep that shape if you extend it.

## Strategy construction (`ls_equity`)

[src/strategies/step_ls.py](../src/strategies/step_ls.py) +
[utils/strategies_opt.py](../src/strategies/utils/strategies_opt.py).

Out-of-sample only — the sleeve starts at the model's `train_end`. Construction is a dollar / beta /
sector-neutral inverse-variance optimizer:

- `risk_model: covariance` = Ledoit-Wolf-shrunk **idiosyncratic covariance** (`w = Σ⁻¹·residual-alpha`),
  which jointly down-weights correlated names so shared risk is not double-counted. `diagonal` is
  plain inverse-variance (weight ~ 1/vol, ignores correlation). `cov_shrink ∈ [0,1]` blends toward
  the diagonal; `1.0` **is** the diagonal.
- Turnover control: partial `step` toward target, a `no_trade_band`, and `rebalance_freq`.
- Multi-horizon combine: `blend: ir` uses the `train_ic_ir` weights from `metadata.json` with
  `blend_shrink`.
- **Integer shares**: you cannot short a fraction of a share. When enabled, each rebalance's
  continuous target is projected to integer shares by a MILP (min tracking error s.t. integer +
  dollar/beta/sector-neutral within tolerance + gross within ±`gross_tol`), then held between
  rebalances. `long_fractional: true` is the retail case — fractional longs absorb the rounding and
  only the short leg is integer. **This needs realistic capital**: with small capital, positions fall
  below one share and the book collapses.

Note **size is neutralized in the label, not the book.** `strategy_ls.yml` enforces beta + sector
neutrality only, so a size tilt would flow straight into the live book — which is exactly why
`build_cube.targets.neutralize_log_mcap` is on.

## Portfolio

[StepPortfolio](../src/portfolio/step_portfolio.py) reads `configs/portfolio.yml`, runs each
configured sleeve on its own `configs/strategy/*.yml`, then blends the daily return streams by
risk-parity / ERC ([utils/blend.py](../src/portfolio/utils/blend.py)) — this **is** the dynamic
dollar allocation — plus one global vol target and leverage cap. It reports **per-strategy Sharpe vs
the global portfolio**. Each sleeve and the portfolio save plots under `data/output/<sleeve>/analysis/`
and `data/output/portfolio/`.

[StepStrategyMoves](../src/portfolio/step_strategy_moves.py) runs the same blend but reports it as a
tradeable ledger: it re-sizes the traded weight panel (`erc_weight × leverage`) before computing share
quantities, FIFO-matches round trips
([utils/positions.py](../src/strategies/utils/positions.py)`::round_trip_ledger`), and **upserts** to
`strategy` so a BUY row written weeks ago gains its `price_sold` / `pnl` on the day it closes.

## Reviewing a model change

The order the model work is expected to be reviewed in:

1. **Prune on evidence** — SHAP importance and coefficient p-values, not intuition.
2. **Sanity-check monotone directions** — a constraint with the wrong sign is worse than none.
3. **Validate the ensemble's predictions, not just its IC.** A better mean IC with a worse
   cross-fold IR is usually the wrong trade; the `weight_decay: false` decision is exactly that
   trade-off made explicitly.
4. Compare runs from `data/output/diagnostics/<run_stamp>/kpis.csv` — every run already writes a flat
   per-horizon KPI table, so a comparison needs no new harness.

Reference figures from a 2026-07-27 run (ensemble CV): h30 IC +0.0404 / IR +2.04 · h60 +0.0453 /
+1.87 · h90 +0.0443 / +1.45. Treat these as an order-of-magnitude sanity bar, not a target.
