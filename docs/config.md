# Configuration

Scope: where every knob lives and how the merge works. Rule: **all numeric hyperparameters, window
sizes and thresholds live in config — never hardcoded.**

## How the config is assembled

[src/utils/config.py](../src/utils/config.py)`::read_config(path)`:

1. loads `configs/configs.yml` (required — missing it raises `FileNotFoundError`),
2. globs `configs/**/*.yml` recursively and `OmegaConf.merge`s each one in, skipping `configs.yml`.

So **each file contributes its own distinct top-level key**, and the whole tree is flat at the top.
Adding a model family or a strategy sleeve is a drop-in new file — no loader change.

> **Glob order is filesystem order, not declared order.** Two files defining the same top-level key
> would merge non-deterministically. Keep one owner per key.

`get_config_context("./configs", use_cache=…, save=…)` in
[src/context.py](../src/context.py) then applies `dictConfig(config.logging)`, seeds RNGs via
`set_seed(config)`, and builds the `Context` (paths → logging → `.env` → `.store`).

Access is always attribute-style: `self._config.<section>.<key>`. Never hardcode a value inline, and
never replace OmegaConf.

## The top-level keys

| Key | File | Owns |
|---|---|---|
| *(root)* `seed` | `configs.yml` | `seed: 4325` — the global RNG seed, read as `context.random_state` |
| `data_extract` | `configs.yml` | extraction scope & cadence |
| `local.paths` | `paths.yml` | `root: ./`, `data_store: data`, `logs: .log` — resolved into `context.paths` |
| `logging` | `logging.yml` | stdlib `dictConfig` tree; `formatters.file.format` also feeds the in-memory log buffer |
| `peers` | `peers.yml` | peer-basket construction |
| `build_cube` | `build_cube.yml` | **the largest config** — betas, targets, features, intrinsic DCF, self-history windows, output flags |
| `model`, `train` | `modellling.yml` | shared modelling config + the training window (note the triple `l` in the filename) |
| `linear`, `lgbm`, `random_forest` | `configs/models/*_modelling.yml` | per-family hyperparameters **and each family's own feature `columns`** |
| `strategy_ls`, `strategy_long_book`, `strategy_trend`, `strategy_eq_long_only` | `configs/strategy/*.yml` | per-sleeve construction params |
| `portfolio` | `portfolio.yml` | sleeve set + global vol / leverage / capital / fees |

## `data_extract` (configs.yml)

```yaml
data_extract:
  years_history: 15                  # equity price / filing history depth
  macro_years_history: 31            # `prices_macro`'s only window (~1995 →)
  notes_years_history: 15
  # NO other_tickers: the market/macro SERIES LIST is a registry, not a tunable — it cannot be
  # split from its symbol→name mapping, so it lives in constants.MACRO_PRICE_SERIES /
  # MACRO_FRED_SERIES. Those series land in `prices_macro`, never in `prices`.
  redundant_ticks: ["GOOG", "FOX", "NWS"]              # dual-class duplicates to skip
  refresh_universe: false            # true -> re-scrape sp500_tickers even when populated
  llm_model: "gpt-5-mini"            # DEF 14A structured extraction
  manifest_full_rescan_days: 30      # force a full EDGAR relist every N days (self-heal)
```

## `build_cube` (build_cube.yml)

The file is heavily commented with the *measured* justification for each number — read those
comments before changing a value; several encode a specific finding.

| Block | Keys | Notes |
|---|---|---|
| — | — | no `market_ticker`: the beta/benchmark series is `constants.MACRO_MARKET_SERIES` (`equity_tr`), a named row in `prices_macro` |
| `betas` | `window: 126`, `min_obs: 80`, `ridge_alpha: 1.5`, `ridge_alpha_market: 0.48`, `market_prior: 1.0`, `step: 1`, `ffill_limit: 21` | `window` was 63; at that length the cross-sectional slope of realized on ex-ante beta is only 0.43, so the hedge over-shot. `ridge_alpha` is a **ratio** (λ = α·N), so shrinkage is window-length invariant. The market shrinks toward 1.0, not 0 |
| `targets` | `horizons: [30, 60, 90]`, `primary_horizon: 60`, `labels: [rank, zscore, epsilon]`, `min_names: 20`, `neutralize_momentum: true`, `neutralize_log_mcap: true`, `vol_standardize: true` | Every label is finally projected cross-sectionally orthogonal to **every fitted loading + momentum + log market cap + GICS industry-group dummies, jointly and after the rank/zscore transform**. Subtracting β·factor alone is insufficient — a signal built from nothing but market beta earned rank-IC +0.073 (t +10.3) against the un-projected label. The loading list is **derived** from the fitted betas, not listed. `vol_63` deliberately stays OUT (it would zero the `vol_63` feature). **Column-naming contract**: each (label, horizon) pair becomes ONE cube column `target_<label>_h<horizon>` — `target_rank_h30`, `target_zscore_h60`, … — so these two lists define the cube's label column SET (9 here), not a `target_horizon` key. The format lives in exactly one place, `utils/assemble/cube.target_column` / `TARGET_COL_RE`; never re-spell it at a call site. ⚠ Editing either list forces a **full** rebuild of `cube_part_targets` (see the runbook) |
| `features` | `standardize_method: rank` | |
| `intrinsic` | `discount_rate: 0.10`, `terminal_growth: 0.025`, `years: 5`, `growth_cap/floor` | two-stage DCF; `terminal_growth` must be `< discount_rate` |
| `hist` | `window: 1260`, `min_periods: 252` | self-history window for `f_<yield>_vs_hist` |
| `institutionals` | `decay_halflife`, `superinvestor.{stale_quarters, selection}`, `coverage_break_pct: 0.50`, `excursion_lookback: 252` | `coverage_break_pct` is D17's coverage-discontinuity guard on the universe-wide 13F **filer count**; measured on the 2026-09-11 full build it fires on **11 quarters** and nulls their QoQ deltas — but 7 of those sit at or before the D16 delta floor and are already nulled by it, so it only removes something new on **4** (2023-12-31, 2024-03-31, 2025-06-30, 2025-09-30). The two broken quarters (463 and 254 filers against ~7,000) are HOLES whose levels are suppressed too. `excursion_lookback` caps the two insider excursion features and is the longest **bounded** look-back in the conditioning group — raising it means raising `cube_part_institutionals.warmup_trading_days` in `parts.py` (390), which `test_part_registry.py` enforces. ⚠ It is **not** the binding look-back of the whole part: the ownership group's `HOLDER_ACTIVE_DAYS` (378) is longer, and several families have no finite look-back at all (age-since-last-event, distinct-actors-to-date, new-holder, the expanding-window owner surprise). That part therefore does not rely on its warm-up — it computes over the full trading calendar and writes only the tail — so this knob is a research dial here, not an incremental-correctness one. See the runbook |
| `output` | `save_cube/panel/signal/cv_results/predictions/shap/models` | |

## `model` / `train` (modellling.yml)

```yaml
model:
  label_column: y
  target_type: rank            # which stored cube target: rank | zscore
  ensemble: [elasticnet, lgbm, random_forest]   # per-day-standardized preds are averaged
  weight_decay: {enabled: false, half_life_years: 10}
  cv: {n_splits: 5, embargo: null}              # null embargo -> primary_horizon
  diagnostics: {enabled: true, top_n_features: 50, shap_sample: 12000, pdp_grid: 30}
train:
  start_date: 2011-01-01
  end_date: 2022-01-01         # the holdout boundary; `full-train` ignores it
```

`target_type: rank` won on both mean IC and cross-fold stability. `weight_decay` is **off**: time
decay up-weighted recent folds and *widened* the cross-fold IC spread; off lifts fold-IR 1.85 → 2.50
(tighter, all-positive folds) at a small mean-IC cost. Re-enable at ~10y half-life for a mild
recency tilt.

Per-family files carry both hyperparameters **and** that family's feature `columns`, so each model
tunes its own variable set. `lgbm` additionally carries `categoricals: [industry_group]` and a
`monotonic.features` list of ~55 signed constraints (`+1`/`-1`). If you add a feature to
`lgbm.columns` and it has an economically unambiguous direction, add its monotone sign too —
`tests/modelling/test_monotone_constraints.py` checks the mapping.

## Strategy & portfolio

`configs/portfolio.yml` picks the sleeves and owns everything global:

```yaml
portfolio:
  sleeves: [ls_equity, eq_long_only, long_book]    # + trend_cta (currently commented out)
  start: "2022-01-01"; end: null
  starting_capital: 1000; sleeve_target_vol: 0.10
  fee_bps: 2.0; spread_bps: 8.0; risk_free_rate: 0.03
  scheme: erc                  # erc (correlation-aware) | inverse_vol
  cov_mode: ewma; cov_halflife: 63; vol_window: 63; rebalance_freq: 21
  portfolio_vol_target: 0.10; max_leverage: 2.0
  plot_analysis: true; save_trades: true
```

Each `configs/strategy/strategy_*.yml` holds only how that sleeve is **constructed**;
`target_ann_vol`, `starting_capital`, `risk_free_rate` and the window come *down* from the portfolio
as `PortfolioInputs`. Do not duplicate a portfolio-level value in a sleeve file — a sleeve may
override `fee_bps`/`spread_bps` and nothing else.

`strategy_ls.yml` is the one to know: `beta_neutral`/`sector_neutral` constraints, `pos_cap: 0.05`,
`gross_cap: 2.0`, turnover control (`step: 0.35`, `no_trade_band: 0.003`, `rebalance_freq: 21`),
`risk_model: covariance` with `cov_shrink`, and an `integer_shares` block (MILP projection to whole
shares, `long_fractional: true` for the retail case where only the short leg must be integer).

## Config conventions checklist

- A new numeric knob goes in the YAML that owns its stage, with a comment saying *why* that value.
- Access via `self._config.<section>.<key>`; provide a default in the YAML rather than in code.
- A new model family → a new `configs/models/<family>_modelling.yml` with its own top-level key.
- A new sleeve → a new `configs/strategy/strategy_<name>.yml`, its `config_key` on the `Strategy`
  subclass, and an entry in `STRATEGY_REGISTRY`.
- Keep global *literals* (URLs, date formats, tag lists, GICS names, plausibility bounds) in
  [src/constants/constants.py](../src/constants/constants.py), **not** in config. Config is for
  tunable numbers; constants are for facts about the world.
