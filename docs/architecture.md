# Architecture

Scope: where code lives, how the stages are wired, and the entry points. For data-layer rules see
[data_conventions.md](data_conventions.md); for tables see [data_schema.md](data_schema.md).

`stock_pick_strat` — a long/short S&P 500 stock-picking pipeline. It extracts market /
fundamental / governance / ownership / alt-data into PostgreSQL, builds a point-in-time,
peer-relative feature **cube**, trains cross-sectional ensembles, and blends strategy sleeves into
one book. 350 Python files in `src/`.

Stack: Python **3.13**, OmegaConf (`DictConfig`), pandas, SQLAlchemy 2.0 + PostgreSQL 16 (Docker),
LightGBM / scikit-learn / SHAP, OpenAI (embeddings + DEF 14A), local FinBERT (torch), click, pytest.
Airflow runs in its own container on Python 3.12 (see [runbook.md](runbook.md)).

## Directory layout

The package lives at the **repo root** (it used to be under `stock_pick_strat/`; any doc or
docstring still saying otherwise is stale).

```
├── configs/                  # OmegaConf YAML, merged by top-level key — see config.md
├── data/                     # NON-tabular artifacts ONLY (models, plots, SEC zip caches,
│                             #   transcripts, peer dict). Created on first run; not in git.
├── docs/                     # this documentation set
├── sql/schema.sql            # generated DDL, applied by Postgres initdb on an EMPTY volume
├── scripts/                  # generate_schema_sql, diagnostics, one-off reports
│   └── dod/                  #   definition-of-done report generators — see definition_of_done.md
├── reports/                  # definition-of-done reports, TRACKED in git; ONE FOLDER PER DAY
│   ├── YYYY-MM-DD/           #   <slug>__<TYPE>.md + assets/<slug>/ (plots copied out of data/)
│   └── baselines/            #   data_profile.json — persistent, NOT per-day (gates D2/D3/D5)
├── .claude/                  # hooks/ (the Stop gate), skills/ (dod-*-report), commands/, settings.json
├── specs/, coverage_bar/     # design notes / coverage tracking (not executed)
├── docker-compose.yml        # `db` (pipeline Postgres) + Airflow scheduler/webserver/metadata-db
├── main.py                   # scratch driver: instantiate a step, uncomment its .run()
├── src/
│   ├── cli.py, __main__.py       # `python -m src <package> <command>` plugin dispatcher
│   ├── context.py               # Context: config, logging, .store (DB), .paths (artifacts)
│   ├── constants/constants.py   # 927 lines — THE global literals (URLs, formats, thresholds)
│   ├── data_store/              # the ONLY SQL in the repo: schema.py, store.py, ddl.py, errors.py
│   ├── data_extract/            # StepExtractAllData + 4 sub-steps + fetchers
│   ├── data_peers/              # StepDeducePeers (return-corr + OpenAI-embedding peer baskets)
│   ├── data_aggregate/          # StepBuildCube + 7 sub-steps -> cube_part_* -> cube
│   ├── modelling/               # long_short/, trend/, long_book/ — signal engines
│   ├── strategies/              # self-contained sleeves + analysis plots
│   ├── portfolio/               # StepPortfolio (ERC blend), StepStrategyMoves (trade ledger)
│   ├── validate/                # THE home for validation code, all domains. Part 2 of 3
│   │                            #   (extract -> VALIDATE -> bugfix). CHECK_REGISTRY drives
│   │                            #   35 checks over 3 tiers -> fundamentals_check. Read-only
│   │                            #   everywhere else; gates nothing. See its README.md
│   ├── dags/                    # 4 Airflow DAGs
│   └── utils/                   # cross-package shared code (step, db, config, universe, http…)
├── tests/                    # mirrors src/; 160 test files; conftest.py has the shared fixtures
└── app/app.py                # Streamlit portfolio dashboard (reads pre-trained artifacts)
```

## The Step pattern

Every `src/` subfolder owns a `step_*.py` orchestrator. This is the primary structural convention.

Base class: [src/utils/step.py](../src/utils/step.py) — 36 lines. It gives you exactly four things:
`self._config`, `self._context`, `self._log` (a stdlib logger), `self._today`.

```python
class StepCubeExtras(Step):
    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)   # ALWAYS call this
        self._cfg = config.build_cube
        self._store = context.store

    def run(self, full: bool = False) -> None:             # the sole public entry point
        ...
```

Rules:

1. Inherit `Step`; always call `super().__init__(context=context, config=config)`.
2. `run()` is the only public method. Logic lives in same-folder private helpers or composed
   sub-steps.
3. **No cross-folder imports between `src/` subfolders.** Shared logic goes in `src/utils/`.
   (A sub-package's own `utils/` — e.g. `data_aggregate/utils/` — is in-folder and fine.)
4. A **super-step** composes sub-steps of the same folder in `__init__` and resolves shared inputs
   once in `run()` (e.g. the ticker universe).
5. Class name `Step` + PascalCase; file `step_<snake_case>.py`.
6. `Strategy` sleeves are the one deliberate exception: they implement
   [src/strategies/base.py](../src/strategies/base.py)`::Strategy.run(inputs) -> StrategyResult`
   rather than `Step`. See [modelling.md](modelling.md).

## The pipeline

```
StepExtractAllData              src/data_extract/step_extract_all_data.py
  ├─ StepExtractStructure         DEF 14A (LLM + edgartools), 8-K, 13D, filing text
  ├─ StepExtractFundamentals      SEC XBRL per-filing facts, earnings surprises, macro, notes
  ├─ StepExtractPrices            prices+dividends, short interest, FTD, 13F, superinvestors,
  │                               market/macro-asset series
  └─ StepExtractBehavioral        Wikipedia pageviews, Google Trends, earnings-call transcripts
StepDeducePeers                 src/data_peers/ -> peer dict JSON at paths["SECTOR_PEERS_PATH"]
StepBuildCube                   src/data_aggregate/ -> 8 cube_part_* tables -> `cube`
StepModelling                   src/modelling/long_short/step_train.py -> predictions, cube_signal
StepPortfolio                   src/portfolio/ -> blend the sleeves vs SP-hold
StepStrategyMoves               src/portfolio/ -> the `strategy` trade ledger
```

Note the extraction order in `StepExtractAllData.run()`: **structure → fundamentals → prices →
behavioral**, not the docstring's numbering.

### StepBuildCube: 7 sub-steps, one part table each

| Sub-step | Writes | Price fields it reads (`_FIELDS`) | Contents |
|---|---|---|---|
| `StepCubePrices` | `cube_part_prices` | `prices_macro` (market series, for the trading calendar) — (**the only reader of raw `prices`**) | pivot, trading calendar, returns, universe restriction, peer sector returns |
| `StepCubeTarget` | `cube_part_targets`, `cube_part_betas` | `close, ret` | factor panel → rolling betas → multi-horizon factor-neutral labels |
| `StepCubeFundamentals` | `cube_part_fundamentals` | (via `PitFrames`) | fundamental, sector-KPI, earnings, workforce, dividend |
| `StepCubeMomentum` | `cube_part_momentum` | `close, open, high, low, volume, ret, sector_ret` | momentum, vol, trend, lottery, liquidity, seasonality, MACD/RSI/ATR |
| `StepCubeText` | `cube_part_text` | `close` | earnings-call FinBERT sentiment + embedding KPIs (both **stream** their sources) |
| `StepCubeExtras` | `cube_part_extras` | `close, volume` | governance, 13F, elite 13F, insider, short interest, attention |
| `StepAssembleCube` | `cube` | — | read the parts → composites → per-horizon streamed write |

**Memory invariant.** Each sub-step keeps its heavy frames LOCAL to `run()` and reads the price
grid back from `cube_part_prices` **projected to its declared `_FIELDS`**. Peak memory is the
largest single sub-step, not the sum. **Never stash a frame on `self`.**

**One code path, two drivers.** `StepBuildCube.run()` drives the same seven objects that
[src/data_aggregate/cli.py](../src/data_aggregate/cli.py) exposes as seven commands and the Airflow
DAG chains as seven tasks.

**Part registry** — [src/data_aggregate/utils/common/parts.py](../src/data_aggregate/utils/common/parts.py)
is the single source of truth for part names, CLI commands, incremental warm-ups and per-group
binding look-backs. The CLI, the DAG chain (`PART_COMMANDS`) and `cube-status` all derive from it.
**Do not hand-list parts anywhere.** `tests/dags/test_dag_matches_part_registry.py` and
`tests/data_aggregate/test_part_registry.py` enforce this.

**Collision protection** — `PanelMerger.add` raises `FeatureCollisionError` on a duplicate feature
name, naming the panel that already owns it. Applied to both the per-step merge and the cross-part
merge in assemble.

### Shared aggregation layer (`src/data_aggregate/utils/common/`)

| Module | Contract |
|---|---|
| `price_frames.py` | the `PriceFrames` contract + `load_price_frames` / `load_trading_calendar` |
| `pit.py` | point-in-time accessors + the memoizing `PitFrames` (one `fundamentals_history` read shared) |
| `panel.py` | peer-relative panel construction |
| `xs.py` | **ONE** cross-sectional z + rank. The 3.0 / 4.0 / 8.0 clips are three deliberate policies (`XS_CLIP_LABEL` / `_CHARACTERISTIC` / `_PEER`); `clip` is a **required** argument so nobody can unify them by omission |
| `frames.py` | `ratio` (frames, column-intersecting) / `safe_div` (series, None-tolerant, does **not** strip inf) / `sanitize` / `downcast_float32` |
| `prices.py` | momentum, trailing vol, forward windows, `price_column_returns` |
| `incremental.py` | part lifecycle + the full-vs-incremental decision (`plan_window`, `write_part`, `COLUMNS_CHANGED`) |
| `parts.py`, `part_status.py` | the part registry and the DAG status gate (its dict shape is a contract) |
| `panel_merge.py`, `capital.py`, `sector_gates.py`, `gics.py`, `peers_io.py`, `sources.py`, `data_utils.py` | see each module docstring |

Domain builders live in `utils/{target,fundamentals,momentum,text,extras,assemble}/`.
`fundamental_features._derived_fields` is a thin composition over ~30 per-block
`_*_fields(daily, …) -> dict` builders.

## Entry points

**1. `main.py`** — a scratch driver. Steps are instantiated and the wanted `.run()` is uncommented.
Not a production path; it also carries a long TODO/results scratchpad at the bottom.

**2. The CLI** — `python -m src <package> <command> [-c ./configs] [-t AAPL,MSFT] [-F]`.
[src/cli.py](../src/cli.py) auto-discovers any `src/*/cli.py` as a sub-command group. Five groups
exist: `data_extract` (~22 per-source commands), `data_peers`, `data_aggregate` (9),
`modelling` (3), `portfolio` (2). Shared click options are in
[src/constants/command_line_interface.py](../src/constants/command_line_interface.py). This is what
the DAGs call. Full list in [runbook.md](runbook.md).

**3. Airflow DAGs** (`src/dags/`) — the production schedule:

| DAG | Schedule | Tasks |
|---|---|---|
| `data_extraction` | daily 01:00 | `seed_universe` → all fetchers in parallel (pool-throttled: `sec_bulk`/`sec_api`/`scrape` = 2 slots each) → `extraction_complete` → trigger `data_aggregation` |
| `data_aggregation` | triggered | `deduce_peers` → the 6 build commands **strictly sequential** (`max_active_tasks=1`, so peak memory = the largest single step) → `assemble_cube` → `cube_status` → trigger `strat_prediction` |
| `strat_prediction` | daily 06:00 | `predict` (→ `predictions_latest`) → `strategy_moves` (→ the `strategy` ledger) |
| `modelling` | weekly, Sat 02:00 | `train_model` (holdout) → `backtest_portfolio` (OOS) → `full_train` (ALL history, no holdout) |

Training is weekly and prediction is daily, so a freshly rebuilt cube is scored every night without
waiting for a retrain; `strat_prediction` reads back the artifacts from the last `full_train`.
Aggregation is triggered on `ALL_DONE`, so a failed fetcher does not block the prediction build.

**4. `app/app.py`** — Streamlit dashboard (`streamlit run app/app.py`). Assumes pre-trained models.

## Current work-in-progress state (verified 2026-08-17, branch `dev`)

Read this before assuming a stage runs end-to-end:

- **[src/data_aggregate/step_build_cube.py:78-85](../src/data_aggregate/step_build_cube.py#L78-L85)
  has 6 of its 7 sub-steps commented out** — `run()` currently executes only
  `self._target.run(full=full)`. `main.py` is likewise set to `StepBuildCube(...).run(full=True)`
  with every other step commented. This is a deliberate in-progress state, not a bug to "fix"
  unless asked; the CLI sub-commands still drive each sub-step individually.
- The live database has **no `prices` table and no cube/prediction/strategy tables at all** — see
  [database.md](database.md). Any code path starting from `prices` cannot run locally as-is.
- `data/` does not exist in a fresh checkout; `Context` creates it on first run.
