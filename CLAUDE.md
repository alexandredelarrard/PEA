## Risk zones — ask before editing

| File / directory | Why it's a risk zone |
|---|---|
| `src/context.py` | Imported by every step class — changes cascade everywhere |
| `src/utils/step.py` | Base class for all steps — changes break all inheritors |
| `src/constants/*.py` | Renaming any constant breaks all downstream references |
| `configs/*.yaml` | Structural changes must be mirrored in all consuming code |
| `src/data_store/*`, `sql/schema.sql` | DB access layer + DDL — a table/PK rename cascades to every read/write |
| `data/` artifacts & the Postgres volume | Model/plot files, `sec_bulk_cache/` JSON, and the DB volume — overwriting or dropping is not recoverable |

For any of the above: propose the change and wait for approval before editing.

---

## Code structure (`stock_pick_strat/`)

The pipeline is a chain of `Step` classes (base: `src/utils/step.py`), each with a
`run()`; `main.py` wires them. **All tabular data lives in PostgreSQL** (docker-compose
+ named volume) and is accessed via `context.store` (`DataStore`) — parquet is retired.

- `src/context.py` — `Context`: config, logging, env, `.store` (DB), `.paths` (non-tabular artifacts only).
- `src/constants/constants.py` — global literals (date formats, SEC URLs).
- `src/data_store/` — DB layer: `store.py` (`DataStore`: `load` / `save` upsert / `replace` / `ensure_columns` / `existing_dates`), `schema_registry.py` (logical table → PK + incremental date col), `schema_sql.py` (generates `sql/schema.sql`), `io.py`.
- `src/data_extract/` — `StepExtractAllData` super-step → 4 sub-steps: **prices** (prices+dividends, short interest, fails-to-deliver [SEC FTD → `fails_to_deliver`], 13F), **fundamentals** (SEC companyfacts history, earnings surprises → historical forward P/E, macro), **structure** (employees, DEF 14A LLM governance/executive-pay; both discover filings on demand via `edgar_fillings.list_filings`), **behavioral** (Wikipedia, Google Trends). Fetchers live under `utils/{prices,fundamentals,structure,behavioral,common}/`. Plus three **SEC bulk data sets** (entry points `fetch_insider_transactions` / `fetch_financial_statements` / `fetch_financial_notes`, all `(context, tickers)`): **insider transactions** (Forms 3/4/5 → `insider_transactions`), **Financial Statement Data Sets** (primary-statement num/sub XBRL → `pension_facts`), and **Financial Statement & Notes Data Sets** (adds the FOOTNOTE facts: numeric → `notes_num` [undimensioned `dimn==0` pension detail: PBO, plan assets, ABO, service cost, contributions, discount rate], and narrative TEXT → `notes_text` [high-signal notes, stored raw for later embedding/sentiment]). Each caches its zips under `data/sec_{financial_statements,financial_notes}/` (insider under `sec_bulk_cache/`) and is incremental by source period (skip periods already in the DB; re-parse cached zips only when the ticker universe grows, tracked via a `<table>_universe.json` sidecar). The notes sets are ~380MB/file (rolling quarterly `YYYYqQ` ↔ monthly `YYYY_MM`), scoped by the dedicated `data_extract.notes_years_history` knob (~26GB at 15y). Downstream: `pension_facts` + `notes_num` feed the off-BS pension leverage / footnote funded-status features, `insider_transactions` feeds the insider-trading cube signal (`insider_features.py` → `comp_insider`).
- `src/data_peers/` — `StepDeducePeers` (return-correlation + OpenAI-embedding peers).
- `src/data_aggregate/` — `StepBuildCube`: peer-relative feature panels (fundamental [incl. footnote pension: `pension_funded_ratio`, `pbo_to_mcap`, `pension_underfunding_to_mcap` from `notes_num`], sector KPIs, forward valuation, earnings, governance/executive-pay [from `def14a_llm`], employee, dividend, attention, institutional, short-interest) → `cube` table.
- `src/modelling/` — `StepModelling` → `predictions`, `cube_signal`.
- `src/post_processing/` — `StepBacktest`.
- Infra: `docker-compose.yml` (Postgres 16 + volume), `Dockerfile`, `sql/schema.sql`, `scripts/` (schema generator, parquet→DB migrator).

---

## Data / DB conventions

- **DB-only I/O.** Read/write tabular data through `context.store` (`load` / `save` / `replace` / `existing_dates`), never parquet. New DataFrame columns auto-add to the table via `ensure_columns`. Only non-tabular artifacts (models, plots, `sec_bulk_cache/*.json`, filing text) stay on disk under `context.paths`.
- **Point-in-time + incremental.** Fetchers resume from the DB's max date per entity, save **per entity** (an interrupted run must never lose expensive work — LLM / 13F / API calls), and lag by filing date so features are leak-free.
- **Cache large downloads** to disk (companyfacts JSON, 13F zips) and only re-download when missing.
- **Coalesce alternative XBRL tags** — union across candidate tags per period, don't take the first present (filers split concepts by era/scope: `Revenues`↔`RevenueFromContractWithCustomer`, `NetIncomeLoss`↔`ProfitLoss`, equity with/without NCI).
- **Booleans → numeric flags** (1.0/0.0) so they are usable as model features.

---

## Workflow for new features

1. Check `src/constants/*.py` before naming anything — add there first if missing
2. Check `src/utils/` for existing helpers before writing a new one
3. Implement the feature
4. Write the test alongside the implementation — not after
5. Sanity-check real-data edge cases before submitting — not just synthetic tests: missing / sparse / NaN inputs, TTM warmup, metrics that are N/A for a sector, alternative XBRL tags. Prove data vs extraction issues against the actual source when values look missing.
6. Run `pytest tests/path/to/new_test.py::test_function -v -s`
7. Show me **only the output of the new test**, not the full pytest summary
8. The test output must include the printed sanity check conclusion — if it doesn't, the work is not done

---

## How to communicate results

- When a task is done: show the new test output + the printed sanity check conclusion
- Do not show the full list of all passing tests — only the new ones
- If a refactor touches existing tests, tell me which ones and why, but don't dump all output
- For multi-step work: confirm each step before moving to the next

---

## What to do automatically

- Always check `src/constants/*.py` before introducing a new column name or string key
- Put global literals (date formats, SEC/API URLs, env-var keys, magic thresholds) in `src/constants/constants.py` — never hardcode them inline
- After implementing a feature, propose the unit test before marking done
- Log via `self._context.logger`, never `print()`
- When writing a new config key, add it to the appropriate `configs/*.yaml`
- When a helper is useful across folders, place it in `src/utils/` not inline
- When a generic, reusable convention emerges from a request, **propose it and ask before adding it to this CLAUDE.md** — don't edit CLAUDE.md unprompted
- Keep `CLAUDE.md`, `AGENTS.md`, and `README.md` in sync with the code: review them regularly and update them **in the same change** whenever the structure or conventions evolve (new step / table / package, moved module, new data source) — the `Code structure` section above must not drift

---

## What NOT to do

- Do not replace OmegaConf with another config system
- Do not restructure the Step inheritance pattern
- Do not cross-import between `src/` subfolders (e.g. data_extraction importing from modelling)
- Do not hardcode strings or paths that belong in `src/constants/`
- Do not reformat code unrelated to the current task
- Do not say work is done without the printed sanity check conclusion in the test output