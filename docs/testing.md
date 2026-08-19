# Testing & validation

Scope: how to write, run and judge a test here. `tests/` mirrors `src/`; 160 test files.
For the interpreter and DB access, see [runbook.md](runbook.md).

## Choose the right data source — this is the first decision

### Feature / economic tests → REAL data, small sample

```python
df = _store().load("prices", where={"ticker": wanted})    # scope it
df = df.head(100)                                          # time-series
df = df.sample(100, random_state=42)                       # non-temporal
```

**Never mock DataFrames with synthetic data for a feature or economic test.** Real data catches NaNs,
delisted tickers, corporate actions, late IPOs and weekend gaps that mocks never will — that is
exactly what surfaced the sector-NaN truncation bug.

### Exception: parsing / derivation MATH → synthetic known-truth

XBRL concept extraction, TTM arithmetic, ratio formulas, KPI math, beta estimation. You can only
assert a value is *correct* if you know the true inputs.

**Pair it with a real-data coverage check** against the cached source — e.g. build a real ticker's
history and confirm the previously-missing field now populates. The known-truth fixture proves the
formula; the coverage check proves it fires on real filings.

`tests/conftest.py::synthetic_factor_model` is the canonical example: it returns
`(y, shared, sector, true_betas)` with **known** loadings so the estimator can be asserted to recover
them.

## Fixtures

Shared loaders live in [tests/conftest.py](../tests/conftest.py), `scope="session"`, reading through
the store.

| Fixture | Gives you |
|---|---|
| `real_frames` | wide close / returns matrices from real `prices`, ~101 EQUITY tickers (always incl. `AMD`), plus the market/commodity/FX series read separately from `prices_macro` (`macro_wide`). Reads a **ticker subset**, never the whole table. It is a two-table read now: those series left `prices` |
| `real_pipeline` | end-to-end real pieces computed once: peers, sector returns, factor panel, rolling betas, multi-horizon targets |
| `fundamental_panel` | the peer-relative fundamental feature panel, scoped to the same universe as `real_frames` |
| `sqlite_store` | a **real `DataStore`** on fresh in-memory SQLite. Function-scoped. **Never skips** |
| `FakeStore` | the in-memory double for the two cases SQLite cannot serve |
| `synthetic_factor_model` | known-truth loadings |

### `sqlite_store` vs `FakeStore` — pick deliberately

**Prefer `sqlite_store`.** It is a genuine second dialect: `DataStore` adapts its upsert per dialect
(`ON CONFLICT DO UPDATE` on both Postgres and SQLite), which is exactly the path under test. And it
**never skips**, so a unit test that depends on store behaviour actually runs on a machine with no
container. It replaced ~18 hand-rolled `_FakeStore` duck types that each implemented a different
subset of the API — only one of which accepted `where=`.

Use `FakeStore` only for the two things SQLite genuinely cannot do:

1. a **vector column** (`ticker_embeddings.embedding`, `earning_calls_embedding.embedding`) —
   SQLite's driver refuses to bind a Python list;
2. a test that asserts the **order of writes** (`FakeStore.writes` / `saved_frames()`), which a real
   store does not expose.

Both doubles are faithful on the contract that matters: keyed by `name_of` so a `Table` and its name
are the same table, and **a read of an absent-or-empty table RAISES unless `optional=True`**.

### Real-data fixtures skip, they do not fail

`_store()` skips when the DB is unreachable, and each fixture skips when its table is empty. That is
intentional — these are integration tests and a machine without the container should report *skipped*,
not a wall of connection tracebacks that hides real failures.

> **Corollary you must not forget:** with `prices` currently absent from the local DB (see
> [database.md](database.md)), `real_frames` and everything derived from it **skip**. `real_frames`
> now also skips when `prices_macro` is empty, since that is where the market series (and so the
> trading calendar) comes from. A green suite here is not full coverage. Say so when you report results.

## Mandatory sanity check

A test must validate logical / financial sense, not just structure, and must **print a conclusion**.

```python
def test_momentum_feature(sample_prices):
    result = compute_momentum(sample_prices, window=20)

    # structural
    assert "momentum_20d" in result.columns
    assert result["momentum_20d"].isna().mean() < 0.1

    # economic direction
    winners = result.nlargest(5, "momentum_20d")
    losers = result.nsmallest(5, "momentum_20d")
    assert (winners["momentum_20d"] > 0).all()
    assert (losers["momentum_20d"] < 0).all()

    # MANDATORY printed conclusion
    print("\n=== SANITY CHECK: momentum_20d ===")
    print(f"  Range : [{result['momentum_20d'].min():.4f}, {result['momentum_20d'].max():.4f}]")
    print(f"  NaN % : {result['momentum_20d'].isna().mean():.1%}")
    print("  ✓ Winners positive, losers negative — direction is correct")
    print("  → Feature validated.")
```

**Do not say "tests pass" without this printed conclusion — it is part of the definition of done.**

Edge cases to cover before declaring done: all-NaN / sparse input, TTM warm-up with `min_periods`,
sector-N/A KPIs, a single ticker, a delisted ticker, idempotency (running twice is identical).

What to assert: output shape & dtypes · NaN rate within bounds · values make economic sense
(prices > 0, ratios non-negative where expected) · direction of the signal.

## Running tests

```bash
PY="$HOME/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe"

"$PY" -m pytest tests/data_aggregate/test_targets.py -v -s          # one file
"$PY" -m pytest tests/data_aggregate/test_targets.py::test_fn -v -s  # one test
```

`-s` is required — it is what shows the sanity-check print. Run from the repo root so `./configs`,
`.env` and `src` resolve.

Naming: file `test_<subject>.py`, function `test_<feature>_<condition>`.

## The architectural guard tests

These are not ordinary unit tests. If you break one, the fix is almost always in your change,
not in the test.

| Test | Invariant |
|---|---|
| `tests/dod/test_agents_md_budget.py` | `AGENTS.md` stays ≤ **70 lines**, still points at [definition_of_done.md](definition_of_done.md), and the stdlib-only hook (`.claude/hooks/dod_lib.py`) has not drifted from the generators (`scripts/dod/report_common.py`) on the report contract or the state directory |
| `tests/data_store/test_store_boundary.py` | `src/data_store/` is the only code that knows SQL exists — greps every `src/**/*.py` for `sqlalchemy` imports and `read_sql`/`to_sql`/`engine.connect`/`raw_connection`/`store.engine`/`information_schema`. Also asserts `DataStore` still exposes all 15 capabilities the former bypasses needed |
| `tests/data_aggregate/test_part_registry.py` | every part's `warmup_trading_days` covers its declared `binding_lookbacks` |
| `tests/dags/test_dag_matches_part_registry.py` | the Airflow chain and `parts.py` cannot drift (they already did once) |
| `tests/data_aggregate/test_cube_incremental.py` | an incremental trailing recompute reproduces the full build's tail exactly; and every source projection covers its builder's needs |

## The definition-of-done tests (`tests/dod/`)

The gate that decides whether a task is finished is itself tested — see
[definition_of_done.md](definition_of_done.md) for what it enforces.

| Test | Covers |
|---|---|
| `test_classify.py` | every R/N rule, both type-resolution paths, the question-turn exemption, incremental scanning, and a perf budget (5k-line transcript) |
| `test_report_contract.py` | the validator rejects a missing section, an empty §5, a hand-edited metrics block, the wrong type, another session's report |
| `test_hook_process.py` | the hooks as **real processes** under `python -S -E`, all five escape hatches, and the enforce-mode refusal text |
| `test_agents_md_budget.py` | the `AGENTS.md` line cap and hook/generator contract agreement |

`.claude/hooks/` is deliberately **not** an importable package (the hooks must stay stdlib-only),
so `tests/dod/conftest.py` loads `dod_lib` by path. The hook tests set `LOCALAPPDATA` to a
`tmp_path`, so they never touch real session state.

## The aggregation fingerprint baseline

[tests/data_aggregate/test_aggregate_regression.py](../tests/data_aggregate/test_aggregate_regression.py)
hashes 35 aggregation outputs (15 panels, 13 deduplicated primitives, 6 labels, the frozen input)
against `aggregate_fingerprint_baseline.json`.

> **The baseline may be regenerated only in a commit that touches no `src/` file, or in a PR that is
> exclusively a declared numeric change.**

The suite also carries a `DECLARED_DRIFT` set — outputs the current baseline knowingly predates by one
declared change (commit `0053dc3`, "removed peers neutrality"). A second test asserts that set stays
*accurate*: the moment a listed output matches the baseline again, it must be removed from the set.
When the baseline is eventually regenerated, `DECLARED_DRIFT` must go to **empty**, not linger.

If your change moves a number here, the right response is to state *which* outputs moved and *why*,
not to regenerate.

## CI

`.github/workflows/pylint.yml` runs `pylint` over all tracked `*.py` on push, on Python 3.8/3.9/3.10.
**This is stale** — the project requires Python ≥3.13 — so treat CI as advisory only. There is no
pytest job; test runs are local.
