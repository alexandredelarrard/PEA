---
title: Testing and validation
description: Real-data testing, known-truth fixtures, architectural guards, sanity conclusions, and validators.
type: guide
tags:
  - wiki
  - guide
  - testing
  - validation
---
# Testing and validation

## Goal

Choose evidence that proves the code path, data contract, and economic interpretation of a change. A passing assertion set is incomplete until the test prints a concise sanity conclusion.

## Choose the data source first

### Feature and economic behavior

Use a small, bounded sample from real stored data. Real observations expose nulls, delistings, corporate actions, short source histories, filing lags, and weekend gaps that synthetic frames hide.

Scope at the database:

```python
df = store.load(
    Tables.prices,
    columns=["date", "ticker", "close_total"],
    where={"ticker": wanted},
    since=start,
)
```

Then take a deterministic sample appropriate to the time-series or cross-sectional test.

### Parser and mathematical identities

Use synthetic known-truth inputs when correctness requires an exact answer: tag resolution, TTM arithmetic, ratio formulas, beta recovery, or parser branching.

Pair the known-truth test with a real-data coverage check when the practical question is whether the fixed path fires on actual filings. The synthetic half proves the math; the real half proves reach.

## Shared fixtures

[tests/conftest.py](../../tests/conftest.py) owns session-scoped real data and store fixtures.

| Fixture | Contract |
| --- | --- |
| `real_frames` | Bounded equity close/return matrices plus separately loaded macro series. |
| `real_pipeline` | Peer, sector-return, factor, beta, and target pieces computed once from real data. |
| `fundamental_panel` | Peer-relative fundamental features on the real fixture universe. |
| `sqlite_store` | Real `DataStore` behavior on fresh in-memory SQLite; function-scoped and never skipped. |
| `FakeStore` | Narrow double for vector binding and write-order assertions that SQLite cannot provide. |
| `synthetic_factor_model` | Known loadings for exact estimator recovery. |

Prefer `sqlite_store` for store behavior. It exercises a real second SQL dialect and catches facade divergence. Use `FakeStore` only when a PostgreSQL vector payload cannot bind in SQLite or when the test must inspect write order.

Both test stores preserve the production rule that an absent or empty table raises unless `optional=True`.

## Integration skips

Real-data fixtures skip when the local database, source table, or required macro leg is unavailable. This is intentional for an integration layer, but a green suite with skipped price-dependent fixtures is reduced evidence.

Always report material skips. Do not substitute synthetic price data just to turn a skip into a pass.

## Mandatory sanity conclusion

A completed test checks structure and meaning, then prints what the result implies:

```python
def test_momentum_direction(real_prices):
    result = compute_momentum(real_prices, window=20)

    assert "momentum_20d" in result
    assert result["momentum_20d"].isna().mean() < 0.10
    assert (result.nlargest(5, "momentum_20d")["momentum_20d"] > 0).all()
    assert (result.nsmallest(5, "momentum_20d")["momentum_20d"] < 0).all()

    print("\n=== SANITY CHECK: momentum_20d ===")
    print(f"NaN rate: {result['momentum_20d'].isna().mean():.1%}")
    print("Winners are positive and losers are negative; direction is correct.")
```

Run with `-s` so the conclusion is visible. Useful assertions include shape, dtype, null bounds, value bounds, economic sign, expected ordering, idempotency, and behavior on sparse/all-null inputs.

## Running tests

Run from the repository root. The host Python and Poetry commands are not available on `PATH`; use the Poetry virtual-environment executable described in [run the pipeline](./run-the-pipeline.md).

```bash
"$PY" -m pytest tests/path/test_file.py -v -s
"$PY" -m pytest tests/path/test_file.py::test_case -v -s
```

Naming:

- file: `test_<subject>.py`;
- function: `test_<feature>_<condition>`.

Report only the targeted test's relevant output and printed sanity conclusion unless a broader run was requested.

## Architectural guard tests

| Test | Invariant |
| --- | --- |
| [test_store_boundary.py](../../tests/data_store/test_store_boundary.py) | Only `src/data_store/` knows SQL, and the facade retains required capabilities. |
| [test_part_registry.py](../../tests/data_aggregate/test_part_registry.py) | Part warm-ups cover every declared binding look-back. |
| [test_dag_matches_part_registry.py](../../tests/dags/test_dag_matches_part_registry.py) | Airflow part commands and the registry cannot drift. |
| [test_cube_incremental.py](../../tests/data_aggregate/test_cube_incremental.py) | Incremental tails reproduce full builds and source projections cover builder requirements. |
| [test_aggregate_regression.py](../../tests/data_aggregate/test_aggregate_regression.py) | Aggregation outputs retain approved numeric fingerprints. |

If an architectural guard fails after a production change, fix the implementation or its registry. Do not weaken the test merely to accept the drift.

## Aggregate fingerprint baseline

The fingerprint test hashes representative panels, primitives, labels, and frozen inputs. Its checked set is owned by the test itself; count it from code rather than repeating a prose number.

Regenerate the baseline only for an isolated declared numerical change or a commit that changes no production source. When an output moves, report which output moved and the causal change. Any temporary declared-drift set must remain accurate and be emptied when the baseline is intentionally regenerated.

The baseline is a risk zone and requires approval before editing.

## Edge cases

For features and transformations, consider:

- all-null or sparse input;
- insufficient TTM or rolling warm-up;
- sector-not-applicable KPIs;
- single-ticker cross-sections;
- listings, delistings, and symbol transitions;
- split and dividend basis;
- amendments and same-day filings;
- source-availability boundaries;
- repeat execution/idempotency; and
- full versus incremental tail equivalence.

For strategies, include exposure residuals, position counts, gross/net constraints, turnover, transaction costs, integer-share feasibility, and realistic capital.

## Validation subsystem

Unit/integration tests prove code behavior. [src/validate](../../src/validate/) evaluates persisted or rebuilt data quality through explicit checks, result gates, scopes, and reports.

Use the validator that matches the changed domain. Validation may abstain when a contract is not applicable; abstention is not a pass. Read check-health diagnostics before ranked findings because a miscalibrated threshold can inflate or hide clusters.

The institutionals validator rebuilds its panel in memory rather than trusting persisted output, so budget it like a full institutional build. Validation is read-only with respect to application tables, but can persist its own run/report metadata.

See [validate a change](./validate-a-change.md) and the [validation module](../modules/validation.md).

## CI caveat

The inspected GitHub workflow runs pylint on older Python versions and does not run pytest. The application requires Python 3.13, so local targeted tests and validators remain the meaningful acceptance path until CI is modernized.

## Completion checklist

1. The test matches the affected contract.
2. Real data is used for economic behavior; known-truth fixtures are used for exact math.
3. Database reads are projected and scoped.
4. The test asserts logical meaning, not only shape.
5. The sanity conclusion is printed and visible under `-s`.
6. Integration skips are disclosed.
7. Architectural guards remain intact.
8. A relevant validator/report is run for important data/model/output changes.

## Related

- [Validate a change](./validate-a-change.md)
- [Coding standards](./coding-standards.md)
- [Tests module](../modules/tests.md)
- [Validation module](../modules/validation.md)
- [Data access and storage](./data-access.md)
