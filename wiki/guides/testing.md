---
title: Testing and validation
description: Real-data testing, known-truth fixtures, architectural guards, sanity conclusions, and validators.
type: guide
tags:
  - wiki
  - guide
  - testing
  - validation
source_commit: 15df4358b2a4ea81c34f02c71076256aed23544a
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

## Cube assembly source replay

[The real-source replay](../../tests/data_aggregate/test_assemble_cube_real.py) reads a bounded AAPL/MSFT/JPM source window and runs the ordinary assembler against disposable SQLite. It checks exact price keys, source values and nulls under the existing precision policy, physical column removal, retained-value equality and latest feature-only reads. It does not write the production cube or rebuild peers. From an isolated worktree, set ROOT_PATH to the original repository containing the existing data/output/peers/sector_peers.json; an unavailable peer artifact is an abstention, not permission to recompute it.

On 2026-10-16, candidate1c2bc428 passed the independent replay over111 sessions/333rows and120,213source cells/nulls, narrowing366physical columns to67. All nine targets remained null on the three latest prediction rows. Two fixture-only corrections supply the existing Step config_dir contract and plain-string column names for synthetic OmegaConf selections; no application calculation or assertion was weakened.

The captured actual candidate's validator returns pull/grain/timeseries0, profile1 and six explicit ABSTAIN3 results. The profile flag is a neutral level_factor constant1, explained by [the price/share basis](../../src/data_aggregate/utils/common/level_basis.py); it is not changed to satisfy a generic variance check. Timeseries has informational findings and abstains on its frozen component without a daily_legs declaration. Subset population checks and undeclared bounds/catalogue do not become passes. Exact commands, raw JSON/logs and independent reviews are kept in the local ignored run reports/validate/2026-10-16-cube-assembly-selection/.

Separate the proof of assembly from [live feature coverage](../reference/live-database.md#cube-frontier-snapshot-2026-10-16) and [table readiness](./run-the-pipeline.md#assemble-selected-inputs-and-check-readiness). A part's maximum date does not prove every ticker/feature cell is present. [Assembly/status fixtures](../../tests/data_aggregate/test_assemble_cube.py) and [readiness fixtures](../../tests/data_aggregate/test_part_status.py) cover required/optional policy, projected reads and malformed-input refusals.

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
rtk "$PY" -m pytest tests/path/test_file.py -v -s
rtk "$PY" -m pytest tests/path/test_file.py::test_case -v -s
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

Unit/integration tests prove code behavior. [validation CLI](../../src/validate/cli.py) evaluates persisted or rebuilt data quality through explicit checks, result gates, scopes, and reports.

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

## Candidate P3 amendment validation

`tests/data_extract/institutionals/test_13f_vintages.py` combines exact parser/math fixtures with a read-only economic check on actual recovered SEC books. The final focused P3/shared availability run on 10 October 2026 passed 31 tests with one opt-in dedicated-PostgreSQL skip (20 P3 plus 11 availability tests,28.29s). It covers prospective replacement/additions, repeated originals grouped by report quarter, latest public prior-quarter baselines, stale versus active value-only revisions, mixed-manager event isolation, source reconciliation, clock exhaustion, accession gaps and immutable replay. Each test prints a targeted SANITY conclusion.

The separate disposable PostgreSQL test passed rollback after holding insertion, retry, empty completion, replay and payload-conflict checks; the disposable container was stopped and removed afterward. No production PostgreSQL volume was used for destructive test setup. The broader P3 sibling regression passed 201 tests with two opt-in skips before the final event-isolation controls. The read-only scoped source validator passed 24 accessions / 118,276 rows / 18 historical-prefix checks. Full-ledger results at a historical as-of must exactly match results after removing all not-yet-public versions.

Exact outputs and source cases are saved in the local ignored artifact `reports/validate/2026-10-09-institutional-13f-audit/p3-amendments.html`. These results validate the candidate bounded scope, not an already adopted production cube. P1's frozen 211-pass result remains a separate phase checkpoint.

The 10 October P3 copied-book refinement passes34 focused vintage/availability tests with one opt-in skip, plus independent typed-leg/provenance controls. Read-only source audit verifies133/143 accessions,64 source-backed quarter histories and122 per-book prefix boundaries;71 later-filing prefixes also remain invariant. Overall amendment source coverage is70/75 (93.33%). The broader regression exposes an existing latest-only first-publication error:5.924pp vs2.5pp ceiling over988 pairs, identically reproduced on the unchanged branch. Keep this failure visible; do not increase the threshold to obtain a green run. Candidate source and five charts are in local ignored `reports/validate/2026-10-09-institutional-13f-audit/p3-amendments.html`; full production PIT acceptance remains gated.

## Candidate genuine-share splits and daily comparable books

On branch harness/institutional-13f-pit-20261009, `StepCubeInstitutionals.build_panel` sends the existing genuine-split classification to broad and elite 13F share comparisons. Parent spinoff price factors do not rebase SEC quantities. The source-backed GE2023/2024 controls remove22.121/20.239 percentage points of false flow on identical eligible rows; NVDA10:1, WST2:1 and DD1:3 remain correctly rebased. This is split-mechanism evidence, not full original-publication coverage.

The candidate `vintage_cohort_snapshot`/`vintage_cohort_publications` in institutional_features.py, under the unmerged .worktrees/institutional-13f-pit candidate, are a **scoped preview**, not activated production consumers. They resolve immutable accession manifests at the requested as-of, select one report quarter and compare identical eligible manager books against the adjacent prior quarter. Completed empty books support reported exits; absent/unfiled/failed/restricted or unsupported quantities remain unknown. New managers affect levels only. Prior amendments, dollar/support changes and newly public books revise later states without altering earlier observations. Security CUSIPs must be independently verified stable across the report pair; the API does not infer DD-style changes or Vanguard reporting-perimeter aliases.

Book counts, quantity/value support, public-prior share coverage and median publication age accompany the preview. The coverage denominator contains only prior books public at the same as-of, on the same genuine split basis. Neither many filer IDs nor a waiting buffer establishes adequate economic coverage. The existing three-session publication policy is preserved; explicit settle_days=0 permits a bounded no-extra-wait study. Carried event snapshots must preserve unknown cells and advance age by elapsed days.

Current exact sibling run:139 passes,one opt-in disposable-PostgreSQL skip,one pre-existing legacy first-publication failure. The latter reproduces the unchanged baseline5.924pp median error over988 pairs against a2.5pp ceiling. Do not raise the threshold or describe the global legacy path as fixed. Independent focused controls and the mixed unknown-quantity probe pass on unchanged candidate hashes. Read-only source validation passes14 recovered accessions/94,341 rows/12 historical prefixes. Root code adoption, full source-version recovery, security/manager identity bridges and the deferred daily DAG lookback remain separate.

Run artifacts are local ignored paths under reports/validate/2026-10-09-institutional-13f-audit/: _out/p4_candidate_manifest.json, _out/p4_integrated_tests.txt, _out/p4_independent_review_final.json and _figures/p4_split_evidence.png. The existing interpreter lacks Ruff; Python syntax/compile, focused behavioral controls and store-boundary guards are recorded instead. No dependency was installed.

The superseding observation-clock regression found and fixed a32-day stale age after an identical current re-filing. The candidate includes publication_reference_at in revision equality, while elapsed publication_age_days alone does not generate revisions. The final vintage/step/availability suite passes51 tests with one opt-in PostgreSQL skip; independent10 focused checks plus the unknown-book probe pass on unchanged final hashes. Real source replay covers28 complete versions/121,324 rows/35 stock-prefix checks, with23 cutoff-available accessions/98,742 rows/12 validator prefixes passing. State Street supplies124 supported AAPL/NVDA daily comparisons from day45; its100% coverage denominator is one manager, not the all-filer universe. Three named combination/confidential scopes abstain on flows. All468 carried states agree on age at checked boundaries. These results are bounded mechanism evidence, not proof of marketwide day45 significance or activated production flows. Current ignored handoff: reports/validate/2026-10-09-institutional-13f-audit/p4-daily-preview.html.

## Ordinary 13F candidate validation — supersedes the preview-only checkpoint

The isolated harness/institutional-13f-pit-20261009 candidate now runs both actual StepCubeInstitutionals methods against immutable verified source versions. The final ordinary proof uses140 packets/691,676 grouped rows for ten managers and49 fresh calls, including20 full calls. Historical feature/sink prefixes and newer-quarter isolation pass. Source validator/raw arithmetic evidence is explicitly reused from an unchanged hash-verified source store; actual candidate cells and ordinary calls run freshly. Broad sample arithmetic uses a test-only zero onset floor; the real100-holder production floor is checked separately. Sparse problem histories and reporting-perimeter guards are not a representative population error rate.

The real replay exposed a shared expiry-calendar duplicate-session crash in three elite histories. The minimal iterator fix passes exact RED/GREEN, the complete independent vintage suite44pass/1 opt-in PostgreSQL skip, and all20 real full calls. Final wider regression318pass/1skip/1 unchanged latest-only legacy failure at5.92378pp vs2.5pp over988 pairs; no threshold was relaxed. Independent correctness/Ponytail review is clear for bounded handoff, not full historical adoption. Local ignored evidence:reports/validate/2026-10-09-institutional-13f-audit/04-validation-05.md and p5-normal-builders.html. Full production source recovery, valid price inputs, identity continuity, all-filer significance, runtime/memory and daily refresh equality remain in the accepted TODOs.

## Cached 13F backfill performance controls

The October 2026 refactor retains exact source/economic/public fields against dev4f125ab7 while shortening grouped book calculation, sharing the full CUSIP cache and caching verified writer preparation. [ZIP controls](../../tests/data_extract/institutionals/test_13f_backfill_zip.py) cover all-window discovery, missing manager copies, CUSIP continuity, ordinary completed-accession skips and forced source replay. [Mapper controls](../../tests/data_extract/institutionals/test_cusip_map.py) distinguish cached NULL misses from failed transports and preserve overrides. [Writer preparation](../../tests/data_store/test_13f_preparation.py) covers lazy warm use, new columns, facade invalidation and concurrent cold use. The existing vintage tests retain atomic rollback, empty-book completion and checksum/cover conflicts.

Use disposable PostgreSQL16 with a dedicated test URL for the opt-in migration/atomicity/concurrency nodes; production volumes are excluded. General fixture tests run with the live-write tripwire. On Windows, use a short workspace-contained `--basetemp` if long temporary paths or external pytest cleanup fail. Performance and read-only source validation evidence is kept in local ignored `reports/validate/2026-10-11-13f-backfill-performance/`; bounded source timings do not establish a whole-history runtime or network latency.

The integrated timing proof compares 240 requested accessions in 2014Q1, 2019Q4 and March–May2026: 222 accepted books/167,130 grouped positions match the approved-rule baseline exactly, with identical rejected siblings and unchanged archive hashes. CPU transformation improves 2.5–2.8x on these bounded samples. The same 20-book/48,911-position actual saver reduces full-map reads from 15 to 1; disposable PostgreSQL cold writes take68.83s→46.41s and immutable replay15.40s→8.09s, with exact stored payload/checksum equality. These workloads overlap and are not summed. First-save latency is not improved in the saver sample (3.08s→5.81s); full-history runtime and live network waits remain unmeasured.

## Generic price and PIT share adjustments

The price register's generic event contract is covered by [resolver controls](../../tests/utils/test_price_adjustments.py), [real original-share/quote cases](../../tests/data_aggregate/test_price_adjustment_cases.py) and [total-return controls](../../tests/data_aggregate/test_price_return_repairs.py). Parser/math fixtures distinguish missing and moved price/vendor/actual events, already-adjusted versus unadjusted bars, exact healed sources, quote units, source absence versus unavailability, repaired anchor loading and out-of-scope policies. Real stored observations test IVZ/HBAN/APA/SJM original counts, BLDR/CCI boundaries and native VTR/SJM returns; economic expected values come from issuer evidence and independent wealth arithmetic.

Always test ordinary-builder refusal with a captured isolated write seam: changed source/anchor evidence or unusable total-return event prices must cause zero writes. An already-correct target must pass without a second adjustment. Test full and incremental windows, unchanged controls, ordinary cash dividends, and separate return versus OHLC masks. Configured carried ownership rebases both quantity legs using actual G while preserving filing/publication clocks and freshness.

Run the relevant existing split-union, level/value, predecessor, insider/institutional, seam-mask, target-factor, price-part roundtrip, incremental and store-boundary siblings. Keep `PEA_LIVE_TESTS` unset. The independent read-only price validator's historical level failures must be inspected individually; its recent-seam gate alone does not certify historical levels. Generic CLI exit3 is ABSTAINED, not a pass. Source/raw and production-derived tables remain unchanged during candidate validation; historical adoption is a separately scoped operation.

## Assessment contracts and financial replay

[Training tests](../../tests/modelling/test_step_long_short.py) verify original folds/cutoff persistence, actual-label-end purging, finite target eligibility and saved full-history fitting settings/schema without repeated CV. Last-two-fold count arithmetic and five-fold/multihorizon routing use no-fit fixtures. [Artifact tests](../../tests/modelling/test_run_artifacts.py) retain production publication, failure/retry and pinned-reader checks.

[Assessment integration](../../tests/modelling/test_assessment_evaluation.py) loads prefitted known-truth members with fitting/CV forbidden, checks all-title scoring and train/assess IC parity, runs independent source/native/effect validation and refreshes after corrupting old outputs. [Publication tests](../../tests/modelling/test_fresh_assessment.py) cover generation/swap failure, same-run concurrency, exact24hour warning boundaries and Windows file locks. [Explanation tests](../../tests/modelling/test_assessment_explanations.py) retain native additivity, RF scale and linear reconstruction; [report tests](../../tests/modelling/test_assessment_report.py) cover declared folds/horizons/features and cutoff PDP.

For the 2026-10-21 train/assess refactor, the user explicitly substituted isolated synthetic data because the live cube/targets are absent, and restricts operational rehearsal to h60, at most three folds and all three families: nine fold fits, three cutoff fits and three full-history fits, plus two fresh assessments, portfolio and one final read-only validator. Use the last three configured windows without changing feature pools, controls or diagnostic budgets. Default production remains five folds and h30/h60/h90. Actual h30/h90 performance is outside this rehearsal.

Run only exact affected pytest function node IDs; no global suite, whole modelling folder or unrelated sibling suites. Record failures, repairs, execution counts and any accidental extra fixture fits explicitly. No automatic repetition of green groups.

The [independent validator](../../src/validate/checks/modelling.py), invoked with validate modelling -c <isolated-configs> -o <dated-run>, independently reconciles finite-pair Spearman/IC-IR, external-test purges, metric inputs against original scores, source-checked explanation inputs, native predictions/contributions/PDP, mature full-history refits and report content. Incomplete evidence exits3 (ABSTAINED); corrupt complete evidence fails. Positive performance is not a correctness threshold.

Offline report checks exercise every available horizon/population/statistic/family control and representative feature states. Curves reflect every declared fold plus cutoff and preserve unsupported calendar gaps. Known-truth fixtures prove math/control flow, while real scoped data supports population/temporal acceptance. Current-roster/GICS/static-peer/vendor limitations and external-test RMSE selection influence remain explicit.

## Modelling workflow acceptance 2026-10-21

Executable revision: `2c9504c51e3c1d230d3d3623ff03eb8cffc4d11a`, after isolated correction and independent correctness/simplicity reviews. The user explicitly replaced real-cube acceptance with disposable synthetic data because the live cube and target tables were absent. Normal operation remains five folds and h30/h60/h90.

The bounded functional chain uses h60, the last three configured test windows and all three families: nine original-fold models, three cutoff models and three full-history models, for 15 actual fitted objects. Two complete assessments reload these models and recompute scores/explanations; the original model bytes remain unchanged. The read-only [model validator](../../src/validate/checks/modelling.py) passes with zero findings over 20,560 scored rows. Independent arithmetic reconciles all fold IC/IR and the last-two-round mean. The offline model report passes 127 browser checks.

The integrated correction batch passes 45 exact affected cases. A subsequent 15-case [CLI/integrity batch](../../tests/modelling/test_assessment_cli.py) covers the final dated-portfolio guard and its connected compatibility paths. The guard refuses absent/null original member inventories before portfolio construction or routing changes; the shared loader retains the sole completeness/hash check. Missing-inventory failures were reproduced before correction, including a silent two-of-three member load. The final independent recheck proves the refusal and preserves complete-cutoff and generic legacy controls.

Only the portfolio refusal changes executable code after the main functional chain. Its actual guarded CLI replay passes using the same 15 models with zero additional fits, preserving active production, peer and root-user identities. The audit wrapper subsequently exits 1 while printing a Unicode arrow under Windows cp1252; the successful child result was already persisted. Independent readback passes and retains that output-only failure; the portfolio replay was not repeated.

Evidence is retained under the local ignored `reports/validate/2026-10-21-modelling-train-assess-simplification/`: `06-final.md`, `report.html`, `04-validation-01.md`, `04-validation-02.md`, exact commands/JUnit and both final review reports. Earlier accidental small legacy-fixture h30/h60 fits are explicitly disclosed and that fixture was converted to no-fit. No global/folder-wide pytest suite or additional operational training was run. This is functional/mathematical acceptance, with no real financial performance, upstream PIT, generalization or production-scale claim. The older acceptance record below remains historical.

## Modelling DAG acceptance 2026-10-19

The completed h30/h60/h90 run is `20261018_w`. Its measured product revision is `48a92a6f`; the final reviewed documentation candidate is `f539ede0` on `harness/modelling-dag-backtest`. This measured run remains frozen at those revisions; later integration with dev is a separate check and does not retrain these models. The configured artifacts and current lifecycle are described in [Modelling](../modules/modelling.md).

| Check or review | Completed result | Retained evidence |
| --- | --- | --- |
| Scoped regression and static checks | 268 tests pass, zero skips; one explicitly deselected active-production-predictor smoke. Thirty changed Python files pass Ruff, formatting and Pyright. | _out/integrated-v5.json and integrated-v5.xml |
| Full actual workflow and read-only validator | Training, saved-model assessment, portfolio backtest, isolated full-history refit, validator and offline rerender all succeed. The validator reports zero findings. | _out/chain-20261018_w.json; actual dated run's _out/modelling.json |
| Independent domain validation | A separate validator examines the outcomes and evidence; final reconciliation passes 147/147 checks. | 04-validation-02.md;_out/independent02-final.json |
| Independent validation after documentation fixes | 52/52 checks pass; source/configuration and 103 prior artifact identities remain unchanged. This reconciles existing measurements rather than repeating training. | 04-validation-03.md;_out/independent03-final-reconciled.json |
| Correctness and Ponytail simplification reviews | Independent connected-source passes, then closing reviews: CLEAR and ship. The final passes find no further product-code refactor needed. Two stale documentation issues are corrected and rechecked. | 05-review-01.md through 05-review-04.md |
| Offline report | 348 actual browser assertions; identical artifact-only model-report rerender; readiness HTML passes 15 integrity checks. | _out/browser-real-w.json;_out/final-report-integrity.json |

Evidence filenames in this table are relative to the retained local audit package `reports/validate/2026-10-18-modelling-dag-backtest/`; the model run is `data/output/modelling_backtest/20261018_w/`. The package's `06-final.md` and `report.html` preserve exact commands, findings, commits, failures and scope. Its frozen candidate remains f539ede0; later wiki-only clarification does not rewrite that review history.

The [read-only modelling validator](../../src/validate/checks/modelling.py) independently reconciles all 8,613,360 scored rows for target/metric arithmetic. Native explanation checks cover 216,000 saved member rows across 54 original/cutoff objects. Source-feature cross-sections, original OOF reload and direct PDP responses are bounded samples; all nine serialized full-history objects were not natively replayed. A simplification pass is a review for unnecessary code, not a promise of additional refactoring.

Historical current-universe/GICS/static-peer/vendor-vintage limitations and 51 unavailable configured feature names remain explicit. The 2022+ period is already inspected, without a new sealed-test or generalization claim. The original strict root user-file audit still exits 1 after separate root-dev edits; its baseline was not reset, while hard candidate/configuration/live-model/peer controls pass. No live production model was promoted. These limits keep the historical interpretation conditional despite passing implementation and workflow checks.

## Assessment rank and ticker diagnostics

[Assessment metric tests](../../tests/modelling/test_assessment_metrics.py) cover ex-ante top/bottom unions before outcome filtering, deterministic ties/overlap, original scoring-session support, pooled daily IC summaries and native error arithmetic independent of normalized scores. [Ensemble tests](../../tests/modelling/test_utils_ensemble.py) verify native outputs are retained from one call/member with identical normalized/blended scores.

[Offline report tests](../../tests/modelling/test_assessment_report.py) cover all horizons/samples/cohorts, missing evidence, native comparison gaps and unsupported views, and a greater-than-150-title fixture with different rankings by model/horizon/sample. Only each combination's 100 worst / 50 best native-RMSE histories are embedded; All IC retains every scored title. Full daily selected histories are preserved losslessly. Offline browser checks also measure settled bar-container geometry, row density and scroll reach after wide → narrow → wide resizing and model changes; trace-array equality alone does not prove chart visibility. Equal-RMSE populations of 40, 125 and 211 titles pin opposite ends of one ascending RMSE/ticker total order and exactly min(eligible, 150) distinct displayed titles; the independent checker also enforces this cardinality.

[Independent modelling checks](../../src/validate/checks/modelling.py) reproduce tail membership, finite-pair rank correlation, pooled/rolling summaries and native MAE/bias/RMSE/counts without importing producer helpers. The exact persisted error floats determine display ordering after their arithmetic is independently reconciled, so independent rounding cannot reorder near ties. [Validator corruption tests](../../tests/validate/test_modelling.py) reject early outcome filtering, wrong summaries/counts, missing sessions, wrong ticker groups/order and altered native history bytes. [The known-coefficient integration](../../tests/modelling/test_assessment_evaluation.py) checks original models, one main scoring call/member, zero inference added by errors/reporting, and fresh replacement of corrupted prior scores without estimator fitting.

Validation for this change uses exact affected pytest nodes only. Operational acceptance reuses compatible saved synthetic models at h60 / three folds / all three families, with fitting and live calls denied. Larger multi-horizon/ticker fixtures exercise chart capacity and control coverage without training. These tests establish functional and mathematical behavior; they do not measure live financial performance, upstream PIT correctness or generalization.

## Text-cube acceptance 2026-10-22

The committed refactor is at `96e77f81e5cf184b513b1e9bc4785aa68945654b`. Its source/config/test content matches the reviewed 24-file candidate `b09669091b48c436e96906fe953ee6c621e05316d037a409534b8c2cec92517b`. The [text step](../../src/data_aggregate/transformers/step_cube_text.py) runs shared preparation and independent sentiment/embedding saves before bounded feature creation. The [module contract](../modules/data-aggregate.md#earnings-call-feature-contract) describes eligibility, partial calls and the 30-calendar-day output overlap.

| Check | Measured result |
| --- | --- |
| Integrated regression | 358 passed, six skipped, zero failures/errors; five existing numeric warnings and zero live-guard trips. Skips are two unrelated opt-in live tests and four cases requiring the absent assembled cube. All 37 aggregate outputs/1,767 columns stay identical without fingerprint changes. |
| Real configured models and PostgreSQL | 253 calls/19,210 paragraphs across eight tickers, including complete stored AAPL/ADP/AMZN histories and real partial calls. FinBERT runs on CPU; actual OpenAI embeddings have 1,536 dimensions. No model outputs are mocked in this integration. |
| Eligibility and output | 251 calls pass shared text quality; two explained rejects remain pending. Dated issuer identity separately excludes one valid-text call, leaving 250 feature-eligible calls and 15,639 daily rows across seven tickers. |
| Independent formulas and PIT | All 12 features, keys, NULL masks and 180,372 supported values match independently at the existing float32 precision, without rtol/atol. Tests cover prior-only history, fiscal gaps, matching coverage, real prices/calendar, lag and original 66-session expiry. |
| Recovery and incrementality | Actual cold/full/daily runs, eight missed sessions, incomplete vectors with API unavailable, real embedding retry and missing sentiment-tag repair pass. Valid partner outputs are reused; unavailable sections retain NULL scores/zero words. Full/daily equality uses the same saved model outputs. |
| Final warm scope and speed | Four active issuers, 62 bounded history calls and one payload read of 2,258 vectors versus 3,806 before the scope fix. Three warm runs take 9.642/10.277/10.525 seconds, with zero model calls; median 10.3 seconds. Shared-machine/cache effects limit timing attribution; the 40.67% row reduction is directly measured. |
| Process memory | Final same-cache full run: 85.160 seconds, sampled peak working set 2,530,570,240 bytes; OS lifetime peak 2,586,226,688 bytes. Maximum final warm sampled working set is 511,156,224 bytes. These cohort measurements do not certify full-archive capacity. |
| Reviews and static checks | Correctness and Ponytail reviews clear, no open product finding. All 22 changed Python files pass Ruff/format; ten application modules pass Pyright. |

Production testing found two defects and preserved their RED/GREEN evidence. Mixed stored ISO dates/timestamps could become NaT and repeat completed embeddings; completion and feature joins now parse ISO8601 explicitly. Daily history also included expired unrelated issuers; [feature_history_scope](../../src/data_aggregate/utils/text/earnings_call_features.py) now limits dated history to active issuer IDs while preserving actual fiscal predecessors and old symbols. [Public step regressions](../../tests/data_aggregate/test_step_cube_text.py) cover both date-format orders with exact features and no repeated inference; the [history-scope test](../../tests/data_aggregate/test_earnings_call_features.py) includes an expired second issuer and a renamed predecessor.

Raw DOUBLE transport differences between COPY and append are retained as diagnostics: the declared float32 feature values are exact. A real repeated API request also returned slightly different valid vectors; independently calculated features match the corresponding saved vectors. Neither observation caused a product precision cast, tolerance change, baseline update or recovery abstraction.

The refactor/code result is PASS. Broader data/model readiness remains CONDITIONAL: native validators retain nine high historical/post-2022 PSI alerts and low prepared-share findings, including some mostly administrative AMZN prepared text. Missing-prior-quarter and 66-session expiry gaps are explained, not filled. Generic coverage compares the deliberately narrow cohort with 491 price issuers. Clip and generic leakage checks abstain for missing applicable declarations; independent PIT arithmetic is separate. Full-universe cold backfill capacity and original first-publication timestamps remain unverified; the original full-source domain scan was stopped after 2,074.38 seconds without a result. No threshold was changed to remove these findings.

Exact commands, raw validator exits/findings, source/data hashes, independent calculations and HTML are retained in the local ignored run `reports/validate/2026-10-22-text-cube-flow/`. The user explicitly authorized paid model calls and disposable PostgreSQL mutation for this run. Its final source/cube state is a dated [test-database observation](../reference/live-database.md#disposable-text-cube-test-state-2026-10-22), not a replacement full-universe census.

## Institutional and SEC thirteen split

The current [non-13F contract](../../tests/data_aggregate/test_institutional_step_contract.py) and [SEC thirteen contract](../../tests/data_aggregate/test_sec_thirteen_step_contract.py) use the ordinary constructors/store interfaces to check panel ownership, unique payload reads, the separate inclusive seven-session-interval tails, schema replacement and same-day reruns. [Global FTD coverage](../../tests/data_aggregate/test_ftd_global_coverage.py) compares frozen real sources with known-truth zero/unknown cases; the step-level test preserves coverage even when the selected ticker has no economic row. This is the explicitly approved bounded historical value correction.

The [fingerprint guard](../../tests/data_aggregate/test_aggregate_regression.py) keeps every surviving record unchanged; only `panel.signal_conditioning` and `panel.cross_source` were removed. [Quality controls](../../tests/data_aggregate/test_institutionals_feature_quality.py) verify both table selections, builder/source identity, strict snapshot metadata ownership and multi-batch count conservation. `scripts/institutionals_feature_quality.py --table` selects the physical output on each subcommand. The `validate institutionals` CLI reconciles source accessions, not the new cube output: use per-table validators for each produced part. Bounded replay/support limits and the final candidate identity are recorded in the local harness report; unit passes do not certify full-universe data or production adoption.
