# Data validation protocol

Use this reference for data or mixed work. It preserves the measure-first discipline and concrete checks from the repository's data-check command while keeping Harness Validate portable.

## Operating contract

- Start read-only. Never mutate production data or product code during a validation pass.
- Read the repository runbook before executing anything. Use its interpreter, database, and CLI forms; prefix shell commands as repository policy requires.
- Re-measure every figure quoted from documentation or an older report. Inherit prior questions and dispositions, never stale numbers.
- No actionable finding without both a measurement and a producing mechanism.
- Treat point-in-time leakage as critical. Validate against the publication/freshness clock, not merely a fiscal or observation date.
- Never hide a likely upstream error with clipping, winsorization, or imputation.

## Bootstrap and scope

1. Resolve every table through the repository table registry. Record declared primary key, grain, entity column, observation date, freshness/publication date, expected population, bounds, label conventions, and PIT sources.
2. Read schema, live-database, data-source, data-convention, modelling, testing, and runbook docs that apply.
3. Search prior reports/specs for the same table or feature. Re-run fixed findings and re-check the quantified rationale for accepted or wont-fix items.
4. Record the live database/snapshot identity, pull time, rows, entities, date span, and feature count.
5. Use one run directory: `reports/validate/<YYYY-MM-DD>-<slug>/`. Put snapshots in `_cache/`, JSON in `_out/`, table-specific rerunnable checks in `_scripts/`, figures in `plots/`, and prose reports at the run root. Never write generated artifacts in `src/validate/`.

If the repository is not PEA, map these obligations to its equivalent registry, validator, and evidence directories. Record the mapping; do not pretend a missing check exists.

## PEA validation library first

In PEA, read `docs/runbook.md` and resolve the table in `src/data_store/schema.py`; do not guess paths or date columns. Run one pull and then all nine generic checks before custom SQL:

```text
rtk <PY> -m src validate pull       -T <TABLE> -o <RUN_DIR>
rtk <PY> -m src validate grain      -T <TABLE> -o <RUN_DIR>
rtk <PY> -m src validate coverage   -T <TABLE> -o <RUN_DIR>
rtk <PY> -m src validate profile    -T <TABLE> -o <RUN_DIR>
rtk <PY> -m src validate bounds     -T <TABLE> -o <RUN_DIR>
rtk <PY> -m src validate redundancy -T <TABLE> -o <RUN_DIR>
rtk <PY> -m src validate clip       -T <TABLE> -o <RUN_DIR>
rtk <PY> -m src validate timeseries -T <TABLE> -o <RUN_DIR>
rtk <PY> -m src validate leakage    -T <TABLE> -o <RUN_DIR>
rtk <PY> -m src validate catalogue  -T <TABLE> -o <RUN_DIR> --catalogue <PATH>
```

Use the actual interpreter token from the runbook in place of `<PY>`. Each check writes `_out/<check>.json` with scope, metrics, findings, and status. Record the command and process exit code:

| Exit | Meaning |
|---:|---|
| 0 | Measured; no finding above the information band |
| 1 | Measured; one or more findings |
| 3 | ABSTAINED; it did not measure and is not a pass |

For every `3`, record the missing declaration or input. Never manufacture a configuration threshold to turn an abstention green; add declarations only from measured semantics. Read each JSON's actual source: the snapshot is a performance control, not a correctness shortcut. In the current PEA validator, grain, redundancy, and bounds preserve database precision and timeseries reads per entity; report what each result actually measured.

Run `scripts/dod/data_profile.py` and the applicable domain validator when their documented scope matches the table. Record their gates, scope, and any abstention. If either tool is absent or inapplicable, say so explicitly rather than silently dropping the obligation.

## Measurement matrix

### Grain and temporal meaning

- Count rows, distinct declared keys, null key components, entities, first/last observation, and first/last publication.
- Establish whether the table is event, daily, periodic, or periodic expanded to daily.
- Explain what every date means and which publication clock drives availability.
- Duplicate or null declared keys are structural defects.

### Coverage and missingness

- Compare entities with the authoritative expected universe, including declared exclusions.
- Measure full-span coverage and recent edge sessions per entity.
- Measure missingness overall, by entity, by time, and by feature cohort.
- Distinguish expected pre-listing/pre-publication nulls from holes, late starts, early stops, and freshness failures.
- Check sector/category-conditional features only against their eligible population.
- Reconcile row-count expectations with cadence and universe; do not use a raw count alone as proof.

### Per-column profile and bounds

For every material column, record non-null/null counts, distinct count, infinities, min, p01, p50, p99, p999 when useful, max, mean, standard deviation, robust centre/scale, and first/last non-null date.

Flag dead, constant, invalid, impossible, unit/sign/horizon mismatches, and bounds violations. Inspect named raw examples before calling an outlier a defect.

### Clipping and cross-sectional ties

- For standardized peer legs, measure share exactly on the declared clip boundary.
- For cross-sectional ranks/scores, measure per-date tie mass and degenerate dates.
- High saturation means the transform lost distance information; it is not automatically solved by more clipping.

### Point-in-time and target leakage

- Prove feature availability against each source's publication/freshness clock for named entity/date examples.
- Hunt forward shifts, centred windows, future-row joins, restatements used before publication, and forward-fills beyond their contract.
- For declared forward labels, verify maximum labelled date recedes strictly as horizon grows within the same label family and never reaches the last source date.
- Do not infer a label from a suffix alone; use the table's declared label pattern and PIT sources.

### Redundancy

- Evaluate every eligible numeric pair, not a curated list.
- Measure pairwise-complete correlation and exact equality.
- Investigate correlations above the repository threshold and identities/tautologies; name what should remain and why.
- One correlated pair can be intended. Explain before filing.

### Per-entity time series

Measure each relevant entity/feature series for:

- Jumps large both in robust-change units and relative to the feature's own span.
- Internal holes distinct from expected leading/trailing nulls.
- Frozen runs only for features declared to update at that cadence.
- Isolated extraction, cleaning, aggregation, or feature-construction kinks.

Plot and replay named examples. An informational market event is not a defect merely because it moved sharply.

### Recent-versus-history behavior

This check is compulsory for predictive data:

- Compare the latest representative 3–6 months with historical training periods.
- Compare categorical support/frequencies and continuous distributions.
- Detect source lateness, missing feature families, scale/sign changes, and coverage shifts that make live predictions unlike training.
- Interpret differences using release cadence and regimes; distribution shift is a suspect until its mechanism and consequence are known.

### Idempotency and recovery

Where the implementation writes or rebuilds data, verify within the accepted safe environment:

- Re-running does not duplicate keys or corrupt state.
- Resume/freshness boundaries do not skip or replay incorrectly.
- Partial failure has a documented recovery point.
- Validation remains read-only unless a separately approved test fixture or isolated database is used.

## Diagnose measured suspects

Only after measurement, trace each suspect:

1. Find the producing orchestrator and calculation.
2. Compare numerator/denominator, sign, window, lag, annualization, minimum history, fiscal/calendar alignment, units, corporate actions, joins, and forward-fill with the observed numbers.
3. Decide whether source, extraction, aggregation, feature logic, model interface, or the check is wrong.
4. Confirm one named entity, date, actual value, and expected value.
5. Drop a flag when the implementation correctly explains intended behavior; record that disposition.

Use this finding shape:

```text
Observed:   measurement, exact command/query, and artifact
Expected:   contract or invariant
Mechanism:  producing file/symbol/line and why
Impact:     data, financial, modelling, or operational consequence
Fix:        minimal concrete correction
Verify:     rerunnable check that fails before and passes after
```

## Scoring

- 9–10 critical: proven wrong or structural contract broken, including PIT leak, duplicate key, materially wrong calculation, or dead required feature.
- 7–8 high: probably wrong with a named mechanism and material impact.
- 4–6 medium: statistical candidate or real robustness/coverage weakness.
- 1–3 info: quantified and declared; no immediate action expected.

Provability anchors the score. Then weigh breadth, duration, silence, and blast radius.

## Required data evidence

The validation attempt and `defects.md` must state:

- Resolved registry declarations and live scope.
- Prior reports reconciled.
- All nine library commands, exit codes, JSON paths, and every abstention.
- Grain, coverage, missingness, profile, bounds, clipping, leakage, redundancy, time-series, drift, idempotency, and recovery outcomes as applicable.
- Named PIT and defect examples.
- Producing mechanisms for every retained material finding.
- Catalogue coverage in both directions: every live feature documented and every catalogue entry live.
- Table-specific checks under `_scripts/` only for gaps the generic library does not cover.
- Checks not completed and the claim each gap prevents.
- Optional PDF path when required; the HTML final report remains the harness's primary human deliverable.

Never say the data is sound because tests are green or nothing obvious appeared. A rerunnable measurement beats an assertion.
