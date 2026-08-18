---
name: dod-data-report
description: >
  Write the DATA definition-of-done report after extraction, aggregation, or validation work —
  a new or changed fetcher, a cube part, a new table or field, a schema change, an audit pass,
  or anything under src/data_extract, src/data_aggregate, src/data_peers, src/data_store,
  src/validate, sql/, configs/build_cube.yml or peers.yml. Also use when the DoD Stop hook says
  "classified DATA" or asks for reports/<YYYY-MM-DD>/<slug>__DATA.md.
---

# DATA definition-of-done report

Contract: [docs/definition_of_done.md](../../../docs/definition_of_done.md). The generator fills
the numbers; you write §1, §5 and §6.

## 1. Profile the tables you touched

```bash
PY="$HOME/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe"
"$PY" scripts/dod/data_profile.py --slug <kebab-slug> --tables fundamentals_facts,sp500_tickers
```

Useful flags — **scope is the point**, not an afterthought:

| Flag | Use |
|---|---|
| `--tickers AAPL,JPM` | iterate fast; a scoped run can never set the baseline |
| `--since 2024-01-01` | bound the window |
| `--limit 0` | no row cap (needed for a full-scope run) |
| `--expect-through 2026-08-01` | turns **D4** from N/A into a real check |
| `--parts` | add the cube-part status block |
| `--update-baseline` | record this shape so D2/D3/D5 can compare next time |
| `--declare-shrink <table>` | a row-count drop you *intended* (deletes, dedupe) |

`docs/database.md` first: the live DB has **no `prices` and no cube/prediction tables**, so pick
a table that is actually populated or D1–D5 all come back N/A and the report says nothing.

## 2. Read the gates

| Gate | Meaning if it FAILS |
|---|---|
| D1 | duplicate rows under the declared PK **or** the declared PK names columns the live table does not have (schema drift — both are real) |
| D2 | row count dropped below the recorded baseline and you did not declare it |
| D3 | a column present in the baseline is gone |
| D4 | date coverage does not reach `--expect-through` |
| D5 | a field's null rate got worse by more than 0.5pp |

## 3. Write §1, §5, §6

- **§1** — the generator filled the sample scope. Add *what was asked*. If you profiled two
  tickers, the report says two tickers; do not describe it as a universe run.
- **§5** — mandatory, non-empty. The rows you could not reconcile, the ticker you skipped, the
  field still null, the gate that came back N/A because no baseline existed yet. If genuinely
  nothing: `- None. Checked: <30+ chars>`.
- **§6** — next actions.

## 4. Rules

- **Never edit the ` ```json dod-metrics ` block** — it is hash-checked.
- Table-wide numbers (`rows`, `date_min`, `date_max`) and sample numbers (null rates,
  percentiles) are **different populations**. The table labels which is which; do not mix them
  in a sentence.
- Do not "fix" a risk zone (`src/data_store/`, `sql/schema.sql`, `configs/`) to make a gate
  pass. Report it and ask.
