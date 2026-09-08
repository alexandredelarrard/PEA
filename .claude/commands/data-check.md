# Data Check

Deeply validate one or more financial feature tables used to model future ticker price moves.
Your role is adversarial: **find what is wrong, incomplete, misleading, leaky, or fragile.**

**This is a DATA exercise.** Measure the table first and let the numbers nominate the suspects —
auditing code for bugs instead produces plausible stories about defects that do not exist and
misses the ones that do. The 2026-09-04 cube audit is the standing example: 65 of 179 features
emitting nothing and three pairs at Pearson r = 1.0000, with **68 tests green throughout**. The
codebase is the second half: once a measurement flags something, the code is where you find out
**why** — mandatory, because a flag without a mechanism is a rumour you cannot score or fix.

## Initial Setup

If no table is provided, respond with:

```text
🔎 Starting data validation

Which Postgres table or tables should I validate? Give the name(s) as they appear in
`Tables` (src/data_store/schema.py) — e.g. cube_part_momentum, fundamentals_history, cube.
```

Otherwise, start immediately.

## Core Rules

- **Prefix every shell command with `rtk`** (project hard rule, `AGENTS.md`).
- **Read-only.** Never `UPDATE`/`DELETE`/`ALTER`, never edit production code unless asked.
- **Re-measure anything you quote from `docs/`** — `docs/database.md` is a dated snapshot, and its
  §"Coverage gotchas" accounts for many holes; check it before filing a coverage bug.
- **No finding without both a query and a mechanism** — the query proves it is real, the code path
  proves you understand it.
- Treat **PIT leakage** as critical: a feature may contain only what was knowable at its observation
  date. Never hide a probable data bug with clipping or winsorization.

## Step 0 — Bootstrap

Interpreter, DB connection and CLI are in [docs/runbook.md](../../docs/runbook.md) — read it
rather than guessing paths. The two you will use constantly, plus the registry lookup that
declares the grain (never guess it):

```bash
PY="$HOME/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe"
MSYS_NO_PATHCONV=1 docker exec pea_db psql -U alexandre -d pea -c "SELECT ...;"
rtk "$PY" -c "from src.data_store.schema import resolve; t = resolve('<table>'); print(t.pk, t.date_col, t.ticker_col, t.freshness_col)"
```

`MSYS_NO_PATHCONV=1` is required — Git Bash otherwise mangles container-side paths; for a frame,
wrap the query in `COPY (...) TO STDOUT WITH CSV HEADER`. Date columns are **not** uniformly named
(cube parts key on `(date, ticker)`, `fundamentals_history` on `(ticker, as_of)`), and
`freshness_col` differs from `date_col` on the four tables publishing on a different clock than the
period they describe — validate PIT against the **publication** clock.

**Then read what has already been found on this table** — prior passes carry defects someone
already paid to discover, fixes that landed, and false positives not worth re-deriving:

```bash
rtk grep -rl "<table>" reports/ specs/ --include=*.md | head -20
```

Hits land under `reports/validate/<slug>/`, `reports/dod/<date>/`, `reports/<YYYY-MM-DD>/`,
`reports/planning/` and `reports/research/`. **Their numbers are as-of their date — inherit the
questions, never the figures.** Re-measure anything recorded as fixed, and check a `wontfix`'s
quantified reason against today's table rather than accepting it.

## Step 1 — Run the instruments that already exist

Never hand-roll a diagnostic the repo ships — two subtly different "outlier counts" for one column
would be worse than one. **Always** start with the profiler: grain uniqueness, per-field nulls,
percentiles, MAD outliers, and five gates against a recorded baseline (D1 pk unique · D2 rows did
not shrink · D3 no column vanished · D4 range reaches `--expect-through` · D5 no null rate worse):

```bash
rtk "$PY" scripts/dod/data_profile.py --slug <slug> --tables <table>[,<table>] \
    [--tickers AAPL,JPM] [--since 2015-01-01] [--expect-through <date>] [--parts]
```

A `--tickers` run is a **partial scope**, blocked from overwriting the baseline so it cannot neuter
D2. Quote the scope with every rate: "null rate 4%" means nothing without "over which tickers, over
which window".

**If the domain already has a validator, run it before measuring anything yourself:**

```bash
rtk "$PY" -m src validate fundamentals --tier 1 [-t AAPL,JPM]   # + --roster, and `validate checks`
rtk "$PY" -m src validate prices [-t AAPL] [--no-write]         # 3 adjustment-basis invariants
rtk "$PY" -m scripts.cube_fundamentals_evidence --check         # cube_part_fundamentals
rtk "$PY" -m src data_aggregate cube-status                     # part freshness; exit 2 if behind
```

⚠ **Read [src/validate/README.md](../../src/validate/README.md) §4 "WHEN IT DOES NOT WORK" before
quoting any result.** An ABSTAINED check examined nothing, which is **not a pass** (`peer_ratio`
below 5 filers in a regime, `level_outlier` below 8 periods, `coverage_field` below 4); a THRESHOLD
BUG check is burying real findings under itself. Either one inflates the rankings — say so.

**Reusable kernels — import, do not reimplement:**

| Need | Use |
|---|---|
| outlier score | `src/validate/outliers.py` → `modified_zscore`, `count_mad_outliers`, `detect_level_outliers` |
| what a cube feature MEANS | `scripts/cube_feature_catalogue.py` → `CATALOGUE` |
| the clip a z-score sits on | `src/data_aggregate/utils/common/xs.py` → `XS_CLIP_LABEL=3.0`, `XS_CLIP_CHARACTERISTIC=4.0`, `XS_CLIP_PEER=8.0` |
| PIT quarterly → daily | `src/data_aggregate/utils/common/pit.py` |
| price / share adjustment basis; worked precedents | `src/validate/prices.py` → `load_panel`, `invariant_*`; `reports/validate/targets/_scripts/*.py` |

The MAD kernel scores a **log change, not a level** — scoring raw levels flags the entire recent
era of any growing company, correctly and uselessly. Read rules are in [docs/data_conventions.md](../../docs/data_conventions.md). The one thing not
there: your scripts live outside `src/`, so the store boundary does not bind them — `context.store`
for row reads, but **aggregate** measurement server-side via `src.utils.db.get_engine`, as
`cube_fundamentals_evidence.py` does. `corr()`, `percentile_cont()` and `count(DISTINCT ...)`
belong in the database; the cube is ~26 GB and materialising it is an OOM.

## Step 2 — Measure the data

### Grain

```sql
SELECT count(*) AS rows, count(DISTINCT (<pk...>)) AS keys, count(DISTINCT ticker) AS tickers,
       min(<date_col>) AS first_date, max(<date_col>) AS last_date
FROM   <table>;
```

Substitute the pk and date column from Step 0; `rows > keys` is a D1 failure. Then establish: daily
/ quarterly / quarterly-expanded-to-daily (if expanded, which publication date drives it and what
the ffill limit is); what the date column *means* — observation, fiscal period end, or publication
event; and expected coverage, from `sp500_tickers` not from the table.

### Coverage 

Identify if the table assessed has all tickers expected (so far focusing on 491 tickers from SP500). 
Example of questions to cover with deep data extraction and analysis are (non exhaustive):
- Does the raw count makes sense, daily its around 3M expected. 
- Do I have a drop of tickers for the latest days ?
- Does missing values make sense for tickers with less coverage (some tickers come later, so normal to have Nan before their listing).
- Is the column sector specific and is it filled for the tickers that are part of the sector ? 

### Per-column profile

`data_profile.py` covers a normal-width table. For a wide one, generate the `UNION ALL` block
programmatically the way `cube_fundamentals_evidence.profile()` does — never type 250 columns:

```sql
SELECT '<col>' AS col, count(*) AS n, count(<col>) AS n_ok, min(<col>) AS lo,
       percentile_cont(0.01) WITHIN GROUP (ORDER BY <col>) AS p1,
       percentile_cont(0.50) WITHIN GROUP (ORDER BY <col>) AS p50,
       percentile_cont(0.99) WITHIN GROUP (ORDER BY <col>) AS p99,
       max(<col>) AS hi, avg(<col>) AS mean, stddev(<col>) AS sd,
       count(DISTINCT <col>) AS n_distinct
FROM   <table>
```

Flag `n_distinct = 0` (a dead feature — 65 of 179 cube features were exactly this and no test
noticed), `n_distinct = 1`, infinities and invalid values; missingness **overall, by ticker, and
over time** (4% global that is 100% for 20 tickers is a different bug from 4% spread evenly);
frozen series (`GROUP BY ticker HAVING count(DISTINCT <col>) <= 2`); and whether name, units, sign
and horizon match the values.

**For a z-scored column, clip saturation replaces the winsorization question** — a clipped value no
longer measures distance from peers, it reports that standardisation ran out of room:

```sql
SELECT avg((abs(<col>) >= 8.0 - 1e-9)::int) AS on_clip
FROM   <table> WHERE <col> IS NOT NULL;      -- 8.0 = XS_CLIP_PEER
```

For raw levels, inspect real examples first. A 10× mis-tagged fact is fixed **upstream**, never
clipped.

### Leakage

```sql
-- a fundamentals-derived daily feature must never precede its own publication date
SELECT count(*) FROM <daily_table> d JOIN fundamentals_history f USING (ticker)
WHERE  f.as_of > d.date AND <the feature is sourced from that row>;

-- a forward label must RECEDE as the horizon grows, and never reach the last price
SELECT target_horizon, max(date) AS last_label
FROM   cube_part_targets WHERE target_rank IS NOT NULL GROUP BY 1 ORDER BY 1;
SELECT max(date) AS last_price FROM prices;
```

A label reaching the last price, or three horizons stopping on the same day,
is a leak. Also hunt forward shifts, centred windows, joins selecting future rows, restated data.

### Redundancy

```sql
SELECT corr(<a>, <b>) AS r FROM <table> WHERE <a> IS NOT NULL AND <b> IS NOT NULL;
```

Flag `|r| > 0.985` and exact duplicates — the 2026-09-04 audit found **seven pairs at r = 1.0000**.
Beware tautologies (`ebt - ebit == -intexp` on Sharadar is an identity); say which one to keep.

## Step 3 — Diagnose each flag in the code

**Only now open the codebase, and only for what the data flagged.** For each suspect, work down
until you can name the mechanism:

1. **Find the producer.** `src/<pkg>/step_*.py` orchestrates; the feature math is under its
   `utils/`. [docs/architecture.md](../../docs/architecture.md) maps stage → file.
2. **Read the calculation** against the numbers you measured: numerator/denominator and sign,
   window/lag/annualisation, minimum history, fiscal-vs-calendar, corporate actions, units, joins
   and forward-fills, and behaviour for IPOs, delistings and sparse histories.
3. **Decide which layer is wrong** — source, extraction, aggregation, or *the check itself*.
   Challenge the check before the data: "the vendor is inconsistent" is the most comfortable wrong
   answer available, and it cost a month on the market-cap defect where our own price leg moved.
4. **Confirm on a named example**: one ticker, one date, the value, what it should have been.

If the code explains the number *correctly* — intended behaviour, right data — say so and drop the
flag. That is a successful outcome, not a wasted step. Record every material finding as:

```text
Observed:   what the data shows, with the query that shows it
Expected:   what should happen
Mechanism:  file.py:line — the code producing the wrong value, and why it does
Impact:     financial / modelling consequence
Fix:        a concrete implementation change
Verify:     the script in _scripts/ that fails now and passes after
```

## Priority Scoring

Score 1-10, anchored on **provability** the way `src/validate/README.md`'s severity ladder is, then
weighted by rows/tickers affected, persistence, silent-failure risk and blast radius.

| Score | Severity | Means |
|---:|---|---|
| 9-10 | `critical` | provably wrong or a structural contract broken: a PIT leak, a duplicate on the declared pk, a materially wrong calculation, a feature emitting nothing |
| 7-8 | `high` | probably wrong, and a **named mechanism** says so |
| 4-6 | `medium` | a statistical candidate — look, do not assume; or a real robustness/coverage improvement |
| 1-3 | `info` | declared, quantified, no action expected |

**Corroboration multiplies**: nine checks agreeing is nine arguments; one check firing 62 times is one opinion repeated.

## Required Output — a PDF, backed by re-runnable scripts

```text
reports/validate/<slug>/
├── report.pdf     REQUIRED — the deliverable
├── report.md      REQUIRED — same content, diffable in-repo
├── defects.md     the issue register, highest score first
├── _scripts/      REQUIRED — each check reads _cache/, writes _out/, exits non-zero on failure
├── _out/ _cache/  the JSON each check emits · one dated snapshot pull, reused by every check
└── plots/         PNGs referenced from the report
```

**`_scripts/` is the point of the exercise.** Every acceptance number in the fundamentals rebuild
came from scratchpad scripts that no longer exist. Numbers re-measurable after the next rebuild are
evidence — and are what makes this report useful to the next pass that greps for it.

**Rendering the PDF.** Write `_scripts/render_pdf.py` following
[cube_fundamentals_evidence.py](../../scripts/cube_fundamentals_evidence.py) (`_meta` / `_pdf_rows`
/ a `narrative` module of `(kind, payload)` blocks: `h1`, `h2`, `p`, `table`, `spacer`,
`pagebreak`), reusing the helpers in
[cube_evidence_pdf.py](../../scripts/cube_evidence_pdf.py) — whose `render()` hardcodes its own
title and footer, so parameterise those. Keep the two things it got right the hard way:
**landscape A3** (the ~8-column feature table has three prose columns; on A4 they collapse to ~4
characters per line) and **DejaVu registered from matplotlib** (Helvetica renders `⚠ ≥ ≈ × σ — →`
as black boxes).

**The report opens with** a `**Pull**` / `**Scope**` line — date, live `pea_db`, table, tickers,
span, rows, features — linking `_scripts/` and `defects.md`. Then: **Headline** (one-paragraph
verdict) · **Method note**, what each check is computed *independently of*, which is what makes it
evidence · **Feature analysis** · **What works well**, evidence-backed only · **Defects register**.

| Feature | What it measures | Missing % | Ticker coverage | p1/p50/p99 | On clip % | Status | Comment |
|---|---|---:|---|---|---:|---|---|

For a cube part, take "what it measures" from `CATALOGUE` and assert **both directions**: every live
column has an entry, every entry has a live column. Where a prior report covered this table, close
the loop — say which of its findings are fixed, which persist, and which you believe were wrong.

## Completion Checklist

- [ ] resolved the table in `Tables`; used its declared pk / date_col / freshness_col
- [ ] read the prior reports on this table and reconciled their findings against yours
- [ ] ran `data_profile.py` (five gates + scope) and the domain validator, naming any ABSTAINED
- [ ] queried the live DB directly; re-measured every figure quoted from `docs/`
- [ ] tested grain uniqueness and coverage against `sp500_tickers`
- [ ] checked missingness overall/by ticker/over time; measured clip saturation
- [ ] tested PIT leakage with a named ticker/date; investigated every `|corr|` above 0.99
- [ ] cross-checked the feature catalogue in BOTH directions
- [ ] **for every flag, opened the producing code and named the mechanism** — or dropped the flag
- [ ] committed every check under `_scripts/`, each exiting non-zero on failure
- [ ] scored every issue 1-10 with a fix and a verification script; produced `report.pdf` + `.md`

State any check you could not complete. **Never say the data is OK simply because nothing obvious
appeared**, and never offer a green suite as evidence — in the audit that motivated this command,
all 68 were green.

## Completion Message

```markdown
✅ Data Validation Complete

- Table(s): [table]   ·   Grain: [pk] — [unique / N duplicates]
- Rows / tickers / span: [N / N / min → max]   ·   Abstained checks: [list, or "none"]
- Profile gates: [D1..D5 pass|fail]   ·   Prior reports reconciled: [paths, or "none found"]
- Critical findings (9-10): [N]   ·   Highest-priority fixes: [short list]
- Report: `reports/validate/<slug>/report.pdf` (+ `report.md`, `defects.md`, `_scripts/`)
```

## Guiding Principle

**Measure first; open the code to explain what you measured.** Optimise for finding the truth, not
for no problems — a measurement you can re-run beats an assertion you have to believe.
