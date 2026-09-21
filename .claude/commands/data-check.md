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
`Tables` (src/data_store/schema.py) — e.g. cube_part_momentum.
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

## Step 1 — Bootstrap

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

## Step 2.0 — Run the check library FIRST

Nine table-agnostic checks live in `src/validate/checks/`, one CLI command each. Run them before
writing a single query: they measure the FULL table, they carry thresholds that were set from
measurements rather than taste, and they already know the false positives this repo has paid to
discover. What they report is your suspect list; the SQL in Step 2 is for what they do not cover
and for the follow-up each finding needs.

```bash
PY="$HOME/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe"
T=cube_part_institutionals            # the table under test
O=reports/validate/$(date +%F)-${T}   # every artifact lands here; NEVER under src/

rtk "$PY" -m src validate pull        -T $T -o $O    # one snapshot, reused by the checks that can

rtk "$PY" -m src validate grain       -T $T -o $O
rtk "$PY" -m src validate coverage    -T $T -o $O
rtk "$PY" -m src validate profile     -T $T -o $O
rtk "$PY" -m src validate bounds      -T $T -o $O
rtk "$PY" -m src validate redundancy  -T $T -o $O
rtk "$PY" -m src validate clip        -T $T -o $O
rtk "$PY" -m src validate timeseries  -T $T -o $O
rtk "$PY" -m src validate leakage     -T $T -o $O
rtk "$PY" -m src validate catalogue   -T $T -o $O --catalogue <path to a JSON or a .py with CATALOGUE>
```

Each writes `$O/_out/<check>.json` — scope, findings with a 1-10 score, and metrics — and prints a
one-line summary.

**THE EXIT CODE IS THE CONTRACT, AND THERE ARE THREE OF THEM:**

| code | meaning |
|---:|---|
| `0` | measured, nothing above the `info` band |
| `1` | measured, a finding stands |
| **`3`** | **ABSTAINED — it did not measure. This is NOT a pass.** |

A check abstains when the table declares nothing it can test against: no `bounds`, no clip
suffix, no `daily_legs`, no `label_pattern`, no `pit_sources`, no `--catalogue`. "0 legs over the limit" reads identically
whether the check looked and found none or never looked at all, so **every `3` must be named in
your report** with what declaration is missing — the check prints it. Add the declaration to
`configs/validate.yml` only with a measurement behind it; an absent entry abstains loudly, an
invented one lies quietly.

`pull` is a memory and time control, not a correctness one: it stores floats as float32, so
`grain`, `redundancy` and `bounds` read float64 from the DB regardless and `timeseries` reads one
ticker at a time. Each result's `scope.source` records what it actually read.

Everything — `_cache/`, `_out/`, `plots/` and the report — lives under `reports/validate/<slug>/`
via `--out`. **`src/validate/` holds functions only; nothing is ever written inside it.**

## Step 2 — Measure the data

Step 2.0 has already answered most of what follows on the full table. Use these to follow up on
what it flagged, and to cover what no generic check can know about this table.

### Grain

```sql
SELECT count(*) AS rows, count(DISTINCT (<pk...>)) AS keys, count(DISTINCT ticker) AS tickers,
       min(<date_col>) AS first_date, max(<date_col>) AS last_date
FROM   <table>;
```

> **`validate grain`** has already done this over every row: rows vs distinct declared-pk keys,
> nulls in a key column, and the panel edges.

Substitute the pk and date column from Step 1; `rows > keys` is a D1 failure. Then establish: daily
/ quarterly / quarterly-expanded-to-daily (if expanded, which publication date drives it and what
the ffill limit is); what the date column *means* — observation, fiscal period end, or publication
event; and expected coverage, from `sp500_tickers` not from the table.

### Coverage

> **`validate coverage`** measures this against `load_universe_tickers` (491 = the roster minus
> `INSUFFICIENT_HISTORY_TICKERS` and `redundant_ticks`) — per ticker against its own `prices` span,
> plus the last `edge_days` sessions one at a time. The nine declared exclusions come back as ONE
> `info` finding, not as missing data: reporting them as defects is the retracted D-08 finding.

Identify if the table assessed has all tickers expected (so far focusing on 491 tickers from SP500).
Example of questions to cover with deep data extraction and analysis are (non exhaustive):
- Does the raw count makes sense, daily its around 3M expected.
- Do I have a drop of tickers for the latest days ?
- Does missing values make sense for tickers with less coverage (some tickers come later, so normal to have Nan before their listing).
- Is the column sector specific and is it filled for the tickers that are part of the sector ?

### Per-column profile

> **`validate profile`** emits n / n_ok / n_null / n_distinct / min / p01 / p50 / p99 / p999 / max
> / mean / sd / MAD centre and scale / first and last non-null date for every leg, and names the
> dead, constant and infinite ones. ⚠ Run `pull` first on a wide part: one column group is one
> pass, so 249 legs is 32 passes — 25+ minutes against Postgres, 184 s off the snapshot.
> **`validate bounds`** asserts the `[lo, hi]` a table declares, and abstains when it declares none.

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

> **`validate clip`** does both halves of this over the full table: on-clip share per `*_vs_peers`
> leg against the declared `clip_peer`, and per-date tie mass per `*_xs` leg. It abstains when the
> table declares no suffix convention, because measuring tie mass over an empty set of legs and
> calling it clean is the exact failure this library exists to prevent.

**For a z-scored column, clip saturation replaces the winsorization question** — a clipped value no
longer measures distance from peers, it reports that standardisation ran out of room:

```sql
SELECT avg((abs(<col>) >= 8.0 - 1e-9)::int) AS on_clip
FROM   <table> WHERE <col> IS NOT NULL;      -- 8.0 = XS_CLIP_PEER
```

For raw levels, inspect real examples first. A 10× mis-tagged fact is fixed **upstream**, never
clipped.

### Leakage

> **`validate leakage`** runs two DECLARED halves and says which ran. Neither is inferred:
> the horizon half needs `label_pattern` (an `_h<n>` suffix does not make a column a forward
> label -- momentum's backward-looking `seasonal_h30/60/90` read as three leaks when it was
> matched blind) and the PIT half needs `pit_sources`. **Horizon recession**: within each
> label family (the leg name with its own `_h<n>` cut out — `target_rank_h30` and
> `target_zscore_h30` end on the same day BECAUSE they are the same horizon, and comparing across
> families reports leaks that are not there), `max(date)` must recede strictly and none may reach
> `max(prices.date)`. **`as_of` PIT**: per TICKER, a feature's first non-null date against its
> source's own **publication** clock (`resolve(src).freshness_col`), which the PIT half abstains
> without (`pit_sources` in `configs/validate.yml`).

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

> **`validate redundancy`** has already done EVERY pair — 30,876 of them on
> `cube_part_fundamentals` — in one streaming pass, exact pairwise-complete Pearson plus an exact
> **equality count**, which is the stronger claim. Do not hand-pick pairs: measured on
> `cube_part_institutionals`, a 249-pair curated candidate list missed
> `f_ic_super_exit_after_top10 ~ f_ic_super_full_exits` at r = 0.9965 entirely. Use the SQL below
> only to re-confirm a single pair by hand.

```sql
SELECT corr(<a>, <b>) AS r FROM <table> WHERE <a> IS NOT NULL AND <b> IS NOT NULL;
```

Flag `|r| > 0.985` and exact duplicates — the 2026-09-04 audit found **seven pairs at r = 1.0000**.
Beware tautologies (`ebt - ebit == -intexp` on Sharadar is an identity); say which one to keep.

### time series per ticker and variable

A strong identifier of data issues is to plot or observe one ticker at a time over time for a specific variable.
If you have an irregular kink, strong drop or jump just for few days or quarter, or a whole or the values do not move for several days or quarters,
It means :
- underlying data extraction have an issue -> edgar
- cleaning of the data has an issue -> codebase to investigate, edge case to identify then fix in next session
- aggregation, feature construction has an issue -> codebase to be fixed later.

> **`validate timeseries`** runs this over every (ticker, leg) pair — 62,357 of them on
> `cube_part_institutionals` — with three kernels: **JUMP** (modified z of the change above
> `jump_z` **AND** the move covering at least `jump_span_frac` of the leg's own p1-p99 span; both
> gates are load-bearing, the z gate alone produced 296,926 "jumps" against 16,410 on the same
> scope, and small-integer legs are excluded rather than thresholded harder), **HOLE** and
> **FROZEN**. Jumps are filed at `info` — a real disclosure legitimately moves these series.
>
> ⚠ **FROZEN abstains unless the table declares `daily_legs`**, and `cadence: daily` is not that
> declaration. Every leg of `cube_part_momentum` is a cross-sectional percentile, so a name that
> holds its rank holds its value: AAPL carried `dollar_volume_63 == 1.0000` for 2,071 consecutive
> sessions. Run on "every leg because the table is daily" that produced 4,661 flat spells, none of
> them a defect.
>
> This is the one check that takes `-t/--tickers`, because its claim is per (ticker, leg). Use it
> to replay a single name against a plot.

Run this analysis per ticker on what `timeseries` flagged, and plot the named examples: a finding
you can see is a finding you can explain.

### Consistency for new predictions vs past history 

Check on 5 different tickers, for the table variables, if the time series behave the same way (same distribution) for the latest few 3-6 months vs past 3-5 years. 
The check is there to understand if the is leakage in terms of repructability of the signal over time. 
Each table does not arrive the same time over the year, so if one is missing, some signal can drop, creating strong issues for predictions since model will be trained, expecting a signal from feature that would be biased by the lack of extracted / freshness information. 

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
Fix:        a concrete implementation change. Explain in clear english what the fix has to be.
Verify:     the script in _scripts/ that fails now and passes after
```

## Priority Scoring

Score 1-10, anchored on **provability**, then weighted by rows/tickers affected, persistence,
silent-failure risk and blast radius. The table below is the ladder — it is the definition, not a
copy of one, and `src/validate/result.py` maps every check's score onto these same four bands.

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

**`_scripts/` holds what the library does not cover — never a re-implementation of what it does.**
The nine checks are importable (`from src.validate import check_grain, ...`), so a runner here is a
few lines that calls them and writes the report. What belongs in `_scripts/` is the table-specific
work: the catalogue file `catalogue` reads, the plots, the query that confirms a named example, and
any check this table needs that no generic one can express. Every acceptance number in the
fundamentals rebuild came from scratchpad scripts that no longer exist; numbers re-measurable after
the next rebuild are evidence, and are what makes this report useful to the next pass that greps
for it.

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
- [ ] **ran all nine `python -m src validate` checks and recorded every exit code — naming each
      ABSTAINED one (`3`) and the declaration it is missing. A zero from a check that abstained is
      not a pass, and an abstain left unnamed is a gap the report claims to have covered**
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
- Rows / tickers / span: [N / N / min → max]
- Library checks: [N pass / N finding / N ABSTAINED]   ·   Abstained: [check — missing declaration]
- Profile gates: [D1..D5 pass|fail]   ·   Prior reports reconciled: [paths, or "none found"]
- Critical findings (9-10): [N]   ·   Highest-priority fixes: [short list]
- Report: `reports/validate/<slug>/report.pdf` (+ `report.md`, `defects.md`, `_scripts/`)
```

## Guiding Principle

**Measure first; open the code to explain what you measured.** Optimise for finding the truth, not
for no problems — a measurement you can re-run beats an assertion you have to believe.
