# `src/validate/` — the validation layer

**This is part 2 of 3.** If you have been dropped into this package with no other context, read
this section and you will know what you are holding.

```
1 EXTRACTION            2 VALIDATION            3 BUGFIX
src/data_extract/       src/validate/           an agent + configs/
      │                       │                       │
      ├─ writes ─────────────►│                       │
      │  fundamentals_facts   │                       │
      │  fundamentals_history │                       │
      │  fundamentals_reason_codes                    │
      │                       ├─ writes ─────────────►│
      │                       │  fundamentals_check   │
      │                       │  (MUTATES NOTHING ELSE)
      │                       │                       ├─ records the outcome in
      │◄──────────────────────┴───────────────────────┤  configs/fundamentals/
      │   rebuild, then re-run part 2 to PROVE it     │  fundamentals_check.json
```

Each part names the artifact it hands to the next. Part 2 — this package — reads the tables
part 1 wrote, writes a ranked queue of findings, and **changes nothing else, ever**. Part 3 is
an agent that reads the queue, challenges it, fixes what is real, and records what it decided.

---

## 1. Why this exists

Nobody validates data by re-deriving a value and checking that it matches itself. Every tier
here is **provenance-independent**: it plays the filer's own disjoint evidence against itself
(a number we derived vs one the filer published separately; one filing's statement of a period
vs another's; a declared total vs the leaves that foot to it), or it plays peers against each
other.

**The goal is not 100%.** Compustat runs more than 2,500 checks *and* mandatory human review,
forever, and still ships errors. The job here is to produce a **short, ranked, explained** list
that a reviewer can actually work — not to certify anything.

**Nothing gates.** Not one finding blocks the nightly fill of `fundamentals_facts` or
`fundamentals_history`. This is the SEC's own warn-over-reject precedent, taken deliberately
(decision 45): one filer's bad quarter must never stall the other 499.

---

## 2. How to run it

```bash
PY="$HOME/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe"
```

| what you want | command |
|---|---|
| the nightly full-table pass | `"$PY" -m src validate fundamentals --tier 1` |
| nightly tiers 2–3, only where a filing landed | `"$PY" -m src validate fundamentals --tier 2,3 --since 2026-08-20` |
| one ticker, everything | `"$PY" -m src validate fundamentals -t AAPL` |
| the tuned stress set | `"$PY" -m src validate fundamentals --roster in_sample` |
| the never-tuned set (a finding here generalises) | `"$PY" -m src validate fundamentals --roster out_of_sample` |
| **a new field's acceptance sheet** | `"$PY" -m src validate fundamentals --field capex --roster in_sample` |
| explore a threshold, touch no table | `"$PY" -m src validate fundamentals -t AAPL --no-write --report /tmp/r.md` |
| one check only | `"$PY" -m src validate fundamentals --check peer_ratio --roster all` |
| what does this tool even test? | `"$PY" -m src validate checks` |

### The rosters are not interchangeable

`in_sample` (26 tickers) **was chosen because it broke things**, and every rule in the resolver
was tuned on it — so a pass there proves CONSISTENCY, not generalisation. `out_of_sample` (26)
has zero overlap and was never tuned, so a finding there is a genuine generalisation failure.
`random_cold` is the only honest estimate of the error rate on an *arbitrary* ticker; both
designed rosters measure robustness to KNOWN-HARD shapes instead. Every ticker carries a
`_why` in `configs/fundamentals/fundamentals_rosters.json`, and that `_why` is the whole point
of having a roster rather than a list.

### Adding a new catalogue field: the acceptance procedure

1. The field is born `status: probation`. Its findings are recorded at `info` and are excluded
   from the work queue, so a half-finished field cannot drown the review.
2. Run `validate fundamentals --field X --roster in_sample`. That is the acceptance sheet.
3. Promotion to `status: active` requires the sheet clean **or** its gaps recorded in
   `configs/fundamentals/fundamentals_check.json` with evidence.

Requiring the register cells to be authored up front was rejected: you would have to guess the
register before measuring it, which is exactly how the withdrawn "UNH has no premiums" edit
nearly got written.

---

## 3. What the output is

Findings land in `fundamentals_check`, **append-only, `run_date` in the primary key**. A re-run
appends rather than overwriting, so "did this fire yesterday?" stays answerable and a
recalibrated check leaves both verdicts on the record.

### The finding payload — a self-contained investigation packet

Decision 47. The reviewing agent must be able to settle a finding **in one hop**, without
re-deriving anything:

| group | columns |
|---|---|
| identity | `run_date`, `check_name`, `ticker`, `field`, `period_key`, `finding_id` |
| classification | `tier`, `severity`, `substrate` |
| the claim | `observed`, `expected`, `deviation` |
| provenance | `as_of`, `source_concept`, `resolution_method`, `roll_up_children`, `root_anchor`, `role_uri`, `accession_number`, `edgar_url` |
| evidence | `detail` (JSON, check-specific, including a `why` string) |

The payload is deliberately fat. An identity-only row plus an on-demand join back to
`fundamentals_facts` does not work: a Tier-2/3 finding on a **derived** value — a TTM, a
`derived_identity` total, a computed ratio — has no single fact row to join to, so half the
queue would arrive with an empty provenance block.

`detail` is JSON and not prose. An agent parsing English to decide what to fix is the failure
mode this rebuild exists to remove. The `why` key inside it is for the human reading the
report.

### `finding_id` is the identity that survives runs

A 16-hex hash of `(check_name, ticker, field, period_key)` — and deliberately **not** of
`run_date`, `severity` or `observed`. A threshold retune must not resurrect a settled finding,
and a re-measured value must not orphan one.

### `period_key` is polymorphic, by grain

TEXT: the `as_of` for a history-grain check, the `period_end` for a facts-grain one, `''` for a
ticker-level check, and `start..end` for a series-grain one. One column rather than three
nullable ones, because a Postgres PK cannot contain a NULL and a sentinel date would be a lie.

### The severity ladder — a PROVABILITY ladder, not an impact one

Since nothing gates, severity is purely the reviewing agent's queue order, and the question it
answers is *how sure are we the number is wrong*:

| severity | means | what it obliges you to do |
|---|---|---|
| `critical` | **provably** wrong, or a structural contract is broken | Work it first. A PIT leak, a `dimensional_scope` hit, `Assets != Liabilities + Equity`, a duplicate `(ticker, as_of)`. These should be zero, and a non-zero is a regression somewhere upstream. |
| `high` | probably wrong, and a **named mechanism** says so | Work it. `basis_step`, `tag_switch_break`, unexplained `series_shape`, `peer_ratio`, a `coverage_field` hole the register does not excuse. |
| `medium` | a statistical **candidate** | Look; do not assume. `level_outlier`, `scale`, `series_shape` gaps with a known benign `dc_code`. |
| `info` | declared, quantified, no action expected | **Never enters the queue.** `register_cost`, `restatement_ledger`, `expected_absent_drift`, abstentions, every probation-field finding. |

An impact ladder was rejected: it needs a per-field weighting nobody has measured, and it hides
small provable defects that indicate a systematic bug.

---

## 4. ⚠ WHEN IT DOES NOT WORK

**The most important section here, and the one usually missing.** A zero from a check that
abstained is not a pass.

### Checks that ABSTAIN, and when

| check | abstains when | consequence |
|---|---|---|
| `peer_ratio` | fewer than 5 filers in the regime | `broker_dealer` is **never checked** — GS is the only one on the roster. `real_estate` has 2 in-sample. Reported by `peer_ratio_abstentions`. |
| `trend_break` | fewer than 4 prior periods | a filer's first year is invisible to it |
| `level_outlier` | fewer than 8 periods, or fewer than 3 defined log changes | ditto, and see the sign rule below |
| `series_shape` | a period grid shorter than 8 events | a recent IPO has no shape to classify |
| `coverage_field` | fewer than 4 filers in the regime | 0-of-4 is **not** evidence, and no `expected_absent` cell may be written from it |
| `sign_convention` *(5b-stats)* | the concept is absent from the balance map | reported, not silently passed |

### Known false-positive populations — expected, not bugs

* **`trend_break` on seasonal filers.** Retail Q4, KR's 16-week Q1, weather-driven utility
  quarters. An accepted cost of decision 59. The calibration pass reports the fire rate **by
  regime** so the population is a number rather than a surprise. Named remedy if it ever
  dominates the queue: `fiscal_quarter` is on every row since Phase 5, so switching the
  reference to the same fiscal quarter of the prior 3 years is a one-line change.
* **`coverage_field` on MIXED cells — which are the MAJORITY.** 31 of 48 industrial fields
  resolve for some filers and not others. That is the validator's real work queue, and no
  config rule can decide it; only the filing can.
* **`leaf_vs_total` at up to 25%.** A leaf sum and a declared total genuinely differ when the
  filer's linkbase omits a caption it nonetheless includes in the total. The finding is an
  invitation to look, not a claim that our roll-up is wrong.

### What the validator STRUCTURALLY CANNOT SEE

* **A defect in the filer's own calculation linkbase.** That is Tier 0 (XBRL-US DQC via
  Arelle), deferred. *Trigger to adopt:* Tier 3's checks converge (`leaf_vs_total` and
  `cross_vintage` stop disagreeing) while a full-universe run still surfaces wrong values no
  check explains. Cheapest first step is a **sampled offline audit** of ~300 filings stratified
  by regime × era, not a per-filing tier.
* **A whole regime wrong in the same direction.** `peer_ratio` goes blind: the peer median is
  wrong too. This is why it is `high` and never `critical`, and why it is corroboration for
  `basis_step` rather than a standalone verdict.
* **A value crossing zero.** `level_outlier` scores a log change, which does not exist across a
  sign flip or a zero. APA's revenue going to 0 / −$467M is `impossible_value`'s, not its.
* **Whether an aggregator agrees with us.** Tier 4 is deferred (decision 51). Boritz & No
  measure aggregators disagreeing with the filed 10-K at **6.5–7.7%**, ~10× the effect sizes
  Tier 3 measures — a noise floor above the signal. *Trigger to adopt:* Phase 9 finds wrong
  values Tiers 1–3 cannot explain, **on a field where the aggregator's basis is known to match
  ours.**
* **A dimensioned fact read as the group's, in general.** `dimensional_scope` tests the
  *provenance* for member-shaped tokens, which is what is decidable on the stored table —
  `entity_scope.consolidated_facts` drops dimensioned rows at extraction and does not keep the
  `dim_*` columns, so a filter regression arrives here as an ordinary-looking row. The primary
  defence stays in `entity_scope`, tested where it lives. This is the second lock.

### Two facts about absence that everything here depends on

* **`companyfacts` can prove a concept PRESENT and can NEVER prove one ABSENT.** It publishes
  no company-extension taxonomy and silently drops dimensioned facts. Every coverage claim must
  be measured off `filing.xbrl()`.
* **A `not_disclosed` code is a statement about OUR CONCEPT MAP, not about the filing.** It
  cannot distinguish "the filer has no such line" from "the filer tagged it under a name we do
  not know". 68% of all reason codes are `not_disclosed`, so most of the absence evidence in
  this system is currently unfalsifiable. Closing that is the E-4 work: recording, on the
  unresolved stub row, the concepts the filer DID tag under the same `role_uri` that we failed
  to recognise.

---

## 5. The rules for changing it

1. **CHALLENGE THE CHECK BEFORE CHALLENGING THE DATA.** XBRL-US's own DQC_0118 documentation:
   *"inconsistencies reported to filers can be overwhelming as many don't represent real
   errors."* This repo has earned that lesson twice independently — **745 correct rows** nulled
   by over-strict Q4 guards, and the **withdrawn "UNH has no premiums"** register edit, where
   the check's premise was wrong and the numbers were right. Every check declares an
   `expected_fire_rate_ceiling`, and the report labels anything above it a **threshold bug**.
2. **Never accept a finding without filing-level evidence.** A reason code is the resolver's
   verdict, not evidence. Open the accession and read the statement.
3. **A `fixed` outcome must leave a regression test.** Four defects were once *created by* a
   set of fixes and were visible only on a full re-sweep. Per-finding closes are provisional
   (`regression_swept: false`) until a batched full-roster sweep has seen them.
4. **`configs/` is a risk zone: PROPOSE, never apply.** A large share of real fundamentals fixes
   *are* config — the `never_use` entry that closed MTB and AXP, a `by_ticker` widening, a
   cutover entry. Record `fix_kind: config_proposed`; the finding stays OPEN until approved.
5. **Thresholds live in code with their measurement in the docstring. Baselines live in
   `configs/fundamentals/fundamentals_baselines.json` and only move with evidence.** A
   threshold is *behaviour* and will be retuned repeatedly; a baseline is a *measurement* and a
   test fails on its degradation.
6. **Never hand-list what `CHECK_REGISTRY` drives.** The CLI, the report, the fire-rate table
   and the calibration pass all enumerate it.
7. **Re-running the validator is NOT verification of a fix.** It reads stale rows and reports a
   false green. The rebuild path branches on what you touched:

   | changed | rebuild before re-validating | cost |
   |---|---|---|
   | `build_history.py`, `reason_codes.py`, a `_FORMULAS` entry | `fundamentals-history --rebuild-history -t X` | ~2.5 min/ticker, **no network** |
   | `xbrl_linkbase.py`, `periods.py`, `fundamentals_kpis.json` — a RESOLUTION bug | `fundamentals --rebuild -t X` (deletes four tables for X, refetches) | network |
   | a register the validator itself reads | none | seconds |

---

## 6. Layout

```
src/validate/
  README.md                   this file
  cli.py                      `python -m src validate fundamentals`
  outliers.py                 the shared MAD kernel (scores a LOG CHANGE -- decision 60)
  external/                   Tier 4 adapters. DEFERRED; see external/__init__.py
  fundamentals/
    validator.py              FundamentalsValidator -- the ONE implementation
    substrate.py              every frame, loaded ONCE, projected, passed down
    finding.py                the investigation packet + `finding_id`
    check_register.py         configs/fundamentals/fundamentals_check.json
    report.py                 fire rates, the queue, register health
    checks/__init__.py        CHECK_REGISTRY
    checks/tier1_value.py     deterministic per-value rules, on `fundamentals_history`
    checks/tier2_series.py    series behaviour, on `fundamentals_facts`
    checks/tier3_internal.py  the filer's own disjoint evidence, on `fundamentals_facts`
```

Tests live in `tests/validate/`, not here. A check is instantiated against a synthetic
`Substrates` — **no DB, no CLI, no network** — and the bar for every check is: *plant exactly
one violation; it fires exactly once and nothing else does.* A check that cannot be planted
cannot be trusted.

Three scripts were absorbed and deleted: `verify_fundamentals_history.py` (its 8 gates are
Tier 1), `audit_absence_evidence.py` (it is `coverage_field`'s oracle) and
`measure_total_liabilities_legs.py` (it is `cross_identity`'s evidence).
`sweep_fundamentals_resolution.py` and `report_fundamentals_sweep.py` stay — they are
threadpool EDGAR fetches, a different job with a different cost model.
