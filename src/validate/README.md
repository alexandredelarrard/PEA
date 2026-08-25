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
      │                       │  fundamentals_check_run
      │                       │  fundamentals_check_status
      │                       │  fundamentals_check_fix
      │                       │  (MUTATES NOTHING ELSE)
      │◄──────────────────────┴───────────────────────┤
      │   rebuild, then re-run part 2 at the SAME     │  the ROW-COUNT DROP
      │   scope — the DELTA is the proof              │  is the record
```

Each part names the artifact it hands to the next. Part 2 — this package — reads the tables
part 1 wrote, writes a ranked list of clusters, and **changes nothing else, ever**. Part 3 is
an agent that reads the list, challenges it, fixes what is real, and proves it by re-running
part 2 at the same scope and measuring the drop.

**There is no outcome file.** A git-tracked JSON register used to record every settled finding
and the validator subtracted them before writing. That is gone (see §5): the ledger records
every finding of every run, so a smaller row count between two comparable runs has exactly one
cause. The only thing a human still asserts is a `wontfix`, and it expires by itself.

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
| **both designed rosters** (the flag repeats) | `"$PY" -m src validate fundamentals --roster in_sample --roster out_of_sample` |
| **re-read a recorded run — no re-run, no writes** | `"$PY" -m src validate report --run-id 3df52ae9af75` |
| **record a wontfix** (a NUMBER in the note is enforced) | `"$PY" -m src validate status set <cluster_id> --note "..."` |
| tolerate only ONE check's findings on a cluster | `"$PY" -m src validate status set <cluster_id> --check peer_ratio --note "..."` |
| undo one (all of them, or just one check's) | `"$PY" -m src validate status clear <cluster_id> [--check peer_ratio]` |
| **record a FIX and waive its benign residue, atomically** | `"$PY" -m src validate fix record <cluster_id> --layer extraction --root-cause "..." --evidence '{"accessions": [...]}' --commit SHA --test PATH --waive "peer_ratio:2 findings, 8.3% vs 3.5% median"` |
| what was done to this cluster, and what is still pending | `"$PY" -m src validate fix show <cluster_id>` |
| what does this tool even test? | `"$PY" -m src validate checks` |

Reports land in `reports/validate/YYYY-MM-DD/<scope>.md`, with a `.json` beside it carrying the
same content as structured data — markdown is the HUMAN artifact, JSON is the AGENT artifact.

**`validate report` re-renders from the tables and never re-runs the checks.** That distinction
matters: re-running to read a report is how a stale-row false green happens. To learn what a
recorded run found, read it; to test a fix, rebuild first and then run again.

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
3. Promotion to `status: active` requires the sheet clean **or** its gaps recorded as
   `wontfix` clusters with quantified evidence (`validate status set`).

Requiring the catalogue's exception cells to be authored up front was rejected: you would have
to guess them before measuring anything, which is exactly how the withdrawn "UNH has no
premiums" edit nearly got written.

---

## 3. What the output is

A **ranked list of clusters**, plus the three tables it is derived from.

### The unit of work is a CLUSTER, not a finding

One `(ticker, field)` defect. Every check that fired on it is a WITNESS to the same thing, not
a separate job.

This is the single most important shape in the package, and it was learned by measurement.
Calibration run 2 produced **11,926 findings** — which were never 11,926 bugs. 739 of 1,893
`(ticker, field)` series carried 8,160 of the queue; MCD `capex` alone tripped **nine checks**
for 54 findings. Ordered by severity and then alphabetically, as the old report did, nothing in
that list said which fix closed the most rows.

`cluster_id` is a 12-hex hash of `(ticker, field)` and nothing else. `check_name` and
`period_key` are excluded on purpose: a field that is wrong for 40 quarters is one fix, not
forty, and nine checks agreeing is the CORROBORATION signal — the strongest prior an agent gets
before opening a filing.

Clusters roll up by field into **families**, because breadth is diagnostic: one ticker with a
broken `capex` means fix the filer; forty tickers with a broken `incomeTaxExpense` means the
catalogue is wrong. `routing_hint` says which, from constants in `clusters.py`. *Read its
caveat: on a roster where a broad statistical check touches nearly every ticker, the hint goes
flat and the report says so rather than pretending it is evidence.*

### The score is a POLICY, not a fact

```
cluster_score = (Σ over findings of w(severity) × w(tier)) × corroboration(n_checks)
tier 1/2/3 = 4/2/1   critical/high/medium/info = 4/2/1/0   corroboration = 1 + 0.25×(checks−1)
```

`info` is weighted **zero**, not small, so no volume of declared-and-expected findings can bury
real work — and because corroboration MULTIPLIES, ten checks agreeing that something is benign
is still benign.

The weights are module constants **printed in every report** precisely so they can be argued
with, and they are meant to be retuned once somebody has read a list and disagreed. That has
already happened once: volume-only scoring put HCA `minorityInterest` (62 findings, **two**
checks) on top and left MCD `capex` (55 findings, **ten** checks) off the menu entirely. One
check firing 62 times is one opinion repeated; ten independent checks agreeing is ten arguments
for the same conclusion. With the corroboration term MCD leads at 481 and HCA sits at 305.

### The four tables

| table | grain | what it is for |
|---|---|---|
| `fundamentals_check` | one FINDING per run | the ledger. Nothing is ever subtracted |
| `fundamentals_check_run` | one `(run_id, check_name)` | WHAT the run looked at, and what each check did |
| `fundamentals_check_status` | one `(cluster_id, check_name)` | a human's `wontfix`, and nothing else. `check_name = ''` is the whole cluster |
| `fundamentals_check_fix` | one `(cluster_id, run_id_after)` | an INTERVENTION: what was done, at which layer, and what it measurably closed |

A **waiver** is a STATE — mutable, self-expiring, and it says "this finding is real and we
tolerate it". A **fix** is an EVENT — append-only, never revised, and it says "we intervened,
here is what and why". They are separate tables because they are separate kinds of claim, and
because **neither ever removes a row from `fundamentals_check`**. A waiver is applied when a
report is RENDERED; a fix row is read only to decide whether a settlement is claimable.

### What it takes to call a cluster SETTLED

A cluster does not have to reach zero. MCD `capex` went 55 → 4 and all 4 are benign. Four
conditions, each blocking a different way to fake a closed defect:

1. **no UNWAIVED queue-severity finding is left.** `info` never needs a waiver — nothing reads
   it as work, so it cannot be hiding any;
2. **every waiver is still within its own `findings_at_decision`.** One that grew has expired,
   and a cluster resting on an expired judgement is `reopened`, not settled;
3. **a fix row exists at this `scope_hash`.** Without this, waiving each check in turn
   manufactures a settlement with nobody having fixed anything — the deleted suppression
   register, reassembled from parts;
4. **that fix row reduced the queue** (`queued_after < queued_before`).

Fail 3 or 4 with 1 and 2 satisfied and the cluster reads `wontfix`: **tolerated, not solved**,
and the report says so. That is why no new status word was added — a fully-waived, unfixed
cluster already *is* what `wontfix` means. `Ledger.qualifying_fix` is the only place conditions
3 and 4 are written down; two copies of a rule this load-bearing drift.

A **no-improvement fix row is still recordable** — correcting a wrong-but-plausible value where
no check was firing is a real fix — but `fix record` warns loudly and it can never settle
anything. Permissive to record, strict to settle.

**`run_id` is what makes a delta mean anything.** It hashes (date, tickers, fields, tiers), and
`scope_hash` is the same without the date. Two runs are comparable **iff their scope hashes
match** — differencing a 54-ticker baseline against a one-ticker re-validation would report
~11,800 findings "closed". When no comparable prior run exists the report omits the delta and
says why; a first run must never render as a trend.

`run_id` is also in the primary key, and that was learned the hard way: without it a `-t MCD`
run and a roster run on the same day collided on every shared ticker, and the first run was
left claiming 35 checks and one surviving finding.

### The finding payload — a self-contained investigation packet

Decision 47. The reviewing agent must be able to settle a finding **in one hop**, without
re-deriving anything:

| group | columns |
|---|---|
| identity | `run_date`, `run_id`, `check_name`, `ticker`, `field`, `period_key`, `finding_id`, `cluster_id` |
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
`run_date`, `severity` or `observed`. A threshold retune moves severities in bulk (347 of them
in one change) and must not re-key a finding, or every delta would read as a mass close-and-
reopen.

It also **is** the primary key, hashed, so two findings sharing one id are two rows that would
upsert onto each other. `findings_frame` refuses them outright: run 2 emitted 12,462 findings
and stored 11,926, and the 536 that vanished did so in silence.

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
| `info` | declared, quantified, no action expected | **Never enters the queue.** `catalogue_exclusion_cost`, `restatement_ledger`, `expected_absent_drift`, abstentions, every probation-field finding. Weighted **zero** in the cluster score, so no volume of it can bury real work. |

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
3. **A `fixed` outcome must leave a regression test**, named with the `cluster_id`. Four
   defects were once *created by* a set of fixes and were visible only on a full re-sweep, so
   a close on the affected tickers alone is provisional until a wider run has seen it.
4. **`configs/` is a risk zone: PROPOSE, never apply.** A large share of real fundamentals fixes
   *are* config — the `never_use` entry that closed MTB and AXP, a `by_ticker` widening, a
   cutover entry. The cluster stays OPEN until a human approves the diff.
5. **NOTHING IS SUBTRACTED FROM THE LEDGER, and a `wontfix` expires by itself.** This replaced
   a git-tracked JSON register that recorded every settled finding and was subtracted before
   the write. Its own documentation opened with "THE REGISTER IS NOT A SUPPRESSION LIST" —
   which is the kind of thing a design only has to say when its shape makes the opposite easy —
   and it made the ledger's row count ambiguous between "fixed" and "hidden".

   Now every finding of every run is written. A human's only assertion is a `wontfix` in
   `fundamentals_check_status`, applied when the report is **rendered** and never when a row is
   written, so the table and the checks always agree. It requires a QUANTIFIED note (the CLI
   refuses one with no numeral in it), it records `findings_at_decision`, and it **REOPENS
   automatically the moment the cluster grows past that** — a judgement made about 3 findings
   is not a judgement about 30. The report's `wontfix` footer is never omitted. Between them,
   those two properties are what the deleted register's staleness report used to do by hand.

   `fundamentals_check_fix` was added for the same reason in reverse: a FIX had nowhere to be
   recorded, so cluster `1c9a517eaa47` was fixed and its only trace was a commit sha. It is
   append-only, `fix record` refuses what it cannot verify, and **no renderer may filter
   findings using it** — which is pinned by a test that counts `fundamentals_check` rows with
   and without any of it present. Everything else here reverts with `git revert`; a suppression
   leak does not.
6. **Thresholds live in code with their measurement in the docstring. Baselines live in
   `configs/fundamentals/fundamentals_baselines.json` and only move with evidence.** A
   threshold is *behaviour* and will be retuned repeatedly; a baseline is a *measurement* and a
   test fails on its degradation.
7. **Never hand-list what `CHECK_REGISTRY` drives.** The CLI, the report, the fire-rate table
   and the calibration pass all enumerate it.
8. **Re-running the validator is NOT verification of a fix.** It reads stale rows and reports a
   false green. The rebuild path branches on what you touched:

   | changed | rebuild before re-validating | cost |
   |---|---|---|
   | `build_history.py`, `reason_codes.py`, a `_FORMULAS` entry | `fundamentals-history --rebuild-history -t X` | ~2.5 min/ticker, **no network** |
   | `xbrl_linkbase.py`, `periods.py`, `fundamentals_kpis.json` — a RESOLUTION bug | `fundamentals --rebuild -t X` (deletes four tables for X, refetches) | network |
   | a check module or a threshold | none — re-validate directly | seconds |

   Then re-validate **at the ORIGINAL scope** and report the delta as a number: `55 → 4`, on
   QUEUE severities. A different scope produces an incomparable `run_id` and the tooling
   refuses to difference them. The drop does not have to reach zero — see the settlement rule
   above — but **a fix with no measured drop cannot settle the cluster**.

   Then **record it**, because the drop alone cannot say what you did:
   `validate fix record <cluster_id> --layer L --root-cause "..." --evidence '{...}'
   --commit SHA --test PATH [--waive "check:quantified note"]`. Everything but those flags is
   derived from the ledger, the CLI refuses what it cannot verify, and the fix row plus its
   waivers land atomically.

   To READ a run that already happened, use `validate report --run-id X`, which re-renders from
   the tables without re-running anything. To read what was DONE to a cluster, use
   `validate fix show <cluster_id>`.

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
    scope.py                  RunScope -- what a run covered, hashed, so runs can be differenced
    substrate.py              every frame a CHECK reads, loaded ONCE, projected, passed down
    finding.py                the investigation packet, `finding_id`, `cluster_id`
    ledger.py                 the only READ-BACK of the three tables; comparable runs
    clusters.py               findings -> clusters -> field families, scored and routed
    report.py                 the health gate, the delta, the rankings, the wontfix footer
    checks/__init__.py        CHECK_REGISTRY
    checks/tier1_value.py     deterministic rules. SIX contract checks on
                              `fundamentals_history`, the other 13 on `fundamentals_facts`
    checks/tier2_series.py    series behaviour, on `fundamentals_facts`
    checks/tier3_internal.py  the filer's own disjoint evidence, on `fundamentals_facts`
```

### Which table a check reads, and why it is not a matter of taste

**A check that asks about the TABLE reads `history`. A check that asks about a NUMBER reads
`facts`.** Six checks are on the first side of that line -- `grain`, `column_contract`,
`code_vocabulary`, `unexplained_null`, `pit_leak`, `coverage_universe` -- and everything else
in all three tiers is on the second.

The line was drawn by a measurement. `Finding.edgar_url` is built from
`(cik, accession_number)`, and `fundamentals_history` carries neither: it has 69 columns and
its only provenance is `publication_form` / `is_amendment` / `amended_fields`, which is a form
TYPE and never a document. So on the 2026-08-24 calibration run:

| tier | substrate | findings | with `edgar_url` |
|---|---|---|---|
| 1 | history | 1,427 | **0 -- 0.0%** |
| 1 | facts | 10 | 0 -- 0.0% |
| 2 | facts | 7,369 | 5,731 -- 77.8% |
| 3 | facts | 3,120 | 3,120 -- **100%** |

Not a partial gap. **No Tier-1 finding could be traced to a filing**, criticals included, so
the tier was unactionable however well it was ranked -- agent B's first move is to open the
filing, and there was nothing to open. After the move, Tier 1 reads **876 of 1,386 (63.2%)**,
and every check that implicates a filing at all is at **100%**. The 510 without a URL are the
500 catalogue-configuration diagnostics, where no accession caused a `never_use` entry, and 10
pre-existing ticker-grain checks.

Nothing was lost by moving and something was gained. The balance-sheet identity fails on the
SAME seven filers on either substrate (UNH, PGR, AMT, EQIX, VRT, SPG, NVDA), but `facts`
exposes 4,763 testable statements against history's 3,229 -- a filing carries comparatives and
each is a separate published claim -- so it finds 144 breaks to history's 64. `filing_lag`
went from 1 finding to 11, all real: SMCI's delisting-era restatements, ADM's 2024 accounting
investigation, an SPG 10-Q/A. History missed them because an amendment that changes no
extracted value never becomes a history row.

**The six that stayed test properties `facts` cannot express** -- a 69-column ORDERED contract
(facts is long), a null CELL (in facts a missing fact is an absent ROW), the reason-code
vocabulary, and the no-leakage snapshot grain. Porting them would delete them, not relocate
them. All six carry `expected_fire_rate_ceiling=0.0` and all six fired **zero** on the live
roster, which is what a tripwire is meant to do: they exist to catch a bug in `build_history`,
the one defect class genuinely history's own.

ETN's 2012-11-14 row is the specimen. `totalLiabilities` of **-$8,237,223,652** against
`totalAssets` of **$4,776,348**, tagged `derived_identity` -- computed across the Irish
redomicile's holdco shell and a carried-forward equity from the operating company. It has no
counterpart anywhere in `facts`, because no filer ever tagged it. That row is why history
keeps its tripwires, and why it keeps nothing else.

The split is pinned by `tests/validate/fundamentals/test_substrate_contract.py`, which fails
if a Tier-1 value check is ever added back on the history substrate.

Tests live in `tests/validate/`, not here. A check is instantiated against a synthetic
`Substrates` — **no DB, no CLI, no network** — and the bar for every check is: *plant exactly
one violation; it fires exactly once and nothing else does.* A check that cannot be planted
cannot be trusted.

Three scripts were absorbed and deleted: `verify_fundamentals_history.py` (its 8 gates are
Tier 1 -- and note that the six still on the history substrate are exactly the ones that gate
the TABLE), `audit_absence_evidence.py` (it is `coverage_field`'s oracle, and it always read
`fundamentals_facts`, which is half of why that check moved) and
`measure_total_liabilities_legs.py` (it is `cross_identity`'s evidence).
`sweep_fundamentals_resolution.py` and `report_fundamentals_sweep.py` stay — they are
threadpool EDGAR fetches, a different job with a different cost model.
