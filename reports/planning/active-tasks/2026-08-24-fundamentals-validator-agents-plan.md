# Implementation Plan: the validator's triage loop — clusters, the ledger, and agents A/B

**Date Created**: 2026-08-24
**Planning Phase**: 2 of 3 (FIC Workflow)
**Based on**: `reports/planning/active-tasks/2026-08-24-fundamentals-validator-phase5b-plan.md`
(5b-core.1 → .4 complete) and calibration run 2 (54 tickers, 11,926 findings)
**Next Phase**: Implementation (`/implement`)

## Overview

5b-core built a validator that produces findings. Calibration run 2 proved it works and, in the
same breath, proved the loop around it does not scale: **11,926 findings is not 11,926 bugs**,
and nothing in the system says which fix closes the most rows.

This plan rebuilds the loop around a **cluster** — one `(ticker, field)` defect — instead of
around a finding. It makes `fundamentals_check` readable (nothing reads it today), replaces the
JSON register with database status, and rewrites agents A and B so A ranks and B fixes.

## Current State Analysis

Measured against run 2, not assumed:

| fact | evidence |
|---|---|
| `fundamentals_check` is **write-only** | no module in `src/validate/` reads it back |
| the report shows **57 of 10,898** queue findings | `render()` truncates to 25/severity; the CLI writes that truncated string to the file |
| its docstring claims the markdown is untruncated | false as written — `report.py:38` |
| prioritisation is **severity then alphabetical** | `ValidationRun.queue` sorts `(rank, check_name, ticker, field)` |
| **no root-cause clustering exists anywhere** | — |
| 79% of findings sit in series that trip **2+ checks** | 739 of 1,893 `(ticker,field)` series carry 8,160 of 10,362 findings |
| the worst single series trips **9 checks** | MCD `capex`, 54 findings |
| **536 findings never reach the table** | 12,462 emitted vs 11,926 stored — PK collisions in `cross_vintage` (526), `q4_footing` (6), `leaf_vs_total` (4) |
| **347 findings are wrongly `info`** | `series_shape` tests the gap code before the shape, applying a start-of-history rationale to `interior_gap` (340) and `early_stop` (7) |
| the register **suppresses** rows before writing | `validator.run()` subtracts settled findings — a row-count drop is then ambiguous |
| **run scope is not recorded** | a scoped run writes a partial `run_date`; nothing distinguishes *fixed* from *out of scope* |
| Tier 1 findings carry **no provenance** | 1,427 history-substrate rows have NULL `source_concept` / `accession_number` / `edgar_url` |

Key files: `src/validate/fundamentals/{validator,report,finding,substrate,check_register}.py`,
`src/validate/fundamentals/checks/tier{1,2,3}_*.py`, `src/validate/cli.py`,
`src/data_store/schema.py`, `sql/schema.sql`, `.claude/agents/fundamentals-{validate,triage}.md`.

## Desired End State

- `fundamentals_check` is an **immutable ledger**. Every finding every run, nothing suppressed.
  A drop in row count between two runs *of the same scope* is the proof a fix worked.
- Scope is recorded, so that comparison is sound.
- **One report**, at `reports/validate/YYYY-MM-DD/<scope>.md`, ranking clusters by leverage.
- **Agent A**: confirms scope → runs → clusters → ranks → reports. **No research, no fixes.**
- **Agent B**: asks which of the top 5 → researches → plans → go/no-go → implements → rebuilds
  → calls A as a sub-agent to re-validate → reports the delta.
- The JSON register is gone; `wontfix` lives in the database and **cannot silently rot**.

## Locked Design Decisions

| # | decision | rationale |
|---|---|---|
| D1 | Cluster key is `(ticker, field)`. `check_name` is **evidence inside** a cluster, never a key | MCD `capex` is one defect, not nine issues. 9 checks agreeing is a corroboration signal |
| D2 | Clusters roll up into **field families**; the report shows both levels | narrow → fix the filer; wide → fix the field |
| D3 | `cluster_score = Σ over findings of w(severity) × w(tier)` | volume enters naturally; mixed-tier clusters need no fudge |
| D4 | Weights: tier 1/2/3 = **4/2/1**; critical/high/medium/info = **4/2/1/0** | starting values, printed in the report, retuned after list #1 |
| D5 | **Stop subtracting** settled findings. Always write everything | a row-count drop must have exactly one cause |
| D6 | Status lives in the **database**, not JSON | `fundamentals_check_status`, keyed on `cluster_id` |
| D7 | The register dies: `fundamentals_check.json`, `check_register.py`, `register_cost`, `register_coverage` | superseded by D5/D6 |
| D8 | A `wontfix` records `findings_at_decision` and **auto-reopens if the cluster grows** | replaces the anti-suppression guard that `register_coverage` provided |
| D9 | Report lives at `reports/validate/YYYY-MM-DD/<scope>.md` | user-specified |
| D10 | Agent A does **no research** | keeps A cheap and stops it anchoring B |
| D11 | Agent B calls A as a sub-agent to verify | the re-validation must be at the **original scope** to be comparable |

**Assumption flagged for confirmation during implementation**: D4's weights make MCD `capex`
(54 × T2 × high ≈ 216) outrank VRT's 7 Tier-1 criticals (7 × 4 × 4 = 112). That follows from
volume-dominant scoring, which was chosen deliberately. Retune after list #1 if it reads wrong.

**Assumption flagged**: a go/no-go gate is retained between B's plan and B's implementation.

## Out of Scope

- ~~**Tier-1 provenance backfill**~~ — **PULLED IN AND DONE, 2026-08-25** (Phase 7). Not as a
  join, which is unsound on a carry-forward snapshot, but by moving eight of the fourteen
  Tier-1 history checks onto `fundamentals_facts`. Coverage 0.0% → 63.2%, and 100% of the
  checks that implicate a filing.
- Tier 0 (DQC/Arelle) and Tier 4 (aggregator cross-checks) — still deferred per 5b-core.
- 5b-stats checks (`frozen_series`, `periodicity`, `sign_convention`) — after Phase 9.
- Field-importance weighting beyond tier. Tier IS the importance proxy, per D3/D4.
- `fundamentals_baselines.json` — remains open from 5b-core.4.

---

## Implementation Approach

### Phase 1: the ledger and the run-scope record ✅

**Goal**: make `fundamentals_check` an honest, comparable, readable ledger. Everything else
depends on this, which is why it is first.

**Changes**:

1. `sql/schema.sql` + `src/data_store/schema.py`:
   - [x] New table `fundamentals_check_run` — one row per `(run_id, check_name)`:
     ```
     run_id TEXT, run_date DATE, check_name TEXT,          -- PK (run_id, check_name)
     scope_roster TEXT, scope_tickers INT, scope_ticker_list TEXT,   -- json array
     scope_fields TEXT, scope_tiers TEXT,                  -- json array / '1,2,3'
     tier INT, substrate TEXT,
     examined INT, queued INT, info INT, ceiling DOUBLE PRECISION,
     abstained BOOLEAN, over_ceiling BOOLEAN
     ```
     Scope columns repeat per check row — denormalised on purpose, matching this repo's flat
     table convention and keeping a single read.
   - [x] `run_id` = sha256[:12] of `(run_date, sorted tickers, sorted fields, sorted tiers)`.
     Two runs are **comparable iff their scope hash matches** ignoring `run_date`.
   - [x] New table `fundamentals_check_status`:
     ```
     cluster_id TEXT PRIMARY KEY, ticker TEXT, field TEXT,
     status TEXT,                  -- 'wontfix' only; 'open'/'settled' are DERIVED
     note TEXT,                    -- quantified evidence, required
     findings_at_decision INT,     -- D8: auto-reopen trigger
     decided_at DATE
     ```
   - [x] Add to `fundamentals_check`: `run_id TEXT`, `cluster_id TEXT`.

2. `src/validate/fundamentals/finding.py`:
   - [x] `cluster_id()` = sha256[:12] of `(ticker, field)`. Stable across runs by construction.
   - [x] Add `cluster_id` and `run_id` to `FINDING_COLUMNS`.

3. `src/validate/fundamentals/validator.py`:
   - [x] **Remove register subtraction** from `run()` (D5). Delete `settled`/`settled_total`.
   - [x] Emit the `fundamentals_check_run` frame from the existing `CheckOutcome` list.
   - [x] `write()` writes both tables in one call.

4. **Fix the 536-finding collision**:
   - [x] `findings_frame()` asserts `finding_id` uniqueness and raises with the offending keys.
   - [x] `cross_vintage`, `q4_footing`, `leaf_vs_total` collapse to **one finding per
     `(ticker, field, period_key)`**, reporting the worst deviation and carrying the rest in
     `detail` (`n_collapsed`, the vintage pairs). The grain must match `finding_id`.

**Verification**:
- [x] `rtk "$PY" -m pytest tests/validate -x`
- [x] New test: two runs with different tickers produce different `run_id`; same scope, same id.
- [x] New test: `findings_frame()` raises on a duplicated `finding_id`.
- [x] Re-run 54-ticker calibration; **emitted count == stored row count** (was 12,462 vs 11,926).

---

### Phase 2: severity correctness — the `series_shape` ladder ✅

**Goal**: the score is a function of severity, so severity must be right *before* anything is
ranked. 347 findings are currently mislabelled `info`.

**Changes**:

1. `src/validate/fundamentals/checks/tier2_series.py`:
   - [x] Reorder `_severity_for_shape`: test **shape before gap code**, or make
     `_BENIGN_GAP_CODES` shape-conditional. `insufficient_quarters` is benign **only** for
     `late_start` — its own rationale says *"at the start of a history"*.
   - [x] `interior_gap` gets an explicit branch instead of falling through to the benign
     short-circuit (340 findings across 45 tickers land there today).
   - [x] `early_stop` reaches its existing HIGH branch (7 findings, currently unreachable).
   - [x] `sparse` → `info` deferral to `periodicity` is **retained** — that one is correct.
   - [x] Investigate the 186 `sparse` rows carrying **no gap code at all**; a null code is not
     the same as `not_disclosed` and should not render identically.

**Verification**:
- [x] Planted test per shape × gap-code combination asserting the severity, including
  `early_stop + insufficient_quarters` → HIGH (the currently-unreachable branch).
- [x] Re-run: `series_shape` info count drops by ~347; queue rises correspondingly.
- [x] `series_shape` stays under its 15% ceiling after the reclassification, or the ceiling is
  re-derived with the measurement recorded.

---

### Phase 3: retire the register ✅

**Goal**: delete the JSON register and the two checks that existed to police it.

**Changes**:
- [x] Delete `configs/fundamentals/fundamentals_check.json`.
- [x] Delete `src/validate/fundamentals/check_register.py`.
- [x] Delete checks `register_cost` (446 info findings) and `register_coverage` (54).
- [x] Remove register wiring from `validator.py`, `report.py`, `cli.py`
      (`_register_section`, `stale()`, `unswept_fixes()`, `open_proposals()`).
- [x] Delete the register assertions. *(No `test_check_register.py` existed — the register tests lived inside `test_planted_violations.py`; three were removed and the D5 one was rewritten as a claim about the RUN.)*
- [x] `src/validate/README.md`: rewrite the lifecycle section around D5/D6/D8.

> **The guard that goes with it**: `register_coverage` existed so the register could not
> silently become a suppression list. D8 replaces it — a `wontfix` reopens automatically if its
> cluster grows, and the report always lists wontfix clusters with their age. Phase 5 must
> deliver that section, or this deletion removes a safeguard without a replacement.

**Verification**:
- [x] `rtk grep -rn "check_register\|fundamentals_check.json" src/ tests/ configs/` → no hits.
- [x] `rtk "$PY" -m pytest tests/validate -x`
- [ ] ~~`validate checks` lists 33 checks (was 35).~~ **35** — the two `register_*` checks were kept and renamed; see deviation 1 below.

---

### Phase 4: clustering and ranking ✅

**Goal**: turn 11,926 rows into a ranked list of fixable issues.

**Changes**:

1. New `src/validate/fundamentals/clusters.py`:
   - [x] `TIER_WEIGHTS = {1: 4, 2: 2, 3: 1}`, `SEVERITY_WEIGHTS = {critical: 4, high: 2,
     medium: 1, info: 0}` — module constants with the D4 rationale in the docstring.
   - [x] `Cluster`: `cluster_id, ticker, field, findings, score, checks (with counts),
     tiers, severities, first_seen, last_seen, runs_open, status`.
   - [x] `Family`: `field, clusters, total_score, tickers, breadth, routing_hint`.
   - [x] `build_clusters(findings, status, runs) -> list[Cluster]`.
   - [x] **Routing hint** (the DQC_0118 lesson, made mechanical): a family is
     `likely-check-or-catalogue` when it spans **≥ 5 tickers AND ≥ 30% of the run's roster**;
     otherwise `likely-filer`. Both numbers are constants, printed in the report, tunable.
   - [x] `derive_status()`: `open` by default; `settled` when a `cluster_id` present in a prior
     comparable run is absent from the latest; `wontfix` from `fundamentals_check_status`,
     **auto-reopened** when `findings > findings_at_decision` (D8).

2. New `src/validate/fundamentals/ledger.py`:
   - [x] The first and only **read-back** of `fundamentals_check` /
     `_check_run` / `_check_status`. Mirrors `substrate.py`'s single-access discipline.
   - [x] `comparable_runs(run_id)` — prior runs whose scope hash matches. This is what makes
     "fewer rows = proof of fix" sound rather than assumed.

**Verification**:
- [x] Unit tests on a synthetic ledger: MCD-shaped 9-check series collapses to ONE cluster.
- [x] Score ordering test pinning D4's weights, including the VRT-vs-MCD case above.
- [x] `derive_status` test: a wontfix cluster that grows reopens.
- [x] Measured against run 2: 1,893 clusters, ~60 families, top family reproduces the
      `incomeTaxExpense`-style breadth signal (47 of 54 tickers → `likely-check-or-catalogue`).

---

### Phase 5: the one report ✅

**Goal**: replace the truncated dump with the artifact agent B consumes.

**Changes**:

1. `src/validate/fundamentals/report.py` — rewrite. Sections in this order:
   - [x] **Header** — scope, `run_id`, run_date, roster, ticker/field/tier counts.
   - [x] **Check-health gate** — abstentions and ceiling breaches FIRST, from
     `fundamentals_check_run`. If any check is over its ceiling or abstained, a banner states
     that the rankings below may be inflated. *A cluster list drawn from a mis-calibrated run
     reads as authoritative regardless; this is the only thing that stops that.*
   - [x] **Delta vs the previous comparable run** — per family and per cluster, `373 → 47`.
     Omitted with an explicit note when no comparable prior run exists.
   - [x] **Top field families**, ranked by summed score, with breadth and routing hint.
   - [x] **Top clusters**, ranked; the **top 5 marked as B's menu** with stable `cluster_id`s.
   - [x] **Per-cluster packet**: every check that fired (the corroboration signal), severity
     mix, tier mix, period range, EDGAR URL where available, and the `detail.why` of the
     highest-severity member.
   - [x] **`wontfix` footer** — every wontfix cluster, its note, its age in runs, and whether
     it is near its reopen threshold. Never omitted (D8).
   - [x] Delete `PRINTED_QUEUE_ROWS` truncation from the FILE path; keep a terminal cap and fix
     the docstring that currently claims the markdown is untruncated.

2. `src/validate/cli.py`:
   - [x] `--roster` becomes **repeatable** (`--roster in_sample --roster out_of_sample`),
     covering "both samples" without a new roster entry.
   - [x] Reports default to `reports/validate/YYYY-MM-DD/<scope>.md` (D9).
   - [x] New `validate report [--run-id X]` — re-renders from the tables **without re-running**
     the checks. B needs this; re-running to read a report is how stale-row false-greens happen.
   - [x] `validate status set|clear <cluster_id>` — writes `fundamentals_check_status`;
     requires `--note` and refuses an empty one (D8's quantified-evidence rule).

**Verification**:
- [x] Golden-file test of the renderer on a synthetic ledger.
- [x] `validate report` on run 2's `run_id` produces a report with no re-run and no DB writes.
- [x] Test: a run with an abstained check renders the banner.
- [x] Test: `validate status set` without `--note` exits non-zero.

---

### Phase 6: rewrite agents A and B ✅

**Goal**: the two agent definitions match the loop that now exists. Both files are rewritten in
full, not patched.

#### 6.0 — the handoff contract (build this first; both agents depend on it)

The interface between A and B is the report file. It is a **contract**, not a convenience: if a
field below is missing, B cannot start and must say so rather than improvise.

A's report MUST contain, for each of the top 5 clusters:

| field | why B needs it |
|---|---|
| `cluster_id` | the name the user says out loud, and the regression-test suffix |
| `ticker`, `field` | what to open |
| `score`, `findings` | why it is ranked here |
| `checks_agreeing` — every check name + its count | corroboration; 9 checks agreeing ≠ 1 check firing |
| `severity_mix`, `tier_mix` | whether the published table is affected |
| `period_range` | which filings to pull |
| `routing_hint` (`likely-filer` / `likely-check-or-catalogue`) | tells B what to challenge first |
| `family_breadth` — tickers in the family / roster size | the evidence behind the hint |
| `edgar_url` (may be NULL for Tier-1-only clusters) | where to read |
| `why` — `detail.why` of the highest-severity member | the check's own stated mechanism |
| `run_id` | the scope B must re-validate at |

- [x] Add a `--format json` flag to `validate report` emitting exactly these fields, so B parses
  rather than scrapes prose. Markdown remains the human artifact; JSON is the agent artifact.
- [x] Both files live under `reports/validate/YYYY-MM-DD/`, same basename, `.md` and `.json`.

#### 6.1 — `.claude/agents/fundamentals-validate.md` (**agent A**)

- [x] **Frontmatter**: `name: fundamentals-validate`, `model: sonnet`, `color: green`,
  `description` rewritten around ranking rather than queue-dumping, with two `<example>` blocks
  in the existing house style (one roster run, one per-field acceptance sheet).
- [x] **Opening**: "You are **agent A**. You RUN the validator and RANK what it found. You are
  deliberately thin — the CLI does the work. **You do not research, you do not edit code, you do
  not decide outcomes.**" Plus: read `src/validate/README.md` first, every time.
- [x] **Interpreter block** — retained verbatim:
  ```bash
  PY="$HOME/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe"
  ```
  and the `rtk` prefix rule.

- [x] **Step 1 — choose the scope**, from this table (retained from today's agent, which gets
  this right):

  | the request | scope | what a pass proves |
  |---|---|---|
  | a named ticker | `-t X` | nothing general |
  | "is extraction consistent?" | `--roster in_sample` | CONSISTENCY, not generalisation |
  | "does it generalise?" | `--roster out_of_sample` | zero overlap, never tuned |
  | "what's the real error rate?" | `--roster random_cold` | the only honest estimate |
  | "both samples" | `--roster in_sample --roster out_of_sample` | repeatable flag, Phase 5 |
  | a new catalogue field | `--field X --roster in_sample` | IS the acceptance sheet |
  | nightly | `--tier 1` over all, then `--tier 2,3 --since <last night>` | — |

- [x] **Step 2 — CONFIRM THE SCOPE WITH THE USER BEFORE RUNNING** (D9). Use `AskUserQuestion`
  with the chosen scope as the first option and the two nearest alternatives. State the ticker
  count and the estimated runtime (~15 min for 54 tickers). A full-universe run is expensive and
  must never start on an inferred scope.

- [x] **Step 3 — run**:
  ```bash
  rtk "$PY" -m src validate fundamentals --roster in_sample \
      --report reports/validate/$(date +%F)/in_sample.md --format json
  ```
  `--no-write` when the user is exploring a threshold rather than recording a run. Say which
  you used.

- [x] **Step 4 — READ THE CHECK-HEALTH GATE BEFORE THE RANKINGS.** Non-negotiable and it leads
  the summary:
  - a check labelled **THRESHOLD BUG** has NOT found that much bad data — it has a threshold
    problem and is burying real findings under itself. **Its clusters are not trustworthy and
    must be reported as such**, even when they rank top.
  - a check labelled **ABSTAINED** examined nothing. That is not a pass. Name what went
    unchecked ("`peer_ratio` never ran for `broker_dealer` — GS is the only one on this roster").
  - if either is present, say explicitly that the rankings below may be inflated.

- [x] **Step 5 — report the delta** vs the previous comparable run: clusters closed, clusters
  reopened (D8), clusters new. If no comparable prior run exists, **say so** — never present a
  first run as if it were a trend.

- [x] **Step 6 — present the top 5** in the contract's shape, one block each, plus the top field
  families with breadth and routing hint. For each cluster, one line on *why it ranks here*
  (volume vs. severity vs. tier), because the score is a policy, not a fact.

- [x] **Hard rules** (a `## What you must NOT do` section, matching the existing house style):
  - **Do not investigate a cluster.** No filings, no EDGAR, no root-cause narrative. Hand it to
    `fundamentals-triage`. *(D10 — this is what stops A anchoring B.)*
  - **Do not edit `configs/`, the catalogue, or any check.** No fix authority.
  - **Do not write `fundamentals_check_status`.** Only B decides a `wontfix`.
  - **Do not present a low finding count as good news** without checking abstentions.
  - **Do not re-run to "verify" a change someone just made** — it reads stale rows and reports a
    false green. The rebuild has to happen first.
  - **Do not compare runs of different scope.** The tooling refuses; do not work around it.

- [x] **Closing format**: tickers in scope; findings by severity; clusters and families;
  abstained checks and what that leaves unchecked; checks over ceiling; the delta; the top 5
  with `cluster_id`s; and the `run_id` B must re-validate at.

#### 6.2 — `.claude/agents/fundamentals-triage.md` (**agent B**)

- [x] **Frontmatter**: `name: fundamentals-triage`, `model: opus`, `color: orange`,
  `description` rewritten around **one cluster** rather than one finding, with two `<example>`
  blocks.
- [x] **Opening**: "You are **agent B**. You take ONE cluster and settle it, with evidence. A
  half-investigated cluster left in the queue is worse than an untouched one, because the next
  reader assumes it was looked at." Read `src/validate/README.md` first, especially §4.
- [x] **Interpreter block + the three rebuild paths** — retained verbatim, minus the deleted
  register row:

  | you changed | rebuild before re-validating | cost |
  |---|---|---|
  | `build_history.py`, `reason_codes.py`, a `_FORMULAS` entry | `fundamentals-history --rebuild-history -t X` | ~2.5 min/ticker, **no network** |
  | `xbrl_linkbase.py`, `periods.py`, `fundamentals_kpis.json` — a RESOLUTION bug | `fundamentals --rebuild -t X` | network, minutes |
  | a check module or threshold | none — re-validate directly | seconds |

**Step 0 — pick the cluster**
- [x] Read the newest `reports/validate/<date>/*.json`. If none exists, **stop and ask the user
  to run agent A** — do not run the validator yourself.
- [x] `AskUserQuestion` with the **top 5**, each labelled `TICKER field` and described with
  score, findings, checks agreeing, and routing hint. Work **exactly one**.

**Step 1 — read the whole cluster**
- [x] Pull every finding in it:
  ```bash
  MSYS_NO_PATHCONV=1 rtk docker exec pea_db psql -U alexandre -d pea -c \
    "SELECT check_name, period_key, severity, tier, observed, expected, deviation, detail
     FROM fundamentals_check WHERE run_id='<run_id>' AND cluster_id='<cluster_id>'
     ORDER BY severity, period_key;"
  ```
- [x] **Weigh the corroboration.** N independent checks agreeing is a far stronger prior than
  one check firing. A cluster carried by a single check that is over its ceiling is weak
  evidence and should be treated as a suspected check defect first.
- [x] **Do not re-derive the numbers.** If the packet is not enough to start, that is a defect
  in the CHECK and worth reporting on its own.

**Step 2 — ⚠ CHALLENGE THE CHECK BEFORE THE DATA**
- [x] Retained in full, including DQC_0118 (*"inconsistencies reported to filers can be
  overwhelming as many don't represent real errors"*) and both earned precedents: the **745
  correct rows nulled** by over-strict Q4 guards, and the **withdrawn "UNH has no premiums"**
  edit (UNH earns $72bn in premiums, CVS $34bn, DE $248M).
- [x] Now mechanical: **`routing_hint: likely-check-or-catalogue` means challenge the spec
  before opening a single filing.** A family spanning ≥5 tickers and ≥30% of the roster is far
  more likely to be our defect than a simultaneous failure by 40 filers.
- [x] Also check: is the check over its ceiling in this run? Is the cluster inside a KNOWN
  false-positive population (`trend_break` on a seasonal filer, `coverage_field` on a MIXED
  cell, `leaf_vs_total` where the linkbase omits a caption)?

**Step 3 — research the filing**
- [x] Open the `edgar_url` and read the filed statement. Not the reason code, not
  `companyfacts` — the filing. Both warnings retained verbatim:
  > A `not_disclosed` code is a statement about OUR CONCEPT MAP, never about the filing. It
  > cannot distinguish "the filer has no such line" from "the filer tagged it under a name we do
  > not recognise". 68% of all reason codes are `not_disclosed`.

  > `companyfacts` can prove a concept PRESENT and can NEVER prove one ABSENT. It publishes no
  > company-extension taxonomy and silently drops dimensioned facts. Every coverage claim must
  > be measured off `filing.xbrl()`.
- [x] If the cluster is Tier-1-only it has **no `edgar_url`** (1,427 such findings today). Say
  so plainly and resolve the accession manually — and note it, because it is Phase 7's trigger.

**Step 4 — plan, then GO / NO-GO**
- [x] Write the plan: the mechanism, the layer to change, the blast radius (how many other
  clusters the same fix likely closes), and the rebuild cost.
- [x] **Stop and get an explicit go/no-go from the user.** Picking the cluster authorised the
  investigation, not the change.

**Step 5 — fix, by layer**
- [x] **Code** (`build_history.py`, `xbrl_linkbase.py`, `periods.py`, a check module): apply it,
  record the commit. `fix_kind: code`.
- [x] **`configs/`** (`fundamentals_kpis.json`, `fundamentals_exceptions.json`,
  `fundamentals_regimes.json`, `fundamentals_cik_cutover.json`): **RISK ZONE — PROPOSE THE DIFF,
  NEVER APPLY IT.** The cluster stays OPEN until a human approves. A large share of real
  fundamentals fixes are config, and a wrong config entry is invisible forever.
- [x] The fundamentals config JSONs are **hand-formatted**; a `json.dumps` round-trip reformats
  the whole file. Splice text or use a validated emitter.

**Step 6 — rebuild, then re-validate. "Re-run the validator" is NOT verification**
- [x] Rebuild the layer touched (table above). Re-validating without rebuilding reads stale rows
  and reports a false green.
- [x] **Call agent A as a sub-agent** at the **ORIGINAL scope** (D11):
  ```
  Agent(subagent_type="fundamentals-validate",
        prompt="Re-validate at exactly this scope: <roster/tickers/fields/tiers> from
                run_id <run_id>. Do not widen or narrow it. Report the row-count delta
                for cluster <cluster_id> and any cluster that reopened.")
  ```
  A different scope produces an incomparable `run_id` and the delta is meaningless.
- [x] **The delta is the proof.** Report it as a number: `14 → 0`. A fix with no measured drop
  is not a fix.

**Step 7 — record the outcome**
- [x] **`fixed`** — requires a commit AND a regression test. Nothing else to write; the ledger's
  row-count drop IS the record (D5).
- [x] **`wontfix`** — real, known, not worth repairing. Requires a **QUANTIFIED cost**, a number
  and not an adjective:
  ```bash
  rtk "$PY" -m src validate status set <cluster_id> --note "<quantified evidence + accession>"
  ```
  Captures `findings_at_decision` automatically. It **auto-reopens if the cluster grows** (D8)
  and appears in every future report's footer. **Never `wontfix` a wide cluster** — a
  `likely-check-or-catalogue` family means the spec is wrong and the fix is the spec.
- [x] **No JSON register writes.** `configs/fundamentals/fundamentals_check.json` no longer
  exists (Phase 3).

**Step 8 — the regression test**
- [x] A `fixed` outcome must leave a named test in `tests/` pinning the case forever. Mandatory:
  it is the acceptance corpus and it grows with every fix. **Name the file with the
  `cluster_id`** (was `finding_id`) so the test traces back to the cluster it closed.

- [x] **Hard rules** — retained verbatim except where the register is referenced:
  - Never accept a cluster without filing-level evidence.
  - Never apply a `configs/` edit. Propose it.
  - Never mutate `fundamentals_history` or `fundamentals_facts` directly. Append-only; the fix
    is upstream and the rebuild is the mechanism.
  - Never claim a fix is verified without the rebuild AND the re-validation delta.
  - Do not "helpfully" add a `roll_up.any_of` for `totalLiabilities`. 0 of 44 10-Ks declare a
    `Liabilities` total; leg-sets vary by filer *and* year; an unlisted us-gaap sibling is
    dropped **silently**, and the failure mode is a balance-sheet total short by a caption that
    looks entirely plausible.
  - Do not treat a `derived_identity` value as corroboration of the identity it was computed
    from. It is an INPUT.
  - Do not assume a NULL `minorityInterest` is zero as a blanket rule. Correct only where the
    filer has never tagged one, tested POINT-IN-TIME: TMO looks refusable on lifetime counts but
    files its first NCI on 2022-02-24 against a history opening 2011-11-04.

- [x] **Closing format**: the `cluster_id` and what it was; how many checks agreed; whether you
  challenged the check and what you concluded; what you read and in which accession; the
  outcome with evidence; what you changed; how you rebuilt; **the measured delta**; the
  regression test added. If the outcome is `config_proposed`, show the diff and say plainly that
  the cluster remains OPEN.

**Verification**:
- [x] Both agent files parse: valid frontmatter, `model` and `color` set, ≥2 `<example>` blocks.
- [ ] Dry run A on `--roster in_sample`: confirms scope before running; leads with check health;
  emits both `.md` and `.json`; the JSON validates against 6.0's field list.
- [ ] Dry run B: refuses to start with no report present; `AskUserQuestion` lists 5 clusters;
  stops for go/no-go before any edit.
- [ ] Confirm B's sub-agent call reuses the original scope and that a deliberately widened scope
  produces a different `run_id` and is refused as incomparable.
- [x] `validate status set` without `--note` exits non-zero (also covered in Phase 5).
- [ ] End-to-end on one real cluster: A → B → A, ending in a measured row-count drop.

---

### Phase 7: Tier-1 provenance ✅ (2026-08-25 — solved by MOVING the checks, not by joining)

**Trigger fired early.** Not by B picking a cluster, but by the user asking whether the seven
criticals failing to reach the menu was "the issue with Tier one on history and not facts
table". It was, and the diagnosis went further than the ranking: on the calibration run
**1,427 of 1,427 Tier-1 findings had a NULL `edgar_url`** (0.0%) against 77.8% on Tier 2 and
100% on Tier 3, so the whole tier was unactionable however it was ranked.

**The plan's proposed change was the wrong one, and was NOT implemented.** It said: keep the
checks on history and join back to `fundamentals_facts` on `(ticker, field, period)` to recover
provenance. That join cannot be made sound. A `fundamentals_history` row is a per-column
carry-forward snapshot collapsed by `(ticker, as_of)`, so its `totalAssets` and its
`stockholdersEquity` can come from different filings; there is no single accession to recover,
and picking one would attach a filing that did not produce the number.

**What was done instead:** the checks moved. Eight of the fourteen Tier-1 history checks now
read `fundamentals_facts` directly — `cross_identity`, `coverage_field`, `coverage_quarters`,
`expected_absent_drift`, `impossible_value`, `filing_lag`, `catalogue_exclusion_cost`,
`catalogue_override_coverage`. Six stayed, and the rule that separates them is: **a check that
asks about the TABLE reads history; a check that asks about a NUMBER reads facts.**

Measured after the move:

| | before | after |
|---|---|---|
| Tier-1 `edgar_url` coverage | 0 / 1,437 = **0.0%** | 876 / 1,386 = **63.2%** |
| ...of checks that implicate a filing | — | **100%** |
| `cross_identity` findings | 254 (history grain) | 174, all with an accession |
| balance sheets tested | 3,229 | **4,763** |
| `filing_lag` findings | 1 | **11**, all real |
| contract checks firing | 0 | 0 |

The 510 findings still without a URL are the 500 catalogue-configuration diagnostics — no
accession causes a `never_use` entry — plus 10 pre-existing ticker-grain checks.

**The six that stayed are the point, not a leftover.** `grain`, `column_contract`,
`code_vocabulary`, `unexplained_null`, `pit_leak` and `coverage_universe` test a 69-column
ORDERED contract, a null CELL, the reason-code vocabulary and the no-leakage snapshot grain.
`facts` expresses none of those, so porting them would have deleted them. All six are
`expected_fire_rate_ceiling=0.0` and all six fire zero — they are tripwires for a
`build_history` bug, which is the only defect class genuinely history's own. ETN's 2012-11-14
row is the specimen: `totalLiabilities` of -$8,237,223,652 against `totalAssets` of $4,776,348,
tagged `derived_identity`, with no counterpart anywhere in `facts` because no filer tagged it.

**Two things fell out that the plan did not anticipate:**

- the `derived_identity` SKIP in `cross_identity` is **deleted**. It existed because history's
  `totalLiabilities` could be `totalAssets - stockholdersEquity`, making the identity
  `A - E + E == A`. `fundamentals_facts` is strictly as-filed and holds no derived total, so
  there is nothing to skip;
- `impossible_value` lost its `epsDiluted` rule — that column is one of the twelve
  `build_history` DERIVES and it exists nowhere in facts. `MAX_ABS_EPS` is kept with its
  reasoning against the day EPS gets a derived-value check.

**Pinned by** `tests/validate/fundamentals/test_substrate_contract.py`: the split itself, the
zero-ceiling property of the six, that no Tier-1 value check reads history, and — parametrised
over the REGISTRY so a new check is covered automatically — that every facts-grain Tier-1
finding carries an `edgar_url`.

#### The move caused a PK collision, and the guard from Phase 1 caught it

The first full run **failed**, and correctly. On the statement grain `cross_identity` emits one
finding per `(ticker, accession_number, period_end)`, but a 10-K carries its COMPARATIVES — so
one balance-sheet date appears in several filings, and AMT's 2016-12-31 sheet appears in five,
all five reporting the same break. `finding_id` hashes `(check_name, ticker, field, period_key)`
and IS the `fundamentals_check` primary key, so those five would have upserted onto each other:
**174 emitted, 83 ids, 91 silently lost.**

`DuplicateFindingError` — added in Phase 1 after run 2 lost 536 rows to the same class of bug —
raised before anything was written. Measured across every Tier-1 check, `cross_identity` was the
only one affected; the other nine were already collision-free.

**Fix:** Tier 3's `_collapse` was promoted to `finding.collapse_by_id(findings, *, why)` rather
than a second copy being written, and Tier 3 now delegates to it. Both collision mechanisms are
documented on it — Tier 3's duration shape, Tier 1's comparatives. Collapsing is also the more
honest shape: one broken balance sheet is one thing to look at, however many filings repeated
it, and reporting it five times is the DQC_0118 drowning this design exists to prevent. The
survivor keeps its accession; the other filings are listed in `detail.collapsed`.

#### Verified live, 54 tickers, run_id `725bae7bf8ed` (2026-08-25)

- **emitted 11,730 == stored 11,730.** No loss;
- Tier 1: 1,295 findings, **785 with an `edgar_url` (60.6%)**, from 0 of 1,437. The 510 without
  one are the 500 catalogue-config diagnostics and 10 ticker-grain checks;
- **all 7 criticals now carry a URL** — VRT's pre-merger SPAC across 7 distinct quarters, each
  pointing at the filing that carries that balance sheet;
- the six contract checks fired **zero**, so Tier 1 wrote no `history`-substrate row at all;
- the delta against the 2026-08-24 run rendered correctly (same scope hash): 1 cluster settled,
  12 new.

**Test status:** `tests/validate` 115 passed, 5 skipped. Full suite outside
`data_aggregate`/`data_extract`: 355 passed, 13 skipped, 0 failed. Those two directories show 13
failures, all reproduced on unmodified HEAD — including the two live-SEC ones
(`test_maa_shares_outstanding_is_the_parents`, `test_apa_revenue_is_a_real_number...`), which
skip in a bare worktree for want of `.env` and fail identically once it is present.

**Two tests were found passing VACUOUSLY** after the move — they planted into
`substrates.history`, which the check no longer reads, so they asserted nothing while staying
green. Three others failed loudly. All five were re-pointed at facts, and the acceptance-corpus
header now says so, because a vacuous green is worse than a red.

---

## Testing Strategy

**Unit** — `tests/validate/fundamentals/test_clusters.py` (scoring, collapsing, routing hints,
status derivation, D8 reopen), `test_ledger.py` (scope hashing, comparable runs),
`test_report.py` (golden file, health banner), Phase 2's shape × gap-code severity matrix,
Phase 1's `finding_id` uniqueness assertion.

**Integration** — full 54-ticker calibration after Phases 1–2 and again after Phase 5. Success
means emitted == stored, and the top-5 list is legible without opening the table.

**Manual** — one full A → B → A cycle on a real cluster, ending in a measured row-count drop.

## Risk Mitigation

1. **Deleting the register removes an anti-suppression guard.**
   *Mitigation*: D8 (auto-reopen on growth) plus the never-omitted wontfix footer. Phase 5 must
   ship the footer or Phase 3's deletion is a net loss of safety.
2. **Ranking a mis-calibrated run produces a confident, wrong work plan.**
   *Mitigation*: the check-health gate renders above the rankings, with a banner. This is why
   Phase 2 precedes Phase 4 — severity feeds the score.
3. **Comparing runs of different scope reads as a fix.**
   *Mitigation*: `run_id` scope hashing; the delta section is omitted, with a note, when no
   comparable prior run exists.
4. **D4's weights may rank badly on first contact.**
   *Mitigation*: weights are module constants printed in every report; retune after list #1.
   Explicitly expected, not a failure.
5. **Ledger growth** — ~12k rows/run.
   *Mitigation*: PK already carries `run_date`; indexes on `cluster_id` and `run_id`. Revisit
   retention only if a run exceeds ~30s to read back.

## Rollback

Every phase is additive or a clean deletion on a feature branch. Phases 1 and 3 change the DB:
Phase 1 is **strictly additive DDL** (two new tables, two new columns) — apply it to the live
volume directly and **never** via `scripts/recreate_fundamentals_tables.py`, which drops and
recreates and would destroy the 4h39m rebuild. Phase 3 deletes only a config file and Python.

## Dependencies

- Postgres 16 (`pea_db`), no new libraries.
- `docs/database.md` and `docs/runbook.md` need the two new tables and the new CLI commands.
- Interpreter:
  `PY="$HOME/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe"`

## Success Criteria

- [x] Emitted findings == stored rows (the 536 gap closed)
- [x] ~347 mislabelled `info` findings correctly severed
- [x] 11,926 findings present as ~1,893 clusters in ~60 families
- [x] Top-5 menu is legible without opening the database
- [x] Two same-scope runs produce a sound delta; different-scope runs refuse to compare
- [x] Register fully deleted, with D8 demonstrably replacing its guard
- [ ] One complete A → B → A cycle ends in a measured row-count drop
- [x] `rtk "$PY" -m pytest tests/validate` green

## Estimated Effort

| phase | estimate |
|---|---|
| 1 — ledger + scope + collision fix | 2.5 h + a calibration run |
| 2 — severity ladder | 1 h |
| 3 — retire register | 1 h |
| 4 — clustering + ranking | 2.5 h |
| 5 — the one report | 2.5 h |
| 6 — handoff contract (6.0) + agents A and B | 3 h |
| **total** | **~12.5 h** plus two 15-min calibration runs |

Phase 6.0's `--format json` flag touches `src/validate/cli.py`, which Phase 5 also edits. Build
Phase 5 first; 6.0 extends the command rather than reworking it.

## Notes for Implementation

- `store.ensure_table` infers column types from the FIRST frame — an all-None object column
  becomes TEXT permanently. Pin dtypes on the new tables' first write.
- Postgres DATE round-trips as `datetime.date`, never `Timestamp`. Normalise on load, once,
  in `ledger.py` — the same discipline `substrate.py` already applies.
- The fundamentals config JSONs are hand-formatted; a `json.dumps` round-trip reformats the
  whole file.
- Prefix every shell command with `rtk`.

---

## Implementation results — measured on calibration run 3 (2026-08-24, `3df52ae9af75`)

Every number below is measured against the live 54-ticker ledger, not asserted.

| claim | before | after |
|---|---|---|
| emitted vs stored findings | 12,462 / 11,926 (**536 lost silently**) | **11,926 / 11,926** |
| mislabelled `info` (`series_shape`) | 340 `interior_gap` + 7 `early_stop` | **0** — all 347 moved to the queue |
| `series_shape` fire rate vs ceiling | 10.45% / 15% (measured while suppressing 347) | **16.63% / 18%** (ceiling re-derived, see below) |
| findings presented as | a queue truncated to 57 of 10,898 rows | **2,323 clusters in 50 field families**, none omitted |
| clusters with work in them | — | **1,939** |
| top family | — | `incomeTaxExpense`, **53 of 54 tickers**, `likely-check-or-catalogue` |
| JSON register entries lost | — | **0** (the file held 0 entries) |
| tests | 54 | **104** |

### Deviations from the plan, and why

1. **`register_cost` / `register_coverage` were NOT deleted (D7).** Neither reads the JSON
   register: both read the CATALOGUE (`fundamentals_kpis.json`) — its `never_use` exclusions
   and its `by_ticker` overrides. D7's rationale ("superseded by D5/D6") did not apply, and
   `register_cost` is the only place NEE's $5.2bn exclusion cost is published, which
   `fundamentals_rosters.json` itself cites. **Confirmed with the user**, then renamed to
   `catalogue_exclusion_cost` and `catalogue_override_coverage` to kill the name collision that
   caused the confusion. `validate checks` therefore lists **35**, not the planned 33.

2. **`run_id` joined the `fundamentals_check` PRIMARY KEY**, beyond the plan's "strictly
   additive DDL". Not theoretical: a `-t MCD` smoke run wrote 270 rows and the 54-ticker run an
   hour later upserted over 269 of them, leaving the first run claiming 35 checks and one
   surviving finding. Two runs of different scope on one day must be able to coexist or every
   delta computed against the earlier one is nonsense. The migration is data-safe
   (`DROP CONSTRAINT` + `ADD PRIMARY KEY`, no rows touched) and was verified live: a 2-ticker
   and a 1-ticker run now coexist on one date.

3. **`series_shape`'s ceiling was re-derived 0.15 → 0.18**, which the plan's Phase-2
   verification allows as its second branch. The old 0.15 was measured while the ladder was
   suppressing 347 real findings; the same data correctly classified sits at 16.63%. Nothing
   got noisier and no threshold moved — but 0.15 would now report a CORRECTED check as a
   threshold bug, which would tell every reader to distrust the clusters it feeds.

4. **`scope_hash` was added to `fundamentals_check_run`** (not in the plan's column sketch). It
   is the honest implementation of "comparable iff their scope hash matches" — a single
   equality test rather than a fragile three-column text comparison.

5. **The `_collapse` fix is generic, not three bespoke ones.** All three colliding checks
   (`cross_vintage` 526, `q4_footing` 6, `leaf_vs_total` 4) group on a key wider than
   `finding_id` hashes, so one helper collapses to the finding's grain and records what it
   merged. `findings_frame` now raises `DuplicateFindingError` so the class cannot recur.

### Retune #1, applied — the plan predicted it and it arrived immediately

D4 said the weights would be retuned "after list #1". List #1 was read and two changes followed.

**1. Corroboration entered the score, as a MULTIPLIER.** Volume-only scoring put HCA
`minorityInterest` on top — 62 findings from **two** checks, score 244 — while MCD `capex`,
55 findings from **ten independent** checks, sat at 148 and never reached the menu. That is
backwards: one check firing 62 times is one opinion repeated; ten checks agreeing is ten
arguments for the same conclusion, and the plan itself calls corroboration "the strongest prior
an agent gets".

```
cluster_score = (Σ w(severity) × w(tier)) × (1 + 0.25 × (checks − 1))
```

A multiplier rather than a bonus, so a large enough pile of single-check findings cannot drown
it — and so it cannot rescue an all-`info` cluster, whose base is 0. The menu is now:

| # | cluster | score | findings | checks |
|---|---|---|---|---|
| 1 | MCD `capex` | 481 | 55 | **10** |
| 2 | BA `incomeTaxExpense` | 390 | 39 | **10** |
| 3 | MCD `dilutedShares` | 374 | 61 | 8 |
| 4 | MCD `basicShares` | 368 | 60 | 8 |
| 5 | ORCL `totalRevenue` | 360 | 47 | 7 |

**2. The ranked table caps at 25 and states the total.** It listed all 1,939, which nobody
reads: agent B works ONE cluster from a menu of five, so rows 26+ cost every reader the scroll
and told them nothing. What a reader needs from the tail is its SIZE. The header now reads
`top 25 of 1939` and the footer names the remainder — capping is fine, capping *silently* is
what the old report did. The markdown went from ~2,000 lines to **436**.

### Two findings worth acting on

- **The routing hint is not discriminating on this roster: 48 of 50 families came back
  `likely-check-or-catalogue`.** A broad statistical check (`peer_ratio`, ~2.4%) touches nearly
  every ticker on nearly every field, which makes every family look wide. The report now
  DETECTS and states this rather than presenting a flat signal as evidence, but the thresholds
  (≥5 tickers AND ≥30% of roster) want retuning against a real roster — deliberately left to a
  human, since moving them to make a list look better turns a diagnostic into a decoration.
- **The 7 Tier-1 criticals still do not surface in the menu**, even after corroboration. They
  are single-check `cross_identity` findings on VRT, and the score is deliberately
  volume-and-corroboration dominant. If a critical should pre-empt the ranking regardless of
  volume, that is a THIRD policy change — a floor rather than a weight — and it is a decision,
  not a bug.

### Not done

- ~~**Phase 7 (Tier-1 provenance)**~~ — done 2026-08-25, by moving the checks rather than
  joining back. See Phase 7 above for the measurements.
  Still measured at 1,427 findings with no `edgar_url`, including all 7 criticals.
- **The end-to-end A → B → A cycle on a real cluster.** The mechanism is proven — two
  same-scope runs difference correctly and a different-scope run is refused — but an actual
  fix-and-close needs agent B to pick a cluster and a human to give the go/no-go.
- **`fundamentals_baselines.json`** — still open from 5b-core.4, as the plan noted.
