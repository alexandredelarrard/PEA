# Implementation Plan: Recording Fixes in the Fundamentals Validator

**Date Created**: 2026-08-25
**Planning Phase**: 2 of 3 (FIC Workflow)
**Based on Research**: live tracing of the validator loop during the `1c9a517eaa47` (MCD `capex`) triage, 2026-08-25
**Next Phase**: Implementation (`/implement`)

## Overview

Agent B fixed cluster `1c9a517eaa47` today and the fix has **no machine-readable trace anywhere**.
Not in `fundamentals_check_status` (which accepts `wontfix` and nothing else), not in
`settled_clusters` (a strict set difference that a 55→4 drop does not satisfy), and not in the
rendered report (the cluster fell off the top-50). The only record is commit `2fb6ef2`.

This plan adds the missing record **without** reintroducing the failure mode that the deleted
`configs/fundamentals/fundamentals_check.json` register was becoming: a stored verdict that
suppresses findings while the check is still firing.

The design rests on one distinction:

| what you are recording | shape | mutable? | suppresses findings? |
|---|---|---|---|
| "we intervened, here is what and why" | an **event** | no, append-only | **never** |
| "this finding is real and we tolerate it" | a **decision** | yes, self-expiring | yes, at render |

A fix is the first. Benign residue is the second. Today only the second exists, and only at
whole-cluster grain — which is the actual reason partial settlement feels like it needs a new
status value. It does not, and this plan adds none.

## Decisions taken in planning (2026-08-25)

Recorded because several are non-obvious and an implementer will otherwise re-litigate them.

| # | decision | why |
|---|---|---|
| 1 | **Settlement requires a fix row**, not just an absence of unwaived findings | Otherwise waiving every finding one check at a time manufactures a SETTLED with nobody having fixed anything — the suppression list, reassembled from parts |
| 2 | **No new status word.** `derive_status` still returns `OPEN` / `WONTFIX` / `REOPENED` | A fully-waived-but-unfixed cluster already *is* what `wontfix` means. The schema comment's boast that the vocabulary is "one word long" survives |
| 3 | `layer` stays **single-valued**, four terms | Coarse grouping, not a taxonomy. `root_cause` carries the precision |
| 4 | `layer` is defined by **what the edit does**, not which file it lives in | `extraction` = any code that *produces* a value (`xbrl_linkbase`, `build_history`, `periods`); `rows` = the code was right and the stored data was stale. Documented in the schema comment so it is not left to judgement |
| 5 | **`evidence` requirements vary by layer** | A `check` fix has no filing to cite. Its evidence is the false-positive population it was measured against. Demanding `accessions` universally would force it to cite an irrelevant filing |
| 6 | `fix record` is **atomic**: it writes the fix row and its per-check waivers in one call | Recording a fix and waiving its residue is one decision. Splitting it leaves a fix row whose cluster still reads OPEN — the half-recorded state this plan exists to remove |
| 7 | `fix record` **pre-fills everything derivable** | Only `cluster_id` is required. The human supplies only what no machine can know |
| 8 | A **no-improvement fix is recordable** (with a loud warning) but **cannot settle** | Keeps the legitimate case recordable — a wrong-but-plausible value corrected where no check was firing — while a row that improved nothing can never carry a settlement. *Permissive to record, strict to settle* |
| 9 | Backfill is **MCD only** | We have first-hand evidence. Anything older would be reconstruction, and a fabricated record is worse than a missing one in a table whose entire purpose is evidence |
| 10 | `adjustment_unguarded` is **folded in fully** | The MCD fix now writes 61 `sibling_rejected` markers into the very column the check cannot read |

## Current State Analysis

**What exists today**

| table | grain | role |
|---|---|---|
| `fundamentals_check` | one finding per run | append-only ledger, ~11.9k rows/run. Nothing is ever subtracted |
| `fundamentals_check_run` | `(run_id, check_name)` | the scope + denominator. `scope_hash` decides comparability |
| `fundamentals_check_status` | `cluster_id` | the only mutable state. `status` = `'wontfix'` only |

**Key files**

- [src/validate/fundamentals/clusters.py](../../../src/validate/fundamentals/clusters.py) — `settled_clusters`, `derive_status`, `build_clusters`
- [src/validate/fundamentals/ledger.py](../../../src/validate/fundamentals/ledger.py) — the only read-back of the three tables; `status_map()` at L201
- [src/validate/fundamentals/report.py](../../../src/validate/fundamentals/report.py) — renders; `settled_clusters` call site at L141
- [src/validate/cli.py](../../../src/validate/cli.py) — `status set` (hardcoded `WONTFIX`, L289) / `status clear`
- [src/data_store/schema.py](../../../src/data_store/schema.py) — `Tables` registry, L476
- [sql/schema.sql](../../../sql/schema.sql) — the mirrored DDL
- [.claude/agents/fundamentals-triage.md](../../../.claude/agents/fundamentals-triage.md) — Step 8, L193

**Current limitations, each measured on today's run `725bae7bf8ed`**

1. **A partial fix cannot settle.** `settled_clusters` differences `cluster_id` sets over *all*
   findings. MCD `capex` retains 4 benign findings (1 weighted-zero `info`, 2 `peer_ratio` on a
   documented blind spot, 1 `series_shape` coverage gap), so it stays in the "after" set forever.
   Today's run settled exactly one cluster, `10f02d649538` — not ours.
2. **A fix has nowhere to be recorded.** Step 8 of agent B says outright: *"Nothing else to write:
   the ledger's row-count drop IS the record."* That is only true while the drop reaches zero.
3. **A waiver is all-or-nothing.** `check_status` pk is `cluster_id`, so you cannot tolerate
   `peer_ratio` on MCD `capex` while keeping the other nine checks live on the same cluster.
4. **`adjustment_unguarded` can never run.** It reads `facts["adjustment"]`, but `FACTS_COLUMNS`
   ([substrate.py:72](../../../src/validate/fundamentals/substrate.py)) omits that column on the
   stated grounds it is *"read by no check"* — which is false. It returns early **before**
   `sub.denominator(...)`, so the health gate reports ABSTAINED (a clean-looking abstention)
   rather than broken.

**Blocking discovery — there is no migration tooling**

All DDL is `CREATE TABLE IF NOT EXISTS` ([ddl.py:90](../../../src/data_store/ddl.py)); `sql/`
contains only `schema.sql`; there is no migrations directory. `store.save` takes its
`ON CONFLICT` target from `resolve(table).pk` ([store.py:541](../../../src/data_store/store.py)),
so editing the pk tuple in `schema.py` without touching the live table produces a hard Postgres
error on the next write: *"there is no unique or exclusion constraint matching the ON CONFLICT
specification."* `ensure_columns` evolves *new columns* automatically; it does not alter a
primary key.

## Desired End State

The settlement rule, complete:

```
SETTLED   zero unwaived queue-severity findings
          AND a fix row exists for a comparable run pair
          AND that fix row shows queued_after < queued_before
WONTFIX   zero unwaived queue findings, but no qualifying fix row
          (tolerated, not solved — and it says so)
REOPENED  a waiver grew past its own findings_at_decision
OPEN      anything else
```

1. `validate fix record` writes one append-only row per intervention plus its waivers, atomically,
   with computed counts and required provenance. "What was the fix?" is one query.
2. A benign residual finding can be waived **per check**, each with its own quantified note and its
   own self-expiring `findings_at_decision`.
3. MCD `capex` settles truthfully while all 4 findings stay on the ledger and every check fires.
4. The report renders `SETTLED (clean)` vs `SETTLED (3 findings waived across 2 checks)`.
5. Agent B cannot close a `fixed` outcome without the record.
6. `adjustment_unguarded` actually runs.

## Out of Scope

- Any change that filters `fundamentals_check` rows on write. The ledger stays append-only and
  nothing is ever subtracted — that property is what makes a row-count drop usable as proof.
- A generalised versioned-migration subsystem for `src/data_store/` (rejected in planning as a new
  subsystem far outside this remit; revisit if a second PK change arrives).
- Re-scoring or re-weighting clusters. `TIER_WEIGHTS`, `SEVERITY_WEIGHTS` and the 0.25
  corroboration constant are untouched.
- A `reopened-after-fix` state. Considered and declined; a fix row plus a REOPENED status already
  makes a regression visible in the render.
- Backfilling any cluster other than `1c9a517eaa47` — including `10f02d649538`, which settled today
  for reasons nobody recorded. It stays unrecorded rather than reconstructed.
- The `coverage_field` ceiling question (26.31% vs 25.0%). Its own measurement, its own decision.
- The pre-existing `test_apa_revenue` failure on `HEAD` (`_materialise` returns a tuple; the test
  still calls `.values()` on it). Unrelated, still broken, still worth its own fix.

## Implementation Approach

### Phase 1: Schema — the new table and the PK widening ✅

**Goal**: both tables in their final shape, in both mirrors, with the live DB migrated.

> ⚠ **Risk zone.** AGENTS.md requires asking before editing `src/data_store/`, `sql/schema.sql`.
> Approval given in planning, 2026-08-25 (drop-and-recreate, gated on the table being empty).

**Changes**:

1. `src/data_store/schema.py`:
   - [x] Add `fundamentals_check_fix` to `Tables`, pk `("cluster_id", "run_id_after")`,
         `date_col="decided_at"`, `date_type_cols=("decided_at",)`
   - [x] Widen `fundamentals_check_status` pk to `("cluster_id", "check_name")`
   - [x] Rationale comments at the density of the neighbouring entries: why a fix is an event and
         not a state; why `check_name=''` means the whole cluster

2. `sql/schema.sql`:
   - [x] Mirror both with the full `-- [aggregate]` prose block. State the invariants outright:
         `commit_sha`/`test_path` required; **no renderer may filter findings using this table**;
         and the `layer` vocabulary with decision 4's semantics

   ```sql
   CREATE TABLE IF NOT EXISTS "fundamentals_check_fix" (
       "cluster_id"       TEXT NOT NULL,
       "run_id_after"     TEXT NOT NULL,   -- the run that PROVED it
       "run_id_before"    TEXT NOT NULL,
       "scope_hash"       TEXT NOT NULL,   -- pinned; both runs must share it
       "ticker"           TEXT,
       "field"            TEXT,
       "findings_before"  BIGINT,
       "findings_after"   BIGINT,
       "queued_before"    BIGINT,          -- the number settlement is judged on
       "queued_after"     BIGINT,
       "layer"            TEXT NOT NULL,   -- closed: check|catalogue|extraction|rows
       "root_cause"       TEXT NOT NULL,
       "evidence"         TEXT NOT NULL,   -- JSON, never prose; shape varies by layer
       "commit_sha"       TEXT NOT NULL,
       "test_path"        TEXT NOT NULL,
       "decided_at"       DATE,
       PRIMARY KEY ("cluster_id", "run_id_after")
   );
   ```

3. Migration (one-off, documented in the phase's DoD report):
   - [x] **Verify empty first**: `SELECT count(*) FROM fundamentals_check_status;`
   - [x] If `0` → `DROP TABLE fundamentals_check_status;`, recreated by `ensure_table` on first save
   - [x] If `> 0` → **stop and fall back** to `ALTER TABLE ... ADD COLUMN check_name TEXT NOT NULL
         DEFAULT ''`, then drop/re-add the PK. Never drop rows a human wrote

4. `src/constants/constants.py`:
   - [x] `FIX_LAYERS: frozenset[str] = frozenset({"check", "catalogue", "extraction", "rows"})` —
         closed vocabulary, on the `reason_codes.ALL_CODES` precedent
   - [x] `FIX_EVIDENCE_KEYS: dict[str, frozenset[str]]` — the per-layer required keys (decision 5):

   ```python
   {"extraction": {"accessions"}, "rows": {"accessions"},
    "catalogue":  {"accessions"},
    "check":      {"examined", "benign"}}   # the false-positive population, not a filing
   ```

**Verification**:
- [x] `"$PY" -m src validate checks` still runs (schema import clean)
- [x] Both PKs confirmed in `information_schema.table_constraints`
- [x] `rtk git diff sql/schema.sql src/data_store/schema.py` — the two mirrors agree

---

### Phase 2: Ledger read layer ✅

**Goal**: the ledger can read fixes and express a per-check waiver.

`status_map()` returns `{cluster_id: row}` today. Under the widened key two rows per cluster
collide and the last silently wins — the main ripple of Phase 1.

**Changes**:

1. `src/validate/fundamentals/ledger.py`:
   - [x] Load `fixes` with a `FIX_READ_COLUMNS` projection (AGENTS.md) and
         `_DATE_COLUMNS["fixes"] = ("decided_at",)` — DATE round-trips as `datetime.date`,
         normalised once here, never in a caller
   - [x] Re-grain `status_map()` → `{cluster_id: {check_name: row}}`, `''` being cluster-wide
   - [x] `waivers_for(cluster_id) -> dict[str, dict]`
   - [x] `fixes_for(cluster_id) -> list[FixRecord]`, newest first
   - [x] `qualifying_fix(cluster_id, scope_hash) -> FixRecord | None` — the settlement predicate:
         matching scope_hash **and** `queued_after < queued_before` (decision 8). One place, so
         the rule cannot drift between the renderer and the tests
   - [x] `FixRecord` dataclass, mirroring `RunRecord`'s discipline

2. `tests/validate/fundamentals/test_ledger.py`:
   - [x] A cluster with both a `''` waiver and a `peer_ratio` waiver resolves to two entries
   - [x] `decided_at` returns `datetime64[ns]`, not `datetime.date`
   - [x] `fixes_for` on a twice-fixed cluster returns both, newest first
   - [x] `qualifying_fix` rejects a no-improvement row and a wrong-scope row

**Verification**:
- [x] `"$PY" -m pytest tests/validate/fundamentals/test_ledger.py -q`
- [x] Sanity conclusion printed: key counts for a synthetic two-waiver cluster

---

### Phase 3: Waiver-aware settlement and rendering ✅

**Goal**: MCD `capex` settles truthfully, and the report says on what basis.

**Changes**:

1. `src/validate/fundamentals/clusters.py`:
   - [x] `settled_clusters(previous, latest, waivers, fixes)` implementing the four-state rule in
         *Desired End State*. Docstring states the invariant plainly: nothing is subtracted from
         the ledger; a waived finding is still written, still counted, and still fires
   - [x] Return `list[SettledCluster]` carrying `cluster_id`, `waived_findings`, `waived_checks`,
         `fix` — so the renderer can distinguish clean from waived without recomputing
   - [x] `derive_status` — read the nested waiver map. **No new status value** (decision 2): a
         cluster reads `WONTFIX` when it has no qualifying fix row and nothing unwaived left;
         `REOPENED` applies per entry against its own `findings_at_decision`

2. `src/validate/fundamentals/report.py`:
   - [x] Pass waivers and fixes into `settled_clusters` at the L141 call site
   - [x] Render `SETTLED (clean)` vs `SETTLED (3 finding(s) waived across 2 check(s))`
   - [x] Attach a fix-history line to any settled, wontfix or reopened cluster carrying a fix row:
         layer, root_cause, commit, before→after
   - [x] Mirror both into `render_json`

3. `tests/validate/fundamentals/test_report.py`, `test_clusters.py`:
   - [x] Waived residue + qualifying fix → settles
   - [x] Waived residue + **no** fix row → `WONTFIX`, **not** settled (decision 1)
   - [x] Waived residue + a **no-improvement** fix row → `WONTFIX`, not settled (decision 8)
   - [x] Residue that is only `info` settles with no waiver at all — the
         `catalogue_exclusion_cost` case. If this needs a waiver, the queue-severity filter is wrong
   - [x] One unwaived `high` finding → `OPEN` regardless of the fix row
   - [x] A waived check growing past `findings_at_decision` reopens and un-settles
   - [x] **Anti-suppression test**: `fundamentals_check` row count identical with and without
         waivers and fix rows present. This is the test that pins the whole design

**Verification**:
- [x] `"$PY" -m pytest tests/validate/fundamentals/ -q`
- [x] `"$PY" -m src validate report --run-id 725bae7bf8ed` (read-back, no re-run, no writes)

---

### Phase 4: `validate fix record` ✅

**Goal**: one atomic command that refuses what it cannot verify and derives what it can.

**Changes**:

1. `src/validate/cli.py` — new `fix` group:

   ```
   "$PY" -m src validate fix record 1c9a517eaa47 \
       --layer extraction \
       --root-cause "route 1 took a total the filer declares beside its own leg" \
       --evidence '{"accessions": [...], "concepts": {...}, "figures": {...}}' \
       --commit 2fb6ef2 \
       --test tests/data_extract/test_linkbase_sibling_total_1c9a517eaa47.py \
       --waive "peer_ratio:2 findings, 8.3% capex/revenue vs 3.5% peer median" \
       --waive "series_shape:1 interior-gap coverage finding"
   ```

   - [x] **Derived, not asked** (decision 7): `--after` defaults to the latest run containing the
         cluster, `--before` to its previous comparable run; `ticker`, `field`, `scope_hash` and all
         four counts come from the ledger. Both run flags remain available as overrides
   - [x] **Atomic** (decision 6): the fix row and every `--waive` land together, or neither does
   - [x] `--waive` notes obey the existing quantified-note rule — reuse `status set`'s validator
         rather than writing a second one
   - [x] Refuse, each message naming the rule being enforced:
         - cluster absent from `fundamentals_check` (nothing was ever measured)
         - the two runs' `scope_hash` differ (not a comparison, so not proof)
         - `--layer` outside `FIX_LAYERS`
         - `--evidence` unparseable, or missing its layer's required keys
         - `--commit` fails `git rev-parse --verify`
         - `--test` missing on disk
         - a `--waive` for a check that has no finding on this cluster in the latest run
   - [x] **Warn, do not refuse**, when `queued_after >= queued_before` (decision 8) — and say in
         the warning that the row is on the record but cannot settle the cluster
   - [x] `validate fix show <cluster_id>` — joins fix rows to their waivers; the read-back that
         answers reason / what / before / after / still-pending
   - [x] Add both to the command table in `src/validate/README.md`

2. `tests/validate/test_cli_fix.py` (new):
   - [x] Every refusal fires, each its own assertion
   - [x] The no-improvement path **warns and writes**
   - [x] A well-formed record round-trips; derived counts match the ledger
   - [x] A failed `--waive` rolls the fix row back (atomicity)

**Verification**:
- [x] `"$PY" -m pytest tests/validate/test_cli_fix.py -q`
- [x] `--help` reads clearly to someone who has not read this plan

---

### Phase 5: Agent B — the hard gate ✅

**Goal**: a `fixed` outcome is not complete until it is recorded.

**Changes**:

1. `.claude/agents/fundamentals-triage.md`:
   - [x] Rewrite Step 8. It currently says a `fixed` outcome writes nothing; it must require
         `validate fix record`, and say the CLI enforces the invariants so an unproven fix cannot
         be recorded
   - [x] Document the per-check waiver as the way to settle benign residue, with the quantified-note
         rule restated
   - [x] State decision 1 explicitly: **waiving everything does not settle a cluster.** An agent
         that cannot fix a cluster records a `wontfix`, and that is a different, visible outcome
   - [x] Update "How to close your turn" so the fix row and its waivers are part of the report
   - [x] Keep the existing prohibition intact: never `wontfix` a wide cluster

2. `.claude/agents/fundamentals-validate.md`:
   - [x] Agent A's Step 5 surfaces fix history in the delta, so a reopened cluster reads as a
         regression rather than a new defect

**Verification**:
- [x] Read both files end to end for contradictions against the new CLI
- [x] Dry-run Step 8's wording against the MCD case: every field it asks for is obtainable

---

### Phase 6: `adjustment_unguarded` — make it actually run ✅

**Goal**: the check that audits adjustment provenance can see adjustments.

**Changes**:

1. `src/validate/fundamentals/substrate.py`:
   - [x] Add `"adjustment"` to `FACTS_COLUMNS`
   - [x] **Correct the comment** asserting `adjustment` is "read by no check" — false, and the
         direct cause of this bug. Name the check that reads it

2. `src/validate/fundamentals/checks/tier1_value.py`:
   - [x] Move `sub.denominator("adjustment_unguarded", len(facts))` **above** the early return, so
         a missing column reports as a failure rather than a clean-looking ABSTAINED
   - [x] Classify `sibling_rejected` alongside the `ppeNet` population; the docstring's "the
         population is known: all 128 `ppeNet` lease adjustments" is now stale

3. `tests/validate/fundamentals/test_substrate_contract.py`:
   - [x] Pin the **general** rule: every column any check reads is in the projection. A test
         pinning only `adjustment` would not catch the next instance of this class

**Verification**:
- [x] Measure the read cost before and after on a 54-ticker load — `fundamentals_facts` is ~28M
      rows universe-wide, so report the row/memory delta rather than guessing
- [x] `adjustment_unguarded` reports non-zero `examined` and leaves the ABSTAINED list
- [x] The 61 MCD `sibling_rejected` rows surface as `info` findings

---

### Phase 7: Backfill the MCD case as the end-to-end proof ✅

**Goal**: the worked example becomes the integration test, and today's fix gets its record.

**Changes**:

1. [ ] Record the fix with the Phase 4 command — `--layer extraction`, accessions
       `0000063908-18-000010` and `0000063908-20-000022`, concepts
       `PaymentsToAcquireProductiveAssets` ($540.9M) vs
       `PaymentsToAcquirePropertyPlantAndEquipment` ($2,393.7M), commit `2fb6ef2`, test
       `tests/data_extract/test_linkbase_sibling_total_1c9a517eaa47.py`, and both `--waive` flags.
       Runs `3df52ae9af75` → `725bae7bf8ed` should be **derived**, not typed — if the defaults
       resolve to anything else, Phase 4 is wrong
2. [ ] `catalogue_exclusion_cost` gets **no waiver**. If the cluster will not settle without one,
       Phase 3's queue-severity filter is wrong
3. [ ] Re-render `725bae7bf8ed` and confirm `SETTLED (3 findings waived across 2 checks)` with the
       fix history attached

**Verification**:
- [x] `"$PY" -m src validate fix show 1c9a517eaa47` answers reason / what / before / after / pending
- [x] `SELECT count(*) FROM fundamentals_check WHERE run_id = '725bae7bf8ed'` **unchanged** by any
      of the above. If it moved, the anti-suppression invariant is broken
- [x] Negative test: inject a synthetic unwaived `high` finding → the cluster un-settles
- [x] Negative test: delete the fix row → the cluster reads `WONTFIX`, not `SETTLED`

## Testing Strategy

**Unit** — `test_ledger.py` (nested waiver map, DATE round-trip, multi-fix ordering,
`qualifying_fix`), `test_clusters.py` (the full settlement algebra, seven cases),
`test_cli_fix.py` (every refusal + the warn-and-write path + atomic rollback),
`test_substrate_contract.py` (the general projection rule).

Settlement algebra is logic, not economics, so synthetic frames are correct here — AGENTS.md's
real-data rule applies to feature/economic tests. Phase 6's cost measurement uses real data.

**Integration** — the Phase 7 backfill on real recorded runs, plus both negative tests.

**The test that matters most**: `fundamentals_check` row counts byte-identical with and without
waivers and fix rows. Everything else here reverts with `git revert`; a suppression leak does not.

**Manual** — read `fix record --help` and the rewritten Step 8 cold. If either needs this plan
open beside it, they are not finished.

## Risk Mitigation

1. **Issue**: `check_status` is not actually empty and drop-and-recreate destroys a human decision.
   **Mitigation**: the count is the first step of Phase 1; the drop is gated on `0`; fall back to
   `ALTER`. Never drop rows a human wrote.

2. **Issue**: the widened PK breaks `store.save` because the live table was not migrated.
   **Mitigation**: it fails loudly (`ON CONFLICT` with no matching constraint), not silently.
   Phase 1 confirms the constraint in `information_schema` before Phase 2 starts.

3. **Issue**: settlement becomes a suppression list by drift — someone later filters findings on
   the fix table, or relaxes the fix-row requirement.
   **Mitigation**: the invariant is written into the `sql/schema.sql` prose *and* pinned by the
   row-count test; the fix-row requirement has its own two negative tests. A comment alone would
   not survive — that is precisely how the false "read by no check" comment caused Phase 6's bug.

4. **Issue**: adding `adjustment` to the projection slows every Tier-2/3 run.
   **Mitigation**: measure in Phase 6. If the cost is real, fall back to the honesty-only fix
   (denominator above the early return) and decide the column separately.

5. **Issue**: per-check waivers become a way to silence checks one at a time.
   **Mitigation**: decision 1 means waivers alone can never settle anything; the quantified-note
   rule applies to each; each expires on its own `findings_at_decision`; and the render prints the
   waived count beside every settled cluster so the basis is never invisible.

6. **Issue**: a recorded no-improvement fix misleads a later reader.
   **Mitigation**: it cannot settle (decision 8), the write warns, and `queued_before/after` are
   stored so any reader can see it moved nothing.

### Rollback Plan

- Phases 2–6 are ordinary code; `git revert`.
- Phase 1 is the only one-way door. `fundamentals_check_fix` can be dropped outright (nothing else
  reads it). The `check_status` PK can be narrowed the way it was widened, **provided** no
  per-check waiver has been written — after that, narrowing collides two rows onto one key. Post
  Phase 7, delete the per-check waiver rows first.

## Dependencies

- No new external libraries.
- Internal: `src/data_store/` and `sql/schema.sql` (risk zones, approved),
  `src/constants/constants.py`, `src/validate/**`, both agent definitions.
- Docs to keep in sync per AGENTS.md: `src/validate/README.md` (command table + table list),
  `docs/database.md`, `docs/data_schema.md`. `AGENTS.md` needs no change (cap 70 lines).
- A DATA definition-of-done report is required at the end — schema plus validator change.

## Success Criteria

- [x] `validate fix show 1c9a517eaa47` answers reason / what was done / before / after / pending
- [x] `1c9a517eaa47` renders settled-with-waivers, and un-settles when a real finding lands
- [x] Deleting the fix row drops it to `WONTFIX` — waivers alone never settle
- [x] A no-improvement fix row is recordable, warns, and cannot settle
- [x] `fundamentals_check` row counts provably unaffected by fix rows and waivers
- [x] A second fix of the same cluster appends rather than overwrites
- [x] `adjustment_unguarded` reports a real `examined` and surfaces the 61 markers
- [x] Agent B cannot close a `fixed` outcome without a recorded row
- [x] All of `tests/validate/` green; every new test prints a sanity conclusion

## Estimated Effort

| phase | estimate |
|---|---|
| 1 — schema + migration | 1.5 h (verification is most of it) |
| 2 — ledger read layer | 2 h (`qualifying_fix` is new since v1) |
| 3 — settlement + rendering | 3.5 h (seven-case algebra carries the test surface) |
| 4 — CLI | 2.5 h (atomicity + derivation added since v1) |
| 5 — agent definitions | 1 h |
| 6 — `adjustment_unguarded` | 1.5 h (incl. measuring read cost) |
| 7 — backfill + proof | 1 h |
| **total** | **~13 h** |

## Notes for Implementation

- **Phases 1→2→3 are strictly ordered.** 4–6 in any order once 3 is green; 7 last.
- Put the settlement predicate in **one** place (`Ledger.qualifying_fix`). Two copies drift.
- Postgres `DATE` returns `datetime.date`, never `Timestamp`. Normalise once on load in
  `ledger.py` — a parquet-cached test harness hides this whole bug class.
- Table names only via `Tables.<name>`; no string literals.
- `evidence` is JSON, never prose — the rule `fundamentals_check.detail` already follows. Prose
  goes in `root_cause`.
- Do **not** store `checks_closed`; it is derivable from the run pair. Counts *are* stored, on the
  `findings_at_decision` precedent: a stored measurement answers the question actually asked, even
  after the ledger is pruned.
- `store.ensure_table` is check-then-create with no lock. Nothing here writes concurrently; do not
  introduce a threaded writer against a cold table.
- Every new test prints a sanity-check conclusion, per AGENTS.md.
