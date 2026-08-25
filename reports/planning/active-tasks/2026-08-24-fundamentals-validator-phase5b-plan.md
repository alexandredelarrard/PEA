# Implementation Plan: Phase 5b — the fundamentals validator toolkit

**Date Created**: 2026-08-24
**Planning Phase**: 2 of 3 (FIC workflow)
**Supersedes**: `2026-08-23-fundamentals-rebuild-plan-v2.md` §Phase 5b (which itself absorbed v1 §Phase 7).
That section's *intent* is preserved; its **shape, ownership and half its check list have changed** as a
result of the planning interview below. Read this file, not §5b.
**Next Phase**: Implementation (`/implement`)

---

## Overview

One committed, reusable toolkit that answers *"are these numbers right, for these tickers?"* — and,
critically, **the instrument every new `fundamentals_history` field must pass before it is trusted.**
That second purpose is the user's stated requirement and it is what most reshaped the design: the
validator is not a one-off acceptance gate for Phase 9, it is a **standing loop** —

> run → findings land in a table → an agent challenges and investigates each one → accept / fix /
> wontfix, recorded with evidence in a config JSON → re-run → the finding never fires again.

Nothing blocks. The nightly build of `fundamentals_facts` and `fundamentals_history` runs to
completion regardless of what the validator finds. This is the SEC's own warn-over-reject precedent
(v2 §5b.0), and it is now a decision rather than an inference.

---

## The substrate 5b runs on — Phase 5a CLOSED, verified 2026-08-24 16:43

**Phases 0 → 5a are done.** All eight §5.8 gates PASS on rebuilt data. These are the actual numbers
5b calibrates against — not estimates, and **not the 52-ticker figures quoted throughout v2**
(the roster is **54**). Source: v2 §"Phase 5 VERIFIED on the rebuilt data" and §"PHASE 5a FINAL STATUS".

| | |
|---|---|
| `fundamentals_facts` | **317,036 rows / 54 tickers**, 252,001 valued. (Was 294,898 under the label PK: **+22,138, +7.5%**) |
| `fundamentals_history` | **3,267 rows / 54 tickers**, 2009-07-31 → 2026-08-10. **69 stored = 69 contracted**, same order |
| `fundamentals_reason_codes` | **76,004 rows**. `ALL_CODES` = **19**, `IS_QUALIFIER` = **6** |
| `fundamentals_employees` | **745 rows / 54 of 54 tickers** |
| the null gate | 196,020 cells, **71,857 null (36.7%)**, **UNEXPLAINED 0** |
| grain | 0 duplicate `(ticker, as_of)`, 0 `fiscal_end` regressions, **0 look-ahead leaks** |
| filing lag | median **34 d**, p90 55 d, **1 of 3,267 beyond 200 d** — and that one is real (SMCI's delinquent FY2017 10-K, 686 d, filed during its Nasdaq delisting) |
| amendments | 21 rows / 14 tickers; 36 accessions (1.09% of filings); **3 of 36 (8%) refused by the >365-day cutoff** |
| same-day collapse | 9 of 3,273 pairs carry >1 accession, max **4** |
| regime top lines | revenue / equity / assets / **liabilities 100% in ALL EIGHT regimes** |
| `totalLiabilities` | NULL fell **210 → 38**; all 38 lack `stockholdersEquity` outright, **0 have both inputs present** |
| the replay | 4 h 39 m for 54 tickers (2.4 min/ticker after the numpy rewrite; was 14 min) |

**Reason-code base rates — these are `coverage_field`'s and `series_shape`'s starting denominators:**

| code | fires | tickers |
|---|---|---|
| `not_disclosed` | **68% of all codes** | — |
| `not_applicable_for_regime` | 14,592 | 25 |
| `period_intersection_partial` | **205** (v2's B.6.6 estimated 128) | 19 |
| `derived_identity` (tagged or **deduced** NCI — evidence) | **749** | 17 |
| `derived_identity_nci_assumed_zero` (the assumption) | **152** | 6 |

**⚠ Sequencing consequence (amends decision 55).** The backfill this plan was written to work around
**has already finished.** 5b-core.4's calibration and the baseline ratchet are **unblocked now** —
there is no reason to defer them behind a network wait. The two-part split of 5b-core.1 still stands
as a work-ordering convenience, not a dependency.

---

## Current State Analysis — what the v2 §5b text gets wrong

The v2 plan's §5b was written before Phases 4c and 5 landed. Ten of its Phase-5b statements are already
done, already contradicted, or wrong:

| v2 §5b says | verified reality |
|---|---|
| 5b creates `configs/fundamentals/fundamentals_rosters.json` | **already exists** — `in_sample` (26), `out_of_sample` (26), `amendment_pair` (SMCI/ADM), each with a per-ticker `_why`. No `random_cold` roster for §9.4 yet. |
| `src/validate/` recreated here | correct — it does not exist |
| `fundamentals_quality` | exists in **no** `.py` and no `.sql`. Plan prose only. |
| the column contract is 68 | it is **69**, and there is a new `fiscal_quarter` column (Q1-Q4 on every row incl. TTM/instants) |
| `unexplained_null` is a Tier-1 check | **already a build-time assertion** — `build_ticker` asserts `ALL_CODES` membership, and `scripts/verify_fundamentals_history.py` gate 3 already computes it over the live tables |
| `pit_leak` is a Tier-1 check | **already exists** as `build_history.diff_against_stored`, exact comparison, `datetime.date` round-trip handled |
| `coverage_field` needs an oracle | `scripts/audit_absence_evidence.py` **is** that oracle — STRUCTURAL / UNIVERSAL / MIXED per (regime, field), from stored facts only, zero rules |
| `level_outlier` reuses `src/utils/outliers.py` | that function's signature is on the **facts** grain (`ticker/field/duration_type/fiscal_year/fiscal_period/value/filing_date`), *not* the wide 69-column history grain |
| `level_outlier` = "MAD modified-z > 3.5 on **QoQ log change**" | `outliers.py:118` actually scores `modified_zscore(vals)` on **raw levels** (plus a YoY `diff(4)` check). **The plan and the code specify different rules.** MAD on raw levels flags the entire recent era of any growing company — a 10× revenue growth over 15 years makes every recent quarter a "level outlier" |
| 5b.4 lists `split_basis_mismatch` as a check | it is a **`dc_code`**, not a check. Nothing in 5b implements it; the row is a mislabel |
| `FAILED_HARD_GUARD` | **already reserved** in `reason_codes.ALL_CODES`, unused |

Existing instruments that overlap: `scripts/verify_fundamentals_history.py` (8 gates),
`scripts/audit_absence_evidence.py`, `scripts/measure_total_liabilities_legs.py`,
`scripts/sweep_fundamentals_resolution.py`, `scripts/report_fundamentals_sweep.py`.

---

## Inherited from Phase 5a — the five items §E hands to 5b, by name

These are not new scope discovered here; they are v2 §E's explicit hand-off. Two were already in this
plan; three were not, and one of them is substantial.

| # | handed-on item | status in this plan |
|---|---|---|
| E-1 | **`cross_identity` must treat `derived_identity` and `derived_identity_nci_assumed_zero` rows as INPUTS, never corroboration** | ✅ already specified, Tier 1. Now sized: **901 rows across 18 tickers** are derived, 152 of them on the NCI assumption |
| E-2 | **The MIXED population is the validator's real work queue** — 31 of 48 industrial fields resolve for some filers and not others, so no config rule can decide whether an absence is legitimate | ✅ already specified: `coverage_field` maps MIXED → `medium`. Confirms the design; MIXED being the *largest* bucket is the honest headline |
| E-3 | **Register the cross-regime top-line family per regime.** `audit_absence_evidence` reports **7 unregistered structural fields for `energy`, 5 for `utility`** — but those regimes have only **4 filers each** here, and 0-of-4 is not evidence | ✅ **DESIGN the bar in 5b, write NO cells until Phase 9 (decision 67).** The bar goes in the register's own `_authority` block: a `by_regime` `expected_absent` cell requires **≥N filers in the regime, 0 of N ever resolving, and no adjacent-regime filer resolving it on the same concept.** 0-of-4 does not become evidence by being written down. On today's roster only `noninterestIncome` (0/27 industrial) would qualify — which is exactly what §C.3 measured |
| E-4 | **Give an ABSENCE the evidence a presence has.** A resolved fact row stores `source_concept` / `role_uri` / `roll_up_children`; an unresolved one stores NULL in all four. So `not_disclosed` — **68% of all codes** — is a verdict about our concept MAP, not a checkable claim about the filing | ✅ **ADOPTED into 5b-core (decision 66)** — built now, backfilled at Phase 9. See below |
| E-5 | `test_composites_config.py::real_panel` and the 9 `tests/data_aggregate` failures | → **Phase 6**, not 5b. Named here so nobody re-adopts them |

### E-4 is the one that changes 5b's shape

`coverage_field`'s severity ladder, `series_shape`'s gap diagnosis and the whole `accepted`-with-evidence
loop all rest on reading a `dc_code`. But `not_disclosed` — **68% of the 76,004 codes** — cannot
distinguish *"the filer has no such line"* from *"the filer tagged it under a name we do not know."*
Agent B currently has to open the filing by hand for every one of them, which is precisely the manual
step the toolkit exists to narrow.

The durable fix is to record, on the unresolved stub row, **what the filer DID tag on the relevant
statement** — the concepts present under the same `role_uri` / `root_anchor` that we did not recognise.
That turns `not_disclosed` from a verdict into evidence, and it is what would let `coverage_field`
separate a genuine structural absence from a gap in our concept map **mechanically**.

**Decision 66 — build it in 5b-core, backfill at Phase 9's widening.** The `fetch_fundamentals_sec.py`
change lands now, so every new ticker and every nightly filing carries the evidence immediately. The
54-ticker backfill **rides along with Phase 9's full-universe fetch**, which is already a multi-hour
job — so E-4 costs approximately **zero extra network**. Rejected: re-fetching the 54 now (hours of
network before calibration can start, for a roster Phase 9 will re-fetch anyway); and deferring to
Phase 9 (68% of codes stay unfalsifiable and agent B opens filings by hand for all of them).

**Shape**: on the unresolved stub row, record the concepts the filer DID tag under the same
`role_uri` / `root_anchor` that the resolver did not recognise.

```
fundamentals_facts, unresolved stub TODAY        AFTER E-4
  dc_code          not_disclosed                   dc_code            not_disclosed
  source_concept   NULL                            role_uri           .../BalanceSheet
  role_uri         NULL                            unmatched_concepts ["aapl:MarketableSec…",
  roll_up_children NULL   <- verdict, no evidence                      "us-gaap:OtherAssets…"]
```

- [ ] Until the backfill lands, `coverage_field` and `series_shape` run on the codes they have **and
      report their own coverage of the new evidence** as a number, so the gap is visible rather than
      assumed away.
- [ ] The columns go on `fundamentals_facts` — **risk zone, propose the DDL with the `fundamentals_check`
      diff** so it is one approval.

### Constraints carried from §C — DO NOT DO. Each would re-introduce a measured defect

1. **Do NOT add `roll_up.any_of` for `totalLiabilities`** so route 3b can sum the legs. 0 of 44 10-Ks
   declare a `Liabilities` total; leg-sets vary by filer *and* year; an unlisted us-gaap sibling is
   dropped **silently**. The failure mode is a balance-sheet total short by a caption that looks
   entirely plausible. `cross_identity` must not "helpfully" propose this.
2. **Do NOT write the absence register from the Reg S-X caption list.** Element names are not evidence
   of what a filer declares — refuted three times now.
3. **Do NOT mark the cross-regime top lines `expected_absent` for `industrial`.** Of 27 industrial
   filers only `noninterestIncome` is 0/27. `premiumsEarned` is real for UNH **($72bn)**, CVS
   ($34bn), DE ($248M); `rentalIncome` for AMT ($3.5bn), CAT, BA, CSCO. This bounds E-3's evidence bar.
4. **Do NOT assume a NULL `minorityInterest` is zero as a blanket rule** — correct only where the filer
   has never tagged one, tested **point-in-time**. TMO looks refusable on lifetime counts but files its
   first NCI on 2022-02-24 against a history opening 2011-11-04.
5. **Do NOT relax route 3b's strict period intersection.** The obvious repair reintroduces mixed bases.
6. **Do NOT key anything on `(fiscal_year, fiscal_period)`.** `period_end` is the fact identity — the
   labels collide **18,604 times in 337,190 rows (5.5%)**. Binds `series_shape`, `cross_vintage`,
   `leaf_vs_total` and `duplicate_fact`, all of which join across filings.

### Two §D items 5b can close cheaply, since it is already measuring

- **5a-4** — v2 says SCHW `cash` is refused with no null for a gate to find. In the rebuilt table SCHW
  `cash` is **60/60 non-null** with only `regime_break` ×4, and `period_intersection_partial` never
  fires on SCHW. **Verify the condition still exists before ticking or removing it** — one measurement,
  and it falls straight out of `series_shape`'s calibration.
- **5a-2 — the ETN dual-registrant shell, and it is an acceptance-corpus case for the NEW checks.**
  Two registrants filed a 10-Q for period 2012-09-30 during Eaton's Irish redomicile: Eaton Corp (Ohio)
  on 2012-10-31 with assets **$18,800M**, and Eaton Corp plc on 2012-11-14 with assets **$5M** — the
  holdco's shell balance sheet. Two rows is the *correct grain*; the second row's **values** are the
  defect, and the series reads as the company evaporating. **`trend_break` (3,760×) and `series_shape`
  must both flag it** — 1 ticker, 1 period, 2 rows of 3,267 (0.06%), precisely bounded.

---

## The defect-class coverage matrix

The question that produced four new checks: *"do you have a check to identify all the issues we
bumped into so far?"* — answered honestly, as a matrix, because v2's check list was assembled
mechanism-by-mechanism and never audited against the archive as a whole.

| # | defect class actually hit | check | added by this plan? |
|---|---|---|---|
| 1 | wrong-parent weight (MSFT SG&A −$34.7bn) | `sign_convention`, `impossible_value` | |
| 2 | note-level fact beats the statement line (AMT $1.9M, CSCO 2.3×, MCD 12×, PG $28.4bn) | `basis_step`, `leaf_vs_total` | |
| 3 | zero / negative top line (APA, ETN; VRT correct) | `impossible_value`, `level_outlier` | |
| 4 | mid-history basis break (MTB post-provision, AXP) | `tag_switch_break` | |
| 5 | annual masquerading as a quarter (ORCL $39,068M) | `q4_footing`, `scale` | |
| 6 | restatement (BAC 98,581→102,769) | `restatement_ledger` | |
| 7 | duplicate fact (ORCL 7,623 / 7,600) | `duplicate_fact` | |
| 8 | frozen / staircase TTM | `frozen_series` | |
| 9 | CIK truncation (APA, GOOGL, ETN) | `filing_continuity` | |
| 10 | quarter-grid holes (MAA 17, JNJ 8, GS 3, DE 1, VRT 1) | `coverage_quarters` | |
| 11 | split basis (AAPL 24.3bn shares) | `split_basis_mismatch` **dc_code**, not a check | |
| 12 | **a field goes dark mid-history** (VLO capex from 2023-07, 21 of 63 filings) | **`series_shape` → `early_stop`** | ✅ **NEW** |
| 13 | **a field starts mid-history** (AAPL `totalDebt` 2013, `operatingLeaseLiability` 2020) | **`series_shape` → `late_start`** | ✅ **NEW** |
| 14 | **random interior holes** in an otherwise complete series | **`series_shape` → `interior_gap`** | ✅ **NEW** |
| 15 | **resolved to an entirely wrong concept** (`totalDebt` = a lease liability: BRK-B $4.9-6.3bn, META $7.6-16.7bn, PGR $179-211M) | **`peer_ratio`** | ✅ **NEW** |
| 16 | dimensioned SUBSIDIARY fact read as the group's (DTE capex 17% low) | **`dimensional_scope`** standing assertion | ✅ **NEW** |
| 17 | **a value several times its own trailing level**, in a field lumpy enough that a MAD z-score misses it | **`trend_break`** | ✅ **NEW** |
| 18 | **MAD scored on raw levels flags the whole recent era of any growing company** | `level_outlier`, kernel fixed to QoQ log change | ✅ **FIX** |
| 19 | **a restatement silently ignored** — history keeps a stale-but-genuinely-filed value. Passes every level, identity, outlier and cross-vintage check | **`vintage_currency`** | ✅ **NEW** |

`coverage_field` fires **per cell**. It could never distinguish a random hole from a legitimate late
start, because it never looks at the **shape of the series**. That was the missing dimension, and #12
was hand-waved in v2 as *"a `coverage_field` + `basis_step` finding"* — neither of which detects a shape.

---

## Decisions taken — 2026-08-24 planning interview (40-68)

Numbering continues from the v2 plan, whose last decision is 39.

| # | question | decision | rejected, and why |
|---|---|---|---|
| 40 | **The builder/validator boundary.** Three Tier-1 checks already exist as build-time assertions that FAIL the build; Layer A wants to mutate an append-only table. | **The builder owns invariants; the validator NEVER mutates.** Hard guards move *into* `build_history`, applied before the row is written. The three existing assertions stay in the builder; the validator calls the same functions to *report the number*. `verify_fundamentals_history.py` is absorbed into `src/validate/`. **v2's "two layers, Layer A mutates" framing is retired** — it contradicted §5.0's immutability and would have broken `diff_against_stored`. | a post-hoc `UPDATE` on `fundamentals_history` — a historical row would change value after the fact, so yesterday's cube and today's disagree; and keeping both implementations — two implementations of one rule is what `reason_codes.py` was created to prevent |
| 41 | **Which substrate does each tier read?** | **Tier 1 on `fundamentals_history` (+ `fundamentals_reason_codes`); Tiers 2 and 3 on `fundamentals_facts`.** Findings from the facts grain map back to a `(ticker, as_of)` so the report stays readable per publication event. | everything on history — `frozen_series` and `level_outlier` fire *by construction* on the ~20 forward-filled instant columns, and `q4_footing`/`holdout_q4` cannot run at all because history carries TTM; everything on facts — the validator would then never check the table the cube reads |
| 42 | **Finding lifecycle.** | **Append-only ledger + an evidence-keyed accept register.** `fundamentals_check` is append-only per run; `configs/fundamentals/fundamentals_check.json` holds settled findings with their evidence. The report shows OPEN findings only. | a mutable `state` column — lost on any table rebuild, and "why was this accepted?" would live in free text outside git; report-only — the same known-good rows get re-investigated forever |
| 43 | **Scope.** | **Split into 5b-core and 5b-stats.** 5b-core is everything Phase 9 blocks on. *(Refined below — `basis_step`, `tag_switch_break`, `series_shape` and `peer_ratio` are in core.)* | one phase — a long unverified stretch, and thresholds calibrated before a single Phase-9 finding exists |
| 44 | **New fields.** | **A `probation` status + a per-field acceptance command.** A new catalogue field is born `status: probation`; its findings are recorded at `info` and excluded from the queue. `validate fundamentals --field X --roster in_sample` prints the acceptance sheet. Promotion to `active` requires the sheet clean or its gaps recorded in `fundamentals_check.json`. | requiring the register cells authored up front — you would have to guess the register before measuring, which is exactly how the withdrawn "UNH has no premiums" edit nearly happened |
| 45 | **What gates.** | **NOTHING gates.** The validator runs ad-hoc and nightly, writes findings, and never blocks the nightly fill of `fundamentals_facts` / `fundamentals_history`. The queue is worked by agents. | quarantining critical cells at the cube boundary (proposed, rejected) and failing the nightly step — one filer's bad quarter would stall 500 tickers |
| 46 | **Hard guards.** | **Four impossible-only rules**, applied in `build_history` before the write: `totalAssets < 0`, `sharesOutstanding <= 0`, `basicShares <= 0`, `dilutedShares <= 0`. Plus a **`rejected_value` payload column** on `fundamentals_reason_codes`, because a nulled DERIVED value (a TTM, a `derived_identity`) has no fact row and would otherwise be lost. Everything else v2 listed becomes a **flag-only** Tier-1 check. | v2's four rules as written — the `[-1,1]` ratio rule nulls HCA's correct negative `debtToEquity` and every filer whose debt exceeds equity, which is the 745-row failure mode repeating |
| 47 | **Finding payload.** | **A self-contained investigation packet.** Identity + observed/expected + full provenance + the EDGAR URL + a check-specific `detail` JSON. Agent B settles a finding in one hop, without re-deriving. | identity-only with an on-demand join — Tier 2/3 findings on derived values have no single fact row to join to; a prose `message` — an agent parsing prose to decide what to fix is the failure mode this rebuild exists to remove |
| 48 | **Outcomes.** | **Three: `accepted` / `fixed` / `wontfix`**, each with evidence, decider and date. **A `fixed` outcome MUST also add its case to the acceptance corpus as a named regression test.** | no mandatory regression test — §3c.8 is the precedent: four defects were *created by* the 3c.1-3c.5 fixes and were only visible on the re-sweep |
| 49 | **Severity.** | **A provability ladder** (see below). Purely a work-queue order, since nothing gates. | an impact ladder — needs a per-field weighting nobody has measured, and it hides small provable defects that indicate a systematic bug |
| 50 | **`sign_convention`'s debit/credit oracle.** | **A generated, committed `configs/fundamentals/fundamentals_concept_balance.json`** covering only the concepts our catalogue enumerates (~200-400 entries, a few KB). A missing concept makes the check **ABSTAIN** on that row, and the abstention is itself reported. | v2's stated default of caching the full `us-gaap-2025` XML under `data/` — a 10-30 MB artifact either committed (bloat) or downloaded on demand (tests need network), for coverage of concepts we do not use |
| 51 | **Tier 4 (Tiingo / Yahoo).** | **Defer, with a named trigger**, exactly as Tier 0 was deferred. The plan's own citation argues against it: Boritz & No measures aggregators disagreeing with the 10-K at **6.5-7.7%**, ~10× the effect sizes Tier 3 measures. Only tier with a network cost and a paid dependency, and it never gated anything. | building it — a noise floor above the signal |
| 52 | **Naming.** | `fundamentals_check` (table) + `configs/fundamentals/fundamentals_check.json`. v2's `fundamentals_quality` / `fundamentals_known_findings` are renamed; stale references fixed in one pass. | — |
| 53 | **Nightly scope.** | **Tier 1 nightly over the full table; Tiers 2-3 nightly only on tickers that received a filing that night** (5-8 typical, up to 80 at earnings peak) — a series can only change where a filing landed. A full Tier 2-3 pass over all 500 is `--roster all`, on demand. | all tiers over all tickers nightly — a full Tier-3 cross-vintage pass over ~28M fact rows, which Phase 10 already flags as the validator's efficiency risk |
| 54 | **The agent loop.** | **5b ships both agents as committed `.claude/agents/` definitions** — `fundamentals-validate` and `fundamentals-triage`. The loop is reproducible rather than re-prompted, and the rules are encoded (notably: *challenge the check before challenging the data*). | leaving it to prompting — the DQC_0118 lesson only holds if "is this a threshold bug?" is asked every time |
| 55 | **Sequencing.** | ⚠ **SUPERSEDED by the substrate section above.** The 54-ticker backfill FINISHED 2026-08-24 16:43, so calibration and the ratchet are unblocked now. The original decision — start the no-data work first — stands only as a work-ordering convenience. | closing Phase 5 first (moot, it is closed); calibrating on a partial table (moot, it is complete) |
| 56 | **Series shape** (`interior_gap` / `late_start` / `early_stop`). | **New check `series_shape`, 5b-core.** A `late_start` is tested against **three oracles in order** — the catalogue's own `regime_break` block (ASC 842 / 606 / ASU 2016-18 / LDTI adoption dates), then the listing / first-trade date or a `cik_cutover` entry, then the modal `dc_code` in the absent stretch. None explains it → `high`. | flagging every late start with no oracle — a one-time queue of every ASC-842 and ASC-606 adopter across ~500 tickers × 60 fields; and ignoring `late_start` — a field the filer started tagging late *because we were missing the earlier tag name* would never surface |
| 57 | **Wrong-concept resolution, generically.** | **New check `peer_ratio`, 5b-core, Tier 2, `high`.** A field's ratio to a stable denominator (`totalAssets` for balance-sheet fields, `totalRevenue` for flows), MAD-scored against the **same-regime peer distribution at the same date**. Abstains below 5 peers. The only rule that catches a value resolved to an entirely wrong concept *without a human noticing first*. | not building it — regime peer groups are small (GS is the only `broker_dealer`), but abstention handles that; deferring to 5b-stats — Phase 9's acceptance would run without the one generic wrong-concept detector |
| 58 | **Dimensional scope.** | **A standing assertion in the validator, `critical`**: no resolved fact carries a dimension member outside `entity_scope`'s legitimate consolidation set. Structurally impossible today — which is why it belongs here, same reasoning as `pit_leak`. §B.5's *"do not relax the dimensional filter to fix capex"* becomes enforceable rather than a note. | leaving it as a resolver invariant — it *is* tested where it lives, but a regression stores a subsidiary's numbers as the group's, 17% low and entirely plausible |
| 59 | **"Is a value 3× its own trend?"** — a MAD z-score is not that rule. For a lumpy field (capex, bank provisions) the MAD is large, so a real 3× jump can score **under** 3.5 and be missed. | **New check `trend_break`, 5b-core, Tier 2.** Value vs the **trailing MEDIAN of the last 4-8 quarters**, flagging ratio > 3× or < 1/3×. Median, not mean, so the outlier cannot drag its own reference. Abstains below 4 prior quarters. | a same-fiscal-quarter trailing median (proposed, **not** chosen) — see the seasonality risk below; and a TTM basis — TTM smears one bad quarter across four windows, so the check would fire 4× per defect and point at the wrong quarter |
| 60 | **`level_outlier`: the plan and the code disagree.** | **The plan wins — fix `src/utils/outliers.py` to score the QoQ LOG CHANGE.** Keep the YoY `diff(4)` check. Add a named test: a synthetic smooth 10× compound-growth series over 60 quarters must produce **ZERO** findings; the same series with one planted 3× spike must produce exactly the spike and its reversion. `outliers.py` has other callers — migrate them in the same change. | keeping raw levels and adding log change alongside — two level-outlier checks in the registry and the raw-level one keeps over-firing on growth; dropping `level_outlier` for `trend_break` — a flat 3× threshold misses anomalies that are large relative to a field's *own* volatility |
| 64 | **What does agent B re-run before closing a finding?** §3c.8's precedent: four defects were *created by* the 3c.1-3c.5 fixes and were visible only on a full re-sweep. | **Two-stage.** Per finding: rebuild + re-validate **only the affected tickers** (minutes), write `outcome: fixed` with **`regression_swept: false`**. Then a **batched full-roster sweep** over all accumulated fixes, before the phase closes, flips the flag — and any NEW finding it surfaces is attributed to a fix. Keeps the per-finding loop cheap while still honouring §3c.8: a fix is not *finally* accepted until a full re-sweep has seen it. | both rosters after every single fix — a resolution-layer fix means a network rebuild plus a full 54-ticker Tier 1-3 pass **per finding**, so the loop is hours and the queue drains slowly; affected tickers with no mandatory re-sweep — a fix that breaks a different ticker sits undetected until someone runs `--roster all` |
| 65 | **Can agent B edit `configs/`?** Many real fundamentals fixes ARE config — the `never_use` entry that closed MTB and AXP, a `by_ticker` register widening, a cutover entry. | **No — propose only.** `configs/` is a risk zone. The JSON records **`fix_kind: code \| config_proposed`**, and a `config_proposed` finding is **not closed** until the diff is approved. | letting the agent apply config edits — the register is the one artifact where a wrong entry is invisible forever, and the withdrawn "UNH has no premiums" edit was one approval away from being written |
| 68 | **The TTM-on-amendment contract is enforced by ONE line and tested by nothing.** User question: *"the TTM should take the 4 last quarters, with the amended value for the amended quarter — not 5 values, and not the stale one. Does it do that?"* **It does** — `_latest_per_window` collapses each window to its latest visible vintage BEFORE `trailing_twelve` sums exactly `iloc[i-3:i+1]`, so five values is structurally impossible and the amended value wins. But the property lives entirely in `duplicated(keep="last")`. | **New check `vintage_currency`, 5b-core, Tier 1, `critical`** — the mirror of `pit_leak`. Recompute the latest-visible-vintage value from the facts and compare against what history stored. A stale-but-filed number passes every level, identity, outlier and cross-vintage check there is; only a direct vintage comparison sees it. | folding it into `pit_leak` — the two fail for opposite reasons (a leak vs a staleness) and an agent needs to know which; relying on `test_amendment_grain.py` — a unit test on 1 filer, not a standing measurement over 21 amendment rows / 14 tickers |
| 62 | **Where does validation CODE live?** It is scattered: `src/utils/outliers.py`, `src/utils/tiingo_comparison.py` (26 KB), `src/utils/yahoo_comparison.py` (20 KB), three `scripts/`, and the checks yet to be written. | **`src/validate/` is the ONE home for validation code, across all domains** — not a fundamentals-only package. Everything check-related moves there; **tests stay in `tests/`**. Measured before proposing: `outliers.py` has exactly one external caller (`scripts/dod/data_profile.py`, using the MAD kernel — `count_mad_outliers`, `mad_center_scale`), and `tiingo_comparison.py` / `yahoo_comparison.py` have **no Python importers at all** — they are orphaned modules referenced only by comments in `constants.py`. | leaving them in `src/utils/` — `utils` is where code goes when nobody decided where it belongs, and it is why there were two absence oracles and two null gates |
| 63 | **`src/validate/README.md`** — the operating manual for future agents. | **A required deliverable of 5b-core.1**, written before the checks so it is a spec rather than a retro-doc. It states the **three-part loop** the whole system runs on: **part 1 EXTRACTION → part 2 VALIDATION (this tool) → part 3 BUGFIX**, and covers how to run it, why, what the output is, and — the part that is usually missing — **when it does NOT work.** | leaving it to the phase report — a report is a record of what happened; a README is an instruction for the next agent |
| 61 | **Where does the outlier family live?** | **`trend_break`, `level_outlier` and `scale` move from 5b-stats into 5b-core.** Phase 9's user check 2 (*"no kink from a definition or tag change"*) is only half-answered without them, and they were named as expected first-report behaviour. `frozen_series`, `periodicity` and `sign_convention` stay in 5b-stats. | leaving the family in 5b-stats — the first acceptance report could not answer *"is any value 3× its own trend?"* |

---

## Desired End State

- `src/validate/` exists again, containing the **one** implementation that knows how to judge a value.
- `fundamentals_check` holds a ranked, explained, **self-contained** finding queue.
- `configs/fundamentals/fundamentals_check.json` holds every settled finding with its evidence, so the
  queue shrinks monotonically and nothing is re-investigated.
- `"$PY" -m src validate fundamentals ...` is a real CLI command with `-t`, `--roster`, `--field`,
  `--tier`, `--since`, `--report`.
- Adding a new catalogue field has a **defined acceptance procedure** ending in `status: active`.
- Every baseline number in the rebuild is a committed, ratcheted measurement rather than a figure in
  a report.
- Two committed agents drive the loop end to end.

---

## Out of Scope

- **Tier 0** (XBRL-US DQC via Arelle) — deferred at v2 decision 25, trigger restated below.
- **Tier 4** (Tiingo / Yahoo) — deferred, decision 51, with a trigger.
- **Register item 19** (`capitalizedSoftware` as a tier-0 companion to R&D) — a *catalogue* decision,
  not a validator check. Removed from 5b's ownership; the report names it as a known residual.
- **Filer-sophistication priors** for Tier-2 thresholds (v2 §5b.3) — recorded as a refinement. Our
  universe is the S&P 500, i.e. the low-error end of that distribution.
- Repairing anything the validator finds. That is the agent loop's job and Phase 6/9's work.

---

## Severity, defined (decision 49)

v2 used four severities and defined none. Since nothing gates, severity is **purely the reviewing
agent's queue order**, and the ladder is *how sure are we the number is wrong*:

| severity | means | examples |
|---|---|---|
| `critical` | **provably** wrong, or a structural contract is broken | an impossible value that reached the table, a PIT leak, a `dimensional_scope` hit, `Assets != Liabilities + Equity` beyond tolerance, a `filing_continuity` gap with no cutover entry and no listing explanation |
| `high` | probably wrong, and a **named mechanism** says so | `basis_step`, `tag_switch_break` with a level step, `series_shape` unexplained, `peer_ratio`, a `coverage_field` hole the register does not excuse, `duplicate_fact` |
| `medium` | a statistical **candidate**; look, do not assume | `level_outlier`, `scale`, `frozen_series`, `series_shape` gaps with a known benign `dc_code` |
| `info` | declared, quantified, no action expected | `register_cost` (NEE $5.2bn), `register_coverage`, `restatement_ledger`, `expected_absent_drift`, a `late_start` matching a standard's adoption date, every probation-field finding |

`info` never enters the work queue. Agent B works `critical` → `high` → `medium`, top-down.

---

## The split

| | 5b-core (blocks Phase 9) | 5b-stats (after Phase 9's first in-sample report) |
|---|---|---|
| infrastructure | table, CLI, `CHECK_REGISTRY`, payload, JSON schema, renderer, both agents | — |
| Tier 1 | all deterministic per-value checks + `dimensional_scope` | — |
| Tier 2 | **`basis_step`, `tag_switch_break`, `series_shape`, `peer_ratio`, `trend_break`, `level_outlier`, `scale`** | `frozen_series`, `periodicity`, `sign_convention` |
| Tier 3 | all eight | — |
| Tier 4 | deferred (design + trigger only) | deferred |
| other | calibration, baseline ratchet, acceptance corpus, script absorption | `constants.py` pass (item 20), the concept-balance map |

`basis_step` and `tag_switch_break` are in core because they are **provenance-change** checks, not
statistical ones — they fire on a `resolution_method` / `source_concept` change coinciding with a level
step — and because §9.2 requires MCD's capex 35.6× step in the first out-of-sample report.
`series_shape` and `peer_ratio` are in core because they cover five archive defect classes nothing else
catches. `trend_break`, `level_outlier` and `scale` are in core by decision 61 — they are the answer to
*"is this value 3× its own trend?"*, which Phase 9's first report must be able to give.

**The four "is this number wrong?" checks are deliberately overlapping and non-redundant**, because each
misses what another catches:

| | catches | misses |
|---|---|---|
| `trend_break` | a flat 3× level jump, interpretable, any field | a 2.3× step (CSCO `depAmort`); anything seasonal is a false positive |
| `level_outlier` | an anomaly large relative to the field's **own** volatility | a large jump in an already-lumpy field, where the MAD is wide |
| `basis_step` | a step at a **route boundary**, at ANY magnitude — 2.3× included | a wrong value that never changes route |
| `peer_ratio` | a value in the wrong **units or concept entirely**, even if its own series is perfectly smooth | anything where the whole regime is wrong the same way |

---

## Implementation Approach

### Phase 5b-core.1 — the foundation that needs no data ✅

**Goal**: everything buildable while the backfill runs. Nothing here reads a populated table.

**Changes**:

1. `src/data_store/schema.py` + `sql/schema.sql` — **risk zone, propose the DDL as a diff**:
   - [ ] `Tables.fundamentals_check`. PK `(run_date, check_name, ticker, field, period_key)`, where
         `period_key` is TEXT — the `as_of` for history-grain checks, the `period_end` for facts-grain
         checks, `''` for ticker-level checks (`register_coverage`, `filing_continuity`), and a range
         for series-grain checks (`series_shape`). Append-only, `run_date` in the key.
   - [ ] A stable `finding_id`: a deterministic hash of `(check_name, ticker, field, period_key)`.
         **This is what `fundamentals_check.json` matches on**, so a finding keeps its identity across runs.
   - [ ] Payload columns implementing decision 47: `severity`, `observed`, `expected`, `deviation`,
         `source_concept`, `resolution_method`, `roll_up_children`, `root_anchor`, `role_uri`,
         `accession_number`, `edgar_url`, `detail` (JSON).
   - [ ] **`rejected_value DOUBLE PRECISION` on `fundamentals_reason_codes`** (decision 46) — same
         payload-column pattern as the existing `combined_into`, NULL for every code but
         `failed_hard_guard`.

2. `src/validate/` — recreated as the **cross-domain home for all validation code** (decision 62).
   Domain-scoped from day 1, so a future prices or insider validator has an obvious place and this does
   not become a fundamentals package with a generic name:

   ```
   src/validate/
     README.md                     the part-2-of-3 operating manual (decision 63)
     outliers.py                   MOVED from src/utils/ -- the shared MAD kernel
     external/
       __init__.py
       tiingo_comparison.py        MOVED from src/utils/ -- fetch-only adapter
       yahoo_comparison.py         MOVED from src/utils/ -- fetch-only adapter
     fundamentals/
       __init__.py
       validator.py                FundamentalsValidator -- the ONE implementation
       check_register.py           reads configs/.../fundamentals_check.json
       report.py                   markdown + printed renderer
       checks/
         __init__.py               CHECK_REGISTRY
         tier1_value.py
         tier2_series.py
         tier3_internal.py
   ```

   - [ ] `validator.py` loads each substrate **once, projected**, and passes the frame down the tiers
         (Phase 10's instruction).
   - [ ] `checks/__init__.py` — `CHECK_REGISTRY`. Never hand-list what a registry drives. Each check
         declares `name`, `tier`, `substrate` (`history` | `facts`), `severity`, `grain`, and an
         **`expected_fire_rate_ceiling`**.
   - [ ] `check_register.py` — subtracts settled findings by `finding_id`, and reports **stale entries**
         (a settled finding whose check no longer fires) so the register decays visibly.

2b. **The code moves** (decision 62). Measured, so the blast radius is known before the diff:
   - [ ] `src/utils/outliers.py` → `src/validate/outliers.py`. **One external caller** —
         `scripts/dod/data_profile.py` imports `count_mad_outliers` and `mad_center_scale`. Update it in
         the same change; it uses the statistical kernel, not `detect_level_outliers`, so decision 60's
         log-change fix does not affect it.
   - [ ] `src/utils/tiingo_comparison.py` → `src/validate/external/tiingo_comparison.py`;
         `src/utils/yahoo_comparison.py` → `src/validate/external/yahoo_comparison.py`. **Zero Python
         importers** — they are referenced only by comment blocks in `constants.py`. Reduce both to
         **fetch-only adapters** (return a comparable frame, hold the URL templates); all ranking,
         bucketing and verdict logic moves into the validator. Update the `constants.py` comment paths
         as part of item 20.
   - [ ] Tests **stay in `tests/`** — the user's explicit carve-out. `tests/validate/` is the new home
         for the validator's own tests.
   - [ ] **`src/data_extract/utils/structure/def14a_validate.py` does NOT move.** *Stated assumption,
         flag it to override*: despite the name it is a **repairer** inside the extraction pipeline (it
         fixes the hardcoded-0.5-pct and missed-thousands defects in edgartools' DEF 14A parse), not a
         check. Moving it would make `src/validate/` a mutating package, which contradicts decision 40's
         "the validator never mutates".

3. `configs/fundamentals/fundamentals_check.json` — **risk zone, propose the diff**. Schema:
   ```json
   {"finding_id": "...", "check": "level_outlier", "ticker": "VRT",
    "field": "totalRevenue", "period_key": "2018-01-01..2020-12-31",
    "outcome": "accepted",
    "evidence": "GS Acquisition Holdings blank-cheque shell pre-merger; $690M IPO, trust dividend income, genuinely no revenue. Verified in accession 0001628280-20-002144.",
    "fix_kind": null, "commit": null, "regression_test": null,
    "regression_swept": false,
    "decided_on": "2026-08-24", "decided_by": "..."}
   ```
   - [ ] `outcome` in `{accepted, fixed, wontfix}`; `fixed` **requires** `commit` and `regression_test`
         non-null (asserted by a test, decision 48).
   - [ ] **`fix_kind` in `{code, config_proposed}`** (decision 65). A `config_proposed` finding is NOT
         closed — it stays in the queue until the diff is approved. Asserted by a test.
   - [ ] **`regression_swept`** (decision 64): `false` when agent B closes a finding on the affected
         tickers alone; flipped to `true` by the batched full-roster sweep. A test asserts no finding
         reaches phase closure with `outcome: fixed` and `regression_swept: false`.
   - [ ] `wontfix` **requires** a quantified cost in `evidence` (NEE $5.2bn, PLD).
   - [ ] **Hand-formatted, like the other fundamentals configs.** A `json.dumps` round-trip reformats
         the whole file; use a validated emitter or a text splice.

4. `src/cli.py` — `validate fundamentals` as a real command:
   `[-t TICKER] [--roster in_sample|out_of_sample|amendment_pair|random_cold|all] [--field FIELD] [--tier 1,2,3] [--since DATE] [--report PATH] [--no-write]`

5. **Script absorption** (decision, Q7):
   - [ ] `verify_fundamentals_history.py` → `tier1_value.py`. Its 8 gates ARE Tier 1, already tested
         against live tables. **Delete the script**, replace the runbook line with the CLI.
   - [ ] `audit_absence_evidence.py` → `coverage_field`'s oracle. **Delete the script.**
   - [ ] `measure_total_liabilities_legs.py` → `cross_identity`'s evidence. **Delete the script.**
   - [ ] `sweep_fundamentals_resolution.py` and `report_fundamentals_sweep.py` **stay** — threadpool
         EDGAR fetches, a different job with a different cost model.

6. `build_history.py` — the hard guards (decision 46), applied **before the row is written**:
   - [ ] `_hard_guard()`: the four impossible rules. Nulls the value, writes a `failed_hard_guard`
         reason-code row carrying `rejected_value`.
   - [ ] History stays immutable; `diff_against_stored` is unaffected.

7. **`src/validate/README.md`** — the operating manual for future agents (decision 63). Written
   **before** the checks, so it is a spec rather than a retro-doc. Required sections:

   - [ ] **Where this sits: part 2 of 3.** The loop the whole fundamentals system runs on —
         **1 EXTRACTION** (`src/data_extract/`, writes `fundamentals_facts` → `fundamentals_history`)
         → **2 VALIDATION** (this package, writes `fundamentals_check`, mutates nothing)
         → **3 BUGFIX** (an agent reads the queue, challenges, fixes, records the outcome in
         `configs/fundamentals/fundamentals_check.json`, and re-runs part 2 to prove it).
         Each part names the artifact it hands to the next, so an agent dropped into any one of them
         knows what it is holding.
   - [ ] **Why it exists.** Nobody validates by re-deriving a value and checking it matches itself.
         Every tier here is *provenance-independent*: it plays the filer's own disjoint evidence
         against itself, or plays peers against each other. And the goal is **not** 100% — Compustat
         runs >2,500 checks *and* mandatory human review, forever. The job is to produce a **short,
         ranked, explained** list.
   - [ ] **How to run it.** Every CLI form, with a worked example of each: nightly, ad-hoc per ticker,
         per roster, per field (the new-field acceptance sheet), per tier.
   - [ ] **What the output is.** The `fundamentals_check` schema, the finding payload field by field,
         the severity ladder and what each level obliges you to do, and a real example report.
   - [ ] **⚠ WHEN IT DOES NOT WORK** — the section that is usually missing, and the most valuable one:
         - checks that **ABSTAIN** and why (`peer_ratio` below 5 peers, so `broker_dealer` is never
           checked; `sign_convention` on a concept absent from the balance map; `trend_break` below 4
           priors; `series_shape` on a short history)
         - checks with a **known false-positive population** (`trend_break` on seasonal filers,
           `coverage_field` on MIXED cells — which are the *majority*, 31 of 48 for industrials)
         - what the validator **structurally cannot see**: a defect in the filer's own calculation
           linkbase (that is deferred Tier 0, with its trigger), and anything where a whole regime is
           wrong in the same direction (`peer_ratio` goes blind)
         - **`companyfacts` can prove a concept PRESENT and can never prove one ABSENT** — every
           coverage claim must be measured off `filing.xbrl()`
         - a `not_disclosed` code is a statement about **our concept map**, not about the filing
   - [ ] **The rules for changing it.** Challenge the check before challenging the data (DQC_0118);
         never accept a finding without filing-level evidence; a `fixed` outcome must leave a
         regression test; thresholds live in code with their measurement in the docstring, baselines
         live in `fundamentals_baselines.json` and only move with evidence.

8. **The two agents** (decision 54). The repo already has three committed agents
   (`codebase-locator`, `pea-architect`, `sec-table-analyst`), so the convention exists.

   - [ ] `.claude/agents/fundamentals-validate.md` — **agent A**. Runs the validator, writes findings,
         summarises and ranks the queue. Thin; the CLI does the work.
   - [ ] `.claude/agents/fundamentals-triage.md` — **agent B**, the read / research / plan / fix / test
         loop. Encoded procedure, in order:

     1. **Read** the investigation packet. It is self-contained by decision 47 — no re-derivation.
     2. **Research**: open the accession on EDGAR at the `edgar_url` and read the filed statement.
        A `not_disclosed` code is a statement about **our concept map**, never about the filing.
     3. **CHALLENGE THE CHECK FIRST.** Is this a threshold bug rather than a data defect? DQC_0118:
        *"inconsistencies reported to filers can be overwhelming as many don't represent real
        errors."* This repo has earned it twice independently — the **745 correct rows** nulled by
        over-strict Q4 guards, and the **withdrawn "UNH has no premiums"** register edit, where the
        *check's premise* was wrong and the numbers were right.
     4. **Plan**, then decide the outcome: `accepted` / `fixed` / `wontfix`, with evidence.
     5. **Fix**, and the terminal state depends on WHERE:
        - **code** (`build_history.py`, `xbrl_linkbase.py`, `periods.py`, a check module) → applied
          by the agent, commit recorded.
        - **`configs/`** (`fundamentals_kpis.json`, `fundamentals_exceptions.json`,
          `fundamentals_regimes.json`, `fundamentals_cik_cutover.json`) → **risk zone, PROPOSE the
          diff, never apply.** A large share of real fundamentals fixes are config: the `never_use`
          entry that closed MTB and AXP, a `by_ticker` extension-register widening, a cutover entry.
          The JSON records `fix_kind: code | config_proposed` — **different terminal states**, and a
          `config_proposed` finding is not closed until the diff is approved.
     6. **Test — and this is the step that is easy to get wrong.** *"Re-run the validator"* is
        **not** a valid verification: it would read stale rows and report a false green. The rebuild
        path branches on which layer was touched, and the CLI already names the two:

        | changed | rebuild before re-validating | cost |
        |---|---|---|
        | `build_history.py`, `reason_codes.py`, a `_FORMULAS` entry | `fundamentals-history --rebuild-history -t X` | ~2.5 min/ticker, **no network** |
        | `xbrl_linkbase.py`, `periods.py`, `fundamentals_kpis.json` — a RESOLUTION bug | `fundamentals --rebuild -t X` (deletes the four tables for X, refetches) | network |
        | a register the validator itself reads (`fundamentals_exceptions.json`, `fundamentals_check.json`) | none | seconds |

     7. **Close**, with the re-run scope set by **decision 64**: rebuild + re-validate the affected
        tickers now, write `outcome: fixed` with `regression_swept: false`. The finding is not
        *finally* accepted until a batched full-roster sweep has seen it.
     8. **Add the regression test** (decision 48) to the acceptance corpus, and record `commit`.

**Verification**:
- [ ] **A synthetic frame with one planted violation per check → each fires exactly once and nothing
      else does.** This is the standard; a check that cannot be planted cannot be trusted.
- [ ] `CHECK_REGISTRY` round-trip: every check has a module, a severity, a substrate and a ceiling; the
      report renderer enumerates the registry, never a hand list.
- [ ] `fundamentals_check.json` schema test: `fixed` without `commit`/`regression_test` fails;
      `wontfix` without a number in `evidence` fails; an unknown `outcome` fails.
- [ ] `finding_id` is stable across two runs of the same data, and changes when any key component does.
- [ ] Hard-guard tests: a planted `totalAssets = -1` is nulled with `rejected_value = -1`; a planted
      `debtToEquity = -3.4` (the HCA shape) is **NOT** nulled and reaches the table.
- [ ] Tests instantiate `FundamentalsValidator` against a frame — **no DB, no CLI**.
- [ ] Every test prints its sanity-check conclusion (AGENTS.md hard rule).

**Effort**: 2 days.

---

### Phase 5b-core.2 — Tier 1, on `fundamentals_history` ✅ (code + unit tests; real-data fire rates in 5b-core.4)

**Goal**: the deterministic per-value tier, vectorised over the whole table, cheap enough for nightly.

| check | rule | notes vs v2 |
|---|---|---|
| `unexplained_null` | 0 rows with `value IS NULL AND` no reason code | **reports the builder's own assertion**; does not re-implement it (decision 40) |
| `pit_leak` | no fact with `filing_date > as_of`; no historical row changed since the last run | calls `diff_against_stored` |
| **`vintage_currency`** | the MIRROR of `pit_leak`: every cell carries the **NEWEST** fact visible at `as_of`, not merely *a* visible one. Recompute the latest-vintage value from `fundamentals_facts` where `filing_date <= as_of` and compare against what `fundamentals_history` stored | **NEW, decision 68.** `pit_leak` only tests that nothing from the FUTURE leaked in. The whole restatement contract rests on one line — `_latest_per_window`'s `duplicated(keep="last")` ([periods.py:196](../../../src/data_extract/utils/fundamentals/periods.py#L196)). Flip it to `"first"` and every amendment is silently ignored: the history keeps a **stale but genuinely-filed** number, so no level, identity or outlier check can see it. `cross_vintage` cannot catch it either — it reads `fundamentals_facts`, which is as-filed and unaffected. Baseline: **21 amendment rows / 14 tickers**, and SPG's `netIncome` 1,949M → 2,155M is the planted case. `critical` |
| `dimensional_scope` | no resolved fact carries a dimension member outside `entity_scope`'s consolidation set | **NEW, decision 58.** `critical`. Makes §B.5's "do not relax the dimensional filter" enforceable |
| `coverage_universe` | every universe ticker has ≥1 row | |
| `coverage_quarters` | contiguous to the **filer's own** projected next filing | **re-specified**: expected = the filer's own `fiscal_quarter` grid + its own median `as_of − fiscal_end` lag (AAPL: 32 d), not a calendar. Works for 52/53-week, Jan/May/Sep year-ends, KR's 16-week Q1 |
| `coverage_field` | a field is expected unless the register excuses it | oracle = the absorbed `audit_absence_evidence` verdicts. **UNIVERSAL + NULL → `high`. STRUCTURAL → no finding. MIXED → `medium`** — and MIXED is the honest majority (industrial: 16 UNIVERSAL / 31 MIXED / 1 STRUCTURAL) |
| `expected_absent_drift` | a value present where the register says `expected_absent` | `info`. The register is measured, so it decays |
| `cross_identity` | `Assets == Liabilities + Equity`; `GrossProfit == Revenue − COGS`; `NetIncome` IS↔CF | **must treat a `derived_identity` / `derived_identity_nci_assumed_zero` `totalLiabilities` as an INPUT, never as independent evidence.** The reason code is already there to make this checkable |
| `filing_continuity` | filings-per-ticker-per-year vs the 4.0-4.2 band | **three excusing mechanisms**: a `fundamentals_cik_cutover.json` entry; a recent first-trade / index-add date; or an `accepted` entry. Only a short history with none of the three is a **missing cutover entry**, i.e. a work item |
| `adjustment_unguarded` | an adjustment fired on silence, not positive evidence | item 10. All 128 `ppeNet` lease adjustments. `info` |
| `register_cost` | the declared, quantified cost of each register exclusion | items 14, 15. `info`, visible every run |
| `register_coverage` | which filers run on a partial register | item 16. 17 of ~500. `info` |
| `impossible_value` | the flag-only residue of v2's Layer A: `abs(epsDiluted) > 1000`, a ratio outside its own measured band, a negative revenue or cost line | **new** — decision 46 moved these out of the mutating guard into a check. `high` |

**Verification**:
- [ ] Fire-rate table printed per check. **A check firing on >2% of rows without a named mechanism is a
      threshold bug, and is reported as such** — enforced against each check's declared
      `expected_fire_rate_ceiling`, not left to a human reading a table.
- [ ] `unexplained_null` = 0 on the built roster (the builder already guarantees it; this proves the
      validator agrees).
- [ ] `dimensional_scope` = 0. If it is ever non-zero, the filter regressed.
- [ ] 0 findings at severity ≥ `high` on AAPL other than ones named in the acceptance corpus.

**Effort**: 1.5 days.

---

### Phase 5b-core.3 — Tier 3, plus the seven Tier-2 checks, on `fundamentals_facts` ✅ (code + unit tests; real-data fire rates in 5b-core.4)

**Goal**: the provenance-independent tier that needs no external data, because it plays the filer's own
disjoint numbers against each other. This is what makes deferring Tiers 0 and 4 defensible.

**Tier 3 — cross-vintage, cross-route, hold-out**

| check | rule | v2 baseline (pre-4c — see the ratchet phase) |
|---|---|---|
| `holdout_q4` | force the derivation where the filer published FY, YTD9 **and** its own discrete Q4 | 591/752 cases; 98.73% / 98.99% within 1% |
| `annual_footing` | four derived quarters vs the filer's own annual, restricted to years whose Q4 is not the identity | 99.12% / 98.78% within 2% |
| `q4_footing` | `Q1+Q2+Q3+Q4 == FY`, only on non-identity years | 99.9% of Q4 rows genuinely testable |
| `leaf_vs_total` | `statement_leaf_sum` in one vintage vs a declared total in another | 89/94 points, 76.40% / 78.72% exact |
| `cross_vintage` | same `(ticker, field, period)` across filings. **Restatement vs defect is separable without external data**: a derivation error leaves ≥1 quarter with `basis != as_reported`; a restatement leaves all four as-filed and they foot to the FIRST-FILED annual | 4.53% of annual windows move >2% |
| `derived_vs_asreported` | our `epsDiluted` vs the filer's; share-day `dilutedShares` vs the filer's annual weighted average | 97.3% of 710 points within 0.5% |
| `duplicate_fact` | one filing tagging `(concept, period)` twice with different values | ORCL $7,623M vs $7,600M |
| `restatement_ledger` | **record, never repair** | BAC FY2023 98,581M as filed vs 102,769M re-presented. `info`. Exists so nobody "fixes" it toward `frames` |

**Tier 2 — the provenance-change and shape checks that belong in core**

| check | rule | why it is here |
|---|---|---|
| `basis_step` | a level step at the exact boundary where `resolution_method` changes | MCD capex **35.6×**, CSCO `depAmort` 2.3×, VLO capex dark from 2023. **No cross-vintage test can see any of them** — the filer tags the same narrow concept consistently in the earlier era |
| `tag_switch_break` | a `source_concept` change coinciding with a level step | user check 2; the SEC's own review category. Base rate 0.67% / 0.71% |
| **`series_shape`** | classify each `(ticker, field)` series against the row grid, and report the **modal `dc_code` inside the gap** | **NEW, decision 56.** See the classification and the oracle ladder below |
| **`peer_ratio`** | ratio to a stable denominator, MAD-scored against the same-regime peer distribution at the same date | **NEW, decision 57.** See below |
| **`trend_break`** | value vs the **trailing median of the last 4-8 quarters**; flag ratio > 3× or < 1/3× | **NEW, decision 59.** The interpretable level rule. Abstains below 4 prior quarters. See the seasonality risk |
| `level_outlier` | MAD modified-z > 3.5 (Iglewicz & Hoaglin) on **QoQ log change**, per `(ticker, field)`, min 8 quarters | **decision 60**: `src/utils/outliers.py` is fixed to log change first; other callers migrate in the same change |
| `scale` | order-of-magnitude jump vs the field's own history | DQC 0091/0095/0103/0139/0157 |

#### `series_shape` — the classification and the oracle ladder

```
complete       value at every expected event                       no finding
interior_gap   present BEFORE and AFTER a hole                     the "random quarters missing" case
late_start     absent, then present from D onward                  the "concept started late" case
early_stop     present until D, absent after                       the "went dark" case (VLO)
sparse         no contiguous run                                   -> hand to `periodicity` (5b-stats)
```

**The gap's modal `dc_code` is the diagnosis** — this is what makes the check precise rather than noisy:

| shape + gap code | verdict | severity |
|---|---|---|
| `interior_gap` + `not_disclosed` | **a missing tag** | `high` |
| `interior_gap` + `period_intersection_partial` | route 3b's strict intersection (item 9 / B.6.6; 128 known rows across EQIX capex 40, EQIX depAmort 40, SCHW cash 34, NEE ppeNet 8, VRT depAmort 6) | `medium` |
| `interior_gap` + `insufficient_quarters` | the TTM window; benign by construction | `info` |
| `late_start` + a `regime_break` date match | ASC 842 / 606 / ASU 2016-18 / LDTI adoption — legitimate | `info` |
| `late_start` + a recent listing / cutover date | legitimate | `info` |
| `late_start`, no oracle explains it | **investigate** | `high` |
| `early_stop`, any code | almost always a defect (VLO capex from 2023-07, 21 of 63 filings) | `high` |

Worked examples the check must reproduce:
- `AAPL totalDebt` → `late_start @2013-07-24`; no `regime_break`, listed 1980, gap code `not_disclosed`
  over 16 events → **`high`** → the agent opens the filing → `accepted`, evidence *"first bond issue
  30 April 2013"*. Settled once, forever.
- `AAPL operatingLeaseLiability` → `late_start @2020-01-29`; matches Apple's ASC 842 adoption →
  **`info`**, never enters the queue.
- `VLO capex` → `early_stop @2023-07`; gap code `not_disclosed` over 21 filings → **`high`**.

#### `peer_ratio`

Ratio to a stable denominator — `totalAssets` for balance-sheet fields, `totalRevenue` for flows —
MAD-scored against the same-regime peer distribution **at the same date**. **Abstains below 5 peers**
(GS is the only `broker_dealer`; `real_estate` has 2 in-sample), and the abstention is reported.

```
BRK-B totalDebt / totalAssets = 0.006
  industrial peers @2021-06-30, n=27
  median 0.281   MAD-z -8.4
  -> HIGH: 47x below the peer median
```

The actual defect it would have caught unaided: `totalDebt` resolved to an operating-lease liability,
$4.9-6.3bn against a real long-term debt in the tens of billions — and the same rule covers META
($7.6-16.7bn), PGR ($179-211M), AMT's $1.9M `longTermDebt`, PG's $28,400M annual revenue and MCD's
12×-low capex, **without anyone knowing the mechanism first**.

#### `trend_break` — and the seasonality cost, stated up front

Value vs the **trailing median of the last 4-8 quarters, any quarter** (decision 59). Median, not mean,
so a spike cannot inflate its own reference. Thresholds `> 3×` or `< 1/3×`. Abstains below 4 priors.

**Known cost, accepted deliberately**: a plain trailing median **will** fire on genuinely seasonal
filers — retail Q4, KR's 16-week Q1, weather-driven utility quarters. A same-fiscal-quarter trailing
median was proposed and not chosen.

- [ ] **The calibration pass must measure the fire rate BY REGIME and report it**, so the seasonal
      false-positive population is a number rather than a surprise.
- [ ] **Named remedy if it dominates the queue**: `fiscal_quarter` is on every row since Phase 5, so
      switching the reference to the same fiscal quarter of the prior 3 years is a one-line change.
      Recorded here so it does not have to be re-derived.

**Verification**:
- [ ] Each check planted synthetically first, then reproduced on real data.
- [ ] **`level_outlier`'s growth test** (decision 60): a synthetic smooth 10× compound-growth series over
      60 quarters produces **0** findings; the same series with one planted 3× spike produces exactly
      the spike and its reversion. This is the test the current raw-levels kernel fails.
- [ ] `trend_break` fires on a planted 3× spike and is silent at 2.9×; the boundary is exact.
- [ ] `trend_break`'s fire rate reported **per regime**, in-sample and out-of-sample.
- [ ] `cross_vintage`'s restatement-vs-defect discriminator proven on **VLO operating income FY2012**:
      first-filed $4,010M = our four quarters to the dollar; last-restated $5,044M → classified
      `restatement`, `info`, not a defect.
- [ ] `series_shape` reproduces all three worked examples above, with the stated severities.
- [ ] `peer_ratio` flags BRK-B `totalDebt` on the **pre-fix** substrate and is silent on the post-fix one.
- [ ] `peer_ratio` abstains, visibly, on `broker_dealer`.
- [ ] Findings from the facts grain map back to a `(ticker, as_of)` and the report reads per event.

**Effort**: 2.5 days (raised from 2 — `series_shape` and `peer_ratio` are new designs).

---

### Phase 5b-core.4 — calibration and the baseline ratchet 🔄

**Goal**: the deliverable v2 correctly refused to make a follow-up. **The substrate is available now** —
the 54-ticker backfill closed 2026-08-24 16:43 (3,267 history rows, 317,036 fact rows, 76,004 codes),
so nothing here waits on network.

**Every baseline in the rebuild is stale** and this phase says so out loud: they predate 4c
(statement-role test, `longTermDebt` reorder, ORCL refusal, CIK cutover) **and** the Phase-5 PK fix
that alone recovered **18,604 rows / 5.5%** that were being silently dropped.

**Changes**:
- [ ] Re-measure every baseline on the current substrate. The v2 figures are relabelled **"pre-4c, the
      number to beat"** — not assertions.
- [ ] `configs/fundamentals/fundamentals_baselines.json` — **risk zone, propose the diff**. Each entry:
      `value`, `measured_on`, `substrate`, `n`, `pre_4c`, `evidence`. A test fails on **degradation**;
      an improvement requires updating the file **with its evidence**. This is finally the mechanism
      register item 6 asked for ("fold the surviving 3c numbers in as standing assertions").
- [ ] Calibrate every threshold **offline** before wiring anything in. Print the fire-rate table.
- [ ] **Thresholds live as module constants with their measurement in the docstring**, not in config —
      a threshold is *behaviour*, and Phase 9 will retune them repeatedly; a `configs/` approval loop
      per tweak would be friction with no safety benefit. Baselines (measurements) go in the JSON;
      thresholds (behaviour) go in code. *Stated assumption — say so to flip it.*
- [ ] **Every rate quoted off `fundamentals_facts_legacy` carries its caveat**: 445 of 500 tickers,
      missing the entire U-W tail (UNH UNP UPS USB V VLO VRT VZ WFC WMT …). It is alphabetically
      biased. Scope every query or it lies. Legacy is a **sanity sample only** — it is a *different
      resolver's* output, so a threshold fitted to its error distribution may not transfer.

**Verification**:
- [ ] Fire-rate table for all checks, in-sample and out-of-sample, side by side.
- [ ] Every baseline entry has `n` and a substrate description; a test asserts that.

**Effort**: 1 day + the backfill's wall-clock (already running).

---

### Phase 5b-core.5 — the acceptance corpus, and the phase's own closure ⬜

**Goal**: a validator that cannot re-detect the defects this rebuild spent three phases finding is not
validated. The archive **is** the test corpus, and it is where every future `fixed` outcome lands.

| historical defect | must be | check |
|---|---|---|
| MSFT `sellingGeneralAdmin` −$34.7bn | silent (regression guard) | `sign_convention` *(5b-stats)* |
| `totalDebt` = a lease liability (BRK-B, META, PGR) | silent after the fix; **`high` on the pre-fix substrate** | **`peer_ratio`**, `cross_identity` |
| APA `totalRevenue` = 0 / −$467M | silent | `basis_step`, `impossible_value` |
| MTB revenue post-provision (110 rows, ~32% low) | silent | `tag_switch_break` |
| MCD capex 35.6× at 2017→2018 | **`high` before 4c, silent after** | `basis_step`, `peer_ratio` |
| CSCO `depAmort` 2.3× two-basis series | same | `basis_step` |
| AMT `longTermDebt` $1.9M note-level | same | `basis_step`, `peer_ratio` |
| **VLO capex dark from 2023-07** | **`high`** | **`series_shape` → `early_stop`** |
| **ETN 2012-09-30: assets $18,800M → $5M** (§5a-2, the Irish-redomicile holdco shell; still OPEN) | **`critical`/`high`** — a 3,760× step. Two rows is the correct grain; the second row's VALUES are the defect | **`trend_break`** + **`series_shape`**. The check that would have caught 5a-2 without anyone reading the filing |
| **SPG 2016-01-13 restates `fiscal_end` 2015-09-30: `netIncome` 1,949M → 2,155M** | the 2015-11-04 row **still reads 1,949M**; the 2016-01-13 row reads 2,155M and its TTM recomputes from four quarters, one of them amended | **`vintage_currency`** (the amended value won) + `pit_leak` (the earlier row did not move). **Plant the inverse**: force `_latest_per_window` to `keep="first"` and `vintage_currency` must fire on all 21 amendment rows |
| **AAPL `totalDebt` from 2013-07** | **`high`, then `accepted` with the bond-issue evidence** | **`series_shape` → `late_start`** |
| **AAPL `operatingLeaseLiability` from 2020-01** | **`info`** (ASC 842 match) | **`series_shape` → `late_start`** |
| ORCL FY2020 Q4 revenue $39,068M | **flagged** | `q4_footing` |
| BAC FY2023 98,581 → 102,769 | **`info`, never repaired** | `restatement_ledger` |
| AAPL FY2012 derived Q4 = 24.3bn shares (7:1 split) | **reason-coded, not flagged** | `split_basis_mismatch` (a `dc_code`, **not** a check — v2's table mislabels it) |
| VRT 2018-2020 zero revenue | **`accepted`** with the SPAC-shell evidence | `impossible_value`, `level_outlier` |
| DTE capex dimensioned to `dte:DTEElectricMember` | **`critical` if it ever resolves** | `dimensional_scope` |

**Also due**:
- [ ] The **new-field acceptance procedure**, documented in the runbook, with `status: probation` added
      to `FieldSpec` and asserted by a test (decision 44).
- [ ] **Item 21** — `src/validate/` exists again; fix AGENTS.md's code-map line by **replacing** it
      (the file is capped at 70 lines).
- [ ] The **deferral statements**, each with its revisit trigger, written into the phase report rather
      than left implied:
      - **Tier 0 (DQC/Arelle)**: adopt if Tier 3's checks *converge* (`leaf_vs_total` and
        `cross_vintage` stop disagreeing) while Phase 9 or the full-universe run still surfaces wrong
        values no check explains. Cheapest first step is a **sampled offline audit** (~300 filings
        stratified by regime × era), not a per-filing tier.
      - **Tier 4 (aggregators)**: adopt if Phase 9 finds wrong values Tiers 1-3 cannot explain, **on a
        field where the aggregator's basis is known to match ours.**
- [ ] `configs/fundamentals/fundamentals_rosters.json` gains a **`random_cold`** roster for §9.4 — 26
      tickers drawn from the ~448 never-swept names, **seed recorded in the file**.

**Effort**: 1 day.

---

### Phase 5b-stats — the residual statistical checks ⬜ (after Phase 9's first in-sample report)

*(`trend_break`, `level_outlier` and `scale` moved to 5b-core by decision 61.)*

- [ ] `frozen_series` — exactly-repeated consecutive TTM values. 6.2% → 0.33%; the 5 survivors are genuine
      (NEE/USB three-significant-digit rounding, VRT's correct zeros).
- [ ] `periodicity` — a field with only annual facts for a ticker. **AFL/CSCO `depAmort` is CORRECT** —
      read `by_ticker_periodicity`, or the check fires on 8 correct tickers (item 18 for `longTermDebt`).
      Consumes `series_shape`'s `sparse` classification.
- [ ] `sign_convention` + the generated `fundamentals_concept_balance.json` (decision 50). Known
      population: 7 rows across both rosters plus 8 from 3c.5. **Abstains, visibly, on any concept
      absent from the map.**
- [ ] **Item 20** — the `constants.py` pass as one proposed diff, now that the validator has decided
      what it re-adopts.

**Effort**: 1.5 days.

---

## Testing Strategy

- **Synthetic, planted** — one violation per check, fires exactly once, nothing else does. The bar.
- **Real-data corpus** — the acceptance table above, as a test, growing with every `fixed` outcome.
- **Contract tests** — `CHECK_REGISTRY` completeness; `fundamentals_check.json` schema; `finding_id`
  stability; the baseline ratchet.
- **No DB, no CLI in unit tests** — the validator is instantiated against a frame.
- **Every test prints its sanity-check conclusion** (AGENTS.md hard rule).

## Risk Mitigation

1. **A check over-fires and drowns the queue.** → `expected_fire_rate_ceiling` on every check, enforced
   in the calibration report. DQC_0118's own documentation is the warning. `series_shape`, `peer_ratio`
   and `trend_break` are the three highest-risk here and get their own calibration rows.
2. **`trend_break` fires on seasonality** (retail Q4, KR's 16-week Q1, weather-driven utility quarters) —
   an accepted cost of decision 59. → measure the fire rate **by regime** in calibration; the named
   remedy is switching the reference to the same fiscal quarter of the prior 3 years, which
   `fiscal_quarter` already makes a one-line change.
3. **`level_outlier`'s kernel fix breaks another caller.** `src/utils/outliers.py` has callers outside
   the validator. → find and migrate them in the same change; the 10×-growth synthetic test is the guard.
4. **The accept register becomes a suppression list.** → `outcome` requires evidence; `fixed` requires a
   commit *and* a regression test; the register reports **stale entries** whose check no longer fires.
5. **A guard nulls a correct row** (the 745-row lesson). → four impossible-only rules, `rejected_value`
   recorded, and a named test that HCA-shaped negative equity survives.
6. **`peer_ratio` imports the peer group's own errors.** → abstain below 5 peers; `severity: high`, never
   `critical`; and it is corroboration for `basis_step`, not a standalone verdict.
7. **Baselines fitted to a partial table.** → decision 55 sequences calibration after the backfill; every
   baseline entry carries `n` and its substrate.
8. **The validator re-reads the tables per check** (Phase 10's named risk). → load once, projected, pass
   the frame down the tiers; asserted by a test that counts `store.load` calls.

## Success Criteria

- [ ] `validate fundamentals --roster in_sample` runs end to end and writes `fundamentals_check`.
- [ ] Every check plantable and planted; each fires exactly once on its own violation.
- [ ] Fire-rate table printed; no check above its declared ceiling without a named mechanism.
- [ ] Every baseline re-measured and committed to the ratchet with `n` + substrate.
- [ ] The acceptance corpus passes: every historical defect either silent or flagged, as specified —
      **including the five classes only `series_shape`, `peer_ratio` and `dimensional_scope` cover.**
- [ ] 0 findings at severity ≥ `high` on AAPL beyond the named ones; 0 hard-guard nulls on AAPL.
- [ ] `validate fundamentals --field X` produces a per-field acceptance sheet.
- [ ] **Agent B exercised on all three fix paths**: a `build_history` fix verified via
      `--rebuild-history`; a resolution-layer fix verified via `fundamentals --rebuild -t X`; and a
      `config_proposed` fix that correctly stays OPEN until approval.
- [ ] **No finding reaches phase closure with `outcome: fixed` and `regression_swept: false`** — the
      batched full-roster sweep ran and attributed every new finding it surfaced.
- [ ] Both agents committed and exercised on at least one real finding, end to end, including the
      `fixed` path with its regression test.
- [ ] Three scripts deleted, their runbook lines replaced by the CLI.
- [ ] **`grep -rn "outliers\|tiingo_comparison\|yahoo_comparison" src/utils/` returns nothing** — all
      validation code lives under `src/validate/`, and `scripts/dod/data_profile.py` still runs.
- [ ] **`src/validate/README.md` exists and covers all six required sections**, including
      "when it does not work". A reader who has never seen this repo can run the tool from it alone.
- [ ] AGENTS.md code map repaired by replacement.

## Estimated Effort

| | |
|---|---|
| 5b-core.1 foundation + E-4 absence evidence | 2.5 d |
| 5b-core.2 Tier 1 | 1.5 d |
| 5b-core.3 Tier 3 + basis_step / tag_switch_break / series_shape / peer_ratio / trend_break / level_outlier / scale | 3 d |
| 5b-core.4 calibration + ratchet | 1 d |
| 5b-core.5 acceptance corpus + closure | 1 d |
| **5b-core total** | **9 d** |
| 5b-stats | 1 d |

v2 estimated 4-5 days for everything. That was low: the phase absorbs three committed instruments, adds
a lifecycle, a per-field acceptance procedure, two agent definitions, a baseline ratchet, **four new
checks covering six archive defect classes, and a fix to the outlier kernel it claimed to reuse** —
none of which existed as a design.

## Notes for Implementation

- **`configs/` is a risk zone** — propose every diff. `fundamentals_check.json`,
  `fundamentals_baselines.json`, `fundamentals_concept_balance.json` and the `random_cold` roster are
  four separate approvals; batch them.
- **`sql/schema.sql` is documentation on a live volume.** A DDL change must be applied deliberately via
  `scripts/recreate_fundamentals_tables.py` (`--dry-run`, then `--yes`). This is how an all-None column
  became TEXT and every later ticker's number was stored as `'1997000000.0'`.
- **Postgres DATE columns return as `datetime.date`**, never `Timestamp`. An unguarded comparison
  reports every stored row as drifted.
- **Never kill a running backfill by image name.** Kill by PID only. The 54-ticker backfill is DONE
  (2026-08-24 16:43), but a Phase-9 widening or an E-4 re-fetch will run for hours.
- **`ALL_CODES` is 19 and `IS_QUALIFIER` is 6** as shipped — not the 18/5 in earlier drafts.
- The fundamentals config JSONs are **hand-formatted**; a `json.dumps` round-trip reformats the whole file.
- `companyfacts` can prove a concept PRESENT and can **never** prove one ABSENT. Every coverage claim
  must be measured off `filing.xbrl()`.

---

## IMPLEMENTATION LOG — 2026-08-24

**5b-core.1 ✅ / 5b-core.2 ✅ / 5b-core.3 ✅ (code) — 5b-core.4 🔄 in progress.**

### What landed

| item | where |
|---|---|
| `src/validate/` recreated, cross-domain | `__init__.py`, `README.md`, `cli.py`, `outliers.py`, `external/`, `fundamentals/` |
| the code moves (decision 62) | `git mv` of `outliers.py`, `tiingo_comparison.py`, `yahoo_comparison.py`; `scripts/dod/data_profile.py` re-pointed; tests moved to `tests/validate/external/` |
| `level_outlier` kernel fix (decision 60) | `outliers.log_change` + `_score_changes`; both passes score a LOG CHANGE |
| `Tables.fundamentals_check` + `rejected_value` | `src/data_store/schema.py`, `sql/schema.sql`, **applied to the live volume 2026-08-24** |
| the hard guards (decision 46) | `build_history.HARD_GUARDS` + `_hard_guard`, applied before the write |
| `CHECK_REGISTRY` | **35 checks**: tier 1 = 19, tier 2 = 8, tier 3 = 8 |
| the finding packet + `finding_id` | `fundamentals/finding.py` |
| the accept register | `fundamentals/check_register.py` + `configs/fundamentals/fundamentals_check.json` (empty, by design) |
| the two agents (decision 54) | `.claude/agents/fundamentals-validate.md`, `.claude/agents/fundamentals-triage.md` |
| script absorption | `verify_fundamentals_history.py`, `audit_absence_evidence.py`, `measure_total_liabilities_legs.py` DELETED; runbook re-pointed at the CLI |
| doc repair (item 21) | `AGENTS.md` code map replaced, `docs/architecture.md`, `docs/data_schema.md`, `docs/runbook.md` |

### Tests — all green

- `tests/validate/test_outliers.py` — 5, including decision 60's two named tests
- `tests/validate/fundamentals/test_planted_violations.py` — 30, one planted violation per check
- `tests/data_extract/test_hard_guards.py` — 8, including the HCA-shaped negative that must SURVIVE
- `tests/data_extract/` fundamentals suite — 23, unchanged by the guards

### DEVIATIONS from the plan, each with its reason

1. **`coverage_field` fires per (ticker, field), not per CELL.** Forced by arithmetic: 71,857 null
   cells on a 54-ticker roster would be 71,857 findings — the DQC_0118 drowning this design exists
   to prevent. The null count and rate are in `detail`; the SHAPE of a gap is `series_shape`'s job,
   which is the division of labour the plan's own critique of per-cell firing argues for.
2. **The YoY pass in `outliers.py` was also converted to a log change.** Decision 60 says only
   "keep the YoY `diff(4)` check". Its kernel had the SAME defect — a 4-period difference on an
   exponentially growing series grows exponentially — so fixing one and not the other would have
   left half the bug in place. The check is kept; its kernel is fixed.
3. **`PEER_RATIO_Z` is 3.5, not the 5.0 first written.** MEASURED while writing the tests: the
   modified Z of a lone outlier among k identical peers is bounded by `0.6745 · k`, so at the
   5-peer minimum the maximum achievable score is **3.37** and a 5.0 threshold made the check
   literally unable to fire at its own declared minimum. A threshold that cannot be reached is
   not strict, it is silent.
4. **`mad_center_scale` gained a relative dispersion floor** (`_DISPERSION_REL_TOL = 1e-12`).
   `mad > 0` accepted floating-point dust as scale: a smooth series' log changes are equal to
   ~1e-17, and a planted 3× spike scored **z = 3.6e15**, decided by rounding error.
5. **`series_shape` runs over `instant` AND `quarterly`.** Found by a failing test: running only
   the quarterly frame made every instant field — `totalDebt`, `cash`, `goodwill`, the share
   counts, roughly a third of the table — invisible to the one check that detects a shape.
6. **The `external/` adapters were MOVED but not cut back to fetch-only.** Tier 4 is deferred, so
   there is no validator to move their ranking/bucketing/verdict logic into; the cut-back would
   delete the only implementation a future Tier 4 would have to re-derive. Recorded as pending on
   Tier 4's adoption, in `src/validate/external/__init__.py`.
7. **`scripts/recreate_fundamentals_tables.py` was NOT used.** Both DDL changes are strictly
   additive (`CREATE TABLE`, `ADD COLUMN`), so the drop-and-recreate path would have destroyed the
   4 h 39 m rebuild for no reason. Applied as two additive statements; 76,004 reason-code rows and
   3,267 history rows verified intact afterwards.
8. **`measure_total_liabilities_legs.py` was deleted as instructed, but its MEASUREMENT is not
   reproduced.** `cross_identity` encodes the CONSTRAINT (never leg-sum `totalLiabilities`); the
   script read the calculation linkbase over the network, which no check does. Noted in the
   runbook: recover it from git history if the leg-set question is reopened.

### Still open in 5b-core

- **E-4 absence evidence** (`unmatched_concepts` on the unresolved stub) — NOT started. It rides
  Phase 9's full-universe fetch by decision 66, and `coverage_field` / `series_shape` currently run
  on the codes they have.
- **5b-core.4** — the fire-rate table and `fundamentals_baselines.json` (in progress).
- **5b-core.5** — the acceptance corpus, the `probation` FieldSpec status, the `random_cold` roster.
- **5b-stats** — `frozen_series`, `periodicity`, `sign_convention`, the concept-balance map, item 20.

### 5b-core.4 — FIRST CALIBRATION RUN, 2026-08-24 20:30

`validate fundamentals --roster all` over the live tables. **54 tickers, 12,501 findings**
(critical 293 / high 8,051 / medium 2,593 / info 1,564). Report:
`reports/2026-08-24/validate_calibration.md`. Findings are in `fundamentals_check`.

**The seven zero-ceiling structural checks all return 0**, which independently confirms the
substrate table at the top of this plan rather than restating it:

| check | examined | findings |
|---|---|---|
| `grain` | 3,267 | **0** |
| `pit_leak` | 3,267 | **0** |
| `unexplained_null` | 196,020 cells | **0** |
| `dimensional_scope` | 252,001 facts | **0** |
| `code_vocabulary` | 76,004 codes | **0** |
| `column_contract` | 69 | **0** |
| `coverage_universe` | 54 | **0** |

**Tier 3 is entirely under ceiling** — `holdout_q4` 1.95%, `q4_footing` 1.72%,
`annual_footing` 1.58%, `cross_vintage` 1.19%, `leaf_vs_total` 0.01%, `duplicate_fact` 0,
`derived_vs_asreported` 0. The provenance-change checks are too: `basis_step` 0.19%,
`tag_switch_break` 0.28%.

#### 8 checks breached their own ceiling — challenged, per the rule

| breach | verdict | action |
|---|---|---|
| `register_cost` 825.9% | **denominator bug.** Divides 446 findings by 54 TICKERS instead of tickers x fields | fix the denominator |
| `cross_identity` 8.97% (293, all `critical`) | **two real check defects**, below | fix both |
| `series_shape` 29.1% | 1,045 of 1,632 are `info` (benign gap codes). The ceiling was set against TOTAL findings, not the queue | ceiling should be measured on queue-severity findings |
| `coverage_field` 20.2% | 567 of 656 are `medium` = MIXED -- which section E-2 of this very plan calls *the honest majority* (31 of 48 industrial fields). **The 10% ceiling contradicted the plan's own measurement** | raise the ceiling with E-2 as the evidence |
| `coverage_quarters` 7.4% | 4 findings on a 54-ticker denominator; 1 finding = 1.85%, so a 2% ceiling cannot survive two | ceiling unrealistic at this denominator |
| `level_outlier` 5.3% / `trend_break` 5.2% / `scale` 1.5% | marginally over; the statistical family | needs the BY-REGIME split (decision 59's seasonality cost) before retuning |

#### The two `cross_identity` defects, measured

**1. The balance-sheet leg ignores NCI.** All 102 failures are NCI-heavy filers at ~1.5-2.7%:
`UNH 42, AMT 18, PGR 12, EQIX 12, SPG 9, NVDA 2` (+ VRT 7, the pre-merger shell at ~96%,
which is the near-zero-denominator case). `stockholdersEquity` resolves to the **ex-NCI**
element for these filers, so the identity needs `+ minorityInterest`. This plan flagged the
mechanism under `derived_identity_nci_assumed_zero` and the check did not carry it.

**2. `grossProfit == totalRevenue - costOfRevenue` IS NOT A UNIVERSAL IDENTITY.** All 191
failures are `industrial`, at 15-74%: `TMO 50 (33.6%), EQIX 48 (15.0%), CVS 31 (73.9%),
CAT 31 (39.3%), COST 28 (39.7%)`. `grossProfit` resolves from the filer's own
`us-gaap:GrossProfit` tag, which each filer computes on ITS OWN COGS basis -- CVS excludes
benefit costs, COST nets membership fees, CAT excludes certain items. Both numbers are right;
the PREMISE is wrong, and `critical` (reserved for *provably* wrong) is the wrong severity for
it. **This is DQC_0118 landing on our own code on the first real run**, which is the outcome
the ceiling mechanism exists to produce.

### Full test suite, 2026-08-24: 867 passed, 13 failed, 20 skipped (1h 20m)

**None of the 13 is in code this phase touched.** Triaged, not assumed:

| n | failures | cause |
|---|---|---|
| 10 | `tests/data_aggregate/*` | section E-5's named hand-off: `test_composites_config::real_panel` + the 9 `data_aggregate` failures are **Phase 6's**, explicitly not 5b's |
| 2 | `test_entity_scope::test_maa_shares_outstanding_is_the_parents`, `test_linkbase_resolution::test_apa_revenue_...` | `fetch_fundamentals_sec._materialise` returns a **tuple** `(resolved, refused)` since Phase 4; both tests still call `periods.values()`. Pre-existing API drift on this branch |
| 1 | `test_def14a_llm::test_llm_extractor_real_apple` | DNS: `Failed to resolve 'data.sec.gov'` -- the machine was offline during the run |

Everything 5b touched is green: `tests/validate/` 49, `test_hard_guards.py` 8,
`test_build_history.py` + amendment + PIT 23, `tests/dod` + `tests/data_store` 97.
