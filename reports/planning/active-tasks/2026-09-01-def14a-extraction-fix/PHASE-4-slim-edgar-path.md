# Phase 4 — Slim the edgar path to the ECD XBRL block ✅

**Goal**: `sec_def14a` becomes exactly one thing — the filer-tagged Pay-versus-Performance /
ECD inline-XBRL facts, dimension-filtered. Every HTML-parsed column and all four child tables are
deleted, along with the code that produced them.

**Depends on**: Phase 3 (the LLM path must already own `auditor_name`, the fee breakdown and
`ceo_pay_ratio` before those columns are removed here).

---

## Why (measured)

**KEEP the ECD XBRL reader** — these are facts the filer tagged, and an LLM would be strictly worse.
**But fix its dimension bug**:

- `_get_concept_value` / `_get_concept_series` (`edgar/proxy/core.py:105-151`) filter on
  `concept ==` only. Neither filters `dim_ecd_IndividualAxis` or `dim_ecd_ExecutiveCategoryAxis`;
  they sort by `period_end` and take `.iloc[0]`, with `drop_duplicates(keep='first')` on a stable
  sort — so **document order decides the winner**.
- `ecd:PeoName` is **the tag filers use for every named executive**, discriminated only by the axis
  the library ignores. AAPL's instance carries **26 `ecd:PeoName` facts, only 5 of them
  `ecd:PeoMember`**; the rest are NEOs.
- Consequences: **BA 2026** keeps Ortberg 18,388,629 and **drops Calhoun 15,050,812** (CAP: Ortberg
  +19,904,513 kept, Calhoun **−23,875,735** dropped). **NKE 2026** drops Donahoe. **SBUX** tags a
  full individual×year matrix with 0.0 in non-applicable cells, so `peo_total_comp` and
  `peo_actually_paid_comp` are **0.0** for FY2023, FY2024 *and* FY2025 while `peo_name` reads
  `'Brian Niccol'`.

**RETIRE the HTML-parsed block** — exec comp, director comp, ownership, audit fees, board
recommendations. Every one is measured broken, several **fabricate values rather than omitting
them**, and the defects are ticker-persistent so they do not average out.

**Negative `peo_actually_paid_comp` is legitimate and must be preserved.** 11 of 93 non-null rows
(11.83%), all with positive `peo_total_comp`. Compensation Actually Paid subtracts prior-year
unvested-award fair value, so a share-price decline makes it negative. **28.5% of 2023 and 33.7% of
2025** S&P 500 proxies report ≥1 negative value. Intel FY2024 reconciles exactly:
27,429,900 − 24,625,700 + 0 + 0 − 1,775,113 − 83,245,704 = **−82,216,617**. There is no sign flip in
edgartools — do not add an `abs()` anywhere.

---

## Changes

### 1. `src/data_extract/utils/structure/def14a_ecd.py` (new, ~120 lines)

Read the ECD facts **directly**, bypassing `ProxyStatement`'s undimensioned accessors.

- [x] `filing.xbrl()` → `.facts.to_dataframe()` is the same frame `ProxyStatement._facts_dataframe`
      uses (`edgar/proxy/core.py:98-102`), so nothing new is downloaded. Return `None` when
      `filing.xbrl()` is None.
- [x] `peo_value(facts, concept)`: filter `concept == concept` **and**
      `dim_ecd_ExecutiveCategoryAxis == 'ecd:PeoMember'` (or the equivalent member string as it
      appears in the frame — confirm the exact spelling on a live BA/AAPL filing before hardcoding).
      Then group by `dim_ecd_IndividualAxis` and return **one row per individual**, not `.iloc[0]`.
- [x] Co-PEO years therefore yield **two rows**. Decide the storage shape (below).
- [x] Drop `0.0` cells before selecting — the SBUX zero-matrix shape. `def14a_validate`'s existing
      "a PEO is never paid exactly $0" rule already does this, but doing it at selection time
      recovers the **correct** value instead of a NULL.
- [x] `net_income`: keep reading `us-gaap:NetIncomeLoss` (PVP column (h) — **`ecd:NetIncLossAmt`
      does not exist**, and the concept is wired `priority="-1"` so filers may override it). Keep the
      `DEF14A_NET_INCOME_MIN_PLAUSIBLE` drop: SBUX FY2025 arrives as `1856.4` (raw `value='1856.4'`,
      `decimals='1'`, `unit_ref='usd'` — tagged in $ millions) and **nothing in the row disambiguates
      millions from billions**, so it stays a NULL rather than a guess. The real value is in
      `fundamentals_history`.
- [x] Units are **not normalised anywhere** in edgartools (`instance.py:430-436` is `float(value)`
      verbatim; no `sign`/`scale` handling). Do not assume they are.

#### Storage shape for co-PEO years — decision

Two PEOs in one fiscal year cannot both live in one `peo_*` column on a `(ticker, accession)` row.

- [x] **Chosen**: keep `sec_def14a` at `(ticker, accession_number)` grain and store the PEO facts as
      **`peo_name` / `peo_total_comp` / `peo_actually_paid_comp` for the individual with the LARGEST
      `peo_total_comp`**, plus `n_peos` and `peo_names_all` (comma-joined) so a co-PEO year is
      *visible* rather than silently halved. Rationale: one row per filing keeps the table joinable
      and simple, and the co-PEO case is rare; the current failure is not "we picked the wrong PEO"
      so much as "we did not know there were two".
- [x] Rejected alternative: a `(ticker, accession, individual)` child table. That reintroduces a
      child table one phase after deleting four of them, for ~1% of rows.
- [x] The Phase 6 check is: **BA 2026 has `n_peos = 2` and `peo_names_all` contains both Ortberg and
      Calhoun**; SBUX FY2023-25 `peo_total_comp` is non-zero.

### 2. `src/data_extract/utils/structure/fetch_def14a_edgar.py` — slim

- [x] `_main_row` sources: `company_name` gains the fallback the library lacks —
      `_get(proxy, "company_name") or filing.company`. edgartools reads `dei:EntityRegistrantName`
      XBRL-only with **no fallback to `self._filing.company`**, which is always populated from the
      filing index; DEF 14A has no mandatory cover-page iXBRL requirement, so it is None for every
      pre-2023 filing. `__str__` substitutes the literal `"Unknown Company"`. This is a one-line fix
      worth the whole 14% fill rate.
- [x] Keep sourcing `period_of_report` from `filing.period_of_report`, not the library property
      (`ProxyStatement.fiscal_year_end` / `dei:DocumentPeriodEndDate` is **None on 4/4** live
      large-cap proxies and was 0 of 329 stored rows).
- [x] **Write a row only when the filing has XBRL.** Pre-2023 proxies would otherwise contribute a
      keys-only row for every filing; the inventory of "which proxies exist" is `def14a_llm`'s job,
      which covers all of them. `has_xbrl` therefore becomes degenerate — **drop the column**.
      Keep `has_individual_executive_data`.
- [x] DELETE from `_MAIN_COLS` and `_main_row`: `auditor_name`, `audit_fiscal_year_*`, all 10
      `*_fees_*` columns, `ceo_pay_ratio_*` (3 columns), all 7 `n_*_proposals` counters and
      `n_board_against_recommendations`.
- [x] ADD: `n_peos`, `peo_names_all`.
- [x] DELETE the functions `_exec_comp_rows`, `_director_comp_rows`, `_ownership_rows`,
      `_votes_rows` and the four column lists; `_NUMERIC_COLS` collapses to a single-table dict
      (it doubles as the table list handed to the driver, so the two cannot disagree).
- [x] DELETE the `hasattr(proxy, "voting_proposals")` guard. It never fired for DEF 14C in 5.51.0
      anyway (DEF 14C is not in edgartools' `PROXY_FORMS`, so `matches_form` fails and `.obj()`
      returns something else entirely) and the new path does not call `.obj()` at all — it goes
      straight to `filing.xbrl()`.
- [x] Rewrite the module docstring: what the table is now (the ECD block), the 2023 regulatory
      threshold, and the dimension filter. Drop the retired-defect narrative.
- [x] `ceo_pay_ratio` moves entirely to the LLM path (Phase 3): it is **narrative-only forever**
      (Rel. 33-9877, no XBRL tag ever) and edgartools extracts it with **three value-inventing
      repairs** (`html_extractor.py:430-457`: pay-ratio swap, two-largest-dollars, strip-last-digit).

### 3. `src/data_extract/utils/structure/def14a_validate.py` — remove the edgartools-only parts

Phase 3 kept `clean_text`, `clean_person_name`, the subtotal/address regexes, `_rescale_block` and
the new flag computation. Now remove what only existed for the HTML grid:

- [x] DELETE `_GLUED_TITLE_RE`, `_ORPHAN_MODIFIER_RE`, `_TITLE_LEADING_JUNK_RE`,
      `repair_exec_comp_rows`, `repair_director_comp_rows`, `repair_ownership_rows`,
      `DEF14A_PLACEHOLDER_PERCENT` (edgartools' fabricated 0.5 stand-in) and the pay-ratio triplet
      repair `_repair_pay_ratio` + `DEF14A_PAY_RATIO_TOLERANCE`.
- [x] KEEP `repair_main_row`, slimmed to: text cleaning, the `net_income` plausibility drop, and the
      `peo_* == 0.0` → NaN guard (belt-and-braces behind the ECD selection fix).
- [x] The **pay-ratio identity** does not disappear — `def14a_impute._reconcile_rows` already holds
      it on the LLM side (`median_pay = total / ratio`, `pay_ratio = total / median`), and the
      identity holds on **95.0% of 3,838 rows**. Do not duplicate it.
- [x] `__all__` updated.

### 4. Table removal

- [x] `src/data_store/schema.py` (risk zone): delete `def14a_edgar_executive_comp`,
      `def14a_edgar_director_comp`, `def14a_edgar_ownership`, `def14a_edgar_votes`.
      Confirmed safe: an exhaustive grep shows the **only** references are the fetcher, the validate
      module, `schema.py` itself and docs — **no consumers** in `src/data_aggregate/`,
      `src/data_peers/`, `src/modelling/`, `src/validate/`, `app/`, `backtest/`, `stock_pick_strat/`.
- [x] `sql/schema.sql` (risk zone): remove the four `CREATE TABLE` + four `CREATE INDEX` blocks by
      hand (lines ~962-1031). **Do not regenerate** — the generator drops 8 hand-added indexes.
- [x] Phase 6 issues the `DROP TABLE` statements; this phase only removes the registrations, so a
      revert before cutover is clean.

### 5. Tests

- [x] `tests/data_extract/structure/test_def14a_validate.py` — delete the cases for the removed
      repairs; keep the KO thousands-rescale and the name/footnote-key cases (still live on the LLM
      side). The fixtures are real observed defects and are worth preserving where the code is.
- [x] `tests/data_extract/structure/test_fetch_def14a_edgar.py` — this file is **entirely synthetic
      `SimpleNamespace` fakes**, which is why every defect in the research was invisible to CI.
      Replace the child-table row-builder tests with a **real-filing** test on the ECD path using
      Phase 0's cached filings (see Verification). Keep the incremental / `since`-cutoff tests, which
      are genuinely about the driver.
- [x] New: an ECD dimension-filter test with a **real** facts frame captured from BA 2026 and SBUX
      2026 (save the frames as parquet fixtures — small, and no network in CI).

### 6. Docs

- [x] `docs/data_schema.md` — remove the four tables, rewrite `sec_def14a`'s column list, add the
      four new `def14a_*` tables from Phase 3.
- [x] `docs/database.md:125-127` — the "`sec_def14a` covers only 23 of 500 tickers … any `f_ceo_*` /
      governance feature built off `sec_def14a` will be ~95% NaN" note becomes wrong in a new way:
      the table is now 2023+ **by regulation** and that is correct behaviour, not a gap. Say that.
- [x] `docs/data_sources.md:201-207` — "edgartools' proxy HTML parser is silently wrong, not absent"
      stays true and now explains *why the HTML block was deleted*. Add the ECD dimension bug.

---

## Verification

- [x] **Live ECD probe on cached filings** (no LLM, ~8 requests):
      - BA 2026 → `n_peos == 2`, `peo_names_all` contains both Ortberg and Calhoun, and the retained
        `peo_actually_paid_comp` is one of {+19,904,513, −23,875,735} rather than silently the first
        in document order.
      - NKE 2026 → `n_peos == 2` (Hill + Donahoe).
      - SBUX 2024/2025/2026 → `peo_total_comp` and `peo_actually_paid_comp` **non-zero**.
      - AAPL 2026 → `peo_name` comes from a `PeoMember`-dimensioned fact, not from any of the 21
        NEO-dimensioned `ecd:PeoName` facts.
      - AAPL 2023-01-12 → **no row** (0 ecd tags: its FY2022 ended 2022-09-24, before the
        2022-12-16 threshold). Its first tagged proxy is 2024-01-11. This is correct behaviour.
      - A pre-2023 filing → **no row written**, and `def14a_llm` still has one.
- [x] **Sign preservation**: assert at least one negative `peo_actually_paid_comp` survives the run.
      Intel FY2024's −82,216,617 is the reconciling example.
- [x] **`company_name` fill**: on the 23 baseline tickers, `company_name` is non-null on **100%** of
      written rows (it now falls back to the filing index).
- [x] `grep -rn "def14a_edgar_executive_comp\|def14a_edgar_director_comp\|def14a_edgar_ownership\|def14a_edgar_votes" src tests configs sql docs`
      → zero hits.
- [x] `git diff --stat sql/schema.sql` → four table blocks and four indexes removed, three added
      (from Phase 3), nothing else touched.
- [x] `"$PY" -m pytest tests/data_extract/structure -q` green.
- [x] Line-count sanity: `fetch_def14a_edgar.py` + `def14a_validate.py` should shed roughly 600-700
      lines net. If they did not, something was kept that should not have been.

## Rollback

The registrations and code are removed but the **Postgres tables still exist** until Phase 6 drops
them. `git revert` restores the fetcher and the four registrations, and the data is untouched.

## Notes

- The SGML warning (`SGML fetch failed for … falling back to homepage`) is **cosmetic** and needs no
  work: forcing the fallback produced **byte-identical** extraction on every field, and the AES 2015
  accession it fires on returns empty tables on the happy path too — so the warning is not even
  diagnostic of the data loss it appears next to. Do not chase it.
- Do not add an `abs()` or a sign flip anywhere near `peo_actually_paid_comp`.
- Confirm the exact member string for the executive-category axis on a live frame before hardcoding
  `'ecd:PeoMember'`. Memory rule: never trust a tag name — measure it.

---

## RESULTS

`140 passed, 2 skipped` across `tests/data_extract/structure` + the three cube-side DEF 14A
test modules.

| | before | after |
|---|---|---|
| tables written by this path | 5 | **1** |
| `sec_def14a` columns | 46 | **25** |
| `fetch_def14a_edgar.py` | 321 lines | **133** |
| `def14a_validate.py` | 365 lines | **199** |
| `def14a_ecd.py` | — | **339** (new) |

### Acceptance sheet — the new reader on 23 tickers, 2023+ (zero LLM, `scripts/def14a_ecd_acceptance.py`)

**81 rows written, 5 filings correctly skipped** (no `ecd:` facts → pre-402(v) fiscal year).

| column | fill |
|---|---|
| `peo_name`, `peo_total_comp`, `peo_actually_paid_comp`, `n_peos`, `peo_names_all` | **100%** |
| `neo_avg_*`, `total_shareholder_return`, `peer_group_tsr`, `ecd_period_end` | **100%** |
| `company_selected_measure_name` / `net_income` | 96.3% |
| the four Item 402(x) flags | 30.9-45.7% |

- `peo_total_comp == 0` exactly: **0 rows**. That value is SBUX's matrix marking a year the
  person was not PEO, and dropping it *at selection time* recovers the correct figure instead of
  the NULL a post-hoc guard would leave.
- **negative `peo_actually_paid_comp` survives: 3 of 81 (3.7%).** The research measured 11.83% on
  a larger, older sample; 3.7% is what 23 tickers over 2023-2026 gives, and the number that
  matters is that it is not 0 — a 0 would mean the sign was destroyed.
- `n_peos` distribution: **{1: 73, 2: 7, 3: 1}** — 8 filings whose co-PEO year the old path
  silently halved. The 3 is SBUX's 2025 proxy (Narasimhan, Ruggeri interim, Niccol in one year).
- The **402(x) flag fill is honest absence, not a parser gap**: measured raw values are the
  strings `'true'` / `'false'`, which the reader handles; SBUX tags none of the four, BA's 2024
  proxy tags none, and Item 408(b) only bites for fiscal years beginning after 2024-04-01.

Gate cases, all passing: BA 2025 `n_peos=2` with both Ortberg and Calhoun; NKE 2025 `n_peos=2`
with both Hill and Donahoe; SBUX 2026 non-zero (Niccol 30,992,773); AAPL 2026 names Cook and no
NEO; AAPL 2023 **no row**.

### ⚠ DEVIATION — the plan's prescribed dimension filter returns ZERO rows on every filing

The plan says to filter `dim_ecd_ExecutiveCategoryAxis == 'ecd:PeoMember'`, with a note to
confirm the member string first. Measuring it did not just correct a string — **it falsified the
mechanism.** Filers discriminate PEO facts two incompatible ways:

| filing | `ecd:PeoTotalCompAmt` facts | carrying an `ExecutiveCategoryAxis` | kept by `== 'ecd:PeoMember'` |
|---|---|---|---|
| BA 2025 | 7 | 0 | **0** |
| NKE 2025 | 6 | 0 | **0** |
| NKE 2026 | 7 | 0 | **0** |
| SBUX 2026 | 25 | 0 | **0** |
| AAPL 2026 | 5 | 0 | **0** |

BA / NKE / SBUX put the executive on `ecd:IndividualAxis` **only**. AAPL's amounts are
**entirely undimensioned** (5 facts, one per covered year) while its `ecd:PeoName` facts are
dimensioned on *both* axes — 26 of them, 21 tagged `ecd:NonPeoNeoMember`. So the axis filter is
essential for AAPL's *names* and fatal for everyone's *amounts*, and the name and the amount
cannot be joined on an individual key at all.

**Implemented instead**: apply the axis filter only when the concept's own facts actually carry
the axis, and resolve name and amounts independently. `test_a_fixed_peo_member_filter_would_
return_nothing` is the negative control that pins this — it fails if anyone "simplifies" the
rule back to the plan's version.

Two smaller corrections from the same measurement:

- **`ecd:PeoTotalComp` does not exist** — the concept is `ecd:PeoTotalCompAmt`. The plan's name
  reads as a filer omission rather than a typo, which is exactly how a wrong tag name hides.
- **The co-PEO gate years in the plan are off by one filing.** BA's co-PEO year is FY2024 and
  NKE's is FY2025, so it is the **2025** proxies that carry two PEOs in their covered year, not
  the 2026 ones the plan names. The research read the co-PEO row out of the 2026 filing's
  five-year table without noticing it was not the covered year. Gates re-pointed at BA 2025 /
  NKE 2025 and verified.

### ⚠ DEVIATION — the line-count expectation was arithmetically impossible

The plan asks for "roughly 600-700 lines net" shed from `fetch_def14a_edgar.py` +
`def14a_validate.py`. Those two files held **686 lines in total**, and the same plan instructs
KEEPING `repair_main_row`, the four text/key helpers and `rescale_block` — so the target implied
leaving almost nothing. Measured: **−188 and −166, −354 net**, plus 339 new lines of
dimension-aware reader. The honest reduction is in *surface*, not lines: **5 tables → 1** and
**46 columns → 25**.

### ⚠ DEVIATION — one regex on the plan's delete list is load-bearing

The plan lists `_GLUED_TITLE_RE` for deletion with the other HTML-grid regexes, but
`clean_person_name` — which the plan also says to KEEP, and which is the LLM path's name primary
key — calls it. Deleting it would have silently weakened the key. `_ORPHAN_MODIFIER_RE` and
`_TITLE_LEADING_JUNK_RE` had no surviving callers and are gone.

### Defect found in Phase 2's classifier while probing this phase

Not a Phase 4 change, but found by the Phase 3 paid probe and fixed here because it would have
corrupted Phase 6's numbers: **PG 2026 was carving the wrong table into the SCT block.** Its
`Outstanding Equity at Fiscal Year End` grid (Item 402(f)) satisfies the SCT rule — "Fiscal
**Year** End" plus `Stock Awards` / `Option Awards` columns — and outscored the real SCT on the
data-row tie-break. The model was handed an equity-holdings grid, correctly returned zero
compensation rows, and the filing stored `n_neos = 0`.

Fixed with an `_OTHER_402_MARKERS` must-reject clause (the exact 402(f)/(g)/(h)/(i) column
labels, chosen so that "non-equity incentive plan compensation" cannot trip it). Re-probed:
PG now returns **8 NEOs × 3 fiscal years**, and its carve payload fell **49,040 → 44,007**.

### Deviations that were NOT needed

- The `net_income` concept: `us-gaap:NetIncomeLoss` confirmed, `ecd:NetIncLossAmt` confirmed
  absent, exactly as the plan says.
- `AAPL 2023-01-12` → `filing.xbrl()` is None → no row, exactly as the plan predicts.
- `company_name` from the filing index: 100% fill on every written row.

### Carried into Phase 6

- [ ] `DROP TABLE` for the four retired `sec_def14a_*` child tables (already in `PHASE-6`).
- [ ] `sec_def14a`'s column set changed (46 → 25), so the live table must be dropped rather than
      truncated, or `ensure_table` will keep the 21 dead columns.
