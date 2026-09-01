# Implementation Plan: DEF 14A extraction fix — one LLM path, a slim XBRL path, and vote tallies

**Date Created**: 2026-09-01
**Planning Phase**: 2 of 3 (FIC Workflow)
**Based on Research**: [2026-09-01-def14a-extraction-gaps.md](../../../research/financial-data/2026-09-01-def14a-extraction-gaps.md)
**Next Phase**: Implementation (`/implement`)

---

## Overview

The five `sec_def14a*` tables are broken for three independent reasons. Only one is a parsing
defect, and it is not fixable inside `edgartools`. The plan resolves the architecture down to
**two honest paths**:

1. **`sec_def14a`** — the ECD / Pay-versus-Performance **inline-XBRL** block only. Filer-tagged,
   machine-readable, 2023+ by regulation. Dimension-filtered (which edgartools does not do).
   Zero HTML parsing. Zero LLM cost.
2. **`def14a_llm`** + 4 new child tables — everything narrative or tabular, extracted by the
   existing `LLMExtractor` over a **table-anchored carve** that is both more accurate and
   *cheaper* than today's fixed-budget anchor carve.

Plus one genuinely new dataset: **`sec_8k_votes`**, the shareholder-vote tallies that already sit
unparsed in 6,657 stored `sec_8k` Item 5.07 narratives.

Everything edgartools parses out of **HTML** is deleted, because every instance measured in the
research was silently wrong rather than absent, and the defects are **ticker-persistent** (AMAT
wrong in 9 consecutive years, PG in 6, A in 2), so they survive cross-sectional ranking as a
fixed per-issuer bias instead of averaging out.

---

## Decisions taken (from the planning interview)

| # | Decision | Consequence |
|---|---|---|
| D1 | **`sec_def14a` slims to the ECD XBRL block.** Drop the 4 child tables and the whole HTML-parsed block. | ~700 lines deleted incl. most of `def14a_validate.py`. One table instead of five. |
| D2 | **All four LLM-side outputs are in scope**: exec-comp flatten, `auditor_name` + fee breakdown, director-comp table, per-holder ownership rows. | New tables off one LLM call per filing. Ownership rows are knowingly redundant with 13F / SC 13D-G / Forms 3-4-5 (see *Accepted redundancy*). |
| D3 | **8-K Item 5.07 vote tallies: LLM-only**, with a fabrication guard and amendment union. | New `sec_8k_votes`. No HTML re-fetch — `item_text` is already usable. |
| D4 | **Adopt table-anchoring (strategy B) for the 5 tabular targets**, keep the anchor carve (A) for director bios and corporate governance. | New `def14a_tables.py`. Payload drops to ~73% of today's while recall rises. |
| D5 | **Baseline via parquet snapshot + parquet output.** No DB writes during validation. | The live backfill runs undisturbed; the comparison is re-runnable offline. |
| D6 | **"Fix the fixable, drop the rest"** on the 5 untrusted fields. | DROP `n_technology_directors`, `pct_technology_directors`. KEEP+FIX `gender`, `avg_other_public_boards`, `majority_voting`, `poison_pill`. |
| D6b | **`gender` stays and gets a real deduction upgrade** — it carries alpha, so the answer is to make it right, not to delete it. | Adds a 4th child table `def14a_directors` (free — the 87,984 rows are already in `def14a_json`), a `gender_basis` provenance field, in-document evidence in the prompt, and a cross-filing consensus pass. Detail in Phase 3. |
| D7 | **Form scope unchanged** (`DEF 14A`, `DEF 14C`); fix the `_doc_url` bug only. | ~400 pre-2001 rows recovered. No DEFR14A, no 10-K/A SCT, no special-meeting detection. |
| D8 | **Extract-only.** The cube keeps reading `def14a_llm`. | Blast radius stays inside `src/data_extract` + `src/data_store/schema.py`. New features are a separate task. |
| D9 | **23 baseline tickers**: 13 named-defect + 10 random (fixed seed). | Proves both "fill went up" and "the known-wrong values are gone". |
| D10 | **Exec-comp reconciliation is a FLAG, not a filter**: `reconciles = 1` when `abs(total − Σ components) <= $10`, else 0. Values are kept either way. | Nothing is discarded; the failure rate becomes measurable. |
| D11 | **`gpt-5-mini` everywhere** (the `configs/configs.yml` value), and the dead `temperature` parameter is removed from `LLMExtractor`. | The research's accuracy numbers were `gpt-4o-mini`; Phase 0/6 re-measures on the real model. |

### Reuse, not rebuild (explicit)

| Need | Existing function — use it as-is |
|---|---|
| LLM structured extraction | [llm_extractor.py](../../../../src/data_extract/utils/common/llm_extractor.py) `LLMExtractor.extract(schema, text, instructions=)` |
| OpenAI embeddings | [openai_embeddings.py](../../../../src/utils/openai_embeddings.py) `embed_texts` — **not needed** by this plan (deterministic header signatures measured 96.8%); listed so nobody writes a second client |
| HTML → text | [edgar_extract.py:28](../../../../src/data_extract/utils/common/edgar_extract.py#L28) `html_to_text` |
| Filing discovery | [edgar_fillings.py:65](../../../../src/data_extract/utils/common/edgar_fillings.py#L65) `list_filings` |
| Rate-limited SEC GET | [sec_utils.py:43](../../../../src/data_extract/utils/common/sec_utils.py#L43) `sec_get` |
| Accession dedup / incremental | `sec_utils.existing_filings`, `run_manifest.manifest_window` / `record_run` |
| edgartools driver (8-K/13D pattern) | [edgar_driver.py](../../../../src/data_extract/utils/common/edgar_driver.py) `new_filings`, `run_edgar_fetch` |
| Fee-block unit rescale | [def14a_validate.py:157](../../../../src/data_extract/utils/structure/def14a_validate.py#L157) `_rescale_block` — survives the slimming, moves to the LLM side |

No new OpenAI client, no new rate limiter, no new filing lister.

---

## Director vote categories — VALIDATED 2026-09-01

The director-election tallies land as **columns per role category**, not per-nominee rows.
Two facts constrain the design:

- The 8-K Item 5.07 director table lists **nominee names only — no titles.** Any role split must
  come from a join back to that ticker's proxy data.
- **Item 402(k) covers non-employee directors only.** So a nominee's presence in the new
  `def14a_director_comp` table *is* the definition of an outside director, and presence in
  `def14a_executive_comp` *is* the definition of an employee-director. That is a much stronger
  signal than the LLM's `is_independent` flag (78% fill, 86% concordance).

### The 4 categories, resolved in order (first match wins)

| # | category | definition | derived from | expected share of nominees |
|---|---|---|---|---|
| 1 | `ceo` | the nominee who is the CEO | name-match to `def14a_llm.ceo_name_proxy`, else the `def14a_executive_comp` row whose title contains "chief executive" | ~1 nominee at ~90% of meetings |
| 2 | `exec_officer` | any OTHER nominee who is a company officer — executive chair, founder, president, vice-chair | name-match to `def14a_executive_comp` (excluding the CEO) | rare; ~10-15% of meetings have one |
| 3 | `non_employee` | outside directors | name-match to `def14a_director_comp` (Item 402(k) ⇒ non-employee by construction) | ~85% of all nominees |
| 4 | `unmatched` | matched nothing — no proxy that year, spelling drift, mid-year nominee | residual | the join's error term |

`unmatched` is deliberately its own bucket: folding it into `non_employee` would make the outside
bucket absorb every join failure and quietly overstate it.

The `exec_officer` row also carries `exec_officer_titles` — a comma-joined text list of the matched
titles (e.g. `"Executive Chairman, President"`). It costs one column and means any finer split of
management board seats is later a pure re-read of that column, with no re-extraction.

### Column layout on the director-election row

4 categories × 5 columns = 20, plus 5 election-level columns:

```
votes_for_{ceo,exec_officer,non_employee,unmatched}
votes_against_{...}            <- "Against" or "Withheld"; see vote_standard
votes_abstain_{...}
votes_broker_non_votes_{...}
n_nominees_{...}
--- election-level (no join needed) ---
n_nominees                     total on the ballot
min_support_pct                min over nominees of for/(for+against)
min_support_name               who it was
n_nominees_below_70pct         the withhold-campaign event counter
vote_standard                  'against' | 'withheld'  (20% of filings use Withheld)
```

Per-category vote counts are **summed across the nominees in that bucket**, so the four buckets
reconstruct the whole election additively. The "a director nearly lost" signal — which is the
highest-value governance event in the table — is carried by `min_support_*` rather than by
averaging away inside a bucket.

Raw per-nominee tallies are preserved in a `nominee_votes_json` TEXT column (same pattern as
today's `def14a_json`), so **recategorising later costs zero LLM calls**.

---

## Phases

| Phase | File | Goal | Independently testable |
|---|---|---|---|
| 0 | [PHASE-0-baseline-harness.md](PHASE-0-baseline-harness.md) | Snapshot today's 6 tables for the 23 tickers; build the comparison script; cache the filings | yes — prints the baseline table |
| 1 | [PHASE-1-no-schema-bugfixes.md](PHASE-1-no-schema-bugfixes.md) | Bugs fixable with no schema change: say-on-pay floor, `_doc_url`, dead `temperature`, prompt fixes | yes — unit tests + live 3-filing probe |
| 2 | [PHASE-2-table-anchored-carve.md](PHASE-2-table-anchored-carve.md) | `def14a_tables.py`: table enumerator + 6 header signatures + TSV; router in `prepare_def14a_sections` | yes — recall harness on cached filings, no LLM |
| 3 | [PHASE-3-schema-and-child-tables.md](PHASE-3-schema-and-child-tables.md) | Pydantic expansion + 4 new tables (incl. `def14a_directors`) + the `reconciles` flag + fee rescale + the gender upgrade | yes — flatten is pure; replay over stored `def14a_json` |
| 4 | [PHASE-4-slim-edgar-path.md](PHASE-4-slim-edgar-path.md) | ECD dimension filter, `company_name` fallback, delete the HTML block and 4 tables | yes — 5-filing live probe incl. BA/NKE/SBUX |
| 5 | [PHASE-5-vote-tallies.md](PHASE-5-vote-tallies.md) | `sec_8k_votes` from stored `item_text` + director categories | yes — hand-read fixtures + guard tests |
| 6 | [PHASE-6-cutover-and-report.md](PHASE-6-cutover-and-report.md) | Run the 23-ticker comparison, gate, then YOU truncate + full rerun; docs + DoD | the gate itself |

### Sequencing constraint (from the research)

`existing_filings` dedups on accession, so **a re-fetch after a parser fix requires TRUNCATE +
refetch, not an incremental run**. That is exactly your plan. Two consequences:

- Phases 0-5 must land **before** any truncation. Phase 6 is the only phase that touches live data.
- Phase 3's flatten can be replayed over the **already-stored** `def14a_json` for free (the tokens
  are paid). That is how Phase 3 is verified without spending a single LLM call.
- The DEF 14A and 8-K backfills are **still running**. Phase 0 snapshots first precisely so the
  baseline is a fixed artifact rather than a moving target.

---

## Out of scope

- **DEFR14A, 10-K/A SCTs, FPI 20-F/6-K, MLPs, special-meeting proxy detection** (D7). Each is a
  few percent of coverage for real complexity.
- **`board_recommendation`** — dies with `sec_def14a_votes`. 39.6% NULL, an unknown share of the 890
  `FOR` values are *fabricated* by a regex that matches the proxy card's column header, and the
  ceiling is Apple's recommendation cell being a JPEG with `alt=""`.
- **Rebuilding the ownership TABLE as a substitute for 13F / SC 13D-G / Forms 3-4-5.** The per-holder
  rows land (D2) but nothing downstream should prefer them to the structured sources.
- **New cube features** off exec/director comp (D8) and any modelling work.
- **Third-party vote data** (ISS Voting Analytics, ProxyMonitor, N-PX aggregation). Settled
  empirically in the research: the issuer's own 8-K is the complete free tally.
- **Fixing edgartools upstream.** Every defect is worked around above the library or the library's
  output is dropped.

## Accepted redundancy / known losses

Recorded here so they are not rediscovered as bugs:

| Item | Why accepted |
|---|---|
| Per-holder ownership rows overlap 13F / SC 13D-G / Forms 3-4-5 | You chose it (D2). ~100% present and 95-97% already inside the carve, so marginal cost ≈ 0. As-of dates never align with 13F. |
| `gender` remains partly an **inference**, even after the D6b upgrade | Only 17.4% of proxies state it. The upgrade replaces a bare first-name prior with in-document honorific/pronoun evidence, records the basis per director, and reconciles across filings — but a residual name-inferred tail stays. `gender_basis` makes that tail measurable and filterable instead of invisible. |
| 9 of 190 (4.7%) multi-filing meetings have genuinely restated tallies with no textual signal | Both filings are stored; a reader unioning on `(ticker, period_of_report)` is right 91% of the time. Not worth a resolution heuristic. |
| ~1% of `item_text` is truncated by edgartools' item splitter (filers numbering proposals "Item 1.") | The Phase-5 fabrication guard turns these into *no rows* rather than invented rows. No HTML re-fetch. |
| XOM's proposal page is a JPG; SYK's fee figures live in `.gif` charts | Unreachable by any text method. |
| Pre-2001 filings are ASCII with no HTML tables | The `_doc_url` fix gets the right *bytes*; the narrative fields extract, the table targets will not. |

---

## Success criteria

Measured by Phase 6 on the 23 baseline tickers, 2000-2026, new vs. snapshot:

- [ ] **Zero** exec-comp rows above $1e9 (baseline: 109 DB-wide, all in `executive_comp`)
- [ ] **Zero** ownership rows where `shares` and `percent_of_class` diverge by ~10× (the PG footnote-digit bug)
- [ ] `n_neos` mean rises materially; the share of accessions with `n_neos == 1` falls well below the current 21.7% (2012+)
- [ ] `auditor_name` fill ≫ 2.05% (research: 98% present, 89% already carved)
- [ ] Director-comp rows exist for ≥ 90% of post-2008 proxies (baseline for the LLM path: 0)
- [ ] Say-on-pay values below 0.50 **survive** — JPM 2023 ≈ 0.31, INTC 2023 ≈ 0.34, SPG 2024 ≈ 0.111 all present
- [ ] `gender` improves, not just persists: `pct_female_directors` fill does **not** fall, `gender_basis` is populated on every non-null gender, and the director-level gender count agrees with the filing's own `n_women_directors` on materially more filings than baseline
- [ ] Pre-2001 rows are no longer fully NULL (baseline: 401 of 422)
- [ ] `sec_def14a` PEO rows are axis-filtered: BA 2026 keeps **both** Ortberg and Calhoun; SBUX FY2023-25 is not 0.0
- [ ] Mean carve payload per filing ≤ 80% of baseline (target ~73%)
- [ ] `sec_8k_votes` populated for ≥ 90% of Item 5.07 filings that contain a comma-grouped number
- [ ] Every new/changed test prints a sanity-check conclusion (AGENTS.md)
- [ ] `pytest tests/data_extract/structure tests/data_aggregate/test_def14a_*` green
- [ ] Docs in sync: `docs/data_schema.md`, `docs/database.md`, `docs/data_sources.md`, `sql/schema.sql`

## Estimated effort

| Phase | Estimate |
|---|---|
| 0 — baseline harness | 0.5 day |
| 1 — no-schema bugfixes | 0.5 day |
| 2 — table-anchored carve | 1.5 days (the real engineering) |
| 3 — schema + 4 child tables | 1 day |
| 4 — slim edgar path | 0.5 day |
| 5 — vote tallies | 1 day |
| 6 — cutover + report | 0.5 day + your full rerun |
| **Total** | **~5.5 days** of implementation, plus the rerun |

LLM spend: ~530 calls for the 23-ticker validation, then ~8,700 proxies + ~6,600 vote filings for
the full rerun. Sized in Phase 6.

## Risk register

| Risk | Mitigation |
|---|---|
| `gpt-5-mini` behaves differently from the measured `gpt-4o-mini` | Phase 0 runs both on 3 filings before Phase 3 commits to the schema; Phase 6 reports the real numbers, not the research's. |
| Table classifier picks a semantically wrong table | Strict signatures (exact fee labels, SCT rejects "compensation actually paid", ownership split in two). Phase 2's harness asserts an identifying VALUE from each table, not just "a table was found". |
| The gender upgrade could *lower* fill instead of raising accuracy (honorifics absent in older ASCII proxies) | `gender_basis` records the provenance, so a fill drop is attributable rather than mysterious. The cross-filing consensus backstops filings with no in-document evidence: a director's gender does not change, and directors recur across tickers and years. Phase 6 gates on fill **not falling**. |
| `sql/schema.sql` regeneration is lossy — the generator drops 8 hand-added indexes | Splice the new table blocks by hand; the diff must be purely additive plus the 4 removed blocks. Verified with `git diff --stat`. |
| Truncating live tables mid-backfill | Phase 6 explicitly stops the running extraction first, and Phase 0's snapshot is the rollback. |
| A stale `def14a_llm` row blocks re-extraction | Full TRUNCATE at cutover (your call, D-preamble), not an incremental run. |
