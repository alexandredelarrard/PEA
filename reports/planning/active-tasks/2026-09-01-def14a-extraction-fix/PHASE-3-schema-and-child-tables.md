# Phase 3 — Pydantic expansion + 4 new LLM-side tables ⬜

**Goal**: turn the data that is *already inside the LLM's input* into queryable tables, add the two
free wins (`auditor_name`, fee breakdown), close the missing seventh SCT component, and drop the
three fields that were never extractions in the first place.

**Depends on**: Phase 2 (the SCT / director-comp / fee / ownership tables must actually reach the
model before asking the model for them).

---

## Why (measured)

- **34,741 per-executive rows across 497 tickers already sit unqueryable inside
  `def14a_llm.def14a_json`.** Against `sec_def14a_executive_comp`'s 2,378 rows on 25 tickers:
  title **100% vs 45.4%**, stock awards 93.8% vs 45.4%, option awards 88.0% vs 27.5%, and
  **2 rows > $1e9 vs 109**. Flattening is free — the tokens are paid.
- **`auditor_name`**: 98% present in the document, **89% already inside today's carve**. It is
  currently the worst column in `sec_def14a` at **2.05%** fill, because edgartools returns `''` on
  failure (40 of 41 cells empty).
- **Audit fee breakdown**: 93% present, **91% already carved**.
- **Director compensation**: absent from the Pydantic contract entirely. The 87,984 `directors[]`
  rows are biographical only.
- **The missing seventh SCT component**: the post-2006 SCT has seven components, the schema models
  six. The residual is **positive on 97.5%** of non-reconciling rows, median **$319,367** — the
  signature of a missing column, not of misread values. That column is *Change in Pension Value and
  Nonqualified Deferred Compensation Earnings*.
- **When the LLM populates a field it is almost always right** (8/8 correct on 6 of 8 audited
  fields); the losses are **recall**, which Phase 2 addresses.

---

## Changes

### 1. `src/data_extract/utils/structure/def14a_schema.py`

#### ADD to `ExecutiveCompensation`
- [ ] `pension_change_usd: Optional[float]` — *"Change in Pension Value and Nonqualified Deferred
      Compensation Earnings" column, USD*.
- [ ] Change `fiscal_year`'s description and the parent field's description so the model returns
      **every (NEO × fiscal year) row the SCT shows**, not only the most recent year. The SCT
      normally carries 3 years; today's contract asks for one, and B now delivers the whole table.

#### ADD `DirectorCompensation` (new model)
- [ ] `name`, `fiscal_year`, `fees_earned_usd`, `stock_awards_usd`, `option_awards_usd`,
      `non_equity_incentive_usd`, `pension_change_usd`, `all_other_comp_usd`,
      `total_compensation_usd`.
- [ ] Attach as `Def14AExtract.director_compensation: list[DirectorCompensation]`.
- [ ] Docstring must record: Item 402(k) covers **non-employee directors only** and requires the
      **last completed fiscal year only**, so this list is single-year by regulation — and
      membership in it *is* the definition of an outside director (Phase 5 depends on that).
      The table exists only from the **2008 proxy season** (Reg S-K 2006, FY ending ≥ 2006-12-15).

#### ADD `BeneficialOwner` (new model)
- [ ] `holder_name`, `holder_type` (`'5pct_holder'` | `'director_officer'`), `shares`,
      `percent_of_class` (decimal; **null** for `'*'` / `'<1%'` — a bound, not a measurement).
- [ ] Attach as `Def14AExtract.ownership_holders: list[BeneficialOwner]`.
- [ ] Description must state: exclude subtotal / "as a group" rows — that aggregate is already the
      `insider_ownership_pct` scalar.

#### ADD to `GovernanceProfile`
- [ ] `auditor_name: Optional[str]` — the registered public accounting firm's name.
- [ ] `auditor_since_year: Optional[int]` — first year of the engagement, when stated (78% present).
- [ ] `audit_fees_audit_usd`, `audit_fees_audit_related_usd`, `audit_fees_tax_usd`,
      `audit_fees_other_usd` — current fiscal year, USD.
- [ ] `auditor_fees_prior_usd` — prior-year **total**, so a fee-growth signal is computable.
- [ ] Existing `auditor_fees_usd` stays as the current-year total.
- [ ] All fee fields: description states the value must be **converted to whole USD**, applying any
      "(in thousands)" / "($ in millions)" note that appears in the table header or the sentence
      before it. This is the fix for the measured **1000× error on 8 of the 10 smallest values**
      (MS 57.6 = $57.6M; TSLA 10,919 = $10.9M *and* the wrong year column).

#### DROP (D6)
- [ ] `GovernanceProfile.n_technology_directors` and `technology_committee`.
- [ ] Rationale to record in the module docstring, once, with the number:
      `n_technology_directors` is **an opinion, not an extraction** — mean |Δ| of 1.06 directors
      between consecutive filings (only 38.8% unchanged) and wrong by 7× on HUBB 2022, where the
      filing's own matrix states "Cybersecurity and Technology 78%" of 9 directors.

#### UPGRADE `DirectorInfo.gender` (D6b) — keep the field, fix the deduction

`gender` stays: it feeds `pct_female_directors`, which carries alpha. But today it is **97.9% filled
while only 17.4% of proxies disclose it**, i.e. it is overwhelmingly a **first-name prior** with no
provenance and no way to tell a stated value from a guess. Four changes, cheapest first:

- [ ] **Use in-document evidence before the name.** A proxy bio almost always contains the
      director's honorific or pronouns (`Mr.` / `Ms.` / `Mrs.`; `he`/`him`/`his` vs `she`/`her`/
      `hers`), and the ownership and comp tables carry honorifics too. Rewrite the field description
      so the model resolves gender in this order:
      1. the proxy **states** it (a diversity matrix / "women directors" identification),
      2. the **honorific** used for that director anywhere in the carved text,
      3. the **pronouns** used in that director's bio,
      4. the first name — **last resort only**.
- [ ] **ADD `DirectorInfo.gender_basis: Optional[str]`** in
      `{'stated', 'honorific', 'pronoun', 'name'}`. This is the whole point of the upgrade: it turns
      an invisible inference into a measurable, filterable one, and it is what lets Phase 6 prove
      accuracy went up rather than just fill. Consistent with the `reconciles` flag philosophy —
      record provenance, don't silently drop.
- [ ] **Cross-filing consensus** (deterministic, no LLM). A director's gender does not change, and
      directors recur across years and across boards — the 87,984 `directors[]` rows collapse to far
      fewer people. After a full run, group the new `def14a_directors` table on a normalised name key
      and take the **highest-provenance majority**: any `stated` value wins outright; else the
      honorific/pronoun majority; else the name-based value. Write the consensus back and recompute
      `pct_female_directors`. This both **fills** directors whose own bio carried no evidence and
      **corrects** unstable inferences.
      - Normalise the name with the existing `clean_person_name`, then key on
        `last name + first initial` (the same key Phase 5 uses for the vote role map), so
        `"Katherine J. Smith"` and `"Kathy Smith"` reconcile.
      - Log the number of people whose inference was *overturned* by a higher-provenance sibling —
        that count is the direct measure of what the upgrade bought.
- [ ] **Cross-check against the filing's own count.** `_flatten` already prefers
      `n_women_directors / board_size` for `pct_female_directors`. Keep that precedence, and add
      `n_women_directors_vs_inferred` = the filing's stated count minus the count derived from
      per-director gender. It is 0 when the two agree; a persistent non-zero is the honest error bar
      on the inference. Do not use it to overwrite anything.

### 2. `fetch_def14a_llm.py` — flatten changes

#### Multi-year SCT breaks three existing computations. All three must be fixed together.
- [ ] `_ceo_from_compensation` must select from the **most recent fiscal year** subset before
      matching on name / title / first-row. Today it scans the whole list, which with multi-year
      rows would return an arbitrary year's CEO row.
- [ ] `n_neos = len(extract.compensation)` becomes **distinct names in the most recent fiscal year**.
      Left as-is it would triple and the `n_neos == 1` metric would become meaningless.
- [ ] `total_neo_comp` becomes the sum over the **most recent fiscal year** only.
- [ ] Add a `sct_years` count column so the "1 year instead of 3" pathology stays measurable.

#### Flat-column changes on `def14a_llm`
- [ ] ADD: `auditor_name`, `auditor_since_year`, `audit_fees_audit`, `audit_fees_audit_related`,
      `audit_fees_tax`, `audit_fees_other`, `auditor_fees_prior`, `sct_years`,
      `n_director_comp_rows`, `n_ownership_rows`, `n_women_directors_vs_inferred`,
      `pct_gender_stated` (share of the board whose `gender_basis` is `stated` or `honorific` —
      the per-filing confidence in `pct_female_directors`).
- [ ] REMOVE: `n_technology_directors`, `pct_technology_directors`, `technology_committee`.
- [ ] Update `_NUMERIC_COLS` in lockstep — it is the list that keeps DB columns numeric.
- [ ] `pct_female_directors` keeps its existing precedence (the filing's own `n_women_directors`
      first, the per-director `genders` ratio as fallback). The fallback now runs on
      consensus-corrected genders, so it gets **better**, not narrower.
- [ ] Keep `def14a_json`. It is what makes re-flattening free, and this phase's verification depends
      on replaying it.

#### New child-table builders (pure functions)
- [ ] `_exec_comp_rows(ticker, filing, extract) -> list[dict]`
- [ ] `_director_comp_rows(ticker, filing, extract) -> list[dict]`
- [ ] `_ownership_rows(ticker, filing, extract) -> list[dict]`
- [ ] `_director_rows(ticker, filing, extract) -> list[dict]` — the `directors[]` array flattened:
      `name`, `age`, `tenure_years`, `is_independent`, `gender`, `gender_basis`,
      `other_public_company_boards`.
- [ ] Each stamps `ticker`, `accession_number`, `as_of` (the filing date — point-in-time, leak-free),
      `cik`.
- [ ] `_save_ticker_rows` extends to upsert all five frames per ticker, keeping the existing
      "persist per ticker so a crash loses nothing" property.

#### Why `def14a_directors` becomes a table (D6b)

The 87,984 `directors[]` rows are the **most trustworthy thing in the extract** — 99.74% of names
appear verbatim in the source, 93% of ages and 98% of tenures are confirmable, and a full hand-check
of HUBB 2022 was 27/27 correct including public-vs-private board judgements. They are currently
unqueryable inside `def14a_json`, and the gender consensus pass **cannot be written without them**
(it needs a GROUP BY over people, across tickers and years). Flattening is free — same pattern, same
paid tokens.

#### Gender consensus finalisation pass

- [ ] Runs **once at the end of `fetch_def14a_llm`**, after the per-ticker loop — a cross-ticker
      consensus needs every ticker's rows, so it cannot live inside the loop.
- [ ] Narrow read of `def14a_directors` (`name`, `gender`, `gender_basis` only — never unprojected),
      compute the consensus, then two narrow writes: the corrected `gender` / `gender_basis` back to
      `def14a_directors`, and the recomputed `pct_female_directors` / `pct_gender_stated` to
      `def14a_llm`.
- [ ] Cheap on a daily rerun: DEF 14A is a yearly filing, so an incremental day adds ~0 rows and the
      pass is a narrow read plus a no-op. Skip it entirely when the run extracted no new filings.
- [ ] Print the consensus summary: distinct people, rows filled, rows **overturned**, and the
      `gender_basis` distribution before/after. That print is the evidence the upgrade worked.

#### The `reconciles` flag (D10)
- [ ] On both comp tables add `reconciles`: `1.0` when `abs(total − Σ components) <= 10.0`, else
      `0.0`; NaN when `total` is null. **Values are never dropped or nulled** — the flag is the
      output, and the failure rate becomes measurable over time.
- [ ] Put the tolerance in `src/constants/constants.py` only if a second non-test consumer appears;
      otherwise it lives next to its single consumer (repo constants-placement rule).
- [ ] For 2023+ filings, `sec_def14a.peo_total_comp` is an **independent filer-tagged check** on the
      CEO row. Do not enforce it in the fetcher (different table, different write) — record it as a
      Phase 6 comparison check.

### 3. `src/data_extract/utils/structure/def14a_validate.py` — repurpose, don't delete

The module was written for edgartools' defects, but four of its parts are still the right tool for
LLM-sourced rows. Keep them; Phase 4 removes the rest.

- [ ] KEEP `clean_text`, `clean_person_name` (footnote-suffix strip keeps the **name primary key**
      stable — without it a director keys as `"Emma N. Walmsley11"` one year and `"...10"` the next).
- [ ] KEEP `_SUBTOTAL_HOLDER_RE` and `_ADDRESS_ONLY_RE` / `_ADDRESS_TAIL_RE` — an LLM can still
      return an "as a group" line or an address as a holder.
- [ ] KEEP `_rescale_block` + `DEF14A_AUDIT_FEE_MIN_PLAUSIBLE` as the **safety net** behind the
      prompt's unit instruction. Same rationale as before: edgartools reported a whole fee table in
      one unit and so does a filer, so the block is rescaled together or not at all.
- [ ] Rewrite `_reconcile_components` to **compute the flag** instead of writing values. Drop the
      duplicated-Total repair (that was an edgartools grid artifact) and the single-missing-component
      residual fill — with `pension_change_usd` in the schema the residual is no longer an
      unattributable gap, and filling it would now overwrite a real column.
- [ ] Update the module docstring to say what it is now: the LLM-side row cleaner. Keep the measured
      numbers, drop the before/after chronology.

### 4. `src/data_store/schema.py` (risk zone — ask before editing)

- [ ] Register four tables next to `def14a_llm`:

```python
# Director / nominee roster flattened out of `def14a_llm.def14a_json`: one row per director per
# filing. The most trustworthy block in the extract -- 99.74% of names appear verbatim in the
# source, 93% of ages and 98% of tenures confirmable. `gender_basis` records HOW gender was
# resolved ('stated' > 'honorific' > 'pronoun' > 'name'); only 17.4% of proxies state it, so the
# provenance is what makes the field auditable. A cross-filing consensus over this table is what
# fills and corrects the inference (see the finalisation pass).
def14a_directors = Table(
    "def14a_directors", ("ticker", "accession_number", "name"),
    date_col="as_of", date_type_cols=("as_of",))
```

```python
# Summary Compensation Table rows flattened out of `def14a_llm.def14a_json` (Item 402(c)):
# one row per NEO per fiscal year, ~3 years per filing. `reconciles` = 1 when the seven
# components sum to `total` within $10 -- a FLAG, not a filter; the values are kept either way.
def14a_executive_comp = Table(
    "def14a_executive_comp", ("ticker", "accession_number", "name", "fiscal_year"),
    date_col="as_of", date_type_cols=("as_of",))
# Non-employee Director Compensation Table (Item 402(k)): one row per director per filing.
# Single-year BY REGULATION -- 402(k) requires the last completed fiscal year only -- and
# membership here IS the definition of an outside director. Exists only from the 2008 season.
def14a_director_comp = Table(
    "def14a_director_comp", ("ticker", "accession_number", "name"),
    date_col="as_of", date_type_cols=("as_of",))
# Beneficial-ownership rows (Item 403). KNOWINGLY redundant with 13F / SC 13D-G / Forms 3-4-5,
# which are the preferred sources; the proxy-only figure is the directors-and-officers group
# aggregate, which is the `insider_ownership_pct` scalar on `def14a_llm`, not a row here.
def14a_ownership = Table(
    "def14a_ownership", ("ticker", "accession_number", "holder_name", "holder_type"),
    date_col="as_of", date_type_cols=("as_of",))
```

### 5. `sql/schema.sql` (risk zone)

- [ ] **Splice the four new blocks by hand.** Do not regenerate: the generator **drops 8
      hand-added indexes**. The diff must be purely additive (plus Phase 4's four removals).
- [ ] Add the `date_col` index for each new table, matching the existing naming
      (`ix_<table>_<date_col>`).
- [ ] Verify with `git diff --stat sql/schema.sql` and read the diff — additive only.

### 6. Prompt (`_DEF14A_PROMPT`)

Additive edits only, matching Phase 2's `=== LABEL ===` blocks:

- [ ] SCT: return **every NEO × fiscal-year row** in the table, with `fiscal_year` from the Year
      column. A `-` or blank cell is 0. Do not confuse with the Pay-versus-Performance table
      ("compensation actually paid") or with the DIRECTOR compensation table.
- [ ] Director comp: one row per director; `fees_earned_usd` is the cash-retainer column whatever it
      is labelled (`Fees Earned or Paid in Cash`, `Cash Fees`, `Retainer`); `stock_awards_usd` covers
      `Stock Awards` / `Restricted Stock Units` / `Share Awards`.
- [ ] Audit fees: apply the units note; return whole USD; report the four categories for the current
      year plus the prior-year total; `auditor_name` is the accounting firm, not a sentence.
- [ ] Ownership: one row per holder; exclude subtotal / group rows; `percent_of_class` null for
      `'*'`/`'<1%'`; **percent of the class of shares outstanding, never a "% of voting power"
      column**.
- [ ] Directors: resolve `gender` from the proxy's own statement, else the honorific used for that
      director, else the pronouns in their bio, else the first name — and set `gender_basis` to
      whichever of `stated` / `honorific` / `pronoun` / `name` was used. Never leave `gender_basis`
      null when `gender` is set.

---

## Verification

### The free replay — no LLM calls at all
- [ ] The flatten and the four row builders are **pure functions of `Def14AExtract`**. Replay them
      over the **already-stored** `def14a_json` for the 23 baseline tickers (Phase 0's parquet) and
      assert: exec-comp rows are produced, the `reconciles` distribution is printed, and no row
      exceeds $1e9. This proves the flatten before a single token is spent.
- [ ] Print the replay conclusion: rows produced per table, `reconciles` rate, max value per column.
      The research's expectation is **2 rows > $1e9 out of 34,741** (versus 109 on the edgar path) —
      if the replay shows more, the flatten has a bug, not the LLM.

### New extraction probe (~12 LLM calls, cached filings)
- [ ] CAT 2026 / PFE 2026 / GE 2026 — the director-comp cases where the value was present in 4 of 4
      opened and the parser returned NULL. Assert `stock_awards_usd` is non-null and matches the
      filed figure (CAT: `175,033` under `Restricted Stock Units (1)`; PFE: `205,000` on 13/13 rows;
      GE: `345,795`).
- [ ] AAPL 2026 — SCT must yield 6 NEOs × the years shown, not 6 rows all `year=2025`.
- [ ] MS / TSLA — audit fees must come back in whole USD (the measured 1000× cases).
- [ ] A filing with `auditor_name` — assert a firm name, not a 148-character sentence.
- [ ] **Gender**: HUBB 2022 (hand-checked 27/27 correct in the research) — every director carries a
      `gender_basis`, and the count of female directors matches the filing's own statement
      (`n_women_directors_vs_inferred == 0`). Also run one **pre-2001 ASCII** proxy to see what
      `gender_basis` degrades to when there is no honorific — that is the case the consensus pass
      exists for.

### Gender consensus test (deterministic, no LLM)
- [ ] Unit test on a synthetic roster where the same person appears 4 times: once `stated` female,
      twice `name`-inferred male, once null. Assert the `stated` value wins outright, all 4 rows end
      up female, and the overturned count is 2. Then the same with no `stated` row, asserting the
      honorific/pronoun majority wins over the name.
- [ ] Real-data run over Phase 0's `def14a_json` for the 23 tickers: print distinct people, rows
      filled, rows overturned, and the `gender_basis` distribution. Sanity conclusion must state the
      overturn count — if it is 0 on real data, the name key is too strict and is not matching people
      across filings.

### Tests
- [ ] Extend `tests/data_extract/structure/test_def14a_llm.py`: schema round-trip for the four new
      models plus `gender_basis`; `_flatten` with a 3-year SCT asserting `n_neos` counts **names**,
      not rows; `reconciles` boundary cases at exactly $10 and $10.01.
- [ ] `tests/data_extract/structure/test_def14a_nul.py` — confirm `_strip_nul` still covers the new
      TEXT columns (`auditor_name`, holder names). Postgres TEXT rejects `\x00`.
- [ ] `"$PY" -m pytest tests/data_extract/structure tests/data_aggregate/test_def14a_impute.py tests/data_aggregate/test_governance_features.py -q`
      — the last one because dropping `pct_technology_directors` touches
      `def14a_impute.INTERP` / `FLAGS` and `governance_features`. `pct_female_directors` must still
      appear as a feature; it is the one field in that panel this phase is trying to *improve*.

### Downstream sweep for the dropped fields
- [ ] `grep -rn "n_technology_directors\|pct_technology_directors\|technology_committee" src configs tests docs`
      → zero references after the change. Known sites: `def14a_impute.INTERP`/`FLAGS`,
      `fetch_def14a_llm._NUMERIC_COLS` and `_flatten`, `configs/build_cube.yml` governance group,
      `def14a_schema`, `sql/schema.sql`, `docs/data_schema.md`.

## Rollback

The four tables are new and unreferenced by the cube (D8), so dropping them is safe. The
`def14a_llm` column removals are the only destructive part — Phase 0's parquet snapshot holds the
old values for the 23 tickers, and the full table is not truncated until Phase 6. The gender
consensus pass writes back to `def14a_directors` / `def14a_llm`, so it must be **idempotent**: it
recomputes from `gender_basis`, which it never downgrades, so a second run is a no-op.

## Notes

- Do **not** wire any of this into the cube (D8). `governance_features` keeps reading `def14a_llm`
  and benefits automatically from the better fill.
- Ownership rows land because you asked for them; nothing downstream should prefer them to 13F /
  SC 13D-G / Forms 3-4-5, and their as-of dates never align with 13F's quarter-end.
- `insider_ownership_pct` is worth lifting from 57.3% fill — that is a *recall* fix from Phase 2 plus
  the dual-class prompt fix from Phase 1, not a schema change.
- The `n_neos == 1` metric is the single best summary of whether Phase 2 + 3 worked. Keep
  `sct_years` so it stays measurable after the rerun.
