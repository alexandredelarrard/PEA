# Phase 1 — Bugs fixable with no schema change ⬜

**Goal**: land every fix that needs no new column and no new table. Each is independently correct,
independently testable, and improves the *currently running* extraction path immediately.

**Why separate**: these are cheap, high-confidence, and touching them here keeps Phase 2 and 3
about one thing each.

---

## Changes

### 1. `src/data_aggregate/utils/extras/def14a_impute.py` — delete the say-on-pay floor

The floor deletes the **highest-signal events in the table**. Measured: **14 of 14** sampled
sub-0.50 values are *correct*, and all three counter-examples the docstring cites as proof of
extraction error are real disclosures:

- JPM 2023 — *"the **31% support** we received for last year's say-on-pay resolution"*
- INTC 2023 — *"received **only 34% support**"*
- SPG 2024 — *"**11.1% of the votes cast** favored our Say-on-Pay"*

- [ ] Delete `SAY_ON_PAY_MIN_SUPPORT` (line 47) and `_drop_implausible_say_on_pay` (line 140).
- [ ] Keep `drop_implausible_def14a` as the **function**, now a no-op stub with a docstring
      explaining what it is for and that it currently drops nothing. Rationale: it is the
      documented seam for "present but known-wrong" cells, `step_cube_extras.py:158` calls it, and
      the drop→impute ordering contract (`impute_def14a` is non-destructive by test) must survive.
      Do **not** delete the call site — that ordering is load-bearing and was broken once already.
- [ ] Rewrite the docstring: state the measurement (14/14 correct, 61 correct rows previously
      nulled) and *why* the guard was wrong, not the history of adding and removing it.

```python
def drop_implausible_def14a(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """VALIDATION seam: null cells that are present but known-wrong. Currently drops NOTHING.

    The only rule this ever held -- a 0.50 floor on `say_on_pay_support_pct` -- was measured
    WRONG: 14 of 14 sampled sub-0.50 values are real shareholder revolts (JPM 2023 disclosed 31%
    support, INTC 2023 34%, SPG 2024 11.1%), and the floor nulled 61 correct rows -- exactly the
    highest-signal governance events in the table. Kept as a function because callers run
    drop -> impute in that order, and `impute_def14a` is non-destructive by contract."""
```

### 2. `tests/data_aggregate/test_def14a_say_on_pay.py` — invert the test

- [ ] Replace the `SAY_ON_PAY_MIN_SUPPORT == 0.50` assertion with the opposite claim: the three
      real sub-0.50 disclosures **survive** `drop_implausible_def14a` → `impute_def14a`.
- [ ] Fixture values from the filings themselves: JPM 2023 `0.31`, INTC 2023 `0.34`, SPG 2024
      `0.111`. Real observed values, consistent with the repo's convention that parsing math gets
      known-truth fixtures.
- [ ] Print a sanity conclusion: `3 real revolts in -> 3 survive; 0 cells nulled`.

### 3. `src/data_extract/utils/common/edgar_fillings.py:24` — the `_doc_url` directory bug

`submissions/CIK*.json` has `primaryDocument = ""` for pre-2001 filings, so `_doc_url` builds a bare
directory URL and the LLM was handed an **EDGAR folder index page**. Measured cost: **401 of 422
pre-2001 rows are fully NULL, versus 18 of 8,245 after.** The carve works fine on those filings
once the right bytes are fetched.

- [ ] Fall back to the full-submission text file when `primary_doc` is empty:

```python
def _doc_url(cik: str, accession: str, primary_doc: str) -> str:
    """Absolute URL of a filing's primary document.

    `primaryDocument` is EMPTY in the submissions JSON for filings up to ~2000 (EDGAR stored no
    per-document index then), and a bare directory URL returns a ~10 KB FOLDER INDEX page. The
    `<accession>.txt` full-submission file is the document for those years -- it is ASCII or early
    HTML wrapped in SGML, which `html_to_text` handles. Measured: 401 of 422 pre-2001 DEF 14A rows
    were extracted from the folder index and came back fully NULL."""
    acc_nodash = accession.replace("-", "")
    base = f"{SEC_ARCHIVES_BASE_URL}/{int(cik)}/{acc_nodash}"
    return f"{base}/{primary_doc}" if primary_doc else f"{base}/{accession}.txt"
```

- [ ] This is a **shared** helper — `fetch_filing_text` and the employees path also use
      `list_filings`. Grep for other `doc_url` consumers and confirm the `.txt` form is acceptable
      to each (it is: all of them run `html_to_text` first).

### 4. `src/data_extract/utils/common/llm_extractor.py` — remove the dead `temperature`

`__init__` accepts and stores `temperature`, and `extract` **never sends it** — the kwargs dict is
`model` / `input` / `instructions` / `text_format` / `prompt_cache_key` only. So the class docstring's
"`temperature=0` makes the extraction deterministic" is false today, and `gpt-5-mini` (a reasoning
model) does not accept the parameter anyway.

- [ ] Delete the `temperature` parameter and the `self._temperature` attribute.
- [ ] Correct the docstring: keep the `cache=True` / `prompt_cache_key` explanation (that one is
      real and load-bearing), drop the determinism claim.
- [ ] Update the two call sites: `fetch_def14a_llm.fetch_def14a_llm(temperature=...)` signature and
      the `LLMExtractor(...)` construction. Check `tests/data_extract/structure/test_def14a_llm.py`
      for a passed `temperature`.
- [ ] Grep the whole repo for `LLMExtractor(` — any other caller must be updated in the same commit.

### 5. `fetch_def14a_llm.py` — prompt fixes (no schema change)

Three measured extraction errors that are pure prompt problems:

- [ ] **`insider_ownership_pct` reads the wrong column** on dual-class issuers (2 of 8 populated
      values wrong — it took a *voting-power* column instead of the economic stake). Add to
      `_DEF14A_PROMPT`: the percent must be the **percent of the class of shares outstanding**
      (economic ownership), never a "% of total voting power" / "combined voting power" column,
      which dual-class issuers print alongside it.
- [ ] **`majority_voting` and `poison_pill` must become tri-state** (D6). The current prompt says
      *"companies disclose these when they exist, so return FALSE when the proxy does not indicate
      the provision is in place — do NOT leave them null"*. That instruction is exactly why
      `poison_pill` is TRUE in 0.1% of rows (degenerate) and `majority_voting` **flips 21.2%
      year-over-year on a bylaw that does not change**. Change to: return TRUE or FALSE only when
      the proxy states the provision's status; **null when the proxy is silent**.
      Keep the infer-FALSE instruction for `classified_board` and `dual_class_shares`, which are
      structurally always disclosed.
- [ ] **`avg_other_public_boards` upward bias** (D6): 37.2% of filings mix null and 0 and the nulls
      are systematically the zero-board directors. Fix in the field *description*
      (`def14a_schema.DirectorInfo.other_public_company_boards`): set `0` **only** when the proxy
      explicitly shows a count of zero or states the director serves on no other public boards;
      leave null when other-board service is simply not disclosed for that director. The flatten's
      `_mean` over non-null values is then unbiased and needs no change.

### 6. `configs/configs.yml` — no change

`llm_model: gpt-5-mini` is already the production value (D11). The `gpt-4o-mini` default inside
`fetch_def14a_llm` is dead in production (the CLI and the Step both pass the config value) but
should be aligned to avoid a misleading default:

- [ ] Change the `model` default in `fetch_def14a_llm` to `gpt-5-mini`, or drop the default and
      require the caller to pass it. Prefer dropping the default — a model name defaulting in two
      places is how the research ended up measuring a different model than production runs.

---

## Verification

- [ ] `"$PY" -m pytest tests/data_aggregate/test_def14a_say_on_pay.py -v -s`
      → the 3 real revolts survive; the printed conclusion says `0 cells nulled`.
- [ ] `"$PY" -m pytest tests/data_extract/structure/test_def14a_llm.py tests/data_aggregate/test_def14a_impute.py -v -s`
      → green (the impute non-destructiveness test is the invariant that must not break).
- [ ] **`_doc_url` live probe**: for one pre-2001 filing with `primaryDocument == ""` (from Phase 0's
      cache), assert the URL now ends in `.txt`, `sec_get` returns > 50 KB, and `html_to_text` yields
      text containing the words `proxy` and `annual meeting`. Print the char count before/after —
      baseline is the ~10 KB folder index.
- [ ] **Prompt probe** (LLM, ~6 calls): re-extract 2 dual-class filings from cache (e.g. GOOGL, or
      any `dual_class_shares=True` row in the baseline) and confirm `insider_ownership_pct` now
      matches the economic-ownership column. Confirm `poison_pill` comes back **null**, not False,
      on a filing that never mentions a rights plan.
- [ ] `grep -rn "LLMExtractor(\|temperature=" src tests` → no stale references.
- [ ] Full suite for the touched packages:
      `"$PY" -m pytest tests/data_extract/structure tests/data_aggregate/test_def14a_impute.py -q`

## Rollback

Every change is a small, self-contained edit. `git revert` per commit. Nothing is deleted from the
DB and no schema changes, so a revert is complete.

## Notes

- `drop_implausible_def14a` becoming a stub is deliberate, not an oversight. The comment must say so
  or a future cleanup will delete the seam and re-break the drop→impute ordering.
- The `_doc_url` fix touches a **shared** helper used by 3 fetchers. It is in Phase 1 rather than
  Phase 3 precisely so its blast radius is verified on its own.
- Do not batch the prompt edits with Phase 3's schema expansion. If a field's fill rate moves, you
  want to know whether it was the prompt or the schema.
