# Phase 1 — Bugs fixable with no schema change ✅

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

- [x] Delete `SAY_ON_PAY_MIN_SUPPORT` (line 47) and `_drop_implausible_say_on_pay` (line 140).
- [x] Keep `drop_implausible_def14a` as the **function**, now a no-op stub with a docstring
      explaining what it is for and that it currently drops nothing. Rationale: it is the
      documented seam for "present but known-wrong" cells, `step_cube_extras.py:158` calls it, and
      the drop→impute ordering contract (`impute_def14a` is non-destructive by test) must survive.
      Do **not** delete the call site — that ordering is load-bearing and was broken once already.
- [x] Rewrite the docstring: state the measurement (14/14 correct, 61 correct rows previously
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

- [x] Replace the `SAY_ON_PAY_MIN_SUPPORT == 0.50` assertion with the opposite claim: the three
      real sub-0.50 disclosures **survive** `drop_implausible_def14a` → `impute_def14a`.
- [x] Fixture values from the filings themselves: JPM 2023 `0.31`, INTC 2023 `0.34`, SPG 2024
      `0.111`. Real observed values, consistent with the repo's convention that parsing math gets
      known-truth fixtures.
- [x] Print a sanity conclusion: `3 real revolts in -> 3 survive; 0 cells nulled`.

### 3. `src/data_extract/utils/common/edgar_fillings.py:24` — the `_doc_url` directory bug

`submissions/CIK*.json` has `primaryDocument = ""` for pre-2001 filings, so `_doc_url` builds a bare
directory URL and the LLM was handed an **EDGAR folder index page**. Measured cost: **401 of 422
pre-2001 rows are fully NULL, versus 18 of 8,245 after.** The carve works fine on those filings
once the right bytes are fetched.

- [x] Fall back to the full-submission text file when `primary_doc` is empty:

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

- [x] This is a **shared** helper — `fetch_filing_text` and the employees path also use
      `list_filings`. Grep for other `doc_url` consumers and confirm the `.txt` form is acceptable
      to each (it is: all of them run `html_to_text` first).

### 4. `src/data_extract/utils/common/llm_extractor.py` — remove the dead `temperature`

`__init__` accepts and stores `temperature`, and `extract` **never sends it** — the kwargs dict is
`model` / `input` / `instructions` / `text_format` / `prompt_cache_key` only. So the class docstring's
"`temperature=0` makes the extraction deterministic" is false today, and `gpt-5-mini` (a reasoning
model) does not accept the parameter anyway.

- [x] Delete the `temperature` parameter and the `self._temperature` attribute.
- [x] Correct the docstring: keep the `cache=True` / `prompt_cache_key` explanation (that one is
      real and load-bearing), drop the determinism claim.
- [x] Update the two call sites: `fetch_def14a_llm.fetch_def14a_llm(temperature=...)` signature and
      the `LLMExtractor(...)` construction. Check `tests/data_extract/structure/test_def14a_llm.py`
      for a passed `temperature`.
- [x] Grep the whole repo for `LLMExtractor(` — any other caller must be updated in the same commit.

### 5. `fetch_def14a_llm.py` — prompt fixes (no schema change)

Three measured extraction errors that are pure prompt problems:

- [x] **`insider_ownership_pct` reads the wrong column** on dual-class issuers (2 of 8 populated
      values wrong — it took a *voting-power* column instead of the economic stake). Add to
      `_DEF14A_PROMPT`: the percent must be the **percent of the class of shares outstanding**
      (economic ownership), never a "% of total voting power" / "combined voting power" column,
      which dual-class issuers print alongside it.
- [x] **`majority_voting` and `poison_pill` must become tri-state** (D6). The current prompt says
      *"companies disclose these when they exist, so return FALSE when the proxy does not indicate
      the provision is in place — do NOT leave them null"*. That instruction is exactly why
      `poison_pill` is TRUE in 0.1% of rows (degenerate) and `majority_voting` **flips 21.2%
      year-over-year on a bylaw that does not change**. Change to: return TRUE or FALSE only when
      the proxy states the provision's status; **null when the proxy is silent**.
      Keep the infer-FALSE instruction for `classified_board` and `dual_class_shares`, which are
      structurally always disclosed.
- [x] **`avg_other_public_boards` upward bias** (D6): 37.2% of filings mix null and 0 and the nulls
      are systematically the zero-board directors. Fix in the field *description*
      (`def14a_schema.DirectorInfo.other_public_company_boards`): set `0` **only** when the proxy
      explicitly shows a count of zero or states the director serves on no other public boards;
      leave null when other-board service is simply not disclosed for that director. The flatten's
      `_mean` over non-null values is then unbiased and needs no change.

### 6. `configs/configs.yml` — no change

`llm_model: gpt-5-mini` is already the production value (D11). The `gpt-4o-mini` default inside
`fetch_def14a_llm` is dead in production (the CLI and the Step both pass the config value) but
should be aligned to avoid a misleading default:

- [x] Change the `model` default in `fetch_def14a_llm` to `gpt-5-mini`, or drop the default and
      require the caller to pass it. Prefer dropping the default — a model name defaulting in two
      places is how the research ended up measuring a different model than production runs.

---

## DEVIATIONS from the plan as written

1. **The `_doc_url` fix needed a second half.** Phase 0 found two pre-2001 shapes, not one:
   88 filings with `primaryDocument == ""` (the folder-index case the plan describes) and **7
   that NAME a document absent from the archive** (`"0001.txt"`, all 2000-08..2001-03: AEE, GE,
   NKE, PEG, REG ×2, T). The second shape 404s, so `_process_filing` swallows the raise and the
   filing produces **no row at all** — a loss invisible in the "401 of 422 fully NULL" count.
   `_doc_url` cannot fix it (`primary_doc` is truthy), so `list_filings` now also emits
   **`txt_url`**, and `_fetch_filing_html` retries it when the primary GET fails. A
   404-triggered fallback needs no date cutoff and no `"0001.txt"` hardcode.
2. **The blast radius is one consumer, not three.** The plan expects `fetch_filing_text` and the
   employees path to use `list_filings`; `grep` shows **only `fetch_def14a_llm`** does today
   (the employees path moved to edgartools). No other call site needed checking.
3. **`fetch_def14a_llm(model=...)` is now REQUIRED**, per the plan's preferred option. Two test
   call sites in `test_def14a_incremental.py` had to pass it.

## Verification

- [x] `"$PY" -m pytest tests/data_aggregate/test_def14a_say_on_pay.py -v -s`
      → **5 passed**, printed conclusion: `3 real revolts in -> 3 survive; 0 cells nulled`.
      The file was rewritten around the opposite claim, incl. a new test that the stub returns a
      **copy** (a stub aliasing its input would make a future drop rule mutate the cube's frame).
- [x] `"$PY" -m pytest tests/data_extract/structure tests/data_aggregate/test_def14a_impute.py tests/data_aggregate/test_def14a_say_on_pay.py -q`
      → **110 passed, 2 skipped**. The impute non-destructiveness invariant holds.
- [x] **`_doc_url` live probe, both shapes** — all four now yield real proxy text:

| ticker | shape | chars | contains proxy + annual meeting |
|---|---|---|---|
| GE | empty `primaryDocument` | 123,176 | ✓ |
| A | empty `primaryDocument` | 133,823 | ✓ |
| T | named-but-missing (404) | 165,380 | ✓ |
| NKE | named-but-missing (404) | 141,936 | ✓ |

      Baseline for all four: the ~10 KB folder index → **1,487 chars**, containing neither word —
      or, for the 404 shape, no row at all. `_doc_url("40545", "...", "")` now ends `.txt`.
- [x] **Prompt probe** (5 LLM calls on cached filings; NKE is the only dual-class ticker in the 23):

| filing | `poison_pill` | `majority_voting` | `insider%` | baseline (pois/maj/ins) |
|---|---|---|---|---|
| NKE 2025 | **None** | **None** | None | 0.0 / 0.0 / nan |
| NKE 2024 | **None** | **None** | 0.011 | 0.0 / 0.0 / 0.011 |
| NKE 2023 | **None** | False | 0.005 | 0.0 / 0.0 / 0.005 |
| AAPL 2026 | **None** | **None** | None | 0.0 / 0.0 / nan |
| CAT 2026 | **None** | **None** | None | 0.0 / 0.0 / nan |

      **`poison_pill` is now NULL on 5/5** where the baseline had a fabricated FALSE on all five
      (DB-wide baseline: FALSE on 432/445 = 97.1%, TRUE on 4 = 0.9% — degenerate). The one
      surviving `False` is NKE 2023, whose proxy *states* plurality voting — i.e. the tri-state
      is discriminating, not just nulling everything.
      **`insider_ownership_pct` did not change on NKE** (0.011 / 0.005, identical to baseline):
      those were already the economic column, so the 2-of-8 miscolumn cases the research found
      are not in this ticker set. Recorded rather than claimed as fixed. It stays **None on 3/5**,
      which confirms Phase 0's finding that its 57.3% fill is a **carve** problem — Phase 2's job,
      not the prompt's.
- [x] `grep -rn "LLMExtractor(\|temperature=" src tests` → no stale references; the only remaining
      `temperature` mentions are the docstring explaining why it is gone.

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
