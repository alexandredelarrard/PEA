# Phase 6 — `constants.py` and the duplicated helpers ⬜

**Goal**: `constants.py` holds only what 2+ non-test `src/` files read; every literal that the
standard says belongs there is there; and the helpers that exist twice exist once.

**Risk zone, approved**: `src/constants/constants.py`.

---

## 6.1 The new rule

`docs/coding_standard.md:12-18` currently says: *"Before naming any column, key, URL, threshold,
or date format: grep constants.py (927 lines). Reuse if present; add it there before referencing
it if not. Never hardcode a global literal inline."* The line count is also stale — the file is
**1,058** lines.

- [ ] Rewrite that section as:

  > **A module-level constant lives in `src/constants/constants.py` only when 2+ different
  > non-test `src/` files read it.** A constant read by exactly one `src/` file (however many
  > tests import it) lives in that file. Tests import it from wherever it lives.
  >
  > Unchanged: never hardcode a *shared* literal inline; table names live only in `schema.py`;
  > tunable numbers live in `configs/`.

- [ ] Update the stale line count, or better, drop it — it goes stale every time.
- [ ] Add the counter-rule so the pendulum does not swing back: a constant used once but
      *conceptually* shared (a SEC URL template, a date format) still goes in `constants.py` the
      moment a second reader appears; the rule is about where it lives *today*, not a licence to
      re-type it.

## 6.2 Delete the 46 zero-consumer symbols

Measured: of 154 symbols, **46 have no `src/` consumer at all**, **38 are fully dead** (0 src,
0 tests, 0 scripts, no self-reference). Verified independently: no `import *`, no
`getattr(constants, ...)` anywhere in the repo, so a grep census is sound.

| Group | Count | Lines | Note |
|---|---|---|---|
| Plausibility / tolerance scalars | 23 | `843-1023` (194 lines, **148 of them comment**) | appear only in `docs/coding_standard.md:24` and two `reports/planning/` files |
| `GICS_SECTOR_*` / `GICS_GROUP_*` | 10 | `794-804` | 7 feed only `SECTOR_KPI_SCOPE` in the same file; **all 10 names are re-typed as dict keys in `common/gics.py:19-85`**, which is the module actually imported |
| `NOTES_*` narrative | 3 | `541-578` | the section's own comment says the consumer "has been removed" |
| Sharadar zero-rule | 4 | `370-394` | declared for `diagnostics.py`, which does not reference them |
| Data-freshness | 2 | `764-770` | the header says the gate that read these was removed; `schema.py::freshness_tables()` also documents "NO caller" |
| SEC URLs | 2 | `42-43` | `sec_utils.py:124` documents that the derived table was dropped |
| Other | 2 | `740`, `929` | plus `FUNDAMENTALS_ROSTERS_FILENAME` (`:112`), used only in `scripts/` |

- [ ] Delete the symbol **and its comment block**. The 23 plausibility scalars are 194 lines of
      which 148 are comment — that alone is 18 % of the file.
- [ ] Keep `FUNDAMENTALS_ROSTERS_FILENAME` (`:112`): `scripts/` is not `src/`, but three
      independent uncached readers of `fundamentals_rosters.json` exist (`validate/cli.py:133`,
      `sweep_fundamentals_resolution.py:66`, `report_fundamentals_sweep.py:598`) and the filename
      is genuinely shared. Note it and move on.
- [ ] Before each delete: `rtk grep -rn "<SYMBOL>" src/ tests/ scripts/ app/ main.py configs/ sql/ docs/`.
      A hit in `docs/` or `reports/` is **not** a consumer — fix the doc instead.

## 6.3 Relocate the 45 `fundamentals/`-exclusive symbols

`constants.py`'s fundamentals slice is 47 symbols over 120 of its 365 definition lines, **45 of
them exclusive** to `fundamentals/**` + `fundamentals_sharadar/**`. `src/validate/**` consumes
exactly **2**, both at `validate/cli.py:79`; nothing under `validate/fundamentals/` (2,891 LOC)
imports from `src.constants` at all. `field_map.py` is the sole consumer of **16**.

- [ ] Census first, in a scratch script: for each of the 47, list every `src/` file that reads it.
      Relocate only those with exactly one. Print the census table into the Phase 7 report.
- [ ] Destination = the single consumer. The 16 `field_map.py`-only symbols go to `field_map.py`.
- [ ] Update the importing tests. 21 test files import from `src.constants.constants`; the ones
      affected here are `test_cik_cutover.py:24` (`FUNDAMENTALS_FORMS` — but that has 2+ src
      consumers, so it **stays**) and the sharadar trio
      (`test_fetch_sharadar.py:19`, `test_sharadar_diagnostics.py:26`, `test_sharadar_field_map.py:25, :173`,
      `test_sharadar_merge.py:390`). Import from the new home; do not re-export from
      `constants.py` for backwards compatibility — a shim defeats the whole exercise.
- [ ] Leave the 6 hand-maintained Sharadar vocabulary subsets (`SHARADAR_SF1_COLUMNS:141` ⊃
      `SHARADAR_ID_COLUMNS:168`, `SHARADAR_ZERO_FILLED_FIELDS:187`, `SHARADAR_FLOW_FIELDS:288`,
      `SHARADAR_NON_NEGATIVE_FIELDS:308`, `SHARADAR_EVENT_FIELDS:370`,
      `SHARADAR_DIAGNOSTIC_EXTRA_COLUMNS:278`) **where they are** — 91 definition lines over one
      112-name vocabulary, read by 2+ files each. Deriving the subsets from one declaration is a
      genuine improvement and a genuine risk; record it as a follow-up, do not do it here.
- [ ] **Formatting**: `constants.py` is hand-formatted. A `json.dumps` round-trip or an
      auto-formatter reflows the whole file and makes the diff unreadable. Edit by text splice
      only, and confirm `git diff` shows only the intended blocks.

## 6.4 The reverse violations on this path

Literals hardcoded where the standard says otherwise. Fix the ones in `fundamentals/**` and its
direct helpers; leave the rest to the later sweep.

- [ ] **Table names as constants — explicitly forbidden.** `fetch_financial_notes.py:70-71`
      `_NUM_TABLE` / `_TXT_TABLE`, `fetch_financial_statements.py:42`
      `_TABLE = "pension_facts"`, while `Tables.pension_facts` exists at `schema.py:282`.
      (Also listed in Phase 5.4 — do it once.)
- [ ] **Form lists restated three times**: `build_history.py:49 FORM_PRECEDENCE`
      (= `FUNDAMENTALS_FORMS`), `fetch_fundamentals_sec.py:176 _ANNUAL_FORMS`,
      `fundamentals_employees.py:46 HEADCOUNT_FORMS` (**byte-identical** to the previous). Import
      `FUNDAMENTALS_FORMS`; keep a local only where the *order* carries meaning
      (`FORM_PRECEDENCE` may) — and if it does, say so in one line and derive it from the shared
      list rather than re-typing the members.
- [ ] `DATE_FORMAT_COMPACT = "%Y%m%d"` re-typed inline at `fetch_financial_notes.py:190, :203,
      :219` and `fetch_financial_statements.py:75, :92`. Import it. (8 sites repo-wide; fix these 5.)
- [ ] SEC URL templates defined locally: `fetch_financial_notes.py:73`. Move to `constants.py` —
      `validate/fundamentals/finding.py:76` and `fetch_fails_to_deliver.py:35-36` also carry
      copies, so this one has 2+ readers and qualifies under the new rule.
- [ ] **Bounds and tolerances local while `constants.py` twins sat unused.** Now that the twins
      are deleted (6.2), the local ones are the only definition — which is *correct* under the
      new rule. So: **keep** `build_history.py:74-79 HARD_GUARDS`, `:393
      GROSS_PROFIT_IDENTITY_TOLERANCE`, `:53 MAX_AMENDMENT_LAG_DAYS`, `:269 TTM_STALENESS_DAYS`,
      `periods.py:172, :175, :232`, and delete the unused `constants.py` twins
      (`SHARES_OUTSTANDING_MIN/MAX`, `Q4_RECONCILIATION_TOLERANCE`, ...). Record in each
      docstring that this is now the single definition.
      **Exception**: `TTM_STALENESS_DAYS` is imported by `build_ttm.py:49`, whose `:34` says
      "IMPORTED rather than restated" — that is 2 consumers, so it **stays** in `constants.py`.
      It is the one place the pattern is already right; do not break it.
- [ ] `PeriodGuards` (`periods.py:58-67`): **two of three field names differ from their YAML
      keys** (`max_opposite_sign_ratio` <- `max_opposite_sign_q4_ratio`,
      `concept_switch_scale_max` <- `q4_tag_mismatch_fy_max`), with the mapping living only in
      the constructor at `:75-77`. Rename the dataclass fields to match `configs/configs.yml`
      exactly and delete the mapping. **`test_periods_q4.py:33` pins the three exact field
      names** — update it in the same commit.
- [ ] `periods.TTM_QUARTERS = 4` duplicates `src/utils/quarters.py:21 QUARTERS_PER_YEAR = 4`,
      which exists precisely because "three unrelated subfolders need it". Import it.
- [ ] `periods.TTM_MIN_DAYS / TTM_MAX_DAYS = 330, 400` restate the `ANNUAL` band in
      `_DURATION_BANDS:91` — same two literals, same file. Derive one from the other.
- [ ] `FUNDAMENTALS_DISCONTINUITY_MIN/MAX:954` self-declares as a clone of
      `HEADCOUNT_CONTINUITY_*`; `5.0` carries four different names
      (`OPERATING_MARGIN_ABS_MAX`, `PROFIT_MARGIN_ABS_MAX`, `HEADCOUNT_CONTINUITY_MAX`,
      `FUNDAMENTALS_DISCONTINUITY_MAX`). Most are in the dead-46 bucket; for any survivor, keep
      **one** name and reference it. Do **not** collapse two thresholds that merely happen to
      share a value today — that is how a tuning change silently moves an unrelated gate.

## 6.5 Merge the duplicated helpers

### Inside the fundamentals family

| Keep | Delete / redirect | Evidence |
|---|---|---|
| one as-of join in `src/utils/` | `build_history.carry_latest_known:224-256` + `merge_history._asof_join:272-290` | both docstrings give the same reason **verbatim**; `merge_history.py:45` names the primitive. Phase 3 already vectorised one — unify on the vectorised version. |
| one same-day collapse | `build_history._collapse_same_day:204-219` + `merge_history.collapse_same_date:255-269` | `merge_history.py:259` states the relationship |
| one dtype-pinning helper | `build_history:825-849` + `merge_history._cast:462-478` | near line-for-line: `:470-471`≡`:825-827`, `:474`≡`:836-837`, `:475-477`≡`:846-848` |
| the **vectorised** TTM | `periods.trailing_twelve:793-854` (python row loop) vs `build_ttm.py:155-161` (vectorised `rolling`) | `build_ttm.py:17` says so explicitly. **Careful**: `trailing_twelve` also emits `basis`, `known_from`, `n_quarters` and `dc_code`, and refuses on `_window_is_contiguous` / `_one_share_basis`. Unify only if the vectorised version can carry all of that. If it cannot, **say so and leave both**, with a comment naming the difference. |
| one contiguity test | `periods._window_is_contiguous:887-897` (python loop) vs `build_ttm._window_is_whole:64-75` (vectorised) | same contract, two implementations |
| one cutover filing walk | `cik_cutover.cutover_filings:130` vs `edgar_driver.new_filings:44` | `cik_cutover.py:141`: "**Mirrors** `edgar_driver.new_filings` — same dedup, same `since` filter, same ordering" |
| one entries comprehension | `kpi_catalogue._data_items:571-573` vs `field_map._entries:174-176` | byte-for-byte identical |

### Across `data_extract`, restricted to the fundamentals path

- [ ] **unzip + read member**: `read_zip_member:103`, `read_zip_members:126`, `read_zip_text:153`
      already exist in `bulk_cache`, yet `fetch_financial_notes.py:258-269` and
      `fetch_financial_statements.py:106-119` open `zipfile.ZipFile` directly, each
      re-implementing case-insensitive member lookup — one with `.lower()`, one with `.upper()`.
      Use the shared helpers. (`fetch_insider_transactions.py:203-213` is the third copy; it is
      called from `StepExtractFundamentals`, so fix it too.)
- [ ] **corrupt-zip self-heal**: `bulk_cache._drop_corrupt:92-100` exists and its docstring
      claims closure, but **3 private copies remain** — `fetch_financial_notes.py:270-273`,
      `fetch_insider_transactions.py:214-217`, `fetch_financial_statements.py:120-123`. Delete
      the copies and fix `bulk_cache.py:96-97`'s claim, which is currently false.
- [ ] **cik -> ticker inverse dict**: byte-identical 3 lines at `fetch_financial_notes.py:286-287`,
      `fetch_insider_transactions.py:226-227`, `fetch_financial_statements.py:135-136`. One helper.
- [ ] **CIK zero-padding**: `src/utils/string.py:16 pad_cik` exists and is ignored by 7 sites;
      3 of them drop the `.0` strip or the empty-string guard (`sec_utils.py:137` is the only
      complete one). Fix the sites on this path.
- [ ] **quarter/period generation**: `bulk_cache.quarter_periods:195-203` exists;
      `fetch_financial_notes._generate_periods:152-165` re-derives the same
      `max(FIRST_YEAR, today.year - years)` clamp. Use the shared one.
      (`fetch_fails_to_deliver._periods:42-57` is the third copy — out of scope, note it.)
- [ ] **JSON config loading**: 6 inline implementations with **4 different missing-file
      behaviours** (raise / `{}` / uncaught / explicit `exists()`). Add one
      `utils/json_config.load_json(path, *, required: bool)` and use it for the fundamentals
      readers (`kpi_catalogue:599`, `cik_cutover:90`, the roster readers). Pick **raise** as the
      default: a silently-empty catalogue is worse than a crash.
- [ ] **ticker normalisation**: 5 copies of `strip().upper()`. One `normalise_ticker` in
      `utils/string.py`.
- [ ] `common/form_registry.py` (115 LOC) is imported by **nothing** in `src/`; its own docstring
      (`:9-17`) says it is "consulted by tests and future orchestration work", and
      `form_registry.py:4` cites `schema_registry.TableSpec`, superseded per `schema.py:5`.
      **Decide and record**: either wire it into the form-list consolidation above (6.4) or delete
      it. Do not leave a third state.

## Verification

- [ ] `rtk "$PY" -m pytest tests/ -v -s` — the **full** suite. Relocating a constant can break
      any importer, and 21 test files import from `constants.py`.
- [ ] Phase 0 harness: **0 differing cells**. A form list or a tolerance that changed value while
      being "deduplicated" shows up here.
- [ ] `wc -l src/constants/constants.py` -> **< 720** (from 1,058).
- [ ] Re-run the consumer census: **0** symbols with zero `src/` consumers, **0** symbols in
      `constants.py` with exactly one non-test consumer *inside `fundamentals/`*.
- [ ] Module-global count in `fundamentals/**`: **< 60** (from 124). Count with the same method
      the research used (module-level `UPPER`/`_UPPER` assignments) and report both numbers.
- [ ] `rtk grep -rn "zipfile.ZipFile" src/data_extract/` -> 1 hit (`bulk_cache.py`).
- [ ] `rtk grep -rn "strip().upper()" src/data_extract/` -> 1 hit (the helper).
- [ ] `git diff --stat src/constants/constants.py` shows deletions concentrated in the blocks
      named above, not a whole-file reflow.

## Risks

| Risk | Mitigation |
|---|---|
| A "dead" symbol is read dynamically | Verified: no `import *`, no `getattr(constants, ...)` repo-wide. Still grep `app/`, `main.py`, `configs/`, `sql/` before each delete. |
| Two thresholds that share a value get collapsed | Only collapse where the code *says* one is a clone of the other (`FUNDAMENTALS_DISCONTINUITY_*:954` does). Otherwise keep both names. |
| Unifying `trailing_twelve` with `build_ttm` loses `basis`/`dc_code`/refusals | Explicitly allowed to abandon that one merge and document why. The Phase 0 harness would catch it, but the honest outcome is "left as two, here is the difference". |
| `PeriodGuards` rename breaks `test_periods_q4.py:33` | Same commit. It is a known pin, listed in the research. |
| A relocated constant creates a circular import | Destination is always the single consumer, so the import direction cannot invert. If it does, the symbol had 2+ consumers and should not have moved. |
| Whole-file reflow of `constants.py` | Text splices only; check `git diff --stat`. |
