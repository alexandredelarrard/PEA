# Phase 1 — Live defects, dead code, silent tripwires ✅

**Goal**: no live bug, no unreachable symbol, no test that fails by skipping. Land before any
optimisation so the Phase 0 baseline is not built on a `NameError`.

## 1. The three defects

### 1a. `cols` is undefined — a `NameError` that zeroes whole tickers

`xbrl_linkbase.py:514` returns `pd.DataFrame(columns=cols)`. The three sibling early-returns
(`:439`, `:441`, `:504`) correctly use `ARC_COLUMNS`. It fired for **NEM, MO and AIZ** in the
run that started 2026-08-27 00:06; each produced **zero facts** while the run reported success.

- [x] `xbrl_linkbase.py:514`: `cols` -> `ARC_COLUMNS`.
- [x] Test: `tests/data_extract/fundamentals/test_linkbase_empty_arcs.py` — 2 tests, both
      pass. Driven with a filing whose arcs are all note-only (the actual `:514` condition is
      "no arc SURVIVED the filter", not "no arcs"), plus the other three empty returns pinned
      to the same shape. **Confirmed to fail at HEAD with the real
      `NameError: name 'cols' is not defined` at `xbrl_linkbase.py:514`.**

### 1b. The swallow that turned a `NameError` into a benign warning

The reason a one-word bug survived 10 hours in production is the error path, not the typo:

- `parallel_fetch.py:41-44` **requires** the worker to catch its own exceptions.
- `filing_rows` (`fetch_fundamentals_sec.py:660-663`) has a bare `except Exception` that reports
  "unreadable XBRL" and returns `[]`.

So a programming error is indistinguishable from a genuinely malformed filing.

- [x] In `filing_rows`, keep swallowing **data** failures per filing (one bad filing must not
      kill a ticker — that convention stays), but **re-raise** the programming-error classes:
      ```python
      except (NameError, AttributeError, TypeError, KeyError, ImportError):
          raise                                   # our bug, not the filer's
      except Exception as exc:                    # noqa: BLE001 -- malformed filing
          failures.append((accession, str(exc)))
      ```
      `KeyError` is in the list deliberately: on this path it means a column contract broke.
- [x] Done as an optional `failures: list[tuple[str, str]]` out-param on `filing_rows` (its
      `-> list[dict]` return is consumed by 4 tests), collected per ticker in
      `build_ticker_fundamentals` and logged at WARNING with the accession list.
      Phase 5 carries it into `extraction_run.tickers_failed`.
- [x] A ticker that produced **0 facts from >= 1 filing** is logged at ERROR, not silence. That
      single line is what would have caught this in hour one.
- [x] Test: `tests/data_extract/fundamentals/test_filing_rows_error_classes.py` (6 tests) at
      the per-filing layer, and
      `test_edgar_driver.py::test_run_edgar_fetch_reraises_a_programming_error_instead_of_warning`
      at the per-ticker layer. Both pass.

**⚠ Plan mismatch, and it widened the edit.** `filing_rows`' `try` covers only
`filing.xbrl()`, and it logs nothing — so it is **not** where the `NameError` was swallowed.
The resolver runs in `rows_from_xbrl`, OUTSIDE that `try`, so the exception propagated to
`edgar_driver._worker`'s blanket `except Exception` (`edgar_driver.py:111`), which logged
`"fundamentals: NEM failed (name 'cols' is not defined)"` at WARNING and dropped the **whole
ticker** — which is why NEM/MO/AIZ have 0 rows rather than one gap each. Fixing only
`filing_rows` would have left the live swallow untouched and failed this phase's own
verification, so `PROGRAMMING_ERRORS` is defined once in `edgar_driver.py` and honoured at
**both** layers (the other 4 fetchers on that driver inherit it).

The two `except`s are deliberately asymmetric: `filing.xbrl()` is edgartools parsing the
filer's XBRL, so it still swallows **every** class — narrowing it would turn one malformed
submission into an aborted 490-ticker run. Only our own `rows_from_xbrl` re-raises.

### 1c. `fetch_earnings_surprises`' empty branch always crashes

`fetch_earnings_surprises.py:149` calls `record_run(..., 0)` on the `if not parts:` branch with
**no `return`**, then falls through to `:154-156`.

Verified at implementation-planning time, and it is worse than the double-record the research
reported: `not parts` means `existing` is empty **and** `new_frames` is empty, so `:153` takes the
`else pd.DataFrame()` path and `:156`'s `new['ticker'].nunique()` raises **`KeyError: 'ticker'`**
on a column-less frame. The branch is not "records twice" — it is **unreachable-without-crashing**,
which is why nobody noticed the double record.

- [x] Add the missing `return` after `:149`.
- [x] Test: `test_fetch_earnings_surprises.py::test_the_no_data_branch_returns_after_recording_exactly_one_run`.
      Passes; prints `record_run called 1 time(s): [('earnings_surprises', 2, 0)]`. The
      research's `KeyError: 'ticker'` diagnosis is confirmed — the test fails at HEAD with it.

## 2. Dead code — delete

All verified in research as having no reference in `src/`, `tests/` or `scripts/` beyond their
own definition. Delete the symbol and any comment block that exists only to explain it.

| File | Symbols | Outcome |
|---|---|---|
| `xbrl_linkbase.py` | `ONLY_WHEN_SIBLING` (`:839`) — only `ONLY_WHEN_DESCENDANT` is ever compared (`:890`) | **deleted.** Verified dead by NAME *and* by VALUE: `"not_a_declared_sibling"` appears in no config, because the sibling test is the `else` branch and has nothing to declare. Its doc block is kept (it documents the live default) and its first line now says so. |
| `entity_scope.py` | `us_gaap_only` (`:142`), `dimensioned_facts` (`:232`), `ENTITY_AXES` (`:39`) | **all three deleted.** `_DIM_PREFIX` stays — still read at `:106`. `DIMENSIONED_EXCEPTIONS` stays: it has a test. |
| `kpi_catalogue.py` | `Catalogue.regime_for_sub_industry` (`:365`) | **deleted** — a thin wrapper over `regime_for_gics(sub_industry=…)`, which superseded it. Its measurement (the 4 forced overrides, 37 live tickers) moved onto `regime_for_gics`' docstring rather than dying with it. |
| `kpi_catalogue.py` | ~~`Kind` (`:61`), `Sign` (`:62`), `EXTRACTED_KINDS` (`:66`), `SCORED_TIERS` (`:71`)~~ | **⚠ NOT DEAD — kept.** `Kind`/`Sign` annotate `FieldSpec.kind`/`.sign` (`:81-82`); `EXTRACTED_KINDS` is read by `is_extracted` (`:106`); `SCORED_TIERS` by `is_scored` (`:91`). |
| `reason_codes.py` | ~~`COMBINED_INTO` (`:68`), `BASIS_EX_IPRD` (`:97`)~~ | **⚠ NOT DEAD — kept.** Both are members of `ALL_CODES`, the closed set `build_history` asserts every written row against; `BASIS_EX_IPRD` is also in `IS_QUALIFIER`. Deleting `BASIS_EX_IPRD` would make the code the catalogue *does* emit illegal and **fail that assertion** — i.e. it would have injected a live defect. This is exactly the plan's own "no external importer is their design" caveat, one paragraph later. |
| `context.py` | `DEF14A_LLM_PATH` (`:45`), `SEC_13F_INSIDERS_DIR` (`:49`) | **deferred to Phase 5**, taking the option this row already offers. Risk zone, and Phase 5 rewrites the file anyway; nothing else in Phase 1 depends on it. |

- [x] `Catalogue.combined_into` (`:514`) returns `None` for every cell today and says so at
      `:524`. **Kept**, as instructed — and so is `COMBINED_INTO`, per the row above: the
      constant is what makes the code this method can produce a LEGAL one.
- [x] Did **not** delete the private module constants that only look unimported
      (`_VALUE_KEY`, `_QUARTER_COLUMNS`, `PERIOD_COLUMNS`, `FACT_COLUMNS`, `AS_REPORTED`,
      `Q2_FROM_YTD6`, `TTM_MIN_DAYS`, ...). "No external importer" is their design. They are
      handled in Phase 6/7 as globals-reduction, not as dead code.

## 3. Name collisions — rename

Two modules that import each other define the same private name with different types; two
`_latest_per_window`s have **different window identities**.

- [x] `_QUARTER_COLUMNS` -> `_QUARTER_LABEL_COLUMNS` in `build_history.py` (2 sites), with the
      difference from `periods._QUARTER_COLUMNS` named in its comment.
- [x] `_normalise` -> `_normalise_facts` in `build_history.py` (def + 1 call + 1 comment
      reference). The other three left alone.
- [x] The validator's `_latest_per_window` -> **`_latest_vintage_per_period_end`** (9 sites in
      `tier3_internal.py`) — its actual identity, chosen over the plan's suggested
      `_latest_per_as_of_window` because it groups on an EXACT `period_end` per
      `duration_type`, with no `as_of` in it. Its docstring now names the difference from
      `periods._latest_per_window` (which buckets ends within `_SAME_PERIOD_DAYS`). Not
      unified.

## 4. The three silent-skip test pins

`pytest.importorskip` with a **dotted string**: a rename turns the test into a *skip*, not a
failure.

- [x] `test_fundamentals_employees.py:62` and `:170` — replaced with top-level imports of
      `periods.instant_stock` and `build_history.carry_latest_known`. The module docstring's
      "they skip until their module lands" paragraph is rewritten: the modules have landed.
- [x] `test_fundamentals_point_in_time.py:173` — same, `build_history.build_ticker_history`.
- [x] Both files: **9 tests, 9 passed, 0 skipped.**
- [x] `test_financial_notes.py:168, :174` — **left alone; the premise is wrong.**
      `monkeypatch.setattr(obj, "name", value)` defaults to `raising=True`, so patching a
      renamed attribute already raises. Measured, not reasoned: patching
      `_scrape_available_periods_RENAMED` on that module fails with
      `AttributeError: <module …fetch_financial_notes> has no attribute
      '_scrape_available_periods_RENAMED'`. The proposed `fn._scrape_available_periods.__name__`
      is the same string resolved one step later and would gain nothing. A rename fails at
      **run** time rather than at collection, which is the only residual gap and is not worth a
      change that pretends to close it.

## 5. `Step._log` attribution

`step.py:19` binds `_log` to `src.utils.step`, so every step's log lines are attributed to
`step.py`. `self._log` is used **twice** in all of `src/data_extract/`
(`step_extract_all_data.py:45, :47`); the 5 transformer steps log nothing.

- [x] `utils/step.py`: `self._log = logging.getLogger(type(self).__module__)`. Risk zone —
      approved in this plan. One line; no signature change. `configs/logging.yml` names no
      per-module logger (only `development`/`staging`/`production` + `root`), so nothing was
      routing on the old name.
- [x] `step_extract_fundamentals.py`: a `_stage()` context manager logs one INFO line entering
      each of the **5** sub-fetchers and one leaving it with its wall clock (`_elapsed`:
      `1h 02m 05s`). The `finally` means a stage that RAISES still reports how long it ran.
- [x] Confirmed on real `LogRecord`s: `name='src.data_extract.transformers.step_extract_fundamentals'`
      (the plan's path predates the `transformers/` move), was `src.utils.step`. Sample line:
      `probe stage: starting, 3 ticker(s)` / `probe stage: done in 0s`.

## 6. Stale self-paths and dangling references (the mechanical subset)

The prose pass is Phase 7; these are wrong *facts* and are cheap to fix now.

- [x] 5 wrong self-paths on line 2 fixed. `edgar_fillings.py` also had its header NAME wrong
      (it said `edgar_filings.py`, one `l`, which is not the file) — corrected to match, since a
      reader following either half of that line landed nowhere.
- [x] `cik_cutover.py:81-88` -> `tests/data_extract/fundamentals/test_cik_cutover.py`.
- [x] `build_history.py:460` now names `data/total_liabilities_legs.json` (present, 23.7 KB) as
      the artefact of record and says the script is gone. **⚠ Plan mismatch:** there is no read
      to keep — `:460` is a docstring, and `grep -rn total_liabilities_legs src/` returns only
      that citation and one in `src/validate/README.md`. Nothing in the code opens that JSON.
- [x] `constants.py:895`: the dangling `_TO_COMMON_TOL` / `fetch_fundamentals.py` precedent is
      replaced by what the 2% actually is. Comment-only, in a risk zone; Phase 6 owns the rest of
      the file.
- [x] `fetch_financial_statements.py:17-19` now says the footnote detail is wired, and where
      (`fetch_financial_notes.py` -> `notes_num` / `notes_text`).
- [x] `run_manifest.py:6` -> "the fetchers the five `step_extract_*` sub-steps call".
- [x] `fundamentals_employees.py:89-93` now names `fundamentals_employees` as the seed and the
      `(ticker, as_of, employees)` -> `(ticker, filing_date, value)` rename the caller does.
- [x] `docs/coding_standard.md:18` **left for Phase 6**, as this item itself instructs.

## Verification

- [x] `rtk "$PY" -m pytest tests/data_extract/fundamentals -v -s` -> **223 passed, 0 failed,
      0 skipped** (37m 11s). The tree had grown past the plan's 201: baseline was **214**
      (213 passed + 1 spurious failure, below) and this phase adds **9** tests -- 2 empty-arcs,
      6 filing-rows error-class, 1 earnings-surprises empty branch. **Skip count 0 -> 0**: the
      three `importorskip` pins were never actually skipping (their modules exist), so the value
      of unpinning them is entirely prospective -- exactly what that section argues.

      One thing later phases need to know: `test_segment_margin_876ab8a57bd8.py::
      test_operating_income_is_deliberately_not_derived` asserts on
      `inspect.getsource(bh._snapshot)`, which re-reads `build_history.py` from disk by the code
      object's line numbers. **Editing that file while a suite is in flight fails that test
      spuriously** (it asserted against `'        return None'`). It passes in every clean run.
      Do not chase it in Phases 2-4; just do not edit under a running suite.
- [x] Phase 0 harness, **tier B** (8 tickers, FULL history -- 6,995 fact rows for MCD, 517
      publication events in total). **PASS — 0 differing cells, 0 dtype drift, 0 code deltas**:
      APA/BA/BAC/BRK-B/MCD 69 rows, KR 70, ORCL 68, VRT 34. Baseline replayed at pristine HEAD
      in the isolated export, after-replay in the working tree, same frozen parquet inputs.
- [x] Phase 0 harness, **db mode** (`compare_against_stored`, read-only, run alongside the
      in-flight walk) on VRT + MCD -- the same two Phase 0 measured, so the numbers are
      directly comparable. **Reproduced Phase 0 exactly: VRT 0 drifted cells, MCD 38.** Those 38
      are the pre-existing staleness in stored `fundamentals_history_sec` that Phase 0 already
      recorded; this phase added none. The moving-target guard
      (`verify_live_matches_manifest`) returned `moved=[]` -- the 8 sample tickers' facts have
      not moved since the freeze, even though the live table grew from 472 to 478 tickers
      during this phase.
- [x] Phase 0 harness, **tier A** (8 tickers x 16 filings), re-frozen from the live DB (472
      tickers / 2.87 M rows at 16:50; VRT capped at 16 -> 1,509 rows, matching Phase 0's
      calibration exactly). Baseline replayed at **pristine HEAD in an isolated `git archive`
      export**, not in the working tree, so no edit could contaminate it.

      **PASS — 0 differing cells, 0 dtype drift, 0 reason codes added or removed, all 8
      tickers**: APA 14->14 rows, BA/BAC/BRK-B/KR/MCD/ORCL/VRT 16->16.
- [x] `columns=cols` in `xbrl_linkbase.py` -> 0 hits. The literal check as written is **not
      the right check**: `src/` has 13 other `columns=cols` sites where `cols` is a real local
      (`price_frames.py:146`, `outliers.py:254`, ...). Replaced with an AST scan for names loaded
      but never bound in any enclosing scope; `src/data_extract/utils/fundamentals/` now reports
      only closure false-positives (`spec`/`graph` in nested functions), and no `cols`.
- [x] `importorskip` in `tests/` -> **1 hit, and it is prose**: the sentence in
      `test_fundamentals_employees.py`'s docstring explaining why the pins are gone. No call
      sites remain. The now-unused `import pytest` came out of that file with them.
- [x] Planted `raise NameError("name 'cols' is not defined")` at the top of `statement_arcs`,
      ran the real CLI, reverted in a `finally` (verified byte-identical after). **Third attempt
      is the one that counts, and the first two are a finding:**

      1. `-t VRT -F` -> `1/1 ticker(s) ok, +0 rows` in 4 s. Every VRT accession is already
         stored, so `done_accessions` emptied the walk and the resolver was never entered.
      2. `-t NEM` (no `-F`) -> same, 3 s. **`manifest_window` is per-TABLE, not per-ticker**, so
         `since` was today and NEM listed nothing — which is exactly why NEM/MO/AIZ will NOT
         self-heal on a nightly run. They need `-F`. (Feeds the plan's decision-3 note below.)
      3. `-t NEM -F` -> **exit code 1**, full traceback through
         `fetch_fundamentals_sec.py -> ArcGraph(statement_arcs(xbrl)) -> NameError` on the
         console, and **no** `"NEM failed"` warning line. Nothing was written: NEM/MO/AIZ are
         still at 0 rows in `fundamentals_facts`.

## Side effects of this phase's verification (recorded, not hidden)

- The two single-ticker CLI runs that COMPLETED (`-t VRT -F`, `-t NEM`) each called
  `record_run`, so `data/extraction_manifest.json` now reads `ticker_count: 1, rows_added: 0`
  for `fundamentals_facts` and `fundamentals_employees`. `manifest_window` compares that count
  against the universe size, so the **next** run takes the full years-history window and marks
  itself a full rescan. That is the conservative direction (more work, no data loss) and the
  in-flight walk overwrites it with the true count when it finishes.
- The third run (`-t NEM -F`, the planted `NameError`) wrote **nothing** and recorded nothing:
  the exception propagates before any `save` and before `record_run`. Confirmed after the fact
  — AIZ/MO/NEM are still at 0 rows.

## Notes

- The `cols` fix cannot help the in-flight run — that process already imported the module,
  **and it is still running** (distinct tickers in `fundamentals_facts`: 402 at 12:05, 472 at
  16:50, 478 at 18:57). Coverage remediation is decision 3: **measure after the run, decide
  then**. The detector query, run at 18:56, returns **22 tickers, not 3**:

  ```
  AIZ, FDXF, GEHC, GEV, HONA, KVUE, MO, NEM, Q, SNDK, SOLV, VLTO,
  WTW, WY, WYNN, XEL, XYL, XYZ, YUM, ZBH, ZBRA, ZTS
  ```

  Only **AIZ, MO, NEM** are `cols` victims. The other 19 are mostly alphabetically late (W-Z)
  or newly-listed/renamed symbols, which is what an unfinished walk looks like — so the query
  must be re-run once the walk stops, before anything is concluded from it. Two further facts
  the planted-`NameError` runs turned up, both of which shape the remediation:
    * `manifest_window` keys on the **table**, not the ticker, so a nightly run will never
      re-list a zero-fact ticker. Remediation needs `-F`.
    * A single-ticker `-F` on an already-walked ticker walks **nothing** — `done_accessions`
      empties it. So `-F` only helps the tickers that are genuinely empty, which is exactly
      the set above.

  The detector query for the Phase 7 report:
  ```sql
  SELECT t.ticker FROM sp500_tickers t
  LEFT JOIN (SELECT DISTINCT ticker FROM fundamentals_facts) f USING (ticker)
  WHERE f.ticker IS NULL;
  ```
- Live bug #2 (`employees` vs `employees_sec`) is **out of scope** — see [deferred.md](deferred.md).
