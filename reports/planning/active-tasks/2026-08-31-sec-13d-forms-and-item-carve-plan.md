# Implementation Plan: sec_13d — post-mandate form types, Item 3/4 carve, text normalization

**Date Created**: 2026-08-31
**Planning Phase**: 2 of 3 (FIC Workflow)
**Based on Research**: in-session research 2026-08-31 (probes 1–16, all measurements inline below)
**Next Phase**: Implementation (`/implement`)

## Overview

`sec_13d` stopped ingesting on **2024-12-16** and every row since is missing. The cause is not a
parser failure: EDGAR renamed the form type at the structured-XML mandate, and
`SEC_13D_FORMS` still lists only the pre-mandate strings, so the listing returns nothing and no
parser ever runs.

Independently, the legacy regex carve has a silent corruption bug: when the Item 4 anchor fails,
Item 3 loses its end boundary and **swallows Item 4's entire body**.

This plan fixes both, adds a guard for a false-zero the modern XML path would otherwise
introduce, and applies only the zero-risk text normalization.

## Current State Analysis

| Fact | Measured value |
|---|---|
| `sec_13d` max `filing_date` | **2024-12-16** (1,666 rows, min 2011-08-22) |
| Filings missed across S&P 500 | **461**, 91 tickers, 2024-12-17 → 2026-08-28 |
| `rp_seq` distinct values in the table | **1** (only `0`) — co-filers never captured |
| `date_of_event` non-null | **0 / 1666** |
| `percent_of_class` non-null | **0 / 1666** |
| item3 bodies contaminated with Item 4 | **4.0%** originals, **6.8%** amendments |

**Root cause of the outage** — EDGAR form strings changed at the mandate:

| Era | Form string |
|---|---|
| ≤ 2024-12-16 | `SC 13D`, `SC 13D/A` |
| ≥ 2024-12-17 | `SCHEDULE 13D`, `SCHEDULE 13D/A` |

`src/constants/constants.py:72` lists only the first pair. `new_filings()`
(`edgar_driver.py:44`) calls `Company(ticker).get_filings(form=SEC_13D_FORMS)`, which matches
exactly — so post-mandate filings are invisible to the fetcher.

**The modern extraction path already works.** `_filing_rows`'s `if has_structured and items:`
branch (`fetch_13d_edgar.py:310`) was run unmodified against real 2025–2026 filings: it returns
`has_structured_data=True`, multiple reporting persons, and full numerics. No new extraction
code is needed.

**Key files**
- `src/constants/constants.py` — `SEC_13D_FORMS`
- `src/data_extract/utils/structure/fetch_13d_edgar.py` — anchors, carve, `_num_or_null`, `_COLS`
- `src/data_extract/cli.py` — `sec_13d` command (no `-F/--full` yet)
- `sql/schema.sql:765` — `sec_13d` DDL
- `tests/data_extract/structure/test_fetch_8k_13d_edgar.py`

## Desired End State

1. `sec_13d` ingests SC 13D **and** SCHEDULE 13D filings, current to today.
2. Item 3 is never contaminated by Item 4's body (measured 0%).
3. Item 4 coverage rises 92.9% → 98.9% on originals, 61.0% → 67.0% on amendments.
4. No row claims a 0% stake that the filer did not actually disclose.
5. Item bodies are free of cp1252 mojibake, box-drawing rules and ragged whitespace.
6. A documented (not executed) rebuild procedure to replace the 1,666 corrupt legacy rows.

## Out of Scope

Explicitly excluded per the scoping decision — the measurement showed a **1.9–2.6%** drop in the
embedding similarity noise floor for the whole cleaning suite, which does not justify the regex risk:

- Legal-boilerplate sentence stripping ("incorporated herein by reference", "is hereby amended…").
- Cover-page row stripping (safe at 0.5% false positives, but only ~1% incremental benefit; affects
  29.6% of item5 bodies — revisit separately if item5 embeddings underperform).
- Page furniture removal (`- 4 -`, `CUSIP No. …`, `Page 3 of 12`) — low risk, not zero.
- Nulling pure-boilerplate bodies (35/1,195 = 2.9% fall under 30 chars when fully cleaned).
- 13G (passive) filings — deliberately excluded from this table.
- **Executing** the rebuild against live Postgres (code + tests only; runbook documented in Phase 5).

---

## Implementation Approach

### Phase 1: Post-mandate form types + `--full` threading ✅

**Goal**: The fetcher can see filings after 2024-12-16.

**Changes**:

1. `src/constants/constants.py`:
   - [x] Extend `SEC_13D_FORMS` with the post-mandate strings and record *why* both eras exist.
   ```python
   # EDGAR renamed the form type at the structured-XML mandate: filings through 2024-12-16 are
   # "SC 13D", filings from 2024-12-17 are "SCHEDULE 13D". `get_filings(form=...)` matches
   # EXACTLY, so dropping either pair silently truncates the table at the changeover -- measured:
   # 461 filings across 91 S&P 500 tickers were invisible until both pairs were listed.
   SEC_13D_FORMS = ["SC 13D", "SC 13D/A", "SCHEDULE 13D", "SCHEDULE 13D/A"]
   ```

2. `src/data_extract/utils/structure/fetch_13d_edgar.py`:
   - [x] Thread `full` through `fetch_13d_edgar` into `run_edgar_fetch` (which already accepts it).
   ```python
   def fetch_13d_edgar(context: Context, tickers: list[str], years_history: int,
                       full: bool = False) -> None:
       run_edgar_fetch(context, tickers, years_history,
                       tables=(Tables.sec_13d, Tables.sec_13d_transactions),
                       build=build_ticker_13d_edgar, desc="SC 13D (edgartools)", full=full)
   ```

3. `src/data_extract/cli.py`:
   - [x] Add `@click.option(*_FULL_ARGS, **_FULL_KWARGS)` to the `sec_13d` command and pass `full=full`.
     Needed because after the Phase 5 DELETE the manifest still holds a recent run date, so an
     incremental run would fetch nothing.

**No change needed** to `form_registry.py` or its test — both derive from `SEC_13D_FORMS`.
**No schema change** for this phase: `is_group_member` is already `TEXT`, `date_of_event` already `DATE`.

**Verification**:
- [x] `Company("CVNA").get_filings(form=SEC_13D_FORMS)` returns filings dated after 2024-12-16.
- [x] `"$PY" -m pytest tests/data_extract/common/test_form_registry.py -v -s`
- [x] Spot-check one modern filing end-to-end: `_filing_rows` returns >1 reporting person with a
      non-null `percent_of_class` and `date_of_event`.

---

### Phase 2: Item 3/4 carve — line-anchored headings with a safe fallback ✅

**Goal**: Eliminate item3 contamination; raise item4 coverage. Measured, zero regressions.

**Three failure modes to fix** (all characterized on real filings):

| Mode | Example | Current anchor result |
|---|---|---|
| Caption says "Purpose of **the** Transaction" | PSA, CVNA, EXPE | no match |
| Captionless bare `Item 4.` heading | FSLR 2016 | no match |
| Caption padded past the 8-char separator budget | AKAM, UAL | no match |

When item4's anchor misses, item3's body runs to the next matching anchor — MNST's `item3` is
**17,776 chars** where the true body is 850.

**Changes**:

1. `src/data_extract/utils/structure/fetch_13d_edgar.py`:
   - [x] Keep `_ITEM_ANCHORS` **exactly as it is today** — it is the measured fallback, and
         widening it in place would invalidate the measurement.
   - [x] Add captions + a line-anchored anchor set:
   ```python
   #: Caption keyword per item, widened where filers measurably diverge from the SEC's own
   #: wording -- "Purpose of THE Transaction" alone accounted for most Item 4 misses.
   _ITEM_CAPTIONS: dict[int, str] = {
       1: r"security\s+and\s+(?:the\s+)?issuer",
       2: r"identity\s+and\s+background",
       3: r"source\s+(?:and\s+amount|of\s+funds)",
       4: r"purpose\s+of\s+(?:the\s+)?transaction",
       5: r"interest\s+in\s+(?:the\s+)?securities",
       6: r"contracts",
       7: r"material\s+to\s+be\s+filed",
   }
   #: A heading STARTS A LINE. That single constraint rejects the mid-prose cross-references
   #: ("...as described in Item 4 of Schedule 13D") that make a looser bare-number anchor
   #: unusable, which in turn lets the caption become OPTIONAL: when the line ends right after
   #: "Item N.", it is a captionless heading, not a cross-reference. The caption, when present,
   #: is consumed to end of line so a body never starts mid-caption ("or Other Consideration...").
   _ITEM_ANCHORS_LINE: dict[int, re.Pattern] = {
       n: re.compile(rf"^[ \t]*item{_SEP}{n}\b[\.\:\)]?[ \t]*(?:{cap}[^\n]*|$)", re.I | re.M)
       for n, cap in _ITEM_CAPTIONS.items()
   }
   #: Any captioned heading, anywhere -- used ONLY to detect that a carved body swallowed a
   #: later item, never to carve.
   _ITEM_HEADING_ANYWHERE: dict[int, re.Pattern] = {
       n: re.compile(rf"item{_SEP}{n}{_SEP}(?:{cap})", re.I)
       for n, cap in _ITEM_CAPTIONS.items()
   }
   ```
   - [x] Extract the existing body-walk into `_carve_with(text, anchors)` so both anchor sets
         share one implementation (start at the heading, end at the earliest of any later item's
         heading or `SIGNATURE`, drop bodies under `_ITEM_TEXT_MIN_CHARS`).
   - [x] Rewrite `_extract_13d_item_sections` as the **union rule**:
   ```python
   def _extract_13d_item_sections(text: str) -> dict[str, str]:
       """Carve Item 3/4/5/6 bodies, preferring the line-anchored headings and falling back to
       the legacy anchors ONLY where line-anchoring found nothing AND the legacy body is not
       contaminated.

       Both halves are load-bearing. Line-anchoring is what fixes the three Item 4 misses, but
       it cannot match a filing rendered as ONE line with no newlines at all (HUBB
       0001162044-13-001406 is such a filing) -- the legacy anchor is the only thing that reads
       those. The contamination test is what stops the fallback reintroducing the bug it exists
       to fix: a legacy item3 body that still contains Item 4's heading has swallowed Item 4
       (measured on 4.0% of originals / 6.8% of amendments) and is worse than no body at all."""
       if not text:
           return {}
       line_sections = _carve_with(text, _ITEM_ANCHORS_LINE)
       legacy_sections = _carve_with(text, _ITEM_ANCHORS)
       out = dict(line_sections)
       for item_no, field in _ITEM_TEXT_FIELD.items():
           if field in out or field not in legacy_sections:
               continue
           body = legacy_sections[field]
           if not _swallowed_a_later_item(item_no, body):
               out[field] = body
       return out


   def _swallowed_a_later_item(item_no: int, body: str) -> bool:
       return any(_ITEM_HEADING_ANYWHERE[later].search(body) for later in range(item_no + 1, 8))
   ```

**Measured effect** (all 182 original SC 13Ds — the ground-truth set, since an original must
answer every item — plus 200 random amendments):

| field | originals now | originals after | amendments now | amendments after |
|---|---|---|---|---|
| item3 | 95.1% | **96.7%** | 36.5% | **37.0%** |
| item4 | 92.9% | **98.9%** | 61.0% | **67.0%** |
| item5 | 97.3% | **98.9%** | 81.0% | **82.5%** |
| item6 | 96.7% | **98.4%** | 52.5% | **53.5%** |
| **item3 contaminated** | **4.0%** | **0%** | **6.8%** | **0%** |

**Zero regressions** on either population.

**Why amendment coverage stays well under 100% — and must.** Carved ÷ *present in the document*
is ≈100% for every item (item3 101.4%, item4 98.5%, item5 101.2%, item6 101.9%; over 100% because
the union also catches captionless headings the detector misses). Under Rule 13d-2(a) an amendment
restates only materially changed items, so a 13D/A legitimately contains a subset. Do **not** treat
the amendment percentages as a carve deficiency to chase.

**Verification**:
- [x] `"$PY" -m pytest tests/data_extract/structure/test_fetch_8k_13d_edgar.py -v -s`
- [x] The two existing fixtures must still pass unchanged — in particular
      `test_item5_body_survives_a_cross_reference_to_item6`, whose "See Item 6 for information…"
      cross-reference is mid-line and therefore correctly rejected by line-anchoring.
- [x] Re-run the measurement harness over the 182 originals; assert item3 contamination is 0
      and item4 coverage ≥ 98%.

---

### Phase 3: Zero-risk text normalization ✅

**Goal**: Remove encoding and formatting artifacts that damage tokenization. Nothing semantic.

Raw bodies currently contain cp1252 mojibake (`\x93group\x94` in PSA; `\x93Asset Purchase
Agreement\x94` in STX) and begin with runs of box-drawing characters (STX's item4 body opens with
40 `─`).

**Changes**:

1. `src/data_extract/utils/structure/fetch_13d_edgar.py`:
   - [x] Add the normalizer and apply it to each carved body inside `_carve_with`, before the
         `_ITEM_TEXT_MIN_CHARS` gate.
   ```python
   #: cp1252 bytes that survive EDGAR's own encoding round-trip (a real PSA filing stores
   #: \x93group\x94 for curly quotes), plus the unicode punctuation and zero-width characters
   #: that split a word into two tokens for no semantic reason. Character-for-character
   #: substitutions only -- no sentence or phrase is ever removed here.
   #: Written as \u escapes, not literal glyphs: the characters this table exists to remove are
   #: exactly the ones an editor or a lossy copy-paste would silently mangle in the source.
   _CHAR_NORMALIZATION = {
       "\x91": "'", "\x92": "'", "\x93": '"', "\x94": '"', "\x95": "-", "\x96": "-",
       "\x97": "-", "\x85": "...", "\xa0": " ", "‘": "'", "’": "'",
       "“": '"', "”": '"', "–": "-", "—": "-", "­": "",
       "​": "", "﻿": "",
   }
   #: Box-drawing / rule lines used as visual separators under a heading (U+2500-U+257F is the
   #: Box Drawing block). Bounded to runs of 3+ so a hyphenated word ("non-transferable") and a
   #: negative number are never touched.
   _RULE_RUN_RE = re.compile(r"[─-╿=_]{3,}|(?<![\w-])-{3,}(?![\w-])")


   def _normalize_item_text(body: str) -> str:
       """Encoding and whitespace only. Deliberately NOT a content cleaner: stripping the legal
       boilerplate and the leaked cover-page rows was measured and moved the embedding similarity
       noise floor by 1.9-2.6%, which does not pay for the regex risk of deleting real prose."""
       if not body:
           return body
       for bad, good in _CHAR_NORMALIZATION.items():
           body = body.replace(bad, good)
       body = _RULE_RUN_RE.sub(" ", body)
       body = re.sub(r"[ \t]+", " ", body)
       body = re.sub(r" *\n[ \t]*", "\n", body)
       body = re.sub(r"\n{3,}", "\n\n", body)
       return body.strip()
   ```

**Verification**:
- [x] Unit test: a body containing `\x93`, `\xa0`, `────────` and ragged spacing normalizes to
      straight quotes, plain spaces, no rules, single blank lines.
- [x] Unit test: `non-transferable`, `--5`, and a `-1,234` numeric survive `_RULE_RUN_RE` untouched.
- [x] Re-run the Phase 2 coverage measurement — normalization strips only rules and whitespace,
      so coverage must not move by more than one filing.

---

### Phase 4: False-zero guard + `reporting_person_comment` ✅

**Goal**: Never write a 0% stake the filer did not disclose.

`_num_or_null` gates on `has_structured_data`, which was always `False` pre-mandate and so
protected every row by accident. Post-mandate it is always `True`, and the guard stops
discriminating. Filers exploit `<commentContent>` to defer the numbers:

```xml
<soleVotingPower>0</soleVotingPower>  ... <percentOfClass>0</percentOfClass>
<commentContent>Rows 7, 8, 9, 10, 11, and 13:  See Item 5 of this Schedule 13D amendment.</commentContent>
```
(EL `0001140361-25-042382`, verified against the raw XML.) **84 of 738 backlog rows (11.4%)** have
`percent_of_class == 0`; some are genuine full disposals, so the comment is the discriminator.

**Changes**:

1. `src/data_extract/utils/structure/fetch_13d_edgar.py`:
   - [x] Add `reporting_person_comment` to `_COLS` (after `type_of_reporting_person`).
   - [x] Add the placeholder test and honour it in `_num_or_null`:
   ```python
   _RP_NUMERIC_ATTRS = ("sole_voting_power", "shared_voting_power", "sole_dispositive_power",
                        "shared_dispositive_power", "aggregate_amount", "percent_of_class")


   def _is_placeholder_numerics(rp) -> bool:
       """A reporting person whose SIX numerics are all 0 while `commentContent` is set has not
       disclosed a zero position -- it has deferred the numbers to the Item 5 narrative ("Rows 7,
       8, 9, 10, 11, and 13: See Item 5"). Writing the literal 0 would make the table claim a 0%
       stake, which is the one thing this module's numeric handling exists to prevent. The
       all-zero AND comment-present conjunction matters: a genuine full disposal reports zeros
       with no comment, and a commented row with real numbers keeps them."""
       if not (getattr(rp, "comment", None) or "").strip():
           return False
       values = [getattr(rp, attr, None) for attr in _RP_NUMERIC_ATTRS]
       present = [v for v in values if v is not None]
       return bool(present) and all(v == 0 for v in present)
   ```
   - [x] In `_filing_rows`, compute `placeholder = _is_placeholder_numerics(rp)` once per person
         and pass `has_structured and not placeholder` as `_num_or_null`'s second argument.
   - [x] Write `"reporting_person_comment": (getattr(rp, "comment", None) or None)`.
         The fallback (no-persons) row gets `None`.

2. `sql/schema.sql`:
   - [x] **Hand-splice** `"reporting_person_comment" TEXT,` into the `sec_13d` block after
         `"type_of_reporting_person" TEXT,`. Do **not** regenerate the file — regeneration drops
         hand-added indexes. The diff must be purely additive: one line.

**Verification**:
- [x] Unit test: all-zero numerics + a comment → six NaN numerics, comment retained.
- [x] Unit test: all-zero numerics + **no** comment → zeros preserved (genuine full disposal).
- [x] Unit test: real numerics + a comment → numerics preserved, comment retained.
- [x] `git diff sql/schema.sql` shows exactly one added line.

---

### Phase 5: Tests, docs, and the rebuild runbook ✅

**Goal**: Regression cover, synced docs, and a documented (not executed) rebuild.

**Changes**:

1. `tests/data_extract/structure/test_fetch_8k_13d_edgar.py`:
   - [x] `test_13d_forms_cover_both_edgar_form_eras` — both `SC 13D` and `SCHEDULE 13D` present.
   - [x] `test_item4_anchor_matches_purpose_of_the_transaction` — the "the" variant carves.
   - [x] `test_item4_anchor_matches_a_captionless_heading` — a bare `Item 4.` line carves.
   - [x] `test_item3_body_stops_at_a_the_transaction_caption` — **the regression test for the
         real bug**: a fixture with `Item 3.` then `Item 4. Purpose of the Transaction` must
         produce an item3 body that does NOT contain item4's text.
   - [x] `test_item_carve_falls_back_on_a_single_line_filing` — a no-newline ASCII body still carves.
   - [x] `test_item_carve_fallback_rejects_a_contaminated_legacy_body`.
   - [x] `test_normalize_item_text_fixes_mojibake_rules_and_whitespace`.
   - [x] `test_normalize_item_text_leaves_hyphenated_words_and_negatives_alone`.
   - [x] Three `_is_placeholder_numerics` tests from Phase 4.
   - [x] Extend the existing sanity-check print with the new conclusions.

2. Docs:
   - [x] `docs/data_sources.md` — note the 2024-12-17 form-string change and that both eras are listed.
   - [x] Module docstring in `fetch_13d_edgar.py` — its opening claim ("`has_structured_data` is
         False for essentially every real 13D") is now **false for every filing after
         2024-12-16** and must be rewritten to describe the two eras.

3. `docs/runbook.md` — add the rebuild procedure (**documented, not executed** in this task):
   ```bash
   # sec_13d rebuild: the 1,666 pre-2024-12-17 rows carry the item3 contamination and only
   # ever captured one reporting person per filing, so they are replaced rather than patched.
   MSYS_NO_PATHCONV=1 docker exec pea_db psql -U alexandre -d pea \
     -c "DELETE FROM sec_13d;" -c "DELETE FROM sec_13d_transactions;"
   # -F is REQUIRED: the manifest still holds a recent run date, so an incremental run
   # would resume from it and refetch nothing.
   rtk "$PY" -m src data_extract sec-13d -F
   ```

**Verification**:
- [x] `"$PY" -m pytest tests/data_extract/structure/test_fetch_8k_13d_edgar.py -v -s`
- [x] `"$PY" -m pytest tests/data_extract -q` — no regressions.
- [x] Docs mention both form eras.

---

## Testing Strategy

**Unit** — synthetic known-truth fixtures for the carve, normalization and placeholder logic
(parsing math, per the repo's testing convention). The existing `_REAL_13D_TEXT_SAMPLE` (a real
ZTS 2014 filing) stays the integration-shaped fixture and must pass unchanged.

**Measured acceptance** — re-run the research harness over all 182 originals + 200 amendments:
item3 contamination `== 0`, item4 originals `>= 98%`, zero regressions vs the current carve.

**Manual** — one modern filing (`APO 0000950142-26-000320`) and one legacy filing
(`MNST 0001341004-15-000486`) through `_filing_rows`; confirm MNST's item3 is ~850 chars, not 17,776.

## Risk Mitigation

1. **Line-anchoring breaks filings with no newlines.**
   Real: HUBB `0001162044-13-001406` renders as a single line. Mitigated by the legacy fallback in
   the union rule; covered by a dedicated test.
2. **The fallback reintroduces contamination.**
   Mitigated by `_swallowed_a_later_item`; measured 0% on both populations.
3. **A widened caption creates false positives.**
   `_ITEM_CAPTIONS` is used only by the line-anchored set, which requires a line start. Measured:
   zero regressions, and one new contaminated item4 body on amendments (0.7%) against +12 recovered.
4. **The placeholder guard nulls a genuine zero position.**
   Requires *both* all-zero numerics and a non-empty comment. A genuine full disposal has no
   comment. Covered by a test in each direction.
5. **`sql/schema.sql` regeneration drops hand-added indexes.**
   Hand-splice one line; verify the diff is purely additive.
6. **Rebuild cost.** ~1,700 filings re-fetched. Not executed here; `-F` is required or it no-ops.

**Rollback** — Phases 1–4 are single-file code changes, revertable by `git revert`. The DELETE in
Phase 5 is destructive and is documented only; every deleted row is reconstructible from EDGAR.

## Dependencies

- `edgartools` 5.51.0 — `Schedule13D.from_filing` already dispatches XML vs SGML header; no upgrade needed.
- No new packages. No `Tables` registry change. No index change.

## Success Criteria

- [ ] `sec_13d` ingests filings dated after 2024-12-16.
- [ ] item3 contamination measured at **0%** on originals and amendments.
- [ ] item4 coverage ≥ **98%** on originals; no field regresses.
- [ ] No row has `percent_of_class == 0` while its `reporting_person_comment` is non-empty.
- [ ] Item bodies contain no `\x80-\x9f` bytes and no runs of 3+ box-drawing characters.
- [ ] Full `tests/data_extract` suite green; the new test prints a sanity conclusion.
- [ ] `sql/schema.sql` diff is exactly one added line.
- [ ] Rebuild procedure documented in `docs/runbook.md`, not executed.

## Estimated Effort

| Phase | Estimate |
|---|---|
| 1 — form types + `--full` | 20 min |
| 2 — carve rewrite | 1.5 h |
| 3 — normalization | 30 min |
| 4 — false-zero guard + column | 45 min |
| 5 — tests, docs, runbook | 1.5 h |
| **Total** | **~4.5 h** (excludes the rebuild run) |

## Notes for Implementation

- Keep `_ITEM_ANCHORS` byte-identical; it is the measured fallback, not dead code.
- The union rule's two halves each fix a failure the other causes — do not simplify to one anchor set.
- Follow the repo's docstring convention: keep the measurement and the evidence, drop the
  chronology. State *what is true and how it was measured*, not "before/after this fix".
- `has_structured_data` is no longer a proxy for "pre-2025". Do not add new logic that keys on it
  to mean an era; it now means "this filing has XML".
- Amendment item coverage below 100% is Rule 13d-2(a) working as intended, not a bug to chase.

---

## Implementation Results (measured 2026-08-31)

All five phases implemented. Every number below was re-measured against live EDGAR during
implementation, not carried over from the research session.

### Phase 1 — form types
- `Company("CVNA").get_filings(form=SEC_13D_FORMS)` now returns 4 post-mandate filings
  (2025-02-28 -> 2026-05-01), all typed `SCHEDULE 13D/A`. Previously zero.
- Modern filing `CVNA 0001104659-25-019162` through `_filing_rows`: **3 reporting persons**,
  `has_structured_data=1.0`, `date_of_event=2025-02-26`, `percent_of_class=41.5`.
- Across the 155 tickers already in `sec_13d`: **322 post-mandate filings / 1,351 rows** are now
  reachable and were never ingested.

### Phase 2 — the union carve (182 originals + 200 seeded-random amendments)

| field | originals legacy | originals union | amendments legacy | amendments union |
|---|---|---|---|---|
| item3 | 95.1% | **96.7%** | 36.5% | **37.5%** |
| item4 | 92.9% | **98.9%** | 61.5% | **70.0%** |
| item5 | 97.3% | **98.9%** | 78.5% | **80.0%** |
| item6 | 96.7% | **98.4%** | 47.0% | **48.0%** |
| **item3 contaminated** | **3.8% (7)** | **0% (0)** | **2.5% (5)** | **0% (0)** |

**Zero regressions** on either population. Originals reproduce the plan's predicted figures
exactly; the amendment sample differs from the research session's draw, and came out better.

MNST `0001341004-15-000486`, the plan's worst case: `item3` **17,776 -> 825 chars**, and `item4`
recovered from a MISS to 16,872 chars -- the swallowed body returned to its own item.

### Phase 3 — normalization
Across all **1,186** carved bodies: **0** contain cp1252 `\x80-\x9f` bytes, **0** a 3+
box-drawing run, **0** a non-breaking space, **0** ragged whitespace. For contrast, in the raw
filing text 42.4% of filings carry cp1252 bytes and 84.0% carry a rule run. Carve coverage was
**bit-identical** before and after normalization (0 filings moved, tolerance was 1).

**One deviation from the plan, driven by measurement.** The plan's `_CHAR_NORMALIZATION` table
covered 8 of the cp1252 C1 block's 32 codepoints, which left 2 bodies still holding `\x80`. That
byte is the **euro sign**, in two real KDP filings reading *"Investor paid EUR 52,544.78 in cash to
Acorn"* -- dropping it would silently change the currency of a disclosed consideration. The block
is therefore **decoded** (derived, not hand-typed) rather than partially deleted, and the success
criterion is met at 0 residual bytes.

### Phase 4 — the false-zero guard
Over the 322-filing post-mandate backlog (`has_structured_data` is **99.9%** True there, so the
old accidental guard is fully disarmed):
- **19** rows had all six numerics 0 *with* a comment -> correctly nulled to NaN.
- **230** rows had all six numerics 0 with *no* comment -> correctly preserved (genuine full
  disposals).
- **3** rows report `percent_of_class == 0` next to a comment. All three were opened and are
  **real**, not false zeros: a holding so small it rounds to 0.0% while the share count is
  non-zero (CALFINCO's 18,632,216 shares against 54,730,851,778,811 outstanding after Azul's
  reorganization; Silver Lake's comment states in as many words *"reflects less than 0.1% of the
  outstanding shares"*).

  **This refines one success criterion.** As literally worded ("no row has `percent_of_class == 0`
  while its comment is non-empty") it would require nulling those three, destroying real disclosed
  share counts. The criterion's intent -- Desired End State #4, *"no row claims a 0% stake that the
  filer did not actually disclose"* -- is fully met. The all-six-zero conjunction is what
  distinguishes them, and is now covered by a test in each direction.

- `sql/schema.sql` diff is **exactly one added line**, hand-spliced.

### Phase 5 — tests and docs
`tests/data_extract/structure/test_fetch_8k_13d_edgar.py`: **25 -> 38 tests**, all passing, the
two pre-existing carve fixtures unchanged. Docs updated: a new "Schedule 13D" trap section in
`docs/data_sources.md`, the corrected NULL-vs-0 rule in `docs/data_schema.md`, the rebuild
procedure in `docs/runbook.md`, and the module docstring rewritten around the two eras.

**The rebuild was NOT executed**, per scope. It is documented in `docs/runbook.md`.

### Test status

Green, and directly covering every change made:
- `tests/data_extract/structure/test_fetch_8k_13d_edgar.py` - **38 passed** (was 25), including
  the two pre-existing carve fixtures unchanged.
- `tests/data_extract/common/test_form_registry.py` - **6 passed**.

`tests/data_extract/structure + common + utils` - **149 passed, 3 skipped, 0 failed**.

That subtree had 17 failures when this task started. They were **pre-existing** (reproduced at
clean HEAD in a detached worktree) and have since been **fixed** in a follow-up: all three causes
were test stubs left behind by commit `10f51d5`, which moved production forward without updating
them - a fake context missing `config.local.filename.extraction`, an `sp500_tickers` fixture
writing 2 of the 6 columns `load_cik_mapping` projects server-side, and two DEF 14A stubs missing
the leading `context` argument their production counterparts had gained.

The whole-directory `tests/data_extract` run was not carried to completion - it contains the
live-network fundamentals/sharadar tests and had run well past an hour. The three subdirectories
that can reach the four files this task touches (`fetch_13d_edgar.py`, `constants.py`, `cli.py`,
`sql/schema.sql`) were run in full, as above.
