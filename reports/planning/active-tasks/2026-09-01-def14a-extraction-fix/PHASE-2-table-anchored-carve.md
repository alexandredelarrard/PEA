# Phase 2 — Table-anchored carve ⬜

**Goal**: locate the five tabular targets by parsing the filing's `<table>` elements and classifying
them on their header signature, instead of guessing their position with a text anchor and a fixed
character budget. Keep the existing anchor carve for the two prose sections it serves well.

**This is the load-bearing phase.** Every recall failure the research traced back to the carve is
fixed here, and the LLM payload gets *smaller*.

---

## Why (measured, follow-up 3, 25 filings / 8,628 tables / 8,580 ground-truth cells)

| target | ground truth | today (A) | table-anchored (B) |
|---|---|---|---|
| Summary Compensation Table | 24 | 20 | **25** |
| **Director compensation table** | 25 | **2** | **24** |
| Insider / group ownership | 24 | 22 | 22 |
| ≥5% holders | 22 | 21 | **25** |
| Audit fee table | 23 | 20 | **25** |

- **B = 121/125 (96.8%)**, 1 false positive, and it never picked a semantically different table.
- **Payload**: A is mean 50,300 chars and hits its cap on *every section of every filing*.
  B is mean **4,283**. The realistic hybrid (B tables + narrative slices + A's 20k bios + 6k
  governance) is mean **36,544 = 73% of today's**.
- **The director-comp table is A's structural blind spot** and always was: it sits at **23-36%** of
  the document, in the gap between A's `DIRECTOR NOMINEES` window (ends ~17-21%) and
  `EXECUTIVE COMPENSATION` (starts ~42-64%). Median miss distance **70,761 chars**. No budget widening
  reaches it.
- **`n_neos == 1` on 21.7% of 2012+ rows is an ANCHOR failure, not truncation.** When A's anchor
  lands, the 7k window never truncates the SCT (`frac_in_A = 1.00` in all 20 hits); it simply misses
  the table entirely in 4 of 24. B captures **354 of 354** NEO×year rows.
- **Strategy C (boundary-aware narrative) is rejected** — net −2 on narrative targets. Its only
  sound element (applying the TOC skip on paths that lack it) is adopted below.

---

## Changes

### 1. `src/data_extract/utils/structure/def14a_tables.py` (new, ~250 lines)

Pure functions, no I/O, no LLM, no `context`. Unit-testable against cached filings.

#### 1a. Cell-grid extraction

- [ ] `iter_tables(html: str) -> list[list[list[str]]]` — every `<table>` as a row-major grid of
      cleaned cell strings, via `lxml.html` (already a transitive dependency through edgartools).
- [ ] **`<br>` and block-element boundaries MUST emit a separator.** Dropping it fuses **180 cells
      (2.1%) across 16 of 25 filings** (`All othercompensation`, `James DimonChairman and CEO`).
      This is load-bearing for **classification**, not just for values: *JPM's SCT was invisible to
      the classifier* until block `<div>` boundaries emitted a separator. Same root cause as the
      Agilent `1.0e19`.
- [ ] **`<sup>` must be stripped before the cell text is read.** 47 cells affected (0.5%, 5 filings),
      **7 of them the bare-digit form that corrupts a value** (`$1,587,852` + `⁶` → `1,5878526`).
      Same root cause as PG's ownership 10×. Note the 2026 PG proxy uses a CSS-positioned `<span>`
      rather than `<sup>` and the bug survives — so strip on **both** `sup` elements and any element
      whose `style` contains `vertical-align:` with `top`/`super`.
- [ ] Expand `colspan`. **Handle `rowspan`** — edgartools ignores it entirely, which shifts every
      subsequent row left by one column.
- [ ] Drop cells that are a standalone currency glyph (`$`, `€`) — a `$` in its own `<td>` doubles the
      effective column count and desynchronises the column map (the GE / CAT mechanism).
- [ ] Collapse whitespace, normalise `\xa0` → space, `html.unescape`. Reuse the exact normalisation
      set from [`edgar_extract.html_to_text`](../../../../src/data_extract/utils/common/edgar_extract.py#L28)
      so the two paths cannot disagree on what a cell says. `html_to_text` already maps `<br>` → `\n`;
      the new code needs the same at cell granularity.
- [ ] `merge_header_rows(grid) -> (header, data_rows)` — merge up to **3** leading rows into one
      header when the continuation rows are non-numeric. edgartools treats exactly one row as the
      header and scans only `grid[:4]`, so multi-row headers (`Stock` / `Awards ($)`) drop the column.
- [ ] Skip tables with fewer than 2 data rows or no numeric cell. Measured: **zero of 8,628 tables
      contain a nested `<table>`**, so no recursion is needed.

#### 1b. Header signatures — strict, not naive

A naive signature scores SCT 25/25 *but admits 2-5 candidates in 8/25*, insider ownership **14/25**,
5% holders **3/25**, audit fees **2/25 (23 wrong picks)**. Three fixes reach 96.8%:

| target | signature | must-reject |
|---|---|---|
| `sct` | a `Year` column **and** ≥1 SEC-mandated SCT column (`Salary`, `Stock Awards`, `Option Awards`, `Non-Equity Incentive`, `All Other Compensation`, `Total`) | **any header containing "compensation actually paid"** — that is the PvP table, not the SCT. Also reject a `Target`/`Realized`/`Realizable` pay table. |
| `director_comp` | (`Fees Earned or Paid in Cash` \| `Fees` \| `Cash Fees` \| `Retainer`) **and** (`Stock Awards` \| `Restricted Stock Units` \| `Share Awards`) **and** `Total` | reject if a `Salary` column is present (that is the SCT) |
| `audit_fees` | **exact** fee-category labels as ROW labels or headers: `Audit Fees`, `Audit-Related Fees`, `Tax Fees`, `All Other Fees`, `Total` — exact match, not substring | reject the footnote table (rows starting `(1)`, `(2)`, … with prose bodies) |
| `ownership_insider` | a name column **and** a share-count column **and** a row matching `as a group` | **do not require a percent column** — the insider table often has none. This is why one signature cannot serve both ownership targets. |
| `ownership_5pct` | (`Percent of Class` \| `% of Class`) **and** (`Number of Shares` \| `Shares Beneficially Owned`) **and** a row matching a known institution or `5%` | reject if the only rows are insiders |

- [ ] `classify_table(header, rows) -> str | None` returning one of the 5 target names.
- [ ] `Restricted Stock Units` and `Cash Fees` are explicitly included: those are the exact labels
      edgartools' synonym list misses, and they are what nulled CAT's and GE's director-comp columns.
- [ ] On multiple candidates for one target, tie-break on **data-row count** (the real table is the
      long one), then on document position (earlier wins). Log a debug line naming the runner-up so
      a wrong pick is diagnosable.

#### 1c. TSV serialization

- [ ] `to_tsv(header, rows) -> str` — tab-separated, one line per row, header first. Real SCT text
      averages **1,686 chars (1,886 as TSV)**, about **25%** of the 7,000-char window A spends to
      sometimes reach it.
- [ ] TSV rather than prose because column identity is the thing the LLM keeps getting wrong
      (a dropped `stock_awards` under a `Restricted Stock Units` header); an explicit delimiter makes
      the column boundary unambiguous.
- [ ] Cap each serialized table at a generous budget (e.g. 20,000 chars) purely as a runaway guard.
      Max observed is 14,169.

### 2. `fetch_def14a_llm.prepare_def14a_sections` — become a router

Signature changes from `(text)` to `(html, text)` — it needs the HTML for B and the flattened text
for A. Router per follow-up 3's recommendation:

| target | strategy |
|---|---|
| SCT, director comp, ≥5% holders, audit fee table | **B** |
| Insider / group ownership | **B**, with A's `SECURITY OWNERSHIP` window as fallback |
| Auditor **NAME** | narrative slice anchored on **B's fee table position** — the firm name is often cell `[0][0]` or the sentence immediately before the table |
| Pay ratio, say-on-pay | **A, with the anchor fixed** (below) — not the budget |
| Director bios, corporate governance | **keep A** — B does not cover prose |

- [ ] Emit one labelled block per target, same `=== LABEL ===` convention the prompt already
      references, so the prompt needs only additive edits.
- [ ] When B finds nothing for a target, fall back to A's window for that target. Never silently
      emit nothing for a target that A could have reached.
- [ ] Log per filing: which targets came from B, which from A, which from neither, and the total
      payload chars. This is the diagnostic that tells you a format era broke.

### 3. `fetch_def14a_llm.py` — the two anchor defects (both bidirectional)

- [ ] **The 5% TOC floor is applied only on the anchor-fallback path** inside
      `_find_content_section` (`min_pos = max(5000, len(text) * 0.05)`), and NEM / ROK / SYK carves
      landed at 3.0% / 2.0% / 3.7% of the document while the pay ratio sat at 49-92%. But the floor
      is *itself* the killer elsewhere: **A-2016's say-on-pay result sits at 4.2%** of the document.
      Fix: apply the TOC skip consistently on the content-regex path **for the sections whose target
      is never in the front matter** (pay ratio, median pay, auditor), and **lower or drop it for
      say-on-pay**, whose result legitimately appears in an early "voting matters" summary.
- [ ] **`last_occurrence=True` overshoots in both directions** — it puts WMT's auditor carve 3,066
      chars *before* the fee table and HUBB / ROK **149k / 165k** chars away. With B owning the fee
      table, `AUDITOR FEES` no longer needs `last_occurrence` at all: **remove it** and anchor the
      auditor-name narrative slice on B's fee-table position.
- [ ] **Widen only the 5 sections that genuinely truncate.** Only 5 of 25 narrative misses are within
      ~3,000 chars of the slice end (HSIC +123, PFE +190, WMT +430, PFE-median +206, XOM say-on-pay
      +2,057); everything else is 10k-500k chars away. Add ~3,000 chars to `PAY RATIO & MEDIAN PAY`
      and `SAY ON PAY`. Do **not** widen `EXECUTIVE COMPENSATION` — B owns it now, so the A window
      for it can be **removed or shrunk**, which is where most of the payload saving comes from.

### 4. `fetch_def14a_llm._process_filing` — pass the HTML through

- [ ] `raw_html = sec_get(...).text` is already available; pass both `raw_html` and
      `html_to_text(raw_html)` into `prepare_def14a_sections`. No extra HTTP request.

---

## Verification

All of it runs off Phase 0's on-disk filing cache — **zero network, zero LLM cost**, so it is cheap
enough to re-run on every edit.

- [ ] **New `tests/data_extract/structure/test_def14a_tables.py`** — the recall harness.
      Ground truth is a checked-in JSON: per cached filing, for each of the 5 targets, either
      `null` (genuinely absent) or **one identifying value** from the real table (e.g. the CEO's
      salary, a named director's total fees, the total audit fee, Vanguard's percent of class).
      Assert the classifier finds the right table **and that the identifying value appears in the
      serialized TSV**. Finding "a table" is not the test — the research's naive signature found
      tables and picked the wrong ones 23 times out of 25 on fees.
- [ ] Print the recall matrix as the sanity conclusion, in the same shape as the table above, so it
      is directly comparable to follow-up 3's numbers.
- [ ] **Payload assertion**: mean serialized payload across the cached corpus ≤ **40,000** chars
      (target ~36,544; baseline 50,300). Print mean / median / max.
- [ ] **Prerequisite regression tests**, as their own cases with real fixtures:
      - `<br>` separator: the Agilent 2005 cell `1,000,000<br>1,000,000<br>925,000` must yield three
        cells, never `10000001000000925000`.
      - `<sup>` strip: PG's `217,956,036<sup>2</sup>` must yield `217956036`, never `2179560362`.
      - `$`-in-own-`<td>`: GE's `['$','0','$','345,795','$','0','$','345,795']` must map to
        4 numeric columns aligned to `['CASH FEES','STOCK AWARDS','ALL OTHER COMP','TOTAL']`.
      - multi-row header: a `Stock` / `Awards ($)` split header must merge into one column.
      - `rowspan`: a table with a `rowspan=2` first cell must not shift subsequent rows left.
- [ ] **Anchor-fix tests**: A-2016 say-on-pay (at 4.2% of the document) is now reached; NEM/ROK/SYK
      carves no longer land in the front matter; WMT/HUBB/ROK auditor slices are near the fee table.
- [ ] **End-to-end LLM probe** (~10 calls, on cached filings): CAT 2026, PFE 2026, GE 2026, JPM 2026,
      WMT, TSLA, T 2011, A 2016 — confirm the SCT now yields ≥ 3 NEOs × the years shown, and that a
      director-comp table is present where it exists. `n_neos == 1` must be gone from these.
- [ ] `"$PY" -m pytest tests/data_extract/structure/ -q` green.

## Rollback

`def14a_tables.py` is new and additive. The router change is one function — revert
`prepare_def14a_sections` to its `(text)` signature and the old path returns. Keep the old anchor
constants in place (they are still used for bios/governance and the fallbacks) so a revert is a
one-file change.

## Notes

- **Do not adopt strategy C.** Measured net loss on narrative targets.
- B's 5 residual failures in 125 are recorded and accepted: 4 genuinely-absent disclosures,
  MMM-2011 and T-2011 headers with no signature (`Stock`/`Total`, no %, group total in a footnote),
  XOM's page being a JPG with white-on-white text, SYK's fees living in `.gif` charts, MS's table
  paginated across two `<table>` elements (loses 15 of 19 holders), TSLA's all-em-dash director fees,
  and T-2011's fees being prose. Chasing these is the "few percentage points" you said to trade away.
- The one measured B false positive is MS — it picked page 2 of a paginated table. The data-row
  tie-break is what mitigates it; the harness should include MS so the behaviour is visible.
- Keep `def14a_tables.py` free of `context` and free of I/O. That is what makes the harness cheap,
  and cheap is what makes it get run.
