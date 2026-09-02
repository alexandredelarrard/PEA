# Phase 2 — Table-anchored carve ✅

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

- [x] `iter_tables(html: str) -> list[list[list[str]]]` — every `<table>` as a row-major grid of
      cleaned cell strings, via `lxml.html` (already a transitive dependency through edgartools).
- [x] **`<br>` and block-element boundaries MUST emit a separator.** Dropping it fuses **180 cells
      (2.1%) across 16 of 25 filings** (`All othercompensation`, `James DimonChairman and CEO`).
      This is load-bearing for **classification**, not just for values: *JPM's SCT was invisible to
      the classifier* until block `<div>` boundaries emitted a separator. Same root cause as the
      Agilent `1.0e19`.
- [x] **`<sup>` must be stripped before the cell text is read.** 47 cells affected (0.5%, 5 filings),
      **7 of them the bare-digit form that corrupts a value** (`$1,587,852` + `⁶` → `1,5878526`).
      Same root cause as PG's ownership 10×. Note the 2026 PG proxy uses a CSS-positioned `<span>`
      rather than `<sup>` and the bug survives — so strip on **both** `sup` elements and any element
      whose `style` contains `vertical-align:` with `top`/`super`.
- [x] Expand `colspan`. **Handle `rowspan`** — edgartools ignores it entirely, which shifts every
      subsequent row left by one column.
- [x] Drop cells that are a standalone currency glyph (`$`, `€`) — a `$` in its own `<td>` doubles the
      effective column count and desynchronises the column map (the GE / CAT mechanism).
- [x] Collapse whitespace, normalise `\xa0` → space, `html.unescape`. Reuse the exact normalisation
      set from [`edgar_extract.html_to_text`](../../../../src/data_extract/utils/common/edgar_extract.py#L28)
      so the two paths cannot disagree on what a cell says. `html_to_text` already maps `<br>` → `\n`;
      the new code needs the same at cell granularity.
- [x] `merge_header_rows(grid) -> (header, data_rows)` — merge up to **3** leading rows into one
      header when the continuation rows are non-numeric. edgartools treats exactly one row as the
      header and scans only `grid[:4]`, so multi-row headers (`Stock` / `Awards ($)`) drop the column.
- [x] Skip tables with fewer than 2 data rows or no numeric cell. Measured: **zero of 8,628 tables
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

- [x] `classify_table(header, rows) -> str | None` returning one of the 5 target names.
- [x] `Restricted Stock Units` and `Cash Fees` are explicitly included: those are the exact labels
      edgartools' synonym list misses, and they are what nulled CAT's and GE's director-comp columns.
- [x] On multiple candidates for one target, tie-break on **data-row count** (the real table is the
      long one), then on document position (earlier wins). Log a debug line naming the runner-up so
      a wrong pick is diagnosable.

#### 1c. TSV serialization

- [x] `to_tsv(header, rows) -> str` — tab-separated, one line per row, header first. Real SCT text
      averages **1,686 chars (1,886 as TSV)**, about **25%** of the 7,000-char window A spends to
      sometimes reach it.
- [x] TSV rather than prose because column identity is the thing the LLM keeps getting wrong
      (a dropped `stock_awards` under a `Restricted Stock Units` header); an explicit delimiter makes
      the column boundary unambiguous.
- [x] Cap each serialized table at a generous budget (e.g. 20,000 chars) purely as a runaway guard.
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

- [x] Emit one labelled block per target, same `=== LABEL ===` convention the prompt already
      references, so the prompt needs only additive edits.
- [x] When B finds nothing for a target, fall back to A's window for that target. Never silently
      emit nothing for a target that A could have reached.
- [x] Log per filing: which targets came from B, which from A, which from neither, and the total
      payload chars. This is the diagnostic that tells you a format era broke.

### 3. `fetch_def14a_llm.py` — the two anchor defects (both bidirectional)

- [x] **The 5% TOC floor is applied only on the anchor-fallback path** inside
      `_find_content_section` (`min_pos = max(5000, len(text) * 0.05)`), and NEM / ROK / SYK carves
      landed at 3.0% / 2.0% / 3.7% of the document while the pay ratio sat at 49-92%. But the floor
      is *itself* the killer elsewhere: **A-2016's say-on-pay result sits at 4.2%** of the document.
      Fix: apply the TOC skip consistently on the content-regex path **for the sections whose target
      is never in the front matter** (pay ratio, median pay, auditor), and **lower or drop it for
      say-on-pay**, whose result legitimately appears in an early "voting matters" summary.
- [x] **`last_occurrence=True` overshoots in both directions** — it puts WMT's auditor carve 3,066
      chars *before* the fee table and HUBB / ROK **149k / 165k** chars away. With B owning the fee
      table, `AUDITOR FEES` no longer needs `last_occurrence` at all: **remove it** and anchor the
      auditor-name narrative slice on B's fee-table position.
- [x] **Widen only the 5 sections that genuinely truncate.** Only 5 of 25 narrative misses are within
      ~3,000 chars of the slice end (HSIC +123, PFE +190, WMT +430, PFE-median +206, XOM say-on-pay
      +2,057); everything else is 10k-500k chars away. Add ~3,000 chars to `PAY RATIO & MEDIAN PAY`
      and `SAY ON PAY`. Do **not** widen `EXECUTIVE COMPENSATION` — B owns it now, so the A window
      for it can be **removed or shrunk**, which is where most of the payload saving comes from.

### 4. `fetch_def14a_llm._process_filing` — pass the HTML through

- [x] `raw_html = sec_get(...).text` is already available; pass both `raw_html` and
      `html_to_text(raw_html)` into `prepare_def14a_sections`. No extra HTTP request.

---

## RESULTS — measured over 64 cached 2024-2026 filings (2.5x the research's sample)

| target | research B (25 filings) | this implementation (64) |
|---|---|---|
| Summary Compensation Table | 25/25 | **64/64 — 100%** |
| Director compensation table | 24/25 | **64/64 — 100%** |
| Insider / group ownership | 22/25 | **64/64 — 100%** |
| ≥5% holders | 25/25 | **64/64 — 100%** |
| Audit fee table | 25/25 | **52/64 — 81%** |
| **TOTAL** | **121/125 = 96.8%** | **308/320 = 96.2%** |

Serialized table payload: **mean 5,316 / median 4,013 / max 25,638** chars (research B: mean 4,283).

### Seven defects found and fixed that the plan did not anticipate

Each was found by measuring, and each was silently costing whole tickers:

1. **`lxml` refuses a `str` carrying an XML encoding declaration.** Every modern inline-XBRL
   filing opens with `<?xml version='1.0' encoding='ASCII'?>`, so the first implementation found
   **0 tables on 63 of 64 filings** while the raw `<table` count was 123-409. Input is encoded to
   **bytes** first. The bug hid behind a bare `except Exception: return []`, which is now a
   logged warning — a silent "this filing has no tables" is indistinguishable from a total
   parser failure, and that is exactly how it went unnoticed.
2. **One table can be TWO targets.** Filers routinely publish a single beneficial-ownership
   table holding both the ≥5% institutions and the directors-and-officers rows (AAPL 2026's is
   16 rows: Vanguard, BlackRock, the individual directors, then
   `All current directors and executive officers as a group (12 persons)`). Under
   first-match-wins it registered only as `ownership_insider` and ≥5% recall sat at **61%**.
   `classify_table` now returns a **list** of targets. → 61% → 80%.
3. **The ownership column labels in the plan's signatures do not match what filers write.** LMT
   uses `Percent of Outstanding Shares` / `Amount of Common Stock`, AMAT
   `Shares Beneficially Owned Percent`; neither contains "of class". Between them that was 6 of
   the 13 remaining ≥5% misses. → 80% → **100%**.
4. **Fee-category row labels are not always in column 0.** NKE colspans its label column, so
   the row reads `['', '', '', 'Audit Fees', 'Audit Fees', …]` and a `r[0]` test finds nothing.
   Now scans the leading 6 columns.
5. **`_MAX_HEADER_ROWS = 3` is too small.** EOG 2026's SCT header spans **four** rows
   (`Non-Equity` / `Stock` / `Name and Fiscal Salary Awards` / `Principal Position Year ($)`) and
   the cap cut off the row carrying `Year` — the SCT signature's own requirement — so the whole
   table went unclassified. Raised to 6 and made a pure runaway guard; what ends the header is
   the first row carrying a data-sized number.
6. **The header-merge terminator must be "contains a number", not "is a number".** AMAT stacks
   all three fiscal years inside ONE `<td>` per column with `<br>`, so after normalisation a
   cell reads `2025 2024 2023` and matches no whole-cell numeric test — the merge then swallowed
   the entire table, leaving **0 data rows** on all 3 AMAT filings. (This is the same filer the
   research measured at `total` = 3.527e23 from the *fused* form; the `<br>` separator fixes the
   value, this fixes the classification.) → SCT 95% → **100%**.
7. **Spacer columns must be judged on DATA rows.** A `colspan` header label propagates into
   every column it spans, so a header-based test finds no spacers. PG's insider table is **20
   columns of which 13 are pure spacers**. Added a post-split pass, plus a collapse of runs of
   columns identical in the header *and* every data row (GE's name column arrives 3× wide).
   Payload fell from mean 8,530 → **5,316**.

### The audit-fee residual is ONE measured format class, not a shrug

All 12 misses are 4 tickers × 3 years — **EOG, GE, JPM, PEG** — and every one puts its fee
**category labels in narrative prose**:

- **GE** prints an unlabelled table (`2024 $ 17.9 $ 3.0 $ 0.0 $ 0.0 $ 20.9`) and labels it in the
  *following* paragraph (`AUDIT FEES. Fees for the audit of…`).
- **JPM** gives the figures in prose: *"fees for the annual integrated audit … were $37.0 million
  and $26.3 million, respectively"*.
- **EOG**: *"Audit Fees . The aggregate fees billed for professional services rendered by
  Deloitte…"*. **PEG**: *"Audit Fees - The audit fees were incurred for audits of…"*.

No table classifier can reach these. The router's `AUDITOR FEES` anchor fallback fires on
**exactly those 12 filings** and on no others — which is the designed behaviour, verified.

### ⚠ DEVIATION — the G10 payload gate is not met, and the gate's own arithmetic is why

Measured full carve: **mean 42,225 / median 41,109 / min 38,059 / max 60,911** — an 18% reduction
from the baseline's 51,198 (which is *identical on every filing*, confirming the research's
"hits its cap on every section of every filing"). The gate asks ≤ 40,000.

The 40,000 came from the research's estimate of 36,544 = `B tables + A's 20k bios + 6k governance
+ narrative slices`. That estimate does **not** include two things this phase's own instructions
add: the **+3,000 widening of PAY RATIO and SAY ON PAY** (+6,000) and the new **AUDITOR NAME**
slice. The estimate and the instructions are mutually inconsistent, so the target was
unreachable as specified.

What was done instead of quietly shrinking the bios to hit a number:

- Windows are sized from the **measured miss distances** rather than round numbers: PAY RATIO
  +1,000 (covers WMT's +430 and PFE's +206 with margin), SAY ON PAY +2,500 (covers XOM's +2,057).
  A flat +3,000 each would have spent 2,500 chars per filing reaching nothing measured as
  reachable.
- `AUDITOR NAME` cut to 1,800 chars — it is adjacent to its landmark, so a wide window buys
  nothing, and it is pure addition to the payload.
- Overlap between A slices was measured before assuming it: **447 chars, 1% of the A payload,
  on 11 of 64 filings.** Deduplication is not the missing lever.
- The remaining payload is dominated by `DIRECTOR NOMINEES` at 20,000 chars — **half of it**.
  Shrinking it is the only way to reach 40,000, and it would be actively harmful: Phase 0
  measured that `board_size` / `n_directors` / `n_women_directors` already disagree between
  models *because the bios are truncated*, and Phase 3's gender upgrade needs the honorifics
  that live in those bios.

**Recommendation for the Phase 6 gate: re-set G10 to ≤ 45,000** (met, with margin) and record
that the binding constraint is the bios window, which the plan deliberately keeps.

**DECISION — approved 2026-09-02: G10 is ≤ 45,000.** Applied in
`scripts/compare_def14a_baseline.py`, `PHASE-0-baseline-harness.md` and
`PHASE-6-cutover-and-report.md`.

Why 45,000 and not 50,000, since the latter would also pass: the baseline is **51,198**, so a gate
at 50,000 certifies only "not worse than the code being replaced" and would tolerate 18.4% growth
over the measured 42,225 — the whole 18% reduction this phase claims could be lost with the gate
still green. 45,000 leaves **6.6%**, which absorbs a different filing mix when the corpus grows
from 64 filings / 23 tickers to the universe, and nothing more: no later phase adds to this payload
(Phase 4 only *removes* the HTML block, Phase 5 carves 8-K text on a separate budget).

The concrete regression the threshold has to catch is the three **fallback** slices —
`EXECUTIVE COMPENSATION` 7,000 + `SECURITY OWNERSHIP` 10,000 + `AUDITOR FEES` 2,500 = **19,500
chars** — which are emitted *only when the table classifier finds nothing*. That is not
hypothetical: this phase hit exactly that failure once, when an lxml encoding-declaration
`ValueError` hidden behind a bare `except` found **0 tables on 63 of 64 filings**. A 45,000 gate
trips on it; a 50,000 gate may not, because the fallbacks partly displace the TSV blocks they
replace. G10 is the only gate positioned to notice this, since table recall is measured
separately and could break at the same time and for the same reason.

If a later need genuinely requires more, raise the gate *then*, with the measurement that
justifies it — a ratchet pre-loosened for hypothetical growth is not a ratchet.

## Verification

All of it runs off Phase 0's on-disk filing cache — **zero network, zero LLM cost**, so it is cheap
enough to re-run on every edit.

- [x] **New `tests/data_extract/structure/test_def14a_tables.py`** — the recall harness.
      Ground truth is a checked-in JSON: per cached filing, for each of the 5 targets, either
      `null` (genuinely absent) or **one identifying value** from the real table (e.g. the CEO's
      salary, a named director's total fees, the total audit fee, Vanguard's percent of class).
      Assert the classifier finds the right table **and that the identifying value appears in the
      serialized TSV**. Finding "a table" is not the test — the research's naive signature found
      tables and picked the wrong ones 23 times out of 25 on fees.
- [x] Print the recall matrix as the sanity conclusion, in the same shape as the table above, so it
      is directly comparable to follow-up 3's numbers.
- [x] **Payload assertion**: mean serialized payload across the cached corpus ≤ **45,000** chars
      — **re-set from 40,000, approved 2026-09-02**; the 36,544 estimate it came from predates this
      phase's own widenings (see the deviation above). Measured 42,225. Print mean / median / max.
- [x] **Prerequisite regression tests**, as their own cases with real fixtures:
      - `<br>` separator: the Agilent 2005 cell `1,000,000<br>1,000,000<br>925,000` must yield three
        cells, never `10000001000000925000`.
      - `<sup>` strip: PG's `217,956,036<sup>2</sup>` must yield `217956036`, never `2179560362`.
      - `$`-in-own-`<td>`: GE's `['$','0','$','345,795','$','0','$','345,795']` must map to
        4 numeric columns aligned to `['CASH FEES','STOCK AWARDS','ALL OTHER COMP','TOTAL']`.
      - multi-row header: a `Stock` / `Awards ($)` split header must merge into one column.
      - `rowspan`: a table with a `rowspan=2` first cell must not shift subsequent rows left.
- [x] **Anchor-fix tests**: the say-on-pay TOC-floor case reproduces on **A 2015**, not A-2016
      (the research's label; A 2016's result sits at 35.4% and is unaffected). A 2015's
      say-on-pay content hit is at **5.0%** of the document: with `toc_frac=0.05` the floor
      rejected it and jumped to **44.6%**, with `toc_frac=0.0` it lands at 4.8% and the block
      carries a real `97%`. The floor was deleting an early "voting matters" result exactly as
      the plan predicted.
      `last_occurrence=True` is REMOVED from the auditor anchor, and the auditor-NAME slice is
      now positioned by finding one of B's fee-table VALUES in the flattened text — which maps
      the HTML-side match to the text offset with no second regex. It fires on 41 of 64 filings
      (it needs B's table plus a locatable landmark).
- [x] **End-to-end LLM probe** (~10 calls, on cached filings): CAT 2026, PFE 2026, GE 2026, JPM 2026,
      WMT, TSLA, T 2011, A 2016 — confirm the SCT now yields ≥ 3 NEOs × the years shown, and that a
      director-comp table is present where it exists. `n_neos == 1` must be gone from these.
- [x] `"$PY" -m pytest tests/data_extract/structure/ -q` green.

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

## POST-PHASE FIX — the SCT signature admitted another Item 402 table

Found by Phase 3's paid probe (PG 2026 returned 0 NEOs while every other field populated), not by
this phase's own recall harness.

PG's `Outstanding Equity at Fiscal Year End` grid satisfies the SCT rule: "Fiscal **Year** End"
supplies the `year` token and its `Stock Awards` / `Option Awards` column groups supply the
mandated-column token. It then **outscored the real SCT on the data-row tie-break**, so the TSV
handed to the model was an equity-holdings table. The model was right to return nothing; the
filing stored `n_neos = 0`.

Why the recall harness missed it: recall is measured as "an identifying value from the right table
appears in the serialized TSV", and a *wrong table that is also a table* passes every check that
asks whether a table was found. Only asking the model produced the disagreement.

Fixed with an `_OTHER_402_MARKERS` must-reject clause — the exact regulatory column labels of
Items 402(f)/(g)/(h)/(i) (`outstanding equity`, `unexercised options`, `have not vested`,
`option expiration`, `equity incentive plan awards`, `value realized`, `shares acquired on`,
`years credited service`, `present value of accumulated`, `aggregate earnings`,
`aggregate withdrawals`, `executive contributions`). Each was chosen so it cannot occur in a
402(c) header: note that `non-equity incentive plan compensation` contains "equity incentive
plan" but never "equity incentive plan **awards**", so the SCT itself is not rejected.

Measured after the fix: PG returns **8 NEOs × 3 fiscal years**, and its carve payload falls
**49,040 → 44,007** — the wrong table was also the larger one, so G10's mean improves.
