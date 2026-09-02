"""
def14a_tables.py  (src/data_extract/utils/structure/def14a_tables.py)
---------------------------------------------------------------------
Locate a DEF 14A's five TABULAR targets by parsing its `<table>` elements and classifying each
on its HEADER SIGNATURE, instead of guessing the table's position with a text anchor and a fixed
character budget.

Measured on 25 filings / 8,628 tables / 8,580 ground-truth cells, against the anchor carve:

    target                          ground truth   anchor carve   table-anchored
    Summary Compensation Table            24            20             25
    Director compensation table           25             2             24
    Insider / group ownership             24            22             22
    >=5% holders                          22            21             25
    Audit fee table                       23            20             25

**121/125 (96.8%)**, one false positive, and never a semantically different table. The director
compensation table is the anchor carve's structural blind spot and always was: it sits at 23-36%
of the document, in the gap between the `DIRECTOR NOMINEES` window (ends ~17-21%) and
`EXECUTIVE COMPENSATION` (starts ~42-64%), median miss distance **70,761 chars**. No budget
widening reaches it.

Payload falls at the same time: a serialized table averages 4,283 chars against the 7,000-char
window the anchor carve spends to *sometimes* reach one.

PURE: no I/O, no `context`, no LLM. That is what lets the recall harness run off a disk cache of
filings on every edit, and cheap is what makes a harness get run.

Three cell-level defects are fixed here because they corrupt CLASSIFICATION, not just values:
  * `<br>` / block boundaries MUST emit a separator -- dropping it fuses 180 cells (2.1%) across
    16 of 25 filings (`All othercompensation`, `James DimonChairman and CEO`), and JPM's SCT was
    invisible to the classifier until block `<div>` boundaries separated.
  * `<sup>` must be stripped BEFORE the cell text is read -- 47 cells (0.5%), 7 of them the
    bare-digit form that corrupts a value (`$1,587,852` + superscript 6 -> `1,5878526`).
  * a `$` alone in its own `<td>` doubles the effective column count and desynchronises the
    column map (the GE / CAT mechanism, 3 of 4 numeric columns dropped).
"""
from __future__ import annotations

import html
import logging
import re
from typing import Callable, Iterable

import lxml.html

logger = logging.getLogger(__name__)

#: Target names. Two ownership targets, not one: the insider table often has NO percent column
#: while the >=5% table is defined by it, so a single signature scores 14/25 and 3/25.
SCT = "sct"
DIRECTOR_COMP = "director_comp"
AUDIT_FEES = "audit_fees"
OWNERSHIP_INSIDER = "ownership_insider"
OWNERSHIP_5PCT = "ownership_5pct"
TARGETS = (SCT, DIRECTOR_COMP, AUDIT_FEES, OWNERSHIP_INSIDER, OWNERSHIP_5PCT)

#: RUNAWAY GUARD on the header-row merge, not the rule itself -- what actually ends the header
#: is the first row carrying a money-shaped cell. Multi-row headers are how edgartools drops a
#: column: it treats exactly one row as the header and scans only `grid[:4]`. Measured need for
#: the guard to be generous: EOG 2026's SCT header spans FOUR rows
#: (`Non-Equity` / `Stock` / `Name and Fiscal Salary Awards` / `Principal Position Year ($)`),
#: and a cap of 3 cut off the row carrying `Year` -- which is the SCT signature's own
#: requirement, so the whole table went unclassified.
_MAX_HEADER_ROWS = 6
#: Runaway guard only. Max observed serialized table is 14,169 chars.
_MAX_TSV_CHARS = 20_000
#: A table with fewer data rows than this is a layout wrapper, not a data table.
_MIN_DATA_ROWS = 2

#: A cell holding only a currency glyph (optionally with a footnote marker) is a LAYOUT cell.
#: Filers put `$` in its own `<td>`, which doubles the column count and shifts the column map.
_CURRENCY_ONLY_RE = re.compile(r"^[\$€£¥]\s*(?:\(\d+\))?$")
#: Any digit run -- the cheapest "is this a data table" test.
_HAS_DIGIT_RE = re.compile(r"\d")
#: A money-shaped or percent-shaped cell -- the WHOLE cell is one number.
_NUMERIC_CELL_RE = re.compile(r"^[\$€£¥(]?\s*[\d,]+(?:\.\d+)?\s*\)?\s*%?$")
#: A cell CONTAINING a data-sized number: a comma-grouped amount or a 4+ digit run (a year).
#: This, not `_NUMERIC_CELL_RE`, is what ends the header merge. AMAT's SCT stacks all three
#: fiscal years inside ONE `<td>` per column with `<br>`, so after normalisation a single cell
#: reads `2025 2024 2023` / `1,030,000 1,030,000 990,000` and matches NO whole-cell numeric
#: test -- with a whole-cell test the header merge swallowed the entire table (0 data rows) and
#: the SCT went unclassified on all 3 AMAT filings. Header labels (`Salary ($)`, `Year`, `($)`)
#: carry no digits, so they still merge.
_DATA_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+|\d{4,}")
#: Footnote-only row label (`(1)`, `(a)`) -- the audit-fee footnote table's giveaway.
_FOOTNOTE_ROW_RE = re.compile(r"^\(?\s*[\da-z]\s*\)")
#: How many leading columns to scan for a fee-category ROW LABEL. A filer that colspans its
#: label column pushes the text right; 6 covers every case measured without reaching the
#: numeric columns.
_FEE_LABEL_COLS = 6


# --------------------------------------------------------------------------- #
# 1a. cell-grid extraction                                                    #
# --------------------------------------------------------------------------- #
#: Elements whose boundary is a visual line break. `html_to_text` already maps `<br>` -> `\n`;
#: the same must happen at CELL granularity or two stacked values fuse into one number.
_BLOCK_TAGS = frozenset({"br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"})


def _is_superscript(el) -> bool:
    """True for a `<sup>`, or for any element positioned as one by CSS.

    The CSS form is not an edge case: PG's 2026 proxy dropped `<sup>` for a
    `style="vertical-align:top"` `<span>` and the 10x share-count bug SURVIVED. Matching on the
    style is what makes the fix outlive one filer's template change.
    """
    if el.tag == "sup":
        return True
    style = (el.get("style") or "").lower().replace(" ", "")
    return "vertical-align:" in style and ("top" in style or "super" in style)


def _cell_text(el) -> str:
    """One cell's visible text, with superscripts removed and block boundaries preserved.

    Walks the subtree rather than using `.text_content()` because that method does BOTH wrong
    things at once: it concatenates across `<br>` with no separator and it includes superscript
    digits, which are the two defects that corrupt values here.
    """
    parts: list[str] = []

    def walk(node) -> None:
        for child in node:
            if isinstance(child.tag, str) and _is_superscript(child):
                # skip the superscript's text but KEEP its tail -- the tail is real cell text
                if child.tail:
                    parts.append(child.tail)
                continue
            if isinstance(child.tag, str) and child.tag in _BLOCK_TAGS:
                parts.append("\n")
            if child.text:
                parts.append(child.text)
            walk(child)
            if isinstance(child.tag, str) and child.tag in _BLOCK_TAGS:
                parts.append("\n")
            if child.tail:
                parts.append(child.tail)

    if el.text:
        parts.append(el.text)
    walk(el)
    return _normalise("".join(parts))


def _normalise(text: str) -> str:
    """The same normalisation set as `edgar_extract.html_to_text`, so the flattened-text path
    and the table path can never disagree about what a cell says.

    A block boundary inside a cell survives as `\\n` up to here and becomes a single space --
    the SEPARATOR is what matters (`1,000,000 1,000,000` not `10000001000000`), not the newline.
    """
    text = html.unescape(text or "")
    text = text.replace("\xa0", " ").replace("​", "").replace("﻿", "")
    return re.sub(r"\s+", " ", text).strip()


def _expand_row(tr, pending: dict[int, tuple[str, int]]) -> list[str]:
    """One `<tr>` as a flat cell list, expanding `colspan` and honouring `rowspan` carried down
    from earlier rows.

    edgartools ignores `rowspan` entirely, which shifts every subsequent row LEFT by one column
    for the rest of the table -- so a "Total" value lands under "All Other Compensation".
    `pending` maps column index -> (text, rows still to fill) and is mutated across rows.
    """
    cells: list[str] = []
    col = 0

    def place_pending() -> None:
        nonlocal col
        while col in pending:
            text, left = pending[col]
            cells.append(text)
            if left <= 1:
                del pending[col]
            else:
                pending[col] = (text, left - 1)
            col += 1

    place_pending()
    for td in tr.iterchildren():
        if not isinstance(td.tag, str) or td.tag not in ("td", "th"):
            continue
        text = _cell_text(td)
        try:
            colspan = max(1, min(int(td.get("colspan") or 1), 40))
            rowspan = max(1, min(int(td.get("rowspan") or 1), 40))
        except ValueError:
            colspan = rowspan = 1
        for _ in range(colspan):
            cells.append(text)
            if rowspan > 1:
                pending[col] = (text, rowspan - 1)
            col += 1
            place_pending()
    place_pending()
    return cells


def _drop_layout_cells(rows: list[list[str]]) -> list[list[str]]:
    """Remove columns that are pure layout: every cell empty, or every non-empty cell a bare
    currency glyph.

    Done column-wise rather than cell-wise on purpose -- dropping a `$` cell only where it
    happens to appear would shorten SOME rows and desynchronise the grid, which is the very
    failure being fixed. GE's `['$','0','$','345,795','$','0','$','345,795']` collapses to
    4 numeric columns that line up with `['CASH FEES','STOCK AWARDS','ALL OTHER COMP','TOTAL']`.
    """
    if not rows:
        return rows
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]
    keep = [i for i in range(width)
            if any(c and not _CURRENCY_ONLY_RE.match(c) for c in (r[i] for r in padded))]
    return [[r[i] for i in keep] for r in padded] if keep else []


def _parse(raw_html: str | bytes):
    """Parse a filing to an lxml tree, or None.

    Input is encoded to BYTES first, which is not optional: every modern inline-XBRL EDGAR
    filing opens with `<?xml version='1.0' encoding='ASCII'?>`, and lxml raises
    `ValueError: Unicode strings with encoding declaration are not supported` on a `str`
    carrying one. Measured: passing `str` found **0 tables on 63 of 64** cached 2024-2026
    filings while the raw `<table` count was 123-409; passing bytes finds every one of them
    (342/342 AAPL, 409/409 JPM, 21/21 on a pre-2001 `.txt`).

    A failure is LOGGED, never swallowed silently -- a bare `except: return []` here is exactly
    what hid the defect above behind a plausible-looking "this filing has no tables".
    """
    if not raw_html:
        return None
    data = raw_html.encode("utf-8", "replace") if isinstance(raw_html, str) else raw_html
    try:
        return lxml.html.fromstring(data)
    except Exception as e:
        logger.warning("def14a_tables: could not parse filing markup (%s: %s)",
                       type(e).__name__, e)
        return None


def iter_tables(raw_html: str | bytes) -> list[list[list[str]]]:
    """Every `<table>` in the filing as a row-major grid of cleaned cell strings.

    No recursion into nested tables: measured **zero of 8,628** DEF 14A tables contain one.
    Tables with fewer than `_MIN_DATA_ROWS` rows or no digit anywhere are dropped as layout.
    """
    root = _parse(raw_html)
    if root is None:
        return []
    for bad in root.xpath("//script | //style"):
        bad.getparent().remove(bad)

    grids: list[list[list[str]]] = []
    for table in root.xpath("//table"):
        pending: dict[int, tuple[str, int]] = {}
        rows = [_expand_row(tr, pending) for tr in table.xpath(".//tr")]
        rows = _drop_layout_cells([r for r in rows if any(c for c in r)])
        if len(rows) < _MIN_DATA_ROWS or not any(_HAS_DIGIT_RE.search(c) for r in rows for c in r):
            continue
        grids.append(rows)
    return grids


def merge_header_rows(grid: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Split a grid into (merged header, data rows).

    Leading rows are merged into one header for as long as they carry no data-sized number --
    that is what rejoins a `Stock` / `Awards ($)` split header into one column label, and it is
    driven by the content rather than by `_MAX_HEADER_ROWS`, which is only a runaway guard.
    The first row containing a comma-grouped amount or a 4-digit year starts the data.
    """
    if not grid:
        return [], []
    n_header = 0
    for row in grid[:_MAX_HEADER_ROWS]:
        if any(_DATA_NUMBER_RE.search(c) for c in row if c):
            break
        n_header += 1
    n_header = max(1, n_header)

    width = max(len(r) for r in grid)
    header = []
    for i in range(width):
        parts = [g[i] for g in grid[:n_header] if i < len(g) and g[i]]
        # de-duplicate: a colspan-expanded label repeats across its span and across rows
        seen, uniq = set(), []
        for part in parts:
            if part.lower() not in seen:
                seen.add(part.lower())
                uniq.append(part)
        header.append(" ".join(uniq))

    data = [r + [""] * (width - len(r)) for r in grid[n_header:]]
    return _drop_spacer_columns(header, data)


def _drop_spacer_columns(header: list[str], data: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Drop columns that hold no DATA, judging on the data rows only.

    `_drop_layout_cells` cannot do this: it runs before the header is identified, and a
    `colspan`-expanded header label propagates into every column it spans, so every column
    looks non-empty. PG's insider table is 20 columns of which 13 are pure spacers -- kept,
    they trebled the serialized payload and left the model to align values across a row of
    mostly empty fields.
    """
    if not data:
        return header, data
    keep = [i for i in range(len(header))
            if any((r[i] or "").strip() and not _CURRENCY_ONLY_RE.match(r[i]) for r in data)]
    if not keep:
        return header, data

    # Collapse a run of columns that is IDENTICAL in the header AND in every data row -- that is
    # a `colspan` expansion, not repeated data. GE's name column arrives 3x wide
    # (`['Stephen Angel*', 'Stephen Angel*', 'Stephen Angel*', '0', ...]`). Requiring the header
    # to match too is what keeps two genuinely equal numeric columns apart: a real repeated
    # value sits under DIFFERENT headers (`2025` vs `2024`).
    collapsed = [keep[0]]
    for i in keep[1:]:
        prev = collapsed[-1]
        if header[i] == header[prev] and all(r[i] == r[prev] for r in data):
            continue
        collapsed.append(i)
    return [header[i] for i in collapsed], [[r[i] for i in collapsed] for r in data]


# --------------------------------------------------------------------------- #
# 1b. header signatures                                                       #
# --------------------------------------------------------------------------- #
# A NAIVE signature scores the SCT 25/25 but admits 2-5 candidates in 8 of 25 filings, insider
# ownership 14/25, >=5% holders 3/25, and audit fees 2/25 with 23 WRONG picks. Strictness is the
# whole design: each signature carries a must-REJECT clause, and the fee labels are matched
# EXACTLY rather than as substrings.

def _blob(cells: Iterable[str]) -> str:
    return " | ".join(c.lower() for c in cells if c)


def _has(blob: str, *needles: str) -> bool:
    return any(n in blob for n in needles)


#: SEC-mandated SCT columns. `Total` alone is far too common to count.
_SCT_COLS = ("salary", "stock award", "option award", "non-equity incentive",
             "all other compensation", "bonus")
#: The Pay-versus-Performance table also has a Year column and dollar columns. Its giveaway
#: phrase is regulatory and exact, so rejecting on it is safe.
_PVP_MARKERS = ("compensation actually paid", "value of initial fixed",
                "peer group total shareholder")
#: A CD&A "target pay" / "realized pay" table mimics the SCT header.
_ALT_PAY_MARKERS = ("target total direct", "realized pay", "realizable pay",
                    "target annual", "target direct compensation")
#: The OTHER Item 402 tables that carry `Stock Awards` / `Option Awards` columns and the word
#: "Year", and therefore satisfy the SCT rule without being an SCT. Measured on PG's 2026 proxy:
#: its `Outstanding Equity at Fiscal Year End` table (402(f)) outscored the real SCT on data
#: rows and won the tie-break, so the model was handed an equity-holdings grid, correctly
#: returned zero compensation rows, and the filing stored `n_neos = 0`.
#: Every phrase here is a regulatory column label from 402(f)/(g)/(h)/(i) that cannot occur in a
#: 402(c) header -- note "non-equity incentive plan compensation" contains "equity incentive
#: plan" but never "equity incentive plan awards", so the SCT itself is not rejected.
_OTHER_402_MARKERS = (
    "outstanding equity", "unexercised options", "have not vested", "option expiration",
    "equity incentive plan awards", "value realized", "shares acquired on",
    "years credited service", "present value of accumulated", "aggregate earnings",
    "aggregate withdrawals", "executive contributions",
)
#: The cash-retainer column, whatever the filer calls it. `Cash Fees` and `Retainer` are the
#: labels edgartools' synonym list misses.
_DIR_FEE_COLS = ("fees earned or paid in cash", "fees earned", "cash fees", "retainer",
                 "fees paid in cash", "annual retainer")
#: `Restricted Stock Units` and `Share Awards` are what nulled CAT's and GE's stock column.
_DIR_STOCK_COLS = ("stock award", "restricted stock unit", "share award", "stock unit award")
#: EXACT fee-category labels. Substring matching is what produced 23 wrong picks.
_FEE_LABELS = ("audit fees", "audit-related fees", "audit related fees", "tax fees",
               "all other fees", "other fees")
#: Percent-column labels, measured across the corpus rather than guessed. LMT writes
#: `Percent of Outstanding Shares` and AMAT `Shares Beneficially Owned Percent`, neither of
#: which contains "of class" -- between them that was 6 of the 13 missed >=5% tables. The bare
#: "percent" / "%" tokens are safe here only because the >=5% rule ALSO requires a share column
#: and either a named institution or a 5% reference.
_PCT_COLS = ("percent of class", "% of class", "percent of shares", "percentage of class",
             "percent of common stock", "% of outstanding", "percent of outstanding",
             "owned percent", "percent", "% of total", "% owned")
#: Share-count column labels. `Amount of Common Stock` (LMT) and `Amount and Nature` are the
#: two that do not contain the word "shares".
_SHARE_COLS = ("number of shares", "shares beneficially owned", "amount and nature",
               "shares owned", "beneficial ownership", "number of common shares",
               "amount of common stock", "amount of shares", "common stock owned",
               # bare "Common Stock" is LMT's entire share-column label. Safe only because
               # the insider rule ALSO requires an "as a group" row, which is highly specific.
               "common stock")
#: Institutions that appear in essentially every >=5% table.
_INSTITUTIONS = ("blackrock", "vanguard", "state street", "fmr llc", "fidelity",
                 "t. rowe price", "capital research", "capital group", "wellington",
                 "massachusetts financial", "dodge & cox", "geode", "berkshire hathaway")


def _exact_cell_match(cells: Iterable[str], labels: tuple[str, ...]) -> int:
    """How many of `labels` appear as a WHOLE cell (footnote markers and units stripped).

    Exact-match on the cell, not a substring of the blob: "audit fees" appears inside the
    *sentence* "the following table shows audit fees billed", which is how a substring test
    picked 23 wrong tables. It also appears inside NKE's Proposal-3 narrative, which the filer
    puts in a `<td>` -- so the whole proposal text becomes one "cell" and a substring test
    matches it.
    """
    norm = set()
    for c in cells:
        c = re.sub(r"\(\s*[\d a-z]+\s*\)|\(\$\)|\$|:", " ", (c or "").lower())
        norm.add(re.sub(r"\s+", " ", c).strip())
    return sum(1 for lab in labels if lab in norm)


def _row_labels(rows: list[list[str]]) -> list[str]:
    return [(r[0] or "").lower() for r in rows if r]


def _n_numeric(rows: list[list[str]]) -> int:
    return sum(1 for r in rows for c in r if _NUMERIC_CELL_RE.match(c or ""))


def classify_table(header: list[str], rows: list[list[str]]) -> list[str]:
    """Every target this table satisfies -- a LIST, because one table can be two targets.

    Filers routinely publish ONE beneficial-ownership table holding both the >=5% institutions
    and the directors-and-officers rows: AAPL 2026's is 16 rows running `The Vanguard Group`,
    `BlackRock, Inc.`, the individual directors, then
    `All current directors and executive officers as a group (12 persons)`. Under first-match-wins
    that table registered only as `ownership_insider` and >=5% recall sat at **61%**. The same
    serialized TSV legitimately answers both targets, so it is emitted for both.
    """
    matched: list[str] = []
    hb = _blob(header)
    all_cells = [c for r in rows for c in r]
    ab = _blob(header) + " || " + _blob(all_cells[:400])
    labels = _row_labels(rows)

    # ---- audit fees: EXACT category labels, as row labels or as headers ----
    # scan the leading COLUMNS, not only column 0: NKE duplicates its row labels across the
    # colspan (`['', '', '', 'Audit Fees', 'Audit Fees', ...]`), leaving `r[0]` empty, which
    # hid its fee table on all 3 of its filings.
    lead_cells = [c for r in rows for c in r[:_FEE_LABEL_COLS]]
    n_fee = max(_exact_cell_match(header, _FEE_LABELS),
                _exact_cell_match(lead_cells, _FEE_LABELS))
    if n_fee >= 2:
        # the footnote table repeats the same labels with prose bodies and numbered rows
        footnoted = sum(1 for lab in labels if _FOOTNOTE_ROW_RE.match(lab))
        if footnoted <= len(labels) // 2 and _n_numeric(rows) >= 2:
            matched.append(AUDIT_FEES)

    # ---- SCT: a Year column AND >=1 mandated SCT column, and NOT another Item 402 table ----
    if _has(hb, "year") and _has(hb, *_SCT_COLS):
        if (not _has(ab, *_PVP_MARKERS) and not _has(hb, *_ALT_PAY_MARKERS)
                and not _has(hb, *_OTHER_402_MARKERS)):
            # a Salary column is the discriminator against the DIRECTOR table
            if _has(hb, "salary") or not _has(hb, *_DIR_FEE_COLS):
                matched.append(SCT)

    # ---- director comp: a fee column AND a stock column AND Total, and NO Salary ----
    if (_has(hb, *_DIR_FEE_COLS) and _has(hb, *_DIR_STOCK_COLS) and _has(hb, "total")
            and not _has(hb, "salary") and not _has(ab, *_PVP_MARKERS)):
        matched.append(DIRECTOR_COMP)

    # ---- ownership: two targets, and ONE table may serve both (see the docstring) ----
    has_share_col = _has(hb, *_SHARE_COLS) or _has(hb, "shares")
    has_pct_col = _has(hb, *_PCT_COLS)
    has_group_row = any("as a group" in lab for lab in labels) or _has(ab, "as a group")
    has_institution = _has(ab, *_INSTITUTIONS)
    has_5pct = _has(ab, "5%", "five percent", "5 percent")

    # insider / group table: deliberately does NOT require a percent column -- many have none
    if has_share_col and has_group_row:
        matched.append(OWNERSHIP_INSIDER)
    # >=5% holders: DEFINED by the percent column plus a named institution or a 5% reference
    if has_pct_col and has_share_col and (has_institution or has_5pct):
        matched.append(OWNERSHIP_5PCT)
    return matched


#: A header cell ending in the word `salary`, tolerating a unit marker and any number of
#: Item 402 column references after it. The reference letters are the trap: the SEC's own table
#: format labels columns `(a) (b) (c)`, so KLAC's real SCT column reads `Salary ($) (c)` and a
#: pattern allowing only `(\d+)` rejects it -- which silently handed six KLAC filings a 5-row
#: CD&A table in place of their 15-row SCT.
_SALARY_TAIL_RE = re.compile(
    r"\bsalary\b\s*(?:\(\s*[\$%]\s*\))?\s*(?:\(\s*\w{1,3}\s*\)\s*)*[\s*†‡§]*$", re.I)
#: A unit marker right after the label. Its presence is what licenses a LONG cell: a colspan'd
#: table title propagates into every cell, so a genuine column can read
#: `Summary Compensation Table Annual Compensation Salary ($)` (57 chars, 8 words).
_SALARY_UNIT_RE = re.compile(r"\bsalary\b\s*\(\s*[\$%]\s*\)", re.I)
#: Word budget for a cell with NO unit marker. Measured over every header cell of every
#: SCT-candidate table in the 656-filing cache: 46 cells end in `salary`, and the split is
#: clean -- genuine labels run 1-6 words (`Salary`, `Base Salary`,
#: `SUMMARY COMPENSATION TABLE Salary ($) (c)`), while every prose cell is 11-13 words
#: (`Year-Over-Year Percentage Increase Represented by the Fiscal Year 2013 Base Salary`,
#: 82 chars). Nothing measured falls between 7 and 10 words. A CHAR cap cannot do this job --
#: legitimate title-propagated labels reach 64 chars and the prose starts at 82.
_SALARY_MAX_WORDS = 6


def _has_salary_column(header: list[str]) -> bool:
    """Does this header carry the SCT's mandatory Salary COLUMN (not prose about salary)?

    Item 402(c)(2)(iii) makes Salary a mandatory SCT column, so a candidate carrying one is
    strictly more likely to be the SCT than one that does not. The whole difficulty is telling
    the column apart from text that merely contains -- or ends with -- the word:

      * `The salary portion of the amounts reflected above is ...` is a merged FOOTNOTE row.
        A "salary appears anywhere" test picks it over the real table on A 2011-2014, CAT
        2008-2009, EOG 2009 and PG 2013-2023.
      * `Year-Over-Year Percentage Increase Represented by the Fiscal Year 2013 Base Salary`
        is a CD&A raise table and it ENDS with the word, so end-anchoring alone is not enough
        either -- this cell is what beat KLAC's real SCT while I was measuring.
    """
    for cell in header:
        s = str(cell).strip()
        if not _SALARY_TAIL_RE.search(s):
            continue
        if _SALARY_UNIT_RE.search(s) or len(s.split()) <= _SALARY_MAX_WORDS:
            return True
    return False


#: Per-target tie-break preference, applied BEFORE the data-row count. Only the SCT needs one:
#: `all other compensation` is in `_SCT_COLS`, so the 402(c) FOOTNOTE breakout table
#: ("All Other Compensation" detail) matches the SCT rule on its own title and then wins on rows.
_PREFER: dict[str, Callable[[list[str]], bool]] = {SCT: _has_salary_column}


def classify_filing(raw_html: str | bytes) -> dict[str, tuple[list[str], list[list[str]]]]:
    """Best table per target for one filing.

    Tie-break, in order: the target's own `_PREFER` predicate (SCT only), then DATA-ROW COUNT
    (the real table is the long one; the measured false positive was page 2 of a paginated
    table), then document position (earlier wins).

    Row count alone is not enough and the failure is not rare. Measured over 656 cached filings,
    a genuine Salary-bearing SCT existed and LOST on 11 of them -- BA 2013-2021 is nine
    consecutive years where `Name and Principal Position | Year | Salary ($)` lost to a longer
    `Name | Year | Annual Incentive Compensation` CD&A table, and GE 2019 lost to a director BIO
    grid. Those filings then stored `n_neos` from the wrong table.
    """
    candidates: dict[str, list[tuple[bool, int, int, list[str], list[list[str]]]]] = {}
    for pos, grid in enumerate(iter_tables(raw_html)):
        header, rows = merge_header_rows(grid)
        if not rows:
            continue
        for target in classify_table(header, rows):
            prefer = _PREFER.get(target)
            candidates.setdefault(target, []).append(
                (bool(prefer(header)) if prefer else False, len(rows), -pos, header, rows))

    best: dict[str, tuple[list[str], list[list[str]]]] = {}
    for target, cands in candidates.items():
        cands.sort(key=lambda c: (c[0], c[1], c[2]), reverse=True)
        best[target] = (cands[0][3], cands[0][4])
        if len(cands) > 1:
            runner = cands[1]
            _log_runner_up(target, cands[0][1], runner[1], runner[3])
    return best


def _log_runner_up(target: str, n_best: int, n_runner: int, runner_header: list[str]) -> None:
    """Name the rejected candidate so a wrong pick is diagnosable from the log alone."""
    logger.debug(
        "def14a_tables: %s kept a %d-row table over a %d-row candidate (%s)",
        target, n_best, n_runner, " | ".join(runner_header[:6]))


# --------------------------------------------------------------------------- #
# 1c. TSV serialization                                                       #
# --------------------------------------------------------------------------- #
def to_tsv(header: list[str], rows: list[list[str]], max_chars: int = _MAX_TSV_CHARS) -> str:
    """Header + rows as tab-separated lines.

    TSV rather than prose because COLUMN IDENTITY is what the model keeps getting wrong (a
    dropped `stock_awards` under a `Restricted Stock Units` header), and an explicit delimiter
    makes the column boundary unambiguous. A real SCT is ~1,886 chars this way -- about a
    quarter of the 7,000-char window the anchor carve spends to sometimes reach it.
    """
    lines = ["\t".join(c.replace("\t", " ") for c in header)]
    lines += ["\t".join((c or "").replace("\t", " ") for c in row) for row in rows]
    out = "\n".join(lines)
    return out[:max_chars] if len(out) > max_chars else out
