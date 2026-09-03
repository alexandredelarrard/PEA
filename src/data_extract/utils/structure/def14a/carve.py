"""
carve.py  (src/data_extract/utils/structure/def14a/carve.py)
-------------------------------------------------------------
Anchor a proxy statement and cut it down to the `=== LABEL ===` blocks the model reads.

LLM extraction is far cheaper AND more accurate on narrow, precise text, so each section is
capped tightly to just its data: ~51k chars versus the old ~122k, i.e. ~2.5x cheaper and
faster with no loss of the target fields.

Two thresholds here are load-bearing and measured, not tuned:
  `_TOC_SKIP_FRAC`      the table-of-contents floor -- an anchor in the first 5% of a proxy
                        is the TOC entry, not the section
  `_DIRECTOR_MAX_FRAC`  the director-window ceiling
"""
from __future__ import annotations

import logging
import re

from src.data_extract.utils.structure.def14a.tables import (
    AUDIT_FEES, DIRECTOR_COMP, OWNERSHIP_5PCT, OWNERSHIP_INSIDER, SCT,
    classify_filing, to_tsv,
)

logger = logging.getLogger(__name__)

# Per-section char budgets. LLM extraction is far cheaper AND more accurate on narrow,
# precise text, so each section is capped tightly to just its data (the Summary Comp
# Table and pay-ratio/auditor lines are compact; only the director bios run long). Total
# ~51k chars vs the old ~122k -> ~2.5x cheaper / faster with no loss of the target fields.
_CONTEXT_PRE = 500        # chars of context before the anchor match

# Content-based patterns that reliably mark the START of each section's actual data.
# These match real content (biographical text, table rows, share tables) rather than
# section headings, which appear in the TOC and in cross-references throughout the doc.

# Director bios: "Director since YEAR" or "Age:" label in a bio block
_DIRECTOR_CONTENT_RE = re.compile(
    r"\bDirector\s+since\s+(?:19|20)\d{2}\b"
    r"|\bAge[:\s]+\d{2}\b"
    r"|\bhas\s+served\s+as.*?(?:Chief Executive|President|Chairman|Chief Financial)",
    re.I,
)
# EXECUTIVE Summary Compensation Table. PRIMARY anchor: the SEC-mandated "Summary Compensation
# Table" title that is actually FOLLOWED (within ~1.8k chars) by a NEO data row — a fiscal year
# then two large dollar figures (salary + an award column). The data-row lookahead is what
# separates the real table from the many TOC / CD&A / pay-vs-performance references to the same
# title (which are not followed by SCT rows). This is far more format-robust than a header-cluster
# regex (some proxies' "Salary" column isn't adjacent to a marker, or a CD&A "target pay" table
# mimics the header). FALLBACK (`_COMPENSATION_CONTENT_RE`): the column-header cluster — "Salary"
# immediately followed by a COLUMN MARKER (($)/footnote digit/Bonus/Stock/Option/Awards) then a
# plain "Total" (negative-lookahead skips "Total Cash/Direct/Realized" realized-pay tables) and an
# "Awards" column — for the rare filing whose SCT title doesn't survive flattening.
_COMPENSATION_TITLE_RE = re.compile(
    r"Summary\s+Compensation\s+Table"
    r"(?=[\s\S]{0,1800}?\b20\d\d\b[\s\S]{0,25}?[\d,]{6,}[\s\S]{0,25}?[\d,]{5,})", re.I)
_COMPENSATION_CONTENT_RE = re.compile(
    r"\bSalary\b\s*(?:\(\$\)|\$|\d|Bonus|Stock|Option|Awards)"
    r"(?=[\s\S]{0,400}?\bTotal\b(?!\s*(?:Cash|Direct|Realized)))(?=[\s\S]{0,400}?Awards)", re.I)
# Ownership: the target line is "all directors and executive officers as a group"; else a
# beneficial-ownership row (a share count of 5+ digits then a percent). Requiring the
# "beneficially"/"as a group" context avoids false matches on $-amount narratives.
_OWNERSHIP_CONTENT_RE = re.compile(
    r"directors\s+and\s+(?:executive\s+)?officers\s+as\s+a\s+group"
    r"|beneficial(?:ly)?\s+own[\s\S]{0,400}?\b\d[\d,]{4,}\b\s+(?:\*|\d+(?:\.\d+)?\s*%)",
    re.I,
)

# ---- Densest-window row tokens (primary anchoring for the two tabular sections) ----
# Proxies scatter the director bios / beneficial-ownership table in different places and formats
# (a summary matrix, per-director blocks, footnotes), so a single "first match" anchor is fragile.
# Instead, anchor each of these sections on the region with the HIGHEST concentration of its row
# pattern — which is, by construction, the table itself (see `_densest_window`). Lone prose hits
# ("independent director since 2005", "communicate with the directors as a group") never cluster.
# The (?![\d.]|,\d) guard stops a 2-digit "age" from matching the start of a larger number —
# critical because a director-COMPENSATION fee row ("Smith, 45,000") would otherwise look like
# "Smith, 45" and pull the densest window onto the fees table instead of the bios. It rejects a
# following digit/decimal or a thousands comma ("45,000") but still allows a grammatical trailing
# comma ("Alice Johnson, 58, has served ...").
_DIRECTOR_ROW_RE = re.compile(
    r"\bAge\b[:*\s]{0,4}(?:4\d|5\d|6\d|7\d|8\d)(?![\d.]|,\d)"   # "Age 62" / "Age: 70" / "Age** 60"
    r"|\b[A-Z][a-zA-Z]+,\s+(?:4\d|5\d|6\d|7\d|8\d)(?![\d.]|,\d)"  # "Douglas, 62" rows (name, age)
    r"|\bDirector\s+Since\b",                                     # "Director Since" column / label
    re.I,
)
_OWNERSHIP_ROW_RE = re.compile(
    r"\b\d[\d,]{4,}\b\s*(?:\(\d+\)\s*)?(?:\*|\d{1,2}(?:\.\d+)?\s*%)"   # share count + percent/'*'
    r"|\b(?:BlackRock|Vanguard|State\s+Street|FMR\s+LLC|T\.?\s*Rowe\s+Price|"
    r"Capital\s+(?:Research|Group)|Wellington|Massachusetts\s+Financial|Dodge\s*&\s*Cox)\b"  # 5% holders
    r"|\bas\s+a\s+group\b",                                            # insider summary row
    re.I,
)

# Fallback text anchors (for filings that don't have rich content patterns)
_DIRECTOR_ANCHORS = (
    "nominees for election",
    "election of directors",
    "director nominees",
    "our board of directors",
    "board of directors",
)
_COMPENSATION_ANCHORS = (
    "summary compensation table",
    "executive compensation",
    "named executive officers",
)
_OWNERSHIP_ANCHORS = (
    "security ownership of certain",
    "security ownership of management",
    "beneficial ownership",
)
# prose sections (no reliable numeric content pattern -> anchor-only)
_GOVERNANCE_ANCHORS = (
    "corporate governance",
    "board leadership structure",
    "governance highlights",
    "board composition",
    "director independence",
)
# pay ratio / median pay, say-on-pay support, and auditor fees live in DIFFERENT parts of
# the proxy, so each gets its OWN small slice (one bundled slice can't reach all three).
_PAYRATIO_ANCHORS = (
    "ceo pay ratio",
    "pay ratio",
    "ratio of the annual total compensation",
    "median employee",
)
_SAYONPAY_ANCHORS = (
    "say-on-pay",
    "say on pay",
    "advisory vote to approve",
    "advisory vote on executive comp",
    "% of the votes cast",
)
_AUDITOR_ANCHORS = (
    "audit fees",
    "fees billed",
    "fees paid to",
    "audit and non-audit fees",
    "independent registered public accounting firm",
)
# say-on-pay result: a percent QUALIFIED as a vote result ("NN% of votes cast", "NN% approval",
# "NN% support", "NN% in favor") that is preceded within ~200 chars by a say-on-pay / advisory-vote
# / executive-compensation phrase. The context requirement anchors the slice on the say-on-pay
# result specifically — without it, an unrelated vote result (e.g. a director election % earlier in
# the doc) is matched first; the old "NN% … votes" pattern also missed "received 91% approval".
# Auditor fees stay a compact $ context.
_SAYONPAY_CONTENT_RE = re.compile(
    r"(?:say[- ]on[- ]pay|advisory\s+vote|executive\s+compensation)[\s\S]{0,200}?"
    r"\d{2}(?:\.\d+)?\s*%\s*(?:of\s+(?:the\s+)?(?:votes?|shares?)\s*(?:cast|voted)"
    r"|in\s+favor|approval|support|were\s+voted)",
    re.I,
)
_AUDITOR_CONTENT_RE = re.compile(r"\baudit\s+fees\b[^\n]{0,20}?[\$\d]", re.I)
# CEO pay-ratio: the heading anchors ("pay ratio") also match the comp-philosophy narrative,
# so anchor on the SEC-mandated disclosure sentence instead — "median … employee" or "ratio of
# the … compensation" — which sits with both the ratio (NNN to 1) and the median-pay $ figure.
_PAYRATIO_CONTENT_RE = re.compile(
    r"median\s+(?:of\s+(?:all\s+)?(?:our\s+)?(?:global\s+)?)?(?:employee|associate|colleague|teammate)"
    r"|\bratio\s+of\s+the\s+(?:annual\s+)?(?:total\s+)?(?:annual\s+)?compensation",
    re.I,
)

#: Front-matter fraction a section match must clear. Applied to the CONTENT-REGEX path as well
#: as the anchor fallback -- NEM / ROK / SYK carves landed at 3.0% / 2.0% / 3.7% of the document
#: while their pay ratio sat at 49-92%. Sections whose target legitimately appears early pass
#: `toc_frac=0.0` instead (see `_ANCHOR_SECTIONS`).
_TOC_SKIP_FRAC = 0.05
#: Ceiling on where the DIRECTOR NOMINEES window may START, as a fraction of the document.
#: `_TOC_SKIP_FRAC` is a FLOOR only, so the row-token optimum was free to land anywhere -- and on
#: two of 22 measured filings it landed in the BACK half: T's at 53.6% (the pension-assumptions
#: discussion, 0 of 11 directors in the block, every `is_independent` null) and EOG's at 70.0%
#: (change-of-control prose, 1 of 10). Both blocks are full of "Age"/"Director Since" tokens, so
#: no density or bio-marker score separates them from a real roster; their POSITION does. The
#: other 20 filings' windows sit at 3.3%-29.1%, so 0.40 clears the widest real case by 11 points.
#: The ceiling cannot move a filing whose window is already below it -- the picker takes the
#: EARLIEST maximal window, so discarding later candidates leaves that choice standing -- and
#: measured: exactly EOG and T move, 191 -> 203 of 243 directors land in the block.
#:
#: Per-section, NOT global: the ownership fallback's table legitimately sits in the back half.
_DIRECTOR_MAX_FRAC = 0.40
#: A fee-table cell that is a plausible dollar figure, used only to locate the table in the
#: flattened text. Deliberately requires a comma group or a decimal so a footnote marker or a
#: year cannot be picked as the landmark.
_FEE_VALUE_RE = re.compile(r"[\d,]*\d,\d{3}[\d,]*(?:\.\d+)?|\d+\.\d+")
#: Chars before / after the fee table for the auditor-NAME slice. The firm name is in the
#: sentence introducing the table, so most of the budget goes BEFORE it. Kept small on purpose:
#: it is ADJACENT to the landmark, so a wide window buys nothing and this slice is pure
#: addition to the payload (it did not exist in the research's 36,544-char hybrid estimate).
_AUDITOR_NAME_PRE = 1_200
_AUDITOR_NAME_CHARS = 1_800

#: The five targets `def14a_tables` owns, with the block label the prompt reads.
_TABLE_TARGETS = (
    ("SUMMARY COMPENSATION TABLE", SCT),
    ("DIRECTOR COMPENSATION TABLE", DIRECTOR_COMP),
    ("AUDIT FEE TABLE", AUDIT_FEES),
    ("FIVE PERCENT HOLDERS", OWNERSHIP_5PCT),
    ("INSIDER OWNERSHIP", OWNERSHIP_INSIDER),
)

#: Anchor-carve sections: (label, densest-window re, content re, anchors, last_occurrence,
#: toc_frac, budget, max_frac).
#:
#: Budgets are sized from the MEASURED miss distances, not by widening everything: only 5 of 25
#: narrative misses were within ~3,000 chars of the slice end (HSIC +123, PFE +190, WMT +430,
#: PFE-median +206, XOM say-on-pay +2,057), and everything else was 10k-500k chars away -- a
#: distance no budget fixes. So PAY RATIO gains 1,000 (covers WMT's +430 and PFE's +206 with
#: margin) and SAY ON PAY gains 2,500 (covers XOM's +2,057); a flat +3,000 on each would spend
#: 2,500 chars per filing to reach nothing that was measured as reachable.
#: `SAY ON PAY` carries `toc_frac=0.0` because its result legitimately appears in an
#: early "voting matters" summary (A-2016's sits at 4.2% of the document).
#:
#: `AUDITOR FEES` lost `last_occurrence=True`: it overshot in both directions (WMT 3,066 chars
#: BEFORE the fee table, HUBB / ROK 149k / 165k away) and B now owns the table anyway, so this
#: window only has to reach the PROSE disclosures.
_ANCHOR_SECTIONS = (
    ("DIRECTOR NOMINEES",      _DIRECTOR_ROW_RE,  _DIRECTOR_CONTENT_RE,     _DIRECTOR_ANCHORS,     False, _TOC_SKIP_FRAC, 20_000, _DIRECTOR_MAX_FRAC),
    ("CORPORATE GOVERNANCE",   None,              None,                     _GOVERNANCE_ANCHORS,   False, _TOC_SKIP_FRAC,  6_000, None),
    ("PAY RATIO & MEDIAN PAY", None,              _PAYRATIO_CONTENT_RE,     _PAYRATIO_ANCHORS,     False, _TOC_SKIP_FRAC,  4_500, None),
    ("SAY ON PAY",             None,              _SAYONPAY_CONTENT_RE,     _SAYONPAY_ANCHORS,     False, 0.0,             4_500, None),
    # ---- fallbacks: emitted only when the table classifier found nothing ----
    ("EXECUTIVE COMPENSATION", None, (_COMPENSATION_TITLE_RE, _COMPENSATION_CONTENT_RE), _COMPENSATION_ANCHORS, False, _TOC_SKIP_FRAC, 7_000, None),
    ("SECURITY OWNERSHIP",     _OWNERSHIP_ROW_RE, _OWNERSHIP_CONTENT_RE,    _OWNERSHIP_ANCHORS,    False, _TOC_SKIP_FRAC, 10_000, None),
    ("AUDITOR FEES",           None,              _AUDITOR_CONTENT_RE,      _AUDITOR_ANCHORS,      False, _TOC_SKIP_FRAC,  2_500, None),
)


def _densest_window(
    text: str,
    row_re: re.Pattern,
    chars: int,
    context_pre: int = _CONTEXT_PRE,
    min_rows: int = 3,
    max_frac: float | None = None,
) -> int:
    """Return the start of the `chars`-wide window holding the MOST `row_re` matches
    (the table), `context_pre` chars before its first row — or -1 if fewer than
    `min_rows` matches exist (caller then falls back to `_find_content_section`).

    Robust to proxies that scatter the director / ownership data or format it as a
    matrix vs per-person blocks: the densest cluster of row tokens IS the table,
    whereas isolated prose mentions never accumulate.

    `max_frac` bounds how far INTO the document the window may start, for a section that is
    always in the front matter. Density alone cannot reject a back-half block: T's winner sat at
    53.6% in the pension-assumptions discussion and EOG's at 70.0% in change-of-control prose,
    both dense in "Age"/"Director Since" tokens (see `_DIRECTOR_MAX_FRAC`). Left None for every
    section whose target may legitimately be anywhere.
    """
    limit = len(text) if max_frac is None else int(len(text) * max_frac)
    starts = [m.start() for m in row_re.finditer(text) if m.start() <= limit]
    if len(starts) < min_rows:
        return -1
    best_start, best_n = starts[0], 0
    for i, p in enumerate(starts):
        n = 0
        for q in starts[i:]:
            if q <= p + chars:
                n += 1
            else:
                break
        if n > best_n:
            best_n, best_start = n, p
    return max(0, best_start - context_pre)


def _find_content_section(
    text: str,
    content_re: "re.Pattern | tuple[re.Pattern, ...] | None",
    fallback_anchors: tuple[str, ...],
    context_pre: int = _CONTEXT_PRE,
    last_occurrence: bool = False,
    toc_frac: float = _TOC_SKIP_FRAC,
) -> int:
    """Find the start of a section using content-based regex(es).

    `content_re` may be a single pattern or a TUPLE of patterns tried in priority
    order (the first that matches wins — used for compensation: prefer the SCT
    title+data-row anchor, else the header-cluster fallback). Returns a position
    `context_pre` chars before the first (or last, if `last_occurrence=True`) match
    so the LLM sees full context. When `content_re` is None (prose sections) or no
    content match is found, falls back to text-anchor scanning.

    `toc_frac` is the front-matter fraction a match must clear, and it now applies to the
    CONTENT-REGEX path as well as the anchor fallback. It used to guard only the fallback,
    which is how the NEM / ROK / SYK carves landed at 3.0% / 2.0% / 3.7% of the document while
    the pay ratio they were looking for sat at 49-92%. Pass `toc_frac=0.0` for a section whose
    target legitimately appears in the front matter — SAY ON PAY does, because filers summarise
    last year's result in an early "voting matters" panel, and A-2016's sits at 4.2%. A floor
    is only correct for the sections that are NEVER in the front matter.
    """
    patterns = (() if content_re is None
                else content_re if isinstance(content_re, tuple) else (content_re,))
    min_pos = int(len(text) * toc_frac) if toc_frac > 0 else 0
    for pattern in patterns:
        matches = [m for m in pattern.finditer(text) if m.start() >= min_pos]
        if matches:
            m = matches[-1] if last_occurrence else matches[0]
            return max(0, m.start() - context_pre)

    # Fallback: text anchors, skipping early TOC hits
    low = text.lower()
    anchor_min = max(5000, min_pos) if toc_frac > 0 else 0
    for anchor in fallback_anchors:
        start = anchor_min
        while True:
            p = low.find(anchor, start)
            if p == -1:
                break
            after = text[p + len(anchor): p + len(anchor) + 200]
            letter_ratio = sum(c.isalpha() for c in after[:100]) / max(len(after[:100]), 1)
            if letter_ratio > 0.20:
                return max(0, p - context_pre)
            start = p + 1
    return -1


def _auditor_name_slice(text: str, fee_table: tuple[list[str], list[list[str]]] | None) -> int:
    """Start of a narrative slice positioned on the FEE TABLE, for the auditor's firm name.

    The firm name is not in the fee table's cells -- it is in the sentence just before it
    ("fees billed by PricewaterhouseCoopers LLP") or in cell [0][0]. Position is derived from
    the table itself by finding one of its comma-grouped values in the flattened text, which
    maps the HTML-side match onto the text-side offset with no second regex.

    This replaces `last_occurrence=True` on the auditor anchor, which overshot in BOTH
    directions -- WMT's carve landed 3,066 chars BEFORE the fee table, HUBB's and ROK's
    149k / 165k chars away.
    """
    if not fee_table:
        return -1
    values = [c for row in fee_table[1] for c in row if _FEE_VALUE_RE.fullmatch(c or "")]
    for value in sorted(values, key=len, reverse=True)[:6]:
        pos = text.find(value)
        if pos != -1:
            return max(0, pos - _AUDITOR_NAME_PRE)
    return -1


def prepare_def14a_sections(html: str, text: str) -> str:
    """Return a focused subset of the DEF 14A, routing each target to the strategy that
    actually reaches it.

    Two strategies, per the measured recall:

    * **Tables (B)** own the five tabular targets -- Summary Compensation Table, director
      compensation, audit fees, >=5% holders, insider ownership. `def14a_tables` classifies
      the filing's `<table>` elements on their header signature and serializes the winner as
      TSV. Measured 308/320 = 96.2% over 64 cached filings, mean 5,316 chars for all five.
    * **Anchors (A)** own the two PROSE targets (director bios, corporate governance) and the
      two narrative numbers (pay ratio, say-on-pay), and stand in as the FALLBACK whenever B
      finds no table -- which is not a rare path: EOG, GE, JPM and PEG all disclose their audit
      fees in narrative prose, so B reaches only 81% of fee disclosures by construction.

    Why B matters most: the director-compensation table is A's structural blind spot and always
    was. It sits at 23-36% of the document, in the gap between A's `DIRECTOR NOMINEES` window
    (ends ~17-21%) and `EXECUTIVE COMPENSATION` (starts ~42-64%), median miss distance 70,761
    chars. A found it in 2 of 25 filings; no budget widening reaches it.

    `=== LABEL ===` blocks, the same convention the prompt references. Never emits nothing for
    a target A could have reached.
    """
    tables = classify_filing(html) if html else {}
    parts: list[str] = []
    from_b: list[str] = []
    from_a: list[str] = []

    for label, target in _TABLE_TARGETS:
        if target in tables:
            parts.append(f"\n\n=== {label} ===\n{to_tsv(*tables[target])}")
            from_b.append(label)

    # A's EXECUTIVE COMPENSATION / SECURITY OWNERSHIP / AUDITOR FEES windows are now
    # FALLBACKS -- emitted only when B found no table for that target. Dropping them on the
    # happy path is where most of the payload saving comes from (three windows, 19,500 chars).
    fallbacks = {
        "EXECUTIVE COMPENSATION": SCT not in tables,
        "SECURITY OWNERSHIP": OWNERSHIP_INSIDER not in tables and OWNERSHIP_5PCT not in tables,
        "AUDITOR FEES": AUDIT_FEES not in tables,
    }

    for label, dense_re, content_re, anchors, use_last, toc_frac, chars, max_frac in _ANCHOR_SECTIONS:
        if label in fallbacks and not fallbacks[label]:
            continue
        pos = (_densest_window(text, dense_re, chars, max_frac=max_frac)
               if dense_re is not None else -1)
        if pos == -1:
            pos = _find_content_section(text, content_re, anchors,
                                        last_occurrence=use_last, toc_frac=toc_frac)
        if pos == -1:
            continue
        parts.append(f"\n\n=== {label} ===\n{text[pos:min(len(text), pos + chars)]}")
        from_a.append(label)

    # the auditor's NAME, positioned on B's fee table (its own small slice, because the name
    # sits in the sentence before the table rather than in its cells)
    name_pos = _auditor_name_slice(text, tables.get(AUDIT_FEES))
    if name_pos != -1:
        end = min(len(text), name_pos + _AUDITOR_NAME_CHARS)
        parts.append(f"\n\n=== AUDITOR NAME ===\n{text[name_pos:end]}")
        from_a.append("AUDITOR NAME")

    payload = "".join(parts) if parts else text[:120_000]
    missing = [lbl for lbl, _ in _TABLE_TARGETS if lbl not in from_b]
    # This log line is the diagnostic that tells you a FORMAT ERA broke: a target that
    # silently moves from B to A across a filer's template change shows up here first.
    logger.debug("def14a carve: B=%s | A=%s | no-table=%s | %d chars",
                 ",".join(from_b) or "-", ",".join(from_a) or "-",
                 ",".join(missing) or "-", len(payload))
    return payload

