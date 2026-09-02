"""
fetch_def14a_llm.py  (src/data_extract/utils/fetch_def14a_llm.py)
-----------------------------------------------------------------
Extract structured governance data from SEC DEF 14A proxy statements using an
LLM with structured output (Def14AExtract schema).

Per ticker, it fetches that ticker's DEF 14A filings from EDGAR, sends targeted
sections to the OpenAI Responses API (constrained to the Def14AExtract Pydantic
schema, prompt caching on), then **immediately upserts that
ticker's rows into the `def14a_llm` Postgres table** before moving to the next
ticker — so an interrupted run never loses the (expensive) LLM calls already made.

Per-filing incremental (gap-filling): each ticker's FULL `years_history` window of DEF 14A
filings is listed, and the LLM is (re-)run ONLY on filings whose `accession_number` is NOT already
in the `def14a_llm` table. So any MISSING year/filing — including a hole in the middle of the
history, not just after the latest — is filled, while every already-extracted filing is skipped
(no repeat LLM cost). Tickers with no rows yet get the whole window; already-complete tickers make
no LLM calls at all.

Requires OPENAI_API_KEY (or OPEN_AI_API_KEY) in the .env file.
If the key is absent the function logs a warning and returns whatever exists.

Output columns (DB table `def14a_llm`), scalar summaries + raw JSON:
    keys        ticker, as_of, period, accession_number, company_name, fiscal_year_extract
    board       n_directors, board_size, avg_director_age, avg_board_tenure,
                pct_independent_directors, pct_female_directors,
                avg_other_public_boards, pct_gender_stated,
                n_women_directors_vs_inferred
    ceo         ceo_name_proxy, ceo_age, ceo_since_year, ceo_is_founder,
                ceo_is_board_chair, ceo_salary, ceo_bonus, ceo_stock_awards,
                ceo_option_awards, ceo_non_equity_incentive, ceo_all_other_comp,
                ceo_total_comp, ceo_equity_pay_pct
    neos        n_neos, total_neo_comp, sct_years
    ownership   insider_ownership_pct, ceo_ownership_pct, n_five_percent_holders,
                n_ownership_rows
    governance  independent_chair, lead_independent_director, classified_board,
                dual_class_shares, poison_pill, majority_voting,
                say_on_pay_support_pct, ceo_pay_ratio, median_employee_pay
    auditor     auditor_name, auditor_since_year, auditor_fees, audit_fees_audit,
                audit_fees_audit_related, audit_fees_tax, audit_fees_other,
                auditor_fees_prior
    counts      n_director_comp_rows, n_ownership_rows
    def14a_json (full Def14AExtract as JSON for downstream use)

`n_technology_directors` / `pct_technology_directors` / `technology_committee` were REMOVED:
they were an opinion, not an extraction (mean |delta| of 1.06 directors between consecutive
filings of the same company, only 38.8% unchanged).

FOUR CHILD TABLES are written alongside, flattened out of the same paid extract:
    def14a_executive_comp   one row per NEO per fiscal year (Item 402(c), ~3 years/filing)
    def14a_director_comp    one row per non-employee director (Item 402(k), single-year)
    def14a_ownership        one row per beneficial holder (Item 403)
    def14a_directors        one row per director -- and the substrate the cross-filing gender
                            consensus pass groups over (see def14a_gender.py)
"""
from __future__ import annotations

import logging
import re

import pandas as pd
from tqdm import tqdm

from src.constants.constants import DATE_FORMAT, DEF14A_FORMS
from src.context import Context
from src.data_extract.utils.structure.def14a_gender import (
    basis_distribution, consensus, log_consensus, recompute_parent_gender,
)
from src.data_extract.utils.structure.def14a_schema import Def14AExtract
from src.data_extract.utils.structure.def14a_validate import (
    DEF14A_AUDIT_FEE_MIN_PLAUSIBLE, clean_holder_name, clean_person_name, clean_text,
    is_subtotal_holder, rescale_block, sum_fee_total,
)
from src.data_extract.utils.structure.def14a_tables import (
    AUDIT_FEES, DIRECTOR_COMP, OWNERSHIP_5PCT, OWNERSHIP_INSIDER, SCT,
    classify_filing, to_tsv,
)
from src.data_extract.utils.common.edgar_extract import html_to_text
from src.data_extract.utils.common.frame_sanitize import strip_nul
from src.data_extract.utils.common.edgar_fillings import list_filings
from src.data_extract.utils.common.llm_extractor import LLMExtractor
from src.data_extract.utils.common.run_manifest import get_entry, manifest_window, record_run
from src.data_extract.utils.common.sec_utils import existing_filings, load_cik_mapping, sec_get
from src.data_store.schema import Tables

# DEF 14A = the shareholder proxy; DEF 14C = the equivalent INFORMATION STATEMENT that
# CONTROLLED companies file instead (no vote solicited because a controlling holder already has
# the votes, e.g. ERIE = Hirt trusts). Same governance / exec-comp content, so both are extracted.
# Centralized in constants.py as DEF14A_FORMS (form_registry.py's single source of truth).
_FORM = DEF14A_FORMS

# flattened output columns that must be stored numeric (float) in the DB
# (governance booleans are surfaced as 1.0/0.0 flags so they are usable features)
_NUMERIC_COLS = [
    "fiscal_year_extract",
    # board / directors
    "n_directors", "board_size", "avg_director_age", "avg_board_tenure",
    "pct_independent_directors", "pct_female_directors", "avg_other_public_boards",
    "pct_gender_stated", "n_women_directors_vs_inferred",
    # CEO
    "ceo_age", "ceo_since_year", "ceo_is_founder", "ceo_is_board_chair",
    "ceo_salary", "ceo_bonus", "ceo_stock_awards", "ceo_option_awards",
    "ceo_non_equity_incentive", "ceo_all_other_comp", "ceo_total_comp", "ceo_equity_pay_pct",
    # NEO aggregate
    "n_neos", "total_neo_comp", "sct_years",
    # child-table row counts (so a silent recall regression is visible on the parent row)
    "n_director_comp_rows", "n_ownership_rows",
    # ownership
    "insider_ownership_pct", "ceo_ownership_pct", "n_five_percent_holders",
    # governance provisions
    "independent_chair", "lead_independent_director", "classified_board",
    "dual_class_shares", "poison_pill", "majority_voting", "say_on_pay_support_pct",
    "ceo_pay_ratio", "median_employee_pay", "auditor_fees",
    # auditor name is TEXT; the rest of the fee block is numeric
    "auditor_since_year", "audit_fees_audit", "audit_fees_audit_related", "audit_fees_tax",
    "audit_fees_other", "auditor_fees_prior",
]

# Task-tailored extraction prompt (cached per model+schema -> stays cheap). Precise
# instructions on WHERE each field lives and how to normalise it materially lift the
# fill rate versus a generic "extract structured data" prompt.
_DEF14A_PROMPT = (
    "You extract structured governance & compensation data from a SEC DEF 14A proxy (or the "
    "equivalent DEF 14C information statement filed by controlled companies). The input is a set "
    "of `=== LABEL ===` blocks. Blocks named SUMMARY COMPENSATION TABLE, DIRECTOR COMPENSATION "
    "TABLE, AUDIT FEE TABLE, FIVE PERCENT HOLDERS and INSIDER OWNERSHIP are TAB-SEPARATED tables "
    "with a header line first — use the header to identify each column. The other blocks are "
    "narrative text.\n"
    "- SUMMARY COMPENSATION TABLE: return EVERY (executive x fiscal year) row the table shows — "
    "it normally carries three years per executive. Take `fiscal_year` from the Year column. A "
    "'-', '—' or blank cell is 0. There are SEVEN dollar components: salary, bonus, stock awards, "
    "option awards, non-equity incentive, CHANGE IN PENSION VALUE, all other compensation. Do NOT "
    "confuse this table with the Pay-versus-Performance table (its column says 'compensation "
    "actually paid') or with the DIRECTOR compensation table.\n"
    "- CEO pay: the CEO's SCT row for the MOST RECENT fiscal year.\n"
    "- DIRECTOR COMPENSATION TABLE: one row per director. `fees_earned_usd` is the cash-retainer "
    "column whatever it is labelled ('Fees Earned or Paid in Cash', 'Cash Fees', 'Retainer'); "
    "`stock_awards_usd` covers 'Stock Awards', 'Restricted Stock Units' or 'Share Awards'.\n"
    "- Board composition: read the governance/board 'highlights' summary for board_size, "
    "n_independent_directors and n_women_directors (e.g. '7 of our 8 directors are independent').\n"
    "- Directors: resolve `gender` from the proxy's own statement, else the HONORIFIC used for "
    "that director, else the PRONOUNS in their bio, else the first name — and set `gender_basis` "
    "to whichever of 'stated'/'honorific'/'pronoun'/'name' you used. Never leave `gender_basis` "
    "null when `gender` is set.\n"
    "- Provisions: classified_board and dual_class_shares are STRUCTURALLY always disclosed, so "
    "return FALSE when the proxy does not indicate them. For poison_pill and "
    "majority_voting_for_directors return TRUE or FALSE only when the proxy STATES the "
    "provision's status, and null when the proxy is SILENT — do not infer FALSE from silence.\n"
    "- Ownership: insider_ownership_pct = the 'all directors and executive officers AS A GROUP' "
    "percent; ceo_ownership_pct = the CEO's own row; both as decimals (a '*' or '<1%' -> null). "
    "Both percents must come from the PERCENT OF THE CLASS of shares outstanding (economic "
    "ownership). Dual-class issuers print a '% of total voting power' / 'combined voting power' "
    "column beside it — NEVER take that one. n_five_percent_holders = count of owners "
    "holding >=5%.\n"
    "- ownership_holders: one entry per HOLDER row across both ownership blocks. Exclude subtotal "
    "and 'as a group' rows, and exclude a row whose name is only a street address. "
    "`percent_of_class` is null for '*' / '<1%'.\n"
    "- Auditor: `auditor_name` is the accounting FIRM NAME only, never a sentence. Report the four "
    "fee categories for the CURRENT year plus the prior-year TOTAL. Every fee must be WHOLE USD — "
    "apply any '(in thousands)' or '($ in millions)' note in the table header or the sentence "
    "before it (a table reading 57.6 under '($ in millions)' is 57,600,000).\n"
    "- say_on_pay_support_pct as a decimal (92% -> 0.92); ceo_pay_ratio as a number (533:1 -> 533).\n"
    "Only use values stated in the text; use null when genuinely absent (except "
    "classified_board / dual_class_shares above)."
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


def _latest_sct_year(extract: Def14AExtract) -> int | None:
    """The most recent fiscal year present in the SCT rows, or None when no row carries one."""
    years = [c.fiscal_year for c in extract.compensation if c.fiscal_year is not None]
    return max(years) if years else None


def _latest_sct_rows(extract: Def14AExtract) -> list:
    """The SCT rows for the most recent fiscal year only.

    The schema now asks for every (NEO x year) cell, so three derived values MUST be restricted
    to one year or they silently change meaning: the CEO row would come from an arbitrary year,
    `n_neos` would triple (making the `n_neos == 1` metric meaningless), and `total_neo_comp`
    would sum three years of pay. Rows with no `fiscal_year` are kept as a fallback -- a filing
    whose Year column did not parse still has to yield a CEO row.
    """
    latest = _latest_sct_year(extract)
    if latest is None:
        return list(extract.compensation)
    return [c for c in extract.compensation if c.fiscal_year == latest]


def _ceo_from_compensation(extract: Def14AExtract) -> "ExecutiveCompensation | None":  # noqa: F821
    """The CEO's Summary-Compensation-Table row for the MOST RECENT fiscal year: match on the
    extracted CEO name first, else on a CEO-like title, else the first (usually highest-paid)
    NEO of that year."""
    rows = _latest_sct_rows(extract)
    ceo_name = (extract.ceo_name or "").strip().lower()
    if ceo_name:
        for c in rows:
            if (c.name or "").strip().lower() == ceo_name:
                return c
    ceo_kws = ("chief executive", "ceo", "president and chief")
    for c in rows:
        if any(kw in (c.title or "").lower() for kw in ceo_kws):
            return c
    return rows[0] if rows else None


def _ceo_age(extract: Def14AExtract) -> int | None:
    """CEO age: the explicit top-level field, else looked up by CEO name in the
    directors list (the CEO is almost always a director nominee)."""
    if extract.ceo_age is not None:
        return extract.ceo_age
    name = (extract.ceo_name or "").strip().lower()
    if name:
        for person in extract.directors:
            if (person.name or "").strip().lower() == name and person.age is not None:
                return person.age
    return None


def _bnum(x: bool | None) -> float | None:
    """Bool -> 1.0/0.0 (numeric flag for the feature store); None stays None."""
    return None if x is None else float(bool(x))


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 3) if xs else None


def _flatten(ticker: str, filing: pd.Series, extract: Def14AExtract) -> dict:
    dirs = extract.directors
    ages = [d.age for d in dirs if d.age is not None]
    tenures = [d.tenure_years for d in dirs if d.tenure_years is not None]
    genders = [(d.gender or "").strip().lower() for d in dirs if d.gender]
    other_boards = [d.other_public_company_boards for d in dirs
                    if d.other_public_company_boards is not None]
    ceo = _ceo_from_compensation(extract)
    latest_rows = _latest_sct_rows(extract)
    gov = extract.governance
    g = lambda a: getattr(gov, a, None) if gov is not None else None  # noqa: E731
    ceo_name = (extract.ceo_name or (ceo.name if ceo else None) or "")

    ceo_equity = None
    if ceo and ceo.total_compensation_usd:
        equity = (ceo.stock_awards_usd or 0) + (ceo.option_awards_usd or 0)
        ceo_equity = round(equity / ceo.total_compensation_usd, 3)

    board_size = g("board_size") or (len(dirs) or None)
    # board composition: prefer the DIRECT governance-highlights counts (robust to text
    # trimming), fall back to computing from the per-director list.
    n_indep = g("n_independent_directors")
    if n_indep is not None and board_size:
        pct_independent = round(n_indep / board_size, 3)
    elif dirs and any(d.is_independent is not None for d in dirs):
        pct_independent = round(sum(bool(d.is_independent) for d in dirs) / len(dirs), 3)
    else:
        pct_independent = None
    # gender provenance: `stated` and `honorific` are DOCUMENT evidence, `name` is a bare
    # prior. Only 17.4% of proxies state gender at all, so this ratio is what makes
    # `pct_female_directors` auditable instead of silently inferred.
    bases = [(d.gender_basis or "").strip().lower() for d in dirs if d.gender]
    pct_gender_stated = (round(sum(b in ("stated", "honorific") for b in bases) / len(bases), 3)
                         if bases else None)
    n_female_inferred = sum(x.startswith("f") for x in genders)

    n_women = g("n_women_directors")
    if n_women is not None and board_size:
        pct_female = round(n_women / board_size, 3)
    elif genders:
        pct_female = round(sum(x.startswith("f") for x in genders) / len(genders), 3)
    else:
        pct_female = None

    row = {
        "ticker": ticker,
        "as_of": filing["filing_date"],
        "period": pd.to_datetime(filing.get("period_of_report"), errors="coerce"),
        "accession_number": filing["accession_number"],
        "company_name": extract.company_name,
        "fiscal_year_extract": extract.fiscal_year,
        # ---- Board / directors ----
        "n_directors": len(dirs) or None,
        "board_size": board_size,
        "avg_director_age": _mean(ages),
        "avg_board_tenure": _mean(tenures),
        "pct_independent_directors": pct_independent,
        "pct_female_directors": pct_female,
        "avg_other_public_boards": _mean(other_boards),
        # ---- CEO ----
        "ceo_name_proxy": ceo_name or None,
        "ceo_age": _ceo_age(extract),
        "ceo_since_year": extract.ceo_since_year,
        "ceo_is_founder": _bnum(extract.ceo_is_founder),
        "ceo_is_board_chair": _bnum(extract.ceo_is_board_chair
                                    if extract.ceo_is_board_chair is not None else g("ceo_is_board_chair")),
        "ceo_salary": ceo.salary_usd if ceo else None,
        "ceo_bonus": ceo.bonus_usd if ceo else None,
        "ceo_stock_awards": ceo.stock_awards_usd if ceo else None,
        "ceo_option_awards": ceo.option_awards_usd if ceo else None,
        "ceo_non_equity_incentive": ceo.non_equity_incentive_usd if ceo else None,
        "ceo_all_other_comp": ceo.all_other_comp_usd if ceo else None,
        "ceo_total_comp": ceo.total_compensation_usd if ceo else None,
        "ceo_equity_pay_pct": ceo_equity,
        # ---- NEO aggregate (most recent fiscal year only -- see `_latest_sct_rows`) ----
        "n_neos": len({(c.name or "").strip().lower() for c in latest_rows if c.name}) or None,
        "total_neo_comp": sum(c.total_compensation_usd for c in latest_rows
                              if c.total_compensation_usd is not None) or None,
        # how many fiscal years the SCT actually yielded. Item 402(c) requires three, so a 1
        # here is the "the carve missed most of the table" pathology -- still measurable after
        # the schema change that stops `n_neos` from tripling.
        "sct_years": len({c.fiscal_year for c in extract.compensation
                          if c.fiscal_year is not None}) or None,
        # ---- Ownership / alignment (direct from the beneficial-ownership summary) ----
        "insider_ownership_pct": g("insider_ownership_pct"),
        "ceo_ownership_pct": g("ceo_ownership_pct"),
        "n_five_percent_holders": g("n_five_percent_holders"),
        # ---- Governance provisions ----
        "independent_chair": _bnum(g("independent_chair")),
        "lead_independent_director": _bnum(g("lead_independent_director")),
        "classified_board": _bnum(g("classified_board")),
        "dual_class_shares": _bnum(g("dual_class_shares")),
        "poison_pill": _bnum(g("poison_pill")),
        "majority_voting": _bnum(g("majority_voting_for_directors")),
        "say_on_pay_support_pct": g("say_on_pay_support_pct"),
        "ceo_pay_ratio": g("ceo_pay_ratio"),
        "median_employee_pay": g("median_employee_pay_usd"),
        # ---- auditor: name, tenure, fee breakdown (2.05% fill on the retired edgar path) ----
        "auditor_name": clean_text(g("auditor_name") or "") or None,
        "auditor_since_year": g("auditor_since_year"),
        "auditor_fees": g("auditor_fees_usd"),
        "audit_fees_audit": g("audit_fees_audit_usd"),
        "audit_fees_audit_related": g("audit_fees_audit_related_usd"),
        "audit_fees_tax": g("audit_fees_tax_usd"),
        "audit_fees_other": g("audit_fees_other_usd"),
        "auditor_fees_prior": g("auditor_fees_prior_usd"),
        # ---- child-table row counts + gender provenance ----
        "n_director_comp_rows": len(extract.director_compensation) or None,
        "n_ownership_rows": len(extract.ownership_holders) or None,
        # share of the board whose gender came from the DOCUMENT rather than a first-name prior
        "pct_gender_stated": pct_gender_stated,
        # the filing's own women-director count minus the count derived from per-director
        # gender. 0 when they agree; a persistent non-zero is the honest error bar on the
        # inference. Never used to overwrite anything.
        "n_women_directors_vs_inferred": (None if n_women is None or not genders
                                          else n_women - n_female_inferred),
        # Full JSON for downstream access. Keeping it is what makes RE-flattening free: the
        # four row builders and every derived column above are pure functions of this blob, so
        # a schema change can be replayed over 8,667 stored filings without an LLM call.
        "def14a_json": extract.model_dump_json(),
    }
    # Safety net BEHIND the prompt's unit instruction, not instead of it. A filer reports every
    # cell of its fee table in one unit, so the block is rescaled together or not at all --
    # rescaling cell-by-cell would invent a table whose categories no longer sum to the total.
    # Fires only when the LARGEST fee in the block is still implausibly small for an S&P 500
    # audit, which means an "(in thousands)" / "($ in millions)" note was missed (measured on 8
    # of the 10 smallest values: MS 57.6 = $57.6M, TSLA 10,919 = $10.9M).
    rescale_block(row, list(_FEE_COLS), DEF14A_AUDIT_FEE_MIN_PLAUSIBLE)
    # AFTER the rescale, so both sides of the comparison are in whole dollars. Recovers the total
    # on the fee tables that have no Total row, where the model reports the `Audit Fees` line as
    # the total (BA 39.1M -> 43.6M, T 34.2M -> 38.9M).
    sum_fee_total(row, "auditor_fees", list(_FEE_CATEGORY_COLS))
    return row


def _fetch_filing_html(context: Context, filing: pd.Series) -> str:
    """The filing's raw markup, retrying the `<accession>.txt` full submission when the primary
    document 404s.

    `primaryDocument` names a file that is genuinely ABSENT from the archive on 7 of 663
    measured DEF 14A filings (all 2000-08..2001-03, all naming `"0001.txt"`). Those produce no
    row at all without this retry, because the raise propagates out of `_process_filing` and
    the filing is silently skipped -- a loss invisible in the "pre-2001 rows are NULL" count
    since there is no row to be null. The `.txt` carries the real proxy (53,661-165,380 chars
    on the four spot-checked).
    """
    try:
        return sec_get(context, filing["doc_url"]).text
    except Exception:
        txt_url = filing.get("txt_url")
        if not txt_url or txt_url == filing["doc_url"]:
            raise
        logger.info("%s: primary document unavailable, falling back to the full submission",
                    filing.get("accession_number", ""))
        return sec_get(context, txt_url).text



# --------------------------------------------------------------------------- #
# Child-table row builders                                                    #
#                                                                             #
# All four are PURE functions of (ticker, filing, Def14AExtract), which is what makes them      #
# replayable over the already-stored `def14a_json` for free -- the tokens are paid, so the      #
# flatten can be verified over 445 real filings without a single LLM call.                      #
# --------------------------------------------------------------------------- #
#: `total - sum(components)` within this many dollars counts as reconciling. A FLAG, not a
#: filter (D10): values are kept either way, and the failure rate becomes measurable over time.
#: $10 absorbs the filers who round each component to the nearest dollar independently.
_RECONCILE_TOLERANCE_USD = 10.0

#: The seven SCT components (Item 402(c)) and the six director-comp components (Item 402(k)),
#: in the flat column vocabulary. Deliberately the SAME names the retired edgar child tables
#: used, so the Phase-6 before/after comparison is column-for-column.
_EXEC_COMPONENT_COLS = ("salary", "bonus", "stock_awards", "option_awards",
                        "non_equity_incentive", "pension_change", "other_compensation")
_DIRECTOR_COMPONENT_COLS = ("fees_earned", "stock_awards", "option_awards",
                            "non_equity_incentive", "pension_change", "other_compensation")
#: The fee block is rescaled TOGETHER or not at all -- a filer reports every cell of one table
#: in one unit, so a cell-by-cell rescale would invent a table whose parts no longer sum.
_FEE_COLS = ("auditor_fees", "audit_fees_audit", "audit_fees_audit_related",
             "audit_fees_tax", "audit_fees_other", "auditor_fees_prior")
#: The four Item 9(e) categories that make up the current-year total — `_FEE_COLS` minus the
#: total itself and minus the prior year, whose categories this schema does not carry.
_FEE_CATEGORY_COLS = ("audit_fees_audit", "audit_fees_audit_related",
                      "audit_fees_tax", "audit_fees_other")


def _keys(ticker: str, filing: pd.Series) -> dict:
    """The point-in-time key stamp every child row carries. `as_of` is the FILING date, never a
    period end -- that is what keeps the tables leak-free."""
    return {
        "ticker": ticker,
        "cik": str(filing.get("cik") or "") or None,
        "accession_number": filing["accession_number"],
        "as_of": filing["filing_date"],
    }


def _reconciles(row: dict, components: tuple[str, ...]) -> float | None:
    """1.0 when the components sum to `total` within `_RECONCILE_TOLERANCE_USD`, else 0.0; None
    when `total` is absent.

    A FLAG rather than a repair. The old edgar path filled a single missing component from the
    residual, which is no longer sound: with `pension_change` now in the schema the residual is
    not an unattributable gap, and filling it would overwrite a real column.
    """
    total = row.get("total")
    if total is None or not isinstance(total, (int, float)):
        return None
    parts = [row.get(c) for c in components]
    if not any(isinstance(v, (int, float)) for v in parts):
        return None
    summed = sum(float(v) for v in parts if isinstance(v, (int, float)))
    return 1.0 if abs(float(total) - summed) <= _RECONCILE_TOLERANCE_USD else 0.0


def _exec_comp_rows(ticker: str, filing: pd.Series, extract: Def14AExtract) -> list[dict]:
    """One row per (NEO x fiscal year) of the Summary Compensation Table.

    34,741 such rows already sat unqueryable inside `def14a_llm.def14a_json`, against the
    retired edgar table's 2,378 on 25 tickers -- and on every measurable axis the LLM rows are
    better: title 100% vs 45.4%, stock awards 93.8% vs 45.4%, option awards 88.0% vs 27.5%, and
    2 rows above $1e9 versus 109.
    """
    rows = []
    for c in extract.compensation:
        name = clean_person_name(c.name)
        if not name or c.fiscal_year is None:
            # `fiscal_year` is Optional on the Pydantic model but PART OF THIS TABLE'S PRIMARY
            # KEY, so a null aborts the whole Postgres insert -- not one row. It is also a
            # useless row: comp that cannot be placed in time. 0 of 1,849 replayed rows lack
            # one, but that is evidence, not a guarantee, so the guard is structural.
            continue
        row = {
            **_keys(ticker, filing),
            "name": name,
            "title": clean_text(c.title or "") or None,
            "fiscal_year": c.fiscal_year,
            "salary": c.salary_usd,
            "bonus": c.bonus_usd,
            "stock_awards": c.stock_awards_usd,
            "option_awards": c.option_awards_usd,
            "non_equity_incentive": c.non_equity_incentive_usd,
            "pension_change": c.pension_change_usd,
            "other_compensation": c.all_other_comp_usd,
            "total": c.total_compensation_usd,
        }
        row["reconciles"] = _reconciles(row, _EXEC_COMPONENT_COLS)
        rows.append(row)
    return rows


def _director_comp_rows(ticker: str, filing: pd.Series, extract: Def14AExtract) -> list[dict]:
    """One row per non-employee director (Item 402(k)).

    Single-year BY REGULATION -- 402(k) requires only the last completed fiscal year -- and
    membership here IS the definition of an outside director, which Phase 5's vote role map
    depends on. Absent from the Pydantic contract entirely before this phase.
    """
    rows = []
    for d in extract.director_compensation:
        name = clean_person_name(d.name)
        if not name:
            continue
        row = {
            **_keys(ticker, filing),
            "name": name,
            "fiscal_year": d.fiscal_year,
            "fees_earned": d.fees_earned_usd,
            "stock_awards": d.stock_awards_usd,
            "option_awards": d.option_awards_usd,
            "non_equity_incentive": d.non_equity_incentive_usd,
            "pension_change": d.pension_change_usd,
            "other_compensation": d.all_other_comp_usd,
            "total": d.total_compensation_usd,
        }
        row["reconciles"] = _reconciles(row, _DIRECTOR_COMPONENT_COLS)
        rows.append(row)
    return rows


def _ownership_rows(ticker: str, filing: pd.Series, extract: Def14AExtract) -> list[dict]:
    """One row per beneficial holder (Item 403).

    Knowingly redundant with 13F / SC 13D-G / Forms 3-4-5, which stay the preferred sources.
    Two row shapes are dropped rather than stored: an "as a group" subtotal (that aggregate is
    already the `insider_ownership_pct` scalar) and a cell that is only a street address.
    """
    rows = []
    for h in extract.ownership_holders:
        if is_subtotal_holder(h.holder_name):
            continue
        name = clean_holder_name(h.holder_name)
        if not name:
            continue
        holder_type = (h.holder_type or "").strip().lower() or None
        if holder_type not in ("5pct_holder", "director_officer", None):
            holder_type = "5pct_holder" if (h.percent_of_class or 0) >= 0.05 else "director_officer"
        rows.append({
            **_keys(ticker, filing),
            "holder_name": name,
            "holder_type": holder_type or "director_officer",
            "shares": h.shares,
            "percent_of_class": h.percent_of_class,
        })
    return rows


def _director_rows(ticker: str, filing: pd.Series, extract: Def14AExtract) -> list[dict]:
    """One row per director per filing -- the `directors[]` array, flattened.

    The most trustworthy block in the extract: 99.74% of names appear verbatim in the source,
    93% of ages and 98% of tenures are confirmable, and a full hand-check of HUBB 2022 was 27/27
    correct including the public-vs-private board judgements. `gender_basis` is what makes the
    gender field auditable, and the cross-filing consensus pass CANNOT be written without this
    table -- it needs a GROUP BY over people, across tickers and years.
    """
    rows = []
    for d in extract.directors:
        name = clean_person_name(d.name)
        if not name:
            continue
        gender = (d.gender or "").strip().lower() or None
        basis = (d.gender_basis or "").strip().lower() or None
        rows.append({
            **_keys(ticker, filing),
            "name": name,
            "age": d.age,
            "tenure_years": d.tenure_years,
            "is_independent": _bnum(d.is_independent),
            "gender": gender,
            # never leave the provenance blank when a gender is set: an unlabelled value is
            # indistinguishable from the first-name prior this upgrade exists to expose
            "gender_basis": basis or ("name" if gender else None),
            "other_public_company_boards": d.other_public_company_boards,
        })
    return rows


def _child_frames(ticker: str, filing: pd.Series, extract: Def14AExtract) -> dict:
    """All four child frames for one filing, keyed by table name."""
    return {
        "def14a_executive_comp": _exec_comp_rows(ticker, filing, extract),
        "def14a_director_comp": _director_comp_rows(ticker, filing, extract),
        "def14a_ownership": _ownership_rows(ticker, filing, extract),
        "def14a_directors": _director_rows(ticker, filing, extract),
    }


def _process_filing(
    context: Context, ticker: str, filing: pd.Series, extractor: LLMExtractor
) -> tuple[dict, dict] | None:
    """(parent row, child frames) for one filing, or None when the extraction failed."""
    try:
        raw_html = _fetch_filing_html(context, filing)
        text = html_to_text(raw_html)
        focused = prepare_def14a_sections(raw_html, text)
        extract = extractor.extract(Def14AExtract, focused, instructions=_DEF14A_PROMPT)
        return _flatten(ticker, filing, extract), _child_frames(ticker, filing, extract)
    except Exception as e:
        logger.warning("%s %s: DEF 14A LLM extraction failed (%s)",
                       ticker, filing.get("filing_date", ""), e)
        return None


def _is_up_to_date(context: Context, requested_tickers: list[str]) -> bool:
    """Up to date only when EVERY requested ticker already has rows in the DB AND
    the shared extraction manifest (`run_manifest.py`) was refreshed today. The old
    check compared a DATE + a stored COUNT (`universe_size`), so a same-day rerun
    skipped tickers that were never actually extracted -- the '~15 tickers then it
    stops' bug. Checking per-ticker coverage (tickers x date) makes a rerun pick up
    the still-missing names; the per-ticker loop then skips already-done filings
    via `seen` (no re-LLM)."""
    if not context.store.exists(Tables.def14a_llm):
        return False
    entry = get_entry(context, Tables.def14a_llm)
    if entry is None or entry.get("last_run_date") != pd.Timestamp.today().strftime(DATE_FORMAT):
        return False
    have = set(context.store.load(Tables.def14a_llm, columns=["ticker"])["ticker"].dropna())
    return set(requested_tickers).issubset(have)


#: Numeric columns on the child tables, and the primary key each is deduped on. The PK must
#: match `schema.py`'s registration exactly -- a mismatch here silently drops rows on a filing
#: that lists the same person twice.
_CHILD_SPEC = {
    "def14a_executive_comp": (
        ("fiscal_year", "salary", "bonus", "stock_awards", "option_awards",
         "non_equity_incentive", "pension_change", "other_compensation", "total", "reconciles"),
        ["ticker", "accession_number", "name", "fiscal_year"]),
    "def14a_director_comp": (
        ("fiscal_year", "fees_earned", "stock_awards", "option_awards", "non_equity_incentive",
         "pension_change", "other_compensation", "total", "reconciles"),
        ["ticker", "accession_number", "name"]),
    "def14a_ownership": (
        ("shares", "percent_of_class"),
        ["ticker", "accession_number", "holder_name", "holder_type"]),
    "def14a_directors": (
        ("age", "tenure_years", "is_independent", "other_public_company_boards"),
        ["ticker", "accession_number", "name"]),
}
_CHILD_TABLES = {
    "def14a_executive_comp": Tables.def14a_executive_comp,
    "def14a_director_comp": Tables.def14a_director_comp,
    "def14a_ownership": Tables.def14a_ownership,
    "def14a_directors": Tables.def14a_directors,
}


def _prepare_frame(rows: list[dict], numeric: tuple[str, ...], pk: list[str]) -> pd.DataFrame:
    """Rows -> a save-ready frame: numeric columns coerced, NULs stripped, `as_of` normalised,
    duplicates collapsed on the table's own primary key."""
    df = pd.DataFrame(rows)
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = strip_nul(df)                          # Postgres TEXT rejects NUL (\x00)
    df["as_of"] = pd.to_datetime(df["as_of"]).dt.normalize()
    return df.drop_duplicates(subset=[c for c in pk if c in df.columns], keep="last")


def _save_ticker_rows(context: Context, rows: list[dict],
                      children: dict[str, list[dict]] | None = None) -> int:
    """Upsert one ticker's freshly-extracted parent + child rows right away.

    LLM calls are expensive, so everything for a ticker is persisted before the next ticker
    starts -- an interrupted run loses no paid tokens. The child frames are written FIRST so a
    crash between the two leaves a child row without a parent (recoverable, because the
    accession dedup keys on `def14a_llm`) rather than a parent that claims children it lacks.
    """
    written = 0
    for name, child_rows in (children or {}).items():
        if not child_rows:
            continue
        numeric, pk = _CHILD_SPEC[name]
        written += context.store.save(_CHILD_TABLES[name],
                                      _prepare_frame(child_rows, numeric, pk), pk=pk)

    df = _prepare_frame(rows, tuple(_NUMERIC_COLS), ["ticker", "accession_number"])
    return written + context.store.save(Tables.def14a_llm, df)



def _finalise_gender(context: Context) -> None:
    """Cross-ticker gender consensus, run ONCE after the per-ticker loop.

    It cannot live inside the loop: a director recurs across COMPANIES as well as years, so the
    consensus needs every ticker's rows before it can group on people. Cheap on a routine rerun
    -- DEF 14A is a yearly filing, so an incremental day adds ~0 rows and the pass is a narrow
    read plus a no-op write.

    Both reads are projected to the three columns the consensus needs; both writes carry only
    the columns they change (AGENTS.md: never read a large table unprojected).
    """
    directors = context.store.load(
        Tables.def14a_directors,
        columns=["ticker", "accession_number", "name", "as_of", "gender", "gender_basis"],
        optional=True)
    if directors is None or directors.empty:
        return

    before = basis_distribution(directors)
    resolved, stats = consensus(directors)
    after = basis_distribution(resolved)
    log_consensus(context.log, stats, before, after)

    if not (stats["filled"] or stats["overturned"]):
        return   

    context.store.save(Tables.def14a_directors,
                       resolved[["ticker", "accession_number", "name", "as_of",
                                 "gender", "gender_basis"]],
                       pk=["ticker", "accession_number", "name"])

    # `pct_female_directors` keeps its existing precedence -- the filing's own
    # `n_women_directors` first, this ratio as the FALLBACK -- so a consensus correction makes
    # the fallback better rather than overriding a stated count.
    parent = recompute_parent_gender(resolved)
    if not parent.empty:
        context.store.save(Tables.def14a_llm, parent, pk=["ticker", "accession_number"])
        context.log.info("gender consensus: refreshed pct_female_directors / pct_gender_stated "
                         "on %d filings", len(parent))


def fetch_def14a_llm(
    context: Context,
    tickers: list[str],
    model: str,
    max_chars: int = 130_000,
    cache: bool = True,
) -> None:
    """Build/refresh the DEF 14A LLM governance extract, one ticker at a time.

    For each ticker only filings AFTER its latest stored `as_of` are sent to the
    LLM (year-incremental), and the ticker's rows are upserted to Postgres
    immediately. Skips gracefully when OPENAI_API_KEY is absent.

    `model` is REQUIRED and has no default on purpose: both callers pass
    `config.data_extract.llm_model`, and a second default here is how the research came to
    measure `gpt-4o-mini` while production had been running `gpt-5-mini` all along.
    """
    years = context.config.data_extract.years_history
    de = context.config.data_extract

    cik_map = load_cik_mapping(context, tickers)

    if _is_up_to_date(context, cik_map["ticker"].tolist()):
        existing = context.store.load(Tables.def14a_llm)
        context.log.info("DEF 14A LLM already up to date — every requested ticker present "
                         "(%d rows) — skipping", len(existing))
        return existing

    # accessions already extracted -> never re-LLM (accession-only dedup, same convention as
    # fetch_8k_edgar.py / fetch_13d_edgar.py / fetch_def14a_edgar.py's `existing_filings`)
    # a mutable copy: this fetcher is serial and adds each accession as it extracts it,
    # so a ticker filing twice in one run is not sent to the LLM twice
    seen = set(existing_filings(context, Tables.def14a_llm))

    # Manifest-driven listing window (see run_manifest.py): a routine run only lists
    # filings from the last run's date onward; a ticker-count change or the
    # `manifest_full_rescan_days` self-heal window falls back to the FULL `years`
    # window (gap-filling, same self-heal rationale as the other 4 EDGAR fetchers).
    # `list_filings`'s own `since` cutoff is STRICTLY AFTER the date passed, so we
    # step back one day to keep the last run's date itself inclusive.
    rescan_days = int(getattr(de, "manifest_full_rescan_days", 30))
    manifest_since, is_full_rescan = manifest_window(
        context, Tables.def14a_llm, len(cik_map),
        fallback_since=pd.Timestamp.today() - pd.DateOffset(years=years),
        full_rescan_days=rescan_days)
    list_since = None if is_full_rescan else (manifest_since - pd.Timedelta(days=1))

    try:
        extractor = LLMExtractor(model=model, max_chars=max_chars, cache=cache)
    except EnvironmentError as e:
        context.log.warning("DEF 14A LLM extraction skipped: %s", e)
        existing = context.store.load(Tables.def14a_llm, optional=True)
        return existing if existing is not None else pd.DataFrame(columns=["ticker", "as_of"])

    total_new, tickers_touched, total_skipped = 0, 0, 0
    for _, r in tqdm(cik_map.iterrows(), total=len(cik_map), desc="DEF 14A LLM"):
        ticker, cik, company = r["ticker"], r["cik"], r.get("company_name", "")
        # `list_since=None` (full-rescan runs) lists the FULL years_history window so a MISSING
        # filing anywhere in the history is discovered; otherwise only filings from the manifest's
        # last run date onward are listed. The accession skip below then sends ONLY the
        # not-yet-stored filings to the LLM (gap-filling, per ticker / per date).
        try:
            filings = list_filings(context, cik, _FORM, years, company, since=list_since)
        except Exception as e:
            context.log.warning("%s: DEF 14A filing list failed (%s)", ticker, e)
            continue

        ticker_rows: list[dict] = []
        ticker_children: dict[str, list[dict]] = {name: [] for name in _CHILD_SPEC}
        skipped = 0
        for _, f in filings.iterrows():
            if f["accession_number"] in seen:      # already in the table -> skip this filing
                skipped += 1
                continue
            result = _process_filing(context, ticker, f, extractor)
            if result is not None:
                row, children = result
                ticker_rows.append(row)
                for name, child_rows in children.items():
                    ticker_children[name].extend(child_rows)
                seen.add(f["accession_number"])
        total_skipped += skipped

        # persist THIS ticker before moving on (don't batch — LLM calls are costly)
        if ticker_rows:
            _save_ticker_rows(context, ticker_rows, ticker_children)
            total_new += len(ticker_rows)
            tickers_touched += 1
            context.log.info("%s: +%d new DEF 14A filing(s) sent to the LLM (%d already in table)",
                             ticker, len(ticker_rows), skipped)

    # A cross-ticker consensus needs every ticker's rows, so this is the only thing that
    # cannot run inside the loop. Skipped entirely when nothing new was extracted.
    if total_new:
        _finalise_gender(context)

    record_run(context, Tables.def14a_llm, len(cik_map), total_new, is_full_rescan=is_full_rescan)
   