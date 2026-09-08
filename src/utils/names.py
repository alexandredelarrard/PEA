"""
names.py  (src/utils/names.py)
--------------------------------------------------------------------------------
ONE person key, shared by the extraction that WRITES the DEF 14A / Item 5.07 tables and the
aggregation that JOINS them.

Why it lives here. `def14a_llm.ceo_name_proxy`, `def14a_executive_comp.name`,
`def14a_directors.name` and `sec_8k_votes.nominee_votes_json[].name` all name the same humans,
and reconciling them across those four tables is what the governance features are built on.
AGENTS.md sends shared vocabulary to `src/utils/`: `src/data_aggregate/` may not import
`src/data_extract/`, and a private second copy of the key would be strictly worse than the
cross-import it dodges -- the two would drift, and the write side is what the read side joins on.

⚠ THE OUTPUT IS BAKED INTO STORED DATA. `sec_8k_votes`' twenty `*_ceo` / `*_exec_officer` /
`*_non_employee` vote sums were bucketed by keying each nominee with this function against the
prior proxy's 402(k) roster. Redefining the key re-partitions those stored columns without
recomputing them, so the aggregation layer would read sums bucketed under one definition while
applying another. A behaviour change here is a RE-EXTRACTION, not a refactor.

What the normalisation buys, measured 2026-09-07 over all 12,343 `def14a_llm` rows / 488 tickers:
  * 1,933 raw CEO name strings collapse to 1,706 identities (-11.7%), on 159 of the 488 tickers;
  * 356 of 1,625 apparent CEO turnovers (21.9%) are one filer respelling one person --
    `Timothy D. Cook` -> `Timothy Cook` -> `Tim Cook` is three spellings of one Apple CEO, and
    `Juan R. Luciano` -> `J. R. LUCIANO` is ADM flipping both the first name and the case;
  * the CEO<->NEO cross-table match rate goes 91.6% -> 98.4% (+769 filings), which is what lets
    the pay-slice family assert the CEO sits inside its own top-5 denominator.

Middle tokens need no rule of their own: the key takes the LAST token and the FIRST token's
initial and ignores everything between, so `Timothy D. Cook` and `Tim Cook` both key `cook|t`.

⚠ KNOWN CEILING, pinned rather than chased: a nickname with a DIFFERENT initial does not
reconcile -- `Bob Smith` keys `smith|b`, `Robert Smith` keys `smith|r`. `tests/utils/test_names.py`
asserts that inequality so it stays a documented limit instead of being rediscovered. A nickname
table is its own project, and the 98.4% above already has this priced in.
"""
from __future__ import annotations

import re
from typing import Any

from src.utils.string import clean_text

# Trailing footnote markers on a person cell: "(3)", "*", "†", or bare digits glued to the surname
# ("Daniel Pinto7", "Emma N. Walmsley11"). Bare digits are only stripped after a letter, so a name
# is never confused with a numbered list item.
_FOOTNOTE_SUFFIX_RE = re.compile(r"(?:\(\d+\)|[*†‡§]|(?<=[a-z])\d{1,2})+\s*$")

# Titles that edgartools glues onto the name when the source cell has the name and the position on
# two visual lines. The leading modifier group matters: without it "Luca Maestri Former Senior Vice
# President" splits at "Senior Vice" and leaves "Luca Maestri Former" as the name (and likewise
# "Bob De Lange Group"), so the modifier is consumed into the TITLE where it belongs.
_GLUED_TITLE_RE = re.compile(
    r"\s*(?:Former\s+|Group\s+|Interim\s+|Acting\s+|Co-)?(?:"
    r"Chairman\b|Chief\s|President\b|Senior\s+Vice\b|Executive\s+Vice\b|Vice\s+Chair\b|"
    r"General\s+Counsel\b|Co-CEO\b|\bCEO\b|\bCFO\b|\bCOO\b"
    r").*$"
)

#: Post-nominals must be matched AFTER the dots are removed, not before. The academic ones are
#: written `Ph.D.` / `M.D.` / `DVM`, and `\bphd\b` does not match `ph.d.` -- so with the dots
#: still in place the suffix survived, `_NON_ALPHA_RE` then split it into `ph d`, and `d` became
#: the SURNAME. Measured: `person_key("Albert Bourla, DVM, Ph.D.")` returned `d|a`, which is not
#: merely a failure to match `A. Bourla` (`bourla|a`) -- it collapses every credentialed
#: director with the same first initial onto ONE key, so a consensus pass would propagate one
#: person's gender onto unrelated people.
_DOT_RE = re.compile(r"\.")
_SUFFIX_RE = re.compile(
    r"\b(?:jr|sr|ii|iii|iv|v|phd|md|dvm|dds|dsc|edd|pharmd|mph|cpa|cfa|esq)\b", re.I)
_NON_ALPHA_RE = re.compile(r"[^a-z ]+")


def clean_person_name(value: Any) -> str | None:
    """Normalise a person cell into a STABLE primary key: collapse whitespace, strip the glued-on
    title and any trailing footnote marker. Casing is left alone (it is source-faithful and
    lower-casing would fight the rest of the repo), but the footnote strip is what actually
    matters -- without it the same director keys as "Emma N. Walmsley11" one year and
    "Emma N. Walmsley10" the next, silently duplicating the row instead of updating it."""
    cleaned = clean_text(value)
    if cleaned is None:
        return None
    cleaned = _GLUED_TITLE_RE.sub("", cleaned).strip()
    cleaned = _FOOTNOTE_SUFFIX_RE.sub("", cleaned).strip()
    cleaned = cleaned.rstrip(",;:-").strip()
    return cleaned or None


def person_key(name: str | None) -> str | None:
    """A person key stable across filings and across companies: `lastname|firstinitial`.

    Keyed on the first INITIAL, not the first name, because a filer's own spelling drifts:
    "Katherine J. Smith" and "Kathy Smith" are one director and must reconcile. Generational
    suffixes and post-nominals are stripped first -- "John Smith Jr." and "John Smith" are the
    same person for this purpose, and treating them as two would split their evidence.

    This is the SAME key the vote role map buckets nominees with, so a nominee that matches
    there matches here.
    """
    cleaned = clean_person_name(name)
    if not cleaned:
        return None
    # dots first, so `Ph.D.` becomes `phd` and the suffix pattern can see it; DELETED rather
    # than replaced with a space, because a space would leave `ph d` and put `d` in surname
    # position. `H.` -> `h` is unaffected either way.
    flat = _NON_ALPHA_RE.sub(" ", _SUFFIX_RE.sub(" ", _DOT_RE.sub("", cleaned.lower())))
    parts = [p for p in flat.split() if p]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return f"{parts[-1]}|{parts[0][0]}"
