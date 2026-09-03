"""
guard.py  (src/data_extract/utils/structure/votes/guard.py)
------------------------------------------------------------
The fabrication guard: what stops the model inventing a vote table.

It exists because the model DID invent one -- "John Doe" / "Jane Smith", 250,000,000
votes, for a filing whose `item_text` was truncated. Two rules, both measured:

1. Emit nothing unless the text is long enough AND carries a comma-grouped number.
2. Drop any nominee whose name is not printed in the source, or whose numbers are all
   absent from it.
"""
from __future__ import annotations

import re

import pandas as pd

from src.data_extract.utils.schemas.vote_schema import ProposalVote
from src.data_extract.utils.structure.def14a.validate import clean_person_name

#: The four vote columns, at proposal level and per role category. Shared vocabulary: the
#: grounding check below reads them off a nominee, and `flatten` builds its per-role column
#: names from them.
_VOTE_FIELDS = ("votes_for", "votes_against", "votes_abstain", "votes_broker_non_votes")

#: Vote tallies in these filings are SHARE COUNTS and are always printed comma-grouped.
#: An Item 5.07 narrative with no comma-grouped number anywhere therefore has no table to
#: read, and asking the model to read one is how "John Doe / 250,000,000" gets invented.
_GROUPED_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+")

#: Floor on `item_text` length. It catches the emptiest stubs and NOT MUCH MORE, and the
#: reason is measured: a truncation stub is not reliably shorter than a real filing. The
#: shortest GENUINE tally in the 333-filing baseline is 554 chars -- TDG's 2014 and 2019
#: special meetings, each a complete `FOR / AGAINST / ABSTAIN` block on a single stock-option
#: plan -- while HWM 2019's truncated stub is 508. So a floor high enough to reject HWM would
#: destroy two correct filings, and this rule cannot be the one that catches a truncation.
#: What actually catches those is the comma-grouped-number test plus the per-row grounding
#: check: a stub that was cut before the tables carries no share count to read.
_MIN_ITEM_TEXT_CHARS = 400

#: A filing that says its own numbers are provisional. Only 14 filings corpus-wide do, so
#: this is a weak signal by construction -- it resolves 8 of the 17 genuine restatements
#: and is stored as a flag rather than used to choose between filings.
_PRELIMINARY_RE = re.compile(
    r"(?i)\b(preliminary|estimated\s+(?:preliminary\s+)?voting|"
    r"subject\s+to\s+(?:final\s+)?certification|not\s+yet\s+certified)\b")

#: The four role buckets a nominee lands in. `unmatched` is a bucket, not an error: it is
#: how the join's failure rate stays measurable instead of being silently folded into

# --------------------------------------------------------------------------- #
# Rule 1 -- the fabrication guard                                              #
# --------------------------------------------------------------------------- #
def has_vote_numbers(text: str | None) -> bool:
    """True when `item_text` contains at least one comma-grouped number.

    Vote tallies are share counts and are always comma-grouped in these filings, so this
    is the cheapest test for "is there a table in here at all". Verified on the baseline:
    the genuine tally-free 5.07(d) board-response filings (NKE 2011/2017, BA 2011,
    TDG 2011/2014/2017/2019) contain none.
    """
    return bool(text) and bool(_GROUPED_NUMBER_RE.search(text))


def mentions_preliminary(text: str | None) -> bool:
    """True when the filing itself flags its numbers as provisional."""
    return bool(text) and bool(_PRELIMINARY_RE.search(text))


def rejection_reason(text: str | None) -> str | None:
    """Why this `item_text` must not be sent to the LLM, or None to proceed.

    Returning a REASON rather than a bool is what lets the run log distinguish the two
    populations that both produce zero rows: a truncated narrative (a known 1.0% loss we
    accepted) and a genuinely tally-free 5.07(d) filing (8.8%, a correct outcome). A
    single "0 rows" counter would have hidden both inside each other.
    """
    if text is None or not str(text).strip():
        return "empty item_text"
    if len(text) < _MIN_ITEM_TEXT_CHARS:
        return f"item_text truncated ({len(text)} chars)"
    if not has_vote_numbers(text):
        return "no comma-grouped number -- no vote table"
    return None


def _norm_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


#: Name tokens too short or too generic to be evidence on their own. A middle initial appears
#: somewhere in any proxy, so counting it as a hit would make the token fallback vacuous.
_WEAK_NAME_TOKENS = frozenset({"jr", "sr", "ii", "iii", "iv", "de", "la", "van", "der", "von"})


def _name_in_source(name: str | None, text: str) -> bool:
    """True when `name` appears in the source, whitespace- and case-insensitively.

    A verbatim name check is a valid hallucination test on filing text: 99.74% of the
    `directors[]` names extracted from DEF 14A bodies appear verbatim in their source.
    A nominee name is printed in the election table it was read from, so a name that is
    absent was not read -- it was written.

    But a CONTIGUOUS check alone over-rejects, and measurably: edgartools renders a narrow name
    column by wrapping it, and the vote numbers land BETWEEN the two halves of the name. PTC's
    2010 filing prints

        Paul                       100,753,338     1,735,851     7,486,441
        A. Lacy

    so "Paul A. Lacy" is not a substring of the rendered text at any whitespace normalisation,
    and three CORRECTLY read nominees were being discarded. Hence the token fallback: every
    substantial token of the name must appear in the source, which survives the wrap while
    still refusing a name the document never mentions. It is weaker than contiguity by exactly
    one thing -- a fabricated name assembled from words present elsewhere in the document --
    and the fabrication actually measured ("John Doe" / "Jane Smith" on a 508-char stub) fails
    it, because neither surname is anywhere in that stub.
    """
    cleaned = clean_person_name(name)
    if not cleaned:
        return False
    haystack = _norm_space(text)
    if _norm_space(cleaned) in haystack:
        return True
    tokens = [t for t in re.findall(r"[a-z]+", cleaned.lower())
              if len(t) > 1 and t not in _WEAK_NAME_TOKENS]
    return bool(tokens) and all(t in haystack for t in tokens)


def _number_in_source(value: float | None, text: str) -> bool:
    """True when `value` is printed in the source, comma-grouped or bare."""
    if value is None or pd.isna(value):
        return False
    n = int(round(float(value)))
    return f"{n:,}" in text or (n < 1000 and str(n) in text)


def _is_grounded(votes: dict, text: str) -> bool:
    """True when at least ONE of a row's vote counts is printed in the source.

    Deliberately ANY and not ALL. All-must-match would reject a whole row over a single
    cell the model rounded -- the measured LLM error set includes fractional votes
    truncated 1000x -- while the failure this guard exists to catch is wholesale
    invention, where a fabricated row's numbers appear nowhere. A row with no numbers at
    all is not grounded either: that is the shape of an invented nominee.
    """
    return any(_number_in_source(votes.get(f), text) for f in _VOTE_FIELDS)


# --------------------------------------------------------------------------- #
# Flatten                                                                      #
# --------------------------------------------------------------------------- #
def _grounded_nominees(proposal: ProposalVote, text: str) -> tuple[list[dict], int]:
    """`(kept nominees, rejected count)` after the fabrication guard.

    Both halves of the guard apply per nominee: the name must be printed in the source,
    and at least one of the four counts must be too.
    """
    kept, rejected = [], 0
    for n in proposal.nominees:
        name = clean_person_name(n.name)
        votes = {f: getattr(n, f) for f in _VOTE_FIELDS}
        if not name or not _name_in_source(name, text) or not _is_grounded(votes, text):
            rejected += 1
            continue
        kept.append({"name": name, **votes})
    return kept, rejected

