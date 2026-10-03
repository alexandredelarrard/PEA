"""The fabrication guard: what stops the model inventing a vote table.

1. Emit nothing unless the text is long enough AND carries a comma-grouped number (`rejection_reason`).
2. Drop any nominee whose name is not printed in the source, or whose vote counts are all absent from it.
"""

from __future__ import annotations

import re

import pandas as pd

from src.data_extract.utils.schemas.vote_schema import ProposalVote
from src.data_extract.utils.structure.def14a.validate import clean_person_name

#: The four vote columns, at proposal level and per role category (also used by `flatten` for per-role names).
_VOTE_FIELDS = ("votes_for", "votes_against", "votes_abstain", "votes_broker_non_votes")

#: Vote tallies are share counts, always printed comma-grouped.
_GROUPED_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+")
#: The column labels of a vote tally table.
_VOTE_TABLE_LABEL_RE = re.compile(r"(?i)\b(?:against|withheld|abstain(?:ed)?|broker\s+non[- ]votes?)\b")

#: Floor on `item_text` length: catches only empty stubs; truncations are caught by the number and grounding tests.
_MIN_ITEM_TEXT_CHARS = 400

#: A filing that says its own numbers are provisional; stored as a flag, never used to choose between filings.
_PRELIMINARY_RE = re.compile(
    r"(?i)\b(preliminary|estimated\s+(?:preliminary\s+)?voting|"
    r"subject\s+to\s+(?:final\s+)?certification|not\s+yet\s+certified)\b"
)


def has_vote_numbers(text: str | None) -> bool:
    """True when `item_text` contains at least one comma-grouped number: the cheapest test for "is there a vote table at all"."""
    return bool(text) and bool(_GROUPED_NUMBER_RE.search(text))


def has_vote_table(text: str) -> bool:
    """True when `text` reads as a vote tally table: at least two comma-grouped share counts and
    a vote column label (against / withheld / abstain / broker non-votes)."""
    return len(_GROUPED_NUMBER_RE.findall(text)) >= 2 and bool(_VOTE_TABLE_LABEL_RE.search(text))


def mentions_preliminary(text: str | None) -> bool:
    """True when the filing itself flags its numbers as provisional."""
    return bool(text) and bool(_PRELIMINARY_RE.search(text))


def rejection_reason(text: str | None) -> str | None:
    """Why this `item_text` must not be sent to the LLM, or None to proceed.

    A reason, not a bool, so the run log separates a truncated narrative from a genuinely tally-free filing.
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


#: Name tokens too generic to be evidence on their own in the token fallback.
_WEAK_NAME_TOKENS = frozenset({"jr", "sr", "ii", "iii", "iv", "de", "la", "van", "der", "von"})


def _name_in_source(name: str | None, text: str) -> bool:
    """True when `name` appears in the source, whitespace- and case-insensitively.

    Falls back to requiring every substantial token of the name in the source, because a wrapped
    name column can put the vote numbers between the two halves of a name.
    """
    cleaned = clean_person_name(name)
    if not cleaned:
        return False
    haystack = _norm_space(text)
    if _norm_space(cleaned) in haystack:
        return True
    tokens = [t for t in re.findall(r"[a-z]+", cleaned.lower()) if len(t) > 1 and t not in _WEAK_NAME_TOKENS]
    return bool(tokens) and all(t in haystack for t in tokens)


def _number_in_source(value: float | None, text: str) -> bool:
    """True when `value` is printed in the source, comma-grouped or bare."""
    if value is None or pd.isna(value):
        return False
    n = int(round(float(value)))
    return f"{n:,}" in text or (n < 1000 and str(n) in text)


def _is_grounded(votes: dict, text: str) -> bool:
    """True when at least ONE of a row's vote counts is printed in the source (any, not all, so one
    mis-rounded cell does not reject a row); a row with no numbers is not grounded."""
    return any(_number_in_source(votes.get(f), text) for f in _VOTE_FIELDS)


def _grounded_nominees(proposal: ProposalVote, text: str) -> tuple[list[dict], int]:
    """`(kept nominees, rejected count)`: a nominee is kept only when its name and at least one vote count are in the source."""
    kept, rejected = [], 0
    for n in proposal.nominees:
        name = clean_person_name(n.name)
        votes = {f: getattr(n, f) for f in _VOTE_FIELDS}
        if not name or not _name_in_source(name, text) or not _is_grounded(votes, text):
            rejected += 1
            continue
        kept.append({"name": name, **votes})
    return kept, rejected
