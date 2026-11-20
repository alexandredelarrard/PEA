"""Refuse unusable vote narratives and ground nominee names/counts in the supplied source."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.data_extract.utils.structure.votes.vote_schema import ProposalVote
from src.utils.names import clean_person_name

_VOTE_FIELDS = ("votes_for", "votes_against", "votes_abstain", "votes_broker_non_votes")
_GROUPED_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+")
_VOTE_TABLE_LABEL_RE = re.compile(r"(?i)\b(?:against|withheld|abstain(?:ed)?|broker\s+non[- ]votes?)\b")
_MIN_ITEM_TEXT_CHARS = 400
_PRELIMINARY_RE = re.compile(
    r"(?i)\b(preliminary|estimated\s+(?:preliminary\s+)?voting|"
    r"subject\s+to\s+(?:final\s+)?certification|not\s+yet\s+certified)\b"
)
_WEAK_NAME_TOKENS = frozenset({"jr", "sr", "ii", "iii", "iv", "de", "la", "van", "der", "von"})


def has_vote_table(text: str) -> bool:
    """At least two grouped share counts and an opposition/abstention/broker column label."""
    return len(_GROUPED_NUMBER_RE.findall(text)) >= 2 and bool(_VOTE_TABLE_LABEL_RE.search(text))


def mentions_preliminary(text: str | None) -> bool:
    """Whether the source itself flags its numbers as provisional."""
    return bool(text) and bool(_PRELIMINARY_RE.search(text))


def rejection_reason(text: str | None) -> str | None:
    """The unusable narrative reason, or None when the source can be sent to the model."""
    if text is None or not str(text).strip():
        return "empty item_text"
    if len(text) < _MIN_ITEM_TEXT_CHARS:
        return f"item_text truncated ({len(text)} chars)"
    if not _GROUPED_NUMBER_RE.search(text):
        return "no comma-grouped number -- no vote table"
    return None


def _source_counts(text: str) -> set[int]:
    """Complete printed count tokens under the existing whole-share rounding policy."""
    pattern = r"(?<![\d.,])(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{1,3})(?!\d|,\d|\.\d|\s*%)"
    return {int(round(float(token.replace(",", "")))) for token in re.findall(pattern, text)}


def _grounded_nominees(proposal: ProposalVote, text: str, counts: set[int]) -> tuple[list[dict[str, Any]], int]:
    """Keep a line only when its cleaned name and at least one rounded count occur in the source."""
    # Normalize source whitespace once, retaining the existing wrapped-name token fallback.
    haystack = re.sub(r"\s+", " ", text).strip().lower()
    kept: list[dict[str, Any]] = []
    rejected = 0

    # Preserve any-one-count grounding; validation independently checks every available cell.
    for nominee in proposal.nominees:
        name = clean_person_name(nominee.name)
        tokens = [token for token in re.findall(r"[a-z]+", (name or "").lower()) if len(token) > 1 and token not in _WEAK_NAME_TOKENS]
        present = bool(name) and (
            re.sub(r"\s+", " ", name or "").strip().lower() in haystack or bool(tokens) and all(token in haystack for token in tokens)
        )
        votes = {field: getattr(nominee, field) for field in _VOTE_FIELDS}
        grounded = any(int(round(float(value))) in counts for value in votes.values() if value is not None and pd.notna(value))
        if not present or not grounded:
            rejected += 1
            continue
        kept.append({"name": name, **votes})
    return kept, rejected
