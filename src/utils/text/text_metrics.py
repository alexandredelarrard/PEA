"""Shared earnings-call tokenization, cleaning and quality checks."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from src.utils.text.earnings_call_split import CALL_SECTIONS, is_pleasantry, split_sentences

_WORD_RE = re.compile(r"[a-z][a-z'-]*[a-z]|[a-z]")


def word_tokens(text: str) -> list[str]:
    """Lowercase alphabetic word tokens (apostrophes/hyphens kept internal). Pure."""
    if not text:
        return []
    return _WORD_RE.findall(str(text).lower())


def word_count(text: str) -> int:
    return len(word_tokens(text))


def clean_earnings_call_text(text: object) -> str:
    """Remove leading/trailing courtesy-only sentences and normalize whitespace. Sentences come from
    the shared lossless splitter ("3.5%" and "U.S." survive byte-identical) and courtesy is the
    shared `is_pleasantry` test, so on a `split_call` text, already cleaned sentence by sentence,
    this is a no-op."""
    if text is None:
        raw = ""
    else:
        try:
            missing = bool(text != text)
        except (TypeError, ValueError):
            missing = True
        raw = "" if missing else str(text)
    sentences = split_sentences(raw)
    while sentences and is_pleasantry(sentences[0]):
        sentences.pop(0)
    while sentences and is_pleasantry(sentences[-1]):
        sentences.pop()
    return " ".join(sentences).strip()


@dataclass(frozen=True)
class EarningsCallQuality:
    cleaned_sections: dict[str, str]
    section_word_counts: dict[str, int]
    combined_word_count: int
    valid: bool
    reason: str | None


def assess_earnings_call_sections(sections: Mapping[str, object]) -> EarningsCallQuality:
    """One shared validity decision for every split call, in the features and the validator."""
    scored_tags = CALL_SECTIONS
    minimum_words = 100
    scored = set(scored_tags)
    cleaned = {str(tag): clean_earnings_call_text(text) if str(tag) in scored else str(text or "") for tag, text in sections.items()}
    counts = {tag: word_count(cleaned.get(tag, "")) for tag in scored_tags}
    total = sum(counts.values())
    reason = "no usable sections" if total == 0 else None
    if reason is None and total < minimum_words:
        reason = f"only {total} cleaned words (<{minimum_words})"
    return EarningsCallQuality(cleaned, counts, total, reason is None, reason)
