"""Shared earnings-call tokenization, cleaning and quality checks."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
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


def clean_earnings_call_paragraphs(paragraphs: Iterable[Mapping[str, object]]) -> tuple[list[dict[str, object]], Counter[str]]:
    """Remove metadata/rosters and balanced parenthetical text; retain Operator split cues."""
    # Provider metadata and balanced parentheticals are not financial speech.
    metadata = re.compile(
        r"(?:executives|analysts|company representatives|corporate participants|participants|transcript\s+(?:sponsor|operator)|ai\s+insights)", re.I
    )
    parenthetical = re.compile(r"\([^()]*\)")
    cleaned: list[dict[str, object]] = []
    removed: Counter[str] = Counter()
    # Retain original paragraph keys while tracking each cleaning exclusion.
    for paragraph in paragraphs:
        speaker = " ".join(str(paragraph.get("speaker") or "").split())
        if metadata.fullmatch(speaker):
            removed["metadata paragraphs"] += 1
            continue
        raw = str(paragraph.get("content") or "")
        text = raw
        while True:
            text, matches = parenthetical.subn(" ", text)
            if not matches:
                break
        if text != raw:
            removed["parenthetical paragraphs"] += 1
            removed["parenthetical characters"] += len(raw) - len(text)
            text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = " ".join(text.split())
        if not text:
            removed["empty paragraphs"] += 1
            continue
        cleaned.append({**paragraph, "content": text})
    return cleaned, removed


def clean_earnings_call_text(text: object) -> str:
    """Remove leading/trailing courtesy-only sentences and normalize whitespace. Sentences come from
    the shared lossless splitter ("3.5%" and "U.S." survive byte-identical) and courtesy is the
    shared `is_pleasantry` test, so on a `split_call` text, already cleaned sentence by sentence,
    this is a no-op."""
    # Normalize nullable source values before splitting financial sentences.
    if text is None:
        raw = ""
    else:
        try:
            missing = bool(text != text)
        except (TypeError, ValueError):
            missing = True
        raw = "" if missing else str(text)
    sentences = split_sentences(raw)
    # Strip courtesy-only edges without altering the sentence interiors.
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
    # Clean the two management documents and retain any auxiliary role text.
    minimum_words = 100
    cleaned = {str(tag): clean_earnings_call_text(text) if str(tag) in CALL_SECTIONS else str(text or "") for tag, text in sections.items()}
    # Only prepared and management-answer words establish call eligibility.
    counts = {tag: word_count(cleaned.get(tag, "")) for tag in CALL_SECTIONS}
    total = sum(counts.values())
    reason = "no usable sections" if total == 0 else None
    if reason is None and total < minimum_words:
        reason = f"only {total} cleaned words (<{minimum_words})"
    return EarningsCallQuality(cleaned, counts, total, reason is None, reason)
