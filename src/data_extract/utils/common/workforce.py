"""Workforce-number spans and span merging, shared by the employees text walk and the employees LLM step."""

from __future__ import annotations

import re
from collections.abc import Iterable

_NUMBER = r"(?P<n>(?<![\d,.])(?!(?:19|20)\d\d\b)(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?\s*(?:thousand|million)\b|\d{2,}))"
_NOUN = r"(?:employees|people|persons|associates|team\s+members|workforce|headcount|full[\s-]*time|part[\s-]*time)"
# A workforce number: a number a few words from a workforce noun or `employ(s/ed)`, or after a headcount label.
_WORKFORCE_NUMBER_RES = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        rf"{_NUMBER}(?:[\s-]+[^\s\d-]+){{0,4}}?[\s-]+(?:{_NOUN}|employ(?:s|ed)?)\b",
        rf"\b(?:{_NOUN}|employ(?:s|ed)?)(?:[\s:,-]+[^\s\d:,-]+){{0,4}}?[\s:,-]+{_NUMBER}",
        rf"\b(?:headcount|number\s+of\s+(?:[a-z]+\s+){{0,2}}?employees)\b[^.\d]{{0,60}}{_NUMBER}",
    )
)
MONTH_RE = re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*$", re.I)


def workforce_number_spans(text: str) -> list[tuple[int, int]]:
    """Matches of a workforce number in document order; a day after a month name is not one."""
    spans = {
        (match.start(), match.end())
        for pattern in _WORKFORCE_NUMBER_RES
        for match in pattern.finditer(text)
        if not MONTH_RE.search(text[max(0, match.start("n") - 12) : match.start("n")])
    }
    return sorted(spans)


def merge_spans(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """`spans` sorted, a span starting at or before the previous end joined to it."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
