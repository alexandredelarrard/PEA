"""Item-heading regexes and the span scan shared by the filing-text carvers (10-K/10-Q, Schedule 13D).

`carve_spans` returns one candidate body per start heading; each caller keeps its own pick rule.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

#: Separator tolerated between "Item", its number and its caption: "Item 4.", "Item 4:", "Item  4 -".
ITEM_SEP = r"[\.\:\)\s–—-]{0,8}"

_CROSS_REF_WINDOW = 25  # chars before a marker searched for a cross-reference cue


@dataclass(frozen=True)
class CrossRefCues:
    """Words just before an item marker that make it a pointer ("see Item 7"), not a heading:
    `start` for start markers, `end` for end markers."""

    start: re.Pattern
    end: re.Pattern


def item_heading(number: str | int, caption: str, *, line_anchored: bool = False) -> re.Pattern:
    """Case-insensitive "Item <number> <caption>" heading. `line_anchored` requires the heading to
    start a line, makes the caption optional (a bare "Item N." line) and consumes it to end of line."""
    if line_anchored:
        return re.compile(rf"^[ \t]*item{ITEM_SEP}{number}\b[\.\:\)]?[ \t]*(?:{caption}[^\n]*|$)", re.I | re.M)
    return re.compile(rf"item{ITEM_SEP}{number}\b{ITEM_SEP}{caption}", re.I)


def _is_cross_ref(text: str, pos: int, cue: re.Pattern) -> bool:
    return bool(cue.search(text[max(0, pos - _CROSS_REF_WINDOW) : pos]))


def _first_marker(text: str, pos: int, pattern: re.Pattern, cue: re.Pattern | None) -> int | None:
    """Start of the first `pattern` match at/after `pos`; with a `cue`, the match must lie strictly
    after `pos` and not be a cross-reference."""
    if cue is None:
        match = pattern.search(text, pos)
        return match.start() if match else None
    for match in pattern.finditer(text, pos):
        if match.start() > pos and not _is_cross_ref(text, match.start(), cue):
            return match.start()
    return None


def _earliest_marker(text: str, pos: int, patterns: Sequence[re.Pattern], cue: re.Pattern | None) -> int | None:
    found = [x for pattern in patterns if (x := _first_marker(text, pos, pattern, cue)) is not None]
    return min(found) if found else None


def carve_spans(
    text: str,
    start_re: re.Pattern,
    end_res: Sequence[re.Pattern],
    *,
    stop_re: re.Pattern | None = None,
    fallback_end_res: Sequence[re.Pattern] = (),
    cross_refs: CrossRefCues | None = None,
) -> list[tuple[int, int]]:
    """`(body_start, body_end)` for every `start_re` heading in `text`, in document order.

    A body starts where its heading ends and runs to the earliest `end_res` marker after it, else
    the earliest `fallback_end_res` marker, else end of text; `stop_re` caps it. With `cross_refs`,
    start and end markers preceded by a cross-reference cue are skipped.
    """
    spans: list[tuple[int, int]] = []
    end_cue = cross_refs.end if cross_refs is not None else None
    for match in start_re.finditer(text):
        if cross_refs is not None and _is_cross_ref(text, match.start(), cross_refs.start):
            continue
        start = match.end()
        end = _earliest_marker(text, start, end_res, end_cue)
        if end is None and fallback_end_res:
            end = _earliest_marker(text, start, fallback_end_res, end_cue)
        end = len(text) if end is None else end
        stop = stop_re.search(text, start) if stop_re is not None else None
        spans.append((start, min(end, stop.start()) if stop else end))
    return spans
