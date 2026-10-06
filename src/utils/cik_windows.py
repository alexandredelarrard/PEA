"""The seam rule for an entity's CIK windows, shared by the identity resolver and the identity validator.

A window is one CIK's declared `[valid_from, valid_to)` tenure over its entity; a bound within
`SEAM_MARGIN_DAYS` of another window's opposite bound is widened by that margin into the listed
`[listed_from, listed_to)`. A None bound is open.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

#: Days a consolidating window is widened on each side of a seam between two windows of one entity.
SEAM_MARGIN_DAYS = 31

#: A declared window: (cik, valid_from, valid_to).
Declared = tuple[str, pd.Timestamp | None, pd.Timestamp | None]
#: A widened window: (cik, valid_from, valid_to, listed_from, listed_to).
Widened = tuple[str, pd.Timestamp | None, pd.Timestamp | None, pd.Timestamp | None, pd.Timestamp | None]


def widen_seams(windows: Sequence[Declared]) -> list[Widened]:
    """Windows oldest first, each bound within `SEAM_MARGIN_DAYS` of another window's opposite bound widened by that margin."""
    margin = pd.Timedelta(days=SEAM_MARGIN_DAYS)
    ordered = sorted(windows, key=lambda window: (window[1] or pd.Timestamp.min, window[0]))
    out: list[Widened] = []
    for i, (cik, start, end) in enumerate(ordered):
        others = ordered[:i] + ordered[i + 1 :]
        seam_before = start is not None and any(other_end is not None and abs(other_end - start) <= margin for _, _, other_end in others)
        seam_after = end is not None and any(other_start is not None and abs(other_start - end) <= margin for _, other_start, _ in others)
        listed_from = start - margin if start is not None and seam_before else start
        listed_to = end + margin if end is not None and seam_after else end
        out.append((cik, start, end, listed_from, listed_to))
    return out
