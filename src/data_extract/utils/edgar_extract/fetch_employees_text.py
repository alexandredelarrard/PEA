"""10-K family employees text: the document a workforce count is read from, as candidate windows and their union.

Writes `sec_employees_text`, one row per filing: `source_document` ("primary" or an annual-report exhibit type),
`windows` (JSON `[start, end]` candidate windows in selection priority order), `pieces` (JSON `[offset, text]` of
the windows' merged union), `n_chars` (the chosen document's length) and the filing stamp. The employees LLM step
rebuilds its excerpt from the stored windows at any budget. A filing with no usable text becomes an empty-filing
marker; a transient SEC failure raises and the filing is listed again.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from functools import partial

from src.data_extract.utils.common.resume import DONE_PER_KEY
from src.data_extract.utils.common.sec_io import forget_sgml
from src.data_extract.utils.common.workforce import merge_spans
from src.data_extract.utils.edgar_extract.edgar_driver import EdgarFetch, FilingStamp, parse_filing_rows
from src.data_extract.utils.edgar_extract.employee_document import employee_text
from src.data_store.schema import Tables

_OPENING = 8_000  # chars of the filing's opening text, a candidate after the workforce-number windows
_WINDOW = (550, 850)  # chars kept before and after a workforce mention

_CONTEXT_RE = re.compile(
    r"\b(?:employees?|workforce|associates?|team\s+members?|human\s+capital|"
    r"personnel|staff|colleagues?|full[- ]time|part[- ]time)\b",
    re.I,
)


def candidate_windows(text: str, spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Windows around the workforce-number `spans`, then the opening, then windows around other workforce mentions,
    in selection priority order, each clipped to the text."""
    before, after = _WINDOW
    windows = [(max(0, start - before), min(len(text), end + after)) for start, end in spans]
    windows.append((0, min(_OPENING, len(text))))
    windows += [(max(0, match.start() - before), min(len(text), match.end() + after)) for match in _CONTEXT_RE.finditer(text)]
    return windows


def _employee_rows(ticker: str, stamp: FilingStamp) -> list[dict]:
    """One row for the filing's chosen document; none when it has no usable text."""
    # read: the stamp (period of report first), then the document; the submission is dropped after
    stamped = stamp.columns(Tables.sec_employees_text)
    document = employee_text(stamp.filing)
    forget_sgml(stamp.filing)
    if not document.text.strip():
        return []

    # compute: candidate windows in priority order and the text of their merged union
    windows = candidate_windows(document.text, document.spans)
    pieces = [[start, document.text[start:end]] for start, end in merge_spans(windows)]
    return [
        {
            "ticker": ticker,
            **stamped,
            "source_document": document.source_document,
            "windows": json.dumps([list(window) for window in windows]),
            "pieces": json.dumps(pieces),
            "n_chars": len(document.text),
        }
    ]


EMPLOYEES_TEXT_FETCH = EdgarFetch(
    tables=(Tables.sec_employees_text,),
    parse=partial(parse_filing_rows, table=Tables.sec_employees_text, row_fn=_employee_rows),
    done_scope=DONE_PER_KEY,
)
