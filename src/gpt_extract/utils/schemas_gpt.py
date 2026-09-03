"""
schemas_gpt.py  (src/gpt_extract/utils/schemas_gpt.py)
------------------------------------------------------
What goes into the queue and what comes back out.

The caller passes the schema CLASS it wants filled -- there is no string registry to keep
in sync with the schemas, and therefore no way for a caller to name a schema that does not
exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from pydantic import BaseModel

from src.data_store.schema import Table


@dataclass(frozen=True, slots=True)
class LlmTask:
    """One payload to read, and everything needed to place its answer."""

    #: Submission order. Results are returned in this order regardless of completion
    #: order, because a caller zips them against its own filing list.
    seq: int
    payload: str
    schema: type[BaseModel]
    #: Where a 1:1 row goes. None when the caller supplies a `flatten` that fans one
    #: answer out to several tables.
    table: Table | None = None
    #: Caller context carried through untouched -- ticker, accession, the filing row.
    meta: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LlmResult:
    """One answer, or the reason there isn't one.

    `parsed is None` with `error` set is a NORMAL outcome, not an exception: one filing
    failing must not abort the other filings of its ticker.
    """

    seq: int
    task: LlmTask
    parsed: BaseModel | None
    error: str | None = None
    usage: Mapping[str, int] | None = None

    @property
    def ok(self) -> bool:
        return self.parsed is not None
