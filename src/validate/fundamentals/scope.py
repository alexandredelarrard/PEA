"""
scope.py  (src/validate/fundamentals/)
--------------------------------------------------------------------------------------------
`RunScope`: WHAT a run looked at, hashed, so that two runs can be differenced honestly.

## The one problem this type exists to solve

The loop's whole proposition is that a row-count drop in `fundamentals_check` PROVES a fix
worked. That proposition is false unless the two runs looked at the same thing. Run 2 covered
54 tickers; a follow-up scoped to the single ticker someone was fixing would report ~11,800
fewer findings and every one of them would read as a triumph.

So the scope is recorded, hashed, and compared. `scope_hash` is the equality test; a run whose
hash differs is INCOMPARABLE and the report says so rather than rendering a delta that means
nothing.

## Two hashes, and why not one

  `scope_hash`  (tickers, fields, tiers)             -- COMPARABILITY. Deliberately no date.
  `run_id`      (run_date, tickers, fields, tiers)   -- IDENTITY of one run.

One hash cannot do both jobs. Including the date makes every run incomparable with every other
run; excluding it makes today's run indistinguishable from last week's in a table keyed on it.

`roster` is carried but NOT hashed. It is a label for a ticker list, and two runs that cover
the same tickers are comparable whether or not someone renamed the roster in between --
hashing the name would make a `configs/` edit look like a scope change.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True, slots=True)
class RunScope:
    """The tickers, fields and tiers one run covered. Immutable; hashed on demand."""

    #: The tickers ACTUALLY LOADED, not the ones requested. A roster naming 54 tickers of
    #: which 3 are absent from `fundamentals_history` covers 51, and a hash of the request
    #: would declare two runs comparable that examined different data.
    tickers: tuple[str, ...] = ()
    #: `--field` narrowing, empty for a full run. Empty and "every field" are the same thing
    #: here, which is why the empty tuple is the natural default rather than a sentinel.
    fields: tuple[str, ...] = ()
    tiers: tuple[int, ...] = ()
    #: The roster NAME, for a human reading the report. Not hashed -- see the module docstring.
    roster: str = ""

    @classmethod
    def build(cls, *, tickers: Iterable[str] | None, fields: Iterable[str] | None,
              tiers: Iterable[int] | None, roster: str = "") -> "RunScope":
        """Normalised and sorted, so two runs given the same scope in a different ORDER hash
        alike. `--tier 3,1` and `--tier 1,3` are one scope."""
        return cls(tickers=tuple(sorted({str(t).upper() for t in (tickers or ())})),
                   fields=tuple(sorted({str(f) for f in (fields or ())})),
                   tiers=tuple(sorted({int(t) for t in (tiers or ())})),
                   roster=str(roster or ""))

    # ------------------------------------------------------------------------ hashes ---
    @property
    def scope_hash(self) -> str:
        """The COMPARABILITY key: 12 hex of (tickers, fields, tiers). No date, no roster."""
        return _digest(self._payload())

    def run_id(self, run_date) -> str:
        """This run's identity: 12 hex of (run_date, scope).

        Same day, same scope -> same id, deliberately. That is what makes a re-run after a
        fix REPLACE its own rows instead of appending a second, half-stale copy of them.
        """
        return _digest({"run_date": _date(run_date), **self._payload()})

    def _payload(self) -> dict[str, Any]:
        return {"tickers": list(self.tickers), "fields": list(self.fields),
                "tiers": list(self.tiers)}

    # ------------------------------------------------------------------------ columns ---
    def as_columns(self, run_date) -> dict[str, Any]:
        """The `fundamentals_check_run` scope columns, repeated on every check row.

        The ticker LIST is written out in full alongside its count. The count alone cannot
        answer "which 54?", and that is the first question anyone differencing two runs asks.
        """
        return {
            "run_id": self.run_id(run_date),
            "scope_hash": self.scope_hash,
            "scope_roster": self.roster or None,
            "scope_tickers": len(self.tickers),
            "scope_ticker_list": json.dumps(list(self.tickers)),
            "scope_fields": json.dumps(list(self.fields)),
            "scope_tiers": json.dumps(list(self.tiers)),
        }

    def describe(self) -> str:
        """One line, for a log and for the report header."""
        return (f"{len(self.tickers)} ticker(s)"
                f"{f' [{self.roster}]' if self.roster else ''}, "
                f"tiers={','.join(map(str, self.tiers)) or 'all'}, "
                f"fields={','.join(self.fields) or 'all'}")


def _digest(payload: dict[str, Any]) -> str:
    """12 hex of the canonical JSON. `sort_keys` so a dict-order change is not a scope change."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _date(value) -> str:
    return str(pd.Timestamp(value).date())


__all__ = ["RunScope"]
