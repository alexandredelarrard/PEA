"""
cik_cutover.py (src/data_extract/utils/fundamentals/cik_cutover.py)
--------------------------------------------------------------------------
`Company(ticker)` resolves only the CURRENT registrant, so a predecessor's decade of
filings is invisible with no error and no gap signal. APA loses **2011-02 to 2021-05**
(Apache Corp, CIK 6769), GOOGL loses 2011-2015 (Google Inc, CIK 1288776), ETN loses the
2012 Irish domestication. A ticker simply arrives with 22 filings where its peers have 62,
and nothing in the pipeline says why.

**The repair is a DATED CUTOVER, never a union of CIKs.** Apache Corp kept filing its own
10-K/10-Q through 2024-11-07 -- it retains registered public debt -- so 2021-2024 is
double-covered by two *different legal entities*. Concatenating both CIKs would duplicate
~15 filings and, worse, blend a subsidiary's statements with the parent's: a fuller-looking
history that is quietly wrong, which is the dangerous direction. So each entry names one
date, predecessor filings are kept strictly BEFORE it and successor filings on or after it,
and the two sets are disjoint by construction.

**A rename is not a cutover.** CVS Caremark -> CVS Health and Facebook -> Meta keep their
CIK; they need no entry at all, and encoding one would double-walk a single CIK. The schema
makes the distinction explicit (`kind`) and the loader rejects `predecessor_cik ==
successor_cik`, so the mistake cannot be made silently.

Curated JSON rather than an `sp500_tickers` column: that table is rebuilt from Wikipedia,
and a roster refresh would silently overwrite hand-established evidence.

`entity_scope` needs no change on either side of the boundary -- pre-reorganisation the
predecessor IS the consolidated parent, so undimensioned facts are the right scope for both.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.constants.constants import (
    FUNDAMENTALS_CATALOGUE_SUBDIR, FUNDAMENTALS_CIK_CUTOVER_FILENAME,
)
from src.data_extract.utils.fundamentals.kpi_catalogue import resolve_config_dir

#: `kind` values. `reorganisation` = a new legal parent (APA -> APA Corp holding company);
#: `domestication` = the same business re-registered in another jurisdiction (ETN's 2012
#: move to Ireland). Both change the CIK. A `rename` does not, and is therefore NOT a
#: permitted `kind` here -- naming it explicitly is what stops the next person adding one.
CUTOVER_KINDS: frozenset[str] = frozenset({"reorganisation", "domestication"})

#: The `kind` a reader will reach for and must not use. Rejected with its reason attached.
RENAME_KIND = "rename"


@dataclass(frozen=True)
class Cutover:
    """One ticker's registrant boundary."""

    ticker: str
    cutover_date: pd.Timestamp
    predecessor_cik: str
    successor_cik: str
    kind: str
    evidence: str

    def cik_for(self, filing_date) -> str:
        """Which registrant a filing on this date belongs to."""
        return (self.predecessor_cik if pd.Timestamp(filing_date) < self.cutover_date
                else self.successor_cik)


def _normalise_cik(value: Any) -> str:
    """CIKs are 10-digit zero-padded everywhere in this repo (see the 13F loader), and a
    config written as an int or a bare string must join against that without a surprise."""
    return str(value).strip().zfill(10)


def load_cutovers(config_dir: str | None = None) -> dict[str, Cutover]:
    """The cutover register, keyed by ticker, cached per config DIRECTORY rather than per
    spelling of it -- see `resolve_config_dir`."""
    return _cutovers_at(config_dir)


@cache
def _cutovers_at(config_dir: str) -> dict[str, Cutover]:
    """The cutover register, validated, keyed by ticker. `{}` when the file is absent --
    the common case for a repo that has not needed one yet, and not an error.

    Validation is strict and happens at LOAD time, because a typo here silently deletes a
    decade of history rather than raising: a `cutover_date` a year too early drops the
    predecessor's last four filings and admits nothing in their place.

    The one check that CANNOT live here is "the date falls inside the predecessor's own
    filing window" -- that needs EDGAR, and a config loader must not make network calls on
    a nightly path. `tests/data_extract/fundamentals/test_cik_cutover.py` asserts it against live
    submissions instead.
    """
    path = Path(config_dir) / FUNDAMENTALS_CATALOGUE_SUBDIR / FUNDAMENTALS_CIK_CUTOVER_FILENAME
    if not path.exists():
        return {}
    blob = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, Cutover] = {}
    for ticker, entry in blob.items():
        if ticker.startswith("_"):
            continue
        missing = [k for k in ("cutover_date", "predecessor_cik", "successor_cik", "kind",
                               "evidence") if k not in entry]
        if missing:
            raise ValueError(f"cik_cutover[{ticker}]: missing key(s) {missing}")
        kind = str(entry["kind"])
        if kind == RENAME_KIND:
            raise ValueError(
                f"cik_cutover[{ticker}]: kind='{RENAME_KIND}' is not a cutover. A rename "
                "keeps the CIK (CVS Caremark -> CVS Health, Facebook -> Meta), so an entry "
                "here would walk one CIK twice and duplicate every filing. Delete it.")
        if kind not in CUTOVER_KINDS:
            raise ValueError(f"cik_cutover[{ticker}]: kind={kind!r} not in "
                             f"{sorted(CUTOVER_KINDS)}")
        predecessor = _normalise_cik(entry["predecessor_cik"])
        successor = _normalise_cik(entry["successor_cik"])
        if predecessor == successor:
            raise ValueError(
                f"cik_cutover[{ticker}]: predecessor_cik == successor_cik ({predecessor}). "
                "Same CIK means no cutover happened -- this is a rename.")
        date = pd.Timestamp(entry["cutover_date"])
        if pd.isna(date):
            raise ValueError(f"cik_cutover[{ticker}]: cutover_date is unparseable")
        if not str(entry["evidence"]).strip():
            raise ValueError(
                f"cik_cutover[{ticker}]: empty `evidence`. An undocumented cutover is a "
                "guess that deletes history, which is exactly what this register replaces.")
        out[ticker] = Cutover(ticker=ticker, cutover_date=date,
                              predecessor_cik=predecessor, successor_cik=successor,
                              kind=kind, evidence=str(entry["evidence"]))
    return out


def cutover_filings(cutover: Cutover, forms: list[str],
                    since: pd.Timestamp | None,
                    done_accessions: frozenset[str]) -> list:
    """Both registrants' filings, split at the cutover date, oldest first.

    The predecessor contributes filings strictly BEFORE `cutover_date`, the successor
    everything on or after it. The two sets are disjoint by construction, which is the whole
    point: Apache Corp filed 4x a year as a SUBSIDIARY through 2024-11-07, so a union of the
    two CIKs would blend two legal entities' consolidated statements over 2021-2024 and look
    like a fuller history rather than a corrupted one.

    Mirrors `edgar_driver.new_filings` -- same dedup, same `since` filter, same ordering --
    so a ticker with a cutover entry and one without differ only in which filings arrive.
    """
    from edgar import Company

    dated: list[tuple[pd.Timestamp, object]] = []
    for cik, keep_before in ((cutover.predecessor_cik, True),
                             (cutover.successor_cik, False)):
        try:
            filings = Company(int(cik)).get_filings(form=list(forms))
        except Exception:                                   # noqa: BLE001 -- dead CIK
            continue
        for filing in filings:
            if filing.accession_number in done_accessions:
                continue
            filed = pd.Timestamp(filing.filing_date)
            if keep_before is (filed >= cutover.cutover_date):
                continue
            if since is not None and filed < since:
                continue
            dated.append((filed, filing))
    dated.sort(key=lambda pair: pair[0])
    return [filing for _, filing in dated]
