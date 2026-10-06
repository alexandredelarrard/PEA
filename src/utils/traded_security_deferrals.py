"""Tickers whose history waits for the traded-security realignment, declared in `configs/sec/security_master_manual.json`.

Each `deferred_to_traded_security` entry names a ticker, what it defers and the decision it cites:
`predecessor_series` keeps the vendor's own rows inside the register predecessor window (no replacement, so no
exchange-ratio conversion); `event_filing_windows` lists and keeps 8-K / 13D / 13G over every CIK of the entity,
undated. Read by the identity resolver, the propagation purge, the merged history and `validate identity` alike.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.utils.string import normalise_ticker

SECTION = "deferred_to_traded_security"
PREDECESSOR_SERIES = "predecessor_series"
EVENT_FILING_WINDOWS = "event_filing_windows"
SCOPES = frozenset({PREDECESSOR_SERIES, EVENT_FILING_WINDOWS})
CONFIG_FILE = Path("sec") / "security_master_manual.json"


class DeferralConfigError(ValueError):
    """A `deferred_to_traded_security` entry without a source, with an unknown or empty scope, or declared twice."""


@dataclass(frozen=True)
class Deferral:
    """One ticker's deferred rules and the decision it cites."""

    ticker: str
    defers: frozenset[str]
    source: str


def load_deferrals(config_dir: str | Path) -> dict[str, Deferral]:
    """`{ticker: Deferral}` from the config; `{}` when the file or the section is absent; a malformed entry raises."""
    path = Path(config_dir) / CONFIG_FILE
    if not path.exists():
        return {}
    out: dict[str, Deferral] = {}
    for row in json.loads(path.read_text(encoding="utf-8")).get(SECTION) or []:
        ticker, source, defers = (
            normalise_ticker(str(row.get("ticker") or "")),
            str(row.get("source") or "").strip(),
            frozenset(row.get("defers") or ()),
        )
        if not source:
            raise DeferralConfigError(f"{CONFIG_FILE}: {SECTION} entry {row} has no source")
        if not defers:
            raise DeferralConfigError(f"{CONFIG_FILE}: {SECTION} entry for {ticker} defers nothing")
        if defers - SCOPES:
            raise DeferralConfigError(f"{CONFIG_FILE}: {SECTION} entry for {ticker} names unknown scope(s) {sorted(defers - SCOPES)}")
        if ticker in out:
            raise DeferralConfigError(f"{CONFIG_FILE}: {SECTION} declares {ticker} twice")
        out[ticker] = Deferral(ticker, defers, source)
    return out


def deferred_tickers(config_dir: str | Path, scope: str) -> frozenset[str]:
    """The tickers deferring `scope` (`PREDECESSOR_SERIES` or `EVENT_FILING_WINDOWS`)."""
    return frozenset(ticker for ticker, deferral in load_deferrals(config_dir).items() if scope in deferral.defers)
