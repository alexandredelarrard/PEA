"""Dated `universe_cik_symbol` fixtures for listing tests: a real `Identity` built from hand-written CIK rows."""

from __future__ import annotations

import pandas as pd

from src.data_extract.utils.common.identity import Identity, build_identity
from tests.support.universe import as_roster

SENTINEL = "1900-01-01"
CHANGED_AT = pd.Timestamp("2026-01-01 09:00:00")

#: (ticker, cik, role, valid_from, valid_to); `role` is `cik_window` or `cik_event`.
CikRow = tuple[str, str, str, str, str | None]


def dated_identity(
    rows: list[CikRow],
    roster: dict[str, str],
    *,
    symbols: tuple[tuple[str, str, str], ...] = (),
    changed_at: dict[str, pd.Timestamp] | None = None,
) -> Identity:
    """A real `Identity` over dated CIK rows (entity = `E` + the ticker's first CIK) and form345 tenure `(symbol, cik, valid_from)`."""
    entity_of = {}
    for ticker, cik, *_ in rows:
        entity_of.setdefault(ticker, f"E{cik}")
    stamps = changed_at or {}
    lineage = pd.DataFrame(
        [
            {
                "company_id": entity_of[ticker],
                "canonical_ticker": ticker,
                "cik": cik,
                "role": role,
                "symbol": "",
                "valid_from": start,
                "valid_to": end,
                "status": "curated",
                "sources": "register",
                "oracle": "register",
                "confidence": None,
                "n_observations": 1,
                "evidence": "fixture",
                "scope_changed_at": stamps.get(ticker, CHANGED_AT),
            }
            for ticker, cik, role, start, end in rows
        ]
    )
    tenure = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "issuer_cik": cik,
                "valid_from": pd.Timestamp(start),
                "valid_to": None,
                "n_filings": 5,
                "source": "form345",
                "evidence": "",
            }
            for symbol, cik, start in (symbols or [(ticker, cik, "2000-01-01") for ticker, cik in roster.items()])
        ]
    )
    return build_identity(lineage=lineage, tenure=tenure, roster_cik=as_roster(pd.DataFrame([{"ticker": t, "cik": c} for t, c in roster.items()])))
