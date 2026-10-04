"""Dated `entity_lineage` fixtures for listing tests: a real `Identity` built from hand-written CIK rows."""

from __future__ import annotations

import types
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common.identity import Identity, build_identity

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
                "entity_id": entity_of[ticker],
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
    return build_identity(lineage=lineage, tenure=tenure, roster=pd.DataFrame([{"ticker": t, "cik": c} for t, c in roster.items()]))


def filing(accession: str, filing_date: str, cik: int | None = None, period: str | None = None) -> types.SimpleNamespace:
    """A stub listed filing; `cik` is the filer EDGAR reports (None = not exposed)."""
    out = types.SimpleNamespace(accession_number=accession, filing_date=filing_date, form="10-K", period_of_report=period)
    if cik is not None:
        out.cik = cik
    return out


def patch_company(monkeypatch: pytest.MonkeyPatch, by_key: dict[Any, list], dead: frozenset = frozenset()) -> list[Any]:
    """`edgar.Company(x)` -> the listing for `x`; keys in `dead` raise. Returns every key `Company` was built with."""
    built: list[Any] = []

    def company(key: Any) -> types.SimpleNamespace:
        built.append(key)
        if key in dead:
            raise ValueError(f"no such company {key}")
        return types.SimpleNamespace(get_filings=lambda form: by_key.get(key, []))

    monkeypatch.setattr("edgar.Company", company)
    return built
