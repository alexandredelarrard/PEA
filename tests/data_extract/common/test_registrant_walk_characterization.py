"""Characterization of the registrant rules on local EDGAR index rows.

Pins the listing CIKs, the UNION / SPLIT policies, accession dedup and the margin owner rule of
`resolve_registrant_entries`. Offline.
"""

from __future__ import annotations

import pandas as pd

from src.data_extract.utils.common.registrant import listing_ciks, resolve_registrant_entries
from tests.data_extract.common.scope_fixtures import SENTINEL, dated_identity

#: XOM's register windows plus one event-only CIK (99).
_XOM = dated_identity(
    [
        ("XOM", "0000034088", "cik_window", SENTINEL, "2026-07-01"),
        ("XOM", "0002115436", "cik_window", "2026-07-01", None),
        ("XOM", "0000000099", "cik_event", SENTINEL, None),
    ],
    {"XOM": "0002115436"},
)


def _rows(*rows: tuple[str, str, str, str]) -> pd.DataFrame:
    """Index rows `(cik, form, filed, accession)`."""
    return pd.DataFrame(
        [
            {"cik": cik, "company": "Fixture", "form": form, "filed": pd.Timestamp(filed), "accession": accession}
            for cik, form, filed, accession in rows
        ]
    )


def test_listing_ciks_is_the_scope_superset_roster_first():
    assert listing_ciks(_XOM.filing_scope("XOM")) == ("0002115436", "0000000099", "0000034088")
    print("\n=== SANITY: index listing CIKs ===")
    print("  roster CIK first, then every event CIK and window CIK of the scope, once each")


def test_index_union_lists_every_event_cik_and_a_co_indexed_accession_goes_to_its_date_owner():
    df = _rows(
        ("0000034088", "4", "2026-05-01", "pred"),
        ("0002115436", "4", "2026-07-07", "suc"),
        ("0000000099", "4", "2025-01-01", "event"),
        ("0000034088", "4", "2026-08-03", "shared"),
        ("0002115436", "4", "2026-08-03", "shared"),
        ("0000555555", "4", "2026-08-04", "foreign"),
        ("0000034088", "10-K", "2026-02-01", "other-form"),
    )

    out = resolve_registrant_entries(_XOM.filing_scope("XOM"), df, ["4"])

    assert out["accession"].tolist() == ["event", "pred", "suc", "shared"]
    assert out.loc[out["accession"] == "shared", "cik"].item() == "0002115436"  # the window that owns 2026-08-03
    print("\n=== SANITY: index UNION (insider forms) ===")
    print("  every event CIK listed, oldest first; a co-indexed accession once (its date owner); a foreign CIK and other forms dropped")


def test_index_split_keeps_each_window_inside_its_widened_dates():
    df = _rows(
        ("0000034088", "10-Q", "2026-05-01", "pred-in"),
        ("0000034088", "10-Q", "2026-07-15", "margin"),
        ("0000034088", "10-Q", "2026-09-01", "late"),
        ("0002115436", "10-Q", "2026-04-01", "early"),
        ("0002115436", "10-Q", "2026-08-03", "suc-in"),
        ("0000000099", "10-Q", "2026-03-01", "event-only"),
    )

    out = resolve_registrant_entries(_XOM.filing_scope("XOM"), df, ["10-Q"])

    assert out["accession"].tolist() == ["pred-in", "margin", "suc-in"]
    print("\n=== SANITY: index SPLIT ===")
    print("  each window lists its widened dates only (the predecessor keeps its 31-day margin 10-Q); the event-only CIK never lists")
