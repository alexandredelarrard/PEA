"""Characterization of the registrant filing walks: the `Company` listing and the local-index rows.

Pins CIK walk order, accession dedup, log text and the margin owner rule of `resolve_registrant_filings`,
and that `resolve_registrant_entries` applies the same rules to local EDGAR index rows. Offline: stub `Company`.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.data_extract.utils.common.registrant import (
    listing_ciks,
    resolve_registrant_entries,
    resolve_registrant_filings,
)
from tests.data_extract.common.scope_fixtures import SENTINEL, dated_identity, filing, patch_company

LOGGER = "src.data_extract.utils.common.registrant"

#: XOM's register windows plus one event-only CIK (99).
_XOM = dated_identity(
    [
        ("XOM", "0000034088", "cik_window", SENTINEL, "2026-07-01"),
        ("XOM", "0002115436", "cik_window", "2026-07-01", None),
        ("XOM", "0000000099", "cik_event", SENTINEL, None),
    ],
    {"XOM": "0002115436"},
)


def _messages(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == LOGGER and record.levelno == level]


# --------------------------------------------------------------------------- #
# (a) UNION over event CIKs (insider forms; 8-Ks split since P35)              #
# --------------------------------------------------------------------------- #
def test_union_walks_every_event_cik_in_order_first_writer_wins_and_logs_contributions(monkeypatch, caplog):
    built = patch_company(
        monkeypatch,
        {
            34088: [filing("b", "2026-02-01"), filing("c", "2010-01-01"), filing("done", "2026-03-01")],
            2115436: [filing("a", "2026-08-01"), filing("b", "2026-02-01"), filing("old", "1999-01-01")],
            99: [filing("d", "2026-09-01"), filing("e", "2005-05-05")],
        },
    )
    stats: dict[str, int] = {}
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = resolve_registrant_filings(
            _XOM.filing_scope("XOM"), ["4", "4/A"], since=pd.Timestamp("2000-01-01"), done_accessions=frozenset({"done"}), stats=stats
        )

    assert [f.accession_number for f in out] == ["e", "c", "b", "a", "d"]
    assert built == [99, 34088, 2115436]
    assert stats == {"skipped_existing": 1, "foreign_skipped": 0}
    assert _messages(caplog, logging.INFO) == ["XOM: 4,4/A listed by union across 2 from 0000000099, 2 from 0000034088, 1 from 0002115436"]
    print("\n=== SANITY: UNION over event CIKs (insider forms) ===")
    print(f"  order {[f.accession_number for f in out]}; CIKs walked {built}; first writer wins; 'done' skipped once")


def test_a_single_cik_walk_applies_since_and_done_before_sort_and_is_silent(monkeypatch, caplog):
    identity = dated_identity([("AAPL", "0000320193", "cik_window", SENTINEL, None)], {"AAPL": "0000320193"})
    built = patch_company(
        monkeypatch, {320193: [filing("c", "2020-03-01"), filing("x", "2010-01-01"), filing("d", "2019-01-01"), filing("a", "2018-01-01")]}
    )
    stats: dict[str, int] = {}
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = resolve_registrant_filings(
            identity.filing_scope("AAPL"), ["8-K"], since=pd.Timestamp("2015-01-01"), done_accessions=frozenset({"d", "x"}), stats=stats
        )
    assert [f.accession_number for f in out] == ["a", "c"]
    assert built == [320193]
    assert stats == {"skipped_existing": 2, "foreign_skipped": 0}
    assert _messages(caplog, logging.INFO) == []
    print("\n=== SANITY: single-CIK walk -> since + done, sorted, silent ===")


# --------------------------------------------------------------------------- #
# (b) SPLIT over CIK windows                                                   #
# --------------------------------------------------------------------------- #
def test_split_walks_windows_only_and_a_dead_window_cik_warns(monkeypatch, caplog):
    built = patch_company(
        monkeypatch,
        {
            34088: [filing("pre", "2026-03-01"), filing("margin", "2026-07-15"), filing("late", "2026-09-01")],
            99: [filing("event-only", "2026-03-01")],
        },
        dead=frozenset({2115436}),
    )
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = resolve_registrant_filings(_XOM.filing_scope("XOM"), ["10-K"], since=None, done_accessions=frozenset())
    assert [f.accession_number for f in out] == ["pre", "margin"]
    assert built == [34088, 2115436]
    assert _messages(caplog, logging.WARNING) == ["XOM: CIK 0002115436 could not be resolved"]
    print("\n=== SANITY: SPLIT walks the windows only ===")
    print("  event-only CIK 99 never listed; predecessor keeps its 31-day margin filing; dead successor CIK warns")


def test_split_accession_admitted_by_two_windows_goes_to_its_date_owner(monkeypatch):
    shared = filing("joint", "2026-07-10")
    patch_company(monkeypatch, {34088: [shared], 2115436: [shared, filing("suc", "2026-08-01")]})
    out = resolve_registrant_filings(_XOM.filing_scope("XOM"), ["10-Q"], since=None, done_accessions=frozenset())
    assert [f.accession_number for f in out] == ["joint", "suc"]
    print("\n=== SANITY: an accession in both widened windows is listed once ===")


# --------------------------------------------------------------------------- #
# (c) the same rules on local EDGAR index rows                                 #
# --------------------------------------------------------------------------- #
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
