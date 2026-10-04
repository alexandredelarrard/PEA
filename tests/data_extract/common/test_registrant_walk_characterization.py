"""Characterization of the registrant filing walks: the `Company` listing and the local-index rows.

Pins provenance order, accession dedup, log text and the SPLIT duplicate-accession warning, and
that `resolve_registrant_entries` applies the same rules to index rows. Offline: stub `Company`.
"""

from __future__ import annotations

import logging
import types
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common.identity import Identity, build_identity
from src.data_extract.utils.common.registrant import (
    Registrant,
    Segment,
    resolve_registrant_entries,
    resolve_registrant_filings,
)

LOGGER = "src.data_extract.utils.common.registrant"
BOUNDARY = pd.Timestamp("2026-07-01")


def _filing(accession: str, filing_date: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(accession_number=accession, filing_date=filing_date)


def _patch_company(monkeypatch: pytest.MonkeyPatch, by_key: dict[Any, list], dead: frozenset = frozenset()) -> list[Any]:
    """`Company(x)` -> listing for `x`; keys in `dead` raise. Returns the construction log."""
    built: list[Any] = []

    def company(key: Any) -> types.SimpleNamespace:
        built.append(key)
        if key in dead:
            raise ValueError(f"no such company {key}")
        return types.SimpleNamespace(get_filings=lambda form: by_key.get(key, []))

    monkeypatch.setattr("edgar.Company", company)
    return built


def _identity(ticker: str, roster_cik: str, entity_by_cik: dict[str, str], *tenure: tuple[str, str]) -> Identity:
    return build_identity(
        lineage=pd.DataFrame([{"cik": cik, "entity_id": entity, "source": "fixture"} for cik, entity in entity_by_cik.items()]),
        tenure=pd.DataFrame(
            [
                {"symbol": symbol, "issuer_cik": cik, "valid_from": pd.Timestamp("2000-01-01"), "valid_to": None, "n_filings": 1, "source": "form345"}
                for symbol, cik in tenure
            ]
        ),
        roster=pd.DataFrame([{"ticker": ticker, "cik": roster_cik}]),
    )


def _xom_register() -> dict[str, Registrant]:
    segments = (
        Segment(cik="0000034088", valid_from=None, valid_to=BOUNDARY, evidence="fixture"),
        Segment(cik="0002115436", valid_from=BOUNDARY, valid_to=None, evidence="fixture"),
    )
    return {"XOM": Registrant(ticker="XOM", kind="reorganisation", segments=segments)}


def _messages(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == LOGGER and record.levelno == level]


# --------------------------------------------------------------------------- #
# (a) register UNION + identity CIK                                            #
# --------------------------------------------------------------------------- #
def test_register_union_with_identity_cik_dedups_first_writer_and_logs_labels(monkeypatch, caplog):
    built = _patch_company(
        monkeypatch,
        {
            "XOM": [_filing("a", "2026-08-01"), _filing("b", "2026-02-01"), _filing("done", "2026-03-01")],
            34088: [_filing("b", "2026-02-01"), _filing("c", "2010-01-01"), _filing("done", "2026-03-01")],
            2115436: [_filing("a", "2026-08-01"), _filing("d", "2026-09-01"), _filing("old", "1999-01-01")],
            99: [_filing("d", "2026-09-01"), _filing("e", "2005-05-05")],
        },
    )
    identity = _identity(
        "XOM",
        "0002115436",
        {"0000034088": "E-XOM", "0002115436": "E-XOM", "0000000099": "E-XOM"},
        ("XOM", "0000034088"),
        ("XOM", "0002115436"),
    )
    stats: dict[str, int] = {}
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = resolve_registrant_filings(
            "XOM",
            ["8-K", "8-K/A"],
            since=pd.Timestamp("2000-01-01"),
            done_accessions=frozenset({"done"}),
            registrants=_xom_register(),
            identity=identity,
            stats=stats,
        )

    assert [f.accession_number for f in out] == ["e", "c", "b", "a", "d"]
    assert built == ["XOM", 34088, 2115436, 99]
    assert stats == {"skipped_existing": 1}
    assert _messages(caplog, logging.INFO) == [
        "XOM: 2 from ticker, 1 from 0000034088, 1 from 0002115436, 1 from 0000000099 across the 0000034088 -> 0002115436 boundary (8-K,8-K/A)"
    ]
    print("\n=== SANITY: register UNION + identity CIK ===")
    print(f"  order {[f.accession_number for f in out]}; first writer wins; 'done' skipped once; log label 'ticker' then CIKs")


def test_register_union_with_only_ticker_contributions_is_silent(monkeypatch, caplog):
    _patch_company(monkeypatch, {"XOM": [_filing("a", "2026-08-01")], 34088: [_filing("a", "2026-08-01")]})
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = resolve_registrant_filings("XOM", ["8-K"], since=None, done_accessions=frozenset(), registrants=_xom_register())
    assert [f.accession_number for f in out] == ["a"]
    assert _messages(caplog, logging.INFO) == []
    print("\n=== SANITY: register UNION, nothing from a segment -> no boundary log ===")


def test_register_entry_ignores_identity_aliases(monkeypatch, caplog):
    built = _patch_company(monkeypatch, {"XOM": [_filing("a", "2026-08-01")], "OLDX": [_filing("alias", "2001-01-01")]})
    identity = _identity("XOM", "0002115436", {"0000034088": "E-XOM", "0002115436": "E-XOM"}, ("XOM", "0002115436"), ("OLDX", "0002115436"))
    out = resolve_registrant_filings("XOM", ["8-K"], since=None, done_accessions=frozenset(), registrants=_xom_register(), identity=identity)
    assert [f.accession_number for f in out] == ["a"]
    assert "OLDX" not in built
    print("\n=== SANITY: a register entry makes aliases irrelevant ===")


# --------------------------------------------------------------------------- #
# (b) no register entry: aliases + additive CIK                                #
# --------------------------------------------------------------------------- #
def test_no_entry_aliases_and_additive_cik_log_identity_scope(monkeypatch, caplog):
    built = _patch_company(
        monkeypatch,
        {
            "ZBH": [_filing("a", "2025-01-01")],
            "ZMH": [_filing("a", "2025-01-01"), _filing("b", "2004-01-01")],
            58766: [_filing("c", "1999-01-01"), _filing("b", "2004-01-01")],
        },
    )
    identity = _identity("ZBH", "0001136869", {"0001136869": "E-ZBH", "0000058766": "E-ZBH"}, ("ZBH", "0001136869"), ("ZMH", "0001136869"))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = resolve_registrant_filings("ZBH", ["8-K"], since=None, done_accessions=frozenset(), registrants={}, identity=identity)

    assert [f.accession_number for f in out] == ["c", "b", "a"]
    assert built == ["ZBH", "ZMH", 58766]
    assert _messages(caplog, logging.INFO) == ["ZBH: identity scope added filings (1 from ZBH, 1 from ZMH, 1 from 0000058766)"]
    print("\n=== SANITY: no entry, alias + additive CIK ===")
    print("  ticker first, alias second, identity CIK third; each accession counted once")


def test_no_entry_dead_alias_and_dead_cik_warn_and_keep_the_walk(monkeypatch, caplog):
    _patch_company(monkeypatch, {"ZBH": [_filing("a", "2025-01-01")]}, dead=frozenset({"ZMH", 58766}))
    identity = _identity("ZBH", "0001136869", {"0001136869": "E-ZBH", "0000058766": "E-ZBH"}, ("ZBH", "0001136869"), ("ZMH", "0001136869"))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = resolve_registrant_filings("ZBH", ["8-K"], since=None, done_accessions=frozenset(), registrants={}, identity=identity)
    assert [f.accession_number for f in out] == ["a"]
    assert _messages(caplog, logging.WARNING) == [
        "ZBH: historical alias ZMH could not be resolved",
        "ZBH: register CIK 0000058766 could not be resolved",
    ]
    assert _messages(caplog, logging.INFO) == []
    print("\n=== SANITY: dead alias / CIK cost only their own filings ===")


def test_no_entry_split_form_walks_aliases_but_not_identity_ciks(monkeypatch, caplog):
    built = _patch_company(monkeypatch, {"ZBH": [_filing("a", "2025-01-01")], "ZMH": [_filing("b", "2004-01-01")]})
    identity = _identity("ZBH", "0001136869", {"0001136869": "E-ZBH"}, ("ZBH", "0001136869"), ("ZMH", "0001136869"))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = resolve_registrant_filings("ZBH", ["10-K"], since=None, done_accessions=frozenset(), registrants={}, identity=identity)
    assert [f.accession_number for f in out] == ["b", "a"]
    assert built == ["ZBH", "ZMH"]
    assert _messages(caplog, logging.INFO) == ["ZBH: identity scope added filings (1 from ZBH, 1 from ZMH)"]
    print("\n=== SANITY: SPLIT form, no entry -> alias walked ===")


def test_plain_walk_applies_since_and_done_before_sort(monkeypatch, caplog):
    built = _patch_company(
        monkeypatch, {"AAPL": [_filing("c", "2020-03-01"), _filing("x", "2010-01-01"), _filing("d", "2019-01-01"), _filing("a", "2018-01-01")]}
    )
    stats: dict[str, int] = {}
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = resolve_registrant_filings(
            "AAPL", ["8-K"], since=pd.Timestamp("2015-01-01"), done_accessions=frozenset({"d", "x"}), registrants={}, stats=stats
        )
    assert [f.accession_number for f in out] == ["a", "c"]
    assert built == ["AAPL"]
    assert stats == {"skipped_existing": 2}
    assert caplog.records == [] or _messages(caplog, logging.INFO) == []
    print("\n=== SANITY: plain walk -> since + done, sorted, silent ===")


# --------------------------------------------------------------------------- #
# (c) SPLIT                                                                    #
# --------------------------------------------------------------------------- #
def test_split_duplicate_accession_warns_and_keeps_the_first_segment(monkeypatch, caplog):
    overlapping = {
        "T": Registrant(
            ticker="T",
            kind="reorganisation",
            segments=(
                Segment(cik="0000000001", valid_from=None, valid_to=pd.Timestamp("2021-01-01"), evidence="fixture"),
                Segment(cik="0000000002", valid_from=pd.Timestamp("2020-01-01"), valid_to=pd.Timestamp("2022-01-01"), evidence="fixture"),
                Segment(cik="0000000003", valid_from=pd.Timestamp("2022-01-01"), valid_to=None, evidence="fixture"),
            ),
        )
    }
    built = _patch_company(
        monkeypatch,
        {
            1: [_filing("dup", "2020-06-01"), _filing("old", "2019-01-01"), _filing("late1", "2023-01-01")],
            2: [_filing("dup", "2020-06-01"), _filing("mid", "2021-06-01")],
        },
        dead=frozenset({3}),
    )
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = resolve_registrant_filings("T", ["10-K"], since=None, done_accessions=frozenset(), registrants=overlapping)

    assert [f.accession_number for f in out] == ["old", "dup", "mid"]
    assert built == [1, 2, 3]
    assert _messages(caplog, logging.WARNING) == [
        "T: accession dup kept by BOTH segment 0000000001 and 0000000002 -- the dated split makes that impossible, so the register's boundary is wrong",
        "T: register CIK 0000000003 could not be resolved",
    ]
    print("\n=== SANITY: SPLIT duplicate accession ===")
    print("  overlap -> warning, first segment keeps it; dead segment CIK warns with the ticker")


# --------------------------------------------------------------------------- #
# (d) the same rules on local EDGAR index rows                                 #
# --------------------------------------------------------------------------- #
def _rows(*rows: tuple[str, str, str, str]) -> pd.DataFrame:
    """Index rows `(cik, form, filed, accession)`."""
    return pd.DataFrame(
        [
            {"cik": cik, "company": "Fixture", "form": form, "filed": pd.Timestamp(filed), "accession": accession}
            for cik, form, filed, accession in rows
        ]
    )


def test_index_union_lists_roster_chain_and_identity_ciks_first_writer_wins():
    identity = _identity("XOM", "0000034088", {"0000034088": "E1", "0000099999": "E1"}, ("XOM", "0000034088"))
    df = _rows(
        ("0000034088", "8-K", "2026-05-01", "pred"),
        ("0002115436", "8-K", "2026-07-07", "suc"),
        ("0000099999", "8-K", "2025-01-01", "lineage"),
        ("0002115436", "8-K", "2026-08-03", "shared"),
        ("0000034088", "8-K", "2026-08-03", "shared"),
        ("0000555555", "8-K", "2026-08-04", "foreign"),
        ("0000034088", "10-K", "2026-02-01", "other-form"),
    )

    out = resolve_registrant_entries("XOM", "0000034088", df, ["8-K"], registrants=_xom_register(), identity=identity)

    assert out["accession"].tolist() == ["lineage", "pred", "suc", "shared"]
    assert out.loc[out["accession"] == "shared", "cik"].item() == "0000034088"  # roster CIK is the first writer
    print("\n=== SANITY: index UNION ===")
    print("  roster + chain + identity CIK listed, oldest first; a co-indexed accession once (roster wins); a foreign CIK and other forms dropped")


def test_index_split_keeps_each_segment_inside_its_dates_and_warns_on_overlap(caplog):
    df = _rows(
        ("0000034088", "10-Q", "2026-05-01", "pred-in"),
        ("0000034088", "10-Q", "2026-08-03", "pred-after"),
        ("0002115436", "10-Q", "2026-06-01", "suc-before"),
        ("0002115436", "10-Q", "2026-08-03", "suc-in"),
    )

    out = resolve_registrant_entries("XOM", "0000034088", df, ["10-Q"], registrants=_xom_register())

    assert out["accession"].tolist() == ["pred-in", "suc-in"]
    overlap = Registrant(
        ticker="T",
        kind="reorganisation",
        segments=(
            Segment(cik="0000000001", valid_from=None, valid_to=pd.Timestamp("2026-08-01"), evidence="fixture"),
            Segment(cik="0000000001", valid_from=pd.Timestamp("2026-06-01"), valid_to=None, evidence="fixture"),
        ),
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        dup = resolve_registrant_entries("T", "0000000001", _rows(("0000000001", "10-Q", "2026-07-01", "dup")), ["10-Q"], registrants={"T": overlap})
    assert dup["accession"].tolist() == ["dup"]
    assert any("kept by two segments" in m for m in _messages(caplog, logging.WARNING))
    print("\n=== SANITY: index SPLIT ===")
    print(
        "  each segment lists only its own dates: the predecessor's post-boundary 10-Q and the successor's pre-boundary one are dropped; an overlap warns and keeps one"
    )


def test_index_split_form_without_register_lists_the_roster_cik_only(caplog):
    identity = _identity("ABC", "0000000001", {"0000000001": "E1", "0000000002": "E1"}, ("ABC", "0000000001"))
    df = _rows(("0000000001", "10-K", "2025-02-01", "roster"), ("0000000002", "10-K", "2024-02-01", "uncurated"))

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        out = resolve_registrant_entries("ABC", "0000000001", df, ["10-K"], registrants={}, identity=identity)

    assert out["accession"].tolist() == ["roster"]
    assert any("uncurated" in m for m in _messages(caplog, logging.WARNING))
    print("\n=== SANITY: index SPLIT without a register ===")
    print("  an identity CIK the register has not dated is not listed for a consolidating form, and the run warns")
