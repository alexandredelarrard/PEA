"""
defeatbeta earnings-call extractor (src/data_extract/utils/behavioral/fetch_earnings_call_transcripts.py).

Known-truth fixtures: a tmp parquet with the source schema and two-call row groups, read into the
real SQLite-backed `DataStore`. Plus one live smoke test on two tickers (skipped offline).
"""

from __future__ import annotations

import datetime as dt
import logging
import types
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

from src.constants.constants import EARNINGS_CALL_SENTIMENT_INVALID_PENDING_MODEL
from src.data_extract.utils.behavioral import fetch_earnings_call_transcripts as ect
from src.data_extract.utils.common.identity import CikWindow, Identity, SymbolInterval, build_identity
from src.data_store.schema import Resume, Tables

_PARAGRAPH = pa.struct([("paragraph_number", pa.int32()), ("speaker", pa.string()), ("content", pa.string())])
_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("fiscal_year", pa.int32()),
        ("fiscal_quarter", pa.int32()),
        ("report_date", pa.string()),
        ("transcripts_id", pa.int32()),
        ("transcripts", pa.list_(_PARAGRAPH)),
    ]
)
_CFG = OmegaConf.create({"read_workers": 2})


def _call(symbol: str, year: int, quarter: int, date: str, tid: int | None, n: int, tag: str = "v1") -> dict[str, Any]:
    paragraphs = [
        {"paragraph_number": i, "speaker": f"Speaker {i % 2}", "content": f"{symbol} {year}Q{quarter} {tag} paragraph {i}"} for i in range(1, n + 1)
    ]
    return {"symbol": symbol, "fiscal_year": year, "fiscal_quarter": quarter, "report_date": date, "transcripts_id": tid, "transcripts": paragraphs}


def _write(path: Path, groups: list[list[dict[str, Any]]], schema: pa.Schema = _SCHEMA) -> Path:
    """One parquet row group per inner list, in order (the source is symbol-sorted)."""
    with pq.ParquetWriter(path, schema) as writer:
        for group in groups:
            writer.write_table(pa.Table.from_pylist(group, schema=schema))
    return path


class _Opener:
    """Counts opens of the fixture file."""

    def __init__(self, path: Path) -> None:
        self.path, self.opens = path, 0

    def __call__(self) -> Any:
        self.opens += 1
        return open(self.path, "rb")


def _source(path: Path, tag: str) -> tuple[ect.TranscriptSource, _Opener]:
    opener = _Opener(path)
    return ect.TranscriptSource(revision=f"rev-{tag}", opener=opener), opener


def _ctx(tmp_path: Path, store: Any) -> Any:
    return types.SimpleNamespace(
        store=store,
        log=logging.getLogger("test_earnings_call_transcripts"),
        paths={"DATA_STORE": tmp_path},
        config=types.SimpleNamespace(data_extract=types.SimpleNamespace(redundant_ticks=["GOOG"])),
        _identity_resolver=_identity(),
    )


def _identity() -> Identity:
    names = ["AAA", "BF-B", "BRK-B", "CCC", "DG", "EEE", "RJF", "AAPL", "MSFT", "ECHO", "BNY", "BBB"]
    roster = pd.DataFrame({"ticker": names, "cik": [str(i + 1).zfill(10) for i in range(len(names))]})
    rows = []
    for symbol, ticker, start, end in [
        ("SATS", "ECHO", "2008-01-01", "2026-06-24"),
        ("ECHO", "ECHO", "2026-06-24", None),
        ("BK", "BNY", "2007-01-01", "2026-05-21"),
        ("BNY", "BNY", "2026-05-21", None),
        ("BF-A", "BF-B", "2000-01-01", None),
    ]:
        cik = roster.set_index("ticker").loc[ticker, "cik"]
        rows.append(
            dict(cik=cik, entity_id=f"E{cik}", role="symbol", symbol=symbol, valid_from=start, valid_to=end, status="curated", sources="manual")
        )
    rows.append(
        dict(
            cik="0000000999",
            entity_id="E0000000999",
            role="symbol",
            symbol="ECHO",
            valid_from="2000-01-01",
            valid_to="2008-01-01",
            status="curated",
            sources="manual",
        )
    )
    tenure = pd.DataFrame(
        [
            dict(symbol=row["symbol"], issuer_cik=row["cik"], valid_from=row["valid_from"], valid_to=row["valid_to"], n_filings=1, source="manual")
            for row in rows
        ]
        + [
            dict(symbol=row.ticker, issuer_cik=row.cik, valid_from="1900-01-01", valid_to=None, n_filings=1, source="manual")
            for row in roster.itertuples()
            if row.ticker not in {"ECHO", "BNY"}
        ]
    )
    return build_identity(
        lineage=pd.DataFrame(rows),
        tenure=tenure,
        roster=roster,
        redundant_symbols=frozenset({"GOOG", "BFA"}),
    )


def _recovery_context(tmp_path: Path, store: Any) -> Any:
    """Real Broadcom intro with a known-truth dated master boundary."""
    ctx = _ctx(tmp_path, store)
    roster = pd.DataFrame({"ticker": ["AVGO"], "name": ["Broadcom"], "cik": ["1730168"]})
    lineage = pd.DataFrame(
        [
            dict(
                cik="1730168",
                entity_id="E1730168",
                role="cik_window",
                symbol="",
                valid_from="2016-01-01",
                valid_to=None,
                status="curated",
                sources="manual",
            ),
            dict(
                cik="1730168",
                entity_id="E1730168",
                role="symbol",
                symbol="AVGO",
                valid_from="2016-01-01",
                valid_to=None,
                status="curated",
                sources="manual",
            ),
        ]
    )
    tenure = pd.DataFrame([dict(symbol="AVGO", issuer_cik="1730168", valid_from="2016-01-01", valid_to=None, n_filings=1, source="manual")])
    ctx._identity_resolver = build_identity(lineage, tenure, roster)
    store.save(Tables.sp500_tickers, roster)
    return ctx


def _broadcom_intro(tid: int | None = 72543, quarter: int = 4) -> dict[str, Any]:
    call = _call("AVGO", 2005, quarter, "2006-01-31", tid, 2)
    # Saved defeatbeta AVGO 2005Q4, paragraph 3; the preceding rows are participant rosters.
    call["transcripts"] = [
        {"paragraph_number": 1, "speaker": "Executives", "content": "Scott McGregor, President and Chief Executive Officer"},
        {
            "paragraph_number": 2,
            "speaker": "Operator",
            "content": "Welcome to the Broadcom’s Fourth Quarter Fiscal Year 2005 Earnings Conference Call. During the presentation, all participants will be in a listen-only mode.",
        },
    ]
    return call


def test_intro_recovers_raw_gap_and_repeats_without_writes(tmp_path: Path, sqlite_store: Any) -> None:
    ctx = _recovery_context(tmp_path, sqlite_store)
    source = _source(_write(tmp_path / "intro.parquet", [[_broadcom_intro()]]), "intro")[0]
    first = ect.extract_earnings_calls(ctx, _CFG, tickers=["AVGO"], source=source)
    assert first.calls_new == 1 and first.rows_written == 2
    rows = _sections(sqlite_store)
    assert set(rows["quarter"]) == {"2005Q4"} and set(rows["as_of"].astype(str)) == {"2006-01-31"}
    for full in (False, True, True):
        assert ect.extract_earnings_calls(ctx, _CFG, tickers=["AVGO"], full=full, source=source).rows_written == 0
    print("SANITY: an early-name-proven raw gap recovers at its source date; ordinary/full repeats preserve it with zero writes.")


@pytest.mark.parametrize("fallback_date", ["2006-01-31", "2026-01-31"])
def test_normal_provider_call_wins_before_it_is_stored(tmp_path: Path, sqlite_store: Any, fallback_date: str) -> None:
    ctx = _recovery_context(tmp_path, sqlite_store)
    identity = ctx._identity_resolver
    entity = identity.universe_entity("AVGO")
    ctx._identity_resolver = replace(
        identity, windows_by_entity={entity: tuple(replace(w, valid_to=pd.Timestamp("2020-01-01")) for w in identity.windows_by_entity[entity])}
    )
    normal = _call("AVGO", 2005, 4, "2016-02-01", 9, 3, "canonical")
    fallback = {**_broadcom_intro(), "report_date": fallback_date}
    source = _source(_write(tmp_path / "priority.parquet", [[fallback, normal]]), "priority")[0]
    result = ect.extract_earnings_calls(ctx, _CFG, tickers=["AVGO"], source=source)
    assert result.calls_new == 1 and set(_sections(sqlite_store)["transcript_id"]) == {9}
    print("SANITY: complete provider canonical occupancy blocks raw recovery even before the normal winner has been downloaded.")


def test_null_id_replacement_compares_content_after_old_source_disappears(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _recovery_context(tmp_path, sqlite_store)
    first = _source(_write(tmp_path / "recovered.parquet", [[_broadcom_intro(None)]]), "recovered")[0]
    ect.extract_earnings_calls(ctx, _CFG, tickers=["AVGO"], source=first)
    _seed_derivatives(sqlite_store, [("AVGO", "2005Q4", "2006-01-31")])
    # Normal ownership becomes available; the former rejected row is gone, with no ID/date signal.
    entity = ctx._identity_resolver.universe_entity("AVGO")
    cik = ctx._identity_resolver.roster_cik["AVGO"]
    ctx._identity_resolver = replace(ctx._identity_resolver, windows_by_entity={entity: (CikWindow(cik, None, None, None, None),)})
    normal = _call("AVGO", 2005, 4, "2006-01-31", None, 3, "normal replacement")
    source = _source(_write(tmp_path / "normal.parquet", [[normal]]), "normal")[0]
    reads: Counter[int] = Counter()
    read = ect._Readers.read

    def spy(reader: Any, group: int, columns: list[str]) -> pa.Table:
        if columns == [ect.TRANSCRIPTS_COLUMN]:
            reads[group] += 1
        return read(reader, group, columns)

    monkeypatch.setattr(ect._Readers, "read", spy)
    result = ect.extract_earnings_calls(ctx, _CFG, tickers=["AVGO"], source=source)
    assert result.calls_reissued == 1 and result.rows_written == 3 and reads == {0: 2}
    assert _sections(sqlite_store)["content"].str.contains("normal replacement").all()
    pending = sqlite_store.load(Tables.earnings_call_sentiment)
    assert set(pending["model"]) == {EARNINGS_CALL_SENTIMENT_INVALID_PENDING_MODEL}
    reads.clear()
    assert ect.extract_earnings_calls(ctx, _CFG, tickers=["AVGO"], full=True, source=source).rows_written == 0
    assert reads == {0: 1}
    print(
        "SANITY: null/null same-date replacement is detected after former fallback disappears; 2 reads replace, 1 read verifies unchanged, caches invalidate."
    )


@pytest.mark.parametrize("comparison", [False, True])
def test_verification_failure_aborts_full_before_deletion(
    tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, comparison: bool
) -> None:
    ctx = _recovery_context(tmp_path, sqlite_store)
    first = [_broadcom_intro(None), _call("AVGO", 2020, 1, "2020-04-21", 7, 2), _call("AVGO", 2020, 2, "2020-07-21", 8, 2)]
    ect.extract_earnings_calls(ctx, _CFG, tickers=["AVGO"], source=_source(_write(tmp_path / "first.parquet", [first]), "first")[0])
    before = _sections(sqlite_store)
    # A valid normal call remains, so existing vanished-ticker protection alone would not help.
    second = [first[0], first[2]] if comparison else [_broadcom_intro(9, quarter=3), first[2]]
    source = _source(_write(tmp_path / "failed.parquet", [second]), "failed")[0]
    read = ect._Readers.read

    def fail(reader: Any, group: int, columns: list[str]) -> pa.Table:
        if columns == [ect.TRANSCRIPTS_COLUMN]:
            raise OSError("required verification unavailable")
        return read(reader, group, columns)

    monkeypatch.setattr(ect._Readers, "read", fail)
    with pytest.raises(OSError, match="verification unavailable"):
        ect.extract_earnings_calls(ctx, _CFG, tickers=["AVGO"], full=True, source=source)
    pd.testing.assert_frame_equal(before, _sections(sqlite_store))
    print(f"SANITY: failed {'null-ID comparison' if comparison else 'intro verification'} leaves all old raw rows untouched before full deletion.")


def test_ambiguous_master_target_is_not_resolved_by_name_or_subset(tmp_path: Path, sqlite_store: Any) -> None:
    ctx = _recovery_context(tmp_path, sqlite_store)
    identity = ctx._identity_resolver
    rows = identity.symbol_intervals["AVGO"]
    other = replace(rows[0], entity="OTHER", cik="0000000002")
    ctx._identity_resolver = replace(
        identity,
        ticker_by_entity={**identity.ticker_by_entity, "OTHER": "OTHER"},
        symbol_intervals={"AVGO": (*rows, other)},
        windows_by_entity={**identity.windows_by_entity, "OTHER": (CikWindow(other.cik, None, None, None, None),)},
    )
    result = ect.extract_earnings_calls(
        ctx, _CFG, tickers=["AVGO"], source=_source(_write(tmp_path / "ambiguous.parquet", [[_broadcom_intro()]]), "ambiguous")[0]
    )
    assert result.rows_written == 0
    print("SANITY: a matching name and single-ticker scope cannot resolve multiple full-master targets.")


def test_symbol_lineage_recovers_old_calls_on_ordinary_resume(tmp_path: Path, sqlite_store: Any) -> None:
    ctx = _ctx(tmp_path, sqlite_store)
    current = [_call("ECHO", 2026, 1, "2026-08-03", 101, 2), _call("BNY", 2026, 1, "2026-07-15", 201, 2)]
    ect.extract_earnings_calls(ctx, _CFG, tickers=["ECHO", "BNY"], source=_source(_write(tmp_path / "current.parquet", [current]), "short")[0])
    groups = [
        [_call("BK", 2008, 1, "2008-04-20", 11, 2)],
        [_call("SATS", 2009, 1, "2009-05-13", 21, 2)],
        current,
        [_call("ECHO", 2007, 1, "2007-05-13", 1, 2)],
    ]
    source = _source(_write(tmp_path / "history.parquet", groups), "history")[0]
    result = ect.extract_earnings_calls(ctx, _CFG, tickers=["ECHO", "BNY"], source=source)
    rows = _sections(sqlite_store)
    assert result.calls_new == 2 and set(rows["ticker"]) == {"ECHO", "BNY"}
    assert rows.groupby("ticker")["quarter"].nunique().to_dict() == {"BNY": 2, "ECHO": 2}
    assert ect.extract_earnings_calls(ctx, _CFG, tickers=["ECHO", "BNY"], source=source).rows_written == 0
    print("SANITY: ordinary resume recovers old BK/SATS, rejects foreign ECHO and repeats with zero writes.")


def test_alias_overlap_deduplicates_after_mapping_and_rejects_secondary_class() -> None:
    rows = [
        _call("ECHO", 2025, 4, "2026-03-02", 3278590, 1),
        _call("SATS", 2025, 4, "2026-03-02", None, 1),
        _call("BNY", 2026, 1, "2026-04-16", 3258718, 1),
        _call("BK", 2026, 1, "2026-04-16", None, 1),
        _call("BF-A", 2026, 1, "2026-04-16", 9, 1),
        _call("BFA", 2026, 1, "2026-04-16", 9, 1),
    ]
    index = pd.DataFrame(rows).assign(row_group=0, row=range(len(rows)))
    calls = ect.calls_from_index(index, {"ECHO", "BNY", "BF-B"}, _identity())[0]
    assert calls.set_index("ticker")["transcript_id"].to_dict() == {"BNY": 3258718, "ECHO": 3278590}
    assert ect.calls_from_index(index, {"ECHO"}, _identity())[0].to_dict("records") == calls.query("ticker == 'ECHO'").to_dict("records")
    print("SANITY: rebased overlaps keep the non-null ID once; scope and sibling classes cannot change ownership.")


def test_call_identity_rejects_conflicts_noise_dei_event_only_and_outside_windows() -> None:
    identity = _identity()
    entity = identity.universe_entity("ECHO")
    cik = identity.roster_cik["ECHO"]
    stamp = pd.Timestamp("2026-03-02")
    interval = SymbolInterval(entity, None, None, "curated", frozenset({"manual"}), cik)
    cases = [
        (replace(interval, status="conflict"),),
        (replace(interval, status="noise"),),
        (replace(interval, sources=frozenset({"dei"})),),
        (replace(interval, cik="0000000888"),),
        (interval, replace(interval, entity=identity.universe_entity("BNY"), cik=identity.roster_cik["BNY"])),
    ]
    for rows in cases:
        altered = replace(identity, symbol_intervals={**identity.symbol_intervals, "OLD": rows})
        assert ect.ticker_for_call(altered, "OLD", stamp).ticker is None
    outside = replace(
        identity, windows_by_entity={**identity.windows_by_entity, entity: (CikWindow(cik, pd.Timestamp("2027-01-01"), None, None, None),)}
    )
    assert ect.ticker_for_call(outside, "SATS", stamp).ticker is None
    assert ect.ticker_for_call(identity, "ECHO", pd.Timestamp("2007-05-13")).ticker is None
    print("SANITY: no foreign, ambiguous, conflicted, noise, DEI-only, acquired or outside-window call is attached.")


@pytest.mark.parametrize("ownership", ["ambiguous", "foreign"])
def test_unsupported_rebasing_rejects_reused_alias_ownership(ownership: str) -> None:
    identity = _identity()
    entity = identity.universe_entity("ECHO")
    alias = SymbolInterval(entity, pd.Timestamp("2027-01-01"), None, "curated", frozenset({"manual"}), identity.roster_cik["ECHO"])
    identity = replace(identity, symbol_intervals={**identity.symbol_intervals, "REBASED": (alias,)})
    if ownership == "foreign":
        identity = replace(identity, tenure_by_symbol={**identity.tenure_by_symbol, "ECHO": (("E0000000999", pd.Timestamp("2000-01-01"), None, 1),)})

    assert identity.entity_for("REBASED", pd.Timestamp("1999-01-01")) is None
    assert ect.ticker_for_call(identity, "REBASED", pd.Timestamp("1999-01-01")).ticker is None
    print(f"SANITY: unsupported rebasing cannot bypass a supported alias with {ownership} lifetime ownership.")


def test_rebasing_keeps_dated_support_and_unreused_early_history() -> None:
    identity = _identity()
    entity = identity.universe_entity("ECHO")
    alias = SymbolInterval(entity, pd.Timestamp("2027-01-01"), None, "curated", frozenset({"manual"}), identity.roster_cik["ECHO"])
    identity = replace(identity, symbol_intervals={**identity.symbol_intervals, "REBASED": (alias,)})
    assert ect.ticker_for_call(identity, "REBASED", pd.Timestamp("2010-01-01")).ticker == "ECHO"
    same_issuer = replace(
        identity,
        tenure_by_symbol={**identity.tenure_by_symbol, "ECHO": tuple(row for row in identity.tenure_by_symbol["ECHO"] if row[0] == entity)},
    )
    assert ect.ticker_for_call(same_issuer, "REBASED", pd.Timestamp("1999-01-01")).ticker == "ECHO"
    assert ect.ticker_for_call(identity, "REBASED", pd.Timestamp("2027-01-01")).ticker == "ECHO"
    foreign_source = replace(
        identity, symbol_intervals={**identity.symbol_intervals, "REBASED": (replace(alias, valid_from=pd.Timestamp("2006-01-01")),)}
    )
    assert ect.ticker_for_call(foreign_source, "ECHO", pd.Timestamp("2007-01-01")).ticker is None
    print("SANITY: dated support, normal dated resolution and unreused early history survive; actual-source foreign ownership still vetoes.")


def test_rebased_support_requires_unambiguous_dated_alias_and_its_own_cik_window() -> None:
    roster = pd.DataFrame({"ticker": ["NEW", "OTHER"], "cik": ["2", "3"]})
    lineage = pd.DataFrame(
        [
            dict(cik=cik, entity_id=entity, role="cik_window", symbol="", valid_from=start, valid_to=end, status="curated", sources="manual")
            for cik, entity, start, end in [("1", "E1", "1900-01-01", "2020-01-01"), ("2", "E1", "1900-01-01", None), ("3", "E2", "1900-01-01", None)]
        ]
        + [
            dict(cik=cik, entity_id=entity, role="symbol", symbol=symbol, valid_from=start, valid_to=end, status="curated", sources="manual")
            for cik, entity, symbol, start, end in [
                ("1", "E1", "OLD", "2000-01-01", "2020-01-01"),
                ("3", "E2", "OLD", "2000-01-01", "2015-01-01"),
                ("2", "E1", "NEW", "2030-01-01", None),
            ]
        ]
    )
    tenure = pd.DataFrame(
        {
            "symbol": ["OLD", "OLD", "NEW"],
            "issuer_cik": ["1", "3", "2"],
            "valid_from": ["2000-01-01", "2000-01-01", "2030-01-01"],
            "valid_to": ["2025-01-01", "2015-01-01", None],
            "n_filings": 1,
            "source": "manual",
        }
    )
    identity = build_identity(lineage, tenure, roster)
    assert ect.ticker_for_call(identity, "OLD", pd.Timestamp("2010-05-01")).ticker is None
    assert ect.ticker_for_call(identity, "NEW", pd.Timestamp("2010-05-01")).ticker is None
    assert ect.ticker_for_call(identity, "NEW", pd.Timestamp("2015-01-01")).ticker == "NEW"
    assert ect.ticker_for_call(identity, "NEW", pd.Timestamp("2024-05-01")).ticker is None
    assert ect.ticker_for_call(identity, "NEW", pd.Timestamp("2030-01-01")).ticker == "NEW"
    same_entity = replace(
        identity,
        tenure_by_symbol={**identity.tenure_by_symbol, "OLD": tuple(("E1", *row[1:]) for row in identity.tenure_by_symbol["OLD"])},
        manual_tenure_by_symbol={},
        symbol_intervals={**identity.symbol_intervals, "OLD": tuple(replace(row, entity="E1") for row in identity.symbol_intervals["OLD"])},
    )
    assert ect.ticker_for_call(same_entity, "NEW", pd.Timestamp("2010-05-01")).ticker == "NEW"
    own = SymbolInterval("E1", pd.Timestamp("2010-01-01"), pd.Timestamp("2011-01-01"), "curated", frozenset({"manual"}), "0000000001")
    rescued = replace(
        identity,
        symbol_intervals={**identity.symbol_intervals, "SAFE": (own,)},
        tenure_by_symbol={**identity.tenure_by_symbol, "SAFE": (("E1", own.valid_from, own.valid_to, 1),)},
    )
    assert ect.ticker_for_call(rescued, "NEW", pd.Timestamp("2010-05-01")).ticker == "NEW"
    assert ect.ticker_for_call(rescued, "OLD", pd.Timestamp("2010-05-01")).ticker is None
    conflicted = replace(rescued, symbol_intervals={**rescued.symbol_intervals, "SAFE": (replace(own, status="conflict"),)})
    assert ect.ticker_for_call(conflicted, "NEW", pd.Timestamp("2010-05-01")).ticker is None
    print(
        "SANITY: dated support rejects overlapping owners and wrong-CIK windows, honors half-open ends and same-entity CIKs; another alias rescues only rebased copies."
    )


def _sections(store: Any) -> pd.DataFrame:
    rows = store.load(Tables.earnings_call_sections)
    return rows.sort_values(["ticker", "quarter", "paragraph"]).reset_index(drop=True)


_AAA = [_call("AAA", 2010, 1, "2010-04-20", 11, 3), _call("AAA", 2026, 1, "2026-04-21", 12, 5)]
_BFB = [_call("BF.B", 2026, 2, "2026-06-05", None, 4), _call("BRK-B", 2026, 2, "2026-08-01", 31, 2)]
_CCC = [_call("CCC", 2012, 1, "2012-04-02", 41, 2), _call("CCC", 2012, 2, "2012-07-02", 42, 2)]
_ZZZ = [_call("ZZZ", 2026, 2, "2026-08-02", 51, 2)]
_BASE = [_AAA, _BFB, _CCC, _ZZZ]


@pytest.mark.parametrize("spelling", ["BF-B", "BF.B", "BF/B"])
def test_row_group_pruning_keeps_history_and_share_class_spellings(tmp_path: Path, spelling: str) -> None:
    group = [{**_BFB[0], "symbol": spelling}, _BFB[1]]
    path = _write(tmp_path / "t.parquet", [_AAA, group, _CCC, _ZZZ])
    every = ect.select_row_groups(pq.ParquetFile(path).metadata, {"AAA", "BF-B", "CCC"})
    assert every == [0, 1, 2]
    missing_stats = tmp_path / "no-stats.parquet"
    pq.write_table(pa.Table.from_pylist(_ZZZ, schema=_SCHEMA), missing_stats, write_statistics=False)
    assert ect.select_row_groups(pq.ParquetFile(missing_stats).metadata, {"CCC"}) == [0]
    print(f"SANITY: {spelling} resolves to BF-B; 2012 history and groups without statistics survive footer pruning.")


def test_first_run_writes_raw_paragraphs_with_repo_ticker_and_call_date(tmp_path: Path, sqlite_store: Any) -> None:
    sqlite_store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": ["AAA", "BF-B", "BRK-B", "CCC", "GOOG"]}))
    source, _ = _source(_write(tmp_path / "t.parquet", _BASE), "f1")
    ctx = _ctx(tmp_path, sqlite_store)

    summary = ect.extract_earnings_calls(ctx, _CFG, source=source)
    rows = _sections(sqlite_store)
    calls = rows.groupby(["ticker", "quarter"]).agg(n=("paragraph", "size"), as_of=("as_of", "first"), tid=("transcript_id", "first"))

    assert not summary.full and summary.calls_new == 6 and summary.calls_reissued == 0
    assert summary.rows_written == len(rows) == 3 + 5 + 4 + 2 + 2 + 2
    assert sorted(rows["ticker"].unique()) == ["AAA", "BF-B", "BRK-B", "CCC"]
    assert calls.loc[("BRK-B", "2026Q2"), "n"] == 2
    assert calls.loc[("BF-B", "2026Q2"), "n"] == 4 and pd.isna(calls.loc[("BF-B", "2026Q2"), "tid"])
    assert calls.loc[("AAA", "2026Q1"), "tid"] == 12
    dates = {key: pd.Timestamp(value).date() for key, value in calls["as_of"].items()}
    assert dates[("AAA", "2026Q1")] == dt.date(2026, 4, 21) and dates[("CCC", "2012Q2")] == dt.date(2012, 7, 2)
    first = rows[(rows["ticker"] == "AAA") & (rows["quarter"] == "2026Q1")]
    assert first["paragraph"].tolist() == [1, 2, 3, 4, 5]
    assert first["content"].iloc[0] == "AAA 2026Q1 v1 paragraph 1" and first["speaker"].iloc[0] == "Speaker 1"

    print("\n=== SANITY CHECK: first defeatbeta load ===")
    print(f"  {summary.calls_new} calls, {summary.rows_written} paragraph rows; BF.B -> {sorted(rows['ticker'].unique())}")
    print(f"  as_of == report_date as a date: AAA 2026Q1 -> {dates[('AAA', '2026Q1')]}")
    print("  OK: raw paragraphs land 1:1, repo ticker form, real call date, NULL transcript id kept")


def test_provider_absence_uses_the_normal_lookup_for_berkshire(tmp_path: Path, sqlite_store: Any) -> None:
    source, opener = _source(_write(tmp_path / "without-berkshire.parquet", [_AAA]), "without-berkshire")
    summary = ect.extract_earnings_calls(_ctx(tmp_path, sqlite_store), _CFG, tickers=["BRK-B"], source=source)
    assert opener.opens == 1 and summary.row_groups == 0 and summary.calls_new == 0 and summary.rows_written == 0
    assert sqlite_store.load(Tables.earnings_call_sections, optional=True) is None
    print("SANITY: BRK-B is checked against the provider footer; absent data yields no paragraphs through the ordinary path.")


def test_shared_reader_retries_open_failures_and_closes_handles(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path / "retry.parquet", [_AAA])
    handles: list[Any] = []
    opens = 0

    def opener() -> Any:
        nonlocal opens
        opens += 1
        if opens == 2:
            raise TimeoutError("worker connection timed out")
        handle = path.open("rb")
        handles.append(handle)
        return handle

    monkeypatch.setattr(ect, "_RETRY_WAIT_SECONDS", 0)
    source = ect.TranscriptSource(revision="retry", opener=opener)
    summary = ect.extract_earnings_calls(_ctx(tmp_path, sqlite_store), _CFG, tickers=["AAA"], source=source)
    assert opens >= 4 and summary.calls_new == 2 and summary.rows_written == 8
    assert all(handle.closed for handle in handles)
    print("SANITY: shared metadata/payload reads retry a transient worker-open failure and close every successful handle.")


def test_incremental_run_writes_only_the_new_call(tmp_path: Path, sqlite_store: Any) -> None:
    ctx = _ctx(tmp_path, sqlite_store)
    first, _ = _source(_write(tmp_path / "a.parquet", _BASE), "f1")
    ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA", "BF-B", "CCC"], source=first)
    before = _sections(sqlite_store)

    grown = [[*_AAA, _call("AAA", 2026, 2, "2026-07-22", 13, 6)], _BFB, _CCC, _ZZZ]
    second, _ = _source(_write(tmp_path / "b.parquet", grown), "f2")
    summary = ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA", "BF-B", "CCC"], source=second)
    after = _sections(sqlite_store)

    assert not summary.full and summary.calls_new == 1 and summary.calls_reissued == 0
    assert summary.rows_written == 6 and len(after) == len(before) + 6
    assert summary.row_groups == 3  # complete scoped metadata includes CCC's historical keys
    assert after.merge(before, how="inner").shape[0] == len(before)  # nothing else rewritten

    print("\n=== SANITY CHECK: incremental diff ===")
    print(f"  new revision adds AAA 2026Q2: {summary.calls_new} new call, {summary.rows_written} rows, {summary.row_groups} groups read")
    print("  OK: the diff against stored (ticker, quarter, transcript_id) writes only the new call")


def test_short_content_is_filtered_without_breaking_call_resume(tmp_path: Path, sqlite_store: Any) -> None:
    first = _call("AAA", 2026, 1, "2026-04-21", 1, 7)
    for paragraph, content in zip(first["transcripts"], [None, "", "retained first", "a", "ab", "retained second", ""], strict=True):
        paragraph["content"] = content
    second = _call("BBB", 2026, 1, "2026-04-22", 2, 2)
    second["transcripts"][0]["content"] = "xy"
    source = _source(_write(tmp_path / "short-content.parquet", [[first, second]]), "short-content")[0]
    ctx = _ctx(tmp_path, sqlite_store)
    result = ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA", "BBB"], source=source)
    rows = _sections(sqlite_store)
    assert result.calls_new == 2 and result.rows_written == 3
    assert (rows["content"].str.len() > 2).all()
    aaa = rows[rows["ticker"].eq("AAA")]
    assert aaa["content"].tolist() == ["retained first", "retained second"]
    assert aaa["speaker"].tolist() == ["Speaker 1", "Speaker 0"]
    assert aaa["paragraph"].tolist() == [1, 2] and set(aaa["transcript_id"]) == {1}
    assert set(pd.to_datetime(aaa["as_of"]).dt.date) == {dt.date(2026, 4, 21)}
    assert rows.loc[rows["ticker"].eq("BBB"), "paragraph"].tolist() == [1]
    assert ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA", "BBB"], source=source).rows_written == 0
    print("SANITY: null/empty/one/two-character content is dropped; retained text/order/dates/IDs survive and the repeat writes nothing.")


def test_reissued_call_replaces_paragraphs_and_invalidates_derivatives(tmp_path: Path, sqlite_store: Any) -> None:
    ctx = _ctx(tmp_path, sqlite_store)
    first, _ = _source(_write(tmp_path / "a.parquet", _BASE), "f1")
    ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA", "BF-B", "CCC"], source=first)
    calls = [("AAA", "2026Q1", "2026-04-21"), ("BF-B", "2026Q2", "2026-06-05"), ("AAA", "2010Q1", "2010-04-20")]
    sqlite_store.save(
        Tables.earnings_call_sentiment,
        pd.DataFrame(
            [
                {"ticker": t, "quarter": q, "tag": tag, "as_of": d, "sent_pos": 0.5, "model": "scored"}
                for t, q, d in calls
                for tag in ("prepared_remarks", "qa")
            ]
        ),
    )
    sqlite_store.save(
        Tables.earning_calls_embedding,
        pd.DataFrame([{"ticker": t, "quarter": q, "seq": s, "as_of": d, "text": "turn"} for t, q, d in calls for s in range(3)]),
    )

    # AAA 2026Q1: new transcript id, 3 paragraphs instead of 5. BF-B 2026Q2: NULL id, call date moved.
    reissued = [
        [_AAA[0], _call("AAA", 2026, 1, "2026-04-21", 99, 3, tag="v2")],
        [_call("BF.B", 2026, 2, "2026-06-09", None, 4, tag="v2"), _BFB[1]],
        _CCC,
        _ZZZ,
    ]
    second, _ = _source(_write(tmp_path / "b.parquet", reissued), "f2")
    summary = ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA", "BF-B", "CCC"], source=second)

    rows = _sections(sqlite_store)
    aaa = rows[(rows["ticker"] == "AAA") & (rows["quarter"] == "2026Q1")]
    bfb = rows[(rows["ticker"] == "BF-B") & (rows["quarter"] == "2026Q2")]
    sentiment = sqlite_store.load(Tables.earnings_call_sentiment)
    embedding = sqlite_store.load(Tables.earning_calls_embedding)
    pending = sentiment[sentiment["model"] == EARNINGS_CALL_SENTIMENT_INVALID_PENDING_MODEL]

    assert summary.calls_reissued == 2 and summary.calls_new == 0
    assert aaa["paragraph"].tolist() == [1, 2, 3] and set(aaa["transcript_id"]) == {99}  # no orphan paragraphs 4-5
    assert aaa["content"].str.contains(" v2 ").all() and bfb["content"].str.contains(" v2 ").all()
    assert {pd.Timestamp(d).date() for d in bfb["as_of"]} == {dt.date(2026, 6, 9)}
    assert set(map(tuple, pending[["ticker", "quarter"]].to_numpy())) == {("AAA", "2026Q1"), ("BF-B", "2026Q2")}
    assert {pd.Timestamp(d).date() for d in pending.loc[pending["ticker"] == "BF-B", "as_of"]} == {dt.date(2026, 6, 5)}  # the earlier date
    assert set(sentiment.loc[sentiment["model"] == "scored", "quarter"]) == {"2010Q1"}
    assert set(map(tuple, embedding[["ticker", "quarter"]].drop_duplicates().to_numpy())) == {("AAA", "2010Q1")}

    print("\n=== SANITY CHECK: re-issued transcript (AC-004) ===")
    print(f"  AAA 2026Q1 id 12 -> 99, 5 -> {len(aaa)} paragraphs; BF-B 2026Q2 (NULL id) date 06-05 -> 06-09")
    print(f"  sentiment + embedding rows of both calls deleted; {len(pending)} pending markers from the earlier date")
    print("  OK: old paragraphs gone, derivatives invalidated, the untouched AAA 2010Q1 cache survives")


def _seed_derivatives(store: Any, calls: list[tuple[str, str, str]]) -> None:
    store.save(
        Tables.earnings_call_sentiment,
        pd.DataFrame(
            [
                {"ticker": t, "quarter": q, "tag": tag, "as_of": d, "sent_pos": 0.5, "model": "scored"}
                for t, q, d in calls
                for tag in ("prepared_remarks", "qa")
            ]
        ),
    )
    store.save(
        Tables.earning_calls_embedding,
        pd.DataFrame([{"ticker": t, "quarter": q, "seq": s, "as_of": d, "text": "turn"} for t, q, d in calls for s in range(3)]),
    )


@pytest.mark.parametrize("failure", ["invalidation", "refresh_marker", "paragraph_write"])
def test_reissued_call_failure_preserves_retry_and_earliest_refresh(
    tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    ctx = _ctx(tmp_path, sqlite_store)
    old = _call("AAA", 2010, 1, "2010-04-20", 1, 3)
    ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA"], source=_source(_write(tmp_path / "old.parquet", [[old]]), "old")[0])
    _seed_derivatives(sqlite_store, [("AAA", "2010Q1", "2010-04-20")])
    corrected = _call("AAA", 2010, 1, "2010-04-22", 99, 2, tag="v2")
    source = _source(_write(tmp_path / "corrected.parquet", [[corrected]]), "corrected")[0]
    save, delete = sqlite_store.save, sqlite_store.delete

    def fail_save(table: Any, frame: pd.DataFrame, **kwargs: Any) -> int:
        if (failure == "refresh_marker" and table == Tables.earnings_call_sentiment) or (
            failure == "paragraph_write" and table == Tables.earnings_call_sections
        ):
            raise RuntimeError(f"injected {failure}")
        return save(table, frame, **kwargs)

    def fail_delete(table: Any, where: dict[str, Any]) -> int:
        if failure == "invalidation" and table == Tables.earning_calls_embedding:
            raise RuntimeError(f"injected {failure}")
        return delete(table, where)

    with monkeypatch.context() as injected:
        injected.setattr(sqlite_store, "save", fail_save)
        injected.setattr(sqlite_store, "delete", fail_delete)
        with pytest.raises(RuntimeError, match=f"injected {failure}"):
            ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA"], source=source)

    if failure != "paragraph_write":
        retained = _sections(sqlite_store)
        assert len(retained) == 3 and set(retained["transcript_id"]) == {1}
    ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA"], source=source)
    rows = _sections(sqlite_store)
    pending = sqlite_store.load(Tables.earnings_call_sentiment)
    embeddings = sqlite_store.load(Tables.earning_calls_embedding, optional=True)
    assert len(rows) == 2 and set(rows["transcript_id"]) == {99} and rows["content"].str.contains(" v2 ").all()
    assert set(pending["model"]) == {EARNINGS_CALL_SENTIMENT_INVALID_PENDING_MODEL}
    assert set(pd.to_datetime(pending["as_of"]).dt.date) == {dt.date(2010, 4, 20)}
    assert embeddings is None or embeddings.empty
    assert ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA"], source=source).rows_written == 0
    print(f"SANITY: {failure} failure retries without stale cache, orphan paragraphs or a lost earliest refresh date.")


def test_metadata_failure_cannot_remove_stored_calls(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(tmp_path, sqlite_store)
    ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA", "BF-B"], source=_source(_write(tmp_path / "old.parquet", _BASE), "old")[0])
    before = _sections(sqlite_store)
    read_index = ect._read_index

    def fail_index(readers: Any, group: int) -> pd.DataFrame:
        if group == 1:
            raise RuntimeError("incomplete source metadata")
        return read_index(readers, group)

    monkeypatch.setattr(ect, "_read_index", fail_index)
    shrunk = _source(_write(tmp_path / "shrunk.parquet", [[_AAA[1]], _BFB]), "shrunk")[0]
    with pytest.raises(RuntimeError, match="incomplete source metadata"):
        ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA", "BF-B"], full=True, source=shrunk)
    pd.testing.assert_frame_equal(_sections(sqlite_store), before)
    print("SANITY: a failed metadata group aborts full reconciliation before any stored paragraph is removed.")


def test_ordinary_run_never_reconciles_from_a_missing_date_frontier(tmp_path: Path, sqlite_store: Any) -> None:
    ctx = _ctx(tmp_path, sqlite_store)
    ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA"], source=_source(_write(tmp_path / "old.parquet", [_AAA]), "old")[0])
    sqlite_store.save(Tables.earnings_call_sections, _sections(sqlite_store).assign(as_of=pd.NaT))
    assert sqlite_store.max_date(Tables.earnings_call_sections) is None
    _seed_derivatives(sqlite_store, [("AAA", "2026Q1", "2026-04-21")])
    corrected = _call("AAA", 2026, 1, "2026-04-21", 99, 2, tag="v2")
    source = _source(_write(tmp_path / "shrunk.parquet", [[corrected]]), "shrunk")[0]
    result = ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA"], source=source)
    assert not result.full and result.calls_removed == 0
    rows = _sections(sqlite_store)
    assert set(rows["quarter"]) == {"2010Q1", "2026Q1"}
    current = rows[rows["quarter"].eq("2026Q1")]
    assert len(current) == 2 and set(current["transcript_id"]) == {99}
    pending = sqlite_store.load(Tables.earnings_call_sentiment)
    assert set(pending["model"]) == {EARNINGS_CALL_SENTIMENT_INVALID_PENDING_MODEL}
    embeddings = sqlite_store.load(Tables.earning_calls_embedding, optional=True)
    assert embeddings is None or embeddings.empty
    print("SANITY: a missing frontier cannot enable deletion; a reissue with a missing old date still invalidates caches and replaces paragraphs.")


def _spy_invalidations(monkeypatch: pytest.MonkeyPatch) -> list[set[tuple[str, str]]]:
    seen: list[set[tuple[str, str]]] = []
    real = ect.invalidate_earnings_call_derivatives

    def spy(context: Any, calls: pd.DataFrame) -> int:
        seen.append(set(map(tuple, calls[["ticker", "quarter"]].astype(str).to_numpy())))
        return real(context, calls)

    monkeypatch.setattr(ect, "invalidate_earnings_call_derivatives", spy)
    return seen


def test_reconcile_deletes_a_call_removed_from_the_source(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(tmp_path, sqlite_store)
    ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA", "BF-B", "CCC"], source=_source(_write(tmp_path / "a.parquet", _BASE), "f1")[0])
    _seed_derivatives(sqlite_store, [("AAA", "2010Q1", "2010-04-20"), ("AAA", "2026Q1", "2026-04-21")])
    seen = _spy_invalidations(monkeypatch)

    # AAA 2010Q1 is withdrawn upstream; CCC leaves the source entirely (a symbol change reads the same).
    second, _ = _source(_write(tmp_path / "b.parquet", [[_AAA[1]], _BFB, _ZZZ]), "f2")
    summary = ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA", "BF-B", "CCC"], full=True, source=second)

    calls = set(map(tuple, _sections(sqlite_store)[["ticker", "quarter"]].drop_duplicates().to_numpy()))
    sentiment = sqlite_store.load(Tables.earnings_call_sentiment)
    embedding = sqlite_store.load(Tables.earning_calls_embedding)
    pending = sentiment[sentiment["model"] == EARNINGS_CALL_SENTIMENT_INVALID_PENDING_MODEL]

    assert summary.full and summary.calls_removed == 1 and summary.calls_new == 0 and summary.calls_reissued == 0
    assert calls == {("AAA", "2026Q1"), ("BF-B", "2026Q2"), ("CCC", "2012Q1"), ("CCC", "2012Q2")}
    assert seen == [{("AAA", "2010Q1")}]
    assert set(map(tuple, pending[["ticker", "quarter"]].to_numpy())) == {("AAA", "2010Q1")}
    assert {pd.Timestamp(d).date() for d in pending["as_of"]} == {dt.date(2010, 4, 20)}
    assert set(sentiment.loc[sentiment["model"] == "scored", "quarter"]) == {"2026Q1"}
    assert set(embedding["quarter"]) == {"2026Q1"}

    print("\n=== SANITY CHECK: call removed from the source (F-001) ===")
    print(f"  reconcile over a source without AAA 2010Q1 -> {summary.calls_removed} call removed, invalidated {seen}")
    print(
        f"  pending marker dated {sorted(pending['as_of'].astype(str).unique())}; CCC (no source call left) kept: {sorted(q for t, q in calls if t == 'CCC')}"
    )
    print("  OK: stored table = deduplicated source for every ticker still in the source")


def test_reconcile_deletes_the_label_a_relabel_flip_superseded(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    # Two labels on 2020-03-01 and no later call: the highest ordinal (2020Q4) is kept. Once
    # 2020Q1 arrives on 2020-06-01, the chain keeps 2019Q4 for 2020-03-01 instead.
    first = [_call("EEE", 2019, 4, "2020-03-01", 21, 2), _call("EEE", 2020, 4, "2020-03-01", 22, 2)]
    ctx = _ctx(tmp_path, sqlite_store)
    ect.extract_earnings_calls(ctx, _CFG, tickers=["EEE"], source=_source(_write(tmp_path / "a.parquet", [first]), "f1")[0])
    before = set(_sections(sqlite_store)["quarter"])
    _seed_derivatives(sqlite_store, [("EEE", "2020Q4", "2020-03-01")])
    seen = _spy_invalidations(monkeypatch)

    grown = [[*first, _call("EEE", 2020, 1, "2020-06-01", 23, 2)]]
    summary = ect.extract_earnings_calls(ctx, _CFG, tickers=["EEE"], full=True, source=_source(_write(tmp_path / "b.parquet", grown), "f2")[0])
    rows = _sections(sqlite_store).drop_duplicates(["ticker", "quarter"])
    sentiment = sqlite_store.load(Tables.earnings_call_sentiment)

    assert before == {"2020Q4"}
    assert summary.calls_new == 2 and summary.calls_removed == 1
    assert set(rows["quarter"]) == {"2019Q4", "2020Q1"} and not rows.duplicated(["ticker", "as_of"]).any()
    assert seen == [{("EEE", "2020Q4")}]
    assert set(sentiment["model"]) == {EARNINGS_CALL_SENTIMENT_INVALID_PENDING_MODEL}
    assert sqlite_store.load(Tables.earning_calls_embedding, optional=True) is None or sqlite_store.load(Tables.earning_calls_embedding).empty

    print("\n=== SANITY CHECK: relabel flip (F-001) ===")
    print(f"  run 1 stores {sorted(before)} on 2020-03-01; run 2 keeps 2019Q4 there -> stored {sorted(rows['quarter'])}")
    print(f"  {summary.calls_removed} superseded label removed and invalidated; 0 dates with two calls")
    print("  OK: the reconcile restores one call per (ticker, as_of)")


def test_incremental_run_never_deletes_a_stored_call(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(tmp_path, sqlite_store)
    ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA", "BF-B", "CCC"], source=_source(_write(tmp_path / "a.parquet", _BASE), "f1")[0])
    before = _sections(sqlite_store)
    seen = _spy_invalidations(monkeypatch)

    # AAA 2026Q1 (inside the window) and CCC (outside it) leave the source; AAA 2026Q2 is new.
    shrunk = [[_AAA[0], _call("AAA", 2026, 2, "2026-07-22", 13, 2)], _BFB, _ZZZ]
    summary = ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA", "BF-B", "CCC"], source=_source(_write(tmp_path / "b.parquet", shrunk), "f2")[0])
    after = _sections(sqlite_store)

    assert not summary.full and summary.calls_removed == 0 and summary.calls_new == 1
    assert after.merge(before, how="inner").shape[0] == len(before) and len(after) == len(before) + 2
    assert seen == []

    print("\n=== SANITY CHECK: incremental run removes nothing (F-001) ===")
    print(f"  window source without AAA 2026Q1 and CCC -> removed {summary.calls_removed}, all {len(before)} stored rows kept")
    print("  OK: only a reconcile, which sees every source call, may delete")


def test_a_repeated_run_writes_nothing_and_full_forces_a_compare(tmp_path: Path, sqlite_store: Any) -> None:
    ctx = _ctx(tmp_path, sqlite_store)
    path = _write(tmp_path / "a.parquet", _BASE)
    ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA", "CCC"], source=_source(path, "f1")[0])

    again, opener = _source(path, "f1")
    repeated = ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA", "CCC"], source=again)
    forced = ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA", "CCC"], full=True, source=_source(path, "f1")[0])
    rescoped = ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA", "BF-B", "CCC"], source=_source(path, "f1")[0])

    assert not repeated.full and repeated.calls_new == 0 and repeated.rows_written == 0 and opener.opens >= 1
    assert forced.full and forced.calls_new == 0 and forced.calls_reissued == 0 and forced.rows_written == 0
    assert not rescoped.full and rescoped.calls_new == 1 and rescoped.rows_written == 4  # BF-B's call sits inside the window

    print("\n=== SANITY CHECK: no file-hash no-op ===")
    print(
        f"  same file again -> incremental read ({repeated.row_groups} groups), {repeated.rows_written} rows; -F -> full compare, {forced.rows_written} rows"
    )
    print(f"  adding BF-B (rowless, not new) -> incremental, {rescoped.calls_new} call inside the window")
    print("  OK: source metadata and stored call keys decide the work; only explicit -F reconciles removals")


def test_a_new_and_a_reissued_call_land_without_any_manifest(tmp_path: Path, sqlite_store: Any) -> None:
    ctx = _ctx(tmp_path, sqlite_store)
    ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA", "BF-B", "CCC"], source=_source(_write(tmp_path / "a.parquet", _BASE), "f1")[0])
    changed = [[_AAA[0], _call("AAA", 2026, 1, "2026-04-21", 99, 3, tag="v2"), _call("AAA", 2026, 2, "2026-07-22", 13, 6)], _BFB, _CCC, _ZZZ]
    path = _write(tmp_path / "b.parquet", changed)

    first = ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA", "BF-B", "CCC"], source=_source(path, "f2")[0])
    second = ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA", "BF-B", "CCC"], source=_source(path, "f2")[0])

    assert first.calls_new == 1 and first.calls_reissued == 1 and first.rows_written == 6 + 3
    assert second.calls_new == 0 and second.calls_reissued == 0 and second.rows_written == 0
    assert not list(tmp_path.rglob("*manifest*"))
    print("\n=== SANITY CHECK: earnings calls without the manifest ===")
    print(
        f"  new AAA 2026Q2 + re-issued AAA 2026Q1 -> new {first.calls_new}, re-issued {first.calls_reissued}, {first.rows_written} rows; same run again -> {second.rows_written} rows"
    )
    print("  OK: the stored calls alone decide what is new; no manifest file is written")


@pytest.mark.parametrize("added_on", ["2026-09-20", "2000-01-01"])
def test_metadata_scan_recovers_rowless_tickers_regardless_of_added_on(tmp_path: Path, sqlite_store: Any, added_on: str) -> None:
    ctx = _ctx(tmp_path, sqlite_store)
    path = _write(tmp_path / "a.parquet", _BASE)
    ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA", "BF-B"], source=_source(path, "f1")[0])
    sqlite_store.save(
        Tables.sp500_tickers,
        pd.DataFrame(
            {"ticker": ["AAA", "BF-B", "CCC"], "added_on": [pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-01"), pd.Timestamp(added_on)]}
        ),
    )

    summary = ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA", "BF-B", "CCC"], source=_source(path, "f1")[0])
    ccc = _sections(sqlite_store).query("ticker == 'CCC'")

    assert not summary.full
    assert (len(ccc), summary.calls_new) == (4, 2)
    print(f"\n=== SANITY CHECK: new-ticker history (added_on {added_on}) ===")
    print(f"  CCC's 2012 calls recovered for either roster age: {len(ccc)} rows, {summary.row_groups} groups")


def test_duplicate_key_keeps_latest_date_then_highest_id() -> None:
    index = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA", "BBB"],
            "fiscal_year": [2026, 2026, 2026, 2026],
            "fiscal_quarter": [1, 1, 1, 1],
            "report_date": ["2026-04-20", "2026-04-21", "2026-04-21", "2026-04-01"],
            "transcripts_id": [7, 5, 6, None],
            "row_group": [0, 0, 1, 1],
            "row": [0, 1, 0, 1],
        }
    )
    calls = ect.calls_from_index(index, {"AAA", "BBB"}, _identity())[0]
    kept = calls.set_index("ticker")

    assert len(calls) == 2
    assert kept.loc["AAA", "transcript_id"] == 6 and (kept.loc["AAA", "row_group"], kept.loc["AAA", "row"]) == (1, 0)
    assert kept.loc["AAA", "quarter"] == "2026Q1" and pd.isna(kept.loc["BBB", "transcript_id"])

    print("\n=== SANITY CHECK: duplicate (symbol, fiscal_year, fiscal_quarter) rule ===")
    print("  3 AAA 2026Q1 rows -> kept 2026-04-21 / id 6 (latest date, then highest id)")
    print("  OK: measured 0 duplicate keys in the 238,891-call index; the rule is deterministic anyway")


def test_fiscal_relabel_keeps_one_call_per_date_chaining_back_from_the_later_call(tmp_path: Path, sqlite_store: Any) -> None:
    # DG (source labels of 2012-2013): four consecutive dates under a FY and a FY+1 label,
    # resolved back from the single-label 2013-03-25 2013Q4 call.
    dg = [
        _call("DG", 2011, 4, "2012-03-22", 1, 2),
        _call("DG", 2012, 4, "2012-03-22", 2, 2),
        _call("DG", 2012, 1, "2012-06-04", 3, 2),
        _call("DG", 2013, 1, "2012-06-04", 4, 2),
        _call("DG", 2012, 2, "2012-08-27", 5, 2),
        _call("DG", 2013, 2, "2012-08-27", 6, 2),
        _call("DG", 2012, 3, "2012-12-11", 7, 2),
        _call("DG", 2013, 3, "2012-12-11", 8, 2),
        _call("DG", 2013, 4, "2013-03-25", 9, 2),
    ]
    # EEE: the next call implies the LOWER label, so the chain beats the highest ordinal;
    # its latest date is multi-label, so it keeps the highest ordinal.
    eee = [
        _call("EEE", 2019, 4, "2020-03-01", 21, 2),
        _call("EEE", 2020, 4, "2020-03-01", 22, 2),
        _call("EEE", 2020, 1, "2020-06-01", 23, 2),
        _call("EEE", 2020, 2, "2020-09-01", 24, 2),
        _call("EEE", 2020, 3, "2020-09-01", 25, 2),
    ]
    # RJF, verbatim from the 2026-10-01 index: 2015Q2/Q3 on 2015-07-23, 2016Q1/Q2 on 2016-04-21.
    rjf = [
        _call("RJF", 2015, 1, "2015-04-24", 261826, 2),
        _call("RJF", 2015, 2, "2015-07-23", 261827, 2),
        _call("RJF", 2015, 3, "2015-07-23", 261829, 2),
        _call("RJF", 2015, 4, "2015-10-22", 261831, 2),
        _call("RJF", 2016, 1, "2016-04-21", 261833, 2),
        _call("RJF", 2016, 2, "2016-04-21", 261836, 2),
        _call("RJF", 2016, 3, "2016-07-21", 261838, 2),
    ]
    ctx = _ctx(tmp_path, sqlite_store)
    source, _ = _source(_write(tmp_path / "r.parquet", [dg, eee, rjf]), "r1")

    summary = ect.extract_earnings_calls(ctx, _CFG, tickers=["DG", "EEE", "RJF"], source=source)
    calls = _sections(sqlite_store).drop_duplicates(["ticker", "quarter"])
    kept = {(t, pd.Timestamp(d).date().isoformat()): q for t, q, d in calls[["ticker", "quarter", "as_of"]].itertuples(index=False)}

    assert summary.calls_new == len(calls) == 5 + 3 + 5
    assert not calls.duplicated(["ticker", "as_of"]).any()
    assert [kept[("DG", d)] for d in ("2012-03-22", "2012-06-04", "2012-08-27", "2012-12-11", "2013-03-25")] == [
        "2012Q4",
        "2013Q1",
        "2013Q2",
        "2013Q3",
        "2013Q4",
    ]
    assert (kept[("EEE", "2020-03-01")], kept[("EEE", "2020-09-01")]) == ("2019Q4", "2020Q3")
    assert (kept[("RJF", "2015-07-23")], kept[("RJF", "2016-04-21")]) == ("2015Q3", "2016Q2")

    print("\n=== SANITY CHECK: fiscal relabel duplicates (one call per ticker and date) ===")
    print(f"  {len(dg) + len(eee) + len(rjf)} source rows -> {len(calls)} calls, 0 dates with two labels")
    print(f"  DG chain back from 2013Q4: {[kept[('DG', d)] for d in ('2012-03-22', '2012-06-04', '2012-08-27', '2012-12-11')]}")
    print(f"  EEE 2020-03-01 -> {kept[('EEE', '2020-03-01')]} (chain, not max); RJF -> {kept[('RJF', '2015-07-23')]}, {kept[('RJF', '2016-04-21')]}")
    print("  OK: each multi-label date keeps next-later ordinal - 1, else the highest ordinal")


def test_schema_drift_fails_loudly(tmp_path: Path, sqlite_store: Any) -> None:
    drifted = _SCHEMA.set(_SCHEMA.get_field_index("transcripts_id"), pa.field("transcripts_id", pa.string()))
    rows = [[{**c, "transcripts_id": None} for c in group] for group in _BASE]
    source, _ = _source(_write(tmp_path / "d.parquet", rows, schema=drifted), "d1")

    with pytest.raises(ValueError, match="schema drift") as err:
        ect.extract_earnings_calls(_ctx(tmp_path, sqlite_store), _CFG, tickers=["AAA"], source=source)
    assert not sqlite_store.exists(Tables.earnings_call_sections)

    print("\n=== SANITY CHECK: source schema assertion ===")
    print(f"  transcripts_id as string -> {str(err.value)[:70]}...")
    print("  OK: drift raises before any row is written")


@pytest.mark.live
def test_live_defeatbeta_smoke_two_tickers(tmp_path: Path, sqlite_store: Any) -> None:
    try:
        source = ect.resolve_hf_source()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"HuggingFace unreachable: {type(exc).__name__}: {exc}")
    ctx = _ctx(tmp_path, sqlite_store)

    summary = ect.extract_earnings_calls(ctx, _CFG, tickers=["AAPL", "MSFT"], source=source)
    rows = sqlite_store.load(Tables.earnings_call_sections, ["ticker", "quarter", "paragraph", "as_of", "transcript_id"])
    per_ticker = rows.groupby("ticker")["quarter"].nunique()
    first = rows.groupby(["ticker", "quarter"])["paragraph"].min()
    again = ect.extract_earnings_calls(ctx, _CFG, tickers=["AAPL", "MSFT"], source=source)

    assert set(per_ticker.index) == {"AAPL", "MSFT"} and per_ticker.min() >= 40
    assert rows["quarter"].str.fullmatch(r"\d{4}Q[1-4]").all() and (first == 1).all()
    assert pd.to_datetime(rows["as_of"]).min() >= pd.Timestamp(cast(str, cast(Resume, Tables.earnings_call_sections.resume).source_start))
    assert again.rows_written == 0

    print("\n=== SANITY CHECK: live defeatbeta smoke (test store) ===")
    print(f"  revision {summary.revision[:12]}: {summary.row_groups} row groups, calls {per_ticker.to_dict()}, {summary.rows_written} rows")
    print(
        f"  as_of {pd.to_datetime(rows['as_of']).min().date()} -> {pd.to_datetime(rows['as_of']).max().date()}; re-run wrote {again.rows_written} rows"
    )
    print("  OK: pinned HF read lands every call with paragraph 1 and a fiscal label; a re-run writes nothing")
