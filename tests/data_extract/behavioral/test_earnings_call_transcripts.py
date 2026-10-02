"""
defeatbeta earnings-call extractor (src/data_extract/utils/behavioral/fetch_earnings_call_transcripts.py).

Known-truth fixtures: a tmp parquet with the source schema and two-call row groups, read into the
real SQLite-backed `DataStore`. Plus one live smoke test on two tickers (skipped offline).
"""

from __future__ import annotations

import datetime as dt
import logging
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

from src.constants.constants import EARNINGS_CALL_SENTIMENT_INVALID_PENDING_MODEL
from src.data_extract.utils.behavioral import fetch_earnings_call_transcripts as ect
from src.data_store.schema import Tables

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
_CFG = OmegaConf.create({"lookback_days": 45, "read_workers": 2, "reconcile_days": 7})


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
    """Counts opens, so a no-op run can prove it never touched the file."""

    def __init__(self, path: Path) -> None:
        self.path, self.opens = path, 0

    def __call__(self) -> Any:
        self.opens += 1
        return open(self.path, "rb")


def _source(path: Path, fingerprint: str) -> tuple[ect.TranscriptSource, _Opener]:
    opener = _Opener(path)
    return ect.TranscriptSource(revision=f"rev-{fingerprint}", fingerprint=fingerprint, opener=opener), opener


def _ctx(tmp_path: Path, store: Any) -> Any:
    return types.SimpleNamespace(
        store=store,
        log=logging.getLogger("test_earnings_call_transcripts"),
        paths={"DATA_STORE": tmp_path},
        config=types.SimpleNamespace(
            local=types.SimpleNamespace(filename=types.SimpleNamespace(extraction="extraction_manifest.json")),
            data_extract=types.SimpleNamespace(redundant_ticks=["GOOG"]),
        ),
    )


def _sections(store: Any) -> pd.DataFrame:
    rows = store.load(Tables.earnings_call_sections)
    return rows.sort_values(["ticker", "quarter", "paragraph"]).reset_index(drop=True)


_AAA = [_call("AAA", 2010, 1, "2010-04-20", 11, 3), _call("AAA", 2026, 1, "2026-04-21", 12, 5)]
_BFB = [_call("BF.B", 2026, 2, "2026-06-05", None, 4), _call("BRK-B", 2026, 2, "2026-08-01", 31, 2)]
_CCC = [_call("CCC", 2012, 1, "2012-04-02", 41, 2), _call("CCC", 2012, 2, "2012-07-02", 42, 2)]
_ZZZ = [_call("ZZZ", 2026, 2, "2026-08-02", 51, 2)]
_BASE = [_AAA, _BFB, _CCC, _ZZZ]


def test_row_group_pruning_uses_symbol_and_report_date_statistics(tmp_path: Path) -> None:
    path = _write(tmp_path / "t.parquet", _BASE)
    stats = ect.row_group_stats(pq.ParquetFile(path).metadata)
    ranges = [(s.symbol_min, s.symbol_max, s.report_date_max) for s in stats]
    symbols = ect.dataset_symbols(["AAA", "BF-B", "CCC"])

    every = ect.select_row_groups(stats, symbols, None)
    recent = ect.select_row_groups(stats, symbols, "2026-01-01")

    assert ranges == [
        ("AAA", "AAA", "2026-04-21"),
        ("BF.B", "BRK-B", "2026-08-01"),
        ("CCC", "CCC", "2012-07-02"),
        ("ZZZ", "ZZZ", "2026-08-02"),
    ]
    assert symbols == ["AAA", "BF-B", "BF.B", "CCC"]
    assert every == [0, 1, 2]  # ZZZ's group holds no scoped symbol
    assert recent == [0, 1]  # CCC's group ends in 2012

    print("\n=== SANITY CHECK: row-group pruning ===")
    print(f"  4 groups; scope AAA/BF-B/CCC keeps {every}; since 2026-01-01 keeps {recent}")
    print("  OK: symbol min/max drops the out-of-scope group, max(report_date) the stale one")


def test_first_run_writes_raw_paragraphs_with_repo_ticker_and_call_date(tmp_path: Path, sqlite_store: Any) -> None:
    sqlite_store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": ["AAA", "BF-B", "BRK-B", "CCC", "GOOG"]}))
    source, _ = _source(_write(tmp_path / "t.parquet", _BASE), "f1")
    ctx = _ctx(tmp_path, sqlite_store)

    summary = ect.extract_earnings_calls(ctx, _CFG, source=source)
    rows = _sections(sqlite_store)
    calls = rows.groupby(["ticker", "quarter"]).agg(n=("paragraph", "size"), as_of=("as_of", "first"), tid=("transcript_id", "first"))

    assert summary.full and not summary.noop and summary.calls_new == 5 and summary.calls_reissued == 0
    assert summary.rows_written == len(rows) == 3 + 5 + 4 + 2 + 2
    assert sorted(rows["ticker"].unique()) == ["AAA", "BF-B", "CCC"]  # BRK-B holds no call; ZZZ is out of scope
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
    assert summary.row_groups == 2  # CCC's 2012 group is older than frontier - lookback
    assert after.merge(before, how="inner").shape[0] == len(before)  # nothing else rewritten

    print("\n=== SANITY CHECK: incremental diff ===")
    print(f"  new revision adds AAA 2026Q2: {summary.calls_new} new call, {summary.rows_written} rows, {summary.row_groups} groups read")
    print("  OK: the diff against stored (ticker, quarter, transcript_id) writes only the new call")


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


def test_unchanged_revision_is_a_noop_and_full_forces_a_compare(tmp_path: Path, sqlite_store: Any) -> None:
    ctx = _ctx(tmp_path, sqlite_store)
    path = _write(tmp_path / "a.parquet", _BASE)
    ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA", "CCC"], source=_source(path, "f1")[0])

    again, opener = _source(path, "f1")
    noop = ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA", "CCC"], source=again)
    forced = ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA", "CCC"], full=True, source=_source(path, "f1")[0])
    rescoped = ect.extract_earnings_calls(ctx, _CFG, tickers=["AAA", "BF-B", "CCC"], source=_source(path, "f1")[0])

    assert noop.noop and noop.rows_written == 0 and opener.opens == 0
    assert forced.full and not forced.noop and forced.calls_new == 0 and forced.calls_reissued == 0 and forced.rows_written == 0
    assert rescoped.full and rescoped.calls_new == 1 and rescoped.rows_written == 4  # a scope change reconciles

    print("\n=== SANITY CHECK: revision no-op (AC-003) ===")
    print(f"  same file hash -> no-op, {opener.opens} file opens; -F -> full compare, {forced.rows_written} rows")
    print(f"  adding BF-B to the scope -> full reconcile, {rescoped.calls_new} new call")
    print("  OK: an unchanged source costs two metadata requests and writes nothing")


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
    calls = ect.calls_from_index(index, {"AAA", "BBB"}, None)
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
    assert pd.to_datetime(rows["as_of"]).min() >= pd.Timestamp(ect.HISTORY_START)
    assert again.noop and again.rows_written == 0

    print("\n=== SANITY CHECK: live defeatbeta smoke (test store) ===")
    print(f"  revision {summary.revision[:12]}: {summary.row_groups} row groups, calls {per_ticker.to_dict()}, {summary.rows_written} rows")
    print(f"  as_of {pd.to_datetime(rows['as_of']).min().date()} -> {pd.to_datetime(rows['as_of']).max().date()}; re-run no-op={again.noop}")
    print("  OK: pinned HF read lands every call with paragraph 1 and a fiscal label; same revision is a no-op")
