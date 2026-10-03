"""
`dei` partition of `symbol_tenure`: cover-page `dei:TradingSymbol` facts captured per Notes zip period.

Parser and aggregation math, so every assertion runs against synthetic Notes zips this file writes and
therefore knows the truth of: distinct-accession counts, monthly/quarterly collapse, idempotent
re-processing, resume on stored periods without a marker file, and partition isolation.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import src.data_extract.utils.fundamentals.fetch_financial_notes as fn
from src.data_extract.utils.common.symbol_tenure import (
    DEI_SOURCE,
    TABLE_COLUMNS,
    aggregate_dei_symbols,
    collapse_dei_periods,
)
from src.data_store.schema import Tables

_SUB_COLS = ["adsh", "cik", "name", "sic", "form", "period", "fy", "fp", "filed"]
_TXT_COLS = ["adsh", "tag", "version", "ddate", "qtrs", "iprx", "lang", "dimh", "dimn", "coreg", "escaped", "txtlen", "value"]

#: (adsh, cik, name, form, filed) of the synthetic filings.
_FILINGS = {
    "A1": ("0000000001-25-000001", "1", "XYZ CORP", "10-Q", "20250710"),
    "A2": ("0000000001-25-000002", "1", "XYZ CORP", "8-K", "20250720"),
    "A3": ("0000000001-25-000003", "1", "XYZ CORP", "8-K", "20250905"),
    "B1": ("0000000002-25-000001", "2", "BROWN CO", "10-K", "20250815"),
    "D1": ("0000000004-25-000001", "4", "DOW CHEMICAL", "10-Q", "20250801"),
}

#: (filing key, tag, dimn, coreg, value) facts; the same accession may tag the same symbol twice.
_FACTS = {
    "A1": [("TradingSymbol", "0", "", "XYZ"), ("SecurityExchangeName", "0", "", "NYSE")],
    "A2": [("TradingSymbol", "0", "", "XYZ"), ("TradingSymbol", "1", "", "xyz"), ("TradingSymbol", "1", "", "XYZ.PRA")],
    "A3": [("TradingSymbol", "0", "", "XYZ")],
    "B1": [("TradingSymbol", "1", "", "BFA, BFB"), ("TradingSymbol", "1", "", "(NONE)")],
    "D1": [("TradingSymbol", "1", "DowInc", "DOW")],
}


def _row(cols: list[str], **values: str) -> str:
    return "\t".join(values.get(col, "") for col in cols)


def _write_notes_zip(cache: Path, period: str, keys: list[str]) -> Path:
    """A minimal `<period>_notes.zip` with `sub.tsv` and `txt.tsv` holding the named filings."""
    sub = ["\t".join(_SUB_COLS)]
    txt = ["\t".join(_TXT_COLS)]
    for key in keys:
        adsh, cik, name, form, filed = _FILINGS[key]
        sub.append(_row(_SUB_COLS, adsh=adsh, cik=cik, name=name, form=form, filed=filed))
        for tag, dimn, coreg, value in _FACTS[key]:
            txt.append(_row(_TXT_COLS, adsh=adsh, tag=tag, ddate=filed, qtrs="0", dimn=dimn, coreg=coreg, value=value))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("sub.tsv", "\n".join(sub) + "\n")
        archive.writestr("txt.tsv", "\n".join(txt) + "\n")
    path = cache / f"{period}_notes.zip"
    path.write_bytes(buffer.getvalue())
    return path


def _context(store: Any) -> Any:
    return SimpleNamespace(
        store=store,
        log=logging.getLogger("test.dei_symbol_tenure"),
        config=SimpleNamespace(local=SimpleNamespace(paths=SimpleNamespace(financial_notes="unused"))),
    )


def _spy_writes(store: Any, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    writes: list[str] = []
    for method in ("save", "replace", "delete"):
        real = getattr(store, method)
        monkeypatch.setattr(store, method, lambda *args, _real=real, _name=method, **kwargs: (writes.append(_name), _real(*args, **kwargs))[1])
    return writes


def _seed_row(symbol: str, cik: str, source: str, evidence_period: str) -> dict[str, object]:
    return {
        "symbol": symbol,
        "issuer_cik": cik,
        "valid_from": pd.Timestamp("2020-01-01"),
        "valid_to": pd.Timestamp("2020-02-01"),
        "n_filings": 1,
        "source": source,
        "evidence_period": evidence_period,
        "evidence": f"{source} seed",
    }


def _day(value: Any) -> str:
    """A stored or computed date as `YYYY-MM-DD` (SQLite returns `datetime.date`, pandas `Timestamp`)."""
    return str(pd.Timestamp(value).date())


def _stored(store: Any) -> pd.DataFrame:
    stored = store.load(Tables.symbol_tenure, project=True)
    assert stored is not None
    return stored.sort_values(["source", "evidence_period", "symbol", "issuer_cik"], ignore_index=True)


def test_one_zip_aggregates_normalised_symbols_over_distinct_accessions(tmp_path: Path) -> None:
    """Known truth for one zip: distinct accessions per (symbol, CIK), symbols normalised, co-registrant facts ignored."""
    path = _write_notes_zip(tmp_path, "2025q3", ["A1", "A2", "A3", "B1", "D1"])
    facts = fn._read_dei_facts(path)
    assert facts is not None
    rows = aggregate_dei_symbols(facts, "2025q3")

    assert list(rows.columns) == list(TABLE_COLUMNS)
    got = {(r.symbol, r.issuer_cik): (r.n_filings, _day(r.valid_from), _day(r.valid_to)) for r in rows.itertuples(index=False)}
    assert got == {
        ("XYZ", "0000000001"): (3, "2025-07-10", "2025-09-06"),  # A2 tags XYZ twice -> still one accession
        ("XYZ-PRA", "0000000001"): (1, "2025-07-20", "2025-07-21"),
        ("BFA", "0000000002"): (1, "2025-08-15", "2025-08-16"),
        ("BFB", "0000000002"): (1, "2025-08-15", "2025-08-16"),
    }, got
    assert set(rows["source"]) == {DEI_SOURCE} and set(rows["evidence_period"]) == {"2025q3"}
    assert "DOW" not in set(rows["symbol"]), "a co-registrant's symbol must not be attributed to the primary filer's CIK"

    print("\n=== SANITY CHECK: one Notes zip -> dei rows ===")
    print(f"  5 filings, 9 symbol facts -> {len(rows)} rows: {sorted(got)}")
    print("  OK: counts are distinct accessions; 'xyz'/'BFA, BFB'/'XYZ.PRA' normalise; '(NONE)' and the coreg DOW fact drop")


def test_monthly_and_quarterly_zips_holding_the_same_accessions_collapse_to_distinct_count(tmp_path: Path, sqlite_store: Any) -> None:
    """2025_07 and 2025_08 roll into 2025q3: per-period rows keep their own counts, the collapse counts each accession once."""
    _write_notes_zip(tmp_path, "2025_07", ["A1", "A2"])
    _write_notes_zip(tmp_path, "2025_08", ["B1"])
    _write_notes_zip(tmp_path, "2025q3", ["A1", "A2", "A3", "B1"])
    written = fn.capture_dei_symbols(_context(sqlite_store), tmp_path, ["2025_07", "2025_08", "2025q3"])

    stored = _stored(sqlite_store)
    per_period = stored[stored["symbol"].eq("XYZ")].set_index("evidence_period")["n_filings"].to_dict()
    assert per_period == {"2025_07": 2, "2025q3": 3}, per_period
    collapsed = collapse_dei_periods(stored).set_index(["symbol", "issuer_cik"])
    distinct_xyz_accessions = len({"A1", "A2", "A3"})
    assert collapsed.loc[("XYZ", "0000000001"), "n_filings"] == distinct_xyz_accessions
    assert collapsed.loc[("BFA", "0000000002"), "n_filings"] == 1
    assert _day(collapsed.loc[("XYZ", "0000000001"), "valid_from"]) == "2025-07-10"
    assert _day(collapsed.loc[("XYZ", "0000000001"), "valid_to"]) == "2025-09-06"
    naive_sum = int(stored.loc[stored["symbol"].eq("XYZ"), "n_filings"].sum())

    print("\n=== SANITY CHECK: monthly + quarterly zips ===")
    print(f"  wrote {written} rows; XYZ per period {per_period}; naive sum {naive_sum}; collapsed {distinct_xyz_accessions}")
    print("  OK: a month inside a captured quarter is superseded, so each accession counts once")


def test_reprocessing_a_captured_zip_is_a_row_identical_no_op(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Capturing the same zip twice leaves the table row-identical and issues no second write."""
    _write_notes_zip(tmp_path, "2025q3", ["A1", "A2", "A3", "B1"])
    context = _context(sqlite_store)
    writes = _spy_writes(sqlite_store, monkeypatch)

    fn.capture_dei_symbols(context, tmp_path, ["2025q3"])
    first, first_writes = _stored(sqlite_store), list(writes)
    fn.capture_dei_symbols(context, tmp_path, ["2025q3"])
    second = _stored(sqlite_store)

    pd.testing.assert_frame_equal(first, second)
    assert writes == first_writes, f"an unchanged period must not be rewritten: {writes}"

    print("\n=== SANITY CHECK: idempotent dei re-processing ===")
    print(f"  {len(first)} rows after the first capture, identical after the second; writes {first_writes} then none")
    print("  OK: re-processing a zip is a no-op")


def test_download_captures_only_new_periods_and_writes_no_marker_file(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stored `dei` periods A..C are skipped, the new D is captured, other partitions survive, no marker file appears."""
    periods = ["2025_01", "2025_02", "2025_03", "2025_04"]
    for period in periods:
        _write_notes_zip(tmp_path, period, ["A1"])
    seeded = [_seed_row("XYZ", "0000000001", DEI_SOURCE, period) for period in periods[:3]]
    seeded.append(_seed_row("XYZ", "0000000001", "form345", ""))
    seeded.append(_seed_row("XYZ", "0000000001", "manual", ""))
    sqlite_store.save(Tables.symbol_tenure, pd.DataFrame(seeded))
    before = _stored(sqlite_store)

    read: list[str] = []
    real_read = fn._read_dei_facts
    monkeypatch.setattr(fn, "_read_dei_facts", lambda path: (read.append(path.name), real_read(path))[1])
    monkeypatch.setattr(fn, "cache_dir", lambda context, key: tmp_path)
    monkeypatch.setattr(fn, "_notes_periods", lambda context, years_history: periods)
    monkeypatch.setattr(fn, "ensure_zip", lambda context, path, urls, **kwargs: path)

    written = fn.download_financial_notes(_context(sqlite_store))
    after = _stored(sqlite_store)

    assert read == ["2025_04_notes.zip"], read
    assert written == 1
    assert sorted(after.loc[after["source"].eq(DEI_SOURCE), "evidence_period"]) == periods
    untouched = after[after["evidence_period"].ne("2025_04")].reset_index(drop=True)
    pd.testing.assert_frame_equal(untouched, before)
    markers = sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".json")
    assert markers == [], markers

    print("\n=== SANITY CHECK: dei resume on stored periods ===")
    print(f"  stored dei periods {periods[:3]} + form345 + manual; read {read}; wrote {written}; marker files {markers}")
    print("  OK: only the new period is read, the other partitions are untouched, and no marker file is written")


def test_full_recapture_replaces_only_that_periods_dei_rows(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """`full` re-derives a captured period: its stale `dei` rows go, form345/manual and other periods stay."""
    _write_notes_zip(tmp_path, "2025q3", ["A1"])
    seeded = [
        _seed_row("OLD", "0000000009", DEI_SOURCE, "2025q3"),
        _seed_row("KEEP", "0000000009", DEI_SOURCE, "2025q2"),
        _seed_row("XYZ", "0000000001", "form345", ""),
        _seed_row("XYZ", "0000000001", "manual", ""),
    ]
    sqlite_store.save(Tables.symbol_tenure, pd.DataFrame(seeded))
    monkeypatch.setattr(fn, "cache_dir", lambda context, key: tmp_path)
    monkeypatch.setattr(fn, "_notes_periods", lambda context, years_history: ["2025q3"])
    monkeypatch.setattr(fn, "ensure_zip", lambda *args, **kwargs: pytest.fail("a cached, captured period needs no download"))

    fn.download_financial_notes(_context(sqlite_store), full=True)
    rows = {(r.symbol, r.source, r.evidence_period) for r in _stored(sqlite_store).itertuples(index=False)}

    assert rows == {("XYZ", DEI_SOURCE, "2025q3"), ("KEEP", DEI_SOURCE, "2025q2"), ("XYZ", "form345", ""), ("XYZ", "manual", "")}, rows

    print("\n=== SANITY CHECK: dei partition isolation ===")
    print(f"  after a full re-capture of 2025q3: {sorted(rows)}")
    print("  OK: the stale 2025q3 dei row is replaced; form345, manual and the 2025q2 dei rows survive")
