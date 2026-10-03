"""
`symbol_tenure` -- the derivation math on synthetic known-truth zips, plus a real-data
coverage check against the cached Form 345 quarters.

The derivation is PARSING MATH (dates, half-open intervals, overlap preservation), so the
correctness assertions are made against zips this file writes and therefore knows the truth
of. The real-data half then proves the same code fires on the actual cache: `IR`, `COR` and
`WM` are the three cases the identity plan turns on.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.data_extract.utils.common import symbol_tenure as tenure_module
from src.data_extract.utils.common.symbol_tenure import (
    ManualSymbolTenureError,
    _parse_filing_dates,
    changed_tenure_symbols,
    derive_symbol_tenure,
    load_manual_symbol_tenure,
    materialize_symbol_tenure,
    scan_form345_cache,
)
from src.data_store.schema import Tables

#: The real cache. Absent on a fresh clone, so the real-data tests skip rather than fail.
CACHE = Path("data/sec_insider_transactions")

_HEADER = "ACCESSION_NUMBER\tISSUERCIK\tISSUERNAME\tISSUERTRADINGSYMBOL\tFILING_DATE"


def _write_zip(directory: Path, quarter: str, rows: list[tuple[str, str, str, str, str]]) -> Path:
    """A minimal `<quarter>.zip` carrying only SUBMISSION.TSV, the one member read."""
    lines = [_HEADER] + ["\t".join(r) for r in rows]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("SUBMISSION.TSV", "\n".join(lines) + "\n")
    path = directory / f"{quarter}.zip"
    path.write_bytes(buffer.getvalue())
    return path


def _write_manual(directory: Path, payload: dict) -> Path:
    path = directory / "sec" / "symbol_tenure_manual.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_filing_dates_parse_in_both_shapes_the_sec_ships():
    """`DD-MON-YYYY` and ISO both appear across the 81 quarters; both must decode."""
    parsed = _parse_filing_dates(pd.Series(["03-MAR-2006", "2006-03-03", "31-DEC-2019", "2019-12-31", "  17-JUL-2012  ", "", "not-a-date"]))
    assert list(parsed[:4]) == [pd.Timestamp("2006-03-03"), pd.Timestamp("2006-03-03"), pd.Timestamp("2019-12-31"), pd.Timestamp("2019-12-31")]
    assert parsed[4] == pd.Timestamp("2012-07-17")
    assert parsed[5:].isna().all()

    print("\n=== SANITY CHECK: FILING_DATE parsing ===")
    print(f"  DD-MON-YYYY '03-MAR-2006' -> {parsed[0].date()}")
    print(f"  ISO         '2006-03-03'  -> {parsed[1].date()}")
    print("  OK: Both SEC shapes decode to the same day; junk becomes NaT, never a wrong date")
    print("  -> A silent parse failure here would be a silent tenure gap; it cannot happen.")


def test_two_quarter_cache_derives_the_expected_three_tenures(tmp_path):
    """Known truth: `AAA` changes hands, `BBB` does not. 2 symbols -> 3 rows.

    Also pins the two boundary rules in one place: `valid_to == last_filed + 1 day` for a
    tenure that ended, and NULL for one whose last filing lands in the most recent quarter.
    """
    _write_zip(
        tmp_path,
        "2020q1",
        [
            ("a1", "111", "OLDCO INC", "AAA", "05-JAN-2020"),
            ("a2", "111", "OLDCO INC", "AAA", "27-FEB-2020"),
            ("b1", "333", "STEADY CORP", "BBB", "10-JAN-2020"),
        ],
    )
    _write_zip(
        tmp_path,
        "2020q2",
        [
            ("a3", "222", "NEWCO INC", "AAA", "2020-05-14"),
            ("b2", "333", "STEADY CORP", "BBB", "2020-06-30"),
        ],
    )
    out = derive_symbol_tenure(scan_form345_cache(tmp_path))

    assert len(out) == 3
    assert set(out["symbol"]) == {"AAA", "BBB"}
    old = out[(out.symbol == "AAA") & (out.issuer_cik == "0000000111")].iloc[0]
    new = out[(out.symbol == "AAA") & (out.issuer_cik == "0000000222")].iloc[0]
    steady = out[out.symbol == "BBB"].iloc[0]

    assert old.valid_from == pd.Timestamp("2020-01-05")
    # last filing 2020-02-27, in a quarter that is NOT the most recent -> closed at +1 day
    assert old.valid_to == pd.Timestamp("2020-02-28")
    assert old.n_filings == 2
    assert pd.isna(new.valid_to) and pd.isna(steady.valid_to)  # both filed in 2020q2
    assert steady.valid_from == pd.Timestamp("2020-01-10") and steady.n_filings == 2
    assert (out["source"] == "form345").all()
    assert old.evidence == "OLDCO INC"

    # half-open, exactly `registrant.Segment.covers`: the last filing is INSIDE its interval
    assert old.valid_from <= pd.Timestamp("2020-02-27") < old.valid_to

    print("\n=== SANITY CHECK: 2-quarter synthetic derivation ===")
    print(f"  AAA {old.issuer_cik} {old.valid_from.date()} .. {old.valid_to.date()} (closed)")
    print(f"  AAA {new.issuer_cik} {new.valid_from.date()} .. open")
    print(f"  BBB {steady.issuer_cik} {steady.valid_from.date()} .. open")
    print("  OK: 3 rows from 2 symbols; valid_to = last_filed + 1d; open tenure is NULL")
    print("  -> The half-open interval contains the last filing it describes.")


def test_overlapping_tenures_survive_as_two_rows(tmp_path):
    """Two issuers filing under one symbol in the SAME window is real (`COR`, 2010-2012).

    A "last writer wins" collapse would delete one of them, so the table would answer a
    lookup instead of the membership question it exists to answer.
    """
    _write_zip(
        tmp_path,
        "2011q1",
        [
            ("x1", "444", "CORTEX PHARMACEUTICALS INC", "ZZZ", "05-JAN-2011"),
            ("y1", "555", "CORESITE REALTY CORP", "ZZZ", "06-JAN-2011"),
        ],
    )
    _write_zip(
        tmp_path,
        "2011q2",
        [
            ("x2", "444", "CORTEX PHARMACEUTICALS INC", "ZZZ", "01-JUN-2011"),
            ("y2", "555", "CORESITE REALTY CORP", "ZZZ", "02-JUN-2011"),
        ],
    )
    out = derive_symbol_tenure(scan_form345_cache(tmp_path))
    assert len(out) == 2 and out["issuer_cik"].nunique() == 2

    first, second = out.sort_values("valid_from").itertuples()
    # genuinely overlapping: the second starts before the first would have "ended"
    assert pd.Timestamp(cast(Any, second.valid_from)) < pd.Timestamp("2011-06-01")

    print("\n=== SANITY CHECK: overlapping tenures ===")
    for row in out.itertuples():
        print(
            f"  ZZZ {row.issuer_cik} {pd.Timestamp(cast(Any, row.valid_from)).date()} .. "
            f"{'open' if pd.isna(row.valid_to) else pd.Timestamp(cast(Any, row.valid_to)).date()}  n={row.n_filings}"
        )
    print("  OK: Both CIKs kept; neither window was truncated by the other")
    print("  -> Resolution stays a membership test, never a single-answer lookup.")


def test_every_drop_reason_is_counted(tmp_path):
    """A silent drop is a silent tenure gap, so each reason is counted separately."""
    path = _write_zip(
        tmp_path,
        "2015q1",
        [
            ("g1", "777", "GOOD CO", "GOOD", "02-JAN-2015"),
            ("d1", "777", "GOOD CO", "NONE", "02-JAN-2015"),  # pseudo-symbol
            ("d2", "777", "GOOD CO", "", "02-JAN-2015"),  # empty symbol
            ("d3", "0", "ZERO CIK CO", "ZERO", "02-JAN-2015"),  # unusable CIK
            ("d4", "777", "GOOD CO", "BAD", "not-a-date"),  # unparseable date
        ],
    )
    assert path.exists()
    scan = scan_form345_cache(tmp_path)
    drops: Counter = scan.drops
    assert len(scan.tenure_parts) == 1
    agg = scan.tenure_parts[0]

    assert drops["rows_read"] == 5 and drops["rows_kept"] == 1
    assert drops["empty_symbol"] == 2  # "" and "NONE"
    assert drops["empty_cik"] == 1
    assert drops["unparseable_filing_date"] == 1
    assert len(agg) == 1 and agg.iloc[0]["symbol"] == "GOOD"

    print("\n=== SANITY CHECK: drop accounting ===")
    print(
        f"  read={drops['rows_read']} kept={drops['rows_kept']} "
        f"empty_symbol={drops['empty_symbol']} empty_cik={drops['empty_cik']} "
        f"bad_date={drops['unparseable_filing_date']}"
    )
    print("  OK: Every dropped row is attributed to exactly one named reason")
    print("  -> read == kept + the sum of the reasons, so a gap can always be explained.")


def test_a_corrupt_zip_is_skipped_not_raised(tmp_path):
    """The research run hit a corrupt quarter; one bad zip must not lose the other 80."""
    _write_zip(tmp_path, "2020q1", [("a1", "111", "OLDCO INC", "AAA", "05-JAN-2020")])
    (tmp_path / "2020q2.zip").write_bytes(b"this is not a zip file")
    out = derive_symbol_tenure(scan_form345_cache(tmp_path))

    assert len(out) == 1 and out.iloc[0]["symbol"] == "AAA"
    # 2020q2 is still the most recent quarter NAME, so AAA's tenure reads as closed
    assert out.iloc[0]["valid_to"] == pd.Timestamp("2020-01-06")

    print("\n=== SANITY CHECK: corrupt zip ===")
    print("  1 good quarter + 1 corrupt quarter -> 1 tenure, a warning, no exception")
    print("  -> A single bad archive costs its own quarter, never the whole derivation.")


def test_store_round_trip_keeps_date_semantics(sqlite_store):
    """A tenure read back from a store must still compare against a `Timestamp` date.

    Postgres `DATE` columns come back as `datetime.date`, not `Timestamp`, and any comparison
    written against the in-memory frame alone would hide that entirely.
    """
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "issuer_cik": ["0000000111", "0000000222"],
            "valid_from": pd.to_datetime(["2020-01-05", "2020-05-14"]),
            "valid_to": pd.to_datetime(pd.Series(["2020-02-28", None], dtype="object")),
            "n_filings": [2, 1],
            "source": ["form345"] * 2,
            "evidence": ["OLDCO", "NEWCO"],
        }
    )
    sqlite_store.replace(Tables.symbol_tenure, frame)
    back = sqlite_store.load(Tables.symbol_tenure)
    assert back is not None

    closed = back[back.issuer_cik == "0000000111"].iloc[0]
    valid_from = pd.Timestamp(closed["valid_from"])
    valid_to = pd.Timestamp(closed["valid_to"])
    assert valid_from <= pd.Timestamp("2020-02-27") < valid_to
    assert pd.isna(pd.Timestamp(back[back.issuer_cik == "0000000222"].iloc[0]["valid_to"]))

    print("\n=== SANITY CHECK: store round-trip ===")
    print(f"  valid_from read back as {type(closed['valid_from']).__name__}")
    print(f"  normalised -> {valid_from.date()} .. {valid_to.date()}")
    print("  OK: Membership holds after the round-trip; the open tenure stays NULL")
    print("  -> Comparisons must normalise, which is the Postgres DATE trap in one assertion.")


def test_tenure_diff_names_added_removed_and_moved_symbols():
    columns = ["symbol", "issuer_cik", "valid_from", "valid_to"]
    existing = pd.DataFrame(
        [
            ("SAME", "1", "2020-01-01", None),
            ("BOUND", "2", "2020-01-01", "2021-01-01"),
            ("REMOVED", "3", "2020-01-01", None),
        ],
        columns=columns,
    )
    derived = pd.DataFrame(
        [
            ("SAME", "1", "2020-01-01", None),
            ("BOUND", "2", "2020-01-01", "2021-02-01"),
            ("ADDED", "4", "2020-01-01", None),
        ],
        columns=columns,
    )

    assert changed_tenure_symbols(existing, existing.copy()) == []
    assert changed_tenure_symbols(existing, derived) == ["ADDED", "BOUND", "REMOVED"]

    print("\n=== SANITY CHECK: symbol-tenure diff ===")
    print("  added alias, moved boundary and removed alias are named; unchanged rebuild is empty")
    print("  OK: routine refresh logs exactly the symbols whose identity evidence moved")


def test_manual_tenure_loader_normalizes_and_preserves_half_open_boundaries(tmp_path):
    _write_manual(
        tmp_path,
        {
            "version": 1,
            "tickers": {
                "tt": [
                    {
                        "symbol": "ir",
                        "issuer_cik": "0001466258",
                        "valid_from": "2009-07-09",
                        "valid_to": "2020-03-02",
                        "evidence": ["SEC accession one"],
                        "reason": "predecessor symbol",
                    },
                    {
                        "symbol": "tt",
                        "issuer_cik": "0001466258",
                        "valid_from": "2020-03-02",
                        "valid_to": None,
                        "evidence": ["SEC accession two"],
                        "reason": "current symbol",
                    },
                ]
            },
        },
    )

    out = load_manual_symbol_tenure(tmp_path)
    old = out[out["symbol"] == "IR"].iloc[0]
    current = out[out["symbol"] == "TT"].iloc[0]
    assert set(out["canonical_ticker"]) == {"TT"}
    assert old["valid_from"] == pd.Timestamp("2009-07-09")
    assert old["valid_to"] == current["valid_from"] == pd.Timestamp("2020-03-02")
    assert pd.isna(current["valid_to"])
    assert (out["source"] == "manual").all() and (out["n_filings"] == 0).all()

    print("\n=== SANITY CHECK: manual ticker-tenure contract ===")
    print(f"  intervals={len(out)} canonical={out['canonical_ticker'].nunique()} conflicts=0")
    print("  IR excludes 2020-03-02; TT includes it; the open interval remains NULL")
    print("  OK: normalization and half-open boundaries are deterministic")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda second: second.update(valid_from="2019-12-01"), "overlapping manual intervals"),
        (lambda second: second.update(evidence=[]), ".evidence must be"),
        (lambda second: second.update(issuer_cik="1466258"), "zero-padded 10-digit"),
    ],
)
def test_manual_tenure_loader_rejects_unsafe_rows(tmp_path, mutation, message):
    second = {
        "symbol": "TT",
        "issuer_cik": "0001466258",
        "valid_from": "2020-03-02",
        "valid_to": None,
        "evidence": ["SEC accession two"],
        "reason": "current symbol",
    }
    mutation(second)
    _write_manual(
        tmp_path,
        {
            "version": 1,
            "tickers": {
                "TT": [
                    {
                        "symbol": "TT",
                        "issuer_cik": "0000000001",
                        "valid_from": "2010-01-01",
                        "valid_to": "2020-01-01",
                        "evidence": ["SEC accession one"],
                        "reason": "older issuer",
                    },
                    second,
                ]
            },
        },
    )
    with pytest.raises(ManualSymbolTenureError, match=message):
        load_manual_symbol_tenure(tmp_path)


def test_materialization_keeps_manual_and_derived_evidence():
    columns = ["symbol", "issuer_cik", "valid_from", "valid_to", "n_filings", "source", "evidence"]
    derived = pd.DataFrame(
        [["TT", "0000000002", pd.Timestamp("2020-03-10"), pd.NaT, 20, "form345", "derived issuer"]],
        columns=columns,
    )
    manual = pd.DataFrame(
        [["TT", "TT", "0000000001", pd.Timestamp("2020-03-02"), pd.NaT, 0, "manual", "SEC evidence", "current symbol"]],
        columns=["canonical_ticker", *columns, "reason"],
    )
    out = materialize_symbol_tenure(derived, manual)
    assert len(out) == 2
    assert list(out["source"]) == ["manual", "form345"]
    assert set(out["issuer_cik"]) == {"0000000001", "0000000002"}

    print("\n=== SANITY CHECK: manual-over-derived auditability ===")
    print("  manual=1 derived=1 materialized=2; manual is ordered first but neither row is hidden")
    print("  OK: precedence is a resolver concern, not destructive evidence replacement")


def test_materialization_coalesces_an_exact_primary_key_collision():
    columns = ["symbol", "issuer_cik", "valid_from", "valid_to", "n_filings", "source", "evidence"]
    derived = pd.DataFrame(
        [["A", "0001090872", pd.Timestamp("2006-02-16"), pd.NaT, 900, "form345", "AGILENT TECHNOLOGIES INC"]],
        columns=columns,
    )
    manual = pd.DataFrame(
        [["A", "A", "0001090872", pd.Timestamp("2006-02-16"), pd.NaT, 0, "manual", "SEC listing evidence", "verified lower bound"]],
        columns=["canonical_ticker", *columns, "reason"],
    )

    out = materialize_symbol_tenure(derived, manual)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["source"] == "manual"
    assert row["n_filings"] == 900
    assert "manual evidence: SEC listing evidence" in row["evidence"]
    assert "form345 evidence: AGILENT TECHNOLOGIES INC" in row["evidence"]
    assert not out.duplicated(["symbol", "issuer_cik", "valid_from"]).any()
    print("\n=== SANITY CHECK: manual/derived primary-key collision ===")
    print("  one manual-precedence row retains both evidence strings and the derived filing count")
    print("  OK: auditability fits the existing table grain without a schema change")


def test_symbol_tenure_build_logs_cold_and_changed_symbols(sqlite_store, monkeypatch, caplog, tmp_path):
    frame = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "issuer_cik": ["0000000001"],
            "valid_from": pd.to_datetime(["2020-01-01"]),
            "valid_to": pd.to_datetime(pd.Series([None], dtype="object")),
            "n_filings": [1],
            "source": ["form345"],
            "evidence": ["AAA INC"],
        }
    )
    current = {"frame": frame}
    monkeypatch.setattr(tenure_module, "derive_symbol_tenure", lambda scan: current["frame"])
    monkeypatch.setattr(
        tenure_module,
        "load_manual_symbol_tenure",
        lambda config_dir: pd.DataFrame(
            columns=[
                "canonical_ticker",
                "symbol",
                "issuer_cik",
                "valid_from",
                "valid_to",
                "n_filings",
                "source",
                "evidence",
                "reason",
            ]
        ),
    )
    monkeypatch.setattr(tenure_module, "record_run", lambda *args, **kwargs: None)
    context: Any = SimpleNamespace(store=sqlite_store, log=logging.getLogger("test.symbol_tenure"))
    caplog.set_level(logging.INFO, logger="test.symbol_tenure")

    tenure_module.build_symbol_tenure(context, cast(Any, None), tmp_path)
    assert "cold build with 1 row(s) over 1 symbol(s)" in caplog.text
    current["frame"] = pd.concat(
        [
            frame,
            frame.assign(symbol="OLD", valid_from=pd.Timestamp("2010-01-01")),
        ],
        ignore_index=True,
    )
    tenure_module.build_symbol_tenure(context, cast(Any, None), tmp_path)
    assert "1 changed symbol(s): OLD" in caplog.text

    print("\n=== SANITY CHECK: identity refresh visibility ===")
    print("  cold build logs scale only; routine rebuild names changed symbol OLD")
    print("  OK: a new former symbol is visible before symbol-only consumers run")


def test_symbol_tenure_replace_is_skipped_when_unchanged(sqlite_store, monkeypatch, caplog, tmp_path):
    """A rebuild that derives exactly the stored rows (read back through the store) writes nothing."""
    frame = pd.DataFrame(
        {
            "symbol": ["BBB", "AAA"],
            "issuer_cik": ["0000000002", "0000000001"],
            "valid_from": pd.to_datetime(["2015-03-01", "2020-01-01"]),
            "valid_to": pd.to_datetime(pd.Series(["2016-01-01", None], dtype="object")),
            "n_filings": [4, 1],
            "source": ["form345", "form345"],
            "evidence": ["BBB INC", "AAA INC"],
        }
    )
    current = {"frame": frame}
    monkeypatch.setattr(tenure_module, "derive_symbol_tenure", lambda scan: current["frame"])
    manual_columns = ["canonical_ticker", "symbol", "issuer_cik", "valid_from", "valid_to", "n_filings", "source", "evidence", "reason"]
    monkeypatch.setattr(tenure_module, "load_manual_symbol_tenure", lambda config_dir: pd.DataFrame(columns=manual_columns))
    monkeypatch.setattr(tenure_module, "record_run", lambda *args, **kwargs: None)
    replaced: list[int] = []
    real_replace = sqlite_store.replace

    def counting_replace(table, df):
        replaced.append(len(df))
        return real_replace(table, df)

    monkeypatch.setattr(sqlite_store, "replace", counting_replace)
    context: Any = SimpleNamespace(store=sqlite_store, log=logging.getLogger("test.symbol_tenure.skip"))
    caplog.set_level(logging.INFO)

    tenure_module.build_symbol_tenure(context, cast(Any, None), tmp_path)
    tenure_module.build_symbol_tenure(context, cast(Any, None), tmp_path)
    assert replaced == [2]
    assert "symbol_tenure: unchanged (2 row(s)); replace skipped" in caplog.text

    current["frame"] = frame.assign(n_filings=[5, 1])
    tenure_module.build_symbol_tenure(context, cast(Any, None), tmp_path)
    assert replaced == [2, 2]

    print("\n=== SANITY CHECK: unchanged symbol_tenure is not rewritten ===")
    print(f"  cold build wrote once; identical rebuild skipped the replace; a changed n_filings wrote again -> replace calls {replaced}")
    print("  OK: the DATE round-trip and row order do not defeat the comparison")


def _per_group_materialize(derived: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    """Reference: the per-primary-key-group coalescing loop `materialize_symbol_tenure` replaced."""
    table_columns = ["symbol", "issuer_cik", "valid_from", "valid_to", "n_filings", "source", "evidence"]
    out = pd.concat([manual[table_columns], derived[table_columns]], ignore_index=True)
    priority = out["source"].map({"manual": 0, "form345": 1}).fillna(2)
    out = (
        out.assign(_source_priority=priority)
        .sort_values(["symbol", "valid_from", "_source_priority", "issuer_cik"], kind="mergesort")
        .drop(columns="_source_priority")
        .reset_index(drop=True)
    )
    coalesced: list[dict[str, object]] = []
    for _, rows in out.groupby(["symbol", "issuer_cik", "valid_from"], sort=False, dropna=False):
        winner = rows.iloc[0].copy()
        if len(rows) > 1:
            winner["n_filings"] = pd.to_numeric(rows["n_filings"], errors="coerce").max()
            labelled = [f"{row.source} evidence: {row.evidence}" for row in rows.itertuples(index=False) if str(row.evidence).strip()]
            winner["evidence"] = " | ".join(dict.fromkeys(labelled))
        coalesced.append(cast(dict[str, object], winner.to_dict()))
    return pd.DataFrame.from_records(coalesced, columns=table_columns)


def test_materialize_collisions_match_the_per_group_rule():
    """Vectorised coalescing equals the per-group loop on a crafted frame, dtypes included."""
    columns = ["symbol", "issuer_cik", "valid_from", "valid_to", "n_filings", "source", "evidence"]
    day = pd.Timestamp
    derived = pd.DataFrame(
        [
            ["A", "0000000001", day("2006-02-16"), pd.NaT, 900, "form345", "AGILENT"],  # collides with manual
            ["B", "0000000002", day("2010-01-01"), day("2012-01-01"), 7, "form345", "BCO"],  # no collision
            ["C", "0000000003", day("2011-05-05"), pd.NaT, 3, "form345", ""],  # collides, blank evidence
            ["C", "0000000004", day("2011-05-05"), pd.NaT, 2, "form345", "OTHER C"],  # same day, other CIK
            ["D", "0000000005", day("2001-01-01"), day("2003-01-01"), 1, "form345", "DCO"],
            ["E", "0000000006", day("2019-09-09"), pd.NaT, 11, "form345", "ECO"],  # collides with a manual twin
        ],
        columns=columns,
    )
    manual_rows = [
        ["A", "A", "0000000001", day("2006-02-16"), pd.NaT, 0, "manual", "SEC listing", "lower bound"],
        ["C", "C", "0000000003", day("2011-05-05"), day("2020-01-01"), 0, "manual", "8-K", "rename"],
        ["E", "E", "0000000006", day("2019-09-09"), pd.NaT, 0, "manual", "S-4", "merger"],
        ["F", "F", "0000000007", day("2022-02-02"), pd.NaT, 0, "manual", "Form 25", "listing"],
    ]
    manual = pd.DataFrame(manual_rows, columns=["canonical_ticker", *columns, "reason"])

    expected = _per_group_materialize(derived, manual)
    out = materialize_symbol_tenure(derived, manual)

    pd.testing.assert_frame_equal(out, expected, check_dtype=True)
    assert list(out.dtypes) == list(expected.dtypes)
    assert len(out) == 7 and not out.duplicated(["symbol", "issuer_cik", "valid_from"]).any()
    c_row = out[(out.symbol == "C") & (out.issuer_cik == "0000000003")].iloc[0]
    assert c_row["source"] == "manual" and c_row["n_filings"] == 3 and c_row["evidence"] == "manual evidence: 8-K"

    print("\n=== SANITY CHECK: vectorised collision coalescing ===")
    print(f"  {len(derived)} derived + {len(manual)} manual rows, 3 primary-key collisions -> {len(out)} rows")
    print(f"  dtypes {dict(out.dtypes.astype(str))}")
    print("  OK: identical to the per-group loop (values, order, dtypes); a blank evidence string is not labelled")


def test_repository_manual_tenure_covers_validated_ia3_boundaries():
    """Every IA-3 boundary is exact, evidenced and half-open in the live config."""
    manual = load_manual_symbol_tenure(Path("configs"))
    transitions = (
        ("APA", "APA", "APA", "2021-03-02", "0000006769", "0001841666"),
        ("BALL", "BLL", "BALL", "2022-05-10", "0000009389", "0000009389"),
        ("BNY", "BK", "BNY", "2026-05-21", "0001390777", "0001390777"),
        ("BG", "BG", "BG", "2023-11-01", "0001144519", "0001996862"),
        ("BLK", "BLK", "BLK", "2024-10-01", "0001364742", "0002012383"),
        ("COHR", "IIVI", "COHR", "2022-09-08", "0000820318", "0000820318"),
        ("DIS", "DIS", "DIS", "2019-03-20", "0001001039", "0001744489"),
        ("EG", "RE", "EG", "2023-07-10", "0001095073", "0001095073"),
        ("ELV", "ANTM", "ELV", "2022-06-28", "0001156039", "0001156039"),
        ("EXE", "CHK", "EXE", "2024-10-02", "0000895126", "0000895126"),
        ("GL", "TMK", "GL", "2019-08-09", "0000320335", "0000320335"),
        ("J", "JEC", "J", "2019-12-10", "0000052988", "0000052988"),
        ("LHX", "HRS", "LHX", "2019-07-01", "0000202058", "0000202058"),
        ("MRSH", "MMC", "MRSH", "2026-01-14", "0000062709", "0000062709"),
        ("RVTY", "PKI", "RVTY", "2023-05-16", "0000031791", "0000031791"),
        ("XYZ", "SQ", "XYZ", "2025-01-21", "0001512673", "0001512673"),
    )

    for ticker, old_symbol, new_symbol, boundary, old_cik, new_cik in transitions:
        stamp = pd.Timestamp(boundary)
        rows = manual[manual["canonical_ticker"].eq(ticker)]
        old = rows[rows["symbol"].eq(old_symbol) & rows["issuer_cik"].eq(old_cik) & rows["valid_to"].eq(stamp)]
        new = rows[rows["symbol"].eq(new_symbol) & rows["issuer_cik"].eq(new_cik) & rows["valid_from"].eq(stamp)]
        assert len(old) == 1 and len(new) == 1, (
            f"{ticker}: expected one half-open {old_symbol}/{old_cik} -> {new_symbol}/{new_cik} transition at {boundary}"
        )

    print("\n=== SANITY CHECK: repository IA-3 manual boundaries ===")
    print(f"  {len(transitions)} transitions have one exact old end and one exact new start")
    print("  OK: ticker changes and successor-CIK changes are explicit; no date is guessed")


# --------------------------------------------------------------------------- #
# Real data: the same code against the actual cache                            #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_tenure() -> pd.DataFrame:
    if not CACHE.exists() or len(list(CACHE.glob("*.zip"))) < 4:
        pytest.skip(f"no cached Form 345 quarters under {CACHE}")
    return derive_symbol_tenure(scan_form345_cache(CACHE))


def test_real_cache_reproduces_the_measured_reuse_cases(real_tenure):
    """`IR`, `COR` and `AVGO` are the plan's worked cases; `WM` is the quote-strip case."""
    per_symbol = real_tenure.groupby("symbol")["issuer_cik"].nunique()
    assert per_symbol.loc["IR"] == 4 and per_symbol.loc["COR"] == 4
    assert per_symbol.loc["AVGO"] == 4 and per_symbol.loc["CEG"] == 2

    # the quote artefact: `"WM"` carried Washington Mutual's 2006-2008 filings and must have
    # merged into the unquoted symbol, so WM's first observation is 2006 and not 2008
    assert not real_tenure["symbol"].str.contains('"', regex=False).any()
    wm = real_tenure[real_tenure.symbol == "WM"].sort_values("valid_from")
    assert wm.iloc[0]["valid_from"] == pd.Timestamp("2006-01-12")
    assert wm.iloc[0]["issuer_cik"] == "0000933136"  # Washington Mutual
    assert wm.iloc[1]["issuer_cik"] == "0000823768"  # Waste Management

    print("\n=== SANITY CHECK: real cache, the worked reuse cases ===")
    for symbol in ("IR", "COR", "AVGO", "WM"):
        rows = real_tenure[real_tenure.symbol == symbol].sort_values("valid_from")
        print(f"  {symbol}: {len(rows)} issuer CIK(s)")
        for row in rows.itertuples():
            end = "open" if pd.isna(row.valid_to) else str(row.valid_to.date())
            print(f"     {row.issuer_cik}  {row.valid_from.date()} .. {end:>10}  n={row.n_filings:>5d}  {row.evidence}")
    print("  OK: Every reuse the plan names is present, with its own dated window")
    print("  -> Symbol-first ticker resolution would import all of these as one company.")


def test_real_cache_scale_and_determinism(real_tenure):
    """Scale is the finding, not a detail: symbol reuse is ~9% of symbols, not a long tail."""
    per_symbol = real_tenure.groupby("symbol")["issuer_cik"].nunique()
    multi = int((per_symbol > 1).sum())
    share = multi / len(per_symbol)
    assert len(real_tenure) > 30_000 and len(per_symbol) > 27_000
    assert 0.05 < share < 0.15
    assert real_tenure["valid_to"].isna().sum() > 0  # some tenures are still open
    # deterministic: the same cache must give the same table, or the build is not rebuildable
    assert real_tenure.equals(derive_symbol_tenure(scan_form345_cache(CACHE)))

    print("\n=== SANITY CHECK: real cache scale ===")
    print(f"  rows={len(real_tenure)}  symbols={len(per_symbol)}  multi-CIK symbols={multi} ({share:.1%})")
    print(f"  open tenures={int(real_tenure['valid_to'].isna().sum())}")
    print("  OK: Re-deriving the same cache reproduces the table exactly")
    print("  -> ~1 symbol in 11 has had more than one issuer; reuse is not a long tail.")
