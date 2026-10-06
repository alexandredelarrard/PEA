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


#: Known truth for every non-alphanumeric alias filed under a roster CIK in the live table
#: (`_out/p1_junk_pairs.txt`), plus the named placeholder and share-class cases.
#: None = placeholder ("no symbol"), () = noise, otherwise the symbols in roster spelling.
_SYMBOL_FIELDS: list[tuple[str, tuple[str, ...] | None]] = [
    ("(BBT)", ("BBT",)),
    ("[FB]", ("FB",)),
    ("NYSE: GLW", ("GLW",)),
    ("NYSE: BAX", ("BAX",)),
    ("NYSE:IRM", ("IRM",)),
    ("OTC: VICI", ("VICI",)),
    ("BAX (NYSE)", ("BAX",)),
    ("BFA/BFB", ("BFA", "BFB")),
    ("BFA, BFB", ("BFA", "BFB")),
    ("BFA,BFB", ("BFA", "BFB")),
    ("BF'B", ("BFB",)),
    ("LEN, LEN.B", ("LEN", "LEN-B")),
    ("LEN,LEN.B", ("LEN", "LEN-B")),
    ("CMG/CMG.B", ("CMG", "CMG-B")),
    ("CMG.B", ("CMG-B",)),
    ("STZ/STZ.B", ("STZ", "STZ-B")),
    ("HUBA, HUBB", ("HUBA", "HUBB")),
    ("HUBA,HUBB", ("HUBA", "HUBB")),
    ("HUB-A", ("HUB-A",)),
    ("TAP.A TAP", ()),
    ("TAP.A, TAP", ("TAP-A", "TAP")),
    ("FCE A/FCE", ("FCE-A", "FCE")),
    ("KVA / KVB", ("KVA", "KVB")),
    ("FNF, FIS", ("FNF", "FIS")),
    ("L; LMC.B", ("L", "LMC-B")),
    ("ALF A", ("ALF-A",)),
    ("XSIAX (A)", ("XSIAX-A",)),
    ("HBC PR A", ("HBC-PR-A",)),
    ("N O G", ("NOG",)),
    ("V S E C", ("VSEC",)),
    ("BWINA / B", ("BWINA-B",)),
    ("NWIN(OB)", ("NWIN",)),
    ("CPTC.OB", ("CPTC",)),
    ("DE/TGAL", ("TGAL",)),
    ("OWL ROCK T", ()),
    ("APP. FOR", ()),
    ("Z AND ZG", ()),
    ("TAP.A; TAP", ("TAP-A", "TAP")),
    ("LTR:CG", ("LTR", "CG")),
    ("LTR; CG", ("LTR", "CG")),
    ("LTR;CG", ("LTR", "CG")),
    ("BRK.B", ("BRK-B",)),
    ("BRK/B", ("BRK-B",)),
    ("BRK-A", ("BRK-A",)),
    ("BRK.A", ("BRK-A",)),
    ("BRK/A", ("BRK-A",)),
    ("BRKB", ("BRKB",)),
    ("KIM-PG", ("KIM-PG",)),
    ('"WM"', ("WM",)),
    ("(AEP)", ("AEP",)),
    ("(AIG)", ("AIG",)),
    ("(BMY)", ("BMY",)),
    ("(BSX)", ("BSX",)),
    ("(CPT", ("CPT",)),
    ("(CPT)", ("CPT",)),
    ("[D]", ("D",)),
    ("(HCA)", ("HCA",)),
    ("[HON]", ("HON",)),
    ("[IBKR]", ("IBKR",)),
    ("[IRM", ("IRM",)),
    ("IRM]", ("IRM",)),
    ("CMCSA]", ("CMCSA",)),
    ("NCLH]", ("NCLH",)),
    ("(LYB)", ("LYB",)),
    ("(MRK)", ("MRK",)),
    ("[OMC]", ("OMC",)),
    ("(PRU)", ("PRU",)),
    ("(RCL)", ("RCL",)),
    ("(REGN)", ("REGN",)),
    ("(RTX", ("RTX",)),
    ("[TER]", ("TER",)),
    ("(TSN)", ("TSN",)),
    ("[TYL]", ("TYL",)),
    ("(WST)", ("WST",)),
    ("(KO)", ("KO",)),
    ("RC:", ("RC",)),
    ("RCL:", ("RCL",)),
    ("KEY--", ("KEY",)),
    ("IDXX`", ("IDXX",)),
    ("DALRQ.PK", ("DALRQ",)),
    ("/DE/CHD", ("CHD",)),
    ("CARR WI", ("CARR",)),
    ("OTIS WI", ("OTIS",)),
    ("CIEN US", ("CIEN",)),
    ("HCA INC.", ("HCA",)),
    ("3M CO", ("3M",)),
    ("(NONE)", None),
    ("[NONE]", None),
    ("[ N/A ]", None),
    ("NONE", None),
    ("NO SYMBOL", None),
    ("N/A", None),
    ("N.A.", None),
    ("-", None),
    ("---", None),
    ("", None),
    ("4", ()),
    ("DEERE & CO", ()),
    ("4$EJNNIU", ()),
    ("@ABC3DEF", ()),
    ("DC18*POL", ()),
    ("DC6*POLK", ()),
    ("EG8R*NSC", ()),
    ("FF9EYD*", ()),
    ("FO9JOD#Z", ()),
    ("I#EK7JYE", ()),
    ("IGEWXR6*", ()),
    ("JPMC*011", ()),
    ("OC6N*NMF", ()),
    ("SPUD*CO2", ()),
    ("T@NET5WKS", ()),
]


def test_symbol_fields_normalise_to_roster_spelling():
    """Every junk spelling filed under a roster CIK maps to its known symbols, a placeholder or noise."""
    parse = tenure_module.parse_symbol_field
    wrong = [(raw, expected, parse(raw)) for raw, expected in _SYMBOL_FIELDS if parse(raw) != expected]
    assert not wrong, wrong
    assert parse(None) is None and parse(pd.NA) is None

    n_lists = sum(1 for _, expected in _SYMBOL_FIELDS if expected and len(expected) > 1)
    n_placeholders = sum(1 for _, expected in _SYMBOL_FIELDS if expected is None)
    n_noise = sum(1 for _, expected in _SYMBOL_FIELDS if expected == ())
    print("\n=== SANITY CHECK: symbol-field normalisation ===")
    print(f"  {len(_SYMBOL_FIELDS)} known-truth fields: {n_lists} multi-symbol lists, {n_placeholders} placeholders, {n_noise} noise")
    print("  '(BBT)'->BBT  '[FB]'->FB  'NYSE: GLW'->GLW  'BFA/BFB'->BFA,BFB  'BRK.B'/'BRK/B'->BRK-B  'BRKB' stays")
    print("  'ALF A'->ALF-A  'N O G'->NOG  'OWL ROCK T'->noise: whitespace never splits a list, so no stray single-letter ticker appears")
    print("  OK: lists split before the share-class rule; a slash before one letter is a class, not a list")


def test_junk_symbol_fields_derive_clean_tenures(tmp_path):
    """Junk spellings of one issuer's symbol collapse into its roster-spelled tenures; noise is counted, never stored."""
    _write_zip(
        tmp_path,
        "2019q4",
        [
            ("t1", "92230", "BB&T CORP", "(BBT)", "05-NOV-2019"),
            ("t2", "92230", "BB&T CORP", "BBT", "06-NOV-2019"),
            ("m1", "1326801", "FACEBOOK INC", "[FB]", "07-NOV-2019"),
            ("g1", "24741", "CORNING INC", "NYSE: GLW", "08-NOV-2019"),
            ("b1", "14693", "BROWN FORMAN CORP", "BFA/BFB", "09-NOV-2019"),
            ("k1", "1067983", "BERKSHIRE HATHAWAY INC", "BRK.B", "10-NOV-2019"),
            ("k2", "1067983", "BERKSHIRE HATHAWAY INC", "BRK/B", "11-NOV-2019"),
            ("n1", "200406", "JOHNSON & JOHNSON", "(NONE)", "12-NOV-2019"),
            ("x1", "1000", "SOME FILER", "DC18*POL", "13-NOV-2019"),
        ],
    )
    scan = scan_form345_cache(tmp_path)
    out = derive_symbol_tenure(scan)
    by_symbol = out.set_index("symbol")

    assert set(out["symbol"]) == {"BBT", "FB", "GLW", "BFA", "BFB", "BRK-B"}
    assert by_symbol.loc["BBT", "n_filings"] == 2 and by_symbol.loc["BRK-B", "n_filings"] == 2
    assert by_symbol.loc["BFA", "issuer_cik"] == by_symbol.loc["BFB", "issuer_cik"] == "0000014693"
    assert scan.drops["empty_symbol"] == 1 and scan.drops["noise_symbol"] == 1
    assert scan.drops["rows_read"] == 9 and scan.drops["rows_kept"] == 7

    print("\n=== SANITY CHECK: junk symbol fields through the derivation ===")
    print(f"  9 filings -> {len(out)} tenures {sorted(out['symbol'])}")
    print(f"  placeholder dropped={scan.drops['empty_symbol']} noise dropped={scan.drops['noise_symbol']}")
    print("  OK: '(BBT)' merges into BBT, BRK.B and BRK/B into BRK-B, 'BFA/BFB' yields two symbols on one CIK")


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
            "evidence_period": [""] * 2,
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
    assert (out["evidence_period"] == "").all()

    print("\n=== SANITY CHECK: manual-over-derived auditability ===")
    print("  manual=1 derived=1 materialized=2; manual is ordered first but neither row is hidden")
    print("  OK: precedence is a resolver concern, not destructive evidence replacement")


def test_materialization_keeps_both_sources_of_a_shared_interval():
    """A manual and a derived row on one (symbol, issuer_cik, valid_from) are two evidence rows, not one."""
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

    assert list(out["source"]) == ["manual", "form345"]
    assert list(out["n_filings"]) == [0, 900]
    assert list(out["evidence"]) == ["SEC listing evidence", "AGILENT TECHNOLOGIES INC"]
    primary_key = list(Tables.symbol_tenure.pk)
    assert primary_key == ["symbol", "issuer_cik", "valid_from", "source", "evidence_period"]
    assert not out.duplicated(primary_key).any()
    print("\n=== SANITY CHECK: cross-source interval kept as evidence ===")
    print(f"  manual + form345 on A/0001090872/2006-02-16 -> {len(out)} rows, sources {list(out['source'])}")
    print("  OK: source is in the primary key, so no evidence string or filing count is merged away")


def _seed_tenure_row(symbol: str, cik: str, source: str, evidence_period: str) -> dict[str, object]:
    """One stored `symbol_tenure` row for the partition tests."""
    return {
        "symbol": symbol,
        "issuer_cik": cik,
        "valid_from": pd.Timestamp("2019-01-01"),
        "valid_to": pd.NaT,
        "n_filings": 3,
        "source": source,
        "evidence_period": evidence_period,
        "evidence": f"{source} seed",
    }


def test_build_rewrites_only_its_partitions_and_keeps_dei_rows(sqlite_store, monkeypatch, tmp_path):
    """The form345/manual build must leave rows of other sources (`dei`) in place and drop its own stale rows."""
    seeded = pd.DataFrame(
        [
            _seed_tenure_row("DEI", "0000000009", "dei", "2024q1"),
            _seed_tenure_row("STALE", "0000000008", "form345", ""),
        ]
    )
    sqlite_store.save(Tables.symbol_tenure, seeded)
    derived = pd.DataFrame(
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
    monkeypatch.setattr(tenure_module, "derive_symbol_tenure", lambda scan: derived)
    manual_columns = ["canonical_ticker", "symbol", "issuer_cik", "valid_from", "valid_to", "n_filings", "source", "evidence", "reason"]
    monkeypatch.setattr(tenure_module, "load_manual_symbol_tenure", lambda config_dir: pd.DataFrame(columns=manual_columns))
    writes: list[str] = []
    for method in ("save", "replace", "delete"):
        real = getattr(sqlite_store, method)
        monkeypatch.setattr(sqlite_store, method, lambda *args, _real=real, _name=method, **kwargs: (writes.append(_name), _real(*args, **kwargs))[1])
    context: Any = SimpleNamespace(store=sqlite_store, log=logging.getLogger("test.symbol_tenure.partition"))

    tenure_module.build_symbol_tenure(context, cast(Any, None), tmp_path)
    stored = sqlite_store.load(Tables.symbol_tenure, project=True)
    assert stored is not None
    rows = {(row.symbol, row.source, row.evidence_period) for row in stored.itertuples(index=False)}
    assert rows == {("DEI", "dei", "2024q1"), ("AAA", "form345", "")}
    assert "replace" not in writes

    first_writes = list(writes)
    tenure_module.build_symbol_tenure(context, cast(Any, None), tmp_path)
    assert writes == first_writes, "an unchanged form345/manual partition must not be rewritten"

    print("\n=== SANITY CHECK: partition-scoped symbol_tenure build ===")
    print(f"  seeded dei + stale form345 -> stored {sorted(rows)}; writes {first_writes}; unchanged rebuild added none")
    print("  OK: the build owns only form345/manual; dei evidence survives and the stale derived row is gone")


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


def test_symbol_tenure_write_is_skipped_when_unchanged(sqlite_store, monkeypatch, caplog, tmp_path):
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
    saved: list[int] = []
    real_save = sqlite_store.save

    def counting_save(table, df):
        saved.append(len(df))
        return real_save(table, df)

    monkeypatch.setattr(sqlite_store, "save", counting_save)
    context: Any = SimpleNamespace(store=sqlite_store, log=logging.getLogger("test.symbol_tenure.skip"))
    caplog.set_level(logging.INFO)

    tenure_module.build_symbol_tenure(context, cast(Any, None), tmp_path)
    tenure_module.build_symbol_tenure(context, cast(Any, None), tmp_path)
    assert saved == [2]
    assert "symbol_tenure: unchanged (2 row(s)); write skipped" in caplog.text

    current["frame"] = frame.assign(n_filings=[5, 1])
    tenure_module.build_symbol_tenure(context, cast(Any, None), tmp_path)
    assert saved == [2, 2]

    print("\n=== SANITY CHECK: unchanged symbol_tenure is not rewritten ===")
    print(f"  cold build wrote once; identical rebuild skipped the write; a changed n_filings wrote again -> save calls {saved}")
    print("  OK: the DATE round-trip and row order do not defeat the comparison")


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
    assert len(real_tenure) > 27_000 and len(per_symbol) > 24_000
    assert 0.05 < share < 0.15
    assert real_tenure["valid_to"].isna().sum() > 0  # some tenures are still open
    # deterministic: the same cache must give the same table, or the build is not rebuildable
    assert real_tenure.equals(derive_symbol_tenure(scan_form345_cache(CACHE)))

    print("\n=== SANITY CHECK: real cache scale ===")
    print(f"  rows={len(real_tenure)}  symbols={len(per_symbol)}  multi-CIK symbols={multi} ({share:.1%})")
    print(f"  open tenures={int(real_tenure['valid_to'].isna().sum())}")
    print("  OK: Re-deriving the same cache reproduces the table exactly")
    print("  -> ~1 symbol in 11 has had more than one issuer; reuse is not a long tail.")
