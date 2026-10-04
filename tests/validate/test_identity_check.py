"""Identity validator and the P15 manual-fix flags: foreign rows per filer-CIK table (margin and sibling
filings count as own), lineage invariants, and one flag per kind classified once for the build and the
validator. Offline: hand-written `entity_lineage` rows, a real `DataStore` on SQLite for the store reads.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pandas as pd

from src.data_store.schema import Tables
from src.utils.identity_flags import cik_activity, identity_flag_block, identity_flags, log_identity_flags
from src.validate.checks.identity import check_identity

SENTINEL = "1900-01-01"
LOGGER = "test.identity_check"


def _cik(ticker: str, cik: str, role: str, *, start: str = SENTINEL, end: str | None = None, oracle: str = "roster", sources: str = "roster") -> dict:
    return {
        "entity_id": f"E-{ticker}",
        "canonical_ticker": ticker,
        "cik": cik,
        "role": role,
        "symbol": "",
        "valid_from": start,
        "valid_to": end,
        "status": "curated" if oracle in ("register", "manual") else "single_source",
        "sources": sources,
        "oracle": oracle,
        "confidence": None,
        "n_observations": 1,
        "evidence": "fixture",
        "scope_changed_at": pd.Timestamp("2026-01-01"),
    }


def _sym(
    ticker: str, cik: str, symbol: str, start: str, end: str | None = None, *, status: str = "corroborated", sources: str = "dei,form345"
) -> dict:
    return {**_cik(ticker, cik, "symbol", start=start, end=end), "symbol": symbol, "status": status, "sources": sources, "oracle": "fixture"}


def _evidence(rows: list[tuple[str, str, str, str | None]]) -> pd.DataFrame:
    """(cik, source, first, last or None for an open form345 tenure) -> symbol_tenure-shaped evidence."""
    return pd.DataFrame(
        [
            {"symbol": "X", "issuer_cik": c, "valid_from": pd.Timestamp(f), "valid_to": pd.Timestamp(t) if t else pd.NaT, "n_filings": 3, "source": s}
            for c, s, f, t in rows
        ]
    )


def _flag_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One entity per flag kind, plus a GOOGL-style register seam that only lags (not flagged)."""
    lineage = pd.DataFrame(
        [
            # SEQ: the old CIK stops filing in 2012, the roster CIK starts in 2014 -> missing_cutover
            _cik("SEQ", "0000000011", "cik_window"),
            _cik("SEQ", "0000000010", "cik_event", oracle="owner_overlap", sources="form345"),
            _sym("SEQ", "0000000011", "SEQ", "2014-01-02"),
            _sym("SEQ", "0000000011", "SQQ", "2015-01-02", "2015-01-03", status="noise", sources="form345"),
            # CON: a subsidiary co-registrant filing beside the roster CIK the whole time -> co_registrant (INFO)
            _cik("CON", "0000000021", "cik_window"),
            _cik("CON", "0000000020", "cik_event", oracle="owner_overlap", sources="form345"),
            _sym("CON", "0000000021", "CON", "2006-01-02"),
            _sym("CON", "0000000021", "CNX", "2010-01-02", "2011-01-02", status="conflict", sources="form345"),
            # MRG: the roster CIK traded as OLD, then took the MRG symbol from the other concurrent filer -> manual_review
            _cik("MRG", "0000000031", "cik_window"),
            _cik("MRG", "0000000030", "cik_event", oracle="owner_overlap", sources="form345"),
            _sym("MRG", "0000000030", "MRG", "2006-01-02", "2011-01-21", status="single_source", sources="form345"),
            _sym("MRG", "0000000031", "OLD", "2006-01-02", "2009-11-05", status="single_source", sources="form345"),
            _sym("MRG", "0000000031", "MRG", "2009-11-05"),
            # REG: the predecessor still files alone five months after the register date -> register_date_disputed
            _cik("REG", "0000000040", "cik_window", end="2015-07-01", oracle="register", sources="register"),
            _cik("REG", "0000000041", "cik_window", start="2015-07-01", oracle="register", sources="register"),
            _sym("REG", "0000000041", "REG", "2016-04-01"),
            # LAG: symbol observations lag the register date by under a month (GOOGL) -> not flagged
            _cik("LAG", "0000000050", "cik_window", end="2015-10-02", oracle="register", sources="register"),
            _cik("LAG", "0000000051", "cik_window", start="2015-10-02", oracle="register", sources="register"),
            _sym("LAG", "0000000051", "LAG", "2015-10-29"),
            # FIT: a preferred line trading beside the common -> manual_review (summed on the tapes)
            _cik("FIT", "0000000060", "cik_window"),
            _sym("FIT", "0000000060", "FIT", "2006-01-02"),
            _sym("FIT", "0000000060", "FITP", "2013-02-14"),
            _sym("FIT", "0000000060", "FIT28", "2019-01-02", sources="dei"),
            _sym("FIT", "0000000060", "FTI", "2019-05-03", "2019-05-04", status="single_source", sources="form345"),
            # GGL: the redundant class GOG resolves to GGL while GGL trades only on cover pages -> manual_review
            _cik("GGL", "0000000070", "cik_window", end="2015-10-02", oracle="register", sources="register"),
            _cik("GGL", "0000000071", "cik_window", start="2015-10-02", oracle="register", sources="register"),
            _sym("GGL", "0000000070", "GOG", "2006-01-04", "2015-10-30"),
            _sym("GGL", "0000000070", "GGL", "2014-07-25", "2015-10-30", status="single_source", sources="dei"),
            _sym("GGL", "0000000071", "GOG", "2015-10-08"),
            _sym("GGL", "0000000071", "GGL", "2015-10-29"),
        ]
    )
    evidence = _evidence(
        [
            ("0000000010", "form345", "2006-01-03", "2012-06-30"),
            ("0000000010", "dei", "2009-08-01", "2012-05-10"),
            ("0000000011", "form345", "2014-01-02", None),
            ("0000000011", "dei", "2014-05-01", "2026-08-01"),
            ("0000000020", "form345", "2006-01-05", "2025-10-11"),
            ("0000000021", "form345", "2006-01-05", None),
            ("0000000030", "form345", "2006-01-04", "2011-01-21"),
            ("0000000031", "form345", "2006-01-04", None),
            ("0000000040", "dei", "2010-01-01", "2015-12-01"),
            ("0000000041", "dei", "2016-04-01", "2026-08-01"),
            ("0000000050", "dei", "2009-08-04", "2015-10-30"),
            ("0000000051", "form345", "2015-10-08", None),
            ("0000000051", "dei", "2015-10-29", "2026-08-01"),
            ("0000000040", "form345", "2006-01-04", "2015-08-31"),  # Form 3/4/5 lag never disputes a seam
            ("0000000060", "form345", "2006-01-06", None),
            ("0000000070", "form345", "2006-01-04", "2015-10-06"),
            ("0000000071", "form345", "2015-10-08", None),
        ]
    )
    backlog = pd.DataFrame(
        [
            {
                "kind": "grey_band",
                "canonical_ticker": "CPT",
                "entity_id": "E-CPT",
                "cik": "0000096345",
                "symbol": "CPT",
                "detail": "owner overlap with 0000906345",
            }
        ]
    )
    return lineage, evidence, backlog


def test_every_flag_kind_is_classified_once_from_the_lineage_and_the_filing_evidence():
    lineage, evidence, backlog = _flag_fixture()
    flags = identity_flags(lineage, cik_activity(evidence), redundant_symbols=frozenset({"GOG"}), backlog=backlog)
    by_ticker = {(kind, ticker) for kind, ticker in zip(flags["kind"], flags["ticker"], strict=True)}

    assert ("missing_cutover", "SEQ") in by_ticker
    assert ("co_registrant", "CON") in by_ticker and ("missing_cutover", "CON") not in by_ticker
    assert ("manual_review", "MRG") in by_ticker and ("co_registrant", "MRG") not in by_ticker
    assert ("register_date_disputed", "REG") in by_ticker
    assert not flags["ticker"].astype(str).str.contains("LAG").any(), flags
    assert ("grey_band", "CPT") in by_ticker
    assert ("manual_review", "FIT") in by_ticker
    assert ("manual_review", "GGL") in by_ticker
    conflict = flags[flags["kind"].eq("conflict")].iloc[0]
    noise = flags[flags["kind"].eq("noise")].iloc[0]
    assert "CNX" in conflict["evidence"] and conflict["ticker"] == "CON"
    assert "SQQ" in noise["evidence"] and noise["ticker"] == "SEQ"
    fit = flags[flags["kind"].eq("manual_review") & flags["ticker"].eq("FIT")]
    assert fit["evidence"].str.contains("FITP").all() and not fit["evidence"].str.contains("FIT28|FTI").any()
    ggl = flags[flags["kind"].eq("manual_review") & flags["ticker"].eq("GGL")].iloc[0]
    assert "2014-07-25" in ggl["evidence"] and "2015-10-29" in ggl["evidence"], ggl["evidence"]
    seq = flags[flags["kind"].eq("missing_cutover")].iloc[0]
    assert "0000000010" in seq["evidence"] and "2012-05-10" in seq["evidence"] and "registrant_cutover.json" in seq["config_file"]
    assert flags.loc[flags["kind"].eq("co_registrant"), "action"].eq(False).all()

    print("\n=== SANITY CHECK: P15 flag kinds ===")
    print(flags[["kind", "action", "ticker", "ciks"]].to_string(index=False))
    print("  OK: sequential pair flagged, concurrent pair INFO, takeover reviewed, disputed seam flagged, GOOGL-style lag not flagged")


def test_the_warning_block_lists_action_items_first_and_is_skipped_when_empty(caplog):
    lineage, evidence, backlog = _flag_fixture()
    flags = identity_flags(lineage, cik_activity(evidence), redundant_symbols=frozenset({"GOG"}), backlog=backlog)
    block = identity_flag_block(flags)
    assert block is not None and block.startswith("IDENTITY ITEMS NEEDING A MANUAL DECISION")
    lines = block.splitlines()[1:]
    actions = [i for i, line in enumerate(lines) if "[ACTION]" in line]
    others = [i for i, line in enumerate(lines) if "[info]" in line]
    assert actions and others and max(actions) < min(others), block
    assert not any("co_registrant" in line for line in lines), "co-registrants are an INFO line, not a block entry"

    log = logging.getLogger(LOGGER)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        log_identity_flags(log, flags)
        log_identity_flags(log, flags.iloc[0:0])
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert len(warnings) == 1 and any("co-registrant" in m and "CON" in m for m in infos)
    assert identity_flag_block(flags.iloc[0:0]) is None

    print("\n=== SANITY CHECK: flag block ===")
    print(block)
    print("  OK: one WARNING block with action items first; co-registrants one INFO line; nothing logged when empty")


# --------------------------------------------------------------------------- #
# validator: foreign rows and lineage invariants through the store             #
# --------------------------------------------------------------------------- #
ALB, AB, TMUS, TMO_USA, PRED, SUCC = "0000915913", "0000825313", "0001283699", "0001097609", "0000000040", "0000000041"


def _store_lineage() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _cik("ALB", ALB, "cik_window"),
            _sym("ALB", ALB, "ALB", "2006-01-02"),
            _cik("TMUS", TMUS, "cik_window"),
            _cik("TMUS", TMO_USA, "cik_event", oracle="manual", sources="manual"),
            _sym("TMUS", TMUS, "TMUS", "2013-05-01"),
            _cik("REG", PRED, "cik_window", end="2015-07-01", oracle="register", sources="register"),
            _cik("REG", SUCC, "cik_window", start="2015-07-01", oracle="register", sources="register"),
            _sym("REG", SUCC, "REG", "2015-07-01"),
        ]
    )


def _context(store: Any) -> Any:
    return SimpleNamespace(store=store, log=logging.getLogger(LOGGER), config=SimpleNamespace(data_extract=SimpleNamespace(redundant_ticks=[])))


def _seed_store(store: Any, lineage: pd.DataFrame) -> None:
    store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": ["ALB", "TMUS", "REG"], "cik": [ALB, TMUS, SUCC]}))
    store.save(Tables.entity_lineage, lineage.assign(valid_from=pd.to_datetime(lineage["valid_from"]), valid_to=pd.to_datetime(lineage["valid_to"])))
    filings = [
        ("ALB", "alb-1", ALB, "2024-02-01"),
        ("ALB", "ab-1", AB, "2019-03-01"),
        ("ALB", "ab-2", AB, "2021-08-01"),
        ("TMUS", "tmus-1", TMUS, "2024-02-01"),
        ("TMUS", "tmo-usa-1", TMO_USA, "2018-02-01"),
        ("REG", "pred-margin", PRED, "2015-07-20"),
        ("REG", "succ-1", SUCC, "2016-02-01"),
        ("REG", "no-cik", None, "2016-03-01"),
    ]
    store.save(
        Tables.sec_8k,
        pd.DataFrame(
            [
                {"ticker": t, "accession_number": a, "item": item, "cik": c, "filing_date": pd.Timestamp(f)}
                for t, a, c, f in filings
                for item in ("2.02", "9.01")
            ]
        ),
    )


def test_the_validator_counts_foreign_rows_and_keeps_margin_and_sibling_filings_as_own(sqlite_store):
    """AC-024/AC-034: only AllianceBernstein's filings under ALB are pending removals."""
    _seed_store(sqlite_store, _store_lineage())
    sqlite_store.save(Tables.symbol_tenure, _evidence([(ALB, "form345", "2006-01-02", None)]).assign(evidence="", evidence_period=""))

    report = check_identity(_context(sqlite_store))

    removals = report.removals
    assert list(zip(removals["table"], removals["ticker"], removals["cik"], removals["rows"], strict=True)) == [("sec_8k", "ALB", AB, 4)], removals
    assert removals.iloc[0]["first_filed"] == "2019-03-01" and removals.iloc[0]["last_filed"] == "2021-08-01"
    assert report.result.status == "fail"
    foreign = [f for f in report.result.findings if f.field == "foreign_rows"]
    assert len(foreign) == 1 and foreign[0].ticker == "ALB" and foreign[0].score >= 7
    assert report.result.metrics["foreign_rows"] == 4

    print("\n=== SANITY CHECK: identity validator foreign rows ===")
    print(removals.to_string(index=False))
    print("  OK: the foreign filer's 4 rows are pending; TMUS's sibling, REG's margin filing and the null-CIK row are own or not judged")


def test_a_clean_lineage_and_clean_tables_pass(sqlite_store):
    lineage = _store_lineage()
    _seed_store(sqlite_store, lineage)
    sqlite_store.delete(Tables.sec_8k, where={"cik": [AB]})

    report = check_identity(_context(sqlite_store))

    assert report.removals.empty and report.result.status == "pass", report.result.summary()
    print("\n=== SANITY CHECK: clean identity ===")
    print(f"  {report.result.summary()}")
    print("  OK: no foreign rows and no invariant breach -> pass (AC-033 shape)")


def test_lineage_invariant_breaches_are_findings(sqlite_store):
    """AC-012, sentinel open start, one entity per CIK, no window overlap inside an entity beyond the margin."""
    lineage = pd.DataFrame(
        [
            _cik("ALB", ALB, "cik_window"),  # ALB: no current symbol row
            _cik("TMUS", TMUS, "cik_window"),
            _sym("TMUS", TMUS, "TMUS", "2013-05-01"),
            _sym("TMUS", TMUS, "TMUS", "2014-05-01"),  # two current rows
            {**_cik("REG", PRED, "cik_window", end="2016-07-01", oracle="register", sources="register")},
            _cik("REG", SUCC, "cik_window", start="2015-07-01", oracle="register", sources="register"),  # overlap 366 days
            _sym("REG", SUCC, "REG", "2015-07-01"),
            {**_cik("ALB", AB, "cik_window", oracle="owner_overlap"), "entity_id": "E-OTHER", "canonical_ticker": None},
            _cik("TMUS", AB, "cik_event", oracle="owner_overlap"),  # AB in two entities
            _cik("TMUS", "0000000099", "cik_event", start="1899-12-31", oracle="owner_overlap"),  # an open start off the sentinel
        ]
    )
    _seed_store(sqlite_store, lineage)

    report = check_identity(_context(sqlite_store))

    fields = {f.field: f for f in report.result.findings}
    assert {"current_symbol_rows", "sentinel_start", "one_entity_per_cik", "window_overlap"} <= set(fields), sorted(fields)
    assert fields["current_symbol_rows"].evidence["tickers"] == {"ALB": 0, "TMUS": 2}
    assert fields["one_entity_per_cik"].evidence["ciks"] == [AB]
    assert report.result.status == "fail"
    print("\n=== SANITY CHECK: lineage invariants ===")
    for finding in report.result.findings:
        if finding.field != "foreign_rows":
            print(f"  [{finding.severity}] {finding.field}: {finding.observed}")
    print("  OK: each breached invariant is one finding")
