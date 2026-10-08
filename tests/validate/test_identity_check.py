"""Identity validator and the P15 manual-fix flags: foreign rows per filer-CIK table (margin and sibling
filings count as own), lineage invariants, and one flag per kind classified once for the build and the
validator. Offline: hand-written `entity_lineage` rows, a real `DataStore` on SQLite for the store reads.
"""

from __future__ import annotations

import json
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


def test_the_validator_counts_foreign_rows_and_keeps_margin_filings_as_own(sqlite_store):
    """AC-024/AC-034 with P35: AllianceBernstein's filings under ALB and the event-only sibling's 8-K under TMUS are pending."""
    _seed_store(sqlite_store, _store_lineage())
    sqlite_store.save(Tables.symbol_tenure, _evidence([(ALB, "form345", "2006-01-02", None)]).assign(evidence="", evidence_period=""))

    report = check_identity(_context(sqlite_store))

    removals = report.removals
    got = list(zip(removals["table"], removals["ticker"], removals["cik"], removals["rows"], strict=True))
    assert got == [("sec_8k", "ALB", AB, 4), ("sec_8k", "TMUS", TMO_USA, 2)], removals
    assert removals.iloc[0]["first_filed"] == "2019-03-01" and removals.iloc[0]["last_filed"] == "2021-08-01"
    assert report.result.status == "fail"
    foreign = [f for f in report.result.findings if f.field == "foreign_rows"]
    assert {f.ticker for f in foreign} == {"ALB", "TMUS"} and all(f.score >= 7 for f in foreign)
    assert report.result.metrics["foreign_rows"] == 6

    print("\n=== SANITY CHECK: identity validator foreign rows ===")
    print(removals.to_string(index=False))
    print(
        "  OK: the foreign filer's 4 rows and the event-only sibling's 8-K (P35) are pending; REG's margin filing and the null-CIK row are own or not judged"
    )


def test_pending_removals_list_an_event_only_ciks_consolidating_rows(sqlite_store):
    """H1 (AC-004): TMUS's event-only sibling has no window, so its facts and proxy are pending like its 8-K; TMUS's own
    rows and REG's predecessor-window facts (outside its window's dates, no date test on a consolidating table) are kept."""
    _seed_store(sqlite_store, _store_lineage())
    filings = [("TMUS", "tmus-10q", TMUS, "2024-05-01"), ("TMUS", "tmo-usa-10k", TMO_USA, "2018-02-20"), ("REG", "pred-10k", PRED, "2019-02-01")]
    facts = pd.DataFrame(
        [
            {
                "ticker": t,
                "accession_number": a,
                "field": f,
                "duration_type": "quarterly",
                "period_end": pd.Timestamp(d),
                "cik": c,
                "filing_date": pd.Timestamp(d),
            }
            for t, a, c, d in filings
            for f in ("totalRevenue", "totalAssets")
        ]
    )
    sqlite_store.save(Tables.fundamentals_facts, facts)
    sqlite_store.save(
        Tables.def14a_edgar,
        pd.DataFrame([{"ticker": t, "accession_number": a, "form": "DEF 14A", "cik": c, "filing_date": pd.Timestamp(d)} for t, a, c, d in filings]),
    )

    report = check_identity(_context(sqlite_store))

    removals = report.removals[report.removals["ticker"].eq("TMUS")]
    got = sorted(zip(removals["table"], removals["cik"], removals["keys"], removals["rows"], strict=True))
    assert got == [("fundamentals_facts", TMO_USA, 1, 2), ("sec_8k", TMO_USA, 1, 2), (Tables.def14a_edgar.name, TMO_USA, 1, 1)], got
    assert report.removals[report.removals["ticker"].eq("REG")].empty, report.removals
    print("\n=== SANITY CHECK: validator pending removals, H1 ===")
    print(removals.to_string(index=False))
    print("  OK: the event-only CIK's facts and DEF 14A are pending with its 8-K; REG's window-CIK facts are not judged by date")


def test_a_clean_lineage_and_clean_tables_pass(sqlite_store):
    lineage = _store_lineage()
    _seed_store(sqlite_store, lineage)
    sqlite_store.delete(Tables.sec_8k, where={"cik": [AB, TMO_USA]})

    report = check_identity(_context(sqlite_store))

    assert report.removals.empty and report.result.status == "pass", report.result.summary()
    print("\n=== SANITY CHECK: clean identity ===")
    print(f"  {report.result.summary()}")
    print("  OK: no foreign rows and no invariant breach -> pass (AC-033 shape)")


def test_a_filer_whose_only_rows_are_empty_markers_is_not_a_removal(sqlite_store):
    """`distinct` sees an empty-filing marker but `load` hides it: the re-read of such a filer is empty, not an error."""
    lineage = _store_lineage()
    _seed_store(sqlite_store, lineage)
    sqlite_store.delete(Tables.sec_8k, where={"cik": [AB, TMO_USA]})
    marker = {"ticker": "REG", "accession_number": "succ-vote", "proposal_seq": 0.0, "cik": SUCC, "filing_date": pd.Timestamp("2016-05-01")}
    sqlite_store.save(Tables.sec_8k_votes, pd.DataFrame([marker]))

    report = check_identity(_context(sqlite_store))

    assert report.removals.empty and report.result.status == "pass", report.result.summary()
    print("\n=== SANITY CHECK: marker-only filer ===")
    print(f"  {report.result.summary()}")
    print("  OK: REG's lone sec_8k_votes marker under its own CIK is read as no rows, not a TableEmptyError")


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


def test_a_pre_cutover_lineage_is_read_as_membership_rows_and_reported_unmigrated(sqlite_store):
    """F-002: the old-shape table (cik, entity_id, ...) reads as event CIKs plus the roster CIK, as the accessor does."""
    old = pd.DataFrame(
        {
            "cik": [ALB, TMUS, TMO_USA, PRED, SUCC],
            "entity_id": ["E-ALB", "E-TMUS", "E-TMUS", "E-REG", "E-REG"],
            "source": "roster",
            "confidence": None,
            "evidence": "old-shape row",
        }
    )
    _seed_store(sqlite_store, _store_lineage())
    sqlite_store.drop(Tables.entity_lineage)
    old.to_sql(Tables.entity_lineage.name, sqlite_store.engine, index=False)

    report = check_identity(_context(sqlite_store))

    removals = report.removals
    assert list(zip(removals["ticker"], removals["cik"], removals["rows"], strict=True)) == [("ALB", AB, 4)], removals
    fields = [f.field for f in report.result.findings]
    assert "lineage_not_migrated" in fields and "current_symbol_rows" not in fields, fields
    assert report.result.metrics["foreign_rows"] == 4 and report.flags.empty
    print("\n=== SANITY CHECK: validator before the cutover (old-shape entity_lineage) ===")
    print(f"  {report.result.summary()}")
    print("  OK: no KeyError; membership CIKs and roster CIKs are own (no dated windows), AllianceBernstein's 4 rows are pending, invariants skipped")


# --------------------------------------------------------------------------- #
# validator: vendor-series continuity at register cutovers (Q2g, AC-114)        #
# --------------------------------------------------------------------------- #
def _quarters(ticker: str, labels: list[str], assets: float) -> pd.DataFrame:
    ends = [pd.Period(q, freq="Q").end_time.normalize() for q in labels]
    return pd.DataFrame(
        {
            "ticker": ticker,
            "dimension": "ARQ",
            "calendardate": ends,
            "reportperiod": ends,
            "date": [e + pd.Timedelta(days=40) for e in ends],
            "assets": assets,
        }
    )


def _all_quarters(first: str, last: str) -> list[str]:
    return [str(p) for p in pd.period_range(first, last, freq="Q")]


def test_continuity_items_carry_label_and_accession_and_heal_through_the_predecessor_series(sqlite_store, tmp_path):
    """REG's seam 2015-07-01: a recorded gap is INFO, an unrecorded one an action, a record the predecessor's series fills
    is healed, and the canonical rows inside the old CIK's window are flagged as another company's."""
    _seed_store(sqlite_store, _store_lineage())
    canonical = [q for q in _all_quarters("2013Q1", "2017Q4") if q not in {"2014Q2", "2014Q3", "2016Q1"}]
    owner = [q for q in _all_quarters("2013Q1", "2015Q2") if q != "2014Q2"]
    sqlite_store.save(Tables.sharadar_fundamentals, pd.concat([_quarters("REG", canonical, 7.0), _quarters("REG1", owner, 20.0)], ignore_index=True))
    url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={}&type=&dateb=&owner=include&count=40"
    # written raw: SQLite refuses the registry's PK, whose first column is the reserved word `table`
    pd.DataFrame(
        {"ticker": ["REG1", "REG"], "secfilings": [url.format(PRED), url.format(SUCC)], "lastquarter": pd.to_datetime(["2015-06-30", "2017-12-31"])}
    ).to_sql(Tables.sharadar_tickers.name, sqlite_store.engine, index=False)
    sqlite_store.save(
        Tables.fundamentals_facts,
        pd.DataFrame(
            [
                {
                    "ticker": "REG",
                    "accession_number": "succ-q1",
                    "field": "x",
                    "duration_type": "instant",
                    "period_end": pd.Timestamp("2016-03-31"),
                    "cik": SUCC,
                    "form": "10-Q",
                    "filing_date": pd.Timestamp("2016-05-02"),
                    "period_of_report": "2016-03-31",
                },
            ]
        ),
    )
    (tmp_path / "sec").mkdir()
    label = "Sharadar/source coverage gap; SEC filings present"
    rows = [
        {
            "ticker": "REG",
            "quarter": "2014Q2",
            "period_end": "2014-06-30",
            "cik": PRED,
            "accession": "0000000040-14-000002",
            "filed": "2014-08-01",
            "label": label,
        },
        {
            "ticker": "REG",
            "quarter": "2014Q3",
            "period_end": "2014-09-30",
            "cik": PRED,
            "accession": "0000000040-14-000003",
            "filed": "2014-11-01",
            "label": label,
        },
    ]
    (tmp_path / "sec" / "vendor_coverage_exceptions.json").write_text(json.dumps({"exceptions": rows}), encoding="utf-8")
    context = SimpleNamespace(**vars(_context(sqlite_store)), config_dir=tmp_path)

    report = check_identity(context)

    flags = report.flags[
        report.flags["kind"].isin(["vendor_coverage_gap", "incorrect_cik_window", "missing_sec_filing", "vendor_series_other_company"])
    ]
    print("\n=== SANITY CHECK: continuity items in validate identity ===")
    print(flags[["kind", "action", "ticker", "ciks", "evidence"]].to_string(index=False))
    by_quarter = {e.split()[0]: (k, a) for k, a, e in zip(flags["kind"], flags["action"], flags["evidence"], strict=True)}
    assert by_quarter["2014Q2"] == ("vendor_coverage_gap", False)
    assert "0000000040-14-000002" in flags.loc[flags["evidence"].str.startswith("2014Q2"), "evidence"].iloc[0] and label in flags["evidence"].iloc[0]
    assert by_quarter["2016Q1"] == ("vendor_coverage_gap", True)
    assert (
        by_quarter["2014Q3"] == ("vendor_coverage_gap", True)
        and "present now" in flags.loc[flags["evidence"].str.startswith("2014Q3"), "evidence"].iloc[0]
    )
    other = flags[flags["kind"].eq("vendor_series_other_company")]
    assert len(other) == 1 and not other["action"].iloc[0] and "REG1" in other["suggested_action"].iloc[0]
    assert report.result.metrics["predecessor_vendor_tickers"] == {"REG": "REG1"}
    assert report.result.metrics["continuity_healed"] == ["REG 2014Q3"]
    manual = [f for f in report.result.findings if f.field == "manual_decision"]
    assert any("2016Q1" in f.observed for f in manual) and not any("2014Q2" in f.observed for f in manual)
    print("  OK: recorded 2014Q2 INFO with its accession; 2016Q1 an action; 2014Q3 healed by REG1; REG's pre-seam rows are another company's.")


def test_without_the_owner_series_the_window_is_an_action_item(sqlite_store, tmp_path):
    _seed_store(sqlite_store, _store_lineage())
    sqlite_store.save(Tables.sharadar_fundamentals, _quarters("REG", _all_quarters("2013Q1", "2017Q4"), 7.0))
    url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={}"
    pd.DataFrame({"ticker": ["REG1"], "secfilings": [url.format(PRED)], "lastquarter": pd.to_datetime(["2015-06-30"])}).to_sql(
        Tables.sharadar_tickers.name, sqlite_store.engine, index=False
    )
    context = SimpleNamespace(**vars(_context(sqlite_store)), config_dir=tmp_path)

    report = check_identity(context)

    other = report.flags[report.flags["kind"].eq("vendor_series_other_company")]
    print("\n=== SANITY CHECK: predecessor vendor series not stored ===")
    print(other[["ticker", "ciks", "evidence"]].to_string(index=False))
    assert len(other) == 1 and other["action"].iloc[0] and "not stored" in other["evidence"].iloc[0] and "10 canonical" in other["evidence"].iloc[0]
    print("  OK: REG1 missing -> one action naming the 10 unverified canonical quarters inside the window.")


# --------------------------------------------------------------------------- #
# validator: traded-security mismatch (REQ-007, AC-008)                        #
# --------------------------------------------------------------------------- #
_DAYS = pd.bdate_range("2012-01-02", periods=80)
_SPLIT_DAY = _DAYS[60]


def _raw_close(day: pd.Timestamp) -> float:
    """The fixture's raw (as-traded) close: 100 before the 2-for-1 split, 50 from it."""
    return 100.0 + 0.1 * _DAYS.get_loc(day) if day < _SPLIT_DAY else 50.0


def _master_row(ticker: str, cusip: str, role: str, start: str, end: str | None = None, *, symbol: str | None = None) -> dict:
    return {
        "security_id": f"C{cusip}",
        "canonical_company": ticker,
        "issuer_cik": "0000000001",
        "source": "ftd",
        "source_symbol": symbol or ticker,
        "market_symbol": symbol or ticker,
        "cusip": cusip,
        "security_class": "common",
        "lineage_role": role,
        "valid_from": pd.Timestamp(start),
        "valid_to": pd.Timestamp(end) if end else pd.NaT,
    }


def _ftd(cusip: str, ratios: list[float], *, scale: float = 1.0) -> list[dict]:
    """One FTD row per day from the second fixture day, priced `ratio x scale x` the raw close of the prior day."""
    return [
        {"cusip": cusip, "date": day, "trade_date": day, "price": ratio * scale * _raw_close(prev)}
        for prev, day, ratio in zip(_DAYS[:-1], _DAYS[1:], ratios, strict=False)
    ]


def _seed_traded(store: Any) -> None:
    """ALN aligned across a split (its secondary class at 30x is not judged), MIS at 0.446, JCO median 0.987 with 6 % inside, FEW with
    10 days, NOP without Yahoo prices, TWO with two mismatched CUSIPs."""
    tickers = ["ALN", "MIS", "JCO", "FEW", "NOP", "TWO"]
    lineage = pd.DataFrame(
        [row for i, t in enumerate(tickers) for row in (_cik(t, f"00000009{i:02d}", "cik_window"), _sym(t, f"00000009{i:02d}", t, "2006-01-02"))]
    )
    store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": tickers, "cik": [f"00000009{i:02d}" for i in range(len(tickers))]}))
    store.save(Tables.entity_lineage, lineage.assign(valid_from=pd.to_datetime(lineage["valid_from"]), valid_to=pd.to_datetime(lineage["valid_to"])))
    master = [
        _master_row("ALN", "ALN000001", "canonical_current", "2011-01-01"),
        _master_row("ALN", "ALN000002", "secondary_class", "2011-01-01", symbol="ALN.A"),
        _master_row("MIS", "MIS000001", "canonical_predecessor", "2011-01-01"),
        _master_row("JCO", "JCO000001", "canonical_predecessor", "2011-01-01"),
        _master_row("FEW", "FEW000001", "canonical_current", "2011-01-01"),
        _master_row("NOP", "NOP000001", "canonical_current", "2011-01-01"),
        _master_row("TWO", "TWO000001", "canonical_predecessor", "2011-01-01", "2012-02-15"),
        _master_row("TWO", "TWO000002", "canonical_current", "2012-02-15"),
        _master_row("TWO", "TWO000009", "excluded", "2011-01-01"),
    ]
    store.save(Tables.security_master, pd.DataFrame(master))
    n = len(_DAYS) - 1
    jco = [0.90] * 24 + [0.974, 1.0, 1.0] + [1.10] * 23
    ftd = [
        *_ftd("ALN000001", [1.0] * n),
        *_ftd("ALN000002", [1.0] * n, scale=30.0),
        *_ftd("MIS000001", [0.446] * n),
        *_ftd("JCO000001", jco),
        *_ftd("FEW000001", [1.0] * 10),
        *_ftd("NOP000001", [1.0] * n),
        *_ftd("TWO000001", [0.5] * n),
        *_ftd("TWO000002", [0.446] * n),
        *_ftd("TWO000009", [5.0] * n),
    ]
    store.save(Tables.sec_fails_to_deliver_security, pd.DataFrame(ftd))
    prices = [
        {"ticker": t, "date": day, "close_split": scale * _raw_close(day) * (0.5 if day < _SPLIT_DAY else 1.0)}
        for t, scale in (("ALN", 1.0), ("ALN-A", 30.0), ("MIS", 1.0), ("JCO", 1.0), ("FEW", 1.0), ("TWO", 1.0))
        for day in _DAYS
    ]
    store.save(Tables.prices, pd.DataFrame(prices))
    store.save(Tables.prices_splits, pd.DataFrame({"ticker": ["ALN", "ALN-A", "MIS", "JCO", "FEW", "TWO"], "date": _SPLIT_DAY, "ratio": 2.0}))


def test_traded_security_mismatch_flags_one_row_per_ticker_and_counts_unverified_lines(sqlite_store):
    """AC-008 fixture: an aligned line across a split passes and a secondary class is not judged; 0.446, a 0.987 median with 6 % of days
    inside +-3 %, and a two-CUSIP ticker are one warning row each; < 20 days and no Yahoo price are unverified counts."""
    _seed_traded(sqlite_store)

    report = check_identity(_context(sqlite_store))

    flags = report.flags[report.flags["kind"].eq("traded_security_mismatch")]
    print("\n=== SANITY CHECK: traded_security_mismatch ===")
    print(flags[["ticker", "action", "ciks", "evidence"]].to_string(index=False))
    metrics = report.result.metrics
    print(f"  unverified: {metrics['traded_security_unverified']} {metrics['traded_security_unverified_lines']}")
    assert sorted(flags["ticker"]) == ["JCO", "MIS", "TWO"], flags
    assert flags["action"].astype(bool).all() and flags["config_file"].eq("configs/sec/security_master_manual.json").all()
    evidence = dict(zip(flags["ticker"], flags["evidence"], strict=True))
    assert "MIS000001" in evidence["MIS"] and "median ratio 0.446" in evidence["MIS"] and "0% of 79" in evidence["MIS"]
    assert "median ratio 0.987" in evidence["JCO"] and "6% of 50" in evidence["JCO"]
    assert "TWO000002" in evidence["TWO"] and "2 flagged line(s)" in evidence["TWO"] and "TWO000009" not in evidence["TWO"]
    assert metrics["traded_security_unverified"] == 2
    assert [line.split()[:2] for line in metrics["traded_security_unverified_lines"]] == [["FEW", "FEW000001"], ["NOP", "NOP000001"]]
    manual = [f for f in report.result.findings if f.field == "manual_decision" and "traded_security_mismatch" in f.observed]
    assert len(manual) == 3 and all(f.score == 2 for f in manual)
    assert not any(f.score >= 4 for f in report.result.findings), [f.observed for f in report.result.findings if f.score >= 4]
    assert report.result.metrics["traded_security_lines"] == 7
    print("  OK: ALN passes across a split, its secondary class is not judged; MIS, JCO, TWO are one score-2 row each; FEW, NOP are counts.")
