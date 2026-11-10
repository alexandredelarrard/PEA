"""`identity --only sync`, the purge half: a lineage contraction purges rows whose filer CIK left the ticker's entity,
once per table logged; the nightly SEC history build then rebuilds a purged ticker (M7). Offline: a real `DataStore` on
SQLite and a real `Identity` over hand-written lineage rows.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common.identity import Identity, build_identity
from src.data_extract.utils.fundamentals import build_history
from src.data_extract.utils.fundamentals.kpi_catalogue import load_catalogue
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils import filer_tables
from tests.data_extract.fake_context import extract_config
from tests.support.identity_sync import run_sync
from tests.support.universe import as_roster, seed_universe, universe_of

SENTINEL = "1900-01-01"
#: The run date: AFTER falls inside the 7-day re-check window before it, BEFORE does not.
RUN_DATE = pd.Timestamp("2026-01-13")
BEFORE, AFTER = pd.Timestamp("2026-01-05 09:00"), pd.Timestamp("2026-01-12 09:00")
ALB, AB, MSFT, MSFT_FOREIGN = "0000915913", "0000825313", "0000789019", "0000000666"
TMUS, TMO_USA, SPRINT = "0001283699", "0001097609", "0001727074"
ROSTER = {"ALB": ALB, "MSFT": MSFT, "TMUS": TMUS}
LOGGER = "test.identity_sync"


def _row(ticker: str, cik: str, role: str, *, symbol: str = "", start: str = SENTINEL, end: str | None = None, stamp=BEFORE, **extra) -> dict:
    return {
        "company_id": f"E{ROSTER[ticker]}",
        "canonical_ticker": ticker,
        "cik": cik,
        "role": role,
        "symbol": symbol,
        "valid_from": start,
        "valid_to": end,
        "status": extra.get("status", "curated"),
        "sources": extra.get("sources", "form345"),
        "oracle": "fixture",
        "confidence": None,
        "n_observations": 1,
        "evidence": "fixture",
        "scope_changed_at": stamp,
    }


def _identity(stamps: dict[str, pd.Timestamp], symbol_rows: list[dict] = ()) -> Any:
    """ALB and MSFT each one open window; TMUS a window plus its T-Mobile USA co-registrant (event only)."""
    rows = [
        _row("ALB", ALB, "cik_window", stamp=stamps.get("ALB", BEFORE)),
        _row("MSFT", MSFT, "cik_window", stamp=stamps.get("MSFT", BEFORE)),
        _row("TMUS", TMUS, "cik_window", stamp=stamps.get("TMUS", BEFORE)),
        _row("TMUS", TMO_USA, "cik_event", stamp=stamps.get("TMUS", BEFORE)),
        *symbol_rows,
    ]
    tenure = pd.DataFrame(
        [
            {
                "symbol": t,
                "issuer_cik": c,
                "valid_from": pd.Timestamp("2000-01-01"),
                "valid_to": None,
                "n_filings": 5,
                "source": "form345",
                "evidence_period": "2000q1",
            }
            for t, c in ROSTER.items()
        ]
    )
    return build_identity(pd.DataFrame(rows), tenure, as_roster(pd.DataFrame([{"ticker": t, "cik": c} for t, c in ROSTER.items()])))


def _stored_identity(store: DataStore, stamps: dict[str, pd.Timestamp]) -> Identity:
    identity = _identity(stamps)
    for table, frame in identity.source_cohort.items():
        if not frame.empty:
            store.save(table, frame)
    return identity


def _context(store, tmp_path) -> SimpleNamespace:
    return SimpleNamespace(
        store=store,
        paths={"DATA_STORE": tmp_path},
        log=logging.getLogger(LOGGER),
        config=extract_config(data_extract={"years_history": 15, "manifest_full_rescan_days": 30}),
        config_dir="./configs",
        universe=seed_universe(store, list(ROSTER), rows={t: {"cik": c} for t, c in ROSTER.items()}),
    )


def _filings(table, ticker: str, rows: list[tuple[str, str, str]], **extra_cols) -> pd.DataFrame:
    """(accession, filer cik, filed) -> two rows per filing in `table`'s shape."""
    if table == Tables.sec_8k:
        out = [
            {"ticker": ticker, "accession_number": a, "item": item, "cik": c, "filing_date": pd.Timestamp(f)}
            for a, c, f in rows
            for item in ("2.02", "9.01")
        ]
    elif table == Tables.sec_fundamentals_facts:
        out = [
            {
                "ticker": ticker,
                "accession_number": a,
                "field": field,
                "duration_type": "quarterly",
                "period_end": pd.Timestamp(f),
                "cik": c,
                "filing_date": pd.Timestamp(f),
                "value": 1.0,
            }
            for a, c, f in rows
            for field in ("totalRevenue", "totalAssets")
        ]
    elif table == Tables.notes_num:
        out = [
            {"ticker": ticker, "adsh": a, "tag": tag, "ddate": pd.Timestamp(f), "qtrs": 0, "cik": c, "filed": pd.Timestamp(f)}
            for a, c, f in rows
            for tag in ("T1", "T2")
        ]
    elif table == Tables.def14a_edgar:
        out = [{"ticker": ticker, "accession_number": a, "form": "DEF 14A", "cik": c, "filing_date": pd.Timestamp(f)} for a, c, f in rows]
    elif table == Tables.insider_transactions:
        out = [
            {
                "ticker": ticker,
                "accession_number": a,
                "security_type": "nonderiv",
                "row_sequence": seq,
                "issuer_cik": c,
                "filing_date": pd.Timestamp(f),
            }
            for a, c, f in rows
            for seq in (1, 2)
        ]
    else:
        raise AssertionError(table)
    frame = pd.DataFrame(out)
    if table == Tables.sec_fundamentals_facts:
        frame = frame.assign(**{column: None for column in build_history.FACT_COLUMNS if column not in frame.columns})
    return frame


PURGED_FAMILIES = (Tables.sec_8k, Tables.sec_fundamentals_facts, Tables.notes_num, Tables.insider_transactions)


def _seed(store) -> None:
    """ALB: two own filings and two AllianceBernstein (AB) filings in each family; MSFT: one own, one foreign."""
    for table in PURGED_FAMILIES:
        store.save(
            table,
            _filings(
                table, "ALB", [("alb-1", ALB, "2024-02-01"), ("alb-2", ALB, "2024-05-01"), ("ab-1", AB, "2019-03-01"), ("ab-2", AB, "2021-08-01")]
            ),
        )
        store.save(table, _filings(table, "MSFT", [("msft-1", MSFT, "2024-02-01"), ("msft-x", MSFT_FOREIGN, "2024-03-01")]))


def _keys(store, table) -> set[tuple[str, str]]:
    key = filer_tables.PURGE_TABLES_BY_NAME[table.name].key_col
    frame = store.load(table, columns=["ticker", key])
    return set(zip(frame["ticker"], frame[key], strict=True))


def test_a_contraction_purges_the_removed_cik_in_every_family_and_logs_once_per_table(sqlite_store, tmp_path, monkeypatch, caplog):
    """AC-031/AC-034: ALB's scope changed after each table's last run; AB's rows go, ALB's own stay, MSFT is not read."""
    context = _context(sqlite_store, tmp_path)
    _seed(sqlite_store)
    identity = _stored_identity(sqlite_store, {"ALB": AFTER})
    resolver = identity

    with caplog.at_level(logging.INFO, logger=LOGGER):
        result = run_sync(context, resolver, monkeypatch, as_of=RUN_DATE)

    for table in PURGED_FAMILIES:
        keys = _keys(sqlite_store, table)
        assert ("ALB", "ab-1") not in keys and ("ALB", "ab-2") not in keys, table.name
        assert {("ALB", "alb-1"), ("ALB", "alb-2"), ("MSFT", "msft-1"), ("MSFT", "msft-x")} <= keys, table.name
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO and "identity sync: purged" in r.getMessage()]
    assert len(warnings) == len(PURGED_FAMILIES), warnings
    for table in PURGED_FAMILIES:
        line = next(w for w in warnings if f"'{table.name}'" in w)
        assert f"ALB cik {AB} 2019-03-01..2021-08-01 4 row(s)" in line, line
    removed = result.removals
    assert set(removed["table"]) == {t.name for t in PURGED_FAMILIES} and set(removed["cik"]) == {AB} and set(removed["rows"]) == {4}
    assert result.moves.empty, "AB is no universe company's CIK: nothing moves"
    print("\n=== SANITY CHECK: contraction purge ===")
    print(
        f"  4 families: AB's 2 filings (4 rows each) purged under ALB; ALB's own and MSFT's (scope unchanged) rows kept; {len(warnings)} INFO lines (PD-14)"
    )
    print("  nothing moved (AB belongs to no universe company); no zip read. Validated.")


def _employee_row(ticker: str, as_of: str, cik: str, accession: str, total: int | None, status: str) -> dict:
    """One `fundamentals_employees` row in its stored shape."""
    return {
        "ticker": ticker,
        "as_of": pd.Timestamp(as_of),
        "cik": cik,
        "accession_number": accession,
        "form": "10-K",
        "employees_total": total,
        "employees_full_time": None,
        "employees_part_time": None,
        "basis": "total" if total is not None else None,
        "status": status,
        "source_document": "primary",
        "source_quote": '{"total": "We had employees."}' if total is not None else None,
        "measurement_period": None,
    }


def test_a_contraction_purges_foreign_employee_rows_by_filer_cik(sqlite_store, tmp_path, monkeypatch):
    """AC-005: ALB's scope changed; AB's employee rows under ALB (a count and a NULL-status row) go, ALB's own and MSFT's stay."""
    context = _context(sqlite_store, tmp_path)
    rows = [
        _employee_row("ALB", "2024-02-20", ALB, "alb-10k", 9000, "found"),
        _employee_row("ALB", "2019-03-01", AB, "ab-10k-1", 4000, "found"),
        _employee_row("ALB", "2021-03-01", AB, "ab-10k-2", None, "not_disclosed"),
        _employee_row("MSFT", "2024-08-01", MSFT_FOREIGN, "msft-x", 1000, "found"),
    ]
    sqlite_store.save(Tables.fundamentals_employees, pd.DataFrame(rows).astype({"employees_total": "Int64"}))
    resolver = _identity({"ALB": AFTER})

    result = run_sync(context, resolver, monkeypatch, as_of=RUN_DATE)

    stored = sqlite_store.load(Tables.fundamentals_employees, columns=["ticker", "accession_number"])
    keys = set(zip(stored["ticker"], stored["accession_number"], strict=True))
    assert keys == {("ALB", "alb-10k"), ("MSFT", "msft-x")}, keys
    removed = result.removals[result.removals["table"] == Tables.fundamentals_employees.name]
    assert removed[["ticker", "cik", "first_filed", "last_filed", "rows"]].values.tolist() == [["ALB", AB, "2019-03-01", "2021-03-01", 2]]
    print("\n=== SANITY CHECK: employee rows purged by filer CIK (AC-005) ===")
    print("  AB's two employee rows under ALB (one count, one NULL-status) purged; ALB's own row and MSFT's (scope unchanged) kept. Validated.")


def test_a_sibling_cik_of_the_entity_survives_the_purge_except_its_event_filings(sqlite_store, tmp_path, monkeypatch):
    """AC-022 with P35: T-Mobile USA's facts under TMUS are the entity's own; its 8-Ks (an event-only CIK) and the
    foreign filer's rows go."""
    context = _context(sqlite_store, tmp_path)
    for table in (Tables.sec_8k, Tables.sec_fundamentals_facts):
        sqlite_store.save(
            table, _filings(table, "TMUS", [("t-1", TMUS, "2024-02-01"), ("t-usa", TMO_USA, "2016-02-01"), ("s-1", SPRINT, "2019-07-01")])
        )
    resolver = _identity({"TMUS": AFTER})

    run_sync(context, resolver, monkeypatch, as_of=RUN_DATE)

    expected = {Tables.sec_8k: {TMUS: 2}, Tables.sec_fundamentals_facts: {TMUS: 2}}
    for table, want in expected.items():
        rows = sqlite_store.load(table, columns=["ticker", "cik"])
        counts = rows.groupby("cik").size().to_dict()
        assert counts == want, (table.name, counts)
    print("\n=== SANITY CHECK: sibling CIK (AC-022, P35, H1) ===")
    print("  TMUS: T-Mobile USA (event-only, no window) loses its 8-K and its facts; the foreign 0001727074 filing purged")


def test_an_event_only_cik_leaves_the_consolidating_tables_of_a_single_window_ticker(sqlite_store, tmp_path, monkeypatch):
    """H1 (AC-004), PLD shape: old ProLogis is a CIK of PLD's entity with no window, so its facts and proxies are purged
    whatever their date; Prologis Inc's (the window) and its insider rows (UNION by policy) stay."""
    prologis, old_prologis = "0001045609", "0000899881"
    rows = [
        {**_row("ALB", prologis, "cik_window", stamp=AFTER), "company_id": "E-PLD", "canonical_ticker": "PLD"},
        {**_row("ALB", old_prologis, "cik_event", stamp=AFTER), "company_id": "E-PLD", "canonical_ticker": "PLD"},
    ]
    tenure = pd.DataFrame(
        [{"symbol": "PLD", "issuer_cik": prologis, "valid_from": pd.Timestamp("2000-01-01"), "valid_to": None, "n_filings": 5, "source": "form345"}]
    )
    identity = build_identity(pd.DataFrame(rows), tenure, as_roster(pd.DataFrame([{"ticker": "PLD", "cik": prologis}])))
    filings = [("old-1", old_prologis, "2010-11-09"), ("old-2", old_prologis, "2011-08-09"), ("new-1", prologis, "2011-08-09")]
    for table in (Tables.sec_fundamentals_facts, Tables.def14a_edgar, Tables.insider_transactions):
        sqlite_store.save(table, _filings(table, "PLD", filings))
    context = _context(sqlite_store, tmp_path)
    context.universe = universe_of(["PLD"], rows={"PLD": {"cik": prologis}})

    result = run_sync(context, identity, monkeypatch, as_of=RUN_DATE)

    for table in (Tables.sec_fundamentals_facts, Tables.def14a_edgar):
        assert _keys(sqlite_store, table) == {("PLD", "new-1")}, table.name
    assert _keys(sqlite_store, Tables.insider_transactions) == {("PLD", "old-1"), ("PLD", "old-2"), ("PLD", "new-1")}
    got = sorted(zip(result.removals["table"], result.removals["cik"], result.removals["keys"], strict=True))
    assert got == [(Tables.def14a_edgar.name, old_prologis, 2), (Tables.sec_fundamentals_facts.name, old_prologis, 2)], got
    print("\n=== SANITY CHECK: H1 purge, single-window ticker ===")
    print("  PLD: old ProLogis's 2 facts filings and 2 proxies purged (before and after the 2011 seam); Prologis Inc's kept;")
    print("  old ProLogis's insider rows kept (Forms 3/4/5 UNION across the entity)")


def test_a_cik_with_no_digit_is_never_judged_as_the_validator_rules(sqlite_store, tmp_path, monkeypatch):
    """D-7: the purge's pre-filter is the validator's. A junk CIK string ('N/A', '-') is not a filer, so its rows stay."""
    context = _context(sqlite_store, tmp_path)
    for table in (Tables.sec_8k, Tables.sec_fundamentals_facts):
        filings = [("alb-1", ALB, "2024-02-01"), ("junk-1", "N/A", "2023-02-01"), ("junk-2", " - ", "2023-03-01"), ("ab-1", AB, "2019-03-01")]
        sqlite_store.save(table, _filings(table, "ALB", filings))
    resolver = _identity({"ALB": AFTER})

    result = run_sync(context, resolver, monkeypatch, as_of=RUN_DATE)

    for table in (Tables.sec_8k, Tables.sec_fundamentals_facts):
        keys = _keys(sqlite_store, table)
        assert {("ALB", "alb-1"), ("ALB", "junk-1"), ("ALB", "junk-2")} <= keys, (table.name, keys)
        assert ("ALB", "ab-1") not in keys, table.name
    assert set(result.removals["cik"]) == {AB}, result.removals
    assert filer_tables.judged_cik_mask(pd.Series(["N/A", " - ", "", None, "789019", 320193.0], dtype=object)).tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
    ]
    print("\n=== SANITY CHECK: junk-CIK pre-filter (D-7) ===")
    print("  'N/A' and ' - ' rows kept in sec_8k and sec_fundamentals_facts; only AB's foreign filing purged")
    print("  OK: a CIK with no digit is never judged, by the purge exactly as by the validator.")


def test_an_unchanged_lineage_purges_nothing_and_full_rechecks_every_ticker(sqlite_store, tmp_path, monkeypatch):
    """AC-032: every scope stamp is older than every table's last run."""
    context = _context(sqlite_store, tmp_path)
    _seed(sqlite_store)
    resolver = _identity({})
    before = {table.name: sqlite_store.row_count(table) for table in PURGED_FAMILIES}

    result = run_sync(context, resolver, monkeypatch, as_of=RUN_DATE)

    assert result.removals.empty
    assert {table.name: sqlite_store.row_count(table) for table in PURGED_FAMILIES} == before
    every = run_sync(context, resolver, monkeypatch, full=True, as_of=RUN_DATE).removals
    assert set(zip(every["ticker"], every["cik"], strict=True)) == {("ALB", AB), ("MSFT", MSFT_FOREIGN)}
    print("\n=== SANITY CHECK: unchanged lineage (AC-032) ===")
    print("  no stamp in the window: nothing purged; `--full` re-checks every ticker and purges both foreign filers")


def _history(ticker: str, dates: list, revenue: float) -> pd.DataFrame:
    """A stub replay result carrying every history column (the build reads them back by projection)."""
    columns = list(load_catalogue("./configs").history_columns)
    return pd.DataFrame({column: [None] * len(dates) for column in columns}).assign(ticker=ticker, as_of=list(dates), totalRevenue=revenue)


def test_the_history_of_a_purged_ticker_is_rebuilt_without_the_foreign_filing(sqlite_store, tmp_path, monkeypatch):
    """M7: published history rows built from a foreign filing are replaced by the nightly history build once the sync
    purged its facts (the ticker's scope changed inside the re-check window)."""
    context = _context(sqlite_store, tmp_path)
    sqlite_store.save(
        Tables.sec_fundamentals_facts, _filings(Tables.sec_fundamentals_facts, "ALB", [("alb-1", ALB, "2024-02-01"), ("ab-1", AB, "2021-08-01")])
    )

    def one_row_per_filing(ticker, facts, **kwargs):
        dates = sorted(pd.to_datetime(facts["filing_date"]).unique())
        return _history(ticker, dates, float(len(dates)))

    monkeypatch.setattr(build_history, "build_ticker", one_row_per_filing)
    # Published before the purge: the history build itself now sets AB aside (no window of ALB), so the stale rows are seeded.
    sqlite_store.save(Tables.sec_fundamentals_history, _history("ALB", [pd.Timestamp("2021-08-01"), pd.Timestamp("2024-02-01")], 2.0))
    assert sqlite_store.row_count(Tables.sec_fundamentals_history) == 2
    resolver = _identity({"ALB": AFTER})
    monkeypatch.setattr(build_history, "load_identity", lambda context: _identity({"ALB": AFTER}))

    run_sync(context, resolver, monkeypatch, as_of=RUN_DATE)
    build_history.build_fundamentals_history(context, ["ALB"], as_of=RUN_DATE)

    history = sqlite_store.load(Tables.sec_fundamentals_history, columns=["ticker", "as_of", "totalRevenue"])
    assert [str(pd.Timestamp(d).date()) for d in history["as_of"]] == ["2024-02-01"] and list(history["totalRevenue"]) == [1.0]
    print("\n=== SANITY CHECK: history rebuild ===")
    print("  2 published rows (one from AB's filing) -> 1 row after the sync purge and the nightly SEC history build")


def test_the_history_build_rebuilds_a_ticker_whose_scope_changed_since_its_last_run(sqlite_store, tmp_path, monkeypatch):
    """Expansion: predecessor facts change published rows; the replay rebuilds instead of refusing."""
    context = _context(sqlite_store, tmp_path)
    sqlite_store.save(Tables.sec_fundamentals_facts, _filings(Tables.sec_fundamentals_facts, "ALB", [("alb-1", ALB, "2024-02-01")]))

    def count_rows(ticker, facts, **kwargs):
        return _history(ticker, [pd.Timestamp("2024-02-01")], float(len(facts)))

    monkeypatch.setattr(build_history, "build_ticker", count_rows)
    monkeypatch.setattr(build_history, "load_identity", lambda context: _identity({"ALB": BEFORE}))
    build_history.build_fundamentals_history(context, ["ALB"], as_of=RUN_DATE)
    sqlite_store.save(Tables.sec_fundamentals_facts, _filings(Tables.sec_fundamentals_facts, "ALB", [("pred-1", ALB, "2010-02-01")]))

    with pytest.raises(ValueError, match="append-only"):
        build_history.build_fundamentals_history(context, ["ALB"], as_of=RUN_DATE)
    monkeypatch.setattr(build_history, "load_identity", lambda context: _identity({"ALB": AFTER}))
    build_history.build_fundamentals_history(context, ["ALB"], as_of=RUN_DATE)

    history = sqlite_store.load(Tables.sec_fundamentals_history, columns=["as_of", "totalRevenue"])
    assert list(history["totalRevenue"]) == [4.0]
    print("\n=== SANITY CHECK: history expansion rebuild ===")
    print("  unchanged scope: a changed published row still refuses; scope changed inside the re-check window: ALB rebuilt (2 -> 4 fact rows)")
