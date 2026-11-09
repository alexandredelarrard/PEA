"""Propagation stage (phase 7, REQ-014): the 13F re-stamp, the first-build full re-check (PD-13) and the dry run (AC-013).

Known truth on SQLite: the accessions holding a CUSIP whose dated `universe_cusip` line names VMRK, or a second class
stored apart from its company row (U-67), are re-read (offline stub) and rewritten: EQR's rows move to VMRK (AC-009),
BRK-A + BRK-B make one BRK-B row in B shares, a non-universe CUSIP keeps its map ticker, a dry run reports without
writing and a second run moves nothing. On the phase-0 fixture store (a read-only copy published by the phase-5 parity): the real
EQR -> VMRK move of CUSIP 29476L107. The publish of a store without a lineage marks a first build, which re-checks every
ticker; the next publish does not.
"""

from __future__ import annotations

import logging
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.context import Context
from src.data_aggregate.utils.sec_thirteen import inputs as thirteen_inputs
from src.data_aggregate.utils.sec_thirteen.holdings_clean import clean_public_batches
from src.data_aggregate.utils.sec_thirteen.institutional_books import prepare_institutional_book_history
from src.data_extract.transformers import step_extract_identity as step_module
from src.data_extract.transformers.step_extract_identity import IdentityOptions
from src.data_extract.utils.identity.identity_propagate import RESTAMP_COLUMNS, purge_rows, rebuilt_since, retag_13f, thirteenf_moves
from src.data_extract.utils.identity.lifecycle import Lifecycle
from src.data_extract.utils.institutionals import fetch_13f_backfill as backfill
from src.data_extract.utils.institutionals.fetch_13f import filing_vintage, replace_hr_filing, save_hr_filing
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.filer_tables import PURGE_TABLES_BY_NAME
from src.utils.identity_tables import LINE_COLUMNS
from tests.data_extract.identity.test_publish import _step, published_store  # noqa: F401 (fixture)
from tests.data_extract.institutionals.test_13f_one_walk import _line as _info_line
from tests.support.propagation import stub_propagation
from tests.support.universe import universe_config

ROOT = Path(__file__).resolve().parents[3]
RUN = Path("reports") / "validate" / "2026-11-03-identity-security-master" / "_cache"
RUN_DATE = pd.Timestamp("2026-11-05")
EQR_CUSIP, OTHER_CUSIP, BRK_A, BRK_B = "29476L107", "999999105", "084670108", "084670702"


def _context(store: DataStore) -> SimpleNamespace:
    return SimpleNamespace(store=store, log=logging.getLogger("test.identity.propagate"), config_dir=str(ROOT / "configs"))


def _line(cusip: str, company: str, role: str, symbol: str, ratio: float, start: str) -> dict[str, Any]:
    return {
        "security_id": f"C{cusip}", "canonical_company": company, "cusip": cusip, "source": "ftd", "source_symbol": symbol,
        "market_symbol": symbol, "security_class": "common", "conversion_ratio": ratio, "lineage_role": role,
        "valid_from": pd.Timestamp(start), "valid_to": pd.NaT, "scope_changed_at": RUN_DATE - pd.Timedelta(days=1),
    }  # fmt: skip


def _packet(cik: str, period: str, positions: list[tuple[str, str, float, int]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One accession's source packet (dollars era); positions are (cusip, issuer, value, shares)."""
    info = pd.DataFrame([_info_line(cusip, name, value, shares) for cusip, name, value, shares in positions])
    filed = (pd.Timestamp(period) + pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    return filing_vintage(
        info,
        cik=cik,
        accession_number=f"acc-{cik}-{period}",
        period=period,
        filing_date=filed,
        amendment_type="original",
        source=f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/acc-{cik}-{period}.txt",
        declared_rows=len(info),
        declared_value=float(info["VALUE"].sum()),
    )


#: Three books as the pre-U-67 rule stored them: one row per CUSIP, EQR on its market ticker, BRK-A beside BRK-B.
PACKETS = [
    _packet("0000000001", "2024-03-31", [(EQR_CUSIP, "EQUITY RESIDENTIAL", 7_000.0, 100), (OTHER_CUSIP, "ZZZ CORP", 500.0, 50)]),
    _packet(
        "0000000001",
        "2024-06-30",
        [(EQR_CUSIP, "EQUITY RESIDENTIAL", 7_000.0, 100), (BRK_A, "BERKSHIRE A", 1_300_000.0, 2), (BRK_B, "BERKSHIRE B", 43_000.0, 100)],
    ),
    _packet("0000000002", "2024-06-30", [(EQR_CUSIP, "EQUITY RESIDENTIAL", 7_000.0, 100), (OTHER_CUSIP, "ZZZ CORP", 500.0, 50)]),
]


def _seed(store: DataStore) -> None:
    """The market map and the three books stored one row per CUSIP, then the published lines (stamped yesterday): VMRK
    over 29476L107 and BRK-B's two classes."""
    store.save(Tables.cusip_ticker_map, pd.DataFrame({"cusip": [EQR_CUSIP, OTHER_CUSIP, BRK_A, BRK_B], "ticker": ["EQR", "ZZZ", "BRK-A", "BRK-B"]}))
    old = {EQR_CUSIP: "EQR", OTHER_CUSIP: "ZZZ", BRK_A: "BRK-B", BRK_B: "BRK-B"}
    old_map = pd.DataFrame({"cusip": list(old), "ticker": list(old.values())})
    for manifest, book in PACKETS:
        save_hr_filing(store, manifest, book, pd.DataFrame(columns=LINE_COLUMNS), old_map)
    lines = [
        _line(EQR_CUSIP, "VMRK", "canonical_current", "EQR", 1.0, "1995-01-01"),
        _line(BRK_B, "BRK-B", "canonical_current", "BRK-B", 1.0, "1998-12-31"),
        _line(BRK_A, "BRK-B", "secondary_class", "BRK-A", 1500.0, "2010-01-21"),
    ]
    store.save(Tables.universe_cusip, pd.DataFrame(lines))


def _rows(store: DataStore) -> pd.DataFrame:
    rows = store.load(Tables.sec13f_hr, columns=["accession_number", "ticker", "cusip", "shares", "value_usd"])
    return rows.sort_values(["accession_number", "cusip"], ignore_index=True)


def _offline_reread(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The re-read serves each accession's source packet as filed (no network); returns the accessions asked."""
    packets = {str(m.iloc[0].accession_number): (m, b) for m, b in PACKETS}
    asked: list[str] = []
    monkeypatch.setattr(backfill, "read_filing_vintage", lambda stamp: asked.append(stamp.accession_number) or packets[stamp.accession_number])
    return asked


def test_13f_accessions_are_rewritten_as_company_rows_and_a_second_run_moves_nothing(
    filing_store: DataStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC-009 + U-67 known truth: EQR's 3 rows of 29476L107 move to VMRK; the 2 BRK-A + 100 BRK-B of one book become one
    BRK-B row of 3,100 B shares; ZZZ keeps its map ticker; a dry run writes nothing; a second run finds nothing."""
    _seed(filing_store)
    context = _context(filing_store)
    asked = _offline_reread(monkeypatch)
    before = _rows(filing_store)

    dry = retag_13f(context, *thirteenf_moves(context, ["VMRK", "BRK-B"], RUN_DATE, False), True)
    after_dry = _rows(filing_store)
    wet = retag_13f(context, *thirteenf_moves(context, ["VMRK", "BRK-B"], RUN_DATE, False), False)
    after = _rows(filing_store)
    again, _ = thirteenf_moves(context, ["VMRK", "BRK-B"], RUN_DATE, False)
    stale, _ = thirteenf_moves(context, ["VMRK", "BRK-B"], RUN_DATE + pd.Timedelta(days=60), False)

    moved = {(r.cusip, r.from_ticker, r.to_ticker): r.rows for r in dry.itertuples(index=False)}
    assert moved == {(EQR_CUSIP, "EQR", "VMRK"): 3, (BRK_A, "BRK-B", "BRK-B"): 1}, moved
    assert after_dry.equals(before), "a dry run writes nothing"
    assert wet.equals(dry) and sorted(asked) == sorted(str(m.iloc[0].accession_number) for m, _ in PACKETS)
    brk = after[after["ticker"].eq("BRK-B")]
    assert brk[["cusip", "shares", "value_usd"]].values.tolist() == [[BRK_B, 3_100.0, 1_343_000.0]]
    assert set(after.loc[after["cusip"].eq(EQR_CUSIP), "ticker"]) == {"VMRK"}
    assert set(after.loc[after["cusip"].eq(OTHER_CUSIP), "ticker"]) == {"ZZZ"}, "a CUSIP outside every line keeps its market ticker"
    assert after["value_usd"].sum() == before["value_usd"].sum() and len(after) == len(before) - 1
    assert again.empty and stale.empty, "a second run (and a stamp outside the window) finds nothing out of place"
    print("\n=== SANITY CHECK: 13F rewrite of out-of-place accessions (AC-009, AC-013, U-67) ===")
    print(f"  dry run: {dry.to_dict('records')}; stored rows unchanged ({len(after_dry)})")
    print(
        f"  real run: {len(asked)} accession(s) re-read offline; rows {len(before)} -> {len(after)}; value {before['value_usd'].sum():,.0f} = {after['value_usd'].sum():,.0f}"
    )
    print(f"  BRK-B: one row of {brk['shares'].iloc[0]:,.0f} B shares (2 x 1,500 + 100); EQR -> VMRK; ZZZ unchanged; second run 0 group(s)")


def test_a_collapsed_accession_marker_counts_the_stored_rows_and_the_cube_reads_it(filing_store: DataStore) -> None:
    """R-01 (U-67): an accession holding BRK-A and BRK-B is stored as one BRK-B row, so its `sec13f_hr` marker counts
    the 2 position rows stored (not the 3 filed) on save and on replace, and the cube's book history reads it."""
    manifest, book = PACKETS[1]
    lines = pd.DataFrame(
        [
            _line(EQR_CUSIP, "VMRK", "canonical_current", "EQR", 1.0, "1995-01-01"),
            _line(BRK_B, "BRK-B", "canonical_current", "BRK-B", 1.0, "1998-12-31"),
            _line(BRK_A, "BRK-B", "secondary_class", "BRK-A", 1500.0, "2010-01-21"),
        ]
    )
    cmap = pd.DataFrame({"cusip": [], "ticker": []})
    grid = pd.bdate_range("2024-06-01", "2024-12-31")

    def marker_and_history() -> tuple[int, int]:
        stored = filing_store.load(Tables.sec13f_hr, columns=["parse_status", "parsed_row_count"], markers=True)
        packets = thirteen_inputs.iter_filing_vintages(filing_store, grid.max())
        with closing(clean_public_batches(packets, ["BRK-B", "VMRK"], None, None)) as batches:
            history = prepare_institutional_book_history(batches, grid, settle_days=3)
        assert history is not None
        return int(stored.loc[stored["parse_status"].eq("complete"), "parsed_row_count"].iloc[0]), len(history[0])

    saved = save_hr_filing(filing_store, manifest, book, lines, cmap)
    saved_count, saved_events = marker_and_history()
    replaced = replace_hr_filing(filing_store, manifest, book, lines, cmap)
    replaced_count, replaced_events = marker_and_history()

    assert len(book) == 3 and saved == replaced == 2
    assert saved_count == replaced_count == 2, (saved_count, replaced_count)
    assert saved_events == replaced_events > 0
    print("\n=== SANITY CHECK: collapsed 13F accession reaches the cube (R-01) ===")
    print(f"  filed CUSIPs {len(book)} -> stored position rows {saved}; marker parsed_row_count save={saved_count}, replace={replaced_count}")
    print(f"  cube book history built: {saved_events} event row(s) after save, {replaced_events} after replace -> no 'Incomplete stored 13F book'")


def test_fixture_eqr_rows_move_to_vmrk_on_the_published_lines() -> None:
    """AC-009 on real data: the fixture's 39,575 EQR rows of 29476L107 move to VMRK (39,573) or unmapped (2, 1998-Q2/Q3,
    before the line's first confirmed 13F period); read-only, dry run."""
    path = next((base / RUN / "parity" / "fixture_copy_5.sqlite" for base in (ROOT, *ROOT.parents) if (base / RUN).exists()), None)
    if path is None or not path.exists():
        pytest.skip("the phase-5 parity copy of the fixture store is not reachable from this checkout")
    store = DataStore(create_engine(f"sqlite:///file:{path.as_posix()}?mode=ro&uri=true"))
    context = _context(store)

    report = retag_13f(context, *thirteenf_moves(context, ["VMRK"], RUN_DATE, True), True)

    moved = {(r.from_ticker, r.to_ticker): r.rows for r in report.itertuples(index=False)}
    assert moved == {("EQR", "VMRK"): 39573, ("EQR", "_empty"): 2}, moved
    print("\n=== SANITY CHECK: fixture 13F re-stamp, VMRK (dry run) ===")
    print(report.to_string(index=False))
    print("  OK: every EQR row of 29476L107 leaves EQR; 0 rows would stay under the market ticker")


def test_a_first_build_rechecks_every_ticker_and_the_next_publish_does_not(published_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811 (fixture)
    """PD-13: the publish of a store without a lineage is a first build, so its propagation re-checks every ticker as
    `--full` does; the next publish finds the lineage and the propagation goes back to the stamps."""
    calls = stub_propagation(monkeypatch)
    everies = []
    for _ in range(2):
        step: Any = _step(published_store)
        step._options, step._run_date = IdentityOptions(), RUN_DATE
        step.run_publish()
        step.run_propagate()
        everies.append(calls["recheck"][-1]["every"])

    assert everies == [True, False], everies
    assert calls["recheck"][0]["tickers"] == ["AAA"] and calls["retag"] == [False, False]
    print("\n=== SANITY CHECK: first-build propagation (PD-13) ===")
    print(f"  publish 1 (no stored lineage): every={everies[0]}; publish 2: every={everies[1]}")


def test_the_bulk_reparse_follows_only_the_scopes_this_run_changed(published_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811 (fixture)
    """X90: the Notes and Form 3/4/5 re-parses read every cached zip, so after the first build they take only the tickers
    whose scope stamp is from this run; a ticker still inside the 7-day re-check window from an earlier night is re-checked
    (purge, re-stamps) but never re-parsed again."""
    since = pd.Timestamp("2026-11-05 10:00:00")
    scopes = {"AAA": SimpleNamespace(scope_changed_at=pd.Timestamp("2026-11-04 22:00:00")), "BBB": SimpleNamespace(scope_changed_at=since)}
    calls = stub_propagation(monkeypatch)
    monkeypatch.setattr(step_module, "recheck_tickers", lambda scopes, as_of, every: sorted(scopes))
    parsed: list[list[str]] = []
    monkeypatch.setattr(step_module, "reparse_notes", lambda context, tickers, *args: parsed.append(list(tickers)) or {})
    step: Any = _step(published_store)
    step._options, step._run_date, step._first_build = IdentityOptions(), RUN_DATE, False

    step.run_propagate()

    assert rebuilt_since(scopes, since) == ["BBB"] and rebuilt_since({"CCC": SimpleNamespace(scope_changed_at=None)}, since) == []
    assert parsed == [[]] and calls["reload"], parsed
    print("\n=== SANITY CHECK: bulk re-parse after the first build (X90) ===")
    print(f"  stamps AAA 2026-11-04 22:00, BBB 2026-11-05 10:00, run started 10:00 -> re-parsed {rebuilt_since(scopes, since)}")
    print(f"  a later run whose scopes are all older re-checks AAA but re-parses {parsed[0]} (no 2 h Notes pass)")


def test_the_purge_leaves_rows_dated_after_the_company_end(sqlite_store: DataStore) -> None:
    """REQ-014: WBD's window closes at its end (2026-10-06 exclusive); its completion 8-K filed 2026-10-06, after
    `ended_on` 2026-10-05, is left as stored, while a foreign filer's 8-K inside the life is purged."""
    wbd, other = "0001437107", "0000000999"
    rows = [("WBD", wbd, "2026-10-06", "0001193125-26-415055"), ("WBD", wbd, "2026-09-01", "own-1"), ("WBD", other, "2020-01-02", "foreign-1")]
    frame = pd.DataFrame(rows, columns=["ticker", "cik", "filing_date", "accession_number"]).assign(item="8.01")
    sqlite_store.save(Tables.sec_8k, frame.assign(filing_date=pd.to_datetime(frame["filing_date"])))
    owned = ({"WBD": frozenset({wbd})}, {"WBD": ((wbd, None, pd.Timestamp("2026-10-06")),)})
    spec = PURGE_TABLES_BY_NAME[Tables.sec_8k.name]
    ended = {"WBD": (None, pd.Timestamp("2026-10-05"))}

    open_life = purge_rows(_context(sqlite_store), spec, ["WBD"], owned, {}, True)
    removed = purge_rows(_context(sqlite_store), spec, ["WBD"], owned, ended, False)

    kept = set(sqlite_store.load(Tables.sec_8k, columns=["accession_number"])["accession_number"])
    assert sum(r["rows"] for r in open_life) == 2, "without an end the post-window 8-K is foreign too"
    assert [(r["cik"], r["rows"]) for r in removed] == [(other, 1)] and kept == {"0001193125-26-415055", "own-1"}, (removed, kept)
    print("\n=== SANITY CHECK: rows after ended_on are left (REQ-014) ===")
    print(f"  no end: {sum(r['rows'] for r in open_life)} row(s) would go; WBD ended 2026-10-05: purged {removed}; kept {sorted(kept)}")


def test_an_end_first_published_this_run_keeps_the_post_end_rows_through_propagate(
    published_store: DataStore,  # noqa: F811 (fixture)
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R-02 (X80 shape): AAA's `ended_on` 2026-10-05 is first published by this run's publish, which closes its window at
    2026-10-06; the propagation of the same context (whose cached universe was read before the publish) re-reads the universe, sees that end and must
    leave AAA's 8-K filed 2026-10-06, while purging a foreign filer's 8-K inside the life."""
    end, own, other = pd.Timestamp("2026-10-05"), "0000000100", "0000000999"
    step: Any = _step(published_store)
    context: Any = object.__new__(Context)
    context.store, context.log, context._config, context._config_dir = published_store, step._log, universe_config(["AAA"]), step._config_dir
    context._universe = None
    step._context = context
    lineage = step._symbols.rows.assign(valid_to=end + pd.Timedelta(days=1))
    step._symbols = SimpleNamespace(rows=lineage, backlog=step._symbols.backlog, blocked=step._symbols.blocked)
    step._lifecycle = Lifecycle(rows=step._lifecycle.rows.assign(ended_on=end, status="ended"), items=step._lifecycle.items)
    step._options, step._run_date = IdentityOptions(), RUN_DATE
    rows = [("AAA", own, "2026-10-06", "post-end-1"), ("AAA", own, "2026-09-01", "own-1"), ("AAA", other, "2020-01-02", "foreign-1")]
    frame = pd.DataFrame(rows, columns=["ticker", "cik", "filing_date", "accession_number"]).assign(item="8.01")
    published_store.save(Tables.sec_8k, frame.assign(filing_date=pd.to_datetime(frame["filing_date"])))
    tenure = {"symbol": "AAA", "issuer_cik": own, "valid_from": pd.Timestamp("2006-01-03"), "valid_to": end, "n_filings": 3}
    published_store.save(Tables.symbol_tenure, pd.DataFrame([{**tenure, "source": "form345", "evidence_period": "", "evidence": "seed"}]))
    for name in ("refresh_insider", "refresh_fails", "refresh_short_volume"):
        monkeypatch.setattr(step_module, name, lambda *args, **kwargs: [])
    for name in ("reparse_notes", "reparse_insider"):
        monkeypatch.setattr(step_module, name, lambda *args, **kwargs: {})
    monkeypatch.setattr(step_module, "thirteenf_moves", lambda *args, **kwargs: (pd.DataFrame(), None))
    monkeypatch.setattr(step_module, "retag_13f", lambda *args, **kwargs: pd.DataFrame(columns=list(RESTAMP_COLUMNS)))
    monkeypatch.setattr(step_module, "rebuild_history", lambda *args, **kwargs: ())

    seen_before = context.universe.bounds["AAA"][1]
    step.run_publish()
    result = step.run_propagate()
    cached_after = context.universe.bounds["AAA"][1]

    kept = sorted(published_store.load(Tables.sec_8k, columns=["accession_number"])["accession_number"])
    bound_end = context._identity_resolver.bounds["AAA"][1]
    assert seen_before is None and bound_end == end, (seen_before, bound_end)
    assert cached_after is None, "the context cache is left as read: only the propagation re-reads the universe"
    assert kept == ["own-1", "post-end-1"], kept
    assert result.removals["rows"].sum() == 1, result.removals
    print("\n=== SANITY CHECK: an end first published this run reaches the propagation (R-02, X80) ===")
    print(
        f"  universe read before publish: AAA ended_on={seen_before}; resolver bound after publish: {bound_end:%Y-%m-%d}; context cache untouched ({cached_after})"
    )
    print(f"  purged {int(result.removals['rows'].sum())} foreign row(s); kept {kept} (the 2026-10-06 8-K stays, REQ-014)")
