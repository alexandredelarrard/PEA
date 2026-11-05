"""Propagation stage (phase 7, REQ-014): the 13F re-stamp, the first-build full re-check (PD-13) and the dry run (AC-013).

Known truth on SQLite: a CUSIP whose dated `universe_cusip` line names VMRK moves its stored `sec13f_hr` rows from the
market ticker EQR to VMRK (AC-009), a non-universe CUSIP keeps its map ticker, a dry run reports without writing and a
second run moves nothing. On the phase-0 fixture store (a read-only copy published by the phase-5 parity): the real
EQR -> VMRK move of CUSIP 29476L107. The publish of a store without a lineage marks a first build, which re-checks every
ticker; the next publish does not.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.data_extract.transformers.step_extract_identity import IdentityOptions
from src.data_extract.utils.identity.identity_propagate import purge_rows, retag_13f, thirteenf_moves
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.filer_tables import PURGE_TABLES_BY_NAME
from tests.data_extract.identity.test_publish import _step, published_store  # noqa: F401 (fixture)
from tests.support.propagation import stub_propagation

ROOT = Path(__file__).resolve().parents[3]
RUN = Path("reports") / "validate" / "2026-11-03-identity-security-master" / "_cache"
RUN_DATE = pd.Timestamp("2026-11-05")
EQR_CUSIP, OTHER_CUSIP = "29476L107", "999999105"


def _context(store: DataStore) -> SimpleNamespace:
    return SimpleNamespace(store=store, log=logging.getLogger("test.identity.propagate"), config_dir=str(ROOT / "configs"))


def _seed(store: DataStore) -> None:
    """VMRK's one canonical line over 29476L107 (stamped yesterday), the market map, and three 13F books."""
    line = {
        "security_id": "C29476L107", "canonical_company": "VMRK", "cusip": EQR_CUSIP, "source": "ftd", "source_symbol": "EQR",
        "market_symbol": "EQR", "security_class": "common", "conversion_ratio": 1.0, "lineage_role": "canonical_current",
        "valid_from": pd.Timestamp("1995-01-01"), "valid_to": pd.NaT, "scope_changed_at": RUN_DATE - pd.Timedelta(days=1),
    }  # fmt: skip
    store.save(Tables.universe_cusip, pd.DataFrame([line]))
    store.save(Tables.cusip_ticker_map, pd.DataFrame({"cusip": [EQR_CUSIP, OTHER_CUSIP], "ticker": ["EQR", "ZZZ"]}))
    books = [
        (cik, period, f"acc-{cik}-{period}", ticker, cusip, 100.0)
        for cik, period in (("0000000001", "2024-03-31"), ("0000000001", "2024-06-30"), ("0000000002", "2024-06-30"))
        for ticker, cusip in (("EQR", EQR_CUSIP), ("ZZZ", OTHER_CUSIP))
    ]
    frame = pd.DataFrame(books, columns=["cik", "period", "accession_number", "ticker", "cusip", "shares"])
    store.save(Tables.sec13f_hr, frame.assign(period=pd.to_datetime(frame["period"])))


def _tickers(store: DataStore, cusip: str) -> dict[str, int]:
    rows = store.load(Tables.sec13f_hr, columns=["ticker", "cusip"])
    return rows[rows["cusip"].eq(cusip)]["ticker"].value_counts().to_dict()


def test_13f_rows_move_to_the_company_of_their_dated_line_and_a_second_run_moves_nothing(sqlite_store: DataStore) -> None:
    """AC-009 known truth: EQR's 3 rows of 29476L107 move to VMRK; ZZZ keeps its map ticker; dry run writes nothing."""
    _seed(sqlite_store)
    context = _context(sqlite_store)

    dry = retag_13f(context, thirteenf_moves(context, ["VMRK"], RUN_DATE, False), True)
    after_dry = _tickers(sqlite_store, EQR_CUSIP)
    wet = retag_13f(context, thirteenf_moves(context, ["VMRK"], RUN_DATE, False), False)
    again = thirteenf_moves(context, ["VMRK"], RUN_DATE, False)
    stale = thirteenf_moves(context, ["VMRK"], RUN_DATE + pd.Timedelta(days=60), False)

    assert dry[["from_ticker", "to_ticker", "rows"]].values.tolist() == [["EQR", "VMRK", 3]], dry
    assert after_dry == {"EQR": 3}, "a dry run writes nothing"
    assert wet.equals(dry) and _tickers(sqlite_store, EQR_CUSIP) == {"VMRK": 3}
    assert _tickers(sqlite_store, OTHER_CUSIP) == {"ZZZ": 3}, "a CUSIP outside every line keeps its market ticker"
    assert again.empty and stale.empty, "a second run (and a stamp outside the window) moves nothing"
    print("\n=== SANITY CHECK: 13F re-stamp (AC-009, AC-013) ===")
    print(f"  dry run: {dry.to_dict('records')}; stored after the dry run {after_dry}")
    print(f"  real run: 29476L107 {_tickers(sqlite_store, EQR_CUSIP)}, 999999105 {_tickers(sqlite_store, OTHER_CUSIP)}; second run 0 move(s)")


def test_fixture_eqr_rows_move_to_vmrk_on_the_published_lines() -> None:
    """AC-009 on real data: the fixture's 39,575 EQR rows of 29476L107 move to VMRK (39,573) or unmapped (2, 1998-Q2/Q3,
    before the line's first confirmed 13F period); read-only, dry run."""
    path = next((base / RUN / "parity" / "fixture_copy_5.sqlite" for base in (ROOT, *ROOT.parents) if (base / RUN).exists()), None)
    if path is None or not path.exists():
        pytest.skip("the phase-5 parity copy of the fixture store is not reachable from this checkout")
    store = DataStore(create_engine(f"sqlite:///file:{path.as_posix()}?mode=ro&uri=true"))
    context = _context(store)

    report = retag_13f(context, thirteenf_moves(context, ["VMRK"], RUN_DATE, True), True)

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
