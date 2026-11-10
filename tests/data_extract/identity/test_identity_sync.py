"""`identity --only sync` (REQ-009, decision Q1): the published identity reaches the CIK-keyed stored rows by SQL only.

A CIK that moved to another universe company within a dated window takes its rows there (delete + insert; a row
already stored under the new key wins), a row filed outside every scope is purged, and the insider lineage stamps are
rewritten. No zip is opened: the real resolver is rebuilt from the published tables of a SQLite store.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.context import Context
from src.data_extract.transformers import step_extract_identity as step_module
from src.data_extract.transformers.step_extract_identity import IdentityOptions, StepExtractIdentity
from src.data_extract.utils.identity.identity_sync import sync_filer_rows
from src.data_extract.utils.identity.lifecycle import Lifecycle
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.filer_tables import PURGE_TABLES_BY_NAME, foreign_filer_rows
from tests.data_extract.fake_context import extract_config
from tests.data_extract.identity.test_publish import _step, published_store  # noqa: F401 (fixture)
from tests.support.universe import seed_universe, universe_config

ROOT = Path(__file__).resolve().parents[3]
SENTINEL = "1900-01-01"
RUN_DATE = pd.Timestamp("2026-01-13")
#: inside the 7-day re-check window before RUN_DATE
CHANGED = pd.Timestamp("2026-01-12 09:00")
AAA, BBB, MOVED = "0000000100", "0000000200", "0000000300"
ROSTER = {"AAA": AAA, "BBB": BBB}
#: the run date of the publish-then-sync cases (`test_publish` fixtures)
PUBLISH_RUN = pd.Timestamp("2026-11-05")


def _lineage_row(ticker: str, cik: str, *, start: str = SENTINEL, end: str | None = None) -> dict[str, Any]:
    return {
        "company_id": f"E{ROSTER[ticker]}",
        "canonical_ticker": ticker,
        "cik": cik,
        "role": "cik_window",
        "symbol": "",
        "valid_from": pd.Timestamp(start),
        "valid_to": pd.Timestamp(end) if end else None,
        "status": "curated",
        "sources": "register",
        "oracle": "fixture",
        "confidence": None,
        "n_observations": 1,
        "evidence": "fixture",
        "scope_changed_at": CHANGED,
    }


def _publish(store: DataStore) -> None:
    """The published identity after the move: MOVED is BBB's window CIK 2015-01-01..2020-01-01 (BBB's own from 2020)."""
    lineage = [
        _lineage_row("AAA", AAA),
        _lineage_row("BBB", MOVED, start="2015-01-01", end="2020-01-01"),
        _lineage_row("BBB", BBB, start="2020-01-01"),
    ]
    store.save(Tables.universe_cik_symbol, pd.DataFrame(lineage))
    tenure = [
        {"symbol": t, "issuer_cik": c, "valid_from": pd.Timestamp("2000-01-01"), "valid_to": None, "n_filings": 5}
        | {"source": "form345", "evidence_period": "2000q1", "evidence": "fixture"}
        for t, c in ROSTER.items()
    ]
    store.save(Tables.symbol_tenure, pd.DataFrame(tenure))


def _context(store: DataStore, tmp_path: Path) -> SimpleNamespace:
    universe = seed_universe(store, list(ROSTER), rows={t: {"cik": c} for t, c in ROSTER.items()})
    config = extract_config(data_extract={"years_history": 15})
    return SimpleNamespace(
        store=store,
        log=logging.getLogger("test.identity.sync"),
        config=config,
        config_dir=str(ROOT / "configs"),
        paths={"DATA_STORE": tmp_path},
        universe=universe,
    )


def _sync(context: SimpleNamespace) -> Any:
    """`identity --only sync --as-of RUN_DATE` through the step's production interface."""
    step = StepExtractIdentity(context, context.config, IdentityOptions(as_of=RUN_DATE))  # type: ignore[arg-type]
    return step.run_sync()


@pytest.fixture
def no_zip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any zip opened during the test raises."""

    def refuse(self: Any, *args: Any, **kwargs: Any) -> None:
        raise AssertionError(f"the sync opened a zip: {args[:1]}")

    monkeypatch.setattr(zipfile.ZipFile, "__init__", refuse)


def _insider(ticker: str, accession: str, cik: str, filed: str, role: str) -> dict[str, Any]:
    return {
        "accession_number": accession,
        "security_type": "nonderiv",
        "row_sequence": 1,
        "ticker": ticker,
        "issuer_cik": cik,
        "owner_cik": "0000009999",
        "document_type": "4",
        "transaction_code": "P",
        "transaction_date": pd.Timestamp(filed),
        "filing_date": pd.Timestamp(filed),
        "security_title": "Common Stock",
        "shares": 10.0,
        "source": "zip",
        "source_symbol": ticker,
        "economic_date": pd.Timestamp(filed),
        "lineage_role": role,
    }


def _keys(store: DataStore, table: Any, key: str) -> set[tuple[str, str]]:
    frame = store.load(table, columns=["ticker", key])
    return set(zip(frame["ticker"], frame[key], strict=True))


def test_a_cik_moved_to_another_company_is_retickered_per_dated_window_with_no_zip(sqlite_store: DataStore, tmp_path: Path, no_zip: None) -> None:
    """MOVED's rows stored under AAA move to BBB where BBB's scope holds them: the 2016 8-K (inside the dated window) and
    both notes filings (a consolidating table holds any window CIK's filing); the 2012 8-K, outside the window, is
    purged. BBB already stores the 2016 8-K under its own key (PK collision): that row is kept, AAA's copy dropped."""
    _publish(sqlite_store)
    filings = [
        ("AAA", "m-2016", MOVED, "2016-03-01", "aaa copy"),
        ("AAA", "m-2012", MOVED, "2012-05-01", "aaa"),
        ("AAA", "a-1", AAA, "2024-02-01", "own"),
    ]
    filings += [("BBB", "m-2016", MOVED, "2016-03-01", "bbb fetch")]
    eight_k = pd.DataFrame(filings, columns=["ticker", "accession_number", "cik", "filing_date", "form"]).assign(item="2.02")
    sqlite_store.save(Tables.sec_8k, eight_k.assign(filing_date=pd.to_datetime(eight_k["filing_date"])))
    notes = [("AAA", "n-2016", MOVED, "2016-03-01"), ("AAA", "n-2012", MOVED, "2012-05-01"), ("AAA", "n-own", AAA, "2024-02-01")]
    frame = pd.DataFrame(notes, columns=["ticker", "adsh", "cik", "filed"]).assign(tag="T1", ddate=pd.Timestamp("2015-12-31"), qtrs=0, value=1.0)
    sqlite_store.save(Tables.notes_num, frame.assign(filed=pd.to_datetime(frame["filed"])))
    sqlite_store.save(Tables.insider_transactions, pd.DataFrame([_insider("AAA", "i-2016", MOVED, "2016-03-01", "canonical_current")]))

    result = _sync(_context(sqlite_store, tmp_path))

    eight_k_keys = _keys(sqlite_store, Tables.sec_8k, "accession_number")
    forms = sqlite_store.load(Tables.sec_8k, columns=["ticker", "accession_number", "form"])
    assert eight_k_keys == {("AAA", "a-1"), ("BBB", "m-2016")}, eight_k_keys
    assert forms.loc[forms["accession_number"].eq("m-2016"), "form"].tolist() == ["bbb fetch"], forms
    assert _keys(sqlite_store, Tables.notes_num, "adsh") == {("BBB", "n-2016"), ("BBB", "n-2012"), ("AAA", "n-own")}
    insider = sqlite_store.load(Tables.insider_transactions, columns=["ticker", "accession_number", "lineage_role"])
    assert insider.values.tolist() == [["BBB", "i-2016", "canonical_predecessor"]], insider
    moves = result.moves.set_index("table")
    assert moves.loc[Tables.sec_8k.name, ["ticker", "new_ticker", "rows", "kept"]].tolist() == ["AAA", "BBB", 1, 1]
    assert result.removals[["table", "ticker", "cik", "rows"]].values.tolist() == [[Tables.sec_8k.name, "AAA", MOVED, 1]]
    print("\n=== SANITY CHECK: dated re-ticker, SQL only (Q1, M7) ===")
    print(f"  sec_8k: {sorted(eight_k_keys)} (m-2016 moved, BBB's stored copy kept on the PK collision; m-2012 outside the window purged)")
    print("  notes_num: both MOVED filings now under BBB; insider i-2016 -> BBB, role canonical_predecessor; no zip opened")


def test_a_role_change_restamps_the_insider_lineage_role(sqlite_store: DataStore, tmp_path: Path, no_zip: None) -> None:
    """BBB's lineage now dates MOVED as its 2015-2020 window CIK: its row stored as `acquired_constituent` (the old
    event-only view) becomes `canonical_predecessor`; BBB's own 2024 row stays `canonical_current`."""
    _publish(sqlite_store)
    rows = [_insider("BBB", "i-old", MOVED, "2017-06-01", "acquired_constituent"), _insider("BBB", "i-own", BBB, "2024-06-01", "canonical_current")]
    sqlite_store.save(Tables.insider_transactions, pd.DataFrame(rows))

    _sync(_context(sqlite_store, tmp_path))

    got = sqlite_store.load(Tables.insider_transactions, columns=["accession_number", "lineage_role"])
    roles = dict(zip(got["accession_number"], got["lineage_role"], strict=True))
    assert roles == {"i-old": "canonical_predecessor", "i-own": "canonical_current"}, roles
    print("\n=== SANITY CHECK: insider role re-stamp in the sync ===")
    print(f"  roles after the sync: {roles} (acquired_constituent -> canonical_predecessor, read from stored rows only)")


# --------------------------------------------------------------------------- the re-check scope and the company end


def _published_step(store: DataStore) -> Any:
    """The publish step of `test_publish` over `store` (AAA only in the universe), with the sync's run date."""
    step: Any = _step(store)
    step._options, step._run_date = IdentityOptions(), PUBLISH_RUN
    return step


def test_a_first_build_rechecks_every_ticker_and_the_next_publish_does_not(published_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811 (fixture)
    """PD-13: the publish of a store without a lineage is a first build, so its sync re-checks every ticker as `--full`
    does; the next publish finds the lineage and the sync goes back to the stamps."""
    everies: list[bool] = []
    tickers: list[list[str]] = []
    real = step_module.recheck_tickers

    def recheck(scopes: Any, as_of: pd.Timestamp, every: bool) -> list[str]:
        everies.append(every)
        tickers.append(sorted(scopes))
        return real(scopes, as_of, every)

    monkeypatch.setattr(step_module, "recheck_tickers", recheck)
    tenure = {"symbol": "AAA", "issuer_cik": "0000000100", "valid_from": pd.Timestamp("2006-01-03"), "valid_to": None, "n_filings": 3}
    published_store.save(Tables.symbol_tenure, pd.DataFrame([{**tenure, "source": "form345", "evidence_period": "", "evidence": "seed"}]))
    for _ in range(2):
        step = _published_step(published_store)
        step.run_publish()
        step.run_sync()

    assert everies == [True, False] and tickers[0] == ["AAA"], (everies, tickers)
    print("\n=== SANITY CHECK: first-build sync (PD-13) ===")
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
    context: Any = SimpleNamespace(store=sqlite_store, log=logging.getLogger("test.identity.sync"))

    open_life = foreign_filer_rows(sqlite_store, spec, ["WBD"], *owned)
    removed, moved = sync_filer_rows(context, spec, ["WBD"], owned, {"WBD": (None, pd.Timestamp("2026-10-05"))})

    kept = set(sqlite_store.load(Tables.sec_8k, columns=["accession_number"])["accession_number"])
    assert len(open_life) == 2, "without an end the post-window 8-K is foreign too"
    assert [(r["cik"], r["rows"]) for r in removed] == [(other, 1)] and moved == [] and kept == {"0001193125-26-415055", "own-1"}, (removed, kept)
    print("\n=== SANITY CHECK: rows after ended_on are left (REQ-014) ===")
    print(f"  no end: {len(open_life)} row(s) foreign; WBD ended 2026-10-05: purged {removed}; kept {sorted(kept)}")


def test_an_end_first_published_this_run_keeps_the_post_end_rows_through_the_sync(
    published_store: DataStore,  # noqa: F811 (fixture)
) -> None:
    """R-02 (X80 shape): AAA's `ended_on` 2026-10-05 is first published by this run's publish, which closes its window at
    2026-10-06; the sync of the same context (whose cached universe was read before the publish) re-reads the universe,
    sees that end and leaves AAA's 8-K filed 2026-10-06, while purging a foreign filer's 8-K inside the life."""
    end, own, other = pd.Timestamp("2026-10-05"), "0000000100", "0000000999"
    step = _published_step(published_store)
    context: Any = object.__new__(Context)
    context.store, context.log, context._config, context._config_dir = published_store, step._log, universe_config(["AAA"]), step._config_dir
    context._universe = None
    step._context = context
    lineage = step._symbols.rows.assign(valid_to=end + pd.Timedelta(days=1))
    step._symbols = SimpleNamespace(rows=lineage, backlog=step._symbols.backlog, blocked=step._symbols.blocked)
    step._lifecycle = Lifecycle(rows=step._lifecycle.rows.assign(ended_on=end, status="ended"), items=step._lifecycle.items)
    rows = [("AAA", own, "2026-10-06", "post-end-1"), ("AAA", own, "2026-09-01", "own-1"), ("AAA", other, "2020-01-02", "foreign-1")]
    frame = pd.DataFrame(rows, columns=["ticker", "cik", "filing_date", "accession_number"]).assign(item="8.01")
    published_store.save(Tables.sec_8k, frame.assign(filing_date=pd.to_datetime(frame["filing_date"])))
    tenure = {"symbol": "AAA", "issuer_cik": own, "valid_from": pd.Timestamp("2006-01-03"), "valid_to": end, "n_filings": 3}
    published_store.save(Tables.symbol_tenure, pd.DataFrame([{**tenure, "source": "form345", "evidence_period": "", "evidence": "seed"}]))

    seen_before = context.universe.bounds["AAA"][1]
    step.run_publish()
    result = step.run_sync()
    cached_after = context.universe.bounds["AAA"][1]

    kept = sorted(published_store.load(Tables.sec_8k, columns=["accession_number"])["accession_number"])
    bound_end = context._identity_resolver.bounds["AAA"][1]
    assert seen_before is None and bound_end == end, (seen_before, bound_end)
    assert cached_after is None, "the context cache is left as read: only the sync re-reads the universe"
    assert kept == ["own-1", "post-end-1"], kept
    assert result.removals["rows"].sum() == 1, result.removals
    print("\n=== SANITY CHECK: an end first published this run reaches the sync (R-02, X80) ===")
    print(f"  universe before publish: AAA ended_on={seen_before}; resolver bound after publish: {bound_end:%Y-%m-%d}; cache untouched")
    print(f"  purged {int(result.removals['rows'].sum())} foreign row(s); kept {kept} (the 2026-10-06 8-K stays, REQ-014)")
