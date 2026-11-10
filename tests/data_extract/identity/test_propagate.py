"""Propagation stage (phase 7, REQ-014): the first-build full re-check (PD-13), the bulk re-parse scope (X90) and the
purge around a company end. The publish of a store without a lineage marks a first build, which re-checks every ticker;
the next publish does not. `sec13f_hr` is stored as filed, so the propagation no longer touches it (REQ-005).
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.context import Context
from src.data_extract.transformers import step_extract_identity as step_module
from src.data_extract.transformers.step_extract_identity import IdentityOptions
from src.data_extract.utils.identity.identity_propagate import purge_rows, rebuilt_since
from src.data_extract.utils.identity.lifecycle import Lifecycle
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.filer_tables import PURGE_TABLES_BY_NAME
from tests.data_extract.identity.test_publish import _step, published_store  # noqa: F401 (fixture)
from tests.support.propagation import stub_propagation
from tests.support.universe import universe_config

ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = pd.Timestamp("2026-11-05")


def _context(store: DataStore) -> SimpleNamespace:
    return SimpleNamespace(store=store, log=logging.getLogger("test.identity.propagate"), config_dir=str(ROOT / "configs"))


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
    assert calls["recheck"][0]["tickers"] == ["AAA"]
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
    for name in ("refresh_insider", "refresh_short_volume"):
        monkeypatch.setattr(step_module, name, lambda *args, **kwargs: [])
    for name in ("reparse_notes", "reparse_insider"):
        monkeypatch.setattr(step_module, name, lambda *args, **kwargs: {})
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
