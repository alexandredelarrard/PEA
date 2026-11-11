"""Incremental `sec_fundamentals_history`: triage from stored facts and stored events alone (no fetch hand-off), new
events only, full replay when the past moved.

Synthetic where the question is which path a ticker takes (a known-truth fixture names the expected path), real
where the question is whether incremental rows equal a full replay (the sweep ledgers in `data/fundamentals_sweep/`
are genuine `sec_fundamentals_facts` rows). The store is the in-memory SQLite double, never the live tables.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from click.testing import CliRunner

import src.data_extract.cli as cli_mod
import src.data_extract.transformers.step_extract_fundamentals as step_mod
from src.data_extract.utils.fundamentals import build_history as mod
from src.data_extract.utils.fundamentals.build_history import FACT_COLUMNS
from src.data_store.schema import Tables
from tests.data_extract.common.scope_fixtures import SENTINEL, dated_identity

LOGGER = "test.history_incremental"
CATALOGUE_DIR = "./configs"


def _fact(**kwargs: Any) -> dict:
    """One `sec_fundamentals_facts` row with every replay column defaulted."""
    row = {
        "ticker": "TST",
        "cik": "0000000001",
        "accession_number": "a-1",
        "field": "totalRevenue",
        "fiscal_year": 2023,
        "fiscal_period": "Q1",
        "duration_type": "quarterly",
        "form": "10-Q",
        "filing_date": pd.Timestamp("2023-05-01"),
        "period_of_report": pd.Timestamp("2023-03-31"),
        "regime": "industrial",
        "period_start": pd.Timestamp("2023-01-01"),
        "period_end": pd.Timestamp("2023-03-31"),
        "period_days": 89.0,
        "value": 100.0,
        "unit": "USD",
        "source_concept": "us-gaap:Revenues",
    }
    row.update(kwargs)
    return row


#: (start, end, label, filed) of four quarters of a calendar-year filer.
_WINDOWS = [
    ("2023-01-01", "2023-03-31", "Q1", "2023-05-01"),
    ("2023-04-01", "2023-06-30", "Q2", "2023-08-01"),
    ("2023-07-01", "2023-09-30", "Q3", "2023-11-01"),
    ("2023-10-01", "2023-12-31", "Q4", "2024-02-15"),
]


def _quarters(ticker: str = "TST", windows: list[tuple[str, str, str, str]] = _WINDOWS) -> pd.DataFrame:
    """One revenue flow and one asset level per filing; one filing = one publication event."""
    rows = []
    for i, (start, end, label, filed) in enumerate(windows):
        common = {
            "ticker": ticker,
            "accession_number": f"{ticker}-acc-{filed}",
            "fiscal_period": label,
            "filing_date": pd.Timestamp(filed),
            "period_of_report": pd.Timestamp(end),
            "form": "10-K" if label == "Q4" else "10-Q",
        }
        rows.append(_fact(**common, period_start=pd.Timestamp(start), period_end=pd.Timestamp(end), value=100.0 + i))
        rows.append(
            _fact(
                **common,
                field="totalAssets",
                duration_type="instant",
                period_start=None,
                period_end=pd.Timestamp(end),
                period_days=None,
                value=1000.0 + i,
                source_concept="us-gaap:Assets",
            )
        )
    return pd.DataFrame(rows)[list(FACT_COLUMNS)]


def _context(store: Any) -> SimpleNamespace:
    return SimpleNamespace(store=store, log=logging.getLogger(LOGGER), config_dir=CATALOGUE_DIR)


@pytest.fixture(autouse=True)
def _identity_without_these_tickers(monkeypatch: pytest.MonkeyPatch) -> None:
    """An identity layer whose roster holds none of the synthetic tickers, so their single-filer facts skip the seam rule."""
    identity = dated_identity([("ZZZ", "0000009999", "cik_window", SENTINEL, None)], {"ZZZ": "0000009999"})
    monkeypatch.setattr(mod, "load_identity", lambda context: identity)


@pytest.fixture
def snapshots(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every `_snapshot` call's `as_of`, in order; the lineage window reads as unchanged for every ticker."""
    calls: list[str] = []
    real = mod._snapshot

    def counted(ticker: str, visible: pd.DataFrame, event: pd.Series, *args: Any, **kwargs: Any) -> dict:
        calls.append(f"{ticker}@{pd.Timestamp(event['as_of']).date()}")
        return real(ticker, visible, event, *args, **kwargs)

    monkeypatch.setattr(mod, "_snapshot", counted)
    monkeypatch.setattr(mod, "_scope_changed_recently", lambda context, tickers, as_of=None: frozenset())
    return calls


def _spy_loads(monkeypatch: pytest.MonkeyPatch, store: Any) -> list[tuple[str, tuple[str, ...]]]:
    """Every `store.load` / `store.iter_load` call as `(table, columns)`."""
    seen: list[tuple[str, tuple[str, ...]]] = []
    load, iter_load = store.load, store.iter_load

    def spy_load(table: Any, columns: Any = None, *args: Any, **kwargs: Any) -> pd.DataFrame | None:
        seen.append((str(getattr(table, "name", table)), tuple(columns or ())))
        return load(table, columns, *args, **kwargs)

    def spy_iter(table: Any, **kwargs: Any) -> Any:
        seen.append((str(getattr(table, "name", table)), tuple(kwargs.get("columns") or ())))
        return iter_load(table, **kwargs)

    monkeypatch.setattr(store, "load", spy_load)
    monkeypatch.setattr(store, "iter_load", spy_iter)
    return seen


def _full_facts_reads(seen: list[tuple[str, tuple[str, ...]]]) -> int:
    """Reads of a ticker's whole replay projection (the cost a skipped ticker must not pay)."""
    return sum(1 for table, columns in seen if table == Tables.sec_fundamentals_facts.name and columns == tuple(FACT_COLUMNS))


# --------------------------------------------------------------------------- #
# T1 (AC-001) an unchanged ticker costs no snapshot                            #
# --------------------------------------------------------------------------- #
def test_unchanged_ticker_computes_no_snapshot(sqlite_store: Any, snapshots: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    sqlite_store.save(Tables.sec_fundamentals_facts, _quarters())
    context = _context(sqlite_store)
    mod.build_fundamentals_history(context, ["TST"])
    first = len(snapshots)
    stored = sqlite_store.row_count(Tables.sec_fundamentals_history)
    assert first == 4 and stored == 4, (first, stored)

    snapshots.clear()
    seen = _spy_loads(monkeypatch, sqlite_store)
    mod.build_fundamentals_history(context, ["TST"])

    assert snapshots == [], f"{len(snapshots)} snapshot(s) recomputed for a ticker with no new filing: {snapshots}"
    assert sqlite_store.row_count(Tables.sec_fundamentals_history) == stored
    assert _full_facts_reads(seen) == 0, "a skipped ticker read its full facts"
    print("\n=== SANITY CHECK: unchanged ticker ===")
    print(f"  first build: {first} snapshots, {stored} rows; re-run with no new filing: 0 snapshots, 0 rows, 0 full facts reads. Validated.")


# --------------------------------------------------------------------------- #
# helpers on the store                                                         #
# --------------------------------------------------------------------------- #
def _stored(store: Any, ticker: str) -> pd.DataFrame:
    """The ticker's stored history, in `as_of` order."""
    history = store.load(Tables.sec_fundamentals_history, where={"ticker": ticker}, optional=True)
    return pd.DataFrame() if history is None else history.sort_values("as_of").reset_index(drop=True)


def _keep_first(store: Any, ticker: str, history: pd.DataFrame, n_rows: int) -> pd.Timestamp:
    """Rewrite the ticker's stored rows as the first `n_rows` events of `history`; returns the newest kept `as_of`."""
    store.delete(Tables.sec_fundamentals_history, {"ticker": ticker})
    kept = history.iloc[:n_rows]
    store.save(Tables.sec_fundamentals_history, kept)
    return pd.Timestamp(kept["as_of"].max())


def _tamper(store: Any, ticker: str, as_of: str, **cells: Any) -> None:
    """Overwrite cells of one stored history row in place (an upsert on its key)."""
    history = _stored(store, ticker)
    row = history[pd.to_datetime(history["as_of"]) == pd.Timestamp(as_of)].copy()
    assert len(row) == 1, as_of
    store.save(Tables.sec_fundamentals_history, row.assign(**cells))


def _built(store: Any, snapshots: list[str], ticker: str = "TST") -> int:
    """Build `ticker` once from its stored facts (first build), clear the snapshot log, return the stored row count."""
    mod.build_fundamentals_history(_context(store), [ticker])
    snapshots.clear()
    return len(_stored(store, ticker))


# --------------------------------------------------------------------------- #
# T2 (AC-002) incremental rows equal a full replay, on real filers             #
# --------------------------------------------------------------------------- #
SWEEP = Path("data/fundamentals_sweep")
GOOGLE, ALPHABET = "0001288776", "0001652044"

#: (ticker, last filing date kept): a plain filer, a bank, and a two-CIK filer whose window spans its seam.
#: Truncated so each full replay stays short; the incremental property does not depend on the window length.
_REAL = [("AAPL", "2012-12-31"), ("JPM", "2012-12-31"), ("GOOGL", "2016-03-31")]


def _real_facts(ticker: str, until: str) -> pd.DataFrame:
    path = SWEEP / f"{ticker}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not present (real `sec_fundamentals_facts` ledger, main checkout only)")
    facts = pd.read_parquet(path)
    if "prefer_structure" in facts:
        facts = facts[facts["prefer_structure"].astype(bool)]
    # The ledger also carries period-less diagnostic rows, which the table's key (`period_end`) cannot hold.
    facts = facts[(pd.to_datetime(facts["filing_date"]) <= pd.Timestamp(until)) & facts["period_end"].notna()]
    return facts[list(FACT_COLUMNS)].reset_index(drop=True)


@pytest.mark.parametrize(("ticker", "until"), _REAL)
def test_incremental_rows_equal_full_replay(
    sqlite_store: Any, snapshots: list[str], monkeypatch: pytest.MonkeyPatch, ticker: str, until: str
) -> None:
    identity = dated_identity(
        [("GOOGL", GOOGLE, "cik_window", SENTINEL, "2015-10-02"), ("GOOGL", ALPHABET, "cik_window", "2015-10-02", None)], {"GOOGL": ALPHABET}
    )
    monkeypatch.setattr(mod, "load_identity", lambda context: identity)
    facts = _real_facts(ticker, until)
    sqlite_store.save(Tables.sec_fundamentals_facts, facts)
    _built(sqlite_store, snapshots, ticker)
    full_history = _stored(sqlite_store, ticker)
    events = len(full_history)
    assert events > 8, f"{ticker}: only {events} events in the window"

    report = []
    for dropped in (1, 4):
        newest = _keep_first(sqlite_store, ticker, full_history, events - dropped)
        snapshots.clear()
        mod.build_fundamentals_history(_context(sqlite_store), [ticker])
        history = _stored(sqlite_store, ticker)
        assert len(snapshots) == dropped, f"{ticker}: {len(snapshots)} snapshots for {dropped} new event(s)"
        assert all(pd.Timestamp(s.split("@")[1]) > newest for s in snapshots), snapshots
        pd.testing.assert_frame_equal(history, full_history)
        report.append(f"drop {dropped}: {len(snapshots)} snapshot(s); {len(history)} rows cell-identical")

    print(f"\n=== SANITY CHECK: incremental == full replay ({ticker}, {len(facts)} facts filed <= {until}) ===")
    print(f"  full replay: {events} events, forms {full_history['publication_form'].value_counts().to_dict()}")
    for line in report:
        print(f"  {line}")
    print("  Each snapshot reads only its own facts prefix, so the new rows equal the full replay's. Validated.")


# --------------------------------------------------------------------------- #
# T3 (AC-007) triage end results from stored facts alone, no hand-off           #
# --------------------------------------------------------------------------- #
def _fresh_build(facts: pd.DataFrame) -> pd.DataFrame:
    """The full replay of `facts` into an empty store: the end result every path must reach."""
    return mod.build_ticker("TST", facts, catalogue=mod.load_catalogue(CATALOGUE_DIR)).reset_index(drop=True)


def _assert_stored_equals(store: Any, facts: pd.DataFrame) -> pd.DataFrame:
    history = _stored(store, "TST")
    expected = _fresh_build(facts)
    keys = ["as_of", "publication_form", "totalRevenue", "totalAssets"]
    floats = {"totalRevenue": float, "totalAssets": float}
    stored = history[keys].astype(floats).assign(as_of=pd.to_datetime(history["as_of"])).reset_index(drop=True)
    pd.testing.assert_frame_equal(stored, expected[keys].astype(floats).assign(as_of=pd.to_datetime(expected["as_of"])), check_dtype=False)
    return history


def test_triage_end_results_without_a_fetch_hand_off(sqlite_store: Any, snapshots: list[str]) -> None:
    """(a) no new filing -> skip; (b) a newer original -> incremental; (c) a back-dated original -> full replay with
    its row added; (d) a back-dated original and a newer one stored the same night -> full replay. Each end result
    equals a fresh full build of the stored facts; the build takes no fetch argument."""
    first = _quarters(windows=_WINDOWS[:2])
    sqlite_store.save(Tables.sec_fundamentals_facts, first)
    _built(sqlite_store, snapshots)
    context = _context(sqlite_store)

    # (a) skip
    mod.build_fundamentals_history(context, ["TST"])
    skipped = len(snapshots)

    # (b) a newer original: only its event is snapshotted
    newer = _quarters(windows=_WINDOWS[2:3])
    sqlite_store.save(Tables.sec_fundamentals_facts, newer)
    snapshots.clear()
    mod.build_fundamentals_history(context, ["TST"])
    incremental = len(snapshots)
    _assert_stored_equals(sqlite_store, pd.concat([first, newer]))

    # (c) a back-dated original (re-files Q1's numbers on 2023-06-15): the full replay adds its row
    late = _quarters(windows=[("2023-01-01", "2023-03-31", "Q1", "2023-06-15")]).assign(accession_number="TST-late")
    sqlite_store.save(Tables.sec_fundamentals_facts, late)
    snapshots.clear()
    mod.build_fundamentals_history(context, ["TST"])
    backdated = len(snapshots)
    _assert_stored_equals(sqlite_store, pd.concat([first, newer, late]))

    # (d) a back-dated original (Q2's numbers on 2023-09-15) and the next 10-K, stored the same night
    late_q2 = _quarters(windows=[("2023-04-01", "2023-06-30", "Q2", "2023-09-15")]).assign(
        accession_number="TST-late-q2", value=lambda d: d["value"] + 1
    )
    q4 = _quarters(windows=_WINDOWS[3:])
    sqlite_store.save(Tables.sec_fundamentals_facts, pd.concat([late_q2, q4]))
    snapshots.clear()
    mod.build_fundamentals_history(context, ["TST"])
    same_night = len(snapshots)
    history = _assert_stored_equals(sqlite_store, pd.concat([first, newer, late, late_q2, q4]))

    assert (skipped, incremental, backdated, same_night) == (0, 1, 4, 6), (skipped, incremental, backdated, same_night)
    assert "fetched" not in mod.build_fundamentals_history.__code__.co_varnames
    print("\n=== SANITY CHECK: triage end results, no hand-off ===")
    print(f"  no new filing: {skipped} snapshots (skip); a newer 10-Q: {incremental} snapshot (incremental)")
    print(f"  back-dated original 2023-06-15: {backdated} snapshots (full replay), its row added")
    print(f"  back-dated 2023-09-15 + newer 10-K the same night: {same_night} snapshots (full replay); {len(history)} rows stored")
    print("  every stored history equals a fresh full build of the stored facts. Validated.")


# --------------------------------------------------------------------------- #
# T4 (AC-004) an event-list mismatch forces the full replay                    #
# --------------------------------------------------------------------------- #
def test_event_list_mismatch_forces_full_replay(sqlite_store: Any, snapshots: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    sqlite_store.save(Tables.sec_fundamentals_facts, _quarters())
    _built(sqlite_store, snapshots)
    context = _context(sqlite_store)

    # (a) Stored provenance disagrees with the rebuilt event list; lineage-recent -> check -> full -> rebuilt.
    _tamper(sqlite_store, "TST", "2023-08-01", publication_form="10-K")
    monkeypatch.setattr(mod, "_scope_changed_recently", lambda context, tickers, as_of=None: frozenset({"TST"}))
    mod.build_fundamentals_history(context, ["TST"])
    provenance = len(snapshots)
    history = _stored(sqlite_store, "TST")
    form = history.loc[pd.to_datetime(history["as_of"]) == pd.Timestamp("2023-08-01"), "publication_form"].iloc[0]
    assert provenance == 4 and form == "10-Q", (provenance, form)

    # (b) A back-dated original: its date is not stored -> check -> full; it adds its own row.
    monkeypatch.setattr(mod, "_scope_changed_recently", lambda context, tickers, as_of=None: frozenset())
    late = _quarters(windows=[("2023-01-01", "2023-03-31", "Q1", "2023-06-15")]).assign(accession_number="TST-late")
    sqlite_store.save(Tables.sec_fundamentals_facts, late)
    snapshots.clear()
    mod.build_fundamentals_history(context, ["TST"])
    backfilled = len(snapshots)
    assert backfilled == 5 and sqlite_store.row_count(Tables.sec_fundamentals_history) == 5, backfilled

    # (c) A late original while a stored row before it no longer recomputes: the drift guard on rows before it refuses.
    _tamper(sqlite_store, "TST", "2023-05-01", totalAssets=999.0)
    sqlite_store.save(Tables.sec_fundamentals_facts, late.assign(accession_number="TST-later", filing_date=pd.Timestamp("2023-09-15"), value=1.0))
    with pytest.raises(ValueError, match="append-only"):
        mod.build_fundamentals_history(context, ["TST"])

    print("\n=== SANITY CHECK: event-list mismatch ===")
    print(f"  tampered stored publication_form (lineage-recent): {provenance} snapshots, row rebuilt to {form!r}")
    print(f"  back-dated original 2023-06-15, no hand-off: {backfilled} snapshots, its row added (5 rows)")
    print("  late original 2023-09-15 with a changed stored row before it: drift error (append-only). Validated.")


# --------------------------------------------------------------------------- #
# T5 (AC-005) the triage classifies every ticker before any replay            #
# --------------------------------------------------------------------------- #
def test_triage_classifies_tickers(sqlite_store: Any, snapshots: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    names = ("NEW", "GROW", "CUR", "RECENT", "MOVED", "AMENDED")
    for name in names:
        sqlite_store.save(Tables.sec_fundamentals_facts, _quarters(name, _WINDOWS[:3]))
    mod.build_fundamentals_history(_context(sqlite_store), [n for n in names if n != "NEW"])
    sqlite_store.save(Tables.sec_fundamentals_facts, _quarters("GROW", _WINDOWS[3:]))
    sqlite_store.save(Tables.sec_fundamentals_facts, _quarters("MOVED", [("2023-01-01", "2023-03-31", "Q1", "2023-06-15")]))
    # An amendment filed after the newest stored row is not an original: it is ignored, so AMENDED stays current.
    sqlite_store.save(Tables.sec_fundamentals_facts, _quarters("AMENDED", [("2023-01-01", "2023-03-31", "Q1", "2023-12-01")]).assign(form="10-Q/A"))

    seen = _spy_loads(monkeypatch, sqlite_store)
    context = _context(sqlite_store)
    work = mod._history_work(context, list(names), seams={}, windows={}, recent=frozenset({"RECENT"}))
    paths = {ticker: item.path for ticker, item in work.items()}
    verify = {ticker: item.path for ticker, item in mod._history_work(context, list(names), seams={}, windows={}, verify=True).items()}

    # MOVED is the late path: a full replay that rewrites its rows from `rewrite_from`.
    assert paths == {"NEW": "full", "GROW": "incremental", "CUR": "skip", "RECENT": "check", "MOVED": "full", "AMENDED": "skip"}, paths
    assert set(verify.values()) == {"full"}
    assert work["GROW"].newest == pd.Timestamp("2023-11-01") and work["NEW"].newest is None
    assert work["MOVED"].rewrite_from == pd.Timestamp("2023-06-15") and work["GROW"].rewrite_from is None
    assert _full_facts_reads(seen) == 0, "the triage read a ticker's full facts"
    narrow = {columns for table, columns in seen if table == Tables.sec_fundamentals_facts.name}
    assert narrow == {mod._TRIAGE_COLUMNS}, narrow

    print("\n=== SANITY CHECK: triage ===")
    for ticker, path in paths.items():
        print(f"  {ticker:6s} -> {path}")
    print(f"  verify -> {sorted(set(verify.values()))}; facts reads: {sorted(narrow)} only, no full facts read. Validated.")


# --------------------------------------------------------------------------- #
# T6 (AC-006) verify and rebuild keep today's full-replay semantics            #
# --------------------------------------------------------------------------- #
def test_verify_history_replays_in_full_under_the_drift_guard(sqlite_store: Any, snapshots: list[str]) -> None:
    sqlite_store.save(Tables.sec_fundamentals_facts, _quarters())
    _built(sqlite_store, snapshots)
    context = _context(sqlite_store)

    mod.build_fundamentals_history(context, ["TST"], verify_history=True)
    verified = len(snapshots)
    assert verified == 4 and sqlite_store.row_count(Tables.sec_fundamentals_history) == 4

    _tamper(sqlite_store, "TST", "2023-08-01", totalAssets=999.0)
    snapshots.clear()
    mod.build_fundamentals_history(context, ["TST"])
    assert snapshots == [], "a tampered stored value is not a triage signal"
    with pytest.raises(ValueError, match="append-only"):
        mod.build_fundamentals_history(context, ["TST"], verify_history=True)

    print("\n=== SANITY CHECK: --verify-history ===")
    print(f"  current ticker: {verified} snapshots, 0 rows written; a tampered stored cell: daily run 0 snapshots, verify -> drift error. Validated.")


def test_rebuild_history_deletes_and_replays_in_full(sqlite_store: Any, snapshots: list[str], caplog: pytest.LogCaptureFixture) -> None:
    sqlite_store.save(Tables.sec_fundamentals_facts, _quarters())
    _built(sqlite_store, snapshots)
    _tamper(sqlite_store, "TST", "2023-08-01", totalAssets=999.0)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        mod.build_fundamentals_history(_context(sqlite_store), ["TST"], rebuild_history=True)

    history = _stored(sqlite_store, "TST")
    restored = history.loc[pd.to_datetime(history["as_of"]) == pd.Timestamp("2023-08-01"), "totalAssets"].iloc[0]
    assert len(snapshots) == 4 and restored == 1001.0 and len(history) == 4, (len(snapshots), restored)
    assert "TST REBUILT -- 4 row(s) deleted and recomputed" in caplog.text
    print("\n=== SANITY CHECK: sec-fundamentals-history --full ===")
    print(f"  tampered 999 -> {restored} after delete + {len(snapshots)}-event replay; the REBUILT warning is logged. Validated.")


# --------------------------------------------------------------------------- #
# `sec-fundamentals-history --verify-history` reaches the history build         #
# --------------------------------------------------------------------------- #
def test_verify_history_flag_reaches_the_history_build(monkeypatch: pytest.MonkeyPatch) -> None:
    builds: list[dict] = []
    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=15))
    monkeypatch.setattr(cli_mod, "get_config_context", lambda path, **kwargs: (config, _context(None)))
    monkeypatch.setattr(cli_mod, "_tickers", lambda ctx, names: [t.strip().upper() for t in names.split(",")])
    monkeypatch.setattr(step_mod, "build_fundamentals_history", lambda context, tickers, **kwargs: builds.append(kwargs))
    runner = CliRunner()
    for args in (
        ["sec-fundamentals-history", "-t", "AAA"],
        ["sec-fundamentals-history", "-t", "AAA", "--verify-history"],
        ["sec-fundamentals-history", "-t", "AAA", "-F"],
    ):
        result = runner.invoke(cli_mod.cli, args)
        assert result.exit_code == 0, result.output
    names = set(cli_mod.cli.list_commands(None))  # type: ignore[arg-type]
    helps = {name: runner.invoke(cli_mod.cli, [name, "--help"]).output for name in ("sec-fundamentals-history", "sec-fundamentals-facts")}

    assert [(b["rebuild_history"], b["verify_history"]) for b in builds] == [(False, False), (False, True), (True, False)], builds
    assert "fundamentals-history-sec" not in names and "sec-fundamentals-history" in names
    assert "--verify-history" in helps["sec-fundamentals-history"] and "--verify-history" not in helps["sec-fundamentals-facts"]
    print("\n=== SANITY CHECK: sec-fundamentals-history wiring ===")
    print(
        f"  (rebuild, verify) per call: {[(b['rebuild_history'], b['verify_history']) for b in builds]} through StepExtractFundamentals.run_history; "
        "fundamentals-history-sec is gone; only the history command lists --verify-history. Validated."
    )


# --------------------------------------------------------------------------- #
# P3 (AC-007) the replay pool saves exactly what the in-process build saves    #
# --------------------------------------------------------------------------- #
class _CountingPool(ProcessPoolExecutor):
    """A real process pool that counts its submissions."""

    submitted: list[str] = []

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
        _CountingPool.submitted.append(str(args[0]))
        return super().submit(fn, *args, **kwargs)


def _pool_run(store: Any, workers: int) -> pd.DataFrame:
    """Three first builds and one incremental ticker (BBB, 3 of 4 events stored) with `workers`; the saved history."""
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    store.delete(Tables.sec_fundamentals_history, {"ticker": tickers})
    serial = _context(store)
    mod.build_fundamentals_history(serial, ["BBB"])
    _keep_first(store, "BBB", _stored(store, "BBB"), 3)
    context = SimpleNamespace(**vars(serial), config=SimpleNamespace(data_extract=SimpleNamespace(fundamentals_workers=workers)))
    mod.build_fundamentals_history(context, tickers)
    return store.load(Tables.sec_fundamentals_history).sort_values(["ticker", "as_of"]).reset_index(drop=True)


def test_pool_build_equals_in_process_build(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_scope_changed_recently", lambda context, tickers, as_of=None: frozenset())
    monkeypatch.setattr(mod, "ProcessPoolExecutor", _CountingPool)
    for k, ticker in enumerate(("AAA", "BBB", "CCC", "DDD")):
        facts = _quarters(ticker)
        sqlite_store.save(Tables.sec_fundamentals_facts, facts.assign(value=facts["value"] * (k + 1)))

    _CountingPool.submitted = []
    serial_history = _pool_run(sqlite_store, workers=1)
    assert _CountingPool.submitted == [], "workers=1 must build in-process"
    pooled_history = _pool_run(sqlite_store, workers=2)

    assert _CountingPool.submitted == ["AAA", "CCC", "DDD"], _CountingPool.submitted
    pd.testing.assert_frame_equal(pooled_history, serial_history)
    per_ticker = pooled_history.groupby("ticker").size().to_dict()
    assert per_ticker == {"AAA": 4, "BBB": 4, "CCC": 4, "DDD": 4}, per_ticker
    print("\n=== SANITY CHECK: replay pool == in-process ===")
    print(f"  workers=2 submitted {_CountingPool.submitted} to a real process pool (BBB stayed incremental in the parent);")
    print(f"  {len(pooled_history)} history rows identical to workers=1. Validated.")


# --------------------------------------------------------------------------- #
# S1b a late filing rewrites only its ticker's rows from its date              #
# --------------------------------------------------------------------------- #
def test_a_late_10q_rewrites_only_its_tickers_rows_from_its_date(sqlite_store: Any, snapshots: list[str], caplog: pytest.LogCaptureFixture) -> None:
    """Q2 is stored after Q3 and the 10-K were published: rows from its date are recomputed (the 10-K's TTM needs
    it), rows before it and the other ticker stay byte-identical, and the next night skips."""
    q2, base = _quarters(windows=_WINDOWS[1:2]), _quarters(windows=[_WINDOWS[0], *_WINDOWS[2:]])
    sqlite_store.save(Tables.sec_fundamentals_facts, base)
    sqlite_store.save(Tables.sec_fundamentals_facts, _quarters("OTH"))
    mod.build_fundamentals_history(_context(sqlite_store), ["OTH", "TST"])
    before, other = _stored(sqlite_store, "TST"), _stored(sqlite_store, "OTH")
    late = pd.Timestamp(q2["filing_date"].iloc[0])

    sqlite_store.save(Tables.sec_fundamentals_facts, q2)
    snapshots.clear()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        mod.build_fundamentals_history(_context(sqlite_store), ["OTH", "TST"])
    after = _stored(sqlite_store, "TST")
    rewritten = snapshots.copy()

    kept = pd.to_datetime(before["as_of"]) < late
    # An all-NULL column reads back as object (None) on SQLite, so the before-frame is compared cell by cell, exactly.
    earlier = after[pd.to_datetime(after["as_of"]) < late].reset_index(drop=True)
    assert list(earlier.columns) == list(before.columns) and len(earlier) == int(kept.sum()) == 1
    assert mod.diff_against_stored(before[kept], earlier).empty
    pd.testing.assert_frame_equal(_stored(sqlite_store, "OTH"), other)
    _assert_stored_equals(sqlite_store, pd.concat([base, q2]))
    assert late in set(pd.to_datetime(after["as_of"])), "the late filing's own row is missing"
    assert f"TST late filing {late.date()}" in caplog.text, caplog.text
    ttm = after["totalRevenue"].iloc[-1], before["totalRevenue"].iloc[-1]

    snapshots.clear()
    mod.build_fundamentals_history(_context(sqlite_store), ["OTH", "TST"])
    assert snapshots == [], f"the night after the rewrite recomputed {snapshots}"
    pd.testing.assert_frame_equal(_stored(sqlite_store, "TST"), after)

    print("\n=== SANITY CHECK: late 10-Q (S1b) ===")
    print(
        f"  late Q2 filed {late.date()} under a history ending {pd.Timestamp(before['as_of'].max()).date()}: {len(rewritten)} snapshots (full replay)"
    )
    print(
        f"  rows before it byte-identical ({int(kept.sum())}), OTH byte-identical ({len(other)}), {len(after) - int(kept.sum())} row(s) from it rewritten"
    )
    print(f"  10-K TTM revenue {ttm[1]} -> {ttm[0]}; the next night: 0 snapshots. Validated.")


@pytest.mark.parametrize("workers", [1, 2])
def test_a_ticker_whose_drift_guard_raises_does_not_stop_the_others(
    sqlite_store: Any, snapshots: list[str], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, workers: int
) -> None:
    """AAA's late original meets a changed stored row before it: AAA is logged and left untouched, BBB (late) and
    CCC (incremental) are still built, and the task then raises one error naming AAA only."""
    monkeypatch.setattr(mod, "ProcessPoolExecutor", _CountingPool)
    tickers = ["AAA", "BBB", "CCC"]
    for ticker in tickers:
        sqlite_store.save(Tables.sec_fundamentals_facts, _quarters(ticker, _WINDOWS[:3]))
    mod.build_fundamentals_history(_context(sqlite_store), tickers)
    _tamper(sqlite_store, "AAA", "2023-05-01", totalAssets=999.0)
    tampered = _stored(sqlite_store, "AAA")
    for ticker in ("AAA", "BBB"):
        late = _quarters(ticker, [("2023-01-01", "2023-03-31", "Q1", "2023-09-15")]).assign(accession_number=f"{ticker}-late")
        sqlite_store.save(Tables.sec_fundamentals_facts, late)
    sqlite_store.save(Tables.sec_fundamentals_facts, _quarters("CCC", _WINDOWS[3:]))

    _CountingPool.submitted = []
    context = SimpleNamespace(**vars(_context(sqlite_store)), config=SimpleNamespace(data_extract=SimpleNamespace(fundamentals_workers=workers)))
    with caplog.at_level(logging.INFO, logger=LOGGER), pytest.raises(ValueError, match="append-only") as raised:
        mod.build_fundamentals_history(context, tickers)
    built = {ticker: len(_stored(sqlite_store, ticker)) for ticker in tickers}

    assert built == {"AAA": 3, "BBB": 4, "CCC": 4}, built
    pd.testing.assert_frame_equal(_stored(sqlite_store, "AAA"), tampered)
    assert "1 ticker(s)" in str(raised.value) and "AAA" in str(raised.value) and "BBB" not in str(raised.value), raised.value
    assert "BBB late filing 2023-09-15" in caplog.text and "CCC +1 event row(s)" in caplog.text, caplog.text
    assert _CountingPool.submitted == ([] if workers == 1 else ["AAA", "BBB"]), _CountingPool.submitted
    print(f"\n=== SANITY CHECK: a per-ticker drift guard failure (workers={workers}) ===")
    print(f"  stored rows after the run: {built}; AAA untouched (its tampered row kept), pool submissions {_CountingPool.submitted}")
    print(f"  the task raised once at the end: {str(raised.value).splitlines()[0]}. Validated.")


def test_a_stored_event_the_replay_no_longer_consumes_is_checked(sqlite_store: Any, snapshots: list[str], caplog: pytest.LogCaptureFixture) -> None:
    """A stored `as_of` whose filing the facts no longer hold, with no new date, routes to `check`, not `skip`: the full
    replay then meets the 10-K row whose TTM needed that filing, and the drift guard refuses it."""
    sqlite_store.save(Tables.sec_fundamentals_facts, _quarters())
    _built(sqlite_store, snapshots)
    stored = _stored(sqlite_store, "TST")
    sqlite_store.delete(Tables.sec_fundamentals_facts, {"accession_number": ["TST-acc-2023-08-01"]})
    caplog.clear()  # the setup build's triage line is captured too when the root logger is already at INFO

    with caplog.at_level(logging.INFO, logger=LOGGER):
        try:
            mod.build_fundamentals_history(_context(sqlite_store), ["TST"])
            refused = ""
        except ValueError as exc:
            refused = str(exc)
    triage = next(line for line in caplog.messages if line.startswith("history: triage"))

    assert "1 check" in triage and "0 current" in triage, triage
    assert "append-only" in refused, refused
    pd.testing.assert_frame_equal(_stored(sqlite_store, "TST"), stored)
    print("\n=== SANITY CHECK: a stored event the replay no longer consumes (R2-C-004) ===")
    print(f"  4 rows stored, the 2023-08-01 filing's facts gone, no new date -> {triage}")
    print(f"  full replay refused, rows untouched: {refused.splitlines()[0]}. Validated.")


PROLOGIS, OLD_PROLOGIS = "0001045609", "0000899881"


def test_a_ticker_whose_seam_rule_sets_filings_aside_is_skipped_when_nothing_is_new(
    sqlite_store: Any, snapshots: list[str], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """PLD shape: an event-only CIK's filings (one inside the history, one after its newest row) are stored but
    never feed the history, so on a night with nothing new the triage skips PLD without reading its facts."""
    identity = dated_identity(
        [("PLD", PROLOGIS, "cik_window", SENTINEL, None), ("PLD", OLD_PROLOGIS, "cik_event", SENTINEL, None)], {"PLD": PROLOGIS}
    )
    monkeypatch.setattr(mod, "load_identity", lambda context: identity)
    own = _quarters("PLD").assign(cik=PROLOGIS)
    aside = _quarters("PLD", [_WINDOWS[0][:3] + ("2023-06-15",), _WINDOWS[3][:3] + ("2024-03-01",)]).assign(cik=OLD_PROLOGIS)
    sqlite_store.save(Tables.sec_fundamentals_facts, pd.concat([own, aside.assign(accession_number=lambda d: "old-" + d["accession_number"])]))
    mod.build_fundamentals_history(_context(sqlite_store), ["PLD"])
    stored = _stored(sqlite_store, "PLD")
    assert len(stored) == 4, stored["as_of"].tolist()

    snapshots.clear()
    caplog.clear()
    seen = _spy_loads(monkeypatch, sqlite_store)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        mod.build_fundamentals_history(_context(sqlite_store), ["PLD"])
    triage = next(line for line in caplog.messages if line.startswith("history: triage"))

    assert "1 current (skipped" in triage, triage
    assert snapshots == [] and _full_facts_reads(seen) == 0, (snapshots, _full_facts_reads(seen))
    pd.testing.assert_frame_equal(_stored(sqlite_store, "PLD"), stored)
    print("\n=== SANITY CHECK: seam-rule ticker on a night with nothing new (R-C-004) ===")
    print(f"  PLD: 4 own events stored, 2 event-only-CIK filings set aside (2023-06-15, 2024-03-01); {triage},")
    print("  0 snapshots, 0 full facts reads, history unchanged. Validated.")
