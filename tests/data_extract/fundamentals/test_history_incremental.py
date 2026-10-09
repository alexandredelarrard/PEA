"""Incremental `fundamentals_history_sec`: triage before replay, new events only, full replay when the past moved.

Synthetic where the question is which path a ticker takes (a known-truth fixture names the expected path), real
where the question is whether incremental rows equal a full replay (the sweep ledgers in `data/fundamentals_sweep/`
are genuine `fundamentals_facts` rows). The store is the in-memory SQLite double, never the live tables.
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
from src.data_extract.utils.common.edgar_driver import FetchSummary, KeyOutcome
from src.data_extract.utils.common.resume import DocumentWork
from src.data_extract.utils.fundamentals import build_history as mod
from src.data_extract.utils.fundamentals.build_history import FACT_COLUMNS
from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import fetched_filing_dates
from src.data_store.schema import Tables
from tests.data_extract.common.scope_fixtures import SENTINEL, dated_identity

LOGGER = "test.history_incremental"
CATALOGUE_DIR = "./configs"


def _fact(**kwargs: Any) -> dict:
    """One `fundamentals_facts` row with every replay column defaulted."""
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
        "is_amendment": False,
        "period_of_report": pd.Timestamp("2023-03-31"),
        "regime": "industrial",
        "period_start": pd.Timestamp("2023-01-01"),
        "period_end": pd.Timestamp("2023-03-31"),
        "period_days": 89.0,
        "value": 100.0,
        "unit": "USD",
        "source_concept": "us-gaap:Revenues",
        "dc_code": None,
        "adjustment": None,
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


@pytest.fixture
def snapshots(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every `_snapshot` call's `as_of`, in order; the lineage window reads as unchanged for every ticker."""
    calls: list[str] = []
    real = mod._snapshot

    def counted(ticker: str, visible: pd.DataFrame, event: pd.Series, *args: Any, **kwargs: Any) -> tuple[dict, list[dict]]:
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
    return sum(1 for table, columns in seen if table == Tables.fundamentals_facts.name and columns == tuple(FACT_COLUMNS))


# --------------------------------------------------------------------------- #
# T1 (AC-001) an unchanged ticker costs no snapshot                            #
# --------------------------------------------------------------------------- #
def test_unchanged_ticker_computes_no_snapshot(sqlite_store: Any, snapshots: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    sqlite_store.save(Tables.fundamentals_facts, _quarters())
    context = _context(sqlite_store)
    mod.build_fundamentals_history(context, ["TST"])
    first = len(snapshots)
    stored = sqlite_store.row_count(Tables.fundamentals_history_sec)
    assert first == 4 and stored == 4, (first, stored)

    snapshots.clear()
    seen = _spy_loads(monkeypatch, sqlite_store)
    mod.build_fundamentals_history(context, ["TST"])

    assert snapshots == [], f"{len(snapshots)} snapshot(s) recomputed for a ticker with no new filing: {snapshots}"
    assert sqlite_store.row_count(Tables.fundamentals_history_sec) == stored
    assert _full_facts_reads(seen) == 0, "a skipped ticker read its full facts"
    print("\n=== SANITY CHECK: unchanged ticker ===")
    print(f"  first build: {first} snapshots, {stored} rows; re-run with no new filing: 0 snapshots, 0 rows, 0 full facts reads. Validated.")


# --------------------------------------------------------------------------- #
# helpers on the store                                                         #
# --------------------------------------------------------------------------- #
def _stored(store: Any, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The ticker's stored history and reason codes, in a deterministic order."""
    history = store.load(Tables.fundamentals_history_sec, where={"ticker": ticker}, optional=True)
    codes = store.load(Tables.fundamentals_reason_codes, where={"ticker": ticker}, optional=True)
    history = pd.DataFrame() if history is None else history.sort_values("as_of").reset_index(drop=True)
    codes = pd.DataFrame() if codes is None else codes.sort_values(["as_of", "field", "dc_code"]).reset_index(drop=True)
    return history, codes


def _keep_first(store: Any, ticker: str, history: pd.DataFrame, codes: pd.DataFrame, n_rows: int) -> pd.Timestamp:
    """Rewrite the ticker's stored rows as the first `n_rows` events of `history`; returns the newest kept `as_of`."""
    store.delete(Tables.fundamentals_history_sec, {"ticker": ticker})
    store.delete(Tables.fundamentals_reason_codes, {"ticker": ticker})
    kept = history.iloc[:n_rows]
    newest = pd.Timestamp(kept["as_of"].max())
    store.save(Tables.fundamentals_history_sec, kept)
    store.save(Tables.fundamentals_reason_codes, codes[pd.to_datetime(codes["as_of"]) <= newest])
    return newest


def _tamper(store: Any, ticker: str, as_of: str, **cells: Any) -> None:
    """Overwrite cells of one stored history row in place (an upsert on its key)."""
    history, _ = _stored(store, ticker)
    row = history[pd.to_datetime(history["as_of"]) == pd.Timestamp(as_of)].copy()
    assert len(row) == 1, as_of
    store.save(Tables.fundamentals_history_sec, row.assign(**cells))


def _built(store: Any, snapshots: list[str], ticker: str = "TST") -> int:
    """Build `ticker` once from its stored facts (first build), clear the snapshot log, return the stored row count."""
    mod.build_fundamentals_history(_context(store), [ticker])
    snapshots.clear()
    return len(_stored(store, ticker)[0])


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
        pytest.skip(f"{path} not present -- run scripts/sweep_fundamentals_resolution.py")
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
    sqlite_store.save(Tables.fundamentals_facts, facts)
    _built(sqlite_store, snapshots, ticker)
    full_history, full_codes = _stored(sqlite_store, ticker)
    events = len(full_history)
    assert events > 8, f"{ticker}: only {events} events in the window"

    report = []
    for dropped in (1, 4):
        newest = _keep_first(sqlite_store, ticker, full_history, full_codes, events - dropped)
        snapshots.clear()
        mod.build_fundamentals_history(_context(sqlite_store), [ticker])
        history, codes = _stored(sqlite_store, ticker)
        assert len(snapshots) == dropped, f"{ticker}: {len(snapshots)} snapshots for {dropped} new event(s)"
        assert all(pd.Timestamp(s.split("@")[1]) > newest for s in snapshots), snapshots
        pd.testing.assert_frame_equal(history, full_history)
        pd.testing.assert_frame_equal(codes, full_codes)
        report.append(f"drop {dropped}: {len(snapshots)} snapshot(s); {len(history)} rows and {len(codes)} codes cell-identical")

    print(f"\n=== SANITY CHECK: incremental == full replay ({ticker}, {len(facts)} facts filed <= {until}) ===")
    print(f"  full replay: {events} events, {len(full_codes)} reason codes, {int(full_history['is_amendment'].astype(bool).sum())} amendment rows")
    for line in report:
        print(f"  {line}")
    print("  Each snapshot reads only its own facts prefix, so the new rows equal the full replay's. Validated.")


# --------------------------------------------------------------------------- #
# T3 (AC-003) a back-dated fetch forces the full replay                        #
# --------------------------------------------------------------------------- #
def test_backdated_fetch_forces_full_replay(sqlite_store: Any, snapshots: list[str]) -> None:
    facts = _quarters()
    sqlite_store.save(Tables.fundamentals_facts, facts)
    stored = _built(sqlite_store, snapshots)
    context = _context(sqlite_store)

    mod.build_fundamentals_history(context, ["TST"], fetched={"TST": [pd.Timestamp("2023-08-01")]})
    backdated = len(snapshots)
    snapshots.clear()
    mod.build_fundamentals_history(context, ["TST"], fetched={"TST": []}, full_fetch=True)
    forced = len(snapshots)
    snapshots.clear()
    mod.build_fundamentals_history(context, ["TST"], fetched={"TST": [pd.Timestamp("2024-05-01")]})
    later = len(snapshots)
    assert backdated == forced == stored == 4 and later == 0, (backdated, forced, later)
    assert sqlite_store.row_count(Tables.fundamentals_history_sec) == 4

    # A re-read filing that now carries another value for an already-published row: the drift guard refuses.
    q2 = facts[(facts["filing_date"] == pd.Timestamp("2023-08-01")) & (facts["field"] == "totalAssets")].assign(value=5000.0)
    sqlite_store.save(Tables.fundamentals_facts, q2)
    snapshots.clear()
    mod.build_fundamentals_history(context, ["TST"])
    silent = len(snapshots)
    with pytest.raises(ValueError, match="append-only"):
        mod.build_fundamentals_history(context, ["TST"], fetched={"TST": [pd.Timestamp("2023-08-01")]})
    assert silent == 0, "without the hand-off a same-date re-read is invisible to triage (the accepted -F residual)"

    print("\n=== SANITY CHECK: back-dated fetch ===")
    print(f"  fetched 2023-08-01 (<= newest 2024-02-15): {backdated} snapshots; -F: {forced}; fetched a later date with no new facts: {later}")
    print("  re-read 2023-08-01 with a changed value: with the hand-off -> drift error (append-only); without it -> 0 snapshots (D4 residual).")
    print("  Validated.")


# --------------------------------------------------------------------------- #
# T4 (AC-004) an event-list mismatch forces the full replay                    #
# --------------------------------------------------------------------------- #
def test_event_list_mismatch_forces_full_replay(sqlite_store: Any, snapshots: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    sqlite_store.save(Tables.fundamentals_facts, _quarters())
    _built(sqlite_store, snapshots)
    context = _context(sqlite_store)

    # (a) Stored provenance disagrees with the rebuilt event list; lineage-recent -> check -> full -> rebuilt.
    _tamper(sqlite_store, "TST", "2023-08-01", publication_form="10-K")
    monkeypatch.setattr(mod, "_scope_changed_recently", lambda context, tickers, as_of=None: frozenset({"TST"}))
    mod.build_fundamentals_history(context, ["TST"])
    provenance = len(snapshots)
    history, _ = _stored(sqlite_store, "TST")
    form = history.loc[pd.to_datetime(history["as_of"]) == pd.Timestamp("2023-08-01"), "publication_form"].iloc[0]
    assert provenance == 4 and form == "10-Q", (provenance, form)

    # (b) A back-dated original with no fetch hand-off: its date is not stored -> check -> full; it adds its own row.
    monkeypatch.setattr(mod, "_scope_changed_recently", lambda context, tickers, as_of=None: frozenset())
    late = _quarters(windows=[("2023-01-01", "2023-03-31", "Q1", "2023-06-15")]).assign(accession_number="TST-late")
    sqlite_store.save(Tables.fundamentals_facts, late)
    snapshots.clear()
    mod.build_fundamentals_history(context, ["TST"])
    backfilled = len(snapshots)
    assert backfilled == 5 and sqlite_store.row_count(Tables.fundamentals_history_sec) == 5, backfilled

    # (c) A back-dated original that restates a published quarter: the full replay's drift guard refuses.
    sqlite_store.save(Tables.fundamentals_facts, late.assign(accession_number="TST-later", filing_date=pd.Timestamp("2023-09-15"), value=1.0))
    with pytest.raises(ValueError, match="append-only"):
        mod.build_fundamentals_history(context, ["TST"])

    print("\n=== SANITY CHECK: event-list mismatch ===")
    print(f"  tampered stored publication_form (lineage-recent): {provenance} snapshots, row rebuilt to {form!r}")
    print(f"  back-dated original 2023-06-15, no hand-off: {backfilled} snapshots, its row added (5 rows)")
    print("  back-dated original that moves a published row: drift error (append-only). Validated.")


# --------------------------------------------------------------------------- #
# T5 (AC-005) the triage classifies every ticker before any replay            #
# --------------------------------------------------------------------------- #
def test_triage_classifies_tickers(sqlite_store: Any, snapshots: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    names = ("NEW", "GROW", "CUR", "RECENT", "FETCH", "MOVED")
    for name in names:
        sqlite_store.save(Tables.fundamentals_facts, _quarters(name, _WINDOWS[:3]))
    mod.build_fundamentals_history(_context(sqlite_store), [n for n in names if n != "NEW"])
    sqlite_store.save(Tables.fundamentals_facts, _quarters("GROW", _WINDOWS[3:]))
    sqlite_store.save(Tables.fundamentals_facts, _quarters("MOVED", [("2023-01-01", "2023-03-31", "Q1", "2023-06-15")]))

    seen = _spy_loads(monkeypatch, sqlite_store)
    context = _context(sqlite_store)
    work = mod._history_work(context, list(names), fetched={"FETCH": [pd.Timestamp("2023-08-01")], "CUR": []}, recent=frozenset({"RECENT"}))
    paths = {ticker: item.path for ticker, item in work.items()}
    verify = {ticker: item.path for ticker, item in mod._history_work(context, list(names), verify=True).items()}

    assert paths == {"NEW": "full", "GROW": "incremental", "CUR": "skip", "RECENT": "check", "FETCH": "full", "MOVED": "check"}, paths
    assert set(verify.values()) == {"full"}
    assert work["GROW"].newest == pd.Timestamp("2023-11-01") and work["NEW"].newest is None
    assert _full_facts_reads(seen) == 0, "the triage read a ticker's full facts"
    narrow = {columns for table, columns in seen if table == Tables.fundamentals_facts.name}
    assert narrow == {("ticker", "filing_date", "is_amendment")}, narrow

    print("\n=== SANITY CHECK: triage ===")
    for ticker, path in paths.items():
        print(f"  {ticker:6s} -> {path}")
    print(f"  verify -> {sorted(set(verify.values()))}; facts reads: {sorted(narrow)} only, no full facts read. Validated.")


# --------------------------------------------------------------------------- #
# T6 (AC-006) verify and rebuild keep today's full-replay semantics            #
# --------------------------------------------------------------------------- #
def test_verify_history_replays_in_full_under_the_drift_guard(sqlite_store: Any, snapshots: list[str]) -> None:
    sqlite_store.save(Tables.fundamentals_facts, _quarters())
    _built(sqlite_store, snapshots)
    context = _context(sqlite_store)

    mod.build_fundamentals_history(context, ["TST"], verify_history=True)
    verified = len(snapshots)
    assert verified == 4 and sqlite_store.row_count(Tables.fundamentals_history_sec) == 4

    _tamper(sqlite_store, "TST", "2023-08-01", totalAssets=999.0)
    snapshots.clear()
    mod.build_fundamentals_history(context, ["TST"])
    assert snapshots == [], "a tampered stored value is not a triage signal"
    with pytest.raises(ValueError, match="append-only"):
        mod.build_fundamentals_history(context, ["TST"], verify_history=True)

    print("\n=== SANITY CHECK: --verify-history ===")
    print(f"  current ticker: {verified} snapshots, 0 rows written; a tampered stored cell: daily run 0 snapshots, verify -> drift error. Validated.")


def test_rebuild_history_deletes_and_replays_in_full(sqlite_store: Any, snapshots: list[str], caplog: pytest.LogCaptureFixture) -> None:
    sqlite_store.save(Tables.fundamentals_facts, _quarters())
    _built(sqlite_store, snapshots)
    _tamper(sqlite_store, "TST", "2023-08-01", totalAssets=999.0)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        mod.build_fundamentals_history(_context(sqlite_store), ["TST"], rebuild_history=True)

    history, _ = _stored(sqlite_store, "TST")
    restored = history.loc[pd.to_datetime(history["as_of"]) == pd.Timestamp("2023-08-01"), "totalAssets"].iloc[0]
    assert len(snapshots) == 4 and restored == 1001.0 and len(history) == 4, (len(snapshots), restored)
    assert "TST REBUILT -- 4 row(s) deleted and recomputed" in caplog.text
    print("\n=== SANITY CHECK: --rebuild-history ===")
    print(f"  tampered 999 -> {restored} after delete + {len(snapshots)}-event replay; the REBUILT warning is logged. Validated.")


# --------------------------------------------------------------------------- #
# P2 (AC-003, AC-006) the `fundamentals` command hands its reads to the triage #
# --------------------------------------------------------------------------- #
def _summary(units: dict[str, list[tuple[str, str]]], failed: dict[str, list[str]]) -> FetchSummary:
    """A `FetchSummary` listing `(accession, filed)` per ticker, with `failed` accessions still missing."""
    frames = {
        ticker: pd.DataFrame(
            {"cik": "0000000001", "company": ticker, "form": "10-Q", "filed": [pd.Timestamp(f) for _, f in rows], "accession": [a for a, _ in rows]}
        )
        for ticker, rows in units.items()
    }
    outcomes = {
        ticker: KeyOutcome(listed=len(df), failed=df[df["accession"].isin(failed.get(ticker, []))].reset_index(drop=True))
        for ticker, df in frames.items()
    }
    return FetchSummary(work=DocumentWork(units=frames, key_class={t: "established" for t in frames}), outcomes=outcomes)


def test_fetched_dates_exclude_missing_documents() -> None:
    summary = _summary(
        {"AAA": [("a-1", "2023-08-01"), ("a-2", "2023-05-01"), ("a-3", "2024-02-15")], "BBB": [("b-1", "2024-01-10")], "CCC": []},
        {"AAA": ["a-3"], "BBB": ["b-1"]},
    )

    fetched = fetched_filing_dates(summary)

    assert summary.missing == {"AAA": ["a-3"], "BBB": ["b-1"]}
    assert fetched == {"AAA": [pd.Timestamp("2023-05-01"), pd.Timestamp("2023-08-01")]}, fetched
    print("\n=== SANITY CHECK: summary -> fetched ===")
    print(
        f"  AAA listed 3, a-3 missing -> {[d.date().isoformat() for d in fetched['AAA']]}; BBB all missing -> left out; CCC nothing listed -> left out."
    )
    print("  Only documents actually read reach the history triage. Validated.")


def _cli(monkeypatch: pytest.MonkeyPatch, store: Any, summary: FetchSummary | None = None) -> list[dict]:
    """Route the CLI to `store`; the fetch returns `summary` (recording its kwargs) instead of walking EDGAR."""
    fetches: list[dict] = []
    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=15))
    monkeypatch.setattr(cli_mod, "get_config_context", lambda path, **kwargs: (config, _context(store)))
    monkeypatch.setattr(cli_mod, "_tickers", lambda ctx, names: [t.strip().upper() for t in names.split(",")])
    monkeypatch.setattr(cli_mod, "fetch_fundamentals_sec", lambda context, **kwargs: fetches.append(kwargs) or summary)
    return fetches


def test_cli_backdated_fetch_reaches_the_history_as_a_full_replay(sqlite_store: Any, snapshots: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    sqlite_store.save(Tables.fundamentals_facts, _quarters())
    stored = _built(sqlite_store, snapshots)
    runner = CliRunner()
    report = []
    cases = (
        ("back-dated read", _summary({"TST": [("TST-acc-2023-08-01", "2023-08-01")]}, {}), [], stored),
        ("back-dated read, missing", _summary({"TST": [("TST-acc-2023-08-01", "2023-08-01")]}, {"TST": ["TST-acc-2023-08-01"]}), [], 0),
        ("nothing read", _summary({}, {}), [], 0),
        ("-F", _summary({"TST": [("TST-acc-2024-02-15", "2024-02-15")]}, {}), ["-F"], stored),
    )
    for label, summary, extra, expected in cases:
        fetches = _cli(monkeypatch, sqlite_store, summary)
        snapshots.clear()
        result = runner.invoke(cli_mod.cli, ["fundamentals", "-t", "tst", *extra], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert len(snapshots) == expected, (label, snapshots)
        assert fetches[0]["full"] is bool(extra), fetches
        report.append(f"{label}: {len(snapshots)} snapshot(s)")
    assert sqlite_store.row_count(Tables.fundamentals_history_sec) == stored

    # A re-read that changed a published row: through the CLI hand-off the drift guard refuses.
    facts = _quarters()
    q2 = facts[(facts["filing_date"] == pd.Timestamp("2023-08-01")) & (facts["field"] == "totalAssets")].assign(value=5000.0)
    sqlite_store.save(Tables.fundamentals_facts, q2)
    _cli(monkeypatch, sqlite_store, _summary({"TST": [("TST-acc-2023-08-01", "2023-08-01")]}, {}))
    with pytest.raises(ValueError, match="append-only"):
        runner.invoke(cli_mod.cli, ["fundamentals", "-t", "TST"], catch_exceptions=False)

    print("\n=== SANITY CHECK: CLI hand-off (stored newest 2024-02-15) ===")
    for line in report:
        print(f"  {line}")
    print("  changed re-read handed off by the CLI -> drift error (append-only). Validated.")


def test_verify_history_flag_reaches_the_history_build(monkeypatch: pytest.MonkeyPatch) -> None:
    builds: list[dict] = []
    _cli(monkeypatch, None, _summary({}, {}))
    monkeypatch.setattr(cli_mod, "build_fundamentals_history", lambda context, **kwargs: builds.append(kwargs))
    runner = CliRunner()
    for args in (
        ["fundamentals", "-t", "AAA"],
        ["fundamentals", "-t", "AAA", "--verify-history"],
        ["fundamentals-history-sec", "-t", "AAA", "--verify-history"],
    ):
        result = runner.invoke(cli_mod.cli, args)
        assert result.exit_code == 0, result.output
    helps = {name: runner.invoke(cli_mod.cli, [name, "--help"]).output for name in ("fundamentals", "fundamentals-history-sec", "fundamentals-facts")}

    assert [b["verify_history"] for b in builds] == [False, True, True], builds
    assert builds[0]["fetched"] == {} and builds[0]["full_fetch"] is False and builds[0]["rebuild_history"] is False
    assert "fetched" not in builds[2], "fundamentals-history-sec has no fetch to hand off"
    assert "--verify-history" in helps["fundamentals"] and "--verify-history" in helps["fundamentals-history-sec"]
    assert "--verify-history" not in helps["fundamentals-facts"]
    print("\n=== SANITY CHECK: --verify-history wiring ===")
    print(
        f"  verify_history per call: {[b['verify_history'] for b in builds]}; both history commands list the flag, fundamentals-facts does not. Validated."
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


def _pool_run(store: Any, workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three first builds and one incremental ticker (BBB, 3 of 4 events stored) with `workers`; the saved tables."""
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    store.delete(Tables.fundamentals_history_sec, {"ticker": tickers})
    store.delete(Tables.fundamentals_reason_codes, {"ticker": tickers})
    serial = _context(store)
    mod.build_fundamentals_history(serial, ["BBB"])
    history, codes = _stored(store, "BBB")
    _keep_first(store, "BBB", history, codes, 3)
    context = SimpleNamespace(**vars(serial), config=SimpleNamespace(data_extract=SimpleNamespace(fundamentals_workers=workers)))
    mod.build_fundamentals_history(context, tickers)
    history = store.load(Tables.fundamentals_history_sec).sort_values(["ticker", "as_of"]).reset_index(drop=True)
    codes = store.load(Tables.fundamentals_reason_codes).sort_values(["ticker", "as_of", "field", "dc_code"]).reset_index(drop=True)
    return history, codes


def test_pool_build_equals_in_process_build(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_scope_changed_recently", lambda context, tickers, as_of=None: frozenset())
    monkeypatch.setattr(mod, "ProcessPoolExecutor", _CountingPool)
    for k, ticker in enumerate(("AAA", "BBB", "CCC", "DDD")):
        facts = _quarters(ticker)
        sqlite_store.save(Tables.fundamentals_facts, facts.assign(value=facts["value"] * (k + 1)))

    _CountingPool.submitted = []
    serial_history, serial_codes = _pool_run(sqlite_store, workers=1)
    assert _CountingPool.submitted == [], "workers=1 must build in-process"
    pooled_history, pooled_codes = _pool_run(sqlite_store, workers=2)

    assert _CountingPool.submitted == ["AAA", "CCC", "DDD"], _CountingPool.submitted
    pd.testing.assert_frame_equal(pooled_history, serial_history)
    pd.testing.assert_frame_equal(pooled_codes, serial_codes)
    per_ticker = pooled_history.groupby("ticker").size().to_dict()
    assert per_ticker == {"AAA": 4, "BBB": 4, "CCC": 4, "DDD": 4}, per_ticker
    print("\n=== SANITY CHECK: replay pool == in-process ===")
    print(f"  workers=2 submitted {_CountingPool.submitted} to a real process pool (BBB stayed incremental in the parent);")
    print(f"  {len(pooled_history)} history rows and {len(pooled_codes)} reason codes identical to workers=1. Validated.")
