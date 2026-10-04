"""Insider bulk resume (M3): the quarters `fetch_insider_transactions` parses come from the stored rows
alone. A new ticker re-reads every cached quarter for itself only, screened against the whole universe;
a scoped (`-t`) run parses no unstored quarter and skips the stored-row sweep."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.institutionals import fetch_insider_transactions as ins
from src.data_store.schema import Tables

_AS_OF = pd.Timestamp("2024-07-01")


def _context(store: Any, tmp_path: Path) -> Any:
    return SimpleNamespace(
        store=store,
        paths={"DATA_STORE": tmp_path},
        config=SimpleNamespace(
            local=SimpleNamespace(paths=SimpleNamespace(insider_transactions="sec_insider_transactions")),
            data_extract=SimpleNamespace(redundant_ticks=[]),
        ),
    )


def _row(ticker: str, quarter: str) -> dict[str, object]:
    return {"accession_number": f"{ticker}-{quarter}", "security_type": "nonderiv", "transaction_sk": "1", "ticker": ticker, "quarter": quarter}


def _parsed(tickers: list[str], quarter: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """What one quarter's parse returns: a kept row, a rejected row claiming the ticker, and a footnote per ticker."""
    kept = pd.DataFrame([_row(t, quarter) for t in tickers])
    quarantine = pd.DataFrame([_row(t, quarter) | {"transaction_sk": "2", "reject_reason": "entity_mismatch"} for t in tickers])
    notes = pd.DataFrame({"accession_number": [f"{t}-{quarter}" for t in tickers], "footnote_id": "F1", "footnote_text": "x"})
    return kept, quarantine, notes


@pytest.fixture
def harness(sqlite_store: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {"screens": [], "sweeps": 0}
    monkeypatch.setattr(ins, "load_identity", lambda context: None)
    monkeypatch.setattr(ins, "quarter_periods", lambda *args: ["2024q1", "2024q2"])
    monkeypatch.setattr(ins, "ensure_zip", lambda context, path, url, **kwargs: path)
    monkeypatch.setattr(ins, "_read_tables", lambda path: (path.stem,))

    def parse(tables: tuple[str], quarter: str, universe: list[str], identity: object) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        calls["screens"].append((quarter, list(universe)))
        return _parsed(list(universe), quarter)

    def sweep(context: object, tickers: list[str], identity: object) -> tuple[int, int]:
        calls["sweeps"] += 1
        return 0, 0

    monkeypatch.setattr(ins, "_parse_quarter", parse)
    monkeypatch.setattr(ins, "_screen_stored_rows", sweep)
    ctx = _context(sqlite_store, tmp_path)
    (tmp_path / "sec_insider_transactions").mkdir()
    (tmp_path / "sec_insider_transactions" / "2024q1.zip").write_bytes(b"cached")
    return {"ctx": ctx, "store": sqlite_store, "calls": calls}


def test_a_new_ticker_re_reads_cached_quarters_for_itself_only(harness: dict[str, Any]) -> None:
    store, calls = harness["store"], harness["calls"]
    store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": ["AAA", "NEW"], "added_on": pd.to_datetime(["2000-01-01", "2024-06-28"])}))
    store.save(Tables.insider_transactions, pd.DataFrame([_row("AAA", "2024q1")]))

    saved = ins.fetch_insider_transactions(harness["ctx"], ["AAA", "NEW"], years_history=15, as_of=_AS_OF)

    rows = store.load(Tables.insider_transactions)
    quarantine = store.load(Tables.insider_transactions_quarantine)
    notes = store.load(Tables.insider_footnotes)
    assert calls["screens"] == [("2024q1", ["AAA", "NEW"]), ("2024q2", ["AAA", "NEW"])]  # always the whole universe
    assert saved == 3 and sorted(rows["accession_number"]) == ["AAA-2024q1", "AAA-2024q2", "NEW-2024q1", "NEW-2024q2"]
    assert sorted(quarantine["accession_number"]) == ["AAA-2024q2", "NEW-2024q1", "NEW-2024q2"]  # 2024q1 rejects: NEW's only
    assert sorted(notes["accession_number"]) == ["AAA-2024q2", "NEW-2024q1", "NEW-2024q2"]
    assert calls["sweeps"] == 1
    print("\n=== SANITY CHECK: insider new-ticker re-read ===")
    print("  2024q2 (unstored) parsed for AAA and NEW; cached 2024q1 re-read for NEW only (its rows, rejects and footnotes);")
    print("  both screened against the whole universe; the unscoped run sweeps stored rows once. Validated.")


def test_a_scoped_run_parses_no_unstored_quarter_and_skips_the_sweep(harness: dict[str, Any]) -> None:
    store, calls = harness["store"], harness["calls"]
    store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": ["AAA", "BBB"], "added_on": pd.to_datetime(["2000-01-01", "2000-01-01"])}))
    store.save(Tables.insider_transactions, pd.DataFrame([_row("AAA", "2024q1"), _row("BBB", "2024q1")]))

    assert ins.fetch_insider_transactions(harness["ctx"], ["AAA"], years_history=15, as_of=_AS_OF) == 0
    assert calls["screens"] == [] and calls["sweeps"] == 0
    assert ins.fetch_insider_transactions(harness["ctx"], ["AAA"], years_history=15, reparse=True, as_of=_AS_OF) == 1
    assert calls["screens"] == [("2024q1", ["AAA", "BBB"])] and calls["sweeps"] == 0
    assert sorted(store.load(Tables.insider_transactions)["ticker"]) == ["AAA", "BBB"]
    print("\n=== SANITY CHECK: insider -t AAA ===")
    print("  2024q2 is left to the full run (a parse for AAA alone would mark it stored for BBB); no sweep, which would")
    print("  screen BBB out against a one-ticker universe; -t AAA --reparse re-reads stored 2024q1 for AAA only. Validated.")
