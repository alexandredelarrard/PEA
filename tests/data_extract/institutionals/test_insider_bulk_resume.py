"""Insider unit completion: normal missing-only ingest and explicit bounded subset recovery."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common.identity import load_source_identity
from src.data_extract.utils.zips import fetch_insider_transactions as ins
from src.data_store.schema import Tables
from src.utils.window import WindowRun
from tests.canonical_schema import create_canonical
from tests.data_extract.fake_context import extract_config
from tests.support.universe import seed_universe

_AS_OF = pd.Timestamp("2024-07-01")
_CIKS = {"AAA": "0000000001", "NEW": "0000000002"}


def _context(store: Any, tmp_path: Path) -> Any:
    return SimpleNamespace(
        store=store,
        paths={"DATA_STORE": tmp_path},
        config=extract_config(),
        run=WindowRun("incremental", "insider unit fixture", _AS_OF),
    )


def _row(ticker: str, quarter: str) -> dict[str, object]:
    return {
        "accession_number": f"{ticker}-{quarter}",
        "security_type": "nonderiv",
        "row_sequence": 1,
        "ticker": ticker,
        "issuer_cik": _CIKS.get(ticker),
        "quarter": quarter,
        "source": "zip",
        "filing_date": pd.Period(quarter, freq="Q").start_time,
    }


def _parsed(tickers: list[str], quarter: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """What one quarter's parse returns: a kept row, a rejected row claiming the ticker, and a footnote per ticker."""
    kept = pd.DataFrame([_row(t, quarter) for t in tickers])
    rejected = pd.DataFrame(
        [{"accession_number": f"X-{t}-{quarter}", "transaction_code": "P", "claimed_ticker": t, "reject_reason": "entity_mismatch"} for t in tickers]
    )
    notes = pd.DataFrame({"accession_number": [f"{t}-{quarter}" for t in tickers], "footnote_id": "F1", "footnote_text": "x"})
    return kept, rejected, notes


@pytest.fixture
def harness(sqlite_store: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    create_canonical(sqlite_store.engine, [Tables.insider_transactions, Tables.insider_footnotes])
    calls: dict[str, Any] = {"screens": []}
    monkeypatch.setattr(ins, "load_identity", lambda context: SimpleNamespace(source_cohort=load_source_identity(context.store)))
    monkeypatch.setattr(ins, "quarter_periods", lambda *args: ["2024q1", "2024q2"])
    monkeypatch.setattr(ins, "ensure_zip", lambda context, path, url, **kwargs: path)
    monkeypatch.setattr(ins, "_read_tables", lambda path: (path.stem,))

    def parse(
        tables: tuple[str], quarter: str, universe: list[str], identity: object, fetched_at: object
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        calls["screens"].append((quarter, list(universe)))
        return _parsed(list(universe), quarter)

    def log_exclusions(label: str, frames: list[pd.DataFrame]) -> None:
        calls["excluded"] = sorted(a for frame in frames for a in frame["accession_number"])

    monkeypatch.setattr(ins, "_parse_quarter", parse)
    monkeypatch.setattr(ins, "log_exclusions", log_exclusions)
    monkeypatch.setattr(ins, "report_zip_quarter", lambda context, quarter, df_kept: calls.setdefault("reported", []).append(quarter))
    ctx = _context(sqlite_store, tmp_path)
    (tmp_path / "sec_insider_transactions").mkdir()
    (tmp_path / "sec_insider_transactions" / "2024q1.zip").write_bytes(b"cached")
    return {"ctx": ctx, "store": sqlite_store, "calls": calls}


def test_a_new_ticker_does_not_reopen_completed_quarters(harness: dict[str, Any]) -> None:
    store, calls, context = harness["store"], harness["calls"], harness["ctx"]
    context.universe = seed_universe(store, ["AAA", "NEW"], rows={"NEW": {"universe_added_on": pd.Timestamp("2024-06-28")}})
    store.save(Tables.insider_transactions, pd.DataFrame([_row("AAA", "2024q1")]))
    cache = context.paths["DATA_STORE"] / "sec_insider_transactions"
    (cache / "2024q2.zip").write_bytes(b"cached")
    saved = ins.fetch_insider_transactions(context, ["AAA", "NEW"])
    rows = store.load(Tables.insider_transactions)
    assert calls["screens"] == [("2024q2", ["AAA", "NEW"])]
    assert saved == 2 and sorted(rows.accession_number) == ["AAA-2024q1", "AAA-2024q2", "NEW-2024q2"]
    assert calls["excluded"] == ["X-AAA-2024q2", "X-NEW-2024q2"]
    assert sorted(store.load(Tables.insider_footnotes).accession_number) == ["AAA-2024q2", "NEW-2024q2"]
    assert calls["reported"] == ["2024q2"]
    (cache / "2024q1.zip").unlink()
    assert ins.fetch_insider_transactions(context, ["AAA", "NEW"]) == 0
    assert calls["screens"] == [("2024q2", ["AAA", "NEW"])]
    pd.testing.assert_frame_equal(store.load(Tables.insider_transactions), rows)
    print(
        "SANITY: new roster keys and a missing completed archive on disk do not re-ingest stored quarters; only the missing source unit parses once."
    )


def test_scoped_recovery_preserves_other_keys_and_holds_missing_global_units(harness: dict[str, Any]) -> None:
    store, calls, context = harness["store"], harness["calls"], harness["ctx"]
    context.universe = seed_universe(store, ["AAA", "BBB"])
    store.save(Tables.insider_transactions, pd.DataFrame([_row("AAA", "2024q1"), _row("BBB", "2024q1")]))
    before = store.load(Tables.insider_transactions, where={"ticker": "BBB"})
    assert ins.fetch_insider_transactions(context, ["AAA"]) == 0 and calls["screens"] == []
    context.run = WindowRun("full", "explicit scoped unit recovery", context.run.instant)
    assert ins.fetch_insider_transactions(context, ["AAA"]) == 1
    assert calls["screens"] == [("2024q1", ["AAA", "BBB"])]
    pd.testing.assert_frame_equal(store.load(Tables.insider_transactions, where={"ticker": "BBB"}), before)
    assert sorted(store.load(Tables.insider_transactions).ticker) == ["AAA", "BBB"]
    print("SANITY: scoped normal work holds globally missing units; explicit recovery screens the whole universe and writes only its requested key.")
