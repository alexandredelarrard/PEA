"""Predecessor vendor series (D-Q2-8, AC-115): the Sharadar fetch list adds the vendor tickers that carry a register
predecessor CIK's own series, and the merged-history build replaces the canonical ARQ rows inside that CIK's window
with them before `build_ttm`. PLD and DD are deferred to the traded-security realignment (P38, plan §13.10) in the
shipped config, so the mechanism tests switch that deferral off and one test checks the shipped behaviour.
Offline: `FakeStore`, no network, no database.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_extract.transformers import step_extract_fundamentals_sharadar as step_module
from src.data_extract.utils.fundamentals_sharadar import fetch_sharadar, merge_history
from src.data_store.schema import Tables
from tests.conftest import FakeStore

LOGGER = "test.predecessor_autofill"
URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={}&type=&dateb=&owner=include&count=40"


def _lineage() -> pd.DataFrame:
    rows = [
        ("PLD", "0000899881", "1900-01-01", "2011-06-03"),
        ("PLD", "0001045609", "2011-06-03", None),
        ("STE", "0000815065", "1900-01-01", "2015-10-31"),
        ("STE", "0001624899", "2015-10-31", "2019-02-12"),
        ("STE", "0001757898", "2019-02-12", None),
        ("XOM", "0000034088", "1900-01-01", "2026-07-01"),
        ("XOM", "0002115436", "2026-07-01", None),
    ]
    return pd.DataFrame(
        [
            {
                "entity_id": f"E-{t}",
                "canonical_ticker": t,
                "cik": c,
                "role": "cik_window",
                "symbol": "",
                "valid_from": pd.Timestamp(f),
                "valid_to": pd.Timestamp(e) if e else pd.NaT,
                "sources": "register",
            }
            for t, c, f, e in rows
        ]
    )


def _vendor_tickers() -> pd.DataFrame:
    rows = [
        ("PLD1", "0000899881", "2011-03-31"),
        ("PLD", "0001045609", "2026-06-30"),
        ("STE1", "0000815065", "2015-09-30"),
        ("XOM", "0000034088", "2026-06-30"),
    ]
    return pd.DataFrame(
        {
            "table": "fundamentals",
            "ticker": [r[0] for r in rows],
            "secfilings": [URL.format(r[1]) for r in rows],
            "lastquarter": pd.to_datetime([r[2] for r in rows]),
        }
    )


def _arq(ticker: str, labels: list[str], revenue: float) -> pd.DataFrame:
    ends = [pd.Period(q, freq="Q").end_time.normalize() for q in labels]
    return pd.DataFrame(
        {
            "ticker": ticker,
            "dimension": "ARQ",
            "calendardate": [e.date() for e in ends],  # a Postgres DATE round-trips as datetime.date
            "reportperiod": [e.date() for e in ends],
            "date": [(e + pd.Timedelta(days=40)).date() for e in ends],
            "fiscalperiod": [f"{q[:4]}-Q{q[-1]}" for q in labels],
            "revenue": revenue,
        }
    )


def _empties() -> dict[Any, pd.DataFrame]:
    """The merge's other inputs, present and empty."""
    return {
        Tables.sharadar_actions: pd.DataFrame(columns=["ticker", "action"]),
        Tables.prices_splits: pd.DataFrame(columns=["ticker", "date", "ratio"]),
        Tables.fundamentals_employees: pd.DataFrame(columns=["ticker", "as_of", "employees"]),
        Tables.fundamentals_history_sec: pd.DataFrame(columns=["ticker", "as_of"]),
    }


def _context(store: FakeStore) -> Any:
    return SimpleNamespace(store=store, log=logging.getLogger(LOGGER))


@pytest.fixture
def no_deferral(monkeypatch: pytest.MonkeyPatch) -> None:
    """The D-Q2-8 mechanism on its own: no ticker deferred to the traded-security realignment."""
    monkeypatch.setattr(fetch_sharadar, "deferred_tickers", lambda config_dir, scope: frozenset())


def test_predecessor_vendor_tickers_are_derived_not_listed(monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeStore({Tables.entity_lineage: _lineage(), Tables.sharadar_tickers: _vendor_tickers()})
    shipped = fetch_sharadar.predecessor_vendor_tickers(_context(store), ["PLD", "STE", "XOM", "AAPL"])
    monkeypatch.setattr(fetch_sharadar, "deferred_tickers", lambda config_dir, scope: frozenset())
    found = fetch_sharadar.predecessor_vendor_tickers(_context(store), ["PLD", "STE", "XOM", "AAPL"])
    print("\n=== SANITY CHECK: predecessor vendor tickers ===")
    print(f"  shipped config (PLD deferred, P38): {shipped}; no deferral: {found}")
    assert shipped == ["STE1"]
    assert found == ["PLD1", "STE1"]
    old_shape = FakeStore({Tables.entity_lineage: pd.DataFrame({"cik": ["1"], "entity_id": ["E"]}), Tables.sharadar_tickers: _vendor_tickers()})
    assert fetch_sharadar.predecessor_vendor_tickers(_context(old_shape), ["PLD"]) == []
    print("  OK: PLD1 and STE1 from sharadar_tickers x register windows; XOM's own series is canonical; an old-shape lineage gives none.")


def test_the_step_fetches_universe_and_predecessor_tickers(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], bool]] = []
    for name in ("fetch_sharadar_tickers", "fetch_sharadar_actions", "fetch_sharadar_sp500", "build_merged_history"):
        monkeypatch.setattr(step_module, name, lambda *a, **k: None)
    monkeypatch.setattr(step_module, "fetch_sharadar_fundamentals", lambda context, tickers, **k: calls.append((list(tickers), k["full"])))
    monkeypatch.setattr(step_module, "predecessor_vendor_tickers", lambda context, tickers, config_dir=None: ["PLD1", "STE1"])
    step = step_module.StepExtractFundamentalsSharadar.__new__(step_module.StepExtractFundamentalsSharadar)
    step._context = SimpleNamespace(log=logging.getLogger(LOGGER))
    step._log = logging.getLogger(LOGGER)
    step._config = SimpleNamespace(data_extract=SimpleNamespace(sharadar_years_history=30))
    step._config_dir = "./configs"
    step.run(["PLD", "STE"])
    print("\n=== SANITY CHECK: Sharadar fetch list ===")
    print(f"  {calls}")
    assert calls == [(["PLD", "STE"], False), (["PLD1", "STE1"], True)]
    print("  OK: the universe resumes; the delisted predecessor series are read over the whole window (a rowless key's resume window is recent).")


@pytest.mark.usefixtures("no_deferral")
def test_the_merge_replaces_inside_the_window_before_ttm(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    canonical = _arq("PLD", ["2010Q3", "2010Q4", "2011Q2", "2011Q3"], 164.5e6)  # AMB's series before the seam; 2011Q1 missing
    owner = _arq("PLD1", ["2010Q3", "2010Q4", "2011Q1"], 500.0e6)  # old ProLogis
    store = FakeStore(
        {
            **_empties(),
            Tables.entity_lineage: _lineage(),
            Tables.sharadar_tickers: _vendor_tickers(),
            Tables.sharadar_fundamentals: pd.concat([canonical, owner], ignore_index=True),
        }
    )
    seen: dict[str, pd.DataFrame] = {}

    def capture(sharadar_arq: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        seen["arq"] = sharadar_arq
        return pd.DataFrame()

    monkeypatch.setattr(merge_history, "build_frame", capture)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        merge_history.build_merged_history(_context(store), ["PLD"])
    arq = (
        seen["arq"].assign(quarter=lambda f: [str(pd.Period(pd.Timestamp(d), freq="Q")) for d in f["calendardate"]]).set_index("quarter").sort_index()
    )
    print("\n=== SANITY CHECK: merge-time replacement (D-Q2-8) ===")
    print(arq[["ticker", "revenue"]].to_string())
    assert arq["ticker"].eq("PLD").all() and arq.index.is_unique
    assert arq.loc[["2010Q3", "2010Q4", "2011Q1"], "revenue"].eq(500.0e6).all()
    assert arq.loc[["2011Q2", "2011Q3"], "revenue"].eq(164.5e6).all()
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "PLD1" in logged and "2011Q1" in logged and "2010Q3" in logged
    print("  OK: old ProLogis replaces AMB inside the window, 2011Q1 filled, rows after the seam untouched; each quarter logged.")


@pytest.mark.usefixtures("no_deferral")
def test_a_missing_owner_series_leaves_the_canonical_rows_and_warns(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    canonical = _arq("PLD", ["2010Q3", "2010Q4"], 164.5e6)
    store = FakeStore(
        {**_empties(), Tables.entity_lineage: _lineage(), Tables.sharadar_tickers: _vendor_tickers(), Tables.sharadar_fundamentals: canonical}
    )
    seen: dict[str, pd.DataFrame] = {}
    monkeypatch.setattr(merge_history, "build_frame", lambda arq, *a, **k: seen.setdefault("arq", arq).iloc[:0])
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        merge_history.build_merged_history(_context(store), ["PLD"])
    assert seen["arq"]["revenue"].eq(164.5e6).all() and len(seen["arq"]) == 2
    assert any("PLD1" in r.getMessage() and "not stored" in r.getMessage() for r in caplog.records)
    print("\n=== SANITY CHECK: owner series not stored ===")
    print("  OK: canonical rows unchanged and one WARNING names PLD1.")


def test_a_deferred_ticker_keeps_the_vendors_own_rows(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """P38: with the shipped config PLD's rows inside old ProLogis's window stay Sharadar's (AMB's, aligned with prices)."""
    canonical = _arq("PLD", ["2010Q3", "2010Q4", "2011Q2", "2011Q3"], 164.5e6)
    owner = _arq("PLD1", ["2010Q3", "2010Q4", "2011Q1"], 500.0e6)
    store = FakeStore(
        {
            **_empties(),
            Tables.entity_lineage: _lineage(),
            Tables.sharadar_tickers: _vendor_tickers(),
            Tables.sharadar_fundamentals: pd.concat([canonical, owner], ignore_index=True),
        }
    )
    seen: dict[str, pd.DataFrame] = {}
    monkeypatch.setattr(merge_history, "build_frame", lambda arq, *a, **k: seen.setdefault("arq", arq).iloc[:0])
    with caplog.at_level(logging.INFO, logger=LOGGER):
        merge_history.build_merged_history(_context(store), ["PLD"])
    arq = seen["arq"]
    print("\n=== SANITY CHECK: deferred predecessor series (P38) ===")
    print(arq[["ticker", "calendardate", "revenue"]].to_string(index=False))
    assert arq["ticker"].eq("PLD").all() and len(arq) == 4 and arq["revenue"].eq(164.5e6).all()
    assert not any("PLD1" in r.getMessage() for r in caplog.records)
    print("  OK: no replacement, no exchange-ratio conversion, no PLD1 log line: Sharadar's own PLD rows stay.")
