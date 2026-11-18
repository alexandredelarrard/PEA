"""Real macro levels retain their cumulative origin through a bounded source rewrite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.prices import fetch_macro as fm
from src.data_store.schema import Table, Tables
from src.data_store.store import WriteCounts
from src.utils.window import Window, WindowRun
from tests.data_extract.edgar_fixtures import fake_context

_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "macro" / "retained_yield_and_market.parquet"


def test_macro_rewrites_its_configured_tail_without_restarting_bond_levels(tmp_path, sqlite_store, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = pd.read_parquet(_FIXTURE)
    rows["date"] = pd.to_datetime(rows["date"])
    wide = rows.pivot(index="date", columns="ticker", values="close").sort_index()
    context = fake_context(tmp_path, sqlite_store, ["AAPL"])
    context.run = WindowRun("incremental", "real macro replay", pd.Timestamp("2026-11-16"))
    item = context.config.data.tables.prices_macro
    item.start_date, item.end_date = "1995-11-13", "2026-11-10"
    future = rows.groupby("ticker").tail(1).assign(date=pd.Timestamp("2026-11-12"), close=137.0)
    sqlite_store.save(Tables.prices_macro, pd.concat([rows, future], ignore_index=True))
    before = sqlite_store.load(Tables.prices_macro).sort_values(["ticker", "date"]).reset_index(drop=True)

    monkeypatch.setenv("FRED_API_KEY", "offline-fixture")
    monkeypatch.setattr(fm, "_fetch_price_leg", lambda context, since, until: wide.loc[since:until, ["equity_tr"]].dropna())
    monkeypatch.setattr(fm, "_fetch_fred_leg", lambda window: wide.loc[window.start : window.end, ["yield_10y"]].dropna())
    writes = []
    original = sqlite_store.replace_window

    def replace_window(table: Table, frame: pd.DataFrame, window: Window, where: dict[str, list[str]], **kwargs: Any) -> WriteCounts:
        writes.append((window, set(where["ticker"])))
        return original(table, frame, window, where, **kwargs)

    monkeypatch.setattr(sqlite_store, "replace_window", replace_window)
    fm.fetch_macro(context)
    after = sqlite_store.load(Tables.prices_macro).sort_values(["ticker", "date"]).reset_index(drop=True)
    expected_window = Window(pd.Timestamp("2026-11-03"), pd.Timestamp("2026-11-10"))
    assert writes and all(window == expected_window for window, keys in writes)
    outside = ~expected_window.mask(before["date"])
    pd.testing.assert_frame_equal(after.loc[outside].reset_index(drop=True), before.loc[outside].reset_index(drop=True))
    after["date"] = pd.to_datetime(after["date"])
    bond = after.loc[after["ticker"].eq("bond_10y_tr")].set_index("date")["close"]
    expected = wide["bond_10y_tr"].dropna()
    pd.testing.assert_series_equal(bond.loc["2026-11-02":"2026-11-10"], expected.loc["2026-11-02":"2026-11-10"], check_names=False)
    pd.testing.assert_series_equal(
        bond.loc["2026-11-02":"2026-11-10"].pct_change(), expected.loc["2026-11-02":"2026-11-10"].pct_change(), check_names=False
    )

    fm.fetch_macro(context)
    repeated = sqlite_store.load(Tables.prices_macro).sort_values(["ticker", "date"]).reset_index(drop=True)
    repeated["date"] = pd.to_datetime(repeated["date"])
    pd.testing.assert_frame_equal(repeated, after)
    item.reprocess.overlap.value = 3
    writes.clear()
    fm.fetch_macro(context)
    assert writes and all(window.start == pd.Timestamp("2026-11-07") for window, keys in writes)
    print(
        "SANITY: real31-year yield context preserves exact bond levels and boundary returns;7CD/3CD writes obey config, protect both ends and repeat idempotently."
    )
