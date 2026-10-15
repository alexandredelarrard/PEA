"""Real, scoped return controls and exact guard behavior for total-only corrections."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from src.context import Context, get_config_context
from src.data_aggregate.transformers import step_cube_prices as price_step
from src.data_aggregate.utils.common.level_basis import apply_return_seams, measure_quote_seams, measure_seams
from src.data_store.schema import Tables
from src.utils.price_adjustments import load_bugfix


@pytest.fixture(scope="module")
def return_sources() -> tuple[Context, dict[str, Any], pd.DataFrame]:
    _, context = get_config_context("./configs", use_cache=False, save=False)
    raw = context.store.load(
        Tables.prices, columns=price_step.PRICE_COLS, where={"ticker": ["VTR", "SJM", "JCI"]}, since="2001-12-01", until="2016-02-18"
    )
    assert raw is not None
    raw["date"] = pd.to_datetime(raw["date"])
    return context, load_bugfix(context.config_dir), raw


@pytest.mark.parametrize(
    ("ticker", "stamp", "target", "cash_dates"),
    [
        ("VTR", "2015-08-18", (59.80 + 35.58 / 4) / (78.2459716796875 * 0.8757), ["2015-03-04", "2015-06-03", "2015-09-11", "2015-12-17"]),
        ("SJM", "2002-06-03", 0.9451 * 36.650001525878906 / 33.29999923706055, ["2002-02-13", "2002-05-16", "2002-08-16", "2002-11-14"]),
        ("JCI", "2007-07-02", (53.36 + 39.97 + 43.41) / (4 * 33.79), []),
    ],
)
def test_real_total_return_target_and_price_isolation(
    return_sources: tuple[Context, dict[str, Any], pd.DataFrame], ticker: str, stamp: str, target: float, cash_dates: list[str]
) -> None:
    _, blob, raw = return_sources
    when = pd.Timestamp(stamp)
    rows = raw.loc[raw["ticker"].eq(ticker) & raw["date"].between(when - pd.Timedelta(days=180), when + pd.Timedelta(days=180))]
    assert len(rows) >= 249
    wide = price_step.StepCubePrices._pivot_fields(rows)
    before = {field: frame.copy() for field, frame in wide.items()}
    assert apply_return_seams(wide, blob, lambda *args: None) == 1
    returns = wide["close_total"].pct_change(fill_method=None)
    original = before["close_total"].pct_change(fill_method=None)
    assert cast(float, returns.at[when, ticker]) + 1 == pytest.approx(target, rel=1e-12)
    other = returns.index != when
    np.testing.assert_allclose(returns.loc[other], original.loc[other], atol=5e-16, rtol=0, equal_nan=True)
    if ticker == "JCI":
        context = return_sources[0]
        dividends = context.store.load(
            Tables.dividends, columns=["date", "dividends"], where={"ticker": ticker}, since=rows["date"].min(), until=rows["date"].max()
        )
        assert dividends is not None
        cash_dates = dividends.loc[dividends["dividends"].ne(0) & pd.to_datetime(dividends["date"]).ne(when), "date"].astype(str).tolist()
        assert len(cash_dates) >= 3
    for date in cash_dates:
        assert returns.at[pd.Timestamp(date), ticker] == pytest.approx(original.at[pd.Timestamp(date), ticker], abs=5e-16)
    pd.testing.assert_frame_equal(wide["close_total"].loc[when:], before["close_total"].loc[when:], check_exact=True)
    for field in set(wide) - {"close_total"}:
        pd.testing.assert_frame_equal(wide[field], before[field], check_exact=True)
    assert measure_seams(wide["close_total"], blob, lambda *args: None) == {}
    assert measure_quote_seams(wide["close_split"], blob, lambda *args: None) == {ticker: [when]}
    healed = wide["close_total"].copy()
    assert apply_return_seams(wide, blob, lambda *args: None) == 0
    pd.testing.assert_frame_equal(healed, wide["close_total"], check_exact=True)
    with pytest.raises(ValueError, match="rebuild historical prices"):
        measure_seams(before["close_total"], blob, lambda *args: None)
    print(
        f"SANITY: {ticker} {len(rows)} real bars recover {(target - 1):+.9%}; cash returns/OHLC/volume survive, quote masks remain and reapplication is exact."
    )


@pytest.mark.parametrize(("ticker", "stamp"), [("VTR", "2015-08-18"), ("JCI", "2007-07-02")])
@pytest.mark.parametrize("state", ["changed", "missing", "nan", "healed"])
def test_refused_return_repair_cannot_emit_unprotected_returns(
    return_sources: tuple[Context, dict[str, Any], pd.DataFrame], monkeypatch: pytest.MonkeyPatch, state: str, ticker: str, stamp: str
) -> None:
    context, blob, raw = return_sources
    when = pd.Timestamp(stamp)
    rows = raw.loc[raw["ticker"].eq(ticker) & raw["date"].between(when - pd.Timedelta(days=8), when + pd.Timedelta(days=7))].copy()
    if state == "missing":
        rows = rows.loc[rows["date"].ne(when)]
    elif state == "nan":
        rows.loc[rows["date"].eq(when), "close_total"] = np.nan
    elif state == "changed":
        rows.loc[rows["date"].eq(when), "close_total"] *= 1.10
    else:
        item = blob["return_seams"][ticker][0]
        pair = rows.set_index("date")["close_total"]
        step = pair.loc[when] / pair.loc[pair.index < when].iloc[-1]
        rows.loc[rows["date"].lt(when), "close_total"] *= step / item["target_step"]

    class CaptureStore:
        def load(self, table: Any, **kwargs: Any) -> pd.DataFrame:
            assert table == Tables.prices
            return rows.copy()

    step = price_step.StepCubePrices.__new__(price_step.StepCubePrices)
    actual = cast(Any, step)
    actual._store = CaptureStore()
    actual._bugfix = blob
    actual._tickers = [ticker]
    actual._log = logging.getLogger(__name__)
    writes: list[pd.DataFrame] = []
    monkeypatch.setattr(step, "_plan_window", lambda full: SimpleNamespace(since=None))
    monkeypatch.setattr(step, "_trading_calendar", lambda close: pd.DatetimeIndex(close.index))
    monkeypatch.setattr(step, "_vendor_price", lambda: None)
    monkeypatch.setattr(step, "_peers", lambda: {})
    monkeypatch.setattr(step, "_universe_frames", lambda wide, returns, peers: {**wide, "ret": returns})
    monkeypatch.setattr(step, "_level_factor", lambda idx, close, vendor: close * 0 + 1)
    monkeypatch.setattr(price_step, "write_part", lambda store, table, frame, window: writes.append(frame))
    if state == "healed":
        step.run(full=True)
        assert len(writes) == 1
        emitted = writes[0].set_index("date")
        assert cast(float, emitted.at[when, "ret"]) + 1 == pytest.approx(blob["return_seams"][ticker][0]["target_step"], rel=1e-12)
    else:
        with pytest.raises(ValueError, match="REFUSED"):
            step.run(full=True)
        assert not writes
    print(f"SANITY: {ticker} ordinary builder state={state}: {'verified return emitted once' if writes else 'refusal before any write'}.")


def test_small_total_repair_applies_once_inside_defect_tolerance() -> None:
    index = pd.bdate_range("2020-01-01", periods=5)
    frame = pd.DataFrame({"TEST": [100.0, 100.0, 100.2, 100.3, 100.4]}, index=index)
    blob = {"return_seams": {"TEST": [{"date": str(index[2].date()), "fields": ["close_total"], "expect_step": 1.002, "target_step": 1.001}]}}
    wide = {"close_total": frame}
    assert apply_return_seams(wide, blob, lambda *args: None) == 1
    first = frame.copy()
    assert apply_return_seams(wide, blob, lambda *args: None) == 0
    pd.testing.assert_frame_equal(first, frame, check_exact=True)
    print("SANITY: a 0.1% repair inside the 0.5% defect band applies once; precise target recognition makes the second call a no-op.")
