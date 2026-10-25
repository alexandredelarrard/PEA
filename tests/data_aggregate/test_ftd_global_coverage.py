"""Global FTD ZIP coverage survives ticker-scoped economic payloads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.common.price_frames import PriceFrames
from src.data_aggregate.utils.institutionals import short_flow_features
from src.data_aggregate.utils.institutionals.short_flow_features import FtdPublication
from tests.data_aggregate.test_short_flow_features import build_short_flow_feature_panel


@pytest.fixture(scope="module")
def frozen_ftd_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = Path("reports/validate/2026-10-22-institutionals-sec-thirteen-split/_cache")
    for root in Path(__file__).resolve().parents:
        source = root / cache / "ftd-coverage-source.parquet"
        prices = root / cache / "ftd-coverage-prices.parquet"
        if source.is_file() and prices.is_file():
            return pd.read_parquet(source), pd.read_parquet(prices)
    pytest.skip("Frozen real FTD coverage evidence is unavailable")


@pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "JPM"])
def test_ftd_scoped_payload_matches_global_rows(frozen_ftd_sources: tuple[pd.DataFrame, pd.DataFrame], ticker: str) -> None:
    df_fails, df_prices = frozen_ftd_sources
    df_bars = df_prices.loc[df_prices["ticker"].eq(ticker)]
    df_volume = df_bars.pivot(index="date", columns="ticker", values="volume").sort_index()
    df_close = df_bars.pivot(index="date", columns="ticker", values="close_split").sort_index()
    frames = PriceFrames(trading_index=df_close.index, universe=(ticker,), peers={}, close_split=df_close, volume=df_volume)
    df_scoped = df_fails.loc[df_fails["ticker"].eq(ticker)]
    df_coverage = df_fails[["date", "period"]].drop_duplicates()
    publication = FtdPublication(df_coverage, short_flow_features._ftd_available_dates(df_coverage, None))
    df_result = build_short_flow_feature_panel(frames, None, fails_history=df_scoped, publication=publication)
    df_reference = build_short_flow_feature_panel(frames, None, fails_history=df_fails)
    df_reference = df_reference.loc[df_reference["ticker"].eq(ticker)]
    pd.testing.assert_frame_equal(
        df_result.set_index(["date", "ticker"]).sort_index(),
        df_reference.set_index(["date", "ticker"]).sort_index(),
        check_exact=True,
    )
    print(f"SANITY: {ticker}'s scoped real FTD payload equals the global-source feature path, including nulls and ZIP publication dates.")


def test_ftd_global_coverage_keeps_unknown_days_and_uses_latest_zip_state(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.DatetimeIndex(pd.bdate_range("2023-12-01", "2024-02-20"), name="date")
    columns = pd.Index(["A", "ZERO", "FUTURE"], name="ticker")
    df_volume = pd.DataFrame(1_000.0, index=idx, columns=columns)
    df_close = pd.DataFrame(100.0, index=idx, columns=columns)
    df_volume.loc[:"2024-01-19", "FUTURE"] = np.nan
    df_close.loc[:"2024-01-19", "FUTURE"] = np.nan
    frames = PriceFrames(trading_index=idx, universe=tuple(columns), peers={}, volume=df_volume, close_split=df_close)
    df_fails = pd.DataFrame(
        [("2024-01-02", "A", 100.0, "202401a"), ("2024-01-16", "A", 200.0, "202401b")],
        columns=["date", "ticker", "fails_quantity", "period"],
    )
    df_coverage = pd.DataFrame(
        [("2024-01-02", "202401a"), ("2024-01-12", "202401a"), ("2024-01-16", "202401b"), ("2024-01-31", "202401b")],
        columns=["date", "period"],
    )
    states: list[pd.DataFrame] = []
    publish = short_flow_features._publish_ftd_zip_states

    def capture_state(
        settlement_state: pd.DataFrame,
        fails_hist: pd.DataFrame,
        idx: pd.DatetimeIndex,
        available_by_period: dict[str, pd.Timestamp] | None = None,
    ) -> pd.DataFrame:
        states.append(settlement_state.copy())
        return publish(settlement_state, fails_hist, idx, available_by_period)

    monkeypatch.setattr(short_flow_features, "_publish_ftd_zip_states", capture_state)
    publication = FtdPublication(df_coverage, short_flow_features._ftd_available_dates(df_coverage, None))
    df_panel = build_short_flow_feature_panel(frames, None, fails_history=df_fails, publication=publication)
    df_ratio = df_panel.pivot(index="date", columns="ticker", values="f_ic_ftd_to_adv20")
    assert states[0].loc[pd.Timestamp("2024-01-03")].isna().all()
    assert states[0].loc[pd.Timestamp("2024-01-12"), ["A", "ZERO"]].eq(0.0).all()
    assert pd.isna(states[0].loc[pd.Timestamp("2024-01-12"), "FUTURE"])
    assert df_ratio.loc[:"2024-01-29"].isna().all().all()
    assert df_ratio.loc[pd.Timestamp("2024-01-30"), ["A", "ZERO"]].eq(0.0).all()
    assert pd.isna(df_ratio.loc[pd.Timestamp("2024-01-30"), "FUTURE"])
    assert df_ratio.loc[pd.Timestamp("2024-02-15"), ["A", "ZERO"]].eq(0.0).all()
    print("SANITY: covered missing rows are zero, unobserved days and an unlisted source basis stay NaN; each ZIP publishes its global latest state.")


@pytest.mark.parametrize("empty_payload", [None, pd.DataFrame(columns=["date", "ticker", "fails_quantity", "period"])])
def test_ftd_global_coverage_publishes_zero_without_selected_rows(empty_payload: pd.DataFrame | None) -> None:
    idx = pd.DatetimeIndex(pd.bdate_range("2023-12-01", "2024-02-14"), name="date")
    df_volume = pd.DataFrame({"NO_FAILS": 1_000.0}, index=idx)
    frames = PriceFrames(trading_index=idx, universe=("NO_FAILS",), peers={}, volume=df_volume)
    df_coverage = pd.DataFrame({"date": ["2024-01-12"], "period": ["202401a"]})
    publication = FtdPublication(df_coverage, short_flow_features._ftd_available_dates(df_coverage, None))
    df_panel = build_short_flow_feature_panel(frames, None, fails_history=empty_payload, publication=publication)
    df_ratio = df_panel.pivot(index="date", columns="ticker", values="f_ic_ftd_to_adv20")
    assert set(df_ratio.columns) == {"NO_FAILS"}
    assert df_ratio.loc[:"2024-01-29"].isna().all().all()
    assert df_ratio.loc["2024-01-30":, "NO_FAILS"].eq(0.0).all()
    print("SANITY: a requested ticker with no selected source rows receives observed zero only after the globally covered ZIP publishes.")
