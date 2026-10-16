"""Known-return arithmetic and report availability for disclosure reactions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.fundamentals.disclosure_features import disclosure_return_fields
from src.data_aggregate.utils.fundamentals.earnings_features import reported_events
from src.data_store.schema import Tables
from tests.conftest import _store


def test_disclosure_reaction_compounds_separately_and_waits_for_window() -> None:
    idx = pd.bdate_range("2026-01-05", periods=12)
    events = pd.DataFrame({"ticker": ["AAA"], "earnings_date": [idx[1]], "eps_actual": [0.0]})
    own = pd.DataFrame({"AAA": 0.1}, index=idx)
    peers = pd.DataFrame({"AAA": 0.02}, index=idx)
    fields = disclosure_return_fields(events, own, peers)
    reaction = fields["earnings_abret_3d"]["AAA"]
    since = fields["earnings_abret_since_report"]["AAA"]
    assert reaction.iloc[:4].isna().all()
    expected = 1.1**3 - 1.02**3
    np.testing.assert_allclose(reaction.iloc[4:], expected)
    assert np.isclose(since.iloc[2], 0.08)
    assert np.isclose(since.iloc[5], 1.1**4 - 1.02**4)
    assert since.iloc[:2].isna().all()
    print("\nSANITY: zero actual is a real report; reaction begins next session and freezes only after three complete returns.")


def test_disclosure_new_event_resets_and_missing_returns_are_not_zero() -> None:
    idx = pd.bdate_range("2026-01-05", periods=15)
    events = pd.DataFrame({"ticker": ["AAA"] * 3, "earnings_date": [idx[1], idx[7], idx[11]], "eps_actual": [1.0, 2.0, np.nan]})
    own = pd.DataFrame({"AAA": 0.1}, index=idx)
    peers = pd.DataFrame({"AAA": 0.02}, index=idx)
    peers.loc[idx[9], "AAA"] = np.nan
    fields = disclosure_return_fields(events, own, peers)
    assert fields["earnings_abret_3d"].loc[idx[8] :, "AAA"].isna().all()
    assert fields["earnings_abret_since_report"].loc[idx[9] :, "AAA"].isna().all()
    assert np.isclose(fields["earnings_abret_since_report"].loc[idx[8], "AAA"], 0.08)
    prefix = disclosure_return_fields(events.iloc[:1], own.iloc[:7], peers.iloc[:7])
    for name in fields:
        pd.testing.assert_frame_equal(fields[name].iloc[:7], prefix[name])
    print(
        "\nSANITY: new reports clear the old three-day reaction, missing return legs poison their window, and scheduled reports do not reset state."
    )


def test_disclosure_window_requires_explicit_feature_schema_migration() -> None:
    idx = pd.bdate_range("2026-01-05", periods=3)
    returns = pd.DataFrame({"AAA": 0.1}, index=idx)
    with pytest.raises(ValueError, match="feature schema"):
        disclosure_return_fields(pd.DataFrame(), returns, returns, reaction_sessions=4)
    print("\nSANITY: a four-session window cannot silently change the three-day feature definition.")


def test_disclosure_rejects_non_date_calendar() -> None:
    returns = pd.DataFrame({"AAA": [0.1]})
    with pytest.raises(TypeError, match="DatetimeIndex"):
        disclosure_return_fields(pd.DataFrame(), returns, returns)
    print("\nSANITY: disclosure windows reject a positional index rather than interpreting it as trading dates.")


def test_disclosure_reactions_real_returns_and_coverage() -> None:
    context = SimpleNamespace(store=_store())
    metadata = context.store.load(Tables.sp500_tickers, columns=["ticker", "sector"])
    names = metadata.ticker.astype(str).to_list()
    prices = pd.concat(
        context.store.iter_load(
            Tables.cube_part_prices,
            columns=["date", "ticker", "ret", "sector_ret"],
            where={"ticker": names},
            since=pd.Timestamp("2021-01-01"),
            chunksize=100_000,
        ),
        ignore_index=True,
    )
    earnings = context.store.load(
        Tables.earnings_surprises,
        columns=["ticker", "earnings_date", "eps_actual"],
        where={"ticker": names},
        since=pd.Timestamp("2020-01-01"),
        optional=True,
    )
    if prices is None or earnings is None or prices.empty or earnings.empty:
        pytest.skip("bounded report/return inputs are unavailable")
    prices["date"] = pd.to_datetime(prices.date)
    own = prices.pivot(index="date", columns="ticker", values="ret").sort_index()
    peer = prices.pivot(index="date", columns="ticker", values="sector_ret").sort_index()
    fields = disclosure_return_fields(earnings, own, peer)
    assert fields["earnings_abret_3d"].notna().any().any()
    observations = reported_events(earnings, own.index)
    checked = 0
    for event in observations.loc[observations.ticker == "MSFT"].itertuples():
        pos = own.index.get_loc(event.as_of)
        if pos + 3 > len(own):
            continue
        a, b = own.MSFT.iloc[pos : pos + 3], peer.MSFT.iloc[pos : pos + 3]
        got = fields["earnings_abret_3d"].loc[own.index[pos + 2], "MSFT"]
        if a.notna().all() and b.notna().all() and np.isfinite(got):
            assert np.isclose(got, (1 + a).prod() - (1 + b).prod())
            checked += 1
    assert checked > 0
    print(f"\nSANITY: {checked} real MSFT report responses match independently compounded own and peer total returns.")
    for name, frame in fields.items():
        frame = frame.reindex(columns=names)
        sample = frame.loc[frame.index >= "2022-01-01"].stack().dropna()
        total = frame.loc[frame.index >= "2022-01-01"].size
        tails = sample.quantile([0, 0.001, 0.01, 0.5, 0.99, 0.999, 1]).round(5).to_list()
        print(f"  {name}: {len(names)}-name coverage={len(sample) / total:.1%}, min/p0.1/p1/p50/p99/p99.9/max={tails}")
        print(f"  extreme observations: {sample.nsmallest(1).to_dict()} / {sample.nlargest(1).to_dict()}")
        for date, ticker in [sample.idxmin(), sample.idxmax()]:
            event = observations.loc[(observations.ticker == ticker) & (observations.as_of <= date)].iloc[-1]
            end = date if name.endswith("since_report") else own.index[own.index.get_loc(event.as_of) + 2]
            own_return = (1 + own.loc[event.as_of : end, ticker]).prod(min_count=1) - 1
            peer_return = (1 + peer.loc[event.as_of : end, ticker]).prod(min_count=1) - 1
            assert np.isclose(own_return - peer_return, sample.loc[(date, ticker)])
            print(
                f"    {ticker} event={event.earnings_date.date()}, return window={event.as_of.date()}..{end.date()}, own={own_return:.5f}, peer={peer_return:.5f}"
            )
        observed = frame.loc[frame.index >= "2022-01-01"].reindex(columns=names).notna()
        yearly = observed.groupby(observed.index.year).mean().mean(axis=1).mul(100).round(1).to_dict()
        sectors = {sector: round(observed.loc[:, rows.ticker].to_numpy().mean() * 100, 1) for sector, rows in metadata.groupby("sector")}
        print(f"    coverage by year %={yearly}; by sector %={sectors}")
    print("  Registered-universe coverage measured; peer graph and portfolio backtest are unchanged.")
