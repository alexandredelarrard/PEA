"""Known-truth identities for fiscal histories and prospective amendments."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.common.price_frames import load_trading_calendar
from src.data_aggregate.utils.fundamentals.fundamental_features import _derived_fields
from src.data_aggregate.utils.fundamentals.history_features import reported_history_fields
from src.data_store.schema import Tables
from tests.conftest import _store


def _history() -> pd.DataFrame:
    ends = pd.date_range("2017-03-31", periods=25, freq="QE")
    q = np.arange(len(ends), dtype=float)
    return pd.DataFrame(
        {
            "ticker": "AAA",
            "fiscal_end": ends,
            "as_of": ends + pd.Timedelta(days=35),
            "netIncome_q": q**2 - 200,
            "netIncome": q * 10 - 200,
            "totalAssetsExLease": 1000.0,
            "freeCashflow": np.where(q % 3 == 0, -1.0, 1.0),
        }
    )


def test_reported_history_exact_math_and_distinct_quarters() -> None:
    hist = _history()
    idx = pd.bdate_range(hist.as_of.min(), hist.as_of.max() + pd.Timedelta(days=10))
    fields = reported_history_fields(hist, idx)
    latest = fields["earnings_sue_lag0"]["AAA"].iloc[-1]
    d = hist.netIncome_q.diff(4).to_numpy()
    assert np.isclose(latest, d[-1] / np.std(d[-9:-1], ddof=1))
    assert np.isclose(fields["earnings_sue_lag7"]["AAA"].iloc[-1], d[-8] / np.std(d[-16:-8], ddof=1))
    assert fields["earnings_improvement_streak"]["AAA"].iloc[-1] == 8
    assert np.isclose(fields["profit_improvement_to_assets"]["AAA"].iloc[-1], 40 / 1000)
    assert np.isclose(fields["roa_vol_8q"]["AAA"].iloc[-1], np.std(hist.netIncome.iloc[-8:] / 1000, ddof=1))
    assert fields["fcf_positive_fraction_8q"]["AAA"].iloc[-1] == 5 / 8
    duplicate = pd.concat([hist, hist.iloc[[-1]].assign(as_of=hist.as_of.iloc[-1] + pd.Timedelta(days=3))])
    again = reported_history_fields(duplicate, idx)
    for name in fields:
        pd.testing.assert_frame_equal(fields[name], again[name])
    print("\nSANITY: seasonal earnings scale excludes the current change; repeated filings add no quarters; persistence uses eight quarters.")


def test_reported_history_amendments_are_prospective_and_nulls_clear() -> None:
    hist = _history()
    idx = pd.bdate_range(hist.as_of.min(), hist.as_of.max() + pd.Timedelta(days=20))
    baseline = reported_history_fields(hist, idx)
    revised_at = idx[-8]
    amendment = hist.iloc[[-2]].assign(as_of=revised_at, netIncome_q=-99999.0)
    revised = reported_history_fields(pd.concat([hist, amendment]), idx)
    for name in baseline:
        pd.testing.assert_frame_equal(baseline[name].loc[idx < revised_at], revised[name].loc[idx < revised_at])
    assert revised["earnings_sue_lag0"].iloc[-1, 0] != baseline["earnings_sue_lag0"].iloc[-1, 0]
    invalid_at = idx[-5]
    invalid = hist.iloc[[-1]].assign(as_of=invalid_at, freeCashflow=np.nan, totalAssetsExLease=0)
    cleared = reported_history_fields(pd.concat([hist, invalid]), idx)
    assert cleared["fcf_positive_fraction_8q"].loc[idx >= invalid_at, "AAA"].isna().all()
    assert cleared["roa_vol_8q"].loc[idx >= invalid_at, "AAA"].isna().all()
    assert cleared["profit_improvement_to_assets"].loc[idx >= invalid_at, "AAA"].isna().all()
    print("\nSANITY: amendments only change then-current knowledge; invalid new fiscal state clears old values.")


def test_reported_history_missing_quarter_zero_scale_and_incremental_tail() -> None:
    hist = _history()
    idx = pd.bdate_range(hist.as_of.min(), hist.as_of.max() + pd.Timedelta(days=10))
    broken = hist.drop(index=20)
    fields = reported_history_fields(broken, idx)
    assert fields["earnings_sue_lag0"].iloc[-1].isna().all()
    assert fields["earnings_improvement_streak"].iloc[-1].isna().all()
    assert fields["fcf_positive_fraction_8q"].iloc[-1].isna().all()
    flat = reported_history_fields(hist.assign(netIncome_q=1.0), idx)
    assert flat["earnings_sue_lag0"].isna().all().all()
    assert flat["earnings_improvement_streak"].iloc[-1, 0] == 0
    full = reported_history_fields(hist, idx)
    tail = reported_history_fields(hist, idx[-25:])
    for name in full:
        pd.testing.assert_frame_equal(full[name].iloc[-25:], tail[name])
    print("\nSANITY: gaps are never bridged, zero scale stays unknown, and a sliced daily grid reproduces the full tail.")


@pytest.mark.parametrize(
    "settings", [{"earnings_lags": 9}, {"earnings_scale_quarters": 9}, {"earnings_scale_quarters": 1}, {"persistence_quarters": 9}]
)
def test_history_windows_require_explicit_schema_and_source_migration(settings: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="feature schema"):
        reported_history_fields(_history(), pd.bdate_range("2026-01-05", periods=3), **settings)
    print(f"\nSANITY: unsupported window {settings} is rejected before an incomplete or misleading feature is emitted.")


def test_ocf_yield_preserves_cash_burn_and_uses_current_market_cap() -> None:
    idx = pd.bdate_range("2026-01-05", periods=3)
    hist = pd.DataFrame({"ticker": ["AAA"], "as_of": [idx[0]], "sharesOutstanding": [10.0], "operatingCashFlow": [-20.0]})
    close = pd.DataFrame({"AAA": [5.0, 10.0, 0.0]}, index=idx)
    fields = _derived_fields(hist, idx, close, level_factor=None)
    np.testing.assert_allclose(fields["ocf_yield"]["AAA"].iloc[:2], [-0.4, -0.2])
    assert np.isnan(fields["ocf_yield"]["AAA"].iloc[2])
    print("\nSANITY: OCF yield retains negative cash flow, updates with price, and rejects nonpositive market cap.")


def test_reported_histories_real_coverage_and_tails() -> None:
    context = SimpleNamespace(store=_store())
    metadata = context.store.load(Tables.sp500_tickers, columns=["ticker", "sector"])
    names = metadata.ticker.astype(str).to_list()
    columns = [
        "ticker",
        "as_of",
        "fiscal_end",
        "netIncome_q",
        "netIncome",
        "freeCashflow",
        "totalAssetsExLease",
        "totalAssets",
        "operatingLeaseRouAsset",
    ]
    available = set(context.store.columns(Tables.fundamentals_history))
    if not {"netIncome_q", "fiscal_end"}.issubset(available):
        pytest.skip("stored fundamentals lack the quarterly history contract")
    hist = context.store.load(
        Tables.fundamentals_history,
        columns=[name for name in columns if name in available],
        where={"ticker": names},
        since=pd.Timestamp("2010-01-01"),
        optional=True,
    )
    if hist is None or hist.empty:
        pytest.skip("no stored fundamentals for the registered universe")
    calendar = load_trading_calendar(context.store)
    idx = calendar[(calendar >= "2022-01-03") & (calendar <= pd.Timestamp(hist.as_of.max()))]
    fields = reported_history_fields(hist, idx)
    assert fields["earnings_sue_lag0"].notna().any().any()
    assert fields["roa_vol_8q"].stack().dropna().ge(0).all()
    assert fields["fcf_positive_fraction_8q"].stack().dropna().between(0, 1).all()
    assert fields["earnings_improvement_streak"].stack().dropna().between(0, 8).all()
    print(f"\nSANITY: read-only registered-universe 2022+ fiscal-history coverage and tails ({len(names)} names):")
    for name, frame in fields.items():
        frame = frame.reindex(columns=names)
        values = frame.stack().dropna()
        tails = values.quantile([0, 0.001, 0.01, 0.5, 0.99, 0.999, 1]).round(5).to_list()
        print(f"  {name}: coverage={len(values) / frame.size:.1%}, zero={values.eq(0).mean():.1%}, min/p0.1/p1/p50/p99/p99.9/max={tails}")
        yearly = frame.notna().groupby(frame.index.year).mean().mean(axis=1).mul(100).round(1).to_dict()
        sectors = {sector: round(frame.loc[:, rows.ticker].notna().to_numpy().mean() * 100, 1) for sector, rows in metadata.groupby("sector")}
        print(f"    coverage by year %={yearly}; by sector %={sectors}")
    signal = fields["earnings_sue_lag0"].stack().dropna()
    for date, ticker in [signal.idxmin(), signal.idxmax()]:
        known = hist.loc[(hist.ticker == ticker) & (pd.to_datetime(hist.as_of) <= date)]
        quarters = known.sort_values("as_of").drop_duplicates("fiscal_end", keep="last").sort_values("fiscal_end")
        changes = quarters.netIncome_q.diff(4)
        print(
            f"  SUE tail {ticker} {date.date()}: value={signal.loc[(date, ticker)]:.5f}; current NIq={quarters.netIncome_q.iloc[-1]:.0f}; prior-year NIq={quarters.netIncome_q.iloc[-5]:.0f}; scale={changes.iloc[-9:-1].std():.0f}"
        )
    latest = hist.loc[hist.ticker == "MSFT"].sort_values("as_of").drop_duplicates("fiscal_end", keep="last").sort_values("fiscal_end")
    if len(latest) >= 13:
        change = latest.netIncome_q.diff(4)
        expected = change.iloc[-1] / change.iloc[-9:-1].std(ddof=1)
        got = fields["earnings_sue_lag0"]["MSFT"].dropna().iloc[-1]
        assert np.isclose(got, expected)
    print("  Mathematical bounds hold and the latest MSFT SUE matches an independent fiscal-quarter calculation.")
    bounded = hist.loc[pd.to_datetime(hist.as_of) >= idx.max() - pd.DateOffset(years=6)]
    tail = reported_history_fields(bounded, idx[-7:])
    for name in fields:
        pd.testing.assert_frame_equal(fields[name].iloc[-7:], tail[name].reindex(columns=fields[name].columns))
    print("  Six-year source scope and a seven-session output tail reproduce the complete-history results exactly.")
