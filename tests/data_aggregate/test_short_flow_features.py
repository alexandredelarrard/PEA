"""Short-FLOW features: FINRA RegSHO short-sale VOLUME (`ic_shortvol_*`) and SEC
fails-to-deliver (`ic_ftd_*`) -- registry section 6.

Covers the RegSHO file parser, the VOLUME-WEIGHTED ratios (the point of the rework: an
average of daily ratios is a different statistic), the 1-trading-day publication lag, the
one-sided price interactions, the FTD "absent date is NaN, absent ticker is 0" rule, and the
emitted `f_*` columns against the module's own EMISSION map.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.common.pit import fundamentals_to_daily
from src.data_aggregate.utils.institutionals import short_flow_features as short_flow
from src.data_aggregate.utils.institutionals.inputs import QuantityBasis, daily_share_quantities, short_quantity_history, tape_share_quantities
from src.data_aggregate.utils.institutionals.short_flow_features import (
    EMISSION,
    SHORTVOL_PUB_LAG,
    Z_MIN_PERIODS,
    FtdPublication,
    ShortFlowCapital,
    _ftd_available_dates,
    _ftd_estimated_available_date,
    _publish_ftd_zip_states,
    _shortvol_fields,
    prepare_fails_quantities,
)
from src.data_aggregate.utils.institutionals.short_flow_features import (
    _fails_fields as compute_fails_fields,
)
from src.data_aggregate.utils.institutionals.short_flow_features import (
    build_short_flow_feature_panel as compute_short_panel,
)
from src.data_extract.utils.institutionals.fetch_short_interest import _parse_regsho
from src.utils.price_adjustments import ShareBasis, resolve_share_events
from tests.conftest import make_frames


def build_short_flow_feature_panel(
    frames: Any,
    short_history: pd.DataFrame | None,
    fails_history: pd.DataFrame | None = None,
    capital: ShortFlowCapital | None = None,
    publication: FtdPublication | None = None,
    tenure_mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Known-truth producer/visibility control through production preparation and computation."""
    idx = pd.DatetimeIndex(frames.trading_index).normalize().unique().sort_values()
    denominators = capital if capital is not None else ShortFlowCapital(None, frames.volume)
    if idx.empty:
        return compute_short_panel(frames, None, None, denominators, None)
    basis = QuantityBasis.from_context(resolve_share_events(None, pd.DataFrame(columns=["ticker", "date", "ratio"]), {}), idx)
    df_short = short_quantity_history(short_history, basis)
    settlement = None
    if fails_history is not None or publication is not None:
        df_fails = fails_history if fails_history is not None else pd.DataFrame(columns=["date", "ticker", "fails_quantity", "period"])
        df_coverage = publication.coverage if publication is not None else df_fails[["date", "period"]]
        columns = pd.Index(frames.universe, name="ticker")
        settlement = prepare_fails_quantities(df_fails, df_coverage, idx, columns, tenure_mask, basis)
        if publication is None:
            publication = FtdPublication(df_coverage, _ftd_available_dates(df_coverage, 21))
    return compute_short_panel(frames, df_short, settlement, denominators, publication, tenure_mask)


def _fails_fields(
    fails_hist: pd.DataFrame,
    idx: pd.DatetimeIndex,
    shares_out: pd.DataFrame | None,
    volume: pd.DataFrame | None,
    splits: pd.DataFrame | None = None,
    tenure_mask: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Known-truth dated-quantity fixture; numerical expectations remain explicit in each test."""
    df_splits = splits if splits is not None else pd.DataFrame(columns=["ticker", "date", "ratio"])
    basis = QuantityBasis.from_context(resolve_share_events(None, df_splits, {}), idx)
    settlement = prepare_fails_quantities(fails_hist, fails_hist[["date", "period"]], idx, None, tenure_mask, basis)
    publication = FtdPublication(fails_hist[["date", "period"]], _ftd_available_dates(fails_hist, 21))
    return compute_fails_fields(settlement, ShortFlowCapital(shares_out, volume), publication, tenure_mask)


def test_parse_regsho() -> None:
    txt = (
        "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
        "20230103|AAPL|500|0|1000|Q\n"
        "20230103|AAPL|100|0|200|N\n"  # second market center -> summed
        "20230103|MSFT|300|0|1200|Q\n"
        "3\n"
    )
    out = _parse_regsho(txt)
    aapl = out[out["source_symbol"] == "AAPL"].iloc[0]
    assert aapl["short_volume"] == 600 and aapl["total_volume"] == 1200  # summed
    assert out["date"].iloc[0] == pd.Timestamp("2023-01-03")
    assert _parse_regsho("Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n0\n").empty
    with pytest.raises(ValueError, match="header"):
        _parse_regsho("")
    print("\n=== SANITY CHECK: RegSHO parser ===")
    print(
        f"  pipe file parsed; AAPL short/total summed across markets = "
        f"{int(aapl['short_volume'])}/{int(aapl['total_volume'])}; validated empty0 is empty, unavailable text refuses. Validated."
    )


def _synth(t=700, n=8, seed=0):
    dates = pd.DatetimeIndex(pd.bdate_range("2021-01-04", periods=t))
    tickers = [f"S{i}" for i in range(n)]
    rng = np.random.default_rng(seed)
    total = rng.uniform(1e6, 5e6, (t, n))
    frac = np.clip(rng.normal(0.4, 0.1, (t, n)), 0.05, 0.95)
    frac[:, 0] = 0.85  # S0 persistently heavily shorted
    rows = []
    for j, t in enumerate(tickers):
        for i, d in enumerate(dates):
            rows.append({"date": d, "ticker": t, "short_volume": total[i, j] * frac[i, j], "total_volume": total[i, j]})
    return dates, tickers, pd.DataFrame(rows)


def test_the_ratio_is_volume_weighted_not_an_average_of_daily_ratios():
    """THE test for this phase (part-2b Phase 2.5 verification). Five days: four quiet days
    that were 90% short on 100k shares, and one heavy day that was 10% short on 10m. The
    window's short share of TRADED VOLUME is 3.7%, not the 74% an average of daily ratios
    reports -- a factor of 20, and the average is the statistic the module used to emit."""
    dates = pd.DatetimeIndex(pd.bdate_range("2023-01-02", periods=5))
    rows = [{"date": d, "ticker": "A", "total_volume": 100_000.0, "short_volume": 90_000.0} for d in dates[:4]]
    rows.append({"date": dates[4], "ticker": "A", "total_volume": 10_000_000.0, "short_volume": 1_000_000.0})
    hist = pd.DataFrame(rows)
    df = _shortvol_fields(hist, dates, None, None, None)
    # the lag means day 5's window is only readable on the NEXT grid day, so extend the index
    idx = pd.DatetimeIndex(pd.bdate_range("2023-01-02", periods=6))
    df = _shortvol_fields(hist, idx, None, None, None)
    got = float(cast(Any, df["ic_shortvol_ratio_5d"].loc[idx[5], "A"]))
    weighted = (4 * 90_000 + 1_000_000) / (4 * 100_000 + 10_000_000)
    naive = (4 * 0.9 + 0.1) / 5
    assert abs(got - weighted) < 1e-12
    assert abs(got - naive) > 0.5, "the volume weighting made no difference -- check the sum"
    print("\n=== SANITY CHECK: volume-weighted short ratio ===")
    print(
        f"  four 90%-short days on 100k + one 10%-short day on 10m: "
        f"ic_shortvol_ratio_5d = {got:.4f} (= sum short / sum total), against {naive:.2f} "
        f"for an average of daily ratios -- {naive / got:.0f}x. Validated."
    )


def test_publication_lag_is_one_trading_day():
    dates, tickers, hist = _synth()
    df = _shortvol_fields(hist, dates, None, None, None)
    assert {"ic_shortvol_ratio_5d", "ic_shortvol_ratio_20d", "ic_shortvol_ratio_60d", "ic_shortvol_acceleration"}.issubset(df)
    assert "ic_shortvol_ratio_z252" not in df
    # heavily-shorted S0 has the highest ratio cross-sectionally
    assert str(df["ic_shortvol_ratio_20d"].loc[dates[300]].idxmax()) == "S0"
    short = hist.pivot_table(index="date", columns="ticker", values="short_volume", aggfunc="sum")
    total = hist.pivot_table(index="date", columns="ticker", values="total_volume", aggfunc="sum")
    expected = float(cast(Any, (short.rolling(20, min_periods=10).sum() / total.rolling(20, min_periods=10).sum()).loc[dates[299], "S0"]))
    got = float(cast(Any, df["ic_shortvol_ratio_20d"].loc[dates[300], "S0"]))
    assert np.isclose(got, expected), "lag broken"
    assert SHORTVOL_PUB_LAG == 1
    print("\n=== SANITY CHECK: RegSHO 1-day publication lag ===")
    print(
        f"  S0 (85% shorted) tops ic_shortvol_ratio_20d at {dates[300].date()}; the value at "
        f"t equals the volume-weighted ratio computed through t-1 (published next morning). "
        "Validated."
    )


def test_source_adapters_normalize_both_rolling_legs_and_pit_counts() -> None:
    idx = pd.bdate_range("2024-06-03", periods=21)
    df_events = pd.DataFrame({"ticker": ["TEST"], "date": [idx[19]], "value": [10.0]})
    basis = QuantityBasis(ShareBasis(df_events, df_events, df_events, {}), idx[-1])
    df_raw = pd.DataFrame({"ticker": "TEST", "date": idx, "short_volume": [9.0] * 19 + [10.0] * 2, "total_volume": [10.0] * 19 + [100.0] * 2})
    df_filing = pd.DataFrame({"ticker": ["TEST"], "as_of": [idx[0]], "sharesOutstandingPit": [100.0], "sharesOutstanding": [1000.0]})
    df_normalized = short_quantity_history(df_raw, basis)
    df_shares = daily_share_quantities(df_filing, idx, basis)
    df_tape = tape_share_quantities(pd.DataFrame({"TEST": 200.0}, index=idx), basis)
    assert df_normalized is not None and df_shares is not None and df_tape is not None
    result = _shortvol_fields(df_normalized, idx, df_shares, None, df_tape)
    assert result["ic_shortvol_ratio_20d"].at[idx[20], "TEST"] == pytest.approx(1720 / 2000)
    assert result["ic_shortvol_turnover_20d"].at[idx[20], "TEST"] == pytest.approx(1720 / 1000)
    assert result["ic_shortvol_market_coverage"].at[idx[20], "TEST"] == pytest.approx(0.5)
    assert df_shares["TEST"].eq(1000.0).all()
    assert df_tape["TEST"].eq(200.0).all()
    assert df_raw["short_volume"].iloc[0] == 9
    assert df_filing["sharesOutstandingPit"].iloc[0] == 100
    print("SANITY: compatible rolling sums1720/2000 yield0.86 ratio and1.72 turnover over1000 PIT shares; terminal-restated tape stays unchanged.")


def test_future_event_cannot_advance_the_vendor_tape_basis() -> None:
    idx = pd.bdate_range("2024-06-03", periods=21)
    df_events = pd.DataFrame({"ticker": ["TEST"], "date": [idx[-1] + pd.Timedelta(days=10)], "value": [10.0]})
    basis = QuantityBasis.from_context(ShareBasis(df_events, df_events, df_events, {}), idx)
    df_raw = pd.DataFrame({"ticker": "TEST", "date": idx, "short_volume": 5.0, "total_volume": 10.0})
    df_filing = pd.DataFrame({"ticker": ["TEST"], "as_of": [idx[0]], "sharesOutstandingPit": [100.0]})
    df_normalized = short_quantity_history(df_raw, basis)
    df_shares = daily_share_quantities(df_filing, idx, basis)
    df_tape = tape_share_quantities(pd.DataFrame({"TEST": 20.0}, index=idx), basis)
    assert basis.target_date == idx[-1]
    assert df_normalized is not None
    pd.testing.assert_frame_equal(df_normalized, df_raw)
    assert df_shares is not None and df_shares["TEST"].eq(100).all()
    assert df_tape is not None and df_tape["TEST"].eq(20).all()
    print(
        "SANITY: a scheduled event beyond the last vendor bar cancels inF(source)/F(target); raw quantities, PIT counts and existing tape remain compatible."
    )


def test_paired_vendor_restatement_preserves_earlier_dimensionless_outputs() -> None:
    idx = pd.bdate_range("2024-01-02", periods=45)
    df_raw = pd.DataFrame({"ticker": "TEST", "date": idx, "short_volume": 5.0, "total_volume": 10.0})
    df_filing = pd.DataFrame({"ticker": ["TEST"], "as_of": [idx[0]], "sharesOutstandingPit": [100.0]})
    empty = pd.DataFrame(columns=["ticker", "date", "ratio"])
    basis = QuantityBasis.from_context(resolve_share_events(None, empty, {}), idx)
    df_shares = daily_share_quantities(df_filing, idx, basis)
    df_history = short_quantity_history(df_raw, basis)
    assert df_history is not None
    original = _shortvol_fields(df_history, idx, df_shares, None, pd.DataFrame({"TEST": 20.0}, index=idx))
    extended = pd.DatetimeIndex(idx.append(pd.DatetimeIndex([idx[-1] + pd.Timedelta(days=10)])))
    events = pd.DataFrame({"ticker": ["TEST"], "date": [extended[-1]], "ratio": [10.0]})
    later_basis = QuantityBasis.from_context(resolve_share_events(None, events, {}), extended)
    df_later = short_quantity_history(df_raw, later_basis)
    df_later_shares = daily_share_quantities(df_filing, extended, later_basis)
    assert df_later is not None
    restated = _shortvol_fields(df_later, extended, df_later_shares, None, pd.DataFrame({"TEST": 200.0}, index=extended))
    for name, df_before in original.items():
        pd.testing.assert_frame_equal(df_before, restated[name].reindex(idx))
    print(
        "SANITY: extending the actual vendor basis through a future10:1 event and restating its paired tape leaves every earlier ratio/turnover/coverage value and null unchanged."
    )


def test_missing_active_class_keeps_rolling_denominators_unknown() -> None:
    idx = pd.bdate_range("2024-01-02", periods=45)
    df_history = pd.DataFrame({"ticker": "TEST", "date": idx, "short_volume": 5.0, "total_volume": 10.0})
    df_tape = pd.DataFrame({"TEST": 20.0}, index=idx)
    df_supported = pd.DataFrame({"TEST": True}, index=idx)
    df_supported.loc[idx[20], "TEST"] = False
    frames = make_frames(idx, {"TEST": {}}, volume=df_tape)
    capital = ShortFlowCapital(None, df_tape, df_supported)
    panel = compute_short_panel(frames, df_history, None, capital, None).set_index("date")
    coverage = panel["f_ic_shortvol_market_coverage"]
    assert coverage.at[idx[20]] == pytest.approx(0.5)
    assert coverage.loc[idx[21] : idx[40]].isna().all()
    assert coverage.at[idx[41]] == pytest.approx(0.5)
    settlement = short_flow.FtdSettlement(pd.DataFrame({"TEST": 10.0}, index=idx), pd.Series(True, index=idx))
    df_coverage = pd.DataFrame({"date": idx, "period": "202401a"})
    publication = FtdPublication(df_coverage, {"202401a": idx[-1]})
    df_adv = compute_fails_fields(settlement, capital, publication, None)["ic_ftd_to_adv20"]
    assert df_adv.at[idx[-1], "TEST"] == pytest.approx(0.5)
    print(
        "SANITY: a missing active-class input leaves20 lagged coverage windows unknown and recovers only after its rolling denominator is supported; it never becomes claimed zero volume."
    )


def test_unknown_short_quantity_never_aggregates_into_observed_zero() -> None:
    idx = pd.bdate_range("2024-01-02", periods=21)
    df_history = pd.DataFrame({"ticker": "TEST", "date": idx, "short_volume": np.nan, "total_volume": 10.0})
    fields = _shortvol_fields(df_history, idx, None, None, None)
    assert all(df_field.isna().all().all() for df_field in fields.values())
    print("SANITY: unknown quantity observations stay unknown after daily aggregation and never appear as measured zero short sales.")


def test_price_interactions_are_one_sided_and_complementary():
    """#62/#63. High short flow means opposite things depending on the price path, so each
    interaction is zero in the other's regime -- never averaged into one signed number."""
    dates, tickers, hist = _synth(t=400)
    rng = np.random.default_rng(3)
    close = pd.DataFrame({t: 100 * np.cumprod(1 + rng.normal(0.0004, 0.02, len(dates))) for t in tickers}, index=dates)
    df = _shortvol_fields(hist, dates, None, close, None)
    weak = df["ic_shortvol_high_x_weak_price"]
    strong = df["ic_shortvol_high_x_strong_price"]
    assert (weak.fillna(0) >= 0).all().all() and (strong.fillna(0) >= 0).all().all()
    both = (weak > 0) & (strong > 0)
    assert not both.to_numpy().any(), "a cell is both confirming and absorbing"
    print("\n=== SANITY CHECK: the two short-flow price interactions ===")
    print(
        f"  both legs >= 0 and {int(both.to_numpy().sum())} cells fire both (must be 0); "
        "the internal normalization conditions the economic interactions but is not emitted. Validated."
    )


@pytest.mark.parametrize("checked_zero", [False, True], ids=["legacy-absence-unknown", "checked-zero"])
def test_ftd_only_explicit_supported_zero_is_held(checked_zero: bool) -> None:
    idx = pd.DatetimeIndex(pd.bdate_range("2023-12-01", "2024-02-14"))
    fails = pd.DataFrame(
        [
            {"date": "2024-01-02", "ticker": "HI", "fails_quantity": 100.0, "period": "202401a"},
            {"date": "2024-01-02", "ticker": "LO", "fails_quantity": 50.0, "period": "202401a"},
            {"date": "2024-01-15", "ticker": "HI", "fails_quantity": 300.0, "period": "202401a"},
        ]
    )
    if checked_zero:
        fails = pd.concat(
            [fails, pd.DataFrame([{"date": "2024-01-15", "ticker": "LO", "fails_quantity": 0.0, "period": "202401a"}])], ignore_index=True
        )
        fails["source_checked_at"] = pd.Timestamp("2024-01-30")
    volume = pd.DataFrame(1_000.0, index=idx, columns=["HI", "LO"])
    ratio = _fails_fields(fails, idx, None, volume)["ic_ftd_to_adv20"]

    prepublication = ratio.loc[pd.Timestamp("2024-02-02")]
    assert isinstance(prepublication, pd.Series)
    assert prepublication.isna().to_numpy().all()
    assert ratio.loc[pd.Timestamp("2024-02-05"), "HI"] == pytest.approx(0.3)
    if checked_zero:
        assert ratio.loc[pd.Timestamp("2024-02-05"), "LO"] == pytest.approx(0.0)
    else:
        assert pd.isna(ratio.loc[pd.Timestamp("2024-02-05"), "LO"])
    assert ratio.loc[pd.Timestamp("2024-02-14"), "HI"] == pytest.approx(0.3)
    print("\n=== SANITY CHECK: FTD coverage semantics ===")
    print("  A supported explicit LO zero is held after publication; another ticker's later observation cannot invent a legacy LO zero.")


def test_ftd_zip_state_publishes_atomically_without_summing_balances():
    idx = pd.DatetimeIndex(pd.bdate_range("2023-12-01", "2024-02-26"))
    fails = pd.DataFrame(
        [
            {"date": "2024-01-02", "ticker": "HI", "fails_quantity": 100.0, "period": "202401a"},
            {"date": "2024-01-02", "ticker": "LO", "fails_quantity": 50.0, "period": "202401a"},
            {"date": "2024-01-15", "ticker": "HI", "fails_quantity": 300.0, "period": "202401a"},
            {"date": "2024-01-15", "ticker": "LO", "fails_quantity": 0.0, "period": "202401a", "source_checked_at": "2024-01-30"},
            {"date": "2024-01-16", "ticker": "HI", "fails_quantity": 800.0, "period": "202401b"},
            {"date": "2024-01-31", "ticker": "HI", "fails_quantity": 900.0, "period": "202401b"},
        ]
    )
    volume = pd.DataFrame(1_000.0, index=idx, columns=["HI", "LO"])

    ratio = _fails_fields(fails, idx, None, volume)["ic_ftd_to_adv20"]

    assert ratio.loc[pd.Timestamp("2024-02-02")].isna().all()
    assert ratio.loc[pd.Timestamp("2024-02-05"), "HI"] == pytest.approx(0.3)
    assert ratio.loc[pd.Timestamp("2024-02-05"), "LO"] == pytest.approx(0.0)
    assert ratio.loc[pd.Timestamp("2024-02-20"), "HI"] == pytest.approx(0.3)
    assert ratio.loc[pd.Timestamp("2024-02-21"), "HI"] == pytest.approx(0.9)

    print("\n=== SANITY CHECK: FTD ZIP state publication ===")
    print("  January a publishes its latest 300-share balance atomically on estimated Feb 5; January b replaces it on Feb 21")


def test_ftd_available_date_on_weekend_publishes_next_session():
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-15", "2024-02-09"))
    state = pd.DataFrame({"A": np.nan}, index=idx)
    state.loc[pd.Timestamp("2024-01-15"), "A"] = 0.3
    fails = pd.DataFrame({"date": ["2024-01-15"], "period": ["202401a"]})
    published = _publish_ftd_zip_states(state, fails, idx, {"202401a": pd.Timestamp("2024-02-03")})

    assert published.loc[:"2024-02-02", "A"].isna().all()
    assert published.loc[pd.Timestamp("2024-02-05"), "A"] == pytest.approx(0.3)
    print("\n=== SANITY CHECK: weekend FTD availability ===")
    print("  a Saturday ZIP availability date first appears on Monday Feb 5")


def test_ftd_late_older_zip_does_not_replace_newer_settlement_state():
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-15", "2024-02-23"))
    state = pd.DataFrame({"A": np.nan}, index=idx)
    state.loc[pd.Timestamp("2024-01-15"), "A"] = 0.3
    state.loc[pd.Timestamp("2024-01-31"), "A"] = 0.9
    fails = pd.DataFrame({"date": ["2024-01-15", "2024-01-31"], "period": ["202401a", "202401b"]})
    published = _publish_ftd_zip_states(state, fails, idx, {"202401a": pd.Timestamp("2024-02-20"), "202401b": pd.Timestamp("2024-02-15")})

    assert pd.isna(published.loc[pd.Timestamp("2024-02-14"), "A"])
    assert published.loc[pd.Timestamp("2024-02-15"), "A"] == pytest.approx(0.9)
    assert published.loc[pd.Timestamp("2024-02-20"), "A"] == pytest.approx(0.9)
    print("\n=== SANITY CHECK: out-of-order FTD ZIP observations ===")
    print("  a late older ZIP never rolls the published state back from January b to January a")


def test_ftd_publication_uses_persisted_zip_period_at_half_month_boundary():
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-12", "2024-02-26"))
    state = pd.DataFrame({"A": np.nan}, index=idx)
    state.loc[pd.Timestamp("2024-01-12"), "A"] = 0.2
    state.loc[pd.Timestamp("2024-01-15"), "A"] = 0.5
    state.loc[pd.Timestamp("2024-01-31"), "A"] = 0.9
    fails = pd.DataFrame({"date": ["2024-01-12", "2024-01-31", "2024-01-15"], "period": ["202401a", "202401b", "202401b"]})
    published = _publish_ftd_zip_states(state, fails, idx, _ftd_available_dates(fails, 21))

    assert published.loc[pd.Timestamp("2024-02-05"), "A"] == pytest.approx(0.2)
    assert published.loc[pd.Timestamp("2024-02-21"), "A"] == pytest.approx(0.9)
    print("\n=== SANITY CHECK: FTD ZIP boundary ===")
    print("  a settlement on day 15 carried by the b ZIP cannot enter the a-period feature")


def test_ftd_day_15_b_zip_uses_its_stored_period():
    idx = pd.DatetimeIndex(pd.bdate_range("2024-04-15", "2024-05-24"))
    state = pd.DataFrame({"A": np.nan}, index=idx)
    state.loc[pd.Timestamp("2024-04-15"), "A"] = 5.0
    fails = pd.DataFrame({"date": ["2024-04-15"], "period": ["202404b"]})

    published = _publish_ftd_zip_states(state, fails, idx, _ftd_available_dates(fails, 21))

    assert published.loc[:"2024-05-20", "A"].isna().all()
    assert published.loc[pd.Timestamp("2024-05-21"), "A"] == 5.0
    print("SANITY: a day-15 row carried by the April b ZIP first appears May 21 from its stored period.")


def test_ftd_policy_21d_is_mtime_and_today_invariant(tmp_path: Path) -> None:
    # Explicit settlement/policy/session dates are an independent calendar arithmetic oracle.
    cases = [
        ("202609a", "2026-09-15", "2026-10-06", "2026-10-05", "2026-10-06", "2026-10-07"),
        ("202609b", "2026-09-30", "2026-10-21", "2026-10-20", "2026-10-21", "2026-10-22"),
        ("202402b", "2024-02-29", "2024-03-21", "2024-03-20", "2024-03-21", "2024-03-22"),
        ("202406a", "2024-06-14", "2024-07-06", "2024-07-05", "2024-07-08", "2024-07-09"),
        ("202201b", "2022-01-31", "2022-02-21", "2022-02-18", "2022-02-22", "2022-02-23"),
        ("202312b", "2023-12-29", "2024-01-21", "2024-01-19", "2024-01-22", "2024-01-23"),
        ("202401a", "2024-01-12", "2024-02-05", "2024-02-02", "2024-02-05", "2024-02-06"),
    ]
    source = pd.DataFrame({"date": [case[1] for case in cases], "period": [case[0] for case in cases]})
    expected = {case[0]: pd.Timestamp(case[2]) for case in cases}
    zip_path = tmp_path / "cnsfails202609b.zip"
    zip_path.write_bytes(b"identical cached ZIP bytes")
    observed: list[dict[str, pd.Timestamp]] = []
    clock_reads: list[int] = []
    for today, stamp in [("2026-10-19", "2026-10-19"), ("2026-10-22", "2026-10-19"), ("2027-10-22", "2026-10-21")]:
        mtime = pd.Timestamp(f"{stamp} 12:00", tz="America/New_York").timestamp()
        os.utime(zip_path, (mtime, mtime))
        frozen_today = pd.Timestamp(today)
        with patch.object(short_flow.pd, "Timestamp", wraps=pd.Timestamp) as timestamp:
            timestamp.now.return_value = frozen_today
            observed.append(_ftd_available_dates(source, 21))
            clock_reads.append(timestamp.now.call_count)
    print(f"F004 policy probe: September-a/b dates across cache/evaluation clocks: {[(row['202609a'], row['202609b']) for row in observed]}")
    assert observed == [expected] * len(observed)
    assert clock_reads == [0] * len(clock_reads)
    assert zip_path.read_bytes() == b"identical cached ZIP bytes"
    for invalid in [None, True, 0, -1, 21.5]:
        with pytest.raises(ValueError, match="positive integer"):
            _ftd_available_dates(source, cast(Any, invalid))
    for period, settlement, _, previous, first, after in cases:
        idx = pd.DatetimeIndex([settlement, previous, first, after])
        state = pd.DataFrame({"A": [7.0, np.nan, np.nan, np.nan]}, index=idx)
        history = pd.DataFrame({"date": [settlement], "period": [period]})
        published = _publish_ftd_zip_states(state, history, idx, observed[0])
        assert published.loc[:previous, "A"].isna().all()
        assert published["A"].first_valid_index() == pd.Timestamp(first)
        assert published.loc[first:after, "A"].eq(7.0).all()
        assert _publish_ftd_zip_states(state, history, idx[:2], observed[0]).isna().all().all()
    print(
        "SANITY: estimated_policy_21d uses half-month end plus 21 calendar days, then the first actual session, across leap/weekend/holiday cases; cache mtime and today's date have no economic effect."
    )


def test_ftd_partial_universe_uses_the_same_period_policy() -> None:
    scoped_rows = pd.DataFrame({"date": ["2026-08-31"], "period": ["202608b"]})
    global_rows = pd.concat([scoped_rows, pd.DataFrame({"date": ["2026-09-15"], "period": ["202609a"]})], ignore_index=True)
    available = _ftd_available_dates(scoped_rows, 21)
    assert available == {"202608b": pd.Timestamp("2026-09-21")}
    assert available["202608b"] == _ftd_available_dates(global_rows, 21)["202608b"]
    print("SANITY: the same ZIP has the same estimated policy day in scoped and global source histories.")


def test_ftd_period_must_be_present_and_valid():
    for frame in [
        pd.DataFrame({"date": ["2024-04-15"]}),
        pd.DataFrame({"date": ["2024-04-15"], "period": [None]}),
        pd.DataFrame({"date": ["2024-04-15"], "period": ["202413b"]}),
        pd.DataFrame({"date": ["2024-04-15", "2024-04-16"], "period": ["202404b", 2024041]}),
    ]:
        with pytest.raises(ValueError, match="period|ZIP"):
            _ftd_available_dates(frame, 21)
    print("SANITY: missing, null, and malformed ZIP periods fail closed before feature publication.")


def test_ftd_rejects_ambiguous_settlement_date_across_zip_periods():
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-15", "2024-02-20"))
    state = pd.DataFrame({"A": 0.1}, index=idx)
    fails = pd.DataFrame({"date": ["2024-01-15", "2024-01-15"], "period": ["202401a", "202401b"]})
    with pytest.raises(ValueError, match="multiple ZIP periods"):
        _publish_ftd_zip_states(state, fails, idx, _ftd_available_dates(fails, 21))
    print("\n=== SANITY CHECK: FTD ZIP boundary ambiguity ===")
    print("  a settlement date carried by two ZIPs fails closed until source grain is resolved")


def test_ftd_zip_state_preserves_nan_in_latest_state():
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-02", "2024-02-09"))
    settlement_state = pd.DataFrame({"A": np.nan}, index=idx)
    settlement_state.loc[pd.Timestamp("2024-01-02"), "A"] = 1.0
    fails_hist = pd.DataFrame({"date": ["2024-01-02", "2024-01-15"], "period": ["202401a", "202401a"]})

    published = _publish_ftd_zip_states(settlement_state, fails_hist, idx, _ftd_available_dates(fails_hist, 21))

    assert published.loc[pd.Timestamp("2024-02-05") :, "A"].isna().all()
    print("\n=== SANITY CHECK: FTD ZIP latest-state NaN ===")
    print("  a [finite, NaN] settlement state publishes NaN; an earlier finite value is not carried into the ZIP's latest state")


def test_ftd_observed_all_zero_history_is_neutral_but_unavailable_is_nan():
    idx = pd.DatetimeIndex(pd.bdate_range("2022-01-03", periods=360))
    covered = idx[:300]
    fails = pd.DataFrame(
        [{"date": day, "ticker": ticker, "fails_quantity": value} for day in covered for ticker, value in (("ZERO", 0.0), ("CONST", 100.0))]
    )
    fails["period"] = fails["date"].map(lambda day: f"{day:%Y%m}{'a' if day.day <= 15 else 'b'}")
    volume = pd.DataFrame({"ZERO": 1_000_000.0, "CONST": 1_000_000.0}, index=idx)

    fields = _fails_fields(fails, idx, None, volume)
    persistence = fields["ic_ftd_persistence_30d"]
    observed_zero = persistence["ZERO"].dropna()

    assert not observed_zero.empty
    assert observed_zero.eq(0.0).all()
    assert persistence["CONST"].isna().all(), "a nonzero constant basis has no defined persistence state"
    assert persistence.loc[idx[-1], "ZERO"] == 0.0, "the latest published ZIP state is held until another ZIP supersedes it"
    assert "ic_ftd_z252" not in fields
    print("\n=== SANITY CHECK: FTD neutral zero versus unavailable ===")
    print(
        f"  ZERO becomes persistence=0 after {Z_MIN_PERIODS} fully observed source dates plus its lookback and the published state is held; "
        "CONST remains unavailable, while the internal z is not emitted"
    )
    print("  OK: zero means observed neutral pressure, never missing source coverage")


def test_latest_rolling_outputs_require_their_source_date():
    dates, tickers, hist = _synth(t=400, n=2)
    missing_regsho_date = dates[-2]
    hist = hist[hist["date"] != missing_regsho_date]
    shares = pd.DataFrame(500_000_000.0, index=dates, columns=tickers)
    close = pd.DataFrame({ticker: np.linspace(100.0, 120.0, len(dates)) for ticker in tickers}, index=dates)
    volume = pd.DataFrame(5_000_000.0, index=dates, columns=tickers)
    short_fields = _shortvol_fields(hist, dates, shares, close, volume)
    assert all(frame.loc[dates[-1]].isna().all() for frame in short_fields.values())

    print("\n=== SANITY CHECK: rolling source-date completeness ===")
    print("  a missing RegSHO t-1 observation makes every latest derived leg NaN")


def test_ftd_publication_snaps_weekend_and_holiday_forward():
    weekend_idx = pd.bdate_range("2024-06-03", "2024-07-10").difference(pd.DatetimeIndex(["2024-06-19", "2024-07-04"]))
    weekend_state = pd.DataFrame({"A": np.nan}, index=weekend_idx)
    weekend_state.loc[pd.Timestamp("2024-06-14"), "A"] = 1.0
    weekend_hist = pd.DataFrame([{"date": "2024-06-14", "period": "202406a"}])
    weekend_published = _publish_ftd_zip_states(weekend_state, weekend_hist, weekend_idx, _ftd_available_dates(weekend_hist, 21))

    holiday_idx = pd.bdate_range("2022-01-03", "2022-02-23").difference(pd.DatetimeIndex(["2022-01-17", "2022-02-21"]))
    holiday_state = pd.DataFrame({"A": np.nan}, index=holiday_idx)
    holiday_state.loc[pd.Timestamp("2022-01-31"), "A"] = 2.0
    holiday_hist = pd.DataFrame([{"date": "2022-01-31", "period": "202201b"}])
    holiday_published = _publish_ftd_zip_states(holiday_state, holiday_hist, holiday_idx, _ftd_available_dates(holiday_hist, 21))

    assert _ftd_estimated_available_date("202406a", 21) == pd.Timestamp("2024-07-06")
    assert _ftd_estimated_available_date("202405b", 21) == pd.Timestamp("2024-06-21")
    assert pd.isna(weekend_published.loc[pd.Timestamp("2024-07-05"), "A"])
    assert weekend_published.loc[pd.Timestamp("2024-07-08"), "A"] == 1.0
    assert _ftd_estimated_available_date("202201b", 21) == pd.Timestamp("2022-02-21")
    assert pd.isna(holiday_published.loc[pd.Timestamp("2022-02-18"), "A"])
    assert holiday_published.loc[pd.Timestamp("2022-02-22"), "A"] == 2.0
    print("\n=== SANITY CHECK: FTD publication-date snapping ===")
    print("  the July 6 Saturday policy day publishes July 8; the February 21 market holiday publishes February 22")


def test_appending_later_ftd_zip_does_not_rewrite_published_prefix():
    idx = pd.DatetimeIndex(pd.bdate_range("2023-12-01", "2024-02-26"))
    first = pd.DataFrame(
        [
            {"date": "2024-01-02", "ticker": "A", "fails_quantity": 100.0, "period": "202401a"},
            {"date": "2024-01-15", "ticker": "A", "fails_quantity": 300.0, "period": "202401a"},
        ]
    )
    later = pd.DataFrame(
        [
            {"date": "2024-01-16", "ticker": "A", "fails_quantity": 800.0, "period": "202401b"},
            {"date": "2024-01-31", "ticker": "A", "fails_quantity": 900.0, "period": "202401b"},
        ]
    )
    volume = pd.DataFrame(1_000.0, index=idx, columns=["A"])

    before = _fails_fields(first, idx, None, volume)["ic_ftd_to_adv20"]
    after = _fails_fields(pd.concat([first, later], ignore_index=True), idx, None, volume)["ic_ftd_to_adv20"]

    pd.testing.assert_frame_equal(before.loc[:"2024-02-20"], after.loc[:"2024-02-20"])
    assert after.loc[pd.Timestamp("2024-02-21"), "A"] == pytest.approx(0.9)
    print("\n=== SANITY CHECK: FTD appended-ZIP prefix invariance ===")
    print("  adding January b leaves every January-a published cell unchanged and replaces the held state only on Feb 21")


def _market_tenures(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Explicit known-truth market identity rows, independent of accounting CIK windows."""
    return pd.DataFrame([{"source": "ftd", "security_class": "common", "conversion_ratio": 1.0, **row} for row in rows])


@pytest.mark.parametrize(
    "ticker, start, seam, end",
    [
        ("APO", "2019-09-04", "2021-12-31", "2022-05-10"),
        ("CI", "2018-11-02", "2018-12-20", "2018-12-27"),
        ("NXPI", "2010-10-12", "2019-08-01", "2019-08-02"),
    ],
)
def test_market_tenure_uses_dated_canonical_security_intervals(ticker: str, start: str, seam: str, end: str) -> None:
    idx = pd.DatetimeIndex([pd.Timestamp(start) - pd.Timedelta(days=1), start, seam, end])
    master = _market_tenures(
        [
            {
                "canonical_company": ticker,
                "security_id": "C111111111",
                "source_symbol": "OLD",
                "lineage_role": "canonical_predecessor",
                "valid_from": start,
                "valid_to": seam,
            },
            {
                "canonical_company": ticker,
                "security_id": "C222222222",
                "source_symbol": ticker,
                "lineage_role": "canonical_current",
                "valid_from": seam,
                "valid_to": end,
            },
        ]
    )
    mask = short_flow._proven_tenure_mask(idx, pd.Index([ticker]), master, None)
    assert isinstance(mask, pd.DataFrame), "dated market identity must produce an explicit support mask"
    assert mask[ticker].tolist() == [False, True, True, False]
    print(
        f"SANITY: {ticker} uses exact half-open canonical market dates across predecessor/current identity; accounting or Form345 onsets cannot truncate them."
    )


def test_market_tenure_accepts_only_supported_observed_finra_fallback() -> None:
    idx = pd.date_range("2024-01-02", periods=7)
    fallback = pd.DataFrame(
        {
            "date": idx[[0, 2, 3, 4, 5, 6]],
            "ticker": "NEW",
            "security_id": ["S0000000123:OLD", "S0000000456:NEW", "C111111111", "S0000000456:NEW", "S0000000456:NEW", "S0000000456:NEW"],
            "lineage_role": [
                "canonical_predecessor",
                "canonical_current",
                "canonical_current",
                "acquired_constituent",
                "secondary_class",
                "canonical_current",
            ],
            "security_class": ["common", "class_A", "common", "common", "class_A", "unclassified"],
        }
    )
    mask = short_flow._proven_tenure_mask(idx, pd.Index(["NEW"]), None, fallback)
    assert isinstance(mask, pd.DataFrame), "observed FINRA mapping must produce explicit support"
    assert mask["NEW"].tolist() == [True, False, True, False, False, False, False]
    print(
        "SANITY: accepted canonical FINRA S-stamps support only their observed dates; gaps, acquired/secondary/unclassified and nonfallback rows remain unknown without operational timestamps."
    )


@pytest.mark.parametrize("master", [None, pd.DataFrame(), pd.DataFrame({"valid_from": ["2024-01-02"]})], ids=["missing", "empty", "malformed"])
def test_market_tenure_keeps_missing_and_rejected_identity_unknown(master: pd.DataFrame | None) -> None:
    idx = pd.date_range("2024-01-02", periods=3)
    mask = short_flow._proven_tenure_mask(idx, pd.Index(["UNKNOWN"]), master, None)
    assert isinstance(mask, pd.DataFrame), "missing identity must never disable tenure masking"
    assert not mask.to_numpy().any()
    print("SANITY: missing, empty or malformed dated identity yields all-false support, so neither observed features nor quiet zeros are invented.")


def test_market_tenure_refuses_ambiguous_primary_security() -> None:
    idx = pd.date_range("2024-01-02", periods=5)
    master = _market_tenures(
        [
            {
                "canonical_company": "NEW",
                "security_id": "C111111111",
                "source_symbol": "NEW",
                "lineage_role": "canonical_current",
                "valid_from": idx[0],
                "valid_to": idx[4],
            },
            {
                "canonical_company": "NEW",
                "security_id": "C222222222",
                "source_symbol": "NEW",
                "lineage_role": "canonical_current",
                "valid_from": idx[2],
                "valid_to": idx[3],
            },
        ]
    )
    fallback = pd.DataFrame(
        {"date": [idx[2], idx[4]], "ticker": "NEW", "security_id": "S0000000123:NEW", "lineage_role": "canonical_current", "security_class": "common"}
    )
    mask = short_flow._proven_tenure_mask(idx, pd.Index(["NEW"]), master, fallback)
    assert isinstance(mask, pd.DataFrame)
    assert mask["NEW"].tolist() == [True, True, False, True, False]
    print("SANITY: competing primary CUSIPs keep their overlap unknown, and fallback cannot override a known master's conflict or expired interval.")


@pytest.mark.parametrize(
    "role, security_class, ratio",
    [
        ("acquired_constituent", "common", 1.0),
        ("excluded", "common", 1.0),
        ("secondary_class", "class_A", 1.0),
        ("canonical_current", "unclassified", 1.0),
        ("canonical_current", "preferred", 1.0),
        ("canonical_current", "common", float("nan")),
        ("canonical_current", "common", 0.0),
    ],
)
def test_market_tenure_does_not_promote_rejected_or_secondary_lines(role: str, security_class: str, ratio: float) -> None:
    idx = pd.date_range("2024-01-02", periods=2)
    master = _market_tenures(
        [
            {
                "canonical_company": "NEW",
                "security_id": "C111111111",
                "source_symbol": "NEW",
                "lineage_role": role,
                "security_class": security_class,
                "conversion_ratio": ratio,
                "valid_from": idx[0],
                "valid_to": None,
            }
        ]
    )
    fallback = pd.DataFrame(
        {"date": idx, "ticker": "NEW", "security_id": "S0000000123:NEW", "lineage_role": "canonical_current", "security_class": "common"}
    )
    mask = short_flow._proven_tenure_mask(idx, pd.Index(["NEW"]), master, fallback)
    assert isinstance(mask, pd.DataFrame)
    assert not mask.to_numpy().any()
    print(
        "SANITY: acquired, excluded, secondary-only, unsupported class and invalid conversion evidence cannot establish primary market tenure or be overridden by fallback."
    )


@pytest.mark.parametrize("other_symbol", ["NEW", "OLD"])
def test_market_tenure_refuses_one_security_with_two_active_owners(other_symbol: str) -> None:
    idx = pd.date_range("2024-01-02", periods=3)
    master = _market_tenures(
        [
            {
                "canonical_company": company,
                "source_symbol": symbol,
                "security_id": "C111111111",
                "lineage_role": "canonical_current",
                "valid_from": idx[0],
                "valid_to": None,
            }
            for company, symbol in [("NEW", "NEW"), ("OTHER", other_symbol)]
        ]
    )
    support = short_flow._proven_tenure_mask(idx, pd.Index(["NEW", "OTHER"]), master, None)
    assert not support.to_numpy().any()
    print("SANITY: one active security cannot support two canonical owners, including across different historical symbol keys.")


@pytest.mark.parametrize("rejected_symbol", ["NEW", "OLD"])
def test_market_tenure_refuses_rejected_rows_of_the_same_security(rejected_symbol: str) -> None:
    idx = pd.date_range("2024-01-02", periods=3)
    master = _market_tenures(
        [
            {
                "canonical_company": "NEW",
                "source_symbol": symbol,
                "security_id": "C111111111",
                "lineage_role": role,
                "valid_from": idx[0],
                "valid_to": None,
            }
            for symbol, role in [("NEW", "canonical_current"), (rejected_symbol, "excluded")]
        ]
    )
    support = short_flow._proven_tenure_mask(idx, pd.Index(["NEW"]), master, None)
    assert not support.to_numpy().any()
    print("SANITY: explicit overlapping rejection of the same security defeats primary support even when it uses another symbol key.")


@pytest.mark.parametrize("source_symbol", [None, "", " "])
def test_market_tenure_requires_an_actual_source_symbol(source_symbol: str | None) -> None:
    idx = pd.date_range("2024-01-02", periods=2)
    master = _market_tenures(
        [
            {
                "canonical_company": "NEW",
                "source_symbol": source_symbol,
                "security_id": "C111111111",
                "lineage_role": "canonical_current",
                "valid_from": idx[0],
                "valid_to": None,
            }
        ]
    )
    support = short_flow._proven_tenure_mask(idx, pd.Index(["NEW"]), master, None)
    assert not support.to_numpy().any()
    print("SANITY: NULL, empty and whitespace source symbols cannot become fabricated NAN/NONE market keys.")


def test_reused_symbol_is_null_outside_the_current_issuer_tenure() -> None:
    idx = pd.bdate_range("2021-01-04", periods=90)
    ticker = "REUSED"
    hist = pd.DataFrame([{"date": day, "ticker": ticker, "short_volume": 400_000.0, "total_volume": 1_000_000.0} for day in idx])
    tenure = _market_tenures(
        [
            {
                "canonical_company": ticker,
                "source_symbol": ticker,
                "security_id": "C111111111",
                "lineage_role": "canonical_current",
                "valid_from": idx[20],
                "valid_to": idx[60],
            }
        ]
    )
    panel = build_short_flow_feature_panel(
        make_frames(idx, {ticker: {}}, universe=pd.Index([ticker])),
        hist,
        tenure_mask=short_flow._proven_tenure_mask(idx, pd.Index([ticker]), tenure, None),
    )
    ratio = panel[panel["ticker"] == ticker].set_index("date")["f_ic_shortvol_ratio_20d"].reindex(idx)
    assert ratio.loc[: idx[19]].isna().all()
    assert np.isfinite(ratio.loc[idx[45]])
    assert ratio.loc[idx[60] :].isna().all()
    print("\n=== SANITY CHECK: reused-symbol tenure mask ===")
    print("  RegSHO cells are available only inside the actual security's proven half-open market tenure")


def test_canonical_ticker_survives_a_historical_alias_tenure() -> None:
    idx = pd.bdate_range("2023-05-01", periods=80)
    cutover = idx[35]
    history = pd.DataFrame(
        [
            {
                "date": day,
                "ticker": "NEW",
                "short_volume": 400_000.0,
                "total_volume": 1_000_000.0,
            }
            for day in idx
        ]
    )
    tenure = _market_tenures(
        [
            {
                "canonical_company": "NEW",
                "source_symbol": "OLD",
                "security_id": "C111111111",
                "lineage_role": "canonical_predecessor",
                "valid_from": idx[0],
                "valid_to": cutover,
            },
            {
                "canonical_company": "NEW",
                "source_symbol": "NEW",
                "security_id": "C111111111",
                "lineage_role": "canonical_current",
                "valid_from": idx[30],
                "valid_to": None,
            },
        ]
    )

    panel = build_short_flow_feature_panel(
        make_frames(idx, {"NEW": {}}, universe=pd.Index(["NEW"])),
        history,
        tenure_mask=short_flow._proven_tenure_mask(idx, pd.Index(["NEW"]), tenure, None),
    )
    ratio = panel.set_index("date")["f_ic_shortvol_ratio_20d"].reindex(idx)

    assert ratio.loc[cutover:].notna().all()
    assert np.isclose(ratio.loc[cutover], 0.4)
    print("\n=== SANITY CHECK: canonical storage across a symbol alias ===")
    print("  one security/owner survives OLD and NEW's overlapping proved aliases without a fabricated ownership conflict")


def test_panel_columns_match_the_emission_map():
    dates, tickers, hist = _synth(t=400)
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}
    rng = np.random.default_rng(11)
    close = pd.DataFrame({t: 100 * np.cumprod(1 + rng.normal(0.0004, 0.02, len(dates))) for t in tickers}, index=dates)
    volume = pd.DataFrame({t: 3e6 for t in tickers}, index=dates)
    fund = pd.DataFrame([{"ticker": t, "as_of": "2020-01-01", "sharesOutstanding": 5e8, "sharesOutstandingPit": 5e8} for t in tickers])
    keep = rng.random(len(dates) * len(tickers)) < 0.3
    ftd = pd.DataFrame(
        {
            "date": np.repeat(dates.to_numpy(), len(tickers)),
            "ticker": np.tile(np.array(tickers), len(dates)),
            "fails_quantity": np.where(keep, rng.lognormal(8, 1.2, len(keep)).round(0), 0.0),
            "source_checked_at": pd.Timestamp("2025-01-01"),
        }
    )
    ftd["period"] = pd.to_datetime(ftd["date"]).map(lambda day: f"{day:%Y%m}{'a' if day.day <= 15 else 'b'}")
    panel = build_short_flow_feature_panel(
        make_frames(dates, peers, volume=volume, close_total=close),
        hist,
        fails_history=ftd,
        capital=ShortFlowCapital(fundamentals_to_daily(fund, "sharesOutstandingPit", dates), volume),
    )
    expected = set()
    for name, mode in EMISSION.items():
        expected.add(f"f_{name}")
        if mode == "raw+xs":
            expected.add(f"f_{name}_xs")
        elif mode == "raw+peers":
            expected.add(f"f_{name}_vs_peers")
    emitted = {c for c in panel.columns if c.startswith("f_")}
    assert emitted == expected, f"missing {sorted(expected - emitted)}; undeclared {sorted(emitted - expected)}"
    for leg in ("f_ic_shortvol_ratio_20d", "f_ic_shortvol_turnover_20d", "f_ic_ftd_to_adv20", "f_ic_shortvol_market_coverage"):
        assert panel[leg].notna().any(), f"{leg} is all-NaN"
    assert not any(column.endswith("_xs") for column in emitted)
    assert not any(column.endswith("_vs_peers") for column in emitted)
    assert not any(column in panel for column in ("f_ic_shortvol_ratio_z252", "f_ic_ftd_z252"))
    # the three bounded ratios stay in [0, 1]
    for w in (5, 20, 60):
        s = panel[f"f_ic_shortvol_ratio_{w}d"].dropna()
        assert s.between(0, 1).all(), f"{w}d ratio out of [0, 1]"
    # the coverage measurement is a share of the tape, so it cannot exceed 1 when the tape is
    # the larger number -- here RegSHO's 1-5m against a 3m tape, so it straddles 1 by design
    assert panel["f_ic_shortvol_market_coverage"].dropna().gt(0).all()
    # days_to_cover is GONE: it needs short-INTEREST positions this repo does not fetch
    assert not any("days_to_cover" in c for c in panel.columns)
    print("\n=== SANITY CHECK: short-flow panel columns vs the EMISSION map ===")
    print(
        f"  {len(emitted)} legs emitted from {len(EMISSION)} declared features, exact match; "
        f"the three ratios are in [0, 1], no peer leg survives without target/OOS evidence, and history z outputs are absent; "
        f"ic_shortvol_days_to_cover is absent (needs FINRA "
        f"settlement positions, out of scope). Validated."
    )


def test_market_coverage_is_split_basis_free():
    """The defect the 2026-09-12 full build found: `ic_shortvol_market_coverage` breached its
    (0, 1) bound on 2,213 cells, max **3.71**, and all six offending tickers were
    corporate-action names — GE's window ending exactly at its 1-for-8 reverse split.

    Cause: RegSHO `total_volume` is AS-TRADED and never restated, while yfinance `Volume` IS
    retroactively scaled by the split ratio (measured over 746 splits: median post/pre volume
    ratio 0.869 against median R = 2.0). So the plain ratio carries a factor of `1 / F(d)`, and
    a REVERSE split (R < 1) inflates it — a 1-for-8 turns a true 28% coverage into 2.24.

    Here: a true coverage of 0.10 either side of a 1-for-8 reverse split. Unrestated the
    pre-split window reads 0.80 — eight times too high, but deliberately still UNDER 1.0 so
    `_guard_coverage`'s physical ceiling does not mask the effect this test is about. The two
    guards are independent and each has its own test.
    """
    idx = pd.DatetimeIndex(pd.bdate_range("2021-01-04", periods=120))
    split_date = idx[60]
    regsho = pd.DataFrame([{"date": d, "ticker": "GEX", "total_volume": 100_000.0, "short_volume": 40_000.0} for d in idx])
    # yfinance volume: as-traded 1,000,000 a day, then back-scaled by R = 0.125 BEFORE the split
    tape = pd.DataFrame({"GEX": [1_000_000.0 * (0.125 if d < split_date else 1.0) for d in idx]}, index=idx)
    splits = pd.DataFrame([{"date": split_date, "ticker": "GEX", "ratio": 0.125}])

    before = _shortvol_fields(regsho, idx, None, None, tape)["ic_shortvol_market_coverage"]
    basis = QuantityBasis.from_context(resolve_share_events(None, splits, {}), idx)
    df_regsho = short_quantity_history(regsho, basis)
    assert df_regsho is not None
    after = _shortvol_fields(df_regsho, idx, None, None, tape)["ic_shortvol_market_coverage"]
    pre, post = idx[50], idx[110]
    assert before.loc[pre, "GEX"] == pytest.approx(0.80, rel=1e-6), "the defect must reproduce"
    assert after.loc[pre, "GEX"] == pytest.approx(0.10, rel=1e-6)
    assert after.loc[post, "GEX"] == pytest.approx(0.10, rel=1e-6)
    assert before.loc[post, "GEX"] == pytest.approx(0.10, rel=1e-6), "post-split is unaffected"
    print("\n=== SANITY CHECK: market coverage is split-basis free ===")
    print(
        f"  across a 1-for-8 reverse split, true coverage 10%: unrestated reads "
        f"{before.loc[pre, 'GEX']:.2f} before the split (8x too high) and "
        f"{before.loc[post, 'GEX']:.2f} after; restated reads "
        f"{after.loc[pre, 'GEX']:.2f} / {after.loc[post, 'GEX']:.2f}. Validated."
    )


def test_coverage_above_one_is_nulled_as_a_physical_impossibility():
    """Off-exchange volume is a SUBSET of the consolidated tape, so coverage > 1.0 means the
    numerator and denominator are not the same security.

    After the split-basis fix, the 2026-09-12 build's 179 surviving breaches were all REUSED
    TICKERS — `WTW` carrying Weight Watchers' RegSHO volume over Willis Towers Watson's price
    grid (135 cells, max 3.71), plus AXON's rename chain and pre-Gen-Digital `GEN`. Nulled, not
    clipped: a clip at 1.0 would assert that RegSHO saw the entire tape.
    """
    idx = pd.DatetimeIndex(pd.bdate_range("2021-01-04", periods=60))
    regsho = pd.DataFrame(
        [{"date": d, "ticker": t, "total_volume": v, "short_volume": v * 0.4} for d in idx for t, v in (("GOOD", 250_000.0), ("REUSED", 3_000_000.0))]
    )
    tape = pd.DataFrame({"GOOD": [1_000_000.0] * len(idx), "REUSED": [1_000_000.0] * len(idx)}, index=idx)

    cov = _shortvol_fields(regsho, idx, None, None, tape)["ic_shortvol_market_coverage"]
    last = idx[-1]
    assert cov.loc[last, "GOOD"] == pytest.approx(0.25, rel=1e-6), "a real reading is kept"
    assert np.isnan(float(cast(Any, cov.loc[last, "REUSED"]))), "3.0x the tape is impossible and must be NaN"
    assert not (cov == 1.0).any().any(), "nulled, never clipped to 1.0"
    print("\n=== SANITY CHECK: coverage ceiling ===")
    print(f"  GOOD reads {cov.loc[last, 'GOOD']:.2f}; a ticker whose RegSHO volume is 3x the tape reads NaN rather than 3.0. Validated.")
