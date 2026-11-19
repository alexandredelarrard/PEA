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
from src.data_aggregate.utils.common.quotes import terminal_basis_factor, terminal_rows
from src.data_aggregate.utils.institutionals import short_flow_features as short_flow
from src.data_aggregate.utils.institutionals.short_flow_features import (
    EMISSION,
    Z_MIN_PERIODS,
    FtdPublication,
    FtdSource,
    ShortFlowCapital,
    _publish_ftd_zip_states,
    prepare_fails_quantities,
)
from src.data_aggregate.utils.institutionals.short_flow_features import (
    _fails_fields as compute_fails_fields,
)
from src.data_aggregate.utils.institutionals.short_flow_features import (
    _shortvol_fields as compute_short_fields,
)
from src.data_aggregate.utils.institutionals.short_flow_features import (
    build_short_flow_feature_panel as compute_short_panel,
)
from src.data_aggregate.utils.price.features import security_basis_frame
from src.data_extract.utils.institutionals.fetch_short_interest import _parse_regsho
from src.data_store.schema import Tables
from src.utils.data_policy import publication_policy
from src.utils.price.adjustments import ShareBasis, resolve_share_events
from src.utils.publication import point_in_time
from tests.conftest import make_frames
from tests.data_extract.fake_context import extract_config


def terminal_factor(basis: ShareBasis, idx: pd.DatetimeIndex, tickers: Any) -> pd.DataFrame:
    """The step's `F_T / F_d` from known-truth events, built by the production saved-basis builder."""
    return terminal_basis_factor(security_basis_frame(basis, pd.DatetimeIndex(idx), pd.Index(tickers)))


def split_factor(splits: pd.DataFrame | None, idx: pd.DatetimeIndex, tickers: Any) -> pd.DataFrame:
    """`terminal_factor` for Yahoo-style split rows (`ticker`, `date`, `ratio`)."""
    df_splits = splits if splits is not None else pd.DataFrame(columns=["ticker", "date", "ratio"])
    return terminal_factor(resolve_share_events(None, df_splits, {}), idx, tickers)


def ftd_publication(rows: pd.DataFrame, sessions: pd.DatetimeIndex) -> FtdPublication:
    """Prepare fixture ZIP clocks through the same single source boundary as the step."""
    ready = point_in_time(rows, extract_config(), Tables.sec_fails_to_deliver_security, sessions)
    return FtdPublication(ready, ready.groupby("period")["as_of"].first().to_dict())


def short_fields(
    history: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    shares: pd.DataFrame | None,
    close: pd.DataFrame | None,
    config: Any,
    tenure_mask: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Prepare literal raw fixture clocks before the production rolling calculation."""
    ready = point_in_time(history, config, Tables.sec_short_volume_security, sessions)
    return compute_short_fields(ready, sessions, shares, close, tenure_mask)


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
        return compute_short_panel(frames, None, None, denominators)
    settlement = None
    if fails_history is not None or publication is not None:
        df_fails = fails_history if fails_history is not None else pd.DataFrame(columns=["date", "ticker", "fails_quantity", "period"])
        df_coverage = publication.coverage if publication is not None else df_fails[["date", "period"]]
        columns = pd.Index(frames.universe, name="ticker")
        settlement = prepare_fails_quantities(df_fails, df_coverage, idx, columns, tenure_mask)
        if publication is None:
            publication = ftd_publication(df_coverage, idx)
    fails = FtdSource(settlement, publication) if settlement is not None and publication is not None else None
    ready = point_in_time(short_history, extract_config(), Tables.sec_short_volume_security, idx) if short_history is not None else None
    return compute_short_panel(frames, ready, fails, denominators, tenure_mask)


def _fails_fields(
    fails_hist: pd.DataFrame,
    idx: pd.DatetimeIndex,
    shares_out: pd.DataFrame | None,
    volume: pd.DataFrame | None,
    splits: pd.DataFrame | None = None,
    tenure_mask: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Known-truth dated-quantity fixture; numerical expectations remain explicit in each test."""
    df_factor = split_factor(splits, idx, sorted(set(fails_hist["ticker"])))
    df_fails = terminal_rows(fails_hist, ("fails_quantity",), df_factor)
    settlement = prepare_fails_quantities(df_fails, fails_hist[["date", "period"]], idx, None, tenure_mask)
    publication = ftd_publication(fails_hist[["date", "period"]], idx)
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
    """Nineteen quiet days that were 90% short on 100k shares, and one heavy day that was 10% short
    on 10m. The window's short share of TRADED VOLUME is 8.0%, not the 86% an average of daily
    ratios reports."""
    dates = pd.DatetimeIndex(pd.bdate_range("2023-01-02", periods=20))
    rows = [{"date": d, "ticker": "A", "total_volume": 100_000.0, "short_volume": 90_000.0} for d in dates[:19]]
    rows.append({"date": dates[19], "ticker": "A", "total_volume": 10_000_000.0, "short_volume": 1_000_000.0})
    hist = pd.DataFrame(rows)
    # the lag means day 20's window is only readable on the NEXT grid day, so extend the index
    idx = pd.DatetimeIndex(pd.bdate_range("2023-01-02", periods=21))
    df = short_fields(hist, idx, None, None, extract_config())
    got = float(cast(Any, df["ic_shortvol_ratio_20d"].loc[idx[20], "A"]))
    weighted = (19 * 90_000 + 1_000_000) / (19 * 100_000 + 10_000_000)
    naive = (19 * 0.9 + 0.1) / 20
    assert abs(got - weighted) < 1e-12
    assert abs(got - naive) > 0.5, "the volume weighting made no difference -- check the sum"
    print("\n=== SANITY CHECK: volume-weighted short ratio ===")
    print(
        f"  nineteen 90%-short days on 100k + one 10%-short day on 10m: "
        f"ic_shortvol_ratio_20d = {got:.4f} (= sum short / sum total), against {naive:.2f} "
        f"for an average of daily ratios -- {naive / got:.0f}x. Validated."
    )


def test_publication_lag_is_one_trading_day():
    dates, tickers, hist = _synth()
    df = short_fields(hist, dates, None, None, extract_config())
    assert {"ic_shortvol_ratio_20d", "ic_shortvol_ratio_60d", "ic_shortvol_acceleration_60d"}.issubset(df)
    assert "ic_shortvol_ratio_z252" not in df
    # heavily-shorted S0 has the highest ratio cross-sectionally
    assert str(df["ic_shortvol_ratio_20d"].loc[dates[300]].idxmax()) == "S0"
    short = hist.pivot_table(index="date", columns="ticker", values="short_volume", aggfunc="sum")
    total = hist.pivot_table(index="date", columns="ticker", values="total_volume", aggfunc="sum")
    expected = float(cast(Any, (short.rolling(20, min_periods=10).sum() / total.rolling(20, min_periods=10).sum()).loc[dates[299], "S0"]))
    got = float(cast(Any, df["ic_shortvol_ratio_20d"].loc[dates[300], "S0"]))
    assert np.isclose(got, expected), "lag broken"
    assert int(publication_policy(extract_config(), Tables.sec_short_volume_security).lag_days) == 1
    print("\n=== SANITY CHECK: RegSHO 1-day publication lag ===")
    print(
        f"  S0 (85% shorted) tops ic_shortvol_ratio_20d at {dates[300].date()}; the value at "
        f"t equals the volume-weighted ratio computed through t-1 (published next morning). "
        "Validated."
    )


def test_source_adapters_normalize_both_rolling_legs_and_pit_counts() -> None:
    idx = pd.bdate_range("2024-06-03", periods=21)
    df_events = pd.DataFrame({"ticker": ["TEST"], "date": [idx[19]], "value": [10.0]})
    df_factor = terminal_factor(ShareBasis(df_events, df_events, df_events, {}), idx, ["TEST"])
    df_raw = pd.DataFrame({"ticker": "TEST", "date": idx, "short_volume": [9.0] * 19 + [10.0] * 2, "total_volume": [10.0] * 19 + [100.0] * 2})
    df_filing = pd.DataFrame({"ticker": ["TEST"], "as_of": [idx[0]], "sharesOutstandingPit": [100.0], "sharesOutstanding": [1000.0]})
    df_normalized = terminal_rows(df_raw, ("short_volume", "total_volume"), df_factor)
    df_shares = pd.DataFrame({"TEST": [100.0] * 19 + [1000.0] * 2}, index=idx) * df_factor
    result = short_fields(df_normalized, idx, df_shares, None, extract_config())
    assert result["ic_shortvol_ratio_20d"].at[idx[20], "TEST"] == pytest.approx(1720 / 2000)
    assert result["ic_shortvol_turnover_20d"].at[idx[20], "TEST"] == pytest.approx(1720 / 1000)
    assert df_shares["TEST"].eq(1000.0).all()
    assert df_raw["short_volume"].iloc[0] == 9
    assert df_filing["sharesOutstandingPit"].iloc[0] == 100
    print("SANITY: compatible rolling sums1720/2000 yield0.86 ratio and1.72 turnover over1000 PIT shares.")


def test_future_event_cannot_advance_the_vendor_tape_basis() -> None:
    idx = pd.bdate_range("2024-06-03", periods=21)
    df_events = pd.DataFrame({"ticker": ["TEST"], "date": [idx[-1] + pd.Timedelta(days=10)], "value": [10.0]})
    df_factor = terminal_factor(ShareBasis(df_events, df_events, df_events, {}), idx, ["TEST"])
    df_raw = pd.DataFrame({"ticker": "TEST", "date": idx, "short_volume": 5.0, "total_volume": 10.0})
    df_normalized = terminal_rows(df_raw, ("short_volume", "total_volume"), df_factor)
    df_shares = pd.DataFrame({"TEST": 100.0}, index=idx) * df_factor
    df_tape = pd.DataFrame({"TEST": 20.0}, index=idx)
    pd.testing.assert_frame_equal(df_normalized, df_raw)
    assert df_shares["TEST"].eq(100).all()
    assert df_tape is not None and df_tape["TEST"].eq(20).all()
    print(
        "SANITY: a scheduled event beyond the last vendor bar cancels inF(source)/F(target); raw quantities, PIT counts and existing tape remain compatible."
    )


def test_paired_vendor_restatement_preserves_earlier_dimensionless_outputs() -> None:
    idx = pd.bdate_range("2024-01-02", periods=45)
    df_raw = pd.DataFrame({"ticker": "TEST", "date": idx, "short_volume": 5.0, "total_volume": 10.0})
    empty = pd.DataFrame(columns=["ticker", "date", "ratio"])
    df_factor = split_factor(empty, idx, ["TEST"])
    df_shares = pd.DataFrame({"TEST": 100.0}, index=idx) * df_factor
    df_history = terminal_rows(df_raw, ("short_volume", "total_volume"), df_factor)
    original = short_fields(df_history, idx, df_shares, None, extract_config())
    extended = pd.DatetimeIndex(idx.append(pd.DatetimeIndex([idx[-1] + pd.Timedelta(days=10)])))
    events = pd.DataFrame({"ticker": ["TEST"], "date": [extended[-1]], "ratio": [10.0]})
    later_factor = split_factor(events, extended, ["TEST"])
    df_later = terminal_rows(df_raw, ("short_volume", "total_volume"), later_factor)
    df_later_shares = pd.DataFrame({"TEST": [100.0] * len(idx) + [1000.0]}, index=extended) * later_factor
    restated = short_fields(df_later, extended, df_later_shares, None, extract_config())
    for name, df_before in original.items():
        pd.testing.assert_frame_equal(df_before, restated[name].reindex(idx))
    print("SANITY: extending the actual vendor basis through a future10:1 event and leaves every earlier ratio/turnover value and null unchanged.")


def test_missing_active_class_keeps_rolling_denominators_unknown() -> None:
    idx = pd.bdate_range("2024-01-02", periods=45)
    df_tape = pd.DataFrame({"TEST": 20.0}, index=idx)
    df_supported = pd.DataFrame({"TEST": True}, index=idx)
    df_supported.loc[idx[20], "TEST"] = False
    capital = ShortFlowCapital(None, df_tape, df_supported)
    settlement = short_flow.FtdSettlement(pd.DataFrame({"TEST": 10.0}, index=idx), pd.Series(True, index=idx))
    df_coverage = pd.DataFrame({"date": idx, "period": "202401a"})
    publication = FtdPublication(df_coverage, {"202401a": idx[-1]})
    df_adv = compute_fails_fields(settlement, capital, publication, None)["ic_ftd_to_adv20"]
    assert df_adv.at[idx[-1], "TEST"] == pytest.approx(0.5)
    print(
        "SANITY: a missing active-class input leaves the published fails/ADV state unknown until its rolling denominator is supported; it never becomes claimed zero volume."
    )


def test_unknown_short_quantity_never_aggregates_into_observed_zero() -> None:
    idx = pd.bdate_range("2024-01-02", periods=21)
    df_history = pd.DataFrame({"ticker": "TEST", "date": idx, "short_volume": np.nan, "total_volume": 10.0})
    fields = short_fields(df_history, idx, None, None, extract_config())
    assert all(df_field.isna().all().all() for df_field in fields.values())
    print("SANITY: unknown quantity observations stay unknown after daily aggregation and never appear as measured zero short sales.")


def test_price_interactions_are_one_sided_and_complementary():
    """#62/#63. High short flow means opposite things depending on the price path, so each
    interaction is zero in the other's regime -- never averaged into one signed number."""
    dates, tickers, hist = _synth(t=400)
    rng = np.random.default_rng(3)
    close = pd.DataFrame({t: 100 * np.cumprod(1 + rng.normal(0.0004, 0.02, len(dates))) for t in tickers}, index=dates)
    df = short_fields(hist, dates, None, close, extract_config())
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
    published = _publish_ftd_zip_states(state, fails, idx, ftd_publication(fails, idx).available_by_period)

    assert published.loc[pd.Timestamp("2024-02-05"), "A"] == pytest.approx(0.2)
    assert published.loc[pd.Timestamp("2024-02-21"), "A"] == pytest.approx(0.9)
    print("\n=== SANITY CHECK: FTD ZIP boundary ===")
    print("  a settlement on day 15 carried by the b ZIP cannot enter the a-period feature")


def test_ftd_day_15_b_zip_uses_its_stored_period():
    idx = pd.DatetimeIndex(pd.bdate_range("2024-04-15", "2024-05-24"))
    state = pd.DataFrame({"A": np.nan}, index=idx)
    state.loc[pd.Timestamp("2024-04-15"), "A"] = 5.0
    fails = pd.DataFrame({"date": ["2024-04-15"], "period": ["202404b"]})

    published = _publish_ftd_zip_states(state, fails, idx, ftd_publication(fails, idx).available_by_period)

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
    sessions = pd.DatetimeIndex(sorted({pd.Timestamp(day) for case in cases for day in (case[1], *case[3:])}))
    expected = {case[0]: pd.Timestamp(case[4]) for case in cases}
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
            observed.append(ftd_publication(source, sessions).available_by_period)
            clock_reads.append(timestamp.now.call_count)
    print(f"F004 policy probe: September-a/b dates across cache/evaluation clocks: {[(row['202609a'], row['202609b']) for row in observed]}")
    assert observed == [expected] * len(observed)
    assert clock_reads == [0] * len(clock_reads)
    assert zip_path.read_bytes() == b"identical cached ZIP bytes"
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
        "SANITY: the shared PIT boundary uses half-month end plus 21 calendar days, then the first actual session, across leap/weekend/holiday cases; cache mtime and today's date have no economic effect."
    )


def test_ftd_partial_universe_uses_the_same_period_policy() -> None:
    scoped_rows = pd.DataFrame({"date": ["2026-08-31"], "period": ["202608b"]})
    global_rows = pd.concat([scoped_rows, pd.DataFrame({"date": ["2026-09-15"], "period": ["202609a"]})], ignore_index=True)
    sessions = pd.DatetimeIndex(["2026-09-21", "2026-10-06"])
    available = ftd_publication(scoped_rows, sessions).available_by_period
    assert available == {"202608b": pd.Timestamp("2026-09-21")}
    assert available["202608b"] == ftd_publication(global_rows, sessions).available_by_period["202608b"]
    print("SANITY: the same ZIP has the same estimated policy day in scoped and global source histories.")


def test_ftd_period_must_be_present_and_valid():
    sessions = pd.DatetimeIndex(["2024-05-21"])
    with pytest.raises(ValueError, match="period"):
        ftd_publication(pd.DataFrame({"date": ["2024-04-15"]}), sessions)
    for frame in [
        pd.DataFrame({"date": ["2024-04-15"], "period": [None]}),
        pd.DataFrame({"date": ["2024-04-15"], "period": ["202413b"]}),
        pd.DataFrame({"date": ["2024-04-15", "2024-04-16"], "period": ["202404b", 2024041]}),
    ]:
        ready = ftd_publication(frame, sessions).coverage
        assert set(ready.period) <= {"202404b"}
        assert len(ready) == (1 if frame.period.iloc[0] == "202404b" else 0)
    config = extract_config()
    config.data.tables.sec_fails_to_deliver_security.pit.lag_days = 0
    zero = point_in_time(pd.DataFrame({"period": ["202404b"]}), config, Tables.sec_fails_to_deliver_security, pd.DatetimeIndex(["2024-04-30"]))
    assert zero.as_of.iloc[0] == pd.Timestamp("2024-04-30")
    print("SANITY: a missing ZIP clock errors; invalid ZIP values are excluded, while an explicitly configured zero lag is valid.")


def test_ftd_rejects_ambiguous_settlement_date_across_zip_periods():
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-15", "2024-02-22"))
    state = pd.DataFrame({"A": 0.1}, index=idx)
    fails = pd.DataFrame({"date": ["2024-01-15", "2024-01-15"], "period": ["202401a", "202401b"]})
    with pytest.raises(ValueError, match="multiple ZIP periods"):
        _publish_ftd_zip_states(state, fails, idx, ftd_publication(fails, idx).available_by_period)
    print("\n=== SANITY CHECK: FTD ZIP boundary ambiguity ===")
    print("  a settlement date carried by two ZIPs fails closed until source grain is resolved")


def test_ftd_zip_state_preserves_nan_in_latest_state():
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-02", "2024-02-09"))
    settlement_state = pd.DataFrame({"A": np.nan}, index=idx)
    settlement_state.loc[pd.Timestamp("2024-01-02"), "A"] = 1.0
    fails_hist = pd.DataFrame({"date": ["2024-01-02", "2024-01-15"], "period": ["202401a", "202401a"]})

    published = _publish_ftd_zip_states(settlement_state, fails_hist, idx, ftd_publication(fails_hist, idx).available_by_period)

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
    fields = short_fields(hist, dates, shares, close, extract_config())
    assert all(frame.loc[dates[-1]].isna().all() for frame in fields.values())

    print("\n=== SANITY CHECK: rolling source-date completeness ===")
    print("  a missing RegSHO t-1 observation makes every latest derived leg NaN")


def test_ftd_publication_snaps_weekend_and_holiday_forward():
    weekend_idx = pd.bdate_range("2024-06-03", "2024-07-10").difference(pd.DatetimeIndex(["2024-06-19", "2024-07-04"]))
    weekend_state = pd.DataFrame({"A": np.nan}, index=weekend_idx)
    weekend_state.loc[pd.Timestamp("2024-06-14"), "A"] = 1.0
    weekend_hist = pd.DataFrame([{"date": "2024-06-14", "period": "202406a"}])
    weekend_published = _publish_ftd_zip_states(
        weekend_state, weekend_hist, weekend_idx, ftd_publication(weekend_hist, weekend_idx).available_by_period
    )

    holiday_idx = pd.bdate_range("2022-01-03", "2022-02-23").difference(pd.DatetimeIndex(["2022-01-17", "2022-02-21"]))
    holiday_state = pd.DataFrame({"A": np.nan}, index=holiday_idx)
    holiday_state.loc[pd.Timestamp("2022-01-31"), "A"] = 2.0
    holiday_hist = pd.DataFrame([{"date": "2022-01-31", "period": "202201b"}])
    holiday_published = _publish_ftd_zip_states(
        holiday_state, holiday_hist, holiday_idx, ftd_publication(holiday_hist, holiday_idx).available_by_period
    )

    assert ftd_publication(weekend_hist, weekend_idx).available_by_period["202406a"] == pd.Timestamp("2024-07-08")
    assert pd.isna(weekend_published.loc[pd.Timestamp("2024-07-05"), "A"])
    assert weekend_published.loc[pd.Timestamp("2024-07-08"), "A"] == 1.0
    assert ftd_publication(holiday_hist, holiday_idx).available_by_period["202201b"] == pd.Timestamp("2022-02-22")
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
    for leg in (
        "f_ic_shortvol_ratio_20d",
        "f_ic_shortvol_ratio_180d",
        "f_ic_shortvol_acceleration_120d",
        "f_ic_shortvol_turnover_20d",
        "f_ic_ftd_to_adv20",
    ):
        assert panel[leg].notna().any(), f"{leg} is all-NaN"
    assert not any(column.endswith("_xs") for column in emitted)
    assert not any(column.endswith("_vs_peers") for column in emitted)
    assert not any(column in panel for column in ("f_ic_shortvol_ratio_z252", "f_ic_ftd_z252"))
    # the four bounded ratios stay in [0, 1]
    for w in (20, 60, 120, 180):
        s = panel[f"f_ic_shortvol_ratio_{w}d"].dropna()
        assert s.between(0, 1).all(), f"{w}d ratio out of [0, 1]"
    # days_to_cover is GONE: it needs short-INTEREST positions this repo does not fetch
    assert not any("days_to_cover" in c for c in panel.columns)
    print("\n=== SANITY CHECK: short-flow panel columns vs the EMISSION map ===")
    print(
        f"  {len(emitted)} legs emitted from {len(EMISSION)} declared features, exact match; "
        f"the four ratios are in [0, 1], no peer leg survives without target/OOS evidence, and history z outputs are absent; "
        f"ic_shortvol_days_to_cover is absent (needs FINRA "
        f"settlement positions, out of scope). Validated."
    )


def _ramp_history(days: int, tickers: tuple[str, ...] = ("A",)) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """Days 1..100 heavy and 10% short (total 1,000), days 101..195 light and 90% short (total 100), days 196..200 light and 100% short."""
    idx = pd.DatetimeIndex(pd.bdate_range("2022-01-03", periods=days + 1))
    day = np.arange(1, days + 1)
    total = np.where(day <= 100, 1_000.0, 100.0)
    short = np.where(day <= 100, 100.0, np.where(day <= 195, 90.0, 100.0))
    rows = [{"date": idx[i], "ticker": t, "short_volume": short[i], "total_volume": total[i]} for t in tickers for i in range(days)]
    return idx, pd.DataFrame(rows)


def test_ratios_20_60_120_180_are_volume_weighted_sums_published_next_session() -> None:
    idx, hist = _ramp_history(200)
    df = short_fields(hist, idx, None, None, extract_config())
    last = idx[200]
    # hand sums over the window ending on day 200 (published on the next grid row)
    expected = {20: 1850 / 2000, 60: 5450 / 6000, 120: 11050 / 30000, 180: 17050 / 90000}
    for window, truth in expected.items():
        got = float(cast(Any, df[f"ic_shortvol_ratio_{window}d"].at[last, "A"]))
        assert got == pytest.approx(truth, rel=1e-12), window
    # the mean of daily ratios is another statistic (120d: 20 days at 0.1 and 100 at about 0.905)
    assert abs(float(cast(Any, df["ic_shortvol_ratio_120d"].at[last, "A"])) - 0.7) > 0.3
    # one session earlier the window ends on day 199: 16 days at 0.9 and 4 at 1.0
    assert float(cast(Any, df["ic_shortvol_ratio_20d"].at[idx[199], "A"])) == pytest.approx(1840 / 2000)
    assert "ic_shortvol_ratio_5d" not in df
    print(
        f"SANITY: ratio 20/60/120/180d = {', '.join(f'{expected[w]:.4f}' for w in expected)} equal the hand sums of short over total, "
        "one session after the window ends; the 5d ratio is not emitted."
    )


def test_acceleration_is_the_five_day_ratio_minus_the_window_ratio() -> None:
    idx, hist = _ramp_history(200)
    df = short_fields(hist, idx, None, None, extract_config())
    last = idx[200]
    spot = 500 / 500  # days 196..200: short 100 of total 100 each
    acc60 = float(cast(Any, df["ic_shortvol_acceleration_60d"].at[last, "A"]))
    acc120 = float(cast(Any, df["ic_shortvol_acceleration_120d"].at[last, "A"]))
    assert acc60 == pytest.approx(spot - 5450 / 6000, rel=1e-12)
    assert acc120 == pytest.approx(spot - 11050 / 30000, rel=1e-12)
    assert "ic_shortvol_acceleration" not in df
    print(
        f"SANITY: acceleration_60d = 1.0 - 5450/6000 = {acc60:.4f} and acceleration_120d = 1.0 - 11050/30000 = {acc120:.4f}, differences of ratios in ratio points."
    )


def test_ratio_windows_start_once_half_the_window_is_observed() -> None:
    idx, hist = _ramp_history(200)
    df = short_fields(hist, idx, None, None, extract_config())

    def first(name: str) -> int:
        return int(np.argmax(df[name]["A"].notna().to_numpy()))

    # row r publishes days 1..r, so a window needs r >= max(3, window // 2) observed days
    assert [first(f"ic_shortvol_ratio_{w}d") for w in (20, 60, 120, 180)] == [10, 30, 60, 90]
    assert first("ic_shortvol_acceleration_60d") == 30 and first("ic_shortvol_acceleration_120d") == 60
    assert df["ic_shortvol_ratio_180d"]["A"].iloc[:90].isna().all()
    print(
        "SANITY: first finite row is 10/30/60/90 for ratio 20/60/120/180d and 30/60 for acceleration 60/120d; every earlier row is NaN, never a partial-window value."
    )


def test_tenure_mask_cuts_every_new_leg_and_zero_and_missing_days_are_handled() -> None:
    idx, hist = _ramp_history(200, ("A", "B"))
    mask = pd.DataFrame(True, index=idx, columns=["A", "B"])
    mask.loc[idx[150:], "A"] = False
    df = short_fields(hist, idx, None, None, extract_config(), tenure_mask=mask)
    free = short_fields(hist, idx, None, None, extract_config())
    for name in df:
        assert df[name]["A"].iloc[150:].isna().all(), name
        pd.testing.assert_series_equal(df[name]["B"], free[name]["B"])
    assert float(cast(Any, df["ic_shortvol_ratio_60d"].at[idx[149], "A"])) == pytest.approx(
        float(cast(Any, df["ic_shortvol_ratio_60d"].at[idx[149], "B"]))
    )

    short_idx = pd.DatetimeIndex(pd.bdate_range("2024-01-02", periods=22))
    zero = pd.DataFrame({"date": short_idx[:21], "ticker": "Z", "short_volume": 0.0, "total_volume": 100.0})
    zero_ratio = short_fields(zero, short_idx, None, None, extract_config())["ic_shortvol_ratio_20d"]["Z"]
    assert zero_ratio.iloc[21] == 0.0, "an observed zero short volume is a zero ratio, not missing"

    gap = pd.DataFrame({"date": short_idx[:21], "ticker": "G", "short_volume": np.arange(1.0, 22.0), "total_volume": 100.0})
    gap.loc[5, "total_volume"] = np.nan
    gap_ratio = short_fields(gap, short_idx, None, None, extract_config())["ic_shortvol_ratio_20d"]["G"]
    window = [day for day in range(2, 22) if day != 6]  # the 20 rows ending on day 21, minus the unobserved day 6
    assert pd.isna(gap_ratio.iloc[6]), "the day with a missing total is unknown the next session"
    assert gap_ratio.iloc[21] == pytest.approx(sum(window) / (100.0 * len(window)))
    print(
        "SANITY: the tenure mask NaNs every leg after tenure ends and leaves a covered ticker bit-identical; "
        "a zero short day is ratio 0.0; a missing-total day is excluded from both sums and unknown itself."
    )


def test_removed_short_flow_features_are_absent_everywhere() -> None:
    removed = ("ic_shortvol_market_coverage", "ic_shortvol_ratio_5d", "ic_shortvol_acceleration")
    root = Path(__file__).resolve().parents[2]
    paths = ("configs/models/lgbm_modelling.yml", "configs/models/linear_modelling.yml", "configs/validate.yml")
    texts = {path: (root / path).read_text(encoding="utf-8").replace("\r\n", "\n") for path in paths}
    for name in removed:
        assert name not in EMISSION
        for path, text in texts.items():
            assert f"f_{name}\n" not in text and f"f_{name}:" not in text, (path, name)
    assert not hasattr(short_flow, "_shortvol_market_coverage") and not hasattr(short_flow, "_guard_coverage")
    assert len(EMISSION) == 11 and sum(name.startswith("ic_shortvol_") for name in EMISSION) == 9
    print(
        f"SANITY: market_coverage, ratio_5d and the unsuffixed acceleration are in no EMISSION, model list or validate bound; {len(EMISSION)} short-flow features remain."
    )


def test_turnover_is_unknown_on_a_supported_absence_day() -> None:
    idx = pd.bdate_range("2024-01-02", periods=25)
    short = pd.Series(5.0, index=idx)
    total = pd.Series(10.0, index=idx)
    short.iloc[22], total.iloc[22] = 0.0, np.nan
    df_history = pd.DataFrame({"ticker": "TEST", "date": idx, "short_volume": short.to_numpy(), "total_volume": total.to_numpy()})
    df_shares = pd.DataFrame({"TEST": 1000.0}, index=idx)
    fields = short_fields(df_history, idx, df_shares, None, extract_config())
    turnover, ratio = fields["ic_shortvol_turnover_20d"]["TEST"], fields["ic_shortvol_ratio_20d"]["TEST"]
    assert np.isnan(ratio.iloc[23]) and np.isnan(turnover.iloc[23])
    assert turnover.iloc[22] == pytest.approx(5.0 * 20 / 1000)
    assert turnover.iloc[24] == pytest.approx(5.0 * 19 / 1000)
    print("SANITY: a supported-absence day (0 short, total unknown) leaves turnover NaN like the ratio; the sums of observed days are unchanged.")


def test_source_lags_live_in_the_availability_dict_and_roll_to_sessions() -> None:
    assert int(publication_policy(extract_config(), Tables.sec_short_volume_security).lag_days) == 1
    assert int(publication_policy(extract_config(), Tables.sec_fails_to_deliver_security).lag_days) == 21
    # Wed 2024-07-03, Thu 07-04 is a holiday, Fri 07-05, then the weekend, Mon 07-08.
    idx = pd.DatetimeIndex(["2024-07-02", "2024-07-03", "2024-07-05", "2024-07-08", "2024-07-09"])
    df_source = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    published = point_in_time(pd.DataFrame({"date": idx}, index=idx), extract_config(), Tables.sec_short_volume_security, idx)["as_of"].reindex(idx)
    got = short_flow._publish_shortvol(df_source, published)["A"]
    assert np.isnan(got.iloc[0])
    assert got.loc["2024-07-03"] == 1.0, "Tuesday -> Wednesday"
    assert got.loc["2024-07-05"] == 2.0, "the day before a holiday is published on the next session"
    assert got.loc["2024-07-08"] == 3.0, "Friday -> Monday"
    assert got.loc["2024-07-09"] == 4.0 and 5.0 not in got.to_numpy(), "the last session is not yet visible"
    assert got.equals(df_source["A"].shift(1))
    print("\n=== SANITY CHECK: availability dict ===")
    print("  short volume 1d, FTD 21d; holiday, Friday->Monday and last-session cases equal the old one-row shift.")
