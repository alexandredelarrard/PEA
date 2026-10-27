"""Saved price-capital inputs and sparse EPS-unit boundaries."""

from __future__ import annotations

import logging
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.transformers import step_cube_fundamentals as module
from src.data_aggregate.transformers.step_cube_fundamentals import StepCubeFundamentals
from src.data_aggregate.utils.fundamentals.dividend_features import _dividend_fields
from src.data_aggregate.utils.fundamentals.fundamental_features import _derived_fields
from src.data_aggregate.utils.fundamentals.intrinsic import intrinsic_value_daily
from src.utils.price.adjustments import ShareBasis
from src.utils.price.per_share import eps_price, ordinary_per_share_to_security, vendor_ordinary_per_share
from tests.conftest import make_frames
from tests.data_aggregate.fundamental_pit_reference import known_no_share_events


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA"],
            "as_of": ["2024-01-02"],
            "totalRevenue": [100.0],
            "netIncome": [10.0],
            "netIncomeCommon": [10.0],
            "freeCashflow": [8.0],
            "sharesOutstanding": [100.0],
            "revenueGrowth": [0.05],
            "dividendsPaid": [200.0],
            "stockholdersEquity": [50.0],
        }
    )


def test_step_projects_all_saved_capital_inputs() -> None:
    assert {"close_pit", "shares_outstanding_pit", "mcap_pit"}.issubset(StepCubeFundamentals._FIELDS)
    print("SANITY: fundamentals requests all saved PIT capital inputs in its price projection.")


def test_earnings_gate_has_no_price_resolution_or_raw_source_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    days = pd.bdate_range("2024-01-02", periods=3)
    frames = make_frames(days, close_split=pd.DataFrame({"AAA": 100.0}, index=days), level_factor=pd.DataFrame({"AAA": 1.0}, index=days))
    step = StepCubeFundamentals.__new__(StepCubeFundamentals)
    step._store = SimpleNamespace(load=lambda *args, **kwargs: pytest.fail("EPS gate reloaded an action/price source"))
    step._context = SimpleNamespace(store=step._store, config_dir="configs", log=logging.getLogger(__name__))
    monkeypatch.setattr(module, "build_earnings_feature_panel", lambda *args, **kwargs: pd.DataFrame({"basis": [kwargs["share_basis"]]}))
    basis = known_no_share_events()
    df_result = step._earnings_panel(frames, pd.DataFrame({"earnings_date": [days[0]]}), None, None, basis)
    assert df_result is not None
    assert df_result.iloc[0]["basis"] is basis
    print("SANITY: EPS eligibility uses only dated quote units and saved price inputs, without replaying price resolution.")


def test_fundamental_yields_use_saved_cap_and_preserve_unsupported_cells() -> None:
    days = pd.bdate_range("2024-01-02", periods=2)
    prices = make_frames(
        days,
        close_split=pd.DataFrame({"AAA": 2.0}, index=days),
        close_pit=pd.DataFrame({"AAA": 200.0}, index=days),
        shares_outstanding_pit=pd.DataFrame({"AAA": 100.0}, index=days),
        mcap_pit=pd.DataFrame({"AAA": [10_000.0, np.nan]}, index=days),
    )
    fields = _derived_fields(_history(), days, prices=prices)
    assert fields["earnings_yield"].iloc[0, 0] == pytest.approx(0.001)
    assert pd.isna(fields["earnings_yield"].iloc[1, 0])
    print("SANITY: saved issuer cap10,000 determines earnings yield0.001; its null successor cannot fall back to quote times cached shares.")


def test_intrinsic_uses_independently_supported_saved_count_and_cap() -> None:
    days = pd.bdate_range("2024-01-02", periods=3)
    prices = make_frames(
        days,
        close_pit=pd.DataFrame({"AAA": 2.0}, index=days),
        shares_outstanding_pit=pd.DataFrame({"AAA": [500.0, 500.0, np.nan]}, index=days),
        mcap_pit=pd.DataFrame({"AAA": [1_000.0, 20_000.0, np.nan]}, index=days),
    )
    fields = intrinsic_value_daily(_history(), prices, days)
    assert fields["per_share"].iloc[0, 0] == pytest.approx(fields["total"].iloc[0, 0] / 500.0)
    assert fields["yield"].iloc[1, 0] == pytest.approx(fields["total"].iloc[1, 0] / 20_000.0)
    assert pd.isna(fields["per_share"].iloc[1, 0])
    assert pd.isna(fields["per_share"].iloc[2, 0]) and pd.isna(fields["yield"].iloc[2, 0])
    print(
        "SANITY: issuer yield uses independently saved cap; per-security DCF also requires matching count coverage, and unsupported inputs stay null."
    )


def test_dividend_cash_fallback_uses_saved_cap_and_keeps_missing_payer_unknown() -> None:
    days = pd.bdate_range("2024-01-02", periods=2)
    prices = make_frames(days, close_split=pd.DataFrame({"AAA": 100.0}, index=days), mcap_pit=pd.DataFrame({"AAA": [20_000.0, np.nan]}, index=days))
    df_dividends = pd.DataFrame({"ticker": ["AAA"], "date": [days[0]], "dividends": [0.0]})
    fields = _dividend_fields(df_dividends, prices, _history())
    assert fields["dividend_yield"].iloc[0, 0] == pytest.approx(0.01)
    assert pd.isna(fields["dividend_yield"].iloc[1, 0])
    print("SANITY: known cash payer200 uses saved cap20,000 for yield1%; unsupported cap stays unknown, not a nonpayer zero.")


def test_enterprise_value_dilution_requires_supported_matching_coverage() -> None:
    days = pd.bdate_range("2024-01-02", periods=3)
    prices = make_frames(
        days,
        close_split=pd.DataFrame({"AAA": 999_999.0}, index=days),
        close_pit=pd.DataFrame({"AAA": 2.0}, index=days),
        shares_outstanding_pit=pd.DataFrame({"AAA": 1_000.0}, index=days),
        mcap_pit=pd.DataFrame({"AAA": [2_000.0, 3_000.0, np.nan]}, index=days),
    )
    df_history = _history().assign(sharesOutstanding=1_000.0, dilutedShares=1_100.0, ebitda=20.0)
    fields = _derived_fields(df_history, days, prices=prices)
    assert fields["ebitda_to_ev"].iloc[0, 0] == pytest.approx(20.0 / 2_200.0)
    assert fields["ebitda_to_ev"].iloc[1, 0] == pytest.approx(20.0 / 3_000.0)
    assert pd.isna(fields["ebitda_to_ev"].iloc[2, 0])
    print("SANITY: unitary supported cap retains10% dilution; issuer/class coverage mismatch retains issuer cap, and missing cap stays unknown.")


def test_carried_capital_does_not_reset_the_public_share_observation_age() -> None:
    days = pd.DatetimeIndex(["2024-01-02", "2024-07-08"])
    prices = make_frames(
        days,
        close_split=pd.DataFrame({"AAA": 100.0}, index=days),
        close_pit=pd.DataFrame({"AAA": 100.0}, index=days),
        shares_outstanding_pit=pd.DataFrame({"AAA": 100.0}, index=days),
        mcap_pit=pd.DataFrame({"AAA": 10_000.0}, index=days),
    )
    df_history = pd.concat([_history(), _history().assign(as_of="2024-05-01", netIncome=20.0, sharesOutstanding=np.nan)], ignore_index=True)
    fields = _derived_fields(df_history, days, prices=prices)
    assert fields["earnings_yield"].iloc[0, 0] == pytest.approx(0.001)
    assert pd.isna(fields["earnings_yield"].iloc[1, 0])
    print("SANITY: fresh price/cash-flow observations cannot refresh a share observation older than185 days.")


def _stock_dividend_basis(day: pd.Timestamp) -> ShareBasis:
    df_actual = pd.DataFrame({"ticker": ["AAA"], "date": [day], "value": [1.1]})
    df_empty = df_actual.iloc[:0].copy()
    return ShareBasis(df_actual, df_empty, df_empty, {})


def test_genuine_stock_dividend_without_vendor_restatement_is_not_issuance() -> None:
    days = pd.DatetimeIndex(["2024-02-02"])
    df_history = pd.concat(
        [
            _history().assign(as_of="2023-02-01", fiscal_end="2022-12-31", sharesOutstanding=100.0),
            _history().assign(as_of="2024-02-01", fiscal_end="2023-12-31", sharesOutstanding=110.0),
        ],
        ignore_index=True,
    )
    df_before = df_history.copy()
    prices = make_frames(
        days,
        close_split=pd.DataFrame({"AAA": 100.0 / 1.1}, index=days),
        close_pit=pd.DataFrame({"AAA": 100.0 / 1.1}, index=days),
        shares_outstanding_pit=pd.DataFrame({"AAA": 110.0}, index=days),
        mcap_pit=pd.DataFrame({"AAA": 10_000.0}, index=days),
    )
    fields = _derived_fields(df_history, days, prices=prices, share_basis=_stock_dividend_basis(pd.Timestamp("2023-06-15")))
    assert fields["shares_growth"].iloc[0, 0] == pytest.approx(0.0)
    pd.testing.assert_frame_equal(df_history, df_before)
    print("SANITY: G1.1 with vendorF1 rebases both fiscal observations to110 shares; share growth is0%, not false10% issuance.")


def test_dividend_buyback_leg_uses_genuine_comparison_units() -> None:
    days = pd.bdate_range("2023-01-02", periods=254)
    df_history = pd.concat(
        [
            _history().assign(as_of=days[0], sharesOutstanding=100.0, dividendsPaid=0.0),
            _history().assign(as_of=days[252], sharesOutstanding=110.0, dividendsPaid=0.0),
        ],
        ignore_index=True,
    )
    prices = make_frames(days, close_split=pd.DataFrame({"AAA": 100.0 / 1.1}, index=days), mcap_pit=pd.DataFrame({"AAA": 10_000.0}, index=days))
    df_dividends = pd.DataFrame({"ticker": ["AAA"], "date": [days[0]], "dividends": [0.0]})
    fields = _dividend_fields(df_dividends, prices, df_history, share_basis=_stock_dividend_basis(days[252]))
    assert fields["shareholder_yield"].loc[days[252], "AAA"] == pytest.approx(0.0)
    print("SANITY: a10% stock dividend with unrestated vendor counts creates neither a buyback nor a dilution penalty.")


def test_financial_eps_matches_the_same_ordinary_vendor_price_basis() -> None:
    days = pd.bdate_range("2024-01-02", periods=2)
    prices = make_frames(
        days,
        close_split=pd.DataFrame({"AAA": 50.0}, index=days),
        level_factor=pd.DataFrame({"AAA": 2.0}, index=days),
        close_pit=pd.DataFrame({"AAA": 100.0}, index=days),
        shares_outstanding_pit=pd.DataFrame({"AAA": 100.0}, index=days),
        mcap_pit=pd.DataFrame({"AAA": 10_000.0}, index=days),
    )
    fields = _derived_fields(_history().assign(dilutedShares=100.0, epsDiluted=0.1), days, prices=prices, share_basis=known_no_share_events())
    assert fields["earnings_yield"].iloc[0, 0] == pytest.approx(0.001)
    assert fields["eps_yield"].iloc[0, 0] == pytest.approx(0.001)
    print("SANITY: same-scope EPS and total earnings yield both equal0.1% using the matched C50-times-S2 price leg.")


def test_financial_eps_has_no_stock_dividend_yield_jump_without_vendor_restatement() -> None:
    days = pd.bdate_range("2024-01-02", periods=3)
    quote = [100.0, 100.0 / 1.1, 100.0 / 1.1]
    prices = make_frames(
        days,
        close_split=pd.DataFrame({"AAA": 50.0}, index=days),
        level_factor=pd.DataFrame({"AAA": np.asarray(quote) / 50.0}, index=days),
        close_pit=pd.DataFrame({"AAA": quote}, index=days),
        shares_outstanding_pit=pd.DataFrame({"AAA": [100.0, 110.0, 110.0]}, index=days),
        mcap_pit=pd.DataFrame({"AAA": 10_000.0}, index=days),
    )
    df_history = _history().assign(dilutedShares=100.0, epsDiluted=0.1)
    fields = _derived_fields(df_history, days, prices=prices, share_basis=_stock_dividend_basis(days[1]))
    assert fields["eps_yield"]["AAA"].tolist() == pytest.approx([0.001] * 3)
    print("SANITY: G1.1/F1 lowers carried canonical EPS and the quoted price together; EPS yield remains0.1% after the stock dividend.")


def test_shared_per_share_conversion_preserves_negative_and_zero_values() -> None:
    source_date, target_date = pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04")
    df_events = pd.DataFrame({"ticker": ["AAA"], "date": [pd.Timestamp("2024-01-03")], "value": [4.0]})
    basis = ShareBasis(df_events, df_events, df_events, {})
    values = pd.Series([-0.25, 0.0])
    tickers, source, target = (
        pd.Series("AAA", index=values.index),
        pd.Series(source_date, index=values.index),
        pd.Series(target_date, index=values.index),
    )
    terminal = vendor_ordinary_per_share(basis, values, tickers, source, target)
    actual = ordinary_per_share_to_security(basis, terminal, tickers, target, source)
    assert terminal.tolist() == [-0.25, 0.0]
    assert actual.tolist() == [-1.0, 0.0]
    print("SANITY: signed EPS-0.25 and zero survive inverse unit conversion; original pre-split EPS is-1.0 and0, without quote-positive filtering.")


def test_invalid_new_financial_eps_clears_the_prior_public_state() -> None:
    days = pd.bdate_range("2024-01-02", periods=3)
    prices = make_frames(days, close_pit=pd.DataFrame({"AAA": 100.0}, index=days))
    df_history = pd.concat(
        [
            _history().assign(epsDiluted=0.1),
            _history().assign(as_of=days[1], epsDiluted=np.nan),
        ],
        ignore_index=True,
    )
    fields = _derived_fields(df_history, days, prices=prices, share_basis=known_no_share_events())
    assert fields["eps_yield"].iloc[0, 0] == pytest.approx(0.001)
    assert fields["eps_yield"].iloc[1:, 0].isna().all()
    print("SANITY: an invalid newly public EPS ends the previous state instead of reviving an older estimate.")


def test_archive_eps_daily_gate_rejects_unresolved_units_without_policy() -> None:
    days = pd.bdate_range("2024-01-02", periods=2)
    df_close = pd.DataFrame({"AAA": 50.0}, index=days)
    basis = replace(known_no_share_events(), unknown_units=frozenset({"AAA"}))
    df_price = eps_price(df_close, df_close * 0.0 + 2.0, basis)
    assert df_price["AAA"].isna().all()
    print("SANITY: unresolved quote units cannot bypass the archive EPS daily eligibility gate when no explicit policy exists.")


@pytest.mark.parametrize("field", ["diluted_shares_growth", "dps_growth"])
def test_vendor_comparison_fields_have_no_mechanical_stock_dividend_growth(field: str) -> None:
    days = pd.DatetimeIndex(["2024-02-02"])
    df_history = pd.concat(
        [
            _history().assign(as_of="2023-02-01", fiscal_end="2022-12-31", dilutedShares=100.0, dividendsPerShare=0.5),
            _history().assign(as_of="2024-02-01", fiscal_end="2023-12-31", dilutedShares=110.0, dividendsPerShare=0.5 / 1.1),
        ],
        ignore_index=True,
    )
    df_before = df_history.copy()
    fields = _derived_fields(df_history, days, share_basis=_stock_dividend_basis(pd.Timestamp("2023-06-15")))
    assert fields[field].iloc[0, 0] == pytest.approx(0.0)
    pd.testing.assert_frame_equal(df_history, df_before)
    print(f"SANITY: {field} compares the same economic unit across G1.1/F1; raw vendor weighted counts and DPS remain untouched.")


def test_piotroski_masks_only_new_unit_uncertainty_in_existing_comparison_pairs() -> None:
    days = pd.DatetimeIndex(["2023-02-02", "2024-02-02"])
    df_history = pd.concat(
        [
            _history().assign(as_of="2023-02-01", fiscal_end="2022-12-31", totalAssets=100.0, operatingCashFlow=15.0),
            _history().assign(as_of="2024-02-01", fiscal_end="2023-12-31", totalAssets=100.0, operatingCashFlow=15.0),
        ],
        ignore_index=True,
    )
    df_history = df_history.assign(longTermDebt=30.0, currentAssets=50.0, currentLiabilities=25.0, grossMargins=0.4)
    fields = _derived_fields(df_history, days, share_basis=None)
    assert fields["piotroski_f_score"].iloc[0, 0] == pytest.approx(3.0)
    assert np.isnan(fields["piotroski_f_score"].iloc[1, 0])
    print("SANITY: the first partial F-score keeps3/9; a later raw count pair with unresolved comparison units is unknown, with no raw fallback.")
