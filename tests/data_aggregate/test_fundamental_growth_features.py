"""Known-truth arithmetic, publication states and domains for the approved growth features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.fundamentals.feature_views import FEATURE_COLUMNS, HISTORY_FEATURES
from src.data_aggregate.utils.fundamentals.fundamental_features import _derived_fields, build_fundamental_feature_panel
from src.data_store.schema import Tables, name_of
from src.utils.config import read_config
from src.validate.spec import load_spec


def _history() -> pd.DataFrame:
    rows = []
    cases = {
        "NARROW": (-20.0, -10.0),
        "DEEP": (-10.0, -20.0),
        "POS": (20.0, 10.0),
        "TURN": (-10.0, 10.0),
        "LOSS": (10.0, -10.0),
        "ZERO": (0.0, 10.0),
        "MISS": (np.nan, 10.0),
        "INF": (20.0, np.inf),
    }
    for ticker, (old_income, income) in cases.items():
        for year, revenue, net_income in [(2021, 100.0, old_income), (2023, 125.0, old_income), (2024, 172.8, income)]:
            rows.append(
                dict(
                    ticker=ticker,
                    as_of=f"{year}-05-01",
                    fiscal_end=f"{year}-03-31",
                    totalRevenue=revenue,
                    netIncome=net_income,
                    totalAssets=200.0,
                    cash=30.0,
                    freeCashflow=-20.0,
                    sector="Information Technology",
                )
            )
    return pd.DataFrame(rows)


def test_signed_annual_growth_and_three_year_cagr_exact() -> None:
    idx = pd.DatetimeIndex(["2024-04-30", "2024-05-01"])
    fields = _derived_fields(_history(), idx, close=None)
    assert "y_earnings_growth_signed" in fields
    assert "revenue_cagr_3y" in fields
    expected = {"NARROW": 0.5, "DEEP": -1.0, "POS": -0.5, "TURN": 2.0, "LOSS": -2.0}
    for ticker, value in expected.items():
        assert fields["y_earnings_growth_signed"].loc[idx[-1], ticker] == pytest.approx(value)
    for ticker in ["ZERO", "MISS", "INF"]:
        assert pd.isna(fields["y_earnings_growth_signed"].loc[idx[-1], ticker])
    assert fields["revenue_cagr_3y"].loc[idx[-1], "POS"] == pytest.approx(0.2)
    assert fields["revenue_cagr_3y"].loc[idx[0]].isna().all()
    assert fields["y_earnings_growth_signed"].loc[idx[0]].isna().all()
    assert not {"earningsGrowth", "y_earnings_growth"} & fields.keys()
    print("SANITY: -20 to -10 is +50%, worsening losses are negative, 100 to 172.8 over three fiscal years is 20% CAGR; nothing leaks before filing.")


def test_cagr_invalid_replacement_clears_state_and_future_amendments_do_not_leak() -> None:
    history = _history().query("ticker == 'POS'").copy()
    amendment = history.iloc[[0]].assign(as_of="2024-06-01", totalRevenue=50.0)
    new_report = history.iloc[[-1]].assign(as_of="2024-07-01", totalRevenue=200.0)
    invalid = history.iloc[[-1]].assign(as_of="2024-08-01", totalRevenue=-1.0)
    history = pd.concat([history, amendment, new_report, invalid], ignore_index=True)
    idx = pd.DatetimeIndex(["2024-05-01", "2024-07-01", "2024-08-01", "2024-11-01"])
    full = _derived_fields(history, idx, close=None)["revenue_cagr_3y"]
    prefix = _derived_fields(history.loc[pd.to_datetime(history.as_of) <= idx[0]], idx[:1], close=None)["revenue_cagr_3y"]
    assert full.loc[idx[0], "POS"] == pytest.approx(0.2)
    assert full.loc[idx[1], "POS"] == pytest.approx(4 ** (1 / 3) - 1)
    assert full.loc[idx[2] :, "POS"].isna().all()
    pd.testing.assert_frame_equal(full.loc[idx[:1]], prefix)
    print("SANITY: a future prior-period amendment cannot change the published prefix; a newly invalid revenue report ends the prior CAGR.")


@pytest.mark.parametrize("prior,current", [(0.0, 100.0), (-100.0, 100.0), (-100.0, -172.8), (100.0, 0.0), (100.0, np.inf), (np.inf, 100.0)])
def test_cagr_requires_positive_finite_endpoints(prior: float, current: float) -> None:
    history = _history().query("ticker == 'POS'").copy()
    history.loc[history.fiscal_end == "2021-03-31", "totalRevenue"] = prior
    history.loc[history.fiscal_end == "2024-03-31", "totalRevenue"] = current
    fields = _derived_fields(history, pd.DatetimeIndex(["2024-05-01"]), close=None)
    assert pd.isna(fields["revenue_cagr_3y"].iloc[0, 0])
    print("SANITY: zero, negative and infinite revenue endpoints cannot become a valid CAGR.")


def test_cash_runway_domains_visibility_and_invalid_replacement() -> None:
    rows = []
    for ticker, cash, fcf, sector in [
        ("BURN", 30.0, -20.0, "Information Technology"),
        ("ZERO", 0.0, -20.0, "Industrials"),
        ("POS", 30.0, 20.0, "Industrials"),
        ("FLAT", 30.0, 0.0, "Industrials"),
        ("NEG", -1.0, -20.0, "Industrials"),
        ("MISS", np.nan, -20.0, "Industrials"),
        ("INF", np.inf, -20.0, "Industrials"),
        ("BANK", 30.0, -20.0, "Financials"),
        ("UNK", 30.0, -20.0, None),
    ]:
        rows.append(dict(ticker=ticker, as_of="2024-05-01", fiscal_end="2024-03-31", cash=cash, freeCashflow=fcf, sector=sector))
    history = pd.DataFrame(rows)
    history = pd.concat([history, history.loc[history.ticker == "BURN"].assign(as_of="2024-06-01", cash=np.nan)], ignore_index=True)
    idx = pd.DatetimeIndex(["2024-04-30", "2024-05-01", "2024-06-01"])
    fields = _derived_fields(history, idx, close=None)
    assert "cash_runway_months" in fields
    runway = fields["cash_runway_months"]
    assert runway.loc[idx[0]].isna().all()
    assert runway.loc[idx[1], "BURN"] == pytest.approx(18.0)
    assert runway.loc[idx[1], "ZERO"] == 0.0
    for ticker in ["POS", "FLAT", "NEG", "MISS", "INF", "BANK", "UNK"]:
        assert pd.isna(runway.loc[idx[1], ticker])
    assert pd.isna(runway.loc[idx[-1], "BURN"])
    print(
        "SANITY: cash 30 and annual burn 20 give 18 months; observed zero cash gives zero; financial/unknown sectors, invalid inputs and stale replacements remain unknown."
    )


def test_new_features_expire_and_missing_fiscal_year_is_not_bridged() -> None:
    history = _history().query("ticker == 'POS'").copy()
    idx = pd.DatetimeIndex(["2024-05-01", "2024-11-02", "2024-11-03"])
    fields = _derived_fields(history, idx, close=None)
    for name in ["revenue_cagr_3y", "y_earnings_growth_signed", "cash_runway_months"]:
        assert pd.notna(fields[name].loc[idx[1], "POS"])
        assert pd.isna(fields[name].loc[idx[2], "POS"])
    history.loc[history.fiscal_end == "2021-03-31", "fiscal_end"] = "2020-12-31"
    fields = _derived_fields(history, idx[:1], close=None)
    assert fields["revenue_cagr_3y"].isna().all().all()
    undated = _derived_fields(history.drop(columns="fiscal_end"), idx[:1], close=None)
    assert undated["revenue_cagr_3y"].isna().all().all()
    assert undated["y_earnings_growth_signed"].isna().all().all()
    print("SANITY: each feature expires after source day 185 and the CAGR cannot borrow a different fiscal quarter.")


def test_output_schema_keeps_one_signed_annual_growth_and_preserves_history_view() -> None:
    idx = pd.bdate_range("2024-05-01", periods=6)
    panel = build_fundamental_feature_panel(_history(), {}, idx, hist_window=3, hist_min_periods=2)
    expected = {"f_revenue_cagr_3y", "f_y_earnings_growth_signed", "f_cash_runway_months"}
    assert expected <= set(panel)
    assert expected <= set(FEATURE_COLUMNS)
    assert "y_earnings_growth_signed" in HISTORY_FEATURES
    assert "f_y_earnings_growth_signed_vs_hist" in FEATURE_COLUMNS
    assert not {"f_earningsGrowth", "f_earningsGrowth_vs_hist", "f_y_earnings_growth"} & set(FEATURE_COLUMNS)
    assert not {"f_earningsGrowth", "f_earningsGrowth_vs_hist", "f_y_earnings_growth"} & set(panel)
    assert all(panel[name].dtype == np.float32 for name in expected)
    print(
        "SANITY: the fixed schema publishes both additions and one canonical signed annual-growth raw/history contract; obsolete aliases are absent."
    )


def test_generic_validator_measures_new_features_instead_of_abstaining() -> None:
    spec = load_spec(read_config("./configs"), Tables.cube_part_fundamentals)
    features = ["f_revenue_cagr_3y", "f_y_earnings_growth_signed", "f_y_earnings_growth_signed_vs_hist", "f_cash_runway_months"]
    for feature in features:
        sources = [source for prefix, source in spec.pit_sources.items() if feature.startswith(prefix)]
        assert sources == [(name_of(Tables.fundamentals_history),)]
    assert spec.bounds["f_revenue_cagr_3y"] == (-1.0, np.inf)
    assert spec.bounds["f_cash_runway_months"] == (0.0, np.inf)
    assert "f_y_earnings_growth_signed" not in spec.bounds
    print("SANITY: generic PIT validation measures all four output legs against the merged filing clock, with only natural CAGR/runway bounds.")
