"""Source-vintage proof for ephemeral financial share-block conversions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.fundamentals.dividend_features import _dividend_fields
from src.data_aggregate.utils.fundamentals.fundamental_features import _financial_eps_yield
from src.data_aggregate.utils.price.inputs import load_fundamental_share_observations
from src.data_aggregate.utils.price.observations import fundamental_share_observations
from src.utils.cutover_continuity import PredecessorSeries
from src.utils.price import basis as basis_module
from src.utils.price.adjustments import ShareBasis, quantity_basis
from tests.conftest import make_frames


def _sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    quarters = pd.to_datetime(["2024-12-31", "2025-03-31", "2025-06-30", "2025-09-30"])
    raw = pd.DataFrame(
        {
            "ticker": "APH",
            "date": quarters + pd.Timedelta(days=24),
            "reportperiod": quarters,
            "calendardate": quarters,
            "lastupdated": pd.Timestamp("2026-08-01"),
            "sharesbas": 1200.0,
            "shareswa": [1000.0, 1100.0, 1200.0, 1300.0],
            "shareswadil": [1100.0, 1200.0, 1300.0, 1400.0],
            "dps": [0.1, 0.2, 0.3, 0.4],
        }
    )
    history = pd.DataFrame(
        {
            "ticker": ["APH"],
            "as_of": [raw.date.iloc[-1]],
            "fiscal_end": [quarters[-1]],
            "sharesOutstanding": [1200.0],
            "basicShares": [1150.0],
            "dilutedShares": [1250.0],
            "epsDiluted": [3.0],
            "netIncome": [3750.0],
            "dividendsPerShare": [1.0],
            "optionOverhang": [1250 / 1150 - 1],
        }
    )
    return history, raw


def _basis() -> ShareBasis:
    events = pd.DataFrame({"ticker": ["APH"], "date": [pd.Timestamp("2026-09-03")], "ratio": [2.0]})
    return quantity_basis(None, events, {}, ["APH"])


def test_coherent_raw_vintage_restores_entire_supported_share_block() -> None:
    history, raw = _sources()
    original = history.copy(deep=True)
    result = fundamental_share_observations(history, raw, _basis(), ())
    assert result.iloc[0].sharesOutstanding == 2400
    assert result.iloc[0].dilutedShares == 2500
    assert result.iloc[0].basicShares == 2300
    assert result.iloc[0].epsDiluted == 1.5
    assert result.iloc[0].dividendsPerShare == 0.5
    pd.testing.assert_frame_equal(history, original)
    assert result.iloc[0]._source_shares_outstanding == 1200
    days = pd.bdate_range(history.as_of.iloc[0], periods=2)
    frames = make_frames(days, close_pit=pd.DataFrame({"APH": 100.0}, index=days))
    yield_frame = _financial_eps_yield(result, frames, days, _basis())
    assert yield_frame.iloc[0, 0] == pytest.approx(0.03)
    print("SANITY: a proven pre-announcement vintage becomes current-vendor units without mutating the raw financial history.")


def test_mixed_quarter_vintages_null_only_ttm_share_unit_legs() -> None:
    history, raw = _sources()
    raw.loc[2:, "lastupdated"] = pd.Timestamp("2026-09-03")
    result = fundamental_share_observations(history, raw, _basis(), ())
    assert result.iloc[0].sharesOutstanding == 1200
    assert result[["basicShares", "dilutedShares", "epsDiluted", "dividendsPerShare", "optionOverhang"]].isna().all().all()
    assert result.iloc[0].netIncome == 3750
    print(
        "SANITY: mixed representation vintages cannot borrow the latest quarter factor for the TTM, while point counts and USD economics remain independent."
    )


def test_unmatched_weighted_source_cannot_validate_merged_eps() -> None:
    history, raw = _sources()
    history.loc[0, "dilutedShares"] = 9999
    result = fundamental_share_observations(history, raw, _basis(), ())
    assert np.isnan(result.iloc[0].dilutedShares) and np.isnan(result.iloc[0].epsDiluted)
    assert result.iloc[0].basicShares == 2300
    assert result.iloc[0].sharesOutstanding == 2400
    print("SANITY: source binding is per field; a different weighted denominator cannot validate merged EPS or erase a separately proven count.")


def test_owner_ambiguity_blocks_source_units_independently_of_quote() -> None:
    history, raw = _sources()
    owner = PredecessorSeries("APH", "OLD", "0000123", pd.Timestamp("1900-01-01"), None)
    result = fundamental_share_observations(history, raw, _basis(), (owner,))
    assert result[["sharesOutstanding", "basicShares", "dilutedShares", "epsDiluted", "dividendsPerShare"]].isna().all().all()
    assert result.iloc[0]._source_shares_outstanding == 1200
    print("SANITY: canonical ticker equality cannot prove financial owner identity; raw availability survives only as an availability mask.")


@pytest.mark.parametrize("missing", ["lastupdated", "calendardate"])
def test_missing_source_metadata_does_not_invent_a_ttm_vintage(missing: str) -> None:
    history, raw = _sources()
    result = fundamental_share_observations(history, raw.drop(columns=missing), _basis(), ())
    assert result[["dilutedShares", "epsDiluted", "dividendsPerShare"]].isna().all().all()
    assert result.iloc[0].netIncome == history.iloc[0].netIncome
    print("SANITY: absent representation/window metadata cannot be replaced with the ticker's maximum update date or a saved count.")


def test_raw_dividend_cash_legs_ignore_normalized_comparison_copy() -> None:
    history, raw = _sources()
    history = history.assign(dividendsPaid=20.0, freeCashflow=100.0)
    comparison = fundamental_share_observations(history, raw, _basis(), ())
    days = pd.bdate_range(history.as_of.iloc[0], periods=3)
    close = pd.DataFrame({"APH": 100.0}, index=days)
    frames = make_frames(days, close_split=close, mcap_pit=close * 1200)
    dividends = pd.DataFrame({"ticker": ["APH"], "date": [days[0]], "dividends": [1.0]})
    before = _dividend_fields(dividends, frames, history, _basis())
    after = _dividend_fields(dividends, frames, history, _basis(), comparison)
    for field in ("dividend_yield", "dividend_payout_ratio", "dividend_coverage"):
        pd.testing.assert_frame_equal(before[field], after[field], check_exact=True)
    assert after["dividend_payout_ratio"].iloc[0, 0] == pytest.approx(1200 / 3750)
    print(
        "SANITY: the actual dividend entry keeps raw cashA, payout, coverage and matched yield unchanged when the buyback comparison copy changes units."
    )


def test_four_quarter_support_cannot_borrow_a_later_publication() -> None:
    history, raw = _sources()
    raw.loc[0, "date"] = pd.Timestamp("2025-11-01")
    result = fundamental_share_observations(history, raw, _basis(), ())
    assert result[["basicShares", "dilutedShares", "epsDiluted", "dividendsPerShare"]].isna().all().all()
    assert result.iloc[0].sharesOutstanding == 2400
    print("SANITY: representation metadata cannot introduce a quarter whose original publication occurs after the target observation.")


def test_equal_endpoint_span_cannot_hide_a_repeated_quarter_label() -> None:
    history, raw = _sources()
    raw.loc[1, "calendardate"] = raw.loc[0, "calendardate"]
    result = fundamental_share_observations(history, raw, _basis(), ())
    assert result[["basicShares", "dilutedShares", "epsDiluted", "dividendsPerShare"]].isna().all().all()
    assert result.iloc[0].sharesOutstanding == 2400
    print("SANITY: a0,0,2,3 quarter-label sequence fails the four-quarter proof even though its endpoints span three quarters.")


def test_fundamental_source_projection_is_loaded_once_and_scoped() -> None:
    history, raw = _sources()
    requests = []
    store = SimpleNamespace(
        columns=lambda table: list(raw.columns),
        load=lambda table, **kwargs: requests.append(kwargs) or raw,
    )
    result = load_fundamental_share_observations(SimpleNamespace(store=store), history, _basis(), ())
    assert len(requests) == 1
    assert requests[0]["where"] == {"ticker": ["APH"], "dimension": "ARQ"}
    assert requests[0]["since"] == history.fiscal_end.iloc[0] - pd.DateOffset(years=1)
    assert requests[0]["until"] == history.as_of.iloc[0]
    assert result.iloc[0].dilutedShares == 2500
    print("SANITY: the adapter performs one ticker/date-projected ARQ read including the preceding original quarter window.")


def test_sparse_quantity_loader_reuses_preloaded_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = []
    store = SimpleNamespace(load=lambda table, **kwargs: requests.append(table) or None, exists=lambda table: True)
    monkeypatch.setattr(basis_module, "load_price_identity", lambda *args: pytest.fail("Identity read twice"))
    basis = basis_module.load_quantity_basis(SimpleNamespace(store=store), ["APH"], {}, ((), None))
    assert len(requests) == 2
    assert basis.actual_available
    print("SANITY: the sparse quantity loader reuses the same selected owner metadata and reads each event table once.")
