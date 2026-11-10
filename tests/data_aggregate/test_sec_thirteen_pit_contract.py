"""Known-truth controls for saved daily PIT inputs and dated 13F quantities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constants.constants import F13_SETTLE_TRADING_DAYS
from src.data_aggregate.transformers import step_cube_sec_thirteen as step_module
from src.data_aggregate.transformers.step_cube_sec_thirteen import SecThirteenResources, StepCubeSecThirteen
from src.data_aggregate.utils.price.features import security_basis_frame
from src.data_aggregate.utils.sec_thirteen.institutional_features import (
    _daily_institutional_value_weight,
    _daily_owned_shares,
    project_institutional_features,
)
from src.data_aggregate.utils.sec_thirteen.superinvestor_metrics import BehaviorMarket, PairTable, prepare_pair_sample
from src.utils.price.adjustments import ShareBasis
from tests.conftest import make_frames
from tests.data_aggregate.test_superinvestor_features import _book


def _basis_frame(basis: ShareBasis, end: str = "2024-06-28") -> pd.DataFrame:
    """The saved forward basis F of `basis` on a grid that starts before the 2024-03-31 report period."""
    return security_basis_frame(basis, pd.bdate_range("2024-03-28", end), pd.Index(["TEST"]))


def test_unconfigured_split_carries_public_book_and_preserves_unknown_successor() -> None:
    index = pd.bdate_range("2024-05-29", "2024-06-05")
    events = pd.DataFrame({"ticker": ["TEST"], "date": [pd.Timestamp("2024-06-03")], "value": [4.0]})
    basis = ShareBasis(events, events, events, {})
    books = pd.DataFrame(
        {
            "ticker": ["TEST", "TEST"],
            "period": pd.to_datetime(["2024-03-31", "2024-03-31"]),
            "as_of": pd.to_datetime(["2024-05-30", "2024-06-05"]),
            "inst_shares": [10.0, np.nan],
        }
    )
    owned = _daily_owned_shares(books, index, _basis_frame(basis))
    assert pd.isna(owned.at[index[0], "TEST"])
    assert owned.at[pd.Timestamp("2024-05-30"), "TEST"] == 10
    assert owned.at[pd.Timestamp("2024-06-03"), "TEST"] == 40
    assert pd.isna(owned.at[pd.Timestamp("2024-06-05"), "TEST"])
    assert books.inst_shares.iloc[0] == 10 and pd.isna(books.inst_shares.iloc[1])
    print("SANITY: unconfigured 4:1 split carries a public 10-share book to40 without a trade; unknown successor ends coverage.")


def test_value_weight_uses_saved_issuer_cap_without_local_fallback() -> None:
    index = pd.bdate_range("2024-05-29", periods=3)
    books = pd.DataFrame({"ticker": ["TEST"], "as_of": [index[0]], "inst_value": [2000.0]})
    quote = pd.DataFrame(10.0, index=index, columns=["TEST"])
    cap = pd.DataFrame({"TEST": [20000.0, np.nan, 40000.0]}, index=index)
    frames = make_frames(index, close_split=quote, level_factor=quote / 10, close_pit=quote * 4, shares_outstanding_pit=quote * 1000, mcap_pit=cap)
    weight = _daily_institutional_value_weight(books, frames)
    assert weight is not None
    assert weight.TEST.iloc[0] == pytest.approx(0.1)
    assert pd.isna(weight.TEST.iloc[1])
    assert weight.TEST.iloc[2] == pytest.approx(0.05)
    assert books.inst_value.iloc[0] == 2000
    print("SANITY: valid historical USD uses saved issuer cap, preserves a cap hole and ignores distinct legacy price/count products.")


def test_sec_thirteen_requests_and_slices_all_saved_pit_inputs() -> None:
    index = pd.bdate_range("2024-05-29", periods=3)
    quote = pd.DataFrame(40.0, index=index, columns=["TEST"])
    frames = make_frames(index, close_pit=quote, shares_outstanding_pit=quote * 100, mcap_pit=quote * 10000)
    assert {"close_pit", "shares_outstanding_pit", "mcap_pit"}.issubset(StepCubeSecThirteen._FIELDS)
    sliced = frames.on(index[-1:], *StepCubeSecThirteen._OUTPUT_FIELDS)
    for field in ("close_pit", "shares_outstanding_pit", "mcap_pit"):
        pd.testing.assert_frame_equal(getattr(sliced, field), getattr(frames, field).iloc[-1:])
    print("SANITY: 13F explicitly requests the three saved PIT inputs and projects their exact effective output dates.")


def test_saved_ownership_count_and_cap_have_independent_support() -> None:
    index = pd.bdate_range("2024-05-29", "2024-06-05")
    events = pd.DataFrame({"ticker": ["TEST"], "date": [pd.Timestamp("2024-06-03")], "value": [4.0]})
    basis = ShareBasis(events, events, events, {})
    books = pd.DataFrame(
        {"ticker": ["TEST"], "period": [pd.Timestamp("2024-03-31")], "as_of": [index[1]], "inst_shares": [10.0], "inst_value": [2000.0]}
    )
    shares = pd.DataFrame({"TEST": [100.0, 100.0, 100.0, 400.0, np.nan, 400.0]}, index=index)
    cap = pd.DataFrame({"TEST": [20000.0, 20000.0, 20000.0, np.nan, 20000.0, 20000.0]}, index=index)
    frames = make_frames(index, shares_outstanding_pit=shares, mcap_pit=cap, security_basis=_basis_frame(basis))
    fields = project_institutional_features(books, frames, 0, trading_calendar=index)
    ownership = fields["ic_inst_ownership_pct"].TEST
    assert ownership.loc["2024-05-30"] == pytest.approx(0.1)
    assert ownership.loc["2024-06-03"] == pytest.approx(0.1)
    assert pd.isna(ownership.loc["2024-06-04"])
    weight = fields["ic_inst_value_to_mcap"].TEST
    assert pd.isna(weight.loc["2024-06-03"])
    assert weight.loc["2024-06-04"] == pytest.approx(0.1)
    print("SANITY: a no-trade split keeps10% ownership from saved counts; count and issuer-cap holes remain independent of stale source caches.")


def test_missing_events_stay_unsupported_and_declared_ads_units_convert() -> None:
    index = pd.bdate_range("2024-05-29", periods=3)
    empty = pd.DataFrame(columns=["ticker", "date", "value"])
    books = pd.DataFrame({"ticker": ["TEST"], "period": [pd.Timestamp("2024-03-31")], "as_of": [index[0]], "inst_shares": [10.0]})
    missing = ShareBasis(empty, empty, empty, {}, actual_available=False)
    policy = {"TEST": {"quote_units": {"date": "2024-06-01", "before": 2.0, "after": 1.0}}}
    ads = ShareBasis(empty, empty, empty, policy)
    assert _daily_owned_shares(books, index, _basis_frame(missing)).TEST.isna().all()
    assert _daily_owned_shares(books, index, _basis_frame(ads)).TEST.tolist() == [10.0, 10.0, 10.0]
    print("SANITY: missing genuine-event evidence leaves ownership unknown; declared ADS units are folded into F, so the book carries 10.")


def test_broad_basis_gate_preserves_reported_usd(monkeypatch: pytest.MonkeyPatch) -> None:
    index = pd.bdate_range("2024-05-29", periods=3)
    keys = pd.MultiIndex.from_tuples([(pd.Timestamp("2024-03-31"), "1", "TEST")], names=["period", "cik", "ticker"])
    compact = pd.DataFrame({"shares": [10.0], "value_usd": [500.0], "call_value": [0.0], "put_value": [0.0]}, index=keys)
    empty = pd.DataFrame(columns=["ticker", "date", "value"])
    basis = ShareBasis(empty, empty, empty, {}, actual_available=False)
    frames = make_frames(index, close_pit=pd.DataFrame(50.0, index=index, columns=["TEST"]), security_basis=_basis_frame(basis))
    resources = SecThirteenResources(frames, None)
    step = object.__new__(StepCubeSecThirteen)
    step._store, step._cfg, step._settle_days = None, {"institutionals": {}}, F13_SETTLE_TRADING_DAYS
    monkeypatch.setattr(step_module.thirteen_inputs, "iter_filing_vintages", lambda *args, **kwargs: iter(()))
    monkeypatch.setattr(step_module, "prepare_institutional_book_history", lambda *args, **kwargs: (compact, pd.DataFrame()))
    monkeypatch.setattr(step_module, "compute_institutional_feature_history", lambda values, states, **kwargs: values)
    result = step._institutional_features(resources)
    assert result is not None and result.shares.isna().all()
    assert result.value_usd.iloc[0] == 500 and compact.shares.iloc[0] == 10
    print("SANITY: quantities without a saved basis F cannot enter broad split-flow math; original500USD and source shares remain intact.")


def test_quote_unknown_keeps_observed_disclosure_keys() -> None:
    index = pd.bdate_range("2024-05-29", periods=3)
    saved = {"close_split": pd.DataFrame(10.0, index=index, columns=["TEST"])}
    saved.update({field: saved["close_split"] * np.nan for field in ("close_pit", "shares_outstanding_pit", "mcap_pit", "security_basis")})
    frames = make_frames(index, **{field: saved[field] for field in StepCubeSecThirteen._FIELDS})
    disclosure = pd.DataFrame({"date": index, "ticker": "TEST", "f_ic_inst_concentration": 0.5})
    result = StepCubeSecThirteen._merge_panels(frames, (disclosure, None))
    assert result.date.tolist() == index.tolist()
    assert result.f_ic_inst_concentration.eq(0.5).all()
    print("SANITY: observed adjusted quote keys anchor source-only13F features even when PIT quote/count/cap support is unknown.")


@pytest.mark.parametrize("units_before,units_after", [(2.0, 1.0), (1.0, 2.0)])
def test_elite_inverse_units_preserve_reference_money(units_before: float, units_after: float) -> None:
    index = pd.DatetimeIndex(["2024-03-28", "2024-06-28"])
    empty = pd.DataFrame(columns=["ticker", "date", "value"])
    policy = {"TEST": {"quote_units": {"date": "2024-06-01", "before": units_before, "after": units_after}}}
    basis = ShareBasis(empty, empty, empty, policy)
    market = BehaviorMarket(index, pd.DataFrame(50.0, index=index, columns=["TEST"]), _basis_frame(basis))
    prior, current = _book("1", "2024-03-31", {}), _book("1", "2024-06-30", {})
    rows = pd.DataFrame({"ticker": ["TEST"], "current": [10.0], "prior": [10.0], "value_current": [np.nan], "value_prior": [np.nan], "known": [True]})
    result = prepare_pair_sample([("1", current, prior)], PairTable({(current, prior): 0}, rows, np.array([0, 1])), market).df_positions
    assert result.prior.iloc[0] * result.price.iloc[0] == pytest.approx(500)
    assert result.prior.iloc[0] == pytest.approx(10.0 * units_before / units_after)
    assert result.known.iloc[0]
    print("SANITY: F folds the declared quote units, so prior shares and the reference quote convert inversely and keep 500 reference USD.")
