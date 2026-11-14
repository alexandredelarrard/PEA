"""An event without supported information is left out of the sums and counts; a window with nothing else is unknown."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_aggregate.utils.institutionals.insider_features import InsiderCoverage
from tests.conftest import make_frames
from tests.data_aggregate.insider_pit_reference import build_prepared_insider_panel as build_insider_feature_panel
from tests.data_aggregate.test_insider_features import _frame, _peers, _txn

IDX = pd.bdate_range("2015-06-01", "2015-09-30")
CLOSE = pd.DataFrame(100.0, index=IDX, columns=["AAA"])
SHARES = pd.DataFrame({"ticker": ["AAA"], "as_of": [pd.Timestamp("2014-01-01")], "sharesOutstanding": [1_000_000.0]})
MCAP = 100.0 * 1_000_000.0


def _panel(rows: list[dict], close: pd.DataFrame | None = CLOSE) -> pd.DataFrame:
    frames = make_frames(IDX, _peers(), close_split=CLOSE)
    coverage = InsiderCoverage.from_availability(None, IDX.max())
    return build_insider_feature_panel(frames, _frame(rows), shares_out_history=SHARES, df_valuation_close=close, coverage=coverage).set_index("date")


def _unpriced(**kw: object) -> dict:
    return _txn(price_per_share=None, value_usd=None, **kw)


def test_an_unpriced_buy_no_longer_blanks_a_window_that_has_a_supported_buy() -> None:
    rows = [
        _txn(
            accession_number="ok", owner_cik="0001", filing_date="2015-06-10", transaction_date="2015-06-09", shares=100.0, shares_owned_after=1_100.0
        ),
        _unpriced(
            accession_number="np", owner_cik="0002", filing_date="2015-06-20", transaction_date="2015-06-19", shares=500.0, shares_owned_after=1_500.0
        ),
    ]
    panel = _panel(rows, close=None)
    day = "2015-07-15"
    assert panel.loc[day, "f_ic_insider_buy_value_mcap_60d"] == pytest.approx(100 * 100.0 / MCAP)
    assert panel.loc[day, "f_ic_insider_distinct_buyers_120d"] == 2
    assert panel.loc[day, "f_ic_insider_net_buy_ratio_60d"] == 1.0
    print("SANITY: the unpriced $50k buy is out of the dollar sum ($10k kept) but its owner counts as a buyer (2).")


def test_a_window_whose_only_buy_is_unpriced_is_unknown_and_recovers_when_it_ages_out() -> None:
    rows = [_unpriced(accession_number="np", owner_cik="0002", filing_date="2015-06-20", transaction_date="2015-06-19", shares=500.0)]
    value = _panel(rows, close=None)["f_ic_insider_buy_value_mcap_60d"]
    assert value.loc["2015-07-15"] != value.loc["2015-07-15"]  # NaN
    assert value.loc["2015-09-01"] == 0.0
    print("SANITY: no supported information -> NaN while the unpriced buy is inside the 60-day window, 0 after it ages out.")


def test_a_price_repaired_from_the_quote_keeps_the_window_known() -> None:
    rows = [_unpriced(accession_number="np", owner_cik="0002", filing_date="2015-06-20", transaction_date="2015-06-19", shares=500.0)]
    value = _panel(rows)["f_ic_insider_buy_value_mcap_60d"]
    assert value.loc["2015-07-15"] == pytest.approx(500 * 100.0 / MCAP)
    print("SANITY: with a quote the missing price is estimated, so the window is the estimated $50k over cap, not NaN.")


def test_an_anonymous_owner_is_left_out_of_the_distinct_counts_only() -> None:
    rows = [
        _txn(accession_number="named", owner_cik="0001", filing_date="2015-06-10", transaction_date="2015-06-09", shares=100.0),
        _txn(accession_number="anon", owner_cik="", filing_date="2015-06-12", transaction_date="2015-06-11", shares=100.0),
    ]
    panel = _panel(rows)
    assert panel.loc["2015-07-01", "f_ic_insider_distinct_buyers_120d"] == 1
    assert panel.loc["2015-07-01", "f_ic_insider_buy_value_mcap_60d"] == pytest.approx(2 * 100 * 100.0 / MCAP)
    print("SANITY: the anonymous buyer's dollars count, the buyer count keeps the one identified owner (no blank window).")
