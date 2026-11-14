"""Known-truth price repair of insider trades against the saved point-in-time close, and the 25% share-count cap."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_aggregate.utils.institutionals.insider_quality import clean_transactions
from tests.conftest import make_frames
from tests.data_aggregate.insider_pit_reference import build_prepared_insider_panel as build_insider_feature_panel
from tests.data_aggregate.test_insider_features import _frame, _peers, _txn

CLOSE = pd.DataFrame(100.0, index=pd.bdate_range("2015-05-01", "2015-06-30"), columns=["AAA"])


def _one(**kw: object) -> tuple[pd.Series, dict]:
    out, diag = clean_transactions(_frame([_txn(accession_number="x", shares=10.0, value_usd=None, **kw)]), df_valuation_close=CLOSE)
    return out.iloc[0], diag


@pytest.mark.parametrize(
    ("filed", "repaired", "price"),
    [(200.0, False, 200.0), (200.01, True, 100.0), (50.0, False, 50.0), (49.99, True, 100.0), (150.0, False, 150.0), (20.0, True, 100.0)],
)
def test_band_edges_are_kept_and_beyond_them_replaced(filed: float, repaired: bool, price: float) -> None:
    got, diag = _one(price_per_share=filed)
    assert bool(got["price_repaired"]) is repaired and bool(got["price_estimated"]) is repaired
    assert got["price_clean"] == pytest.approx(price) and got["value"] == pytest.approx(10 * price)
    print(f"SANITY: filed {filed} vs close 100 -> {'replaced' if repaired else 'kept'} at {price}; a ratio exactly 2.0 or 0.5 is kept.")


def test_high_and_low_counted_apart_and_filed_value_evidence_untouched() -> None:
    rows = [
        _txn(accession_number="h", owner_cik="1", shares=10.0, price_per_share=1_000.0, value_usd=10_000.0),
        _txn(accession_number="l", owner_cik="2", shares=10.0, price_per_share=1.0, value_usd=10.0),
    ]
    out, diag = clean_transactions(_frame(rows), df_valuation_close=CLOSE)
    assert (diag["repaired_high"], diag["repaired_low"], diag["repaired_rows"]) == (1, 1, 2)
    assert out["value"].tolist() == [1000.0, 1000.0] and out["value_usd"].tolist() == [10_000.0, 10.0]
    assert diag["value_before"] == pytest.approx(10_010.0) and diag["value_after"] == pytest.approx(2_000.0)
    print(f"SANITY: one 10x-high and one 100x-low price repaired to close; total ${diag['value_before']:,.0f} -> ${diag['value_after']:,.0f}.")


def test_a_quote_up_to_seven_days_old_is_used_and_an_older_one_is_not() -> None:
    quotes = pd.DataFrame({"AAA": [100.0]}, index=pd.to_datetime(["2015-05-20"]))
    near = _txn(accession_number="n", owner_cik="1", transaction_date="2015-05-27", shares=10.0, price_per_share=1_000.0)
    far = _txn(accession_number="f", owner_cik="2", transaction_date="2015-05-28", shares=10.0, price_per_share=1_000.0)
    out, diag = clean_transactions(_frame([near, far]), df_valuation_close=quotes)
    by = out.set_index("accession_number")
    assert by.loc["n", "price_clean"] == pytest.approx(100.0) and by.loc["f", "price_clean"] == pytest.approx(1_000.0)
    assert diag["no_consensus_rows"] == 1
    print(
        "SANITY: a quote 7 days before is used, one 8 days before is not; the trade without a quote keeps its filed price and counts as no-consensus."
    )


def test_a_class_titled_row_gets_no_reference() -> None:
    got, diag = _one(price_per_share=1_000.0, security_title="Class B Common Stock")
    assert got["price_clean"] == pytest.approx(1_000.0) and not got["price_repaired"] and diag["no_consensus_rows"] == 1
    print("SANITY: a class-titled trade needs its own quote; the canonical close cannot reprice it.")


@pytest.mark.parametrize("filed", [None, 0.0])
def test_a_missing_or_zero_price_is_priced_from_the_quote_and_flagged(filed: float | None) -> None:
    got, diag = _one(price_per_share=filed)
    assert got["price_clean"] == pytest.approx(100.0) and got["value"] == pytest.approx(1_000.0)
    assert got["price_estimated"] and not got["price_repaired"] and diag["priced_from_quote"] == 1
    print(f"SANITY: filed price {filed} with a quote -> priced $100 and flagged estimated; unpriced rows left: {diag['dropped_unpriced']}.")


def test_a_missing_price_without_a_quote_stays_unpriced() -> None:
    out, diag = clean_transactions(_frame([_txn(accession_number="x", price_per_share=None, value_usd=None)]), df_valuation_close=None)
    assert out.empty and diag["dropped_unpriced"] == 1 and len(diag["unpriced_events"]) == 1
    print("SANITY: no filed price and no quote -> the share claim stays an unpriced event.")


def _shares_out(value: float) -> pd.DataFrame:
    return pd.DataFrame(value, index=pd.bdate_range("2015-01-01", "2015-12-31"), columns=["AAA"])


def test_a_trade_above_a_quarter_of_the_shares_outstanding_is_removed_and_the_account_marked() -> None:
    rows = [
        _txn(accession_number="big", owner_cik="1", shares=300.0, price_per_share=100.0),
        _txn(accession_number="edge", owner_cik="2", shares=250.0, price_per_share=100.0),
        _txn(accession_number="small", owner_cik="3", shares=10.0, price_per_share=100.0),
    ]
    out, diag = clean_transactions(_frame(rows), df_valuation_close=CLOSE, df_shares_outstanding=_shares_out(1_000.0))
    assert sorted(out["accession_number"]) == ["edge", "small"]
    removed = diag["oversized_events"]
    assert removed["accession_number"].tolist() == ["big"] and removed["pct_shares_outstanding"].iloc[0] == pytest.approx(0.3)
    print("SANITY: 30% of shares outstanding is removed, exactly 25% and 1% are kept.")


def test_the_removed_account_leaves_the_ratio_and_the_value_features_exclude_the_trade() -> None:
    idx = pd.bdate_range("2015-06-01", "2015-08-31")
    close = pd.DataFrame(100.0, index=pd.bdate_range("2015-01-01", "2015-08-31"), columns=["AAA"])
    shares = pd.DataFrame({"ticker": ["AAA"], "as_of": [pd.Timestamp("2014-01-01")], "sharesOutstanding": [1_000.0]})
    rows = [
        _txn(
            accession_number="big", owner_cik="0001", filing_date="2015-06-02", transaction_date="2015-06-01", shares=300.0, shares_owned_after=400.0
        ),
        _txn(accession_number="ok", owner_cik="0002", filing_date="2015-06-03", transaction_date="2015-06-02", shares=10.0, shares_owned_after=110.0),
    ]
    frames = make_frames(idx, _peers(), close_split=close, shares_outstanding_pit=pd.DataFrame(1_000.0, index=close.index, columns=["AAA"]))
    panel = build_insider_feature_panel(frames, _frame(rows), shares_out_history=shares, df_valuation_close=close).set_index("date")
    mcap = 100.0 * 1_000.0
    assert panel.loc["2015-07-01", "f_ic_insider_buy_value_mcap_60d"] == pytest.approx(10 * 100.0 / mcap)
    print("SANITY: the 30% trade is absent from the buy value.")
