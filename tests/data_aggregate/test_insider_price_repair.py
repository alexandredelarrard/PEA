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


@pytest.mark.parametrize(
    ("filed", "repaired", "price"), [(200.0, False, 200.0), (200.01, True, 100.0), (50.0, False, 50.0), (49.99, True, 100.0), (1_000.0, True, 100.0)]
)
def test_a_class_titled_row_is_priced_from_the_ticker_quote(filed: float, repaired: bool, price: float) -> None:
    got, diag = _one(price_per_share=filed, security_title="Class B Common Stock")
    assert bool(got["price_repaired"]) is repaired and got["price_clean"] == pytest.approx(price) and diag["no_consensus_rows"] == 0
    print(f"SANITY: class-titled filed {filed} vs ticker close 100 -> {'replaced' if repaired else 'kept'} at {price}; band edges 2.0/0.5 kept.")


def test_a_filing_with_an_impossible_quantity_loses_its_other_rows_too() -> None:
    outstanding = pd.DataFrame(10_000.0, index=CLOSE.index, columns=["AAA"])
    rows = [
        _txn(accession_number="bad", owner_cik="1", shares=3_000.0, value_usd=300_000.0),  # 30% of outstanding
        _txn(accession_number="bad", owner_cik="1", transaction_date="2015-05-28", shares=800.0, value_usd=80_000.0),  # 8%, same filing
        _txn(accession_number="ok", owner_cik="2", shares=800.0, value_usd=80_000.0),  # 8%, other filing
    ]
    out, diag = clean_transactions(_frame(rows), df_valuation_close=CLOSE, df_shares_outstanding=outstanding)
    removed = diag["oversized_events"]
    assert sorted(removed["shares_n"]) == [800.0, 3_000.0] and set(removed["accession_number"]) == {"bad"}
    assert out["accession_number"].tolist() == ["ok"]
    print("SANITY: the 30% trade and the 8% trade of its filing are removed (owner 1 disputed); the 8% trade of another filing stays.")


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
