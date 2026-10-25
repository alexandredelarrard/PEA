"""Phase 2.3 -- the insider panel (registry #28-#41) and the transaction-quality layer.

Every test prints a sanity-check conclusion, because a green assert that nobody reads is how
the two defects these tests exist for survived: a `value_usd` total of $182,982,720tn, and a
"distinct buyer" count of 91 for a company with 13 insiders.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.institutionals.inputs import load_insider_role_evidence
from src.data_aggregate.utils.institutionals.insider_capital import InsiderCapital, normalize_trade_capital
from src.data_aggregate.utils.institutionals.insider_features import (
    CLUSTER_MIN,
    EMISSION,
    INSIDER_FLOOR,
    TEN_B5_1_FLOOR,
    _owner_surprise,
    _unpriced_masks,
    build_insider_feature_panel,
)
from src.data_aggregate.utils.institutionals.insider_quality import (
    OPEN_MARKET_CODES,
    clean_transactions,
    common_stock_mask,
    consensus_price,
    officer_role,
    security_class,
)
from src.data_store.schema import Tables
from tests.conftest import make_frames

TRADING_INDEX = pd.bdate_range("2015-01-01", "2016-12-31")


@contextmanager
def _capture_warnings(logger_name: str):
    """Collect WARNING records from one logger as formatted strings.

    A plain `caplog` would do, but these builders log through a module-level `logging`
    handle and the assertion here is about WHICH logger spoke -- an empty market cap has to
    be reported by the builder that lost the feature, not by something upstream.
    """
    records: list[str] = []

    class _Sink(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    log = logging.getLogger(logger_name)
    handler = _Sink(level=logging.WARNING)
    log.addHandler(handler)
    previous, log.disabled = log.disabled, False
    try:
        yield records
    finally:
        log.removeHandler(handler)
        log.disabled = previous


def _txn(**kw) -> dict:
    """One transaction row with the columns every builder path reads."""
    row = dict(
        accession_number="0000000000-00-000000",
        ticker="AAA",
        owner_cik="0001",
        owner_name="Doe Jane",
        filing_date="2015-06-01",
        transaction_date="2015-05-29",
        transaction_code="P",
        shares=1_000.0,
        price_per_share=100.0,
        value_usd=100_000.0,
        shares_owned_after=11_000.0,
        security_type="nonderiv",
        security_title="Common Stock",
        direct_indirect="D",
        officer_title="",
        is_director=1.0,
        is_officer=0.0,
        is_ten_pct_owner=0.0,
        is_10b5_1=np.nan,
    )
    row.update(kw)
    if "value_usd" in kw and "shares" not in kw and pd.notna(row["value_usd"]) and row["price_per_share"] > 0:
        row["shares"] = row["value_usd"] / row["price_per_share"]
    return row


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _prices(tickers=("AAA",), price=100.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = pd.DataFrame(price, index=TRADING_INDEX, columns=list(tickers))
    fh = pd.DataFrame(
        [{"ticker": t, "as_of": pd.Timestamp("2014-01-01"), "sharesOutstanding": 1_000_000.0, "sharesOutstandingPit": 1_000_000.0} for t in tickers]
    )
    return fh, close


def _peers(tickers=("AAA",)) -> dict:
    return {t: {t: 1.0} for t in tickers}


def test_canonical_cleaner_uses_independent_transaction_quote() -> None:
    quotes = pd.DataFrame({"AAA": [100.0]}, index=pd.to_datetime(["2015-05-29"]))
    rows = [_txn(price_per_share=105, value_usd=105_000), _txn(accession_number="bad", price_per_share=100_000, value_usd=100_000_000)]
    out, diag = clean_transactions(_frame(rows), as_traded_quotes=quotes)
    assert out.loc[0, "value"] == 105_000
    assert out.loc[1, "value"] == 100_000
    assert out.loc[1, "price_estimated"]
    assert diag["repaired_rows"] == 1
    print("SANITY: valid filed execution retained; isolated gross price uses an independently dated, explicitly estimated quote.")


def test_source_price_typo_abstains_without_erasing_known_shares() -> None:
    rows = [
        _txn(
            accession_number="0001127602-15-015564",
            ticker="GOOGL",
            filing_date="2015-04-29",
            transaction_date="2015-04-29",
            transaction_code="S",
            shares=2475,
            shares_owned_after=14191,
            price_per_share=5638853,
            value_usd=13956161175,
            security_title="Class A Common Stock",
            officer_title="CEO",
            is_officer=1,
        ),
        _txn(
            accession_number="0001127602-15-015564",
            ticker="GOOGL",
            filing_date="2015-04-29",
            transaction_date="2015-04-28",
            transaction_code="S",
            shares=2475,
            price_per_share=562,
            value_usd=1390950,
            security_title="Class A Common Stock",
        ),
    ]
    out, diag = clean_transactions(_frame(rows))
    assert len(out) == 1 and out["transaction_day"].eq(pd.Timestamp("2015-04-28")).all()
    claim = diag["unpriced_events"]
    assert len(claim) == 1 and claim["value"].isna().all() and claim["price_clean"].isna().all()
    assert claim["shares_n"].iloc[0] == 2475 and claim["shares_owned_after"].iloc[0] == 14191
    assert not claim["quality_uncertain"].any()
    assert claim["anchor"].eq(pd.Timestamp("2015-04-29")).all()
    assert claim["visible_from"].eq(pd.Timestamp("2015-04-29")).all()
    print(
        "SANITY: footnote contradicts only the exact filed price; abstention keeps known shares, holdings and disclosure anchor; other accession lines survive."
    )


def test_canonical_titles_are_dated_and_identity_safe() -> None:
    rows = [
        _txn(accession_number="a", officer_title="C.F.O.", is_officer=1),
        _txn(accession_number="b", shares=1001, filing_date="2015-06-10", officer_title="", is_officer=1),
        _txn(accession_number="c", shares=1002, filing_date="2015-05-20", officer_title="", is_officer=1),
        _txn(accession_number="d", shares=1003, filing_date="2015-06-15", officer_title="President", is_officer=1),
    ]
    out, _ = clean_transactions(_frame(rows))
    assert list(out["role"]) == ["CFO", "CFO", "other_named_officer", "other_named_officer"]
    assert out.loc[1, "role_source"] == "prior_form4"
    assert out.loc[2, "role_source"] == "filed"
    print("SANITY: punctuation normalized; blank title uses prior same-owner evidence; no future fill or contradictory-title promotion.")


def test_dated_share_claim_masks_tail_and_retires_prospectively() -> None:
    rows = [
        _txn(accession_number="0001584209-16-000008", filing_date="2016-10-19", transaction_date="2016-10-19"),
        _txn(accession_number="known", owner_cik="999", filing_date="2016-10-18", shares=100),
    ]
    cleaned, diag = clean_transactions(_frame(rows))
    claim = cleaned[cleaned["accession_number"].eq(rows[0]["accession_number"])]
    assert claim["visible_until"].eq(pd.Timestamp("2017-03-22")).all()
    assert claim["quality_uncertain"].all()
    idx = pd.bdate_range("2016-10-19", "2017-06-01")
    mask = _unpriced_masks(diag["uncertain_events"], idx)["ic_insider_buy_value_mcap_60d"]
    assert mask.loc["2016-12-16", "AAA"]
    assert not mask.loc["2016-12-19", "AAA"]
    assert not mask.loc["2017-01-20", "AAA"]
    assert not mask.loc["2017-03-22", "AAA"]
    print("SANITY: disputed dollars contaminate only their real60-day contributor window; public invalidation never revives them.")


def test_same_day_surprise_uses_only_strictly_earlier_day() -> None:
    rows = [
        _txn(accession_number="a", filing_date="2015-01-02", transaction_date="2015-01-01", value_usd=50_000),
        _txn(accession_number="b", filing_date="2015-02-02", transaction_date="2015-01-30", value_usd=100_000),
        _txn(accession_number="c", filing_date="2015-02-02", transaction_date="2015-01-30", value_usd=75_000),
    ]
    out, _ = clean_transactions(_frame(rows))
    rank = _owner_surprise(out)
    assert rank["surprise"].eq(1.0).all()
    other = _owner_surprise(out.iloc[::-1]).set_index("accession_number")["surprise"]
    pd.testing.assert_series_equal(rank.set_index("accession_number")["surprise"].sort_index(), other.sort_index())
    print("SANITY: both same-day buys compare only with earlier50k; row permutation leaves ranks unchanged.")


def test_surprise_expires_at_original_anchor_plus_120() -> None:
    rows = [
        _txn(accession_number="a", filing_date="2015-01-02", transaction_date="2015-01-01", value_usd=50_000),
        _txn(accession_number="b", filing_date="2015-02-02", transaction_date="2015-01-30", value_usd=100_000),
    ]
    idx = pd.date_range("2015-01-01", "2015-07-01")
    panel = build_insider_feature_panel(make_frames(idx, _peers()), _frame(rows)).set_index("date")
    score = panel["f_ic_insider_owner_surprise_120d"]
    anchor = pd.Timestamp("2015-02-02")
    assert score.loc[anchor + pd.Timedelta(days=119)] == 1
    assert pd.isna(score.loc[anchor + pd.Timedelta(days=120)])
    print("SANITY: surprise is present at age119, null at age120; old expanding reference history does not extend contributor life.")


def test_daily_lots_reconstruct_prior_without_row_order() -> None:
    rows = [
        _txn(accession_number="a", shares=10, value_usd=1000, shares_owned_after=110),
        _txn(accession_number="a", shares=20, value_usd=2000, shares_owned_after=130),
    ]
    idx = pd.bdate_range("2015-06-01", "2015-06-15")
    for order in (rows, rows[::-1]):
        panel = build_insider_feature_panel(make_frames(idx, _peers()), _frame(order)).set_index("date")
        assert panel["f_ic_insider_purchase_pct_prior"].iloc[0] == pytest.approx(0.3)
    print("SANITY: two supported lots consolidate to30/100, independent of input order; terminal balances are not summed.")


@pytest.mark.parametrize("buy_price,sale_only,expected", [(100.0, False, 0.0), (np.nan, False, 0.0), (np.nan, True, -0.1)])
def test_position_arithmetic_keeps_known_shares_without_prices(buy_price: float, sale_only: bool, expected: float) -> None:
    rows = [
        _txn(
            accession_number="position",
            transaction_code="S",
            shares=10,
            price_per_share=np.nan,
            shares_owned_after=90 if sale_only else 100,
            officer_title="CEO" if sale_only else "",
            is_officer=1,
        )
    ]
    if not sale_only:
        rows.append(
            _txn(accession_number="position", shares=10, price_per_share=buy_price, shares_owned_after=100, officer_title="CEO", is_officer=1)
        )
    idx = pd.date_range("2015-06-01", periods=62)
    for order in (rows, rows[::-1]):
        panel = build_insider_feature_panel(make_frames(idx, _peers()), _frame(order)).set_index("date")
        management = panel["f_ic_insider_management_position_change_60d"]
        assert management.iloc[0] == pytest.approx(expected)
        assert management.iloc[59] == pytest.approx(expected)
        assert pd.isna(management.iloc[60])
        if np.isfinite(buy_price):
            assert panel["f_ic_insider_purchase_pct_prior"].iloc[0] == pytest.approx(0.1)
    print("SANITY: known disclosed shares survive missing execution prices; account math and 60-day expiry are order invariant.")


def test_uncertain_sale_masks_same_account_purchase_and_management() -> None:
    rows = [
        _txn(
            accession_number="0001584209-16-000008",
            filing_date="2016-10-19",
            transaction_date="2016-10-19",
            transaction_code="S",
            shares=10,
            shares_owned_after=100,
            is_officer=1,
        ),
        _txn(
            accession_number="valid",
            filing_date="2016-10-19",
            transaction_date="2016-10-19",
            shares=10,
            shares_owned_after=100,
            officer_title="CEO",
            is_officer=1,
        ),
        _txn(
            accession_number="earlier",
            owner_cik="999",
            filing_date="2016-10-18",
            transaction_date="2016-10-18",
            transaction_code="S",
            shares=10,
            shares_owned_after=90,
            officer_title="CFO",
            is_officer=1,
        ),
        _txn(
            accession_number="known-buy",
            owner_cik="998",
            filing_date="2016-10-18",
            transaction_date="2016-10-18",
            shares=10,
            shares_owned_after=110,
            officer_title="CEO",
            is_officer=1,
        ),
    ]
    idx = pd.bdate_range("2016-10-18", "2017-04-01")
    panel = build_insider_feature_panel(make_frames(idx, _peers()), _frame(rows)).set_index("date")
    assert panel.loc["2016-10-18", "f_ic_insider_purchase_pct_prior"] == pytest.approx(0.1)
    assert pd.isna(panel.loc["2016-10-19", "f_ic_insider_management_position_change_60d"])
    assert pd.isna(panel.loc["2017-03-21", "f_ic_insider_purchase_pct_prior"])
    assert pd.isna(panel.loc["2017-03-22", "f_ic_insider_purchase_pct_prior"])
    print(
        "SANITY: an uncertain blank-title sale contaminates same-account holdings math; retirement after the60-day expiry cannot revive an aged-out purchase ratio."
    )


def test_disputed_prior_masks_later_cross_ticker_score() -> None:
    rows = [
        _txn(accession_number="first", filing_date="2016-01-04", transaction_date="2016-01-01", value_usd=50_000),
        _txn(accession_number="0001584209-16-000008", filing_date="2016-10-19", transaction_date="2016-10-19", value_usd=1_000_000),
        _txn(accession_number="later", ticker="BBB", filing_date="2017-02-01", transaction_date="2017-01-30", value_usd=100_000),
    ]
    idx = pd.bdate_range("2016-01-01", "2017-06-15")
    panel = build_insider_feature_panel(make_frames(idx, _peers(("AAA", "BBB"))), _frame(rows))
    score = (
        panel[panel["ticker"].eq("BBB")].set_index("date").reindex(columns=["f_ic_insider_owner_surprise_120d"])["f_ic_insider_owner_surprise_120d"]
    )
    assert score.loc["2017-02-02":"2017-05-31"].isna().all()
    print("SANITY: uncertain prior onAAA masks derived BBB score, including after public invalidation; contaminated score is never merely unmasked.")


def test_management_pools_holdings_and_expires_after_60_days() -> None:
    rows = [
        _txn(accession_number="ceo", owner_cik="11", officer_title="CEO", is_officer=1, shares=1000, shares_owned_after=11000),
        _txn(accession_number="cfo", owner_cik="12", officer_title="C.F.O.", is_officer=1, transaction_code="S", shares=50, shares_owned_after=50),
    ]
    idx = pd.date_range("2015-06-01", "2015-08-15")
    for ordered in (rows, rows[::-1]):
        panel = build_insider_feature_panel(make_frames(idx, _peers()), _frame(ordered)).set_index("date")
        score = panel["f_ic_insider_management_position_change_60d"]
        assert score.iloc[0] == pytest.approx(950 / 10100)
        assert score.loc["2015-07-30"] == pytest.approx(950 / 10100)
        assert pd.isna(score.loc["2015-07-31"])
    print("SANITY: CEO buy and CFO sell pool to950/10100; permutation unchanged, age59 included, age60 null.")


@pytest.mark.parametrize("code,shares,after,expected", [("P", 200, 300, 2.0), ("S", 100, 0, -1.0), ("P", 100, 100, None)])
def test_management_respects_opening_positions_and_full_exits(code: str, shares: int, after: int, expected: float | None) -> None:
    rows = [_txn(officer_title="CEO and CFO", is_officer=1, transaction_code=code, shares=shares, shares_owned_after=after)]
    idx = pd.date_range("2015-06-01", periods=3)
    panel = build_insider_feature_panel(make_frames(idx, _peers()), _frame(rows)).set_index("date")
    score = panel.reindex(columns=["f_ic_insider_management_position_change_60d"]).iloc[0, 0]
    assert pd.isna(score) if expected is None else score == pytest.approx(expected)
    print(f"SANITY: {code} {shares}, ending{after}: signed supported change={score}; opening zero-prior account is undefined.")


def test_management_later_disclosure_updates_one_day_without_reaging() -> None:
    rows = [
        _txn(accession_number="ceo", owner_cik="11", officer_title="CEO", is_officer=1, shares=10, shares_owned_after=110),
        _txn(accession_number="coo", owner_cik="12", officer_title="COO", is_officer=1, shares=100, shares_owned_after=200, filing_date="2015-06-10"),
    ]
    idx = pd.date_range("2015-06-01", "2015-08-10")
    panel = build_insider_feature_panel(make_frames(idx, _peers()), _frame(rows)).set_index("date")
    score = panel["f_ic_insider_management_position_change_60d"]
    assert score.loc["2015-06-09"] == pytest.approx(0.1)
    assert score.loc["2015-06-10"] == pytest.approx(110 / 200)
    assert score.loc["2015-07-31"] == pytest.approx(1.0)
    assert pd.isna(score.loc["2015-08-09"])
    rows[1]["security_title"] = "Class B Common Stock"
    mixed = build_insider_feature_panel(make_frames(idx, _peers()), _frame(rows)).set_index("date")
    mixed_score = mixed["f_ic_insider_management_position_change_60d"]
    assert mixed_score.loc["2015-06-09"] == pytest.approx(0.1)
    assert mixed_score.loc["2015-06-10":"2015-07-30"].isna().all()
    assert mixed_score.loc["2015-07-31"] == pytest.approx(1.0)
    assert mixed_score.loc["2015-08-09":].isna().all()
    print("SANITY: later same-economic-day leg updates one pool prospectively; incompatible share classes abstain, original age preserved.")


def test_management_future_supported_account_cannot_shorten_past_window() -> None:
    rows = [
        _txn(
            accession_number="opening",
            owner_cik="11",
            officer_title="CEO",
            is_officer=1,
            shares=100,
            shares_owned_after=100,
            filing_date="2015-06-01",
            document_type="4",
        ),
        _txn(
            accession_number="supported",
            owner_cik="12",
            officer_title="CFO",
            is_officer=1,
            shares=10,
            shares_owned_after=110,
            filing_date="2015-06-10",
            document_type="4",
        ),
        _txn(
            accession_number="amend",
            owner_cik="11",
            officer_title="CEO",
            is_officer=1,
            shares=100,
            shares_owned_after=200,
            filing_date="2015-09-01",
            document_type="4/A",
            original_submission_date="2015-06-01",
        ),
    ]
    idx = pd.date_range("2015-06-01", "2015-09-15")
    frames = make_frames(idx, _peers())
    before = build_insider_feature_panel(frames, _frame(rows[:2])).set_index("date")
    after = build_insider_feature_panel(frames, _frame(rows)).set_index("date")
    name = "f_ic_insider_management_position_change_60d"
    pd.testing.assert_series_equal(before.loc[:"2015-08-31", name], after.loc[:"2015-08-31", name])
    assert after.loc["2015-08-01", name] == pytest.approx(0.1)
    print("SANITY: a future correction making an opening account eligible cannot shorten an already-known account's earlier window.")


@pytest.mark.parametrize("later_titles", [["Former Chief Executive Officer"], ["President"], ["Chief Financial Officer", "Chief Operating Officer"]])
def test_proxy_role_barriers_prevent_stale_management_fill(later_titles: list[str]) -> None:
    rows = [
        _txn(accession_number="identity", issuer_cik="123", filing_date="2024-01-01", transaction_code="A", is_officer=1),
        _txn(accession_number="blank", issuer_cik="123", filing_date="2024-05-01", is_officer=1),
    ]
    proxies = [
        {"ticker": "AAA", "cik": "123", "name": "Doe Jane", "title": title, "as_of": day, "fiscal_year": 2023}
        for title, day in [("Chief Executive Officer", "2024-02-01"), *[(title, "2024-04-01") for title in later_titles]]
    ]

    class Store:
        def load(self, table: object, **kwargs: object) -> pd.DataFrame:
            assert table is Tables.def14a_executive_comp
            return pd.DataFrame(proxies)

    evidence = load_insider_role_evidence(Store(), _frame(rows))
    clean, _ = clean_transactions(_frame(rows), role_evidence=evidence)
    assert clean["role"].eq("other_named_officer").all() and not clean["is_management"].any()
    print(f"SANITY: dated proxy evidence {later_titles} blocks stale CEO carry-forward into a later blank title.")


# --------------------------------------------------------------------------- #
# the quality layer                                                             #
# --------------------------------------------------------------------------- #


def test_only_open_market_codes_survive_the_scope_cut():
    """Grants, exercises, withholding and gifts are compensation mechanics, not trades."""
    rows = [_txn(transaction_code=c, accession_number=f"a{i}") for i, c in enumerate(["P", "S", "A", "M", "F", "G", "J", "C", "D", "X"])]
    out, diag = clean_transactions(_frame(rows))
    kept = sorted(out["code"].unique())
    assert kept == sorted(OPEN_MARKET_CODES)
    print(
        f"SANITY: 10 codes in, {kept} out -- the 8 non-discretionary codes ({diag['input_rows'] - diag['scoped_rows']} rows) never reach a feature."
    )


def test_derivatives_and_preferred_are_not_common_stock_purchases():
    rows = [
        _txn(accession_number="a1"),
        _txn(accession_number="a2", security_type="deriv", security_title="Employee Stock Option"),
        _txn(accession_number="a3", security_title="Preferred Stock, Series H"),
    ]
    out, _ = clean_transactions(_frame(rows))
    assert len(out) == 1 and out.iloc[0]["accession_number"] == "a1"
    print(
        "SANITY: 3 purchases in -- 1 common-stock row survives; the option and the "
        "preferred series are dropped, so neither can price a common-share signal."
    )


def test_an_overpriced_row_is_repaired_to_shares_times_the_consensus():
    """The $182,982,720tn defect, in miniature: one filer types 100x the real price."""
    rows = [
        _txn(
            accession_number=f"g{i}",
            owner_cik=f"{i:04d}",
            transaction_date=f"2015-05-{20 + i:02d}",
            shares=10.0,
            price_per_share=100.0,
            value_usd=1_000.0,
        )
        for i in range(6)
    ]
    rows.append(
        _txn(accession_number="bad", owner_cik="9999", transaction_date="2015-05-25", shares=10.0, price_per_share=10_000.0, value_usd=100_000.0)
    )
    quote = pd.DataFrame(100.0, index=pd.bdate_range("2015-05-01", "2015-06-01"), columns=["AAA"])
    out, diag = clean_transactions(_frame(rows), as_traded_quotes=quote)
    bad = out[out["accession_number"].eq("bad")].iloc[0]
    assert diag["repaired_rows"] == 1
    assert bad["price_repaired"] and bad["value"] == pytest.approx(1_000.0)
    print(
        f"SANITY: 7 purchases, one filed at $10,000 against a $100 consensus -> repaired "
        f"to shares x consensus = ${bad['value']:,.0f}, not ${100_000:,}. Table total "
        f"${diag['value_before']:,.0f} -> ${diag['value_after']:,.0f}."
    )


def test_consensus_never_uses_a_later_transaction():
    """Appending later filings cannot change an earlier filing's repaired value."""
    earlier = [
        _txn(
            accession_number=f"prior-{i}",
            owner_cik=f"{i:04d}",
            filing_date=f"2015-05-{17 + i:02d}",
            transaction_date=f"2015-05-{16 + i:02d}",
            shares=10.0,
            price_per_share=100.0,
            value_usd=1_000.0,
        )
        for i in range(3)
    ]
    earlier.append(
        _txn(
            accession_number="suspect",
            owner_cik="9999",
            filing_date="2015-05-21",
            transaction_date="2015-05-20",
            shares=10.0,
            price_per_share=10_000.0,
            value_usd=100_000.0,
        )
    )
    later = [
        _txn(
            accession_number=f"later-{i}",
            owner_cik=f"8{i:03d}",
            filing_date=f"2015-05-{22 + i:02d}",
            # Deliberately on/before the suspect trade date: merely switching to a trailing
            # transaction-date window would still leak these later publications.
            transaction_date=f"2015-05-{17 + i:02d}",
            shares=10.0,
            price_per_share=10_000.0,
            value_usd=100_000.0,
        )
        for i in range(4)
    ]

    quote = pd.DataFrame(100.0, index=pd.bdate_range("2015-05-01", "2015-06-01"), columns=["AAA"])
    prefix, _ = clean_transactions(_frame(earlier), as_traded_quotes=quote)
    full, _ = clean_transactions(_frame(earlier + later), as_traded_quotes=quote)
    prefix_row = prefix.set_index("accession_number").loc["suspect"]
    full_row = full[full["accession_number"].eq("suspect") & full["visible_from"].eq(full["anchor"])].iloc[0]

    assert prefix_row["price_repaired"] and prefix_row["value"] == pytest.approx(1_000.0)
    assert full_row["price_repaired"] and full_row["value"] == pytest.approx(prefix_row["value"])
    print(
        "SANITY: the suspect filing is repaired to $1,000 with the publication prefix "
        f"and remains ${full_row['value']:,.0f} after four later-filed $10,000 prices are appended."
    )


@pytest.mark.parametrize(("unknown_code", "known_code"), [("P", "S"), ("S", "P")])
def test_unpriced_insider_is_unknown_not_zero(unknown_code, known_code):
    """A complete source does not make an unpriced P/S transaction a zero-sized one."""
    unknown_day = pd.Timestamp("2015-08-03")
    rows = [
        _txn(
            accession_number="old-priced",
            transaction_code=known_code,
            filing_date="2015-01-05",
            transaction_date="2015-01-02",
        ),
        _txn(
            accession_number="other-ticker-priced",
            ticker="BBB",
            owner_cik="0002",
            transaction_code=known_code,
            filing_date="2015-01-05",
            transaction_date="2015-01-02",
        ),
        _txn(
            accession_number="unpriced",
            transaction_code=unknown_code,
            filing_date=unknown_day,
            transaction_date="2015-07-31",
            price_per_share=np.nan,
            value_usd=np.nan,
        ),
    ]
    fh, close = _prices(("AAA", "BBB"))
    panel = build_insider_feature_panel(
        make_frames(TRADING_INDEX, _peers(("AAA", "BBB")), close_split=close),
        _frame(rows),
        capital=InsiderCapital(shares_out_history=fh),
        complete_through=TRADING_INDEX.max(),
    )
    panel = panel[panel["ticker"].eq("AAA")].set_index("date")
    ratio = panel["f_ic_insider_net_buy_ratio_180d"]
    affected = ratio.loc[unknown_day : unknown_day + pd.Timedelta(days=179)]
    after = ratio.loc[unknown_day + pd.Timedelta(days=180) :]

    assert affected.isna().all()
    assert not after.empty and after.eq(0.0).all()
    print(
        f"SANITY: the unpriced {unknown_code} leaves all {len(affected)} sessions in its "
        f"180-day net-buy window unknown; after it ages out, {len(after)} complete sessions "
        "return to a genuine zero."
    )


def test_an_underpriced_row_is_counted_but_never_inflated():
    """The one-sided rule. Repairing upward is what turned nine real ~$25m AXON purchases
    into $326m ones, because that ticker carries another company's rows."""
    rows = [
        _txn(
            accession_number=f"g{i}",
            owner_cik=f"{i:04d}",
            transaction_date=f"2015-05-{20 + i:02d}",
            shares=10.0,
            price_per_share=100.0,
            value_usd=1_000.0,
        )
        for i in range(6)
    ]
    rows.append(_txn(accession_number="low", owner_cik="9999", transaction_date="2015-05-25", shares=10.0, price_per_share=1.0, value_usd=10.0))
    quote = pd.DataFrame(100.0, index=pd.bdate_range("2015-05-01", "2015-06-01"), columns=["AAA"])
    out, diag = clean_transactions(_frame(rows), as_traded_quotes=quote)
    low = out[out["accession_number"].eq("low")].iloc[0]
    assert diag["underpriced_rows"] == 1 and diag["repaired_rows"] == 0
    assert not low["price_repaired"] and low["value"] == pytest.approx(10.0)
    print(
        f"SANITY: a price 100x BELOW the consensus is counted ({diag['underpriced_rows']} "
        f"row) and left at its filed ${low['value']:,.0f}. A too-low price understates a "
        f"flow; a too-high one invented $182,982,720tn, and only that direction is "
        f"repaired."
    )


def test_the_consensus_is_built_within_a_share_class():
    """ERIE: Class A near $35, Class B near $32,740, both titled '... Common Stock'. A
    ticker-wide median prices a Class A purchase off the Class B tape."""
    rows = []
    for i in range(6):
        rows.append(
            _txn(
                accession_number=f"a{i}",
                owner_cik=f"a{i}",
                transaction_date=f"2015-05-{20 + i:02d}",
                security_title="Class A Common Stock",
                price_per_share=35.0,
            )
        )
        rows.append(
            _txn(
                accession_number=f"b{i}",
                owner_cik=f"b{i}",
                transaction_date=f"2015-05-{20 + i:02d}",
                security_title="Class B Common Stock",
                price_per_share=32_740.0,
            )
        )
    df = _frame(rows)
    ref = consensus_price(df)
    a = ref[df["security_title"].eq("Class A Common Stock")]
    b = ref[df["security_title"].eq("Class B Common Stock")]
    assert a.max() == pytest.approx(35.0) and b.min() == pytest.approx(32_740.0)
    out, diag = clean_transactions(df)
    assert diag["repaired_rows"] == 0
    print(
        f"SANITY: per-class consensus = ${a.iloc[0]:,.0f} / ${b.iloc[0]:,.0f}; pooled it "
        f"would be ${np.median([35.0, 32_740.0]):,.0f} and every Class A row would be "
        f"'repaired' to 468x its value. 0 rows repaired, which is correct here."
    )


def test_an_exercise_and_sell_package_is_not_a_discretionary_sale():
    """35.0% of all `S` rows share an accession and a transaction date with an `M`."""
    acc = "0001234567-23-000001"
    rows = [
        _txn(
            accession_number=acc,
            transaction_code="M",
            security_type="deriv",
            security_title="Stock Option",
            filing_date="2024-06-03",
            transaction_date="2024-06-01",
        ),
        _txn(accession_number=acc, transaction_code="S", filing_date="2024-06-03", transaction_date="2024-06-01", is_10b5_1=0.0),
        _txn(
            accession_number="0001234567-23-000002",
            transaction_code="S",
            filing_date="2024-06-03",
            transaction_date="2024-06-01",
            is_10b5_1=0.0,
            shares_owned_after=12_000.0,
        ),
    ]
    out, _ = clean_transactions(_frame(rows))
    packaged = out.set_index("accession_number")["in_exercise_package"]
    assert bool(packaged[acc]) and not bool(packaged["0001234567-23-000002"])

    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    fh, close = _prices()
    close = pd.DataFrame(100.0, index=idx, columns=["AAA"])
    panel = build_insider_feature_panel(make_frames(idx, _peers(), close_split=close), _frame(rows), capital=InsiderCapital(shares_out_history=fh))
    col = "f_ic_insider_discretionary_sell_mcap_60d"
    after = panel[panel["date"].ge("2024-06-03")][col].dropna()
    one_sale = 100_000.0 / (1_000_000.0 * 100.0)
    assert after.max() == pytest.approx(one_sale, rel=1e-6)
    print(
        f"SANITY: two identical $100k sales filed the same day, one of them the sell leg "
        f"of an option exercise. Discretionary selling peaks at {after.max():.6f} of "
        f"market cap = exactly ONE sale, not two."
    )


def test_the_role_map_reads_a_title_the_way_a_human_would():
    cases = {
        "Chairman, President & CEO": "CEO",
        "Chief Executive Officer": "CEO",
        "SVP & CFO": "CFO",
        "Chief Financial Officer": "CFO",
        "Treasurer": "other_named_officer",
        "President and Chief Operating Officer": "COO",
        "Executive Chairman": "other_named_officer",
        "": "other_named_officer",
    }
    got = {t: officer_role(t) for t in cases}
    assert got == cases
    print(
        f"SANITY: 'Chairman, President & CEO' -> {got['Chairman, President & CEO']} "
        f"(longest concept first, so it is not read as a President). Measured "
        f"fall-through to other_named_officer on live data: 32.0% of 10,390 titled "
        f"officer purchases -- reported, not tuned away."
    )


def test_security_class_and_common_stock_mask_agree_with_the_measured_titles():
    titles = pd.Series(
        [
            "Common Stock",
            "Class A Common Stock",
            "Ordinary Shares",
            "Preferred Stock, Series H",
            "Employee Stock Option",
            "Common Stock, par value $0.01 per share",
        ]
    )
    mask = common_stock_mask(titles)
    assert mask.tolist() == [True, True, True, False, False, True]
    assert security_class(titles).tolist() == ["COMMON", "A", "COMMON", "H", "COMMON", "COMMON"]
    print(
        f"SANITY: {int(mask.sum())}/6 titles read as common stock; the preferred series "
        f"keys to class 'H' so it can never share a consensus with the common line."
    )


# --------------------------------------------------------------------------- #
# point-in-time                                                                 #
# --------------------------------------------------------------------------- #


def test_owner_surprise_uses_only_that_owners_prior_purchases():
    """The easiest place in the family to leak the future, so it is tested against a
    deliberately-leaky full-sample version rather than by eye."""
    # Two purchases, the SECOND smaller. Two is deliberate: with only one prior in the
    # window the decay weighting cannot blur the answer, so the number the panel prints is
    # the percentile itself and the honest and leaky versions give different numbers.
    sizes = [100.0, 50.0]
    rows = [
        _txn(
            accession_number=f"s{i}",
            owner_cik="0007",
            shares=v,
            value_usd=v * 100.0,
            price_per_share=100.0,
            filing_date=f"2015-0{i + 1}-15",
            transaction_date=f"2015-0{i + 1}-14",
            shares_owned_after=v + 10_000.0,
        )
        for i, v in enumerate(sizes)
    ]
    fh, close = _prices()
    panel = build_insider_feature_panel(
        make_frames(TRADING_INDEX, _peers(), close_split=close), _frame(rows), capital=InsiderCapital(shares_out_history=fh)
    )
    got = panel.set_index("date")["f_ic_insider_owner_surprise_120d"]
    on_second = got.loc["2015-02-16"]
    leaky = float(pd.Series(sizes).rank(pct=True).iloc[1])  # = 0.5
    assert on_second == pytest.approx(0.0, abs=1e-7)
    assert on_second != pytest.approx(leaky, abs=1e-3)
    assert got.loc[:"2015-01-14"].dropna().empty
    print(
        f"SANITY: a $5k purchase following a $10k one by the same owner scores "
        f"{on_second:.2f} -- the smallest they have ever made. A full-sample percentile "
        f"would score it {leaky:.2f} by ranking it against itself. The first purchase is "
        f"NaN ({int(got.loc[:'2015-01-14'].notna().sum())} values before it): 'unusually "
        f"large for this person' is undefined before there is a person."
    )


def test_owner_surprise_counts_ties_at_or_below_per_owner():
    """Known truth by hand: each purchase's share of that owner's earlier purchases at or below it,
    with ties counted, two owners interleaved, and the first purchase of each owner dropped."""
    days = pd.to_datetime(["2015-01-05", "2015-01-06", "2015-01-07", "2015-01-08", "2015-01-09", "2015-01-12"])
    ev = pd.DataFrame(
        {
            "owner_cik": ["A", "B", "A", "A", "B", "A"],
            "value": [10.0, 7.0, 10.0, 5.0, 3.0, 7.0],
            "anchor": days,
            "visible_from": days,
            "visible_until": pd.Series(pd.NaT, index=range(6), dtype="datetime64[ns]"),
        }
    )
    got = _owner_surprise(ev)
    assert got is not None
    # A: 10 (first, dropped), 10 -> 1/1 (a tie), 5 -> 0/2, 7 -> 1/3 (the 5 only); B: 7 (first, dropped), 3 -> 0/1.
    expected = pd.Series([1.0, 0.0, 0.0, 1 / 3], index=[2, 3, 4, 5], name="surprise")
    pd.testing.assert_series_equal(got["surprise"].sort_index(), expected)
    print(
        f"=== SANITY CHECK: owner surprise on 6 hand-checked purchases = {got['surprise'].sort_index().round(3).tolist()}: "
        f"ties count as at or below, each owner ranks only its own earlier purchases, first purchases are dropped ==="
    )


def test_nothing_is_emitted_before_the_measured_availability_floors():
    rows = [
        _txn(filing_date="2015-06-01", transaction_date="2015-05-29"),
        _txn(accession_number="s1", transaction_code="S", filing_date="2015-06-02", transaction_date="2015-05-29", is_10b5_1=1.0),
    ]
    idx = pd.bdate_range("2004-01-01", "2026-01-01")
    fh, close = _prices()
    close = pd.DataFrame(100.0, index=idx, columns=["AAA"])
    panel = build_insider_feature_panel(make_frames(idx, _peers(), close_split=close), _frame(rows), capital=InsiderCapital(shares_out_history=fh))
    cols = [c for c in panel.columns if c not in ("date", "ticker")]
    before = panel[panel["date"] < INSIDER_FLOOR]
    assert before[cols].notna().to_numpy().sum() == 0
    plan_cols = [c for c in cols if "sell_mcap" in c]
    early = panel[panel["date"] < TEN_B5_1_FLOOR]
    assert early[plan_cols].notna().to_numpy().sum() == 0
    print(
        f"SANITY: {len(before):,} ticker-days before {INSIDER_FLOOR.date()} carry 0 "
        f"non-null values across {len(cols)} columns, and the two 10b5-1 legs are empty "
        f"across {len(early):,} rows before {TEN_B5_1_FLOOR.date()} -- NaN, never 0, so "
        f"'not collected yet' cannot read as 'no insider sold'."
    )


def test_complete_through_preserves_covered_zero_and_masks_the_unobserved_tail():
    """A zero inside source coverage is evidence; a zero after it is fabricated evidence."""
    idx = pd.bdate_range("2024-01-01", "2024-09-30")
    complete_through = pd.Timestamp("2024-08-31")
    rows = [
        _txn(
            accession_number="frontier-purchase",
            filing_date="2024-01-02",
            transaction_date="2023-12-29",
        )
    ]
    fh, _ = _prices()
    close = pd.DataFrame(100.0, index=idx, columns=["AAA"])

    panel = build_insider_feature_panel(
        make_frames(idx, _peers(), close_split=close), _frame(rows), capital=InsiderCapital(shares_out_history=fh), complete_through=complete_through
    ).set_index("date")

    covered_zero = panel.loc[pd.Timestamp("2024-07-31"), "f_ic_insider_buy_value_mcap_180d"]
    feature_cols = [column for column in panel.columns if column.startswith("f_ic_insider_")]
    uncovered = panel.loc[panel.index > complete_through, feature_cols]
    assert covered_zero == 0.0
    assert uncovered.isna().to_numpy().all()
    print(
        "SANITY: the 180-day buy window is a real 0 on 2024-07-31 while the source is "
        f"covered, and all {len(feature_cols)} emitted insider legs are NaN on the "
        f"{len(uncovered)} trading days after {complete_through.date()}."
    )


def test_a_feature_never_sees_a_transaction_filed_after_the_date():
    """Truncation: rebuilding on a shorter history must not move a single earlier value."""
    rows = [
        _txn(accession_number=f"p{i}", owner_cik=f"{i:04d}", filing_date=f"2015-{m:02d}-15", transaction_date=f"2015-{m:02d}-14")
        for i, m in enumerate([2, 4, 6, 8, 10])
    ]
    fh, close = _prices()
    full = build_insider_feature_panel(
        make_frames(TRADING_INDEX, _peers(), close_split=close), _frame(rows), capital=InsiderCapital(shares_out_history=fh)
    )
    cut = pd.Timestamp("2015-07-01")
    truncated = build_insider_feature_panel(
        make_frames(TRADING_INDEX, _peers(), close_split=close),
        _frame([r for r in rows if pd.Timestamp(r["filing_date"]) < cut]),
        capital=InsiderCapital(shares_out_history=fh),
    )
    cols = [c for c in full.columns if c in truncated.columns and c not in ("date", "ticker")]
    a = full[full["date"] < cut].set_index(["date", "ticker"])[cols]
    b = truncated[truncated["date"] < cut].set_index(["date", "ticker"])[cols]
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12)
    print(
        f"SANITY: {len(a):,} pre-cut rows x {len(cols)} columns are byte-identical whether "
        f"or not the three later purchases exist. Nothing before {cut.date()} depends on "
        f"anything filed after it."
    )


# --------------------------------------------------------------------------- #
# the features themselves                                                       #
# --------------------------------------------------------------------------- #


def test_distinct_buyers_counts_people_not_filings():
    """The TPL defect: 13 owners buying daily read as 91.4 'distinct buyers' once a decayed
    event sum was called a distinct count."""
    rows = []
    for d in pd.bdate_range("2015-03-02", "2015-05-29"):  # 65 filing days
        for owner in ("0001", "0002", "0003"):
            rows.append(
                _txn(
                    accession_number=f"{owner}-{d:%Y%m%d}",
                    owner_cik=owner,
                    filing_date=d,
                    transaction_date=d,
                    shares=3.0,
                    value_usd=300.0,
                    shares_owned_after=10_003.0 + int(owner),
                )
            )
    fh, close = _prices()
    panel = build_insider_feature_panel(
        make_frames(TRADING_INDEX, _peers(), close_split=close),
        _frame(rows),
        capital=InsiderCapital(shares_out_history=fh),
        complete_through=TRADING_INDEX.max(),
    )
    n = panel.set_index("date")["f_ic_insider_distinct_buyers_120d"]
    peak = n.max()
    assert peak == 3.0
    assert n.loc["2015-10-01"] == 0.0  # the window has emptied
    assert n.loc[:"2015-03-01"].dropna().empty  # nothing before the first purchase
    print(
        f"SANITY: 3 owners filing on 65 consecutive days peak at {peak:.0f} distinct "
        f"buyers, not 195 filings and not the 91.4 a 63-day decay would report. The count "
        f"returns to {n.loc['2015-10-01']:.0f} once the 120-day window empties, and is NaN "
        f"before the first purchase."
    )


def test_cluster_is_two_distinct_buyers_not_two_filings():
    solo = [
        _txn(accession_number=f"x{i}", owner_cik="0001", filing_date=f"2015-03-{2 + i:02d}", transaction_date=f"2015-03-{2 + i:02d}")
        for i in range(4)
    ]
    pair = solo + [_txn(accession_number="y", owner_cik="0002", filing_date="2015-03-06", transaction_date="2015-03-06")]
    fh, close = _prices()
    col = "f_ic_insider_cluster_buy_120d"
    got = {}
    for name, rows in (("one buyer, 4 filings", solo), ("two buyers", pair)):
        panel = build_insider_feature_panel(
            make_frames(TRADING_INDEX, _peers(), close_split=close), _frame(rows), capital=InsiderCapital(shares_out_history=fh)
        )
        got[name] = panel.set_index("date")[col].loc["2015-03-10"]
    assert got["one buyer, 4 filings"] == 0.0
    assert got["two buyers"] == 2.0 >= CLUSTER_MIN
    print(
        f"SANITY: {got['one buyer, 4 filings']:.0f} for one insider filing four times, "
        f"{got['two buyers']:.0f} for two insiders filing once each. The threshold is "
        f"{CLUSTER_MIN} PEOPLE -- measured occupancy 6.9% of ticker-days, against 0.6% "
        f"for the 3-in-20-days the source research proposed."
    )


def test_the_ten_b5_1_split_separates_planned_from_discretionary():
    rows = [
        _txn(accession_number="d", transaction_code="S", filing_date="2024-03-01", transaction_date="2024-02-28", is_10b5_1=0.0, value_usd=100_000.0),
        _txn(
            accession_number="p",
            transaction_code="S",
            filing_date="2024-03-01",
            transaction_date="2024-02-28",
            is_10b5_1=1.0,
            value_usd=300_000.0,
            shares_owned_after=12_000.0,
        ),
        _txn(
            accession_number="u",
            transaction_code="S",
            filing_date="2024-03-04",
            transaction_date="2024-02-28",
            is_10b5_1=np.nan,
            value_usd=900_000.0,
            shares_owned_after=13_000.0,
        ),
    ]
    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    fh, _ = _prices()
    close = pd.DataFrame(100.0, index=idx, columns=["AAA"])
    panel = build_insider_feature_panel(make_frames(idx, _peers(), close_split=close), _frame(rows), capital=InsiderCapital(shares_out_history=fh))
    day = panel[panel["date"].eq("2024-03-01")].iloc[0]
    mcap = 1_000_000.0 * 100.0
    assert day["f_ic_insider_discretionary_sell_mcap_60d"] == pytest.approx(100_000 / mcap)
    assert day["f_ic_insider_planned_sell_mcap_60d"] == pytest.approx(300_000 / mcap)
    unknown_day = panel[panel["date"].eq("2024-03-04")].iloc[0]
    assert pd.isna(unknown_day["f_ic_insider_discretionary_sell_mcap_60d"])
    assert pd.isna(unknown_day["f_ic_insider_planned_sell_mcap_60d"])
    print(
        f"SANITY: $100k flagged not-on-plan, $300k on-plan and $900k with an UNKNOWN flag. "
        f"Discretionary reads {day['f_ic_insider_discretionary_sell_mcap_60d']:.7f} and "
        f"planned {day['f_ic_insider_planned_sell_mcap_60d']:.7f}; the later unknown flag "
        f"makes both full totals unknown prospectively, since it cannot prove either classification."
    )


def test_net_buy_ratio_is_zero_only_inside_a_complete_observation_frontier():
    rows = [_txn(filing_date="2015-06-01", transaction_date="2015-05-29")]
    fh, close = _prices()
    panel = build_insider_feature_panel(
        make_frames(TRADING_INDEX, _peers(), close_split=close),
        _frame(rows),
        capital=InsiderCapital(shares_out_history=fh),
        complete_through=pd.Timestamp("2016-06-30"),
    )
    s = panel.set_index("date")["f_ic_insider_net_buy_ratio_180d"]
    assert pd.isna(s.loc["2015-05-01"]) and s.loc["2015-06-01"] == 1.0
    assert s.loc["2016-06-01"] == 0.0
    assert pd.isna(s.loc["2016-07-01"])
    print(
        f"SANITY: no issuer corroboration stays unknown; one purchase makes "
        f"the ratio {s.loc['2015-06-01']:.0f}, and it returns to {s.loc['2016-06-01']:.0f} "
        "after 180 days; past the complete frontier it is NaN."
    )


def test_every_emitted_column_is_declared_and_every_declaration_is_emitted():
    # Owner CIKs REPEAT on purpose: `owner_surprise` needs a prior purchase by the same
    # person to exist at all, so a fixture of five one-off buyers would silently emit
    # thirteen features and this test would pass against a map of fourteen.
    rows = [
        _txn(
            accession_number=f"p{i}",
            owner_cik=f"{i % 2:04d}",
            officer_title=title,
            is_officer=1.0 if title else 0.0,
            is_director=0.0 if title else 1.0,
            filing_date=f"2024-0{i + 1}-15",
            transaction_date=f"2024-0{i + 1}-14",
            shares=1_000.0 + 100 * i,
            value_usd=100_000.0 + 10_000 * i,
            shares_owned_after=11_000.0 + i,
        )
        for i, title in enumerate(["Chief Executive Officer", "Chief Financial Officer", "", "Executive Vice President", ""])
    ]
    rows += [
        _txn(accession_number="s1", transaction_code="S", filing_date="2024-07-01", transaction_date="2024-06-28", is_10b5_1=0.0),
        _txn(
            accession_number="s2",
            transaction_code="S",
            filing_date="2024-07-01",
            transaction_date="2024-06-28",
            is_10b5_1=1.0,
            shares_owned_after=12_000.0,
        ),
    ]
    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    fh, _ = _prices()
    close = pd.DataFrame(100.0, index=idx, columns=["AAA"])
    panel = build_insider_feature_panel(make_frames(idx, _peers(), close_split=close), _frame(rows), capital=InsiderCapital(shares_out_history=fh))
    cols = {c for c in panel.columns if c not in ("date", "ticker")}
    expected = {f"f_{n}" for n in EMISSION} | {f"f_{n}_xs" for n, m in EMISSION.items() if m == "raw+xs"}
    assert cols == expected, f"missing {sorted(expected - cols)}, extra {sorted(cols - expected)}"
    assert not any(c.endswith("_vs_peers") for c in cols)
    print(
        f"SANITY: {len(EMISSION)} declared features -> {len(cols)} columns "
        f"({sum(m == 'raw+xs' for m in EMISSION.values())} carry an _xs leg), and 0 "
        f"_vs_peers legs, which is D25 for this family: an insider purchase is a fact "
        f"about one company, not about its basket."
    )


def test_without_a_market_cap_the_dollar_features_are_absent_not_counts():
    """`decay_events` counts a NaN magnitude as 1.0 -- correct policy there, and a trap here:
    with no market cap every magnitude is NaN and the four value-scaled intensities would
    silently become event counts under names that promise dollars over market cap."""
    rows = [
        _txn(
            accession_number=f"p{i}",
            owner_cik=f"{i % 2:04d}",
            officer_title="Chief Executive Officer",
            is_officer=1.0,
            is_director=1.0,
            filing_date=f"2015-0{i + 1}-15",
            transaction_date=f"2015-0{i + 1}-14",
            shares=1_000.0 + i,
            shares_owned_after=11_000.0 + i,
        )
        for i in range(3)
    ]
    panel = build_insider_feature_panel(make_frames(TRADING_INDEX, _peers()), _frame(rows))
    got = {c.removeprefix("f_").removesuffix("_xs") for c in panel.columns if c not in ("date", "ticker")}
    scale_free = {
        "ic_insider_distinct_buyers_120d",
        "ic_insider_cluster_buy_120d",
        "ic_insider_net_buy_ratio_180d",
        "ic_insider_purchase_pct_prior",
        "ic_insider_owner_surprise_120d",
        "ic_insider_management_position_change_60d",
    }
    assert got == scale_free, f"unexpected without mcap: {sorted(got - scale_free)}"
    print(
        f"SANITY: with no shares outstanding and no close, {len(got)} scale-free features "
        f"are emitted and all {len(EMISSION) - len(scale_free)} dollar-scaled ones are "
        f"ABSENT -- not present as disguised event counts."
    )


def test_no_usable_transaction_returns_an_empty_panel_not_a_crash():
    for _, rows in (("nothing at all", []), ("grants only", [_txn(transaction_code="A")]), ("derivatives only", [_txn(security_type="deriv")])):
        panel = build_insider_feature_panel(make_frames(TRADING_INDEX, _peers()), _frame(rows) if rows else None)
        assert list(panel.columns) == ["date", "ticker"] and panel.empty
    print(
        "SANITY: an absent table, a grants-only table and a derivatives-only table each "
        "return an empty (date, ticker) frame -- the merge chain reports 'No "
        "insider-trading features built.' rather than raising."
    )


def test_the_mcap_features_are_present_and_the_vendor_basis_is_what_builds_them():
    """The seven `*_mcap_*` insider features must EXIST, and losing them must be AUDIBLE.

    ⚠ THE FAILURE MODE IS ABSENCE, WHICH IS WHY THIS TEST IS SHAPED AS A DELTA.
    `daily_market_cap` reads `sharesOutstanding` -- the VENDOR basis -- while the 13F and
    insider ownership-% denominator is `sharesOutstandingPit`. Both columns exist in
    `fundamentals_history` and both are fully populated (measured 2026-09-14: 51,504 rows,
    51,504 non-null each, differing on 14,296 of them), so it is easy to read one as a synonym
    of the other. It is not: projecting the read to the PIT column alone made
    `fundamentals_to_daily` take its `if field not in columns` branch, `daily_market_cap`
    return a COLUMN-LESS frame, and every `if not mcap.empty` branch in three builders simply
    not fire.

    Nothing logged when they went, so 10 features (~20 emitted columns) disappeared from the
    cube behind a clean build log. Both halves are pinned here: the seven build on the VENDOR
    basis, and dropping that column now produces a WARNING NAMING IT rather than silence.
    """
    # the fixture of `test_every_emitted_column_is_declared_and_every_declaration_is_emitted`:
    # repeated owner CIKs, both officer roles, and a discretionary / 10b5-1 sell pair, which
    # is what it takes for all seven size-scaled legs to be non-degenerate at once.
    rows = [
        _txn(
            accession_number=f"p{i}",
            owner_cik=f"{i % 2:04d}",
            officer_title=title,
            is_officer=1.0 if title else 0.0,
            is_director=0.0 if title else 1.0,
            filing_date=f"2024-0{i + 1}-15",
            transaction_date=f"2024-0{i + 1}-14",
            shares=1_000.0 + 100 * i,
            value_usd=100_000.0 + 10_000 * i,
            shares_owned_after=11_000.0 + i,
        )
        for i, title in enumerate(["Chief Executive Officer", "Chief Financial Officer", "", "Executive Vice President", ""])
    ]
    rows += [
        _txn(accession_number="s1", transaction_code="S", filing_date="2024-07-01", transaction_date="2024-06-28", is_10b5_1=0.0),
        _txn(
            accession_number="s2",
            transaction_code="S",
            filing_date="2024-07-01",
            transaction_date="2024-06-28",
            is_10b5_1=1.0,
            shares_owned_after=12_000.0,
        ),
    ]
    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    fh, _ = _prices()
    close = pd.DataFrame(100.0, index=idx, columns=["AAA"])

    mcap_feats = [f for f in EMISSION if "mcap" in f]
    assert len(mcap_feats) == 7, mcap_feats

    panel = build_insider_feature_panel(make_frames(idx, _peers(), close_split=close), _frame(rows), capital=InsiderCapital(shares_out_history=fh))
    missing = [f for f in mcap_feats if f"f_{f}" not in panel.columns]
    assert not missing, f"size-scaled insider features absent from the panel: {missing}"
    built = {f: int(panel[f"f_{f}"].notna().sum()) for f in mcap_feats}
    assert all(built.values()), f"an mcap feature is all-NaN: {built}"

    # ... and the PIT column ALONE is not a substitute.
    pit_only = fh.drop(columns=["sharesOutstanding"])
    with _capture_warnings("src.data_aggregate.utils.institutionals.insider_features") as logged:
        degraded = build_insider_feature_panel(
            make_frames(idx, _peers(), close_split=close), _frame(rows), capital=InsiderCapital(shares_out_history=pit_only)
        )
    survived = [f for f in mcap_feats if f"f_{f}" in degraded.columns]
    assert not survived, survived
    assert any("sharesOutstanding" in m for m in logged), f"the empty-market-cap path must name the missing column; logged: {logged}"

    print()
    print("=== SANITY: the size-scaled insider family is built on the VENDOR share basis ===")
    print(f"  shares_out_history columns passed: {sorted(fh.columns)}")
    for f, n in built.items():
        print(f"    {f:<42} {n:>4} non-null rows")
    print(f"  with `sharesOutstanding` dropped -> {len(survived)} of {len(mcap_feats)} features survive, and the builder logs:")
    for m in logged:
        print(f"    WARNING {m}")
    print(
        "  CONCLUSION: all 7 mcap features build from `sharesOutstanding`; projecting only "
        "`sharesOutstandingPit` deletes every one of them, and that deletion is now "
        "REPORTED instead of silent. Validated."
    )


def _action_cap_inputs() -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame]:
    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    close = pd.DataFrame(100.0, index=idx, columns=["AAA"])
    close.loc[idx >= pd.Timestamp("2024-03-06"), "AAA"] = 200.0
    close.loc[idx >= pd.Timestamp("2024-03-11"), "AAA"] = 400.0
    history = pd.DataFrame([{"ticker": "AAA", "as_of": pd.Timestamp("2023-01-01"), "sharesOutstanding": 1e6, "sharesOutstandingPit": 1e6}])
    return idx, close, history


@pytest.mark.parametrize(
    ("metric", "code", "title", "plan"),
    [
        ("buy_value_mcap_60d", "P", "", 0.0),
        ("buy_value_mcap_180d", "P", "", 0.0),
        ("ceo_buy_mcap_180d", "P", "Chief Executive Officer", 0.0),
        ("cfo_buy_mcap_180d", "P", "Chief Financial Officer", 0.0),
        ("director_buy_mcap_180d", "P", "", 0.0),
        ("discretionary_sell_mcap_60d", "S", "", 0.0),
        ("planned_sell_mcap_60d", "S", "", 1.0),
    ],
)
def test_action_cap_normalizes_each_trade_before_rolling(metric: str, code: str, title: str, plan: float) -> None:
    idx, close, history = _action_cap_inputs()
    rows = [
        _txn(
            accession_number=f"action-{n}",
            owner_cik=f"{n:04d}",
            filing_date=filing,
            transaction_date=action,
            transaction_code=code,
            officer_title=title,
            is_officer=float(bool(title)),
            is_10b5_1=plan,
            value_usd=1e6,
        )
        for n, (action, filing) in enumerate([("2024-03-04", "2024-03-05"), ("2024-03-06", "2024-03-07")], start=1)
    ]
    panel = build_insider_feature_panel(
        make_frames(idx, _peers(), close_split=close), _frame(rows), capital=InsiderCapital(shares_out_history=history), complete_through=idx[-1]
    ).set_index("date")
    feature = f"f_ic_insider_{metric}"

    assert panel.loc["2024-03-08", feature] == pytest.approx(0.015)
    assert panel.loc["2024-03-11", feature] == pytest.approx(0.015)
    assert pd.isna(panel.loc["2024-03-04", feature])
    print(f"SANITY: {metric} sums 1m/100m + 1m/200m = 1.5%; later 400m cap never revalues disclosed trades.")


def _build_action_cap(rows: list[dict], close: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    idx = pd.DatetimeIndex(close.index)
    return build_insider_feature_panel(
        make_frames(idx, _peers(), close_split=close), _frame(rows), capital=InsiderCapital(shares_out_history=history), complete_through=idx[-1]
    ).set_index("date")


@pytest.mark.parametrize("bad_cap_price", [np.nan, 0.0, -1.0, np.inf])
def test_action_cap_exact_session_gap_does_not_fall_back(bad_cap_price: float) -> None:
    _, close, history = _action_cap_inputs()
    close.loc["2024-03-04", "AAA"] = bad_cap_price
    rows = [_txn(filing_date="2024-03-05", transaction_date="2024-03-04", value_usd=1e6)]
    panel = _build_action_cap(rows, close, history)

    assert pd.isna(panel.loc["2024-03-07", "f_ic_insider_buy_value_mcap_60d"])
    assert panel.loc["2024-03-07", "f_ic_insider_distinct_buyers_120d"] == 1.0
    assert panel.loc["2024-05-06", "f_ic_insider_buy_value_mcap_60d"] == 0.0
    print("SANITY: an exact-session invalid cap remains unknown until expiry; counts and covered quiet zeros survive.")


def test_action_cap_missing_transaction_date_never_uses_filing_date() -> None:
    _, close, history = _action_cap_inputs()
    panel = _build_action_cap([_txn(filing_date="2024-03-05", transaction_date=None, value_usd=1e6)], close, history)

    assert pd.isna(panel.loc["2024-03-07", "f_ic_insider_buy_value_mcap_60d"])
    assert panel.loc["2024-03-07", "f_ic_insider_distinct_buyers_120d"] == 1.0
    print("SANITY: missing economic date cannot borrow filing-day capital or erase a disclosed buyer.")


def test_action_cap_non_session_uses_only_immediate_previous_session() -> None:
    _, close, history = _action_cap_inputs()
    history = pd.concat([history, history.assign(as_of=pd.Timestamp("2024-03-09"), sharesOutstanding=9e6)], ignore_index=True)
    row = _txn(filing_date="2024-03-11", transaction_date="2024-03-09", value_usd=1e6)
    panel = _build_action_cap([row], close, history)
    assert panel.loc["2024-03-12", "f_ic_insider_buy_value_mcap_60d"] == pytest.approx(0.005)

    close.loc["2024-03-08", "AAA"] = np.nan
    unknown = _build_action_cap([row], close, history)
    assert pd.isna(unknown.loc["2024-03-12", "f_ic_insider_buy_value_mcap_60d"])
    print("SANITY: Saturday selects Friday exactly; missing Friday cap cannot skip to Thursday or Monday.")


def test_action_cap_non_session_previous_session_older_than_seven_days_is_unknown() -> None:
    _, close, history = _action_cap_inputs()
    close = close.drop(index=pd.bdate_range("2024-03-04", "2024-03-08"))
    panel = _build_action_cap([_txn(filing_date="2024-03-11", transaction_date="2024-03-09", value_usd=1e6)], close, history)

    assert pd.isna(panel.loc["2024-03-12", "f_ic_insider_buy_value_mcap_60d"])
    print("SANITY: a non-session action cannot borrow an eight-day-old previous session.")


def test_action_cap_late_form5_uses_action_day_public_shares() -> None:
    _, close, history = _action_cap_inputs()
    later = history.assign(as_of=pd.Timestamp("2024-03-05"), fiscal_end=pd.Timestamp("2023-12-31"), sharesOutstanding=9e6)
    history = pd.concat([history, later], ignore_index=True)
    row = _txn(filing_date="2024-06-03", transaction_date="2024-03-04", document_type="5", value_usd=1e6)
    panel = _build_action_cap([row], close, history)

    assert pd.isna(panel.loc["2024-05-31", "f_ic_insider_buy_value_mcap_60d"])
    assert panel.loc["2024-06-03", "f_ic_insider_buy_value_mcap_60d"] == pytest.approx(0.01)
    assert panel.loc["2024-06-04", "f_ic_insider_buy_value_mcap_60d"] == pytest.approx(0.01)
    print("SANITY: late Form5 remains private until disclosure and uses old action-date capital, never later public shares/fiscal end.")


def test_action_cap_current_cap_loss_preserves_valid_trade_and_quiet_zero() -> None:
    _, close, history = _action_cap_inputs()
    history = pd.concat([history, history.assign(as_of=pd.Timestamp("2024-03-11"), sharesOutstanding=0.0)], ignore_index=True)
    panel = _build_action_cap([_txn(filing_date="2024-03-05", transaction_date="2024-03-04", value_usd=1e6)], close, history)

    assert panel.loc["2024-03-12", "f_ic_insider_buy_value_mcap_60d"] == pytest.approx(0.01)
    assert panel.loc["2024-05-06", "f_ic_insider_buy_value_mcap_60d"] == 0.0
    print("SANITY: loss of today capital cannot erase a valid action ratio or a later source-complete quiet zero.")


def test_action_cap_unknown_event_masks_only_its_eligible_scopes() -> None:
    _, close, history = _action_cap_inputs()
    close.loc["2024-03-04", "AAA"] = np.nan
    rows = [
        _txn(
            accession_number="unknown-ceo",
            owner_cik="0001",
            filing_date="2024-03-05",
            transaction_date="2024-03-04",
            value_usd=1e6,
            officer_title="Chief Executive Officer",
            is_officer=1.0,
        ),
        _txn(
            accession_number="known-cfo",
            owner_cik="0002",
            filing_date="2024-03-07",
            transaction_date="2024-03-06",
            value_usd=1e6,
            officer_title="Chief Financial Officer",
            is_officer=1.0,
        ),
    ]
    panel = _build_action_cap(rows, close, history)

    assert pd.isna(panel.loc["2024-03-08", "f_ic_insider_buy_value_mcap_60d"])
    assert pd.isna(panel.loc["2024-03-08", "f_ic_insider_ceo_buy_mcap_180d"])
    assert panel.loc["2024-03-08", "f_ic_insider_cfo_buy_mcap_180d"] == pytest.approx(0.005)
    assert panel.loc["2024-03-08", "f_ic_insider_distinct_buyers_120d"] == 2.0
    assert panel.loc["2024-03-08", "f_ic_insider_net_buy_ratio_180d"] == 1.0
    print("SANITY: missing CEO cap blocks the aggregate/CEO sum, retaining supported CFO/count/raw-dollar behavior.")


@pytest.mark.parametrize("original_known", [False, True])
def test_action_cap_amendment_known_unknown_visibility_and_expiry(original_known: bool) -> None:
    _, close, history = _action_cap_inputs()
    rows = [
        _txn(
            accession_number="original",
            filing_date="2024-03-05",
            transaction_date="2024-03-04",
            price_per_share=100.0 if original_known else np.nan,
            shares=10_000.0,
        ),
        _txn(
            accession_number="amended",
            filing_date="2024-03-12",
            transaction_date="2024-03-04",
            document_type="4/A",
            original_submission_date="2024-03-05",
            price_per_share=np.nan if original_known else 100.0,
            shares=10_000.0,
        ),
    ]
    panel = _build_action_cap(rows, close, history)
    before = panel.loc["2024-03-11", "f_ic_insider_buy_value_mcap_60d"]
    after = panel.loc["2024-03-12", "f_ic_insider_buy_value_mcap_60d"]

    assert (before == pytest.approx(0.01)) if original_known else pd.isna(before)
    assert pd.isna(after) if original_known else (after == pytest.approx(0.01))
    assert panel.loc["2024-05-06", "f_ic_insider_buy_value_mcap_60d"] == 0.0
    print("SANITY: corrected known/unknown dollars switch only at disclosure and expire at the original60-day anchor.")


def test_action_cap_market_evidence_does_not_change_non_cap_features() -> None:
    _, close, history = _action_cap_inputs()
    rows = [_txn(filing_date="2024-03-05", transaction_date="2024-03-04", value_usd=1e6)]
    supported = _build_action_cap(rows, close, history)
    unsupported = _build_action_cap(rows, close, history.assign(sharesOutstanding=np.inf))
    columns = [name for name in supported if "mcap" not in str(name)]

    pd.testing.assert_frame_equal(supported[columns], unsupported[columns])
    assert pd.isna(unsupported.loc["2024-03-07", "f_ic_insider_buy_value_mcap_60d"])
    print("SANITY: nonfinite capitalization affects only cap-dependent fields; all other values/nulls remain equal.")


def test_action_cap_prepared_lineage_and_finite_execution() -> None:
    calendar = pd.bdate_range("2024-03-01", "2024-03-15")
    trades = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "transaction_day": pd.to_datetime(["2024-03-09", "2024-03-09"]),
            "visible_from": pd.to_datetime(["2024-03-11", "2024-03-11"]),
            "value": [1e6, np.inf],
        }
    )
    caps = pd.DataFrame(200e6, index=calendar, columns=["AAA"])
    history = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "as_of": pd.to_datetime(["2024-01-02", "2024-03-05", "2024-03-09"]),
            "sharesOutstanding": [1e6, np.nan, 9e6],
        }
    )
    prepared = normalize_trade_capital(trades, caps, calendar, history)

    assert prepared["cap_session"].eq(pd.Timestamp("2024-03-08")).all()
    assert prepared["cap_price_date"].eq(pd.Timestamp("2024-03-08")).all()
    assert prepared["cap_share_as_of"].eq(pd.Timestamp("2024-01-02")).all()
    assert prepared.loc[0, "value_mcap"] == pytest.approx(0.005)
    assert prepared.loc[0, "cap_status"] == "supported"
    assert prepared.loc[1, "cap_status"] == "unsupported_execution"
    assert pd.isna(prepared.loc[1, "value_mcap"])
    pd.testing.assert_series_equal(prepared["value"], trades["value"])
    overflow = normalize_trade_capital(trades.assign(value=1e308), caps * 1e-308 / 200e6, calendar, history)
    assert overflow["value_mcap"].isna().all()
    assert overflow["cap_status"].eq("unsupported_ratio").all()
    print("SANITY: prepared ratios retain exact prior-session/public-share lineage, reject infinite clean dollars and preserve raw amounts.")


@pytest.mark.parametrize("invalid_leg", ["close", "level_factor"])
def test_action_cap_double_negative_components_cannot_certify_capital(invalid_leg: str) -> None:
    idx, close, history = _action_cap_inputs()
    history["sharesOutstanding"] = -1e6
    factor = pd.DataFrame(1.0, index=idx, columns=["AAA"])
    if invalid_leg == "close":
        close.loc["2024-03-04", "AAA"] = -100.0
    else:
        factor.loc["2024-03-04", "AAA"] = -1.0
    row = _txn(filing_date="2024-03-05", transaction_date="2024-03-04", value_usd=1e6)
    panel = build_insider_feature_panel(
        make_frames(idx, _peers(), close_split=close, level_factor=factor),
        _frame([row]),
        capital=InsiderCapital(shares_out_history=history),
        complete_through=idx[-1],
    ).set_index("date")

    assert pd.isna(panel.loc["2024-03-07", "f_ic_insider_buy_value_mcap_60d"])
    assert panel.loc["2024-03-07", "f_ic_insider_distinct_buyers_120d"] == 1.0
    print("SANITY: two invalid negative inputs cannot manufacture a compatible positive capitalization.")


@pytest.mark.parametrize("action", ["2024-03-11", "2024-03-09"])
def test_action_cap_future_transaction_never_publishes_or_resurrects(action: str) -> None:
    _, close, history = _action_cap_inputs()
    row = _txn(filing_date="2024-03-08", transaction_date=action, value_usd=1e6)
    panel = _build_action_cap([row], close, history)

    assert pd.isna(panel.loc["2024-03-08", "f_ic_insider_buy_value_mcap_60d"])
    assert pd.isna(panel.loc["2024-03-12", "f_ic_insider_buy_value_mcap_60d"])
    assert panel.loc["2024-03-12", "f_ic_insider_distinct_buyers_120d"] == 1.0
    assert panel.loc["2024-05-07", "f_ic_insider_buy_value_mcap_60d"] == 0.0
    print("SANITY: a future trading/non-session action is unsupported from disclosure through expiry; passing its date never resurrects it.")


def test_action_cap_future_transaction_recovers_only_on_new_public_version() -> None:
    _, close, history = _action_cap_inputs()
    original = _txn(accession_number="future-original", filing_date="2024-03-08", transaction_date="2024-03-11", value_usd=1e6)
    amended = _txn(
        accession_number="future-amended",
        filing_date="2024-03-12",
        transaction_date="2024-03-11",
        document_type="4/A",
        original_submission_date="2024-03-08",
        value_usd=2e6,
    )
    panel = _build_action_cap([original, amended], close, history)

    assert pd.isna(panel.loc["2024-03-08", "f_ic_insider_buy_value_mcap_60d"])
    assert pd.isna(panel.loc["2024-03-11", "f_ic_insider_buy_value_mcap_60d"])
    assert panel.loc["2024-03-12", "f_ic_insider_buy_value_mcap_60d"] == pytest.approx(0.005)
    assert panel.loc["2024-05-07", "f_ic_insider_buy_value_mcap_60d"] == 0.0
    print(
        "SANITY: a later real correction can establish valid action/publication order prospectively, without resetting the first disclosure expiry."
    )
