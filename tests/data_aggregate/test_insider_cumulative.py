"""Known-truth arithmetic for cumulative insider disclosure windows."""

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.common.availability import InstitutionalAvailability
from src.data_aggregate.utils.institutionals.inputs import QuantityBasis
from src.data_aggregate.utils.institutionals.insider_features import InsiderCoverage
from src.data_aggregate.utils.institutionals.insider_positions import position_units
from src.data_aggregate.utils.institutionals.insider_quality import clean_transactions, officer_role
from src.data_store.schema import Tables
from src.utils.price_adjustments import ShareBasis
from tests.conftest import make_frames
from tests.data_aggregate.insider_pit_reference import build_prepared_insider_panel as build_insider_feature_panel
from tests.data_aggregate.test_insider_features import _frame, _peers, _txn


def test_quantity_rebasing_preserves_clean_execution_usd() -> None:
    idx = pd.bdate_range("2024-06-03", periods=10)
    df_events = pd.DataFrame({"ticker": ["AAA"], "date": [pd.Timestamp("2024-06-10")], "value": [10.0]})
    share_basis = ShareBasis(df_events, df_events, df_events, {})
    rows = [
        _txn(
            accession_number="before", transaction_date="2024-06-07", filing_date="2024-06-11", shares=5, price_per_share=100, shares_owned_after=105
        ),
        _txn(
            accession_number="after", transaction_date="2024-06-10", filing_date="2024-06-12", shares=50, price_per_share=10, shares_owned_after=1100
        ),
    ]
    df_quotes = pd.DataFrame({"AAA": np.where(idx < pd.Timestamp("2024-06-10"), 100.0, 10.0)}, index=idx)
    df_clean, _ = clean_transactions(_frame(rows), df_valuation_close=df_quotes, share_basis=share_basis)
    df_rebased = position_units(df_clean, QuantityBasis(share_basis, idx[-1]))
    df_before = df_rebased.loc[df_rebased["transaction_day"].eq(pd.Timestamp("2024-06-07"))]
    assert df_before["shares_n"].tolist() == [50]
    assert df_before["shares_owned_after"].tolist() == [1050]
    assert df_rebased["units_known"].all()
    assert not df_clean["price_repaired"].any()
    pd.testing.assert_series_equal(df_rebased["value"], df_clean["value"])
    assert df_clean["value"].eq(500.0).all()
    shares = pd.DataFrame(
        {"ticker": ["AAA", "AAA"], "as_of": pd.to_datetime(["2024-06-03", "2024-06-12"]), "sharesOutstanding": [100_000_000.0, 200_000_000.0]}
    )
    for price, factor in ((10.0, 1.0), (5.0, 2.0)):
        frames = make_frames(
            idx, _peers(), close_split=pd.DataFrame(price, index=idx, columns=["AAA"]), level_factor=pd.DataFrame(factor, index=idx, columns=["AAA"])
        )
        panel = build_insider_feature_panel(
            frames, _frame(rows), shares_out_history=shares, df_valuation_close=df_quotes, share_basis=share_basis
        ).set_index("date")
        assert panel.loc["2024-06-11", "f_ic_insider_buy_value_mcap_60d"] == pytest.approx(0.0000005)
        assert panel.loc["2024-06-12", "f_ic_insider_buy_value_mcap_60d"] == pytest.approx(0.000001)
    print(
        "SANITY: source-date quotes validate100/10 execution prices before5→50 share rebasing; both500USD values remain exact after delayed disclosure."
    )


def test_quantity_rebasing_requires_explicit_nonunitary_titles() -> None:
    df_events = pd.DataFrame({"ticker": ["AAA"], "date": pd.to_datetime(["2024-06-10"]), "value": [0.5]})
    policy = {
        "AAA": {
            "quote_units": {
                "date": "2024-06-10",
                "before": 2,
                "after": 1,
                "quoted_titles": ["American Depositary Shares"],
                "ordinary_titles": ["Ordinary Shares"],
            }
        }
    }
    basis = QuantityBasis(ShareBasis(df_events, df_events, df_events, policy), pd.Timestamp("2024-06-11"))
    df_trades = pd.DataFrame(
        {
            "ticker": ["AAA"] * 3,
            "transaction_day": pd.to_datetime(["2024-06-07"] * 3),
            "security_title": ["Ordinary Shares", "American Depositary Shares", "Common Stock"],
            "shares_n": [10.0] * 3,
            "shares_owned_after": [100.0] * 3,
            "value": [500.0] * 3,
        }
    )
    df_rebased = position_units(df_trades, basis)
    assert df_rebased["shares_n"].tolist() == [5, 10, 10]
    assert df_rebased["shares_owned_after"].tolist() == [50, 100, 100]
    assert df_rebased["units_known"].tolist() == [True, True, False]
    assert df_rebased["declared_units_known"].tolist() == [True, True, False]
    pd.testing.assert_series_equal(df_rebased["value"], df_trades["value"])
    print(
        "SANITY: declared ordinary/ADS titles apply0.5/1 quantity conversion once; ambiguous common title stays unverified and all executionUSD is unchanged."
    )


@pytest.mark.parametrize(
    "title,role",
    [
        ("EVP,CFO & Treasurer", "CFO"),
        ("Executive Vice President,CFO", "CFO"),
        ("C.F.O.", "CFO"),
        ("CFO, Controller", "CFO"),
        ("CEO & Co-Founder", "CEO"),
        ("Principal Financial Officer", "CFO"),
        ("Group Chief Executive Officer", "CEO"),
        ("CEO, President", "CEO"),
        ("Group Chief Exec-Health & PS", "other_named_officer"),
        ("CEO of Diagnostics", "other_named_officer"),
        ("CEO,Royal Caribbean Intl", "other_named_officer"),
        ("Division Chief Financial Officer", "other_named_officer"),
        ("CFO, Sr. VP & Treasurer", "CFO"),
        ("CFO and VP", "CFO"),
        ("Chief Financial Officer and Senior Vice President, Finance", "CFO"),
        ("CFO, Sr. Vice Pres. & Treasurer", "CFO"),
        ("EVP; CFO; Treasurer & Asst Sect", "CFO"),
        ("CFO & MD Finance & Corp Dev", "CFO"),
        ("CFO & CBO", "CFO"),
        ("CEO - Interim", "CEO"),
        ("Former CFO", "other_named_officer"),
        ("Chief Financial Officer of Acme LLC", "other_named_officer"),
    ],
)
def test_role_boundaries_and_issuer_scope(title: str, role: str) -> None:
    assert officer_role(title) == role
    print(f"SANITY: {title!r} -> {role}; punctuation retains role and issuer scope.")


def _observed_panel(rows: list[dict], *, complete: bool = True) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", "2024-12-31")
    tickers = ("AAA", "BBB", "CCC")
    close = pd.DataFrame(100.0, index=idx, columns=tickers)
    history = pd.DataFrame(
        [{"ticker": t, "as_of": pd.Timestamp("2023-01-01"), "sharesOutstanding": 1000.0, "sharesOutstandingPit": 1000.0} for t in tickers]
    )
    return build_insider_feature_panel(
        make_frames(idx, _peers(tickers), close_split=close),
        _frame(rows),
        shares_out_history=history,
        df_valuation_close=close,
        coverage=InsiderCoverage.from_availability(
            InstitutionalAvailability.from_config({Tables.insider_transactions.name: {"__all__": "2024-01-01"}}),
            pd.Series(idx.max(), index=tickers) if complete else None,
        ),
    ).set_index(["date", "ticker"])


def _source_row(**changes: object) -> dict:
    row = _txn(accession_number="source", filing_date="2024-01-02", transaction_date="2024-01-02", transaction_code="G")
    row.update(changes)
    return row


def test_complete_quiet_windows_do_not_require_a_prior_purchase() -> None:
    rows = [_source_row(), _source_row(ticker="BBB", accession_number="future", filing_date="2024-08-01")]
    panel = _observed_panel(rows)
    ratios = [
        "f_ic_insider_net_buy_ratio_180d",
        "f_ic_insider_owner_surprise_120d",
        "f_ic_insider_purchase_pct_prior",
        "f_ic_insider_management_position_change_60d",
    ]
    assert len(panel.columns) == 9 and not set(ratios).intersection(panel.columns)
    assert panel.loc[("2024-07-01", "AAA")].eq(0).all()
    assert panel.loc[("2024-07-01", "BBB")].isna().all()
    assert panel.loc[("2024-09-01", "BBB")].eq(0).all()
    assert panel.loc[("2024-09-01", "CCC")].isna().all()
    assert panel.reindex(columns=ratios).isna().all().all()
    print("SANITY: nine confirmed quiet gross/count signals are zero; four undefined ratios remain NULL and future/no issuer evidence stays unknown.")


@pytest.mark.parametrize(
    "changes",
    [
        {"shares": np.nan},
        {"shares": 0.0},
        {"shares_owned_after": np.nan},
        {"shares": 10.0, "shares_owned_after": 10.0},
        {"direct_indirect": "I"},
        {"n_reporting_owners": 2},
        {"officer_title": ""},
        {"security_title": ""},
        {"security_type": None},
    ],
)
def test_active_unsupported_trade_never_becomes_quiet_zero(changes: dict) -> None:
    active = _txn(accession_number="active", filing_date="2024-07-01", transaction_date="2024-07-01", officer_title="CEO", is_officer=1)
    active.update(changes)
    panel = _observed_panel([_source_row(), active])
    ratios = ["f_ic_insider_management_position_change_60d", "f_ic_insider_owner_surprise_120d", "f_ic_insider_purchase_pct_prior"]
    panel = panel.reindex(columns=list(dict.fromkeys([*panel.columns, *ratios])))
    assert pd.isna(panel.loc[("2024-07-01", "AAA"), "f_ic_insider_management_position_change_60d"])
    assert pd.isna(panel.loc[("2024-07-01", "AAA"), "f_ic_insider_owner_surprise_120d"])
    if changes.get("officer_title") != "":
        assert pd.isna(panel.loc[("2024-07-01", "AAA"), "f_ic_insider_purchase_pct_prior"])
    assert pd.isna(panel.loc[("2024-08-30", "AAA"), "f_ic_insider_management_position_change_60d"])
    assert pd.isna(panel.loc[("2024-10-29", "AAA"), "f_ic_insider_owner_surprise_120d"])
    print("SANITY: active unsupported position/rank remain NULL; expiry cannot turn no-prior or no-scored-buy division into quiet0.")


def test_quiet_zero_requires_frontier_and_available_denominator() -> None:
    active = _txn(accession_number="sale", filing_date="2024-01-02", transaction_date="2024-01-02", transaction_code="S")
    panel = _observed_panel([active], complete=False)
    for name in ("f_ic_insider_purchase_pct_prior", "f_ic_insider_management_position_change_60d", "f_ic_insider_owner_surprise_120d"):
        assert name not in panel or pd.isna(panel.loc[("2024-07-01", "AAA"), name])
    print("SANITY: max filing date alone does not establish a complete quiet window.")


def test_unknown_quantity_amendment_retires_without_reaging() -> None:
    original = _txn(
        accession_number="original",
        filing_date="2024-07-01",
        transaction_date="2024-07-01",
        shares=np.nan,
        value_usd=np.nan,
        shares_owned_after=110,
        officer_title="CEO",
        is_officer=1,
        document_type="4",
        original_submission_date=None,
    )
    amended = dict(
        original,
        accession_number="amended",
        filing_date="2024-07-10",
        document_type="4/A",
        original_submission_date="2024-07-01",
        shares=10.0,
        value_usd=1000.0,
    )
    panel = _observed_panel([_source_row(), original, amended])
    for name, expected in (
        ("f_ic_insider_buy_value_mcap_60d", 0.01),
        ("f_ic_insider_purchase_pct_prior", 0.1),
        ("f_ic_insider_management_position_change_60d", 0.1),
    ):
        assert pd.isna(panel.loc[("2024-07-09", "AAA"), name])
        assert panel.loc[("2024-07-10", "AAA"), name] == pytest.approx(expected)
        expired = panel.loc[("2024-08-30", "AAA"), name]
        assert expired == 0 if name == "f_ic_insider_buy_value_mcap_60d" else pd.isna(expired)
    print("SANITY: 4/A retires missing quantity on publication, restores exact amounts, and expires at original disclosure+60.")


def test_unknown_sale_quantity_blocks_same_owner_management() -> None:
    buy = _txn(
        accession_number="buy",
        filing_date="2024-07-01",
        transaction_date="2024-07-01",
        shares=10.0,
        shares_owned_after=110.0,
        officer_title="CEO",
        is_officer=1,
    )
    sell = _txn(
        accession_number="unknown_sale",
        filing_date="2024-07-10",
        transaction_date="2024-07-10",
        transaction_code="S",
        shares=np.nan,
        value_usd=np.nan,
        officer_title="President",
        is_officer=1,
    )
    panel = _observed_panel([_source_row(), buy, sell])
    assert panel.loc[("2024-07-09", "AAA"), "f_ic_insider_management_position_change_60d"] == pytest.approx(0.1)
    assert pd.isna(panel.loc[("2024-07-10", "AAA"), "f_ic_insider_management_position_change_60d"])
    print("SANITY: unknown same-owner sale contaminates current management holdings prospectively even under a different title.")


def test_expired_unknown_quantity_does_not_mask_new_position() -> None:
    old = _txn(
        accession_number="old_unknown",
        filing_date="2024-01-02",
        transaction_date="2024-01-02",
        transaction_code="S",
        shares=np.nan,
        officer_title="CEO",
        is_officer=1,
    )
    new = _txn(
        accession_number="new_known",
        filing_date="2024-07-01",
        transaction_date="2024-07-01",
        shares=10.0,
        shares_owned_after=110.0,
        officer_title="CEO",
        is_officer=1,
    )
    panel = _observed_panel([old, new])
    assert panel.loc[("2024-07-01", "AAA"), "f_ic_insider_management_position_change_60d"] == pytest.approx(0.1)
    assert panel.loc[("2024-07-01", "AAA"), "f_ic_insider_purchase_pct_prior"] == pytest.approx(0.1)
    print("SANITY: an expired unknown sale cannot mask a separately supported later position window.")


@pytest.mark.parametrize("same_economic_day", [False, True])
def test_account_counted_once_and_lots_expire_separately(same_economic_day: bool) -> None:
    rows = [
        _txn(accession_number="first", shares=10, value_usd=1000, shares_owned_after=110, officer_title="CEO", is_officer=1),
        _txn(
            accession_number="second",
            filing_date="2015-06-10",
            transaction_date="2015-05-29" if same_economic_day else "2015-06-09",
            shares=20,
            value_usd=2000,
            shares_owned_after=130,
            officer_title="CEO",
            is_officer=1,
        ),
    ]
    idx = pd.date_range("2015-06-01", "2015-08-10")
    quotes = pd.DataFrame({"AAA": 100.0}, index=pd.date_range("2015-05-01", "2015-08-10"))
    frames = make_frames(idx, _peers(), close_split=quotes)
    expected = {"2015-06-09": 0.1, "2015-06-10": 0.3, "2015-07-31": 20 / 110}
    for order in (rows, rows[::-1]):
        panel = build_insider_feature_panel(frames, _frame(order), df_valuation_close=quotes).set_index("date")
        for name in ("f_ic_insider_management_position_change_60d", "f_ic_insider_purchase_pct_prior"):
            for day, value in expected.items():
                assert panel.loc[day, name] == pytest.approx(value)
            assert pd.isna(panel.loc["2015-08-09", name])
        prefix = build_insider_feature_panel(frames, _frame(rows[:1]), df_valuation_close=quotes).set_index("date")
        pd.testing.assert_series_equal(
            panel.loc[:"2015-06-09", "f_ic_insider_management_position_change_60d"],
            prefix.loc[:"2015-06-09", "f_ic_insider_management_position_change_60d"],
        )
    print("SANITY: 10/100 then30/100; first expiry20/110; second expiry unknown; one account denominator and invariant historical prefix/order.")


def test_dollar_totals_use_action_cap_and_real_expiry() -> None:
    rows = [_txn(accession_number="flow", shares=10_000, value_usd=1_000_000, shares_owned_after=110_000, officer_title="CEO", is_officer=1)]
    idx = pd.date_range("2015-05-29", "2015-12-01")
    close = pd.DataFrame({"AAA": np.where(idx < pd.Timestamp("2015-06-10"), 100.0, 200.0)}, index=idx)
    shares = pd.DataFrame([{"ticker": "AAA", "as_of": pd.Timestamp("2014-01-01"), "sharesOutstanding": 10_000_000.0}])
    panel = build_insider_feature_panel(
        make_frames(idx, _peers(), close_split=close),
        _frame(rows),
        shares_out_history=shares,
        coverage=InsiderCoverage.from_availability(None, pd.Series({"AAA": idx.max()})),
    ).set_index("date")
    assert pd.isna(panel.loc["2015-05-29", "f_ic_insider_buy_value_mcap_60d"])
    assert panel.loc["2015-06-01", "f_ic_insider_buy_value_mcap_60d"] == pytest.approx(0.001)
    assert panel.loc["2015-06-10", "f_ic_insider_buy_value_mcap_60d"] == pytest.approx(0.001)
    assert panel.loc["2015-07-30", "f_ic_insider_buy_value_mcap_60d"] == pytest.approx(0.001)
    assert panel.loc["2015-07-31", "f_ic_insider_buy_value_mcap_60d"] == 0
    assert panel.loc["2015-11-27", "f_ic_insider_ceo_buy_mcap_180d"] == pytest.approx(0.001)
    assert panel.loc["2015-11-28", "f_ic_insider_ceo_buy_mcap_180d"] == 0
    later = _txn(
        accession_number="later",
        filing_date="2015-06-12",
        transaction_date="2015-06-11",
        shares=5000,
        price_per_share=200,
        value_usd=1_000_000,
        shares_owned_after=115_000,
        officer_title="CEO",
        is_officer=1,
    )
    multiple = build_insider_feature_panel(
        make_frames(idx, _peers(), close_split=close), _frame([*rows, later]), shares_out_history=shares
    ).set_index("date")
    assert multiple.loc["2015-06-12", "f_ic_insider_buy_value_mcap_60d"] == pytest.approx(0.0015)
    print("SANITY: 1m/1bn stays .001 after cap doubles; two action caps sum .0015; public start and original 60/180-day expiry are unchanged.")


@pytest.mark.parametrize("bad_cap", [np.nan, 0.0, -1.0, np.inf])
def test_action_cap_missing_active_trade_and_strict_prior_session(bad_cap: float) -> None:
    idx = pd.bdate_range("2024-06-03", "2024-06-14")
    close = pd.DataFrame(100.0, index=idx, columns=["AAA"])
    if pd.isna(bad_cap):
        close.loc["2024-06-07", "AAA"] = np.nan
    shares = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "as_of": pd.to_datetime(["2024-06-03", "2024-06-07", "2024-06-10"]),
            "sharesOutstanding": [10_000_000.0, bad_cap, 10_000_000.0],
        }
    )
    rows = [
        _txn(accession_number="known", filing_date="2024-06-05", transaction_date="2024-06-04", shares=10_000, shares_owned_after=110_000),
        _txn(accession_number="missing", filing_date="2024-06-10", transaction_date="2024-06-07", shares=10_000, shares_owned_after=120_000),
    ]
    panel = build_insider_feature_panel(make_frames(idx, _peers(), close_split=close), _frame(rows), shares_out_history=shares).set_index("date")
    assert panel.loc["2024-06-05", "f_ic_insider_buy_value_mcap_60d"] == pytest.approx(0.001)
    assert pd.isna(panel.loc["2024-06-10", "f_ic_insider_buy_value_mcap_60d"])
    assert panel.loc["2024-06-10", "f_ic_insider_distinct_buyers_120d"] == 1
    close.loc["2024-06-07", "AAA"] = np.nan
    weekend = dict(rows[1], transaction_date="2024-06-09")
    missing_bar = build_insider_feature_panel(
        make_frames(idx, _peers(), close_split=close), _frame([rows[0], weekend]), shares_out_history=shares
    ).set_index("date")
    assert pd.isna(missing_bar.loc["2024-06-10", "f_ic_insider_buy_value_mcap_60d"])
    print(
        "SANITY: an unsupported active cap masks the sum without erasing independent buyer count; weekend lookup cannot jump over a missing Friday."
    )


def _action_panel(
    rows: list[dict], close: pd.DataFrame, shares: pd.DataFrame | None, identity: pd.DataFrame | None = None, complete: bool = False
) -> pd.DataFrame:
    coverage = InsiderCoverage.from_availability(None, pd.Series({"AAA": close.index.max()}) if complete else None, identity)
    return build_insider_feature_panel(
        make_frames(close.index, _peers(), close_split=close), _frame(rows), shares_out_history=shares, df_valuation_close=close, coverage=coverage
    ).set_index("date")


@pytest.mark.parametrize("case", ["supported", "missing", "seam", "conflict", "bar", "holiday", "class"])
def test_action_cap_non_session_requires_continuous_identity(case: str) -> None:
    idx = pd.bdate_range("2024-06-03", "2024-06-14")
    if case == "holiday":
        idx = idx.drop(pd.Timestamp("2024-06-07"))
    close = pd.DataFrame(100.0, index=idx, columns=["AAA"])
    shares = pd.DataFrame({"ticker": ["AAA"], "as_of": [pd.Timestamp("2024-06-03")], "sharesOutstanding": [10_000_000.0]})
    row = _txn(
        accession_number="weekend",
        filing_date="2024-06-10",
        transaction_date="2024-06-07" if case == "holiday" else "2024-06-09",
        shares=10_000,
        shares_owned_after=110_000,
    )
    line = dict(
        canonical_company="AAA",
        source="ftd",
        security_id="C000000000",
        source_symbol="AAA",
        security_class="common",
        conversion_ratio=1.0,
        lineage_role="canonical_current",
        valid_from="2024-06-03",
        valid_to=None,
    )
    if case == "seam":
        line["valid_from"] = "2024-06-08"
    if case == "bar":
        close.loc["2024-06-07", "AAA"] = np.nan
    if case == "class":
        row["security_title"] = "Class A Common Stock"
        line["security_class"] = "class_B"
    identity = pd.DataFrame([line, dict(line, canonical_company="BBB")] if case == "conflict" else [line])
    panel = _action_panel([_source_row(), row], close, shares, None if case == "missing" else identity, complete=True)
    value = panel.loc["2024-06-10", "f_ic_insider_buy_value_mcap_60d"]
    assert pd.isna(value) if case not in {"supported", "holiday"} else value == pytest.approx(0.001)
    assert panel.loc["2024-06-06", "f_ic_insider_buy_value_mcap_60d"] == 0
    print(f"SANITY: {case} non-session action uses only the chosen prior exchange session with one coherent compatible canonical identity.")


def test_action_cap_survives_missing_current_cap_and_quiet_needs_none() -> None:
    idx = pd.bdate_range("2024-06-03", "2024-06-14")
    close = pd.DataFrame(100.0, index=idx, columns=["AAA"])
    shares = pd.DataFrame({"ticker": ["AAA", "AAA"], "as_of": pd.to_datetime(["2024-06-03", "2024-06-10"]), "sharesOutstanding": [10_000_000.0, 0.0]})
    rows = [_txn(accession_number="known", filing_date="2024-06-05", transaction_date="2024-06-04", shares=10_000, shares_owned_after=110_000)]
    panel = _action_panel(rows, close, shares, complete=True)
    for window in (60, 180):
        assert panel.loc["2024-06-11", f"f_ic_insider_buy_value_mcap_{window}d"] == pytest.approx(0.001)
    quiet = _action_panel([_source_row()], close, None, complete=True)
    columns = [name for name in quiet if "mcap" in name]
    assert len(columns) == 7 and quiet.loc["2024-06-11", columns].eq(0).all()
    print(
        "SANITY: supported historical ratios survive missing current cap; all seven source-complete quiet intensities are zero without an action denominator."
    )


def test_action_cap_all_selectors_preserve_raw_dollar_consumers() -> None:
    idx = pd.bdate_range("2024-06-03", "2024-06-14")
    close = pd.DataFrame({"AAA": np.where(idx < pd.Timestamp("2024-06-07"), 100.0, 200.0)}, index=idx)
    shares = pd.DataFrame({"ticker": ["AAA"], "as_of": [pd.Timestamp("2024-06-03")], "sharesOutstanding": [10_000_000.0]})
    base = dict(shares=10_000, shares_owned_after=110_000, is_officer=1, is_director=0, is_10b5_1=0)
    rows = [
        _txn(**base, accession_number="ceo", filing_date="2024-06-05", transaction_date="2024-06-04", officer_title="CEO"),
        _txn(
            **dict(base, shares=5000, shares_owned_after=115_000),
            accession_number="cfo",
            filing_date="2024-06-10",
            transaction_date="2024-06-07",
            price_per_share=200,
            officer_title="CFO",
        ),
        _txn(
            **dict(base, owner_cik="0002", is_director=1),
            accession_number="director",
            filing_date="2024-06-06",
            transaction_date="2024-06-05",
            officer_title="President",
        ),
        _txn(**dict(base, owner_cik="0003"), accession_number="disc", filing_date="2024-06-06", transaction_date="2024-06-05", transaction_code="S"),
        _txn(
            **dict(base, owner_cik="0004", is_10b5_1=1),
            accession_number="plan",
            filing_date="2024-06-10",
            transaction_date="2024-06-07",
            price_per_share=200,
            transaction_code="S",
        ),
    ]
    panel = _action_panel(rows, close, shares, complete=True)
    expected = {
        "buy_value_mcap_60d": 0.0025,
        "buy_value_mcap_180d": 0.0025,
        "ceo_buy_mcap_180d": 0.001,
        "cfo_buy_mcap_180d": 0.0005,
        "director_buy_mcap_180d": 0.001,
        "discretionary_sell_mcap_60d": 0.001,
        "planned_sell_mcap_60d": 0.001,
    }
    for field, value in expected.items():
        assert panel.loc["2024-06-10", "f_ic_insider_" + field] == pytest.approx(value)
    assert panel.loc["2024-06-10", "f_ic_insider_net_buy_ratio_180d"] == pytest.approx(0.0)
    assert panel.loc["2024-06-10", "f_ic_insider_owner_surprise_120d"] == pytest.approx(1.0)
    print(
        "SANITY: all seven role/plan selectors sum individual action ratios; gross buys, raw-USD net ratio and raw-USD owner surprise retain their definitions."
    )


@pytest.mark.parametrize("case", ["remarks", "quantity", "fractional", "indirect", "joint", "former"])
def test_prepared_role_and_unsupported_scope_mask_finite_management(case: str) -> None:
    idx = pd.bdate_range("2024-06-03", "2024-06-14")
    close = pd.DataFrame(100.0, index=idx, columns=["AAA"])
    manager = _txn(
        accession_number="supported",
        filing_date="2024-06-05",
        transaction_date="2024-06-04",
        shares=10,
        shares_owned_after=110,
        owner_cik="0001",
        owner_ciks="0001",
        n_reporting_owners=1,
        is_officer=1,
        officer_title="CFO",
    )
    other = dict(
        manager,
        accession_number="possible",
        filing_date="2024-06-06",
        transaction_date="2024-06-05",
        owner_cik="0002",
        owner_ciks="0002",
        officer_title="See Remarks",
    )
    if case == "quantity":
        other.update(officer_title="CEO", shares=np.nan)
    if case == "fractional":
        other.update(officer_title="CEO", n_reporting_owners=1.5)
    if case == "indirect":
        other["direct_indirect"] = "I"
    if case == "joint":
        other.update(n_reporting_owners=2, owner_ciks="0002,0003")
    if case == "former":
        other["officer_title"] = "Former CFO"
    shares = pd.DataFrame({"ticker": ["AAA"], "as_of": [pd.Timestamp("2024-06-03")], "sharesOutstanding": [10_000_000.0]})
    panel = _action_panel([manager, other], close, shares, complete=True)
    field = "f_ic_insider_management_position_change_60d"
    assert panel.loc["2024-06-05", field] == pytest.approx(0.1)
    value = panel.loc["2024-06-06", field]
    assert pd.isna(value) if case in {"remarks", "quantity", "fractional"} else value == pytest.approx(0.1)
    if case == "remarks":
        assert pd.isna(panel.loc["2024-06-06", "f_ic_insider_ceo_buy_mcap_180d"])
        assert pd.isna(panel.loc["2024-06-06", "f_ic_insider_cfo_buy_mcap_180d"])
    print(
        f"SANITY: prepared {case} classification gates the finite full-management pool prospectively; proved indirect/joint/former exclusions preserve the supported CFO."
    )


def test_action_cap_economic_date_amendment_preserves_public_prefix_and_expiry() -> None:
    idx = pd.bdate_range("2024-06-03", "2024-08-12")
    close = pd.DataFrame({"AAA": np.where(idx < pd.Timestamp("2024-06-07"), 100.0, 200.0)}, index=idx)
    shares = pd.DataFrame({"ticker": ["AAA"], "as_of": [pd.Timestamp("2024-06-03")], "sharesOutstanding": [10_000_000.0]})
    original = _txn(
        accession_number="original",
        filing_date="2024-06-05",
        transaction_date="2024-06-04",
        shares=10_000,
        shares_owned_after=110_000,
        owner_ciks="0001",
        n_reporting_owners=1,
        document_type="4",
        original_submission_date=None,
        row_sequence=1,
    )
    amended = dict(
        original,
        accession_number="amended",
        filing_date="2024-06-11",
        transaction_date="2024-06-07",
        document_type="4/A",
        original_submission_date="2024-06-05",
    )
    panel = _action_panel([original, amended], close, shares, complete=True)
    prefix = _action_panel([original], close, shares, complete=True)
    field = "f_ic_insider_buy_value_mcap_60d"
    pd.testing.assert_series_equal(panel.loc[:"2024-06-10", field], prefix.loc[:"2024-06-10", field])
    assert panel.loc["2024-06-10", field] == pytest.approx(0.001)
    assert panel.loc["2024-06-11", field] == pytest.approx(0.0005)
    assert panel.loc["2024-08-02", field] == pytest.approx(0.0005)
    assert panel.loc["2024-08-05", field] == 0
    print(
        "SANITY: the uniquely corresponding 4/A economic-date correction switches .001 to .0005 only when public and expires from the original June5 disclosure."
    )


@pytest.mark.parametrize("case", ["known", "absent", "ambiguous"])
def test_action_cap_requires_known_source_security_units(case: str) -> None:
    idx = pd.bdate_range("2024-06-03", "2024-06-14")
    close = pd.DataFrame(100.0, index=idx, columns=["AAA"])
    shares = pd.DataFrame({"ticker": ["AAA"], "as_of": [pd.Timestamp("2024-06-03")], "sharesOutstanding": [10_000_000.0]})
    events = pd.DataFrame(columns=["ticker", "date", "value"])
    policy = {
        "AAA": {
            "quote_units": {
                "date": "2024-06-10",
                "before": 2,
                "after": 1,
                "quoted_titles": ["American Depositary Shares"],
                "ordinary_titles": ["Ordinary Shares"],
            }
        }
    }
    basis = None if case == "absent" else ShareBasis(events, events, events, policy if case == "ambiguous" else {})
    rows = [
        _source_row(),
        _txn(accession_number="unit-control", filing_date="2024-06-05", transaction_date="2024-06-04", shares=10_000, shares_owned_after=110_000),
    ]
    panel = build_insider_feature_panel(
        make_frames(idx, _peers(), close_split=close),
        _frame(rows),
        shares_out_history=shares,
        df_valuation_close=close,
        share_basis=basis,
        coverage=InsiderCoverage.from_availability(None, pd.Series({"AAA": idx.max()})),
    ).set_index("date")
    value = panel.loc["2024-06-05", "f_ic_insider_buy_value_mcap_60d"]
    assert value == pytest.approx(0.001) if case == "known" else pd.isna(value)
    assert panel.loc["2024-06-04", "f_ic_insider_buy_value_mcap_60d"] == 0
    assert panel.loc["2024-06-05", "f_ic_insider_distinct_buyers_120d"] == 1
    print(
        f"SANITY: {case} source-unit evidence gates action capital alone; ordinary empty-event basis proves .001, unavailable/ambiguous units stay NULL without erasing the buyer."
    )


@pytest.mark.parametrize("case", ["missing_row", "owner", "account", "class", "cardinality", "ambiguous_original", "amount"])
def test_economic_date_correction_needs_proved_row_correspondence(case: str) -> None:
    original = _txn(
        accession_number="a-original",
        filing_date="2024-06-05",
        transaction_date="2024-06-04",
        shares=10_000,
        shares_owned_after=110_000,
        owner_ciks="0001",
        n_reporting_owners=1,
        document_type="4",
        original_submission_date=None,
        row_sequence=1,
    )
    amended = dict(
        original,
        accession_number="amended",
        filing_date="2024-06-11",
        transaction_date="2024-06-07",
        document_type="4/A",
        original_submission_date="2024-06-05",
    )
    if case == "missing_row":
        amended["row_sequence"] = None
    if case == "owner":
        amended["owner_cik"] = "0002"
    if case == "account":
        amended["direct_indirect"] = "I"
    if case == "class":
        amended["security_title"] = "Class B Common Stock"
    if case == "amount":
        amended["shares"] = 20_000
    rows = [original, amended]
    if case == "cardinality":
        rows.append(dict(amended, row_sequence=2, shares_owned_after=120_000))
    if case == "ambiguous_original":
        rows.append(dict(original, accession_number="b-original", shares=5000, shares_owned_after=105_000))
    clean, _ = clean_transactions(_frame(rows))
    kept = clean.loc[clean["accession_number"].eq("a-original")]
    assert len(kept) == 1 and kept["visible_until"].isna().all()
    amendment = clean.loc[clean["accession_number"].eq("amended")]
    assert not amendment.empty and amendment["anchor"].eq(pd.Timestamp("2024-06-11")).all()
    print(f"SANITY: {case} cannot prove a date-only replacement; the original is not silently withdrawn or its anchor transferred.")


@pytest.mark.parametrize("intervening_gift", [False, True])
def test_unsupported_account_span_abstains_prospectively(intervening_gift: bool) -> None:
    rows = [
        _txn(accession_number="first", shares=10, value_usd=1000, shares_owned_after=110, officer_title="CEO", is_officer=1),
        _txn(
            accession_number="second",
            filing_date="2015-06-10",
            transaction_date="2015-06-09",
            shares=20,
            value_usd=2000,
            shares_owned_after=140 if not intervening_gift else 130,
            officer_title="CEO",
            is_officer=1,
        ),
    ]
    if intervening_gift:
        rows.append(_txn(accession_number="gift", transaction_code="G", filing_date="2015-06-12", transaction_date="2015-06-05"))
    idx = pd.date_range("2015-06-01", "2015-06-15")
    quotes = pd.DataFrame({"AAA": 100.0}, index=pd.date_range("2015-05-01", "2015-06-15"))
    panel = build_insider_feature_panel(make_frames(idx, _peers(), close_split=quotes), _frame(rows), df_valuation_close=quotes).set_index("date")
    name = "f_ic_insider_management_position_change_60d"
    assert panel.loc["2015-06-09", name] == pytest.approx(0.1)
    blocked = "2015-06-12" if intervening_gift else "2015-06-10"
    assert panel.loc[blocked:, name].isna().all()
    if intervening_gift:
        assert panel.loc["2015-06-10", name] == pytest.approx(0.3)
    print("SANITY: inconsistent balances or a newly public intervening gift make the account unknown without rewriting its earlier history.")


def test_genuine_split_units_cancel_level_actions() -> None:
    rows = [
        _txn(accession_number="before", shares=10, value_usd=1000, shares_owned_after=110, officer_title="CEO", is_officer=1),
        _txn(
            accession_number="after",
            filing_date="2015-06-10",
            transaction_date="2015-06-09",
            shares=20,
            price_per_share=50,
            value_usd=1000,
            shares_owned_after=240,
            officer_title="CEO",
            is_officer=1,
        ),
    ]
    dates = pd.date_range("2015-05-01", "2015-08-10")
    idx = dates[dates >= pd.Timestamp("2015-06-01")]
    split_basis = pd.DataFrame({"AAA": 40.0}, index=dates)
    level = pd.DataFrame({"AAA": 1.25}, index=dates)
    quotes = pd.DataFrame({"AAA": np.where(dates < pd.Timestamp("2015-06-05"), 100.0, 50.0)}, index=dates)
    frames = make_frames(idx, _peers(), close_split=split_basis, level_factor=level)
    df_events = pd.DataFrame({"ticker": ["AAA"], "date": pd.to_datetime(["2015-06-05"]), "value": [2.0]})
    share_basis = ShareBasis(df_events, df_events, df_events, {})
    panel = build_insider_feature_panel(frames, _frame(rows), df_valuation_close=quotes, share_basis=share_basis).set_index("date")
    name = "f_ic_insider_management_position_change_60d"
    assert panel.loc["2015-06-10", name] == pytest.approx(40 / 200)
    assert panel.loc["2015-07-31", name] == pytest.approx(20 / 220)
    print("SANITY: pre-split10 becomes20 shares; pooled40/200, later20/220; level1.25 cancels instead of altering share units.")


def test_unknown_split_conversion_abstains_for_multiple_days() -> None:
    rows = [
        _txn(accession_number="first", shares=10, value_usd=1000, shares_owned_after=110, officer_title="CEO", is_officer=1),
        _txn(
            accession_number="later",
            filing_date="2015-06-10",
            transaction_date="2015-06-09",
            shares=20,
            value_usd=2000,
            shares_owned_after=130,
            officer_title="CEO",
            is_officer=1,
        ),
    ]
    idx = pd.date_range("2015-06-01", "2015-06-15")
    panel = build_insider_feature_panel(make_frames(idx, _peers()), _frame(rows), share_basis=None).set_index("date")
    name = "f_ic_insider_management_position_change_60d"
    assert panel.loc["2015-06-09", name] == pytest.approx(0.1)
    assert panel.loc["2015-06-10":, name].isna().all()
    print("SANITY: a single economic day is scale-free; missing cross-day share conversion cannot certify a rolling account ledger.")


def test_late_economic_trade_uses_latest_included_terminal() -> None:
    rows = [
        _txn(
            accession_number="early_public",
            filing_date="2015-06-01",
            transaction_date="2015-06-01",
            shares=20,
            value_usd=2000,
            shares_owned_after=130,
            officer_title="CEO",
            is_officer=1,
        ),
        _txn(
            accession_number="late_public",
            filing_date="2015-06-20",
            transaction_date="2015-05-01",
            shares=10,
            value_usd=1000,
            shares_owned_after=110,
            officer_title="CEO",
            is_officer=1,
        ),
    ]
    dates = pd.date_range("2015-05-01", "2015-08-20")
    idx = dates[dates >= pd.Timestamp("2015-06-01")]
    quotes = pd.DataFrame({"AAA": 100.0}, index=dates)
    panel = build_insider_feature_panel(make_frames(idx, _peers(), close_split=quotes), _frame(rows), df_valuation_close=quotes).set_index("date")
    name = "f_ic_insider_management_position_change_60d"
    assert panel.loc["2015-06-19", name] == pytest.approx(20 / 110)
    assert panel.loc["2015-06-20", name] == pytest.approx(30 / 100)
    assert panel.loc["2015-07-31", name] == pytest.approx(10 / 100)
    assert pd.isna(panel.loc["2015-08-19", name])
    print("SANITY: late economic trade becomes usable only when public; after newer trade expiry its own110 terminal gives10/100.")


def test_amendment_changes_shares_prospectively_without_refreshing_expiry() -> None:
    rows = [
        _txn(accession_number="first", shares=10, value_usd=1000, shares_owned_after=110, officer_title="CEO", is_officer=1, document_type="4"),
        _txn(
            accession_number="second",
            filing_date="2015-06-10",
            transaction_date="2015-06-09",
            shares=20,
            value_usd=2000,
            shares_owned_after=140,
            officer_title="CEO",
            is_officer=1,
        ),
        _txn(
            accession_number="correction",
            filing_date="2015-06-20",
            shares=20,
            value_usd=2000,
            shares_owned_after=120,
            officer_title="CEO",
            is_officer=1,
            document_type="4/A",
            original_submission_date="2015-06-01",
        ),
    ]
    dates = pd.date_range("2015-05-01", "2015-08-25")
    idx = dates[dates >= pd.Timestamp("2015-06-01")]
    quotes = pd.DataFrame({"AAA": 100.0}, index=dates)
    frames = make_frames(idx, _peers(), close_split=quotes)
    name = "f_ic_insider_management_position_change_60d"
    panel = build_insider_feature_panel(frames, _frame(rows), df_valuation_close=quotes).set_index("date")
    prefix = build_insider_feature_panel(frames, _frame(rows[:2]), df_valuation_close=quotes).set_index("date")
    pd.testing.assert_series_equal(panel.loc[:"2015-06-19", name], prefix.loc[:"2015-06-19", name])
    assert panel.loc["2015-06-09", name] == pytest.approx(0.1)
    assert pd.isna(panel.loc["2015-06-19", name])
    assert panel.loc["2015-06-20", name] == pytest.approx(0.4)
    assert panel.loc["2015-07-31", name] == pytest.approx(20 / 120)
    assert pd.isna(panel.loc["2015-08-09", name])
    rows[2]["filing_date"] = "2015-08-15"
    late = build_insider_feature_panel(frames, _frame(rows), df_valuation_close=quotes).set_index("date")
    assert late.loc["2015-08-15":, name].isna().all()
    print("SANITY: amendment restores a supported40/100 prospectively; original expiry leaves20/120; an amendment after expiry revives nothing.")


def test_ineligible_owner_does_not_refresh_another_owners_clock() -> None:
    rows = [
        _txn(accession_number="ceo", owner_cik="11", officer_title="CEO", is_officer=1, shares=10, value_usd=1000, shares_owned_after=110),
        _txn(
            accession_number="cfo",
            owner_cik="12",
            officer_title="CFO",
            is_officer=1,
            filing_date="2015-06-10",
            shares=20,
            value_usd=2000,
            shares_owned_after=120,
        ),
        _txn(accession_number="gift", owner_cik="11", transaction_code="G", filing_date="2015-06-15"),
    ]
    idx = pd.date_range("2015-06-01", "2015-08-10")
    panel = build_insider_feature_panel(make_frames(idx, _peers()), _frame(rows)).set_index("date")
    name = "f_ic_insider_management_position_change_60d"
    assert panel.loc["2015-06-14", name] == pytest.approx(30 / 200)
    assert panel.loc["2015-06-15":"2015-07-30", name].isna().all()
    assert panel.loc["2015-07-31", name] == pytest.approx(0.2)
    assert panel.loc["2015-08-08", name] == pytest.approx(0.2)
    assert pd.isna(panel.loc["2015-08-09", name])
    print("SANITY: an unsupported active CEO keeps the pooled value unknown until its expiry; the CFO's June10 anchor remains valid through August8.")


def test_balance_identity_does_not_hide_one_share_on_a_large_account() -> None:
    rows = [
        _txn(accession_number="first", shares=1, value_usd=100, shares_owned_after=1_000_001, officer_title="CEO", is_officer=1),
        _txn(
            accession_number="later",
            filing_date="2015-06-10",
            transaction_date="2015-06-09",
            shares=1,
            value_usd=100,
            shares_owned_after=1_000_003,
            officer_title="CEO",
            is_officer=1,
        ),
    ]
    quotes = pd.DataFrame({"AAA": 100.0}, index=pd.date_range("2015-05-01", "2015-06-15"))
    idx = pd.date_range("2015-06-01", "2015-06-15")
    panel = build_insider_feature_panel(make_frames(idx, _peers(), close_split=quotes), _frame(rows), df_valuation_close=quotes).set_index("date")
    name = "f_ic_insider_management_position_change_60d"
    assert panel.loc["2015-06-09", name] == pytest.approx(1 / 1_000_000)
    assert panel.loc["2015-06-10":, name].isna().all()
    print("SANITY: exact share identity rejects an unexplained one-share gap even in a million-share holding.")


@pytest.mark.parametrize(
    "title,event_evidence,quote_mode,same_day",
    [
        ("Class A Common Stock", True, "both", False),
        ("Common Stock", True, "both", False),
        ("Common Stock", False, "none", False),
        ("Common Stock", False, "partial", False),
        ("Common Stock", True, "none", False),
        ("Class A Common Stock", True, "both", True),
    ],
)
def test_cross_account_pool_requires_compatible_share_units(title: str, event_evidence: bool, quote_mode: str, same_day: bool) -> None:
    rows = [
        _txn(
            accession_number="before",
            owner_cik="11",
            shares=10,
            value_usd=1000,
            shares_owned_after=110,
            officer_title="CEO",
            is_officer=1,
            security_title=title,
        ),
        _txn(
            accession_number="after",
            owner_cik="12",
            filing_date="2015-06-10",
            transaction_date="2015-05-29" if same_day else "2015-06-09",
            shares=60,
            price_per_share=100 if same_day else 50,
            value_usd=6000 if same_day else 3000,
            shares_owned_after=160 if same_day else 260,
            officer_title="CFO",
            is_officer=1,
            security_title=title,
        ),
    ]
    dates = pd.date_range("2015-05-01", "2015-08-10")
    idx = dates[dates >= pd.Timestamp("2015-06-01")]
    quotes = pd.DataFrame({"AAA": np.where(dates < pd.Timestamp("2015-06-05"), 100.0, 50.0)}, index=dates)
    if quote_mode == "partial":
        quotes.loc[dates < pd.Timestamp("2015-06-05")] = np.nan
    adjusted = pd.DataFrame({"AAA": 50.0}, index=dates)
    df_events = pd.DataFrame({"ticker": ["AAA"], "date": pd.to_datetime(["2015-06-05"]), "value": [2.0]})
    share_basis = ShareBasis(df_events, df_events, df_events, {}) if event_evidence else None
    panel = build_insider_feature_panel(
        make_frames(idx, _peers(), close_split=adjusted),
        _frame(rows),
        df_valuation_close=quotes if quote_mode != "none" else None,
        share_basis=share_basis,
    ).set_index("date")
    expected = 0.35 if same_day else 0.2 if title == "Common Stock" and event_evidence else np.nan
    for name in ("f_ic_insider_management_position_change_60d", "f_ic_insider_purchase_pct_prior"):
        assert panel.loc["2015-06-09", name] == pytest.approx(0.1)
        if pd.isna(expected):
            assert pd.isna(panel.loc["2015-06-10", name])
        else:
            assert panel.loc["2015-06-10", name] == pytest.approx(expected)
        assert panel.loc["2015-07-31", name] == pytest.approx(0.6 if same_day else 0.3)
    print(
        "SANITY: single-account ratios stay scale-free; pooled split units must be certified or share one economic date; genuine normalized pool is80/400."
    )
