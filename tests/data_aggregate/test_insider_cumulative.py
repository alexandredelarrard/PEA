"""Known-truth arithmetic for cumulative insider disclosure windows."""

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.common.availability import InstitutionalAvailability
from src.data_aggregate.utils.common.quotes import terminal_basis_factor
from src.data_aggregate.utils.institutionals.insider_features import BUY_VALUE_FEATURES, EMISSION, InsiderCoverage
from src.data_aggregate.utils.institutionals.insider_positions import _accumulate_intervals, position_units
from src.data_aggregate.utils.institutionals.insider_quality import clean_transactions, officer_role
from src.data_store.schema import Tables
from src.utils.price.adjustments import ShareBasis
from tests.conftest import make_frames
from tests.data_aggregate.insider_pit_reference import basis_frame
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
    df_clean, _ = clean_transactions(_frame(rows), df_valuation_close=df_quotes)
    df_rebased = position_units(df_clean, terminal_basis_factor(basis_frame(share_basis, ["AAA"], idx[-1])))
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


def test_single_class_split_converts_through_saved_basis_and_class_rows_use_the_ticker_basis() -> None:
    # Saved forward basis F: AAA splits 10:1 on 2024-06-10; BBB's saved rows stop before the grid end.
    idx = pd.bdate_range("2024-06-03", "2024-06-14")
    df_basis = pd.DataFrame({"AAA": np.where(idx < pd.Timestamp("2024-06-10"), 1.0, 10.0), "BBB": 1.0}, index=idx)
    df_basis.loc[idx > pd.Timestamp("2024-06-12"), "BBB"] = np.nan
    df_trades = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA", "BBB"],
            "transaction_day": pd.to_datetime(["2024-06-07", "2024-06-11", "2024-06-07", "2024-06-12"]),
            "security_title": ["Common Stock", "Common Stock", "Class B Common Stock", "Common Stock"],
            "shares_n": [5.0, 50.0, 5.0, 7.0],
            "shares_owned_after": [105.0, 1100.0, 105.0, 70.0],
            "value": [500.0] * 4,
        }
    )

    df_units = position_units(df_trades, terminal_basis_factor(df_basis))

    assert df_units["shares_n"].tolist() == [50.0, 50.0, 50.0, 7.0]
    assert df_units["shares_owned_after"].tolist() == [1050.0, 1100.0, 1050.0, 70.0]
    assert df_units["units_known"].tolist() == [True] * 4
    assert df_units["class_approximated"].tolist() == [False, False, True, False]
    pd.testing.assert_series_equal(df_units["value"], df_trades["value"])
    print(
        "SANITY: a pre-split common trade converts 5->50 by F_T/F_d=10; the class-B row takes the ticker factor too (5->50, flagged class_approximated) so one unit basis holds; "
        "a ticker whose saved basis ends early converts at its last saved F; execution USD unchanged."
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
            idx.max() if complete else None,
        ),
    ).set_index(["date", "ticker"])


def _source_row(**changes: object) -> dict:
    row = _txn(accession_number="source", filing_date="2024-01-02", transaction_date="2024-01-02", transaction_code="G")
    row.update(changes)
    return row


def test_complete_quiet_windows_do_not_require_a_prior_purchase() -> None:
    rows = [_source_row(), _source_row(ticker="BBB", accession_number="future", filing_date="2024-08-01")]
    panel = _observed_panel(rows)
    ratios = [f"f_{name}" for name in EMISSION if "pct_prior" in name or "net_buy_ratio" in name or "breadth" in name]
    counts = [f"f_{name}" for name in EMISSION if f"f_{name}" not in ratios]
    assert len(ratios) == 8 and len(counts) == 10
    assert panel.reindex(columns=counts).loc[("2024-07-01", "AAA")].eq(0).all()
    assert panel.loc[("2024-07-01", "BBB")].isna().all()
    assert panel.reindex(columns=counts).loc[("2024-09-01", "BBB")].eq(0).all()
    assert panel.loc[("2024-09-01", "CCC")].isna().all()
    assert panel.reindex(columns=ratios).isna().all().all()
    print(
        f"SANITY: {len(counts)} confirmed quiet gross/count signals are zero; {len(ratios)} undefined ratios remain NULL and future/no issuer evidence stays unknown."
    )


def test_quiet_zero_requires_frontier_and_available_denominator() -> None:
    active = _txn(accession_number="sale", filing_date="2024-01-02", transaction_date="2024-01-02", transaction_code="S")
    panel = _observed_panel([active], complete=False)
    for name in ("f_ic_insider_purchase_pct_prior_60d", "f_ic_insider_net_buy_ratio_60d", "f_ic_insider_buyer_breadth_120d"):
        assert name not in panel or pd.isna(panel.loc[("2024-07-01", "AAA"), name])
    print("SANITY: max filing date alone does not establish a complete quiet window.")


def test_dollar_totals_use_action_cap_and_real_expiry() -> None:
    rows = [_txn(accession_number="flow", shares=10_000, value_usd=1_000_000, shares_owned_after=110_000, officer_title="CEO", is_officer=1)]
    idx = pd.date_range("2015-05-29", "2015-12-01")
    close = pd.DataFrame({"AAA": np.where(idx < pd.Timestamp("2015-06-10"), 100.0, 200.0)}, index=idx)
    shares = pd.DataFrame([{"ticker": "AAA", "as_of": pd.Timestamp("2014-01-01"), "sharesOutstanding": 10_000_000.0}])
    panel = build_insider_feature_panel(
        make_frames(idx, _peers(), close_split=close),
        _frame(rows),
        shares_out_history=shares,
        coverage=InsiderCoverage.from_availability(None, idx.max()),
    ).set_index("date")
    assert pd.isna(panel.loc["2015-05-29", "f_ic_insider_buy_value_mcap_60d"])
    assert panel.loc["2015-06-01", "f_ic_insider_buy_value_mcap_60d"] == pytest.approx(0.001)
    assert panel.loc["2015-06-10", "f_ic_insider_buy_value_mcap_60d"] == pytest.approx(0.001)
    assert panel.loc["2015-07-30", "f_ic_insider_buy_value_mcap_60d"] == pytest.approx(0.001)
    assert panel.loc["2015-07-31", "f_ic_insider_buy_value_mcap_60d"] == 0
    assert panel.loc["2015-11-27", "f_ic_insider_c_suite_buy_mcap_180d"] == pytest.approx(0.001)
    assert panel.loc["2015-11-28", "f_ic_insider_c_suite_buy_mcap_180d"] == 0
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
    assert panel.loc["2024-06-10", "f_ic_insider_buy_value_mcap_60d"] == pytest.approx(0.001)
    assert panel.loc["2024-06-10", "f_ic_insider_distinct_buyers_120d"] == 1
    close.loc["2024-06-07", "AAA"] = np.nan
    weekend = dict(rows[1], transaction_date="2024-06-09")
    missing_bar = build_insider_feature_panel(
        make_frames(idx, _peers(), close_split=close), _frame([rows[0], weekend]), shares_out_history=shares
    ).set_index("date")
    assert missing_bar.loc["2024-06-10", "f_ic_insider_buy_value_mcap_60d"] == pytest.approx(0.001)
    print(
        "SANITY: a trade without a supported cap is left out of the sum (the known trade still counts) and the buyer count is kept; weekend lookup cannot jump over a missing Friday."
    )


def _action_panel(
    rows: list[dict], close: pd.DataFrame, shares: pd.DataFrame | None, identity: pd.DataFrame | None = None, complete: bool = False
) -> pd.DataFrame:
    coverage = InsiderCoverage.from_availability(None, close.index.max() if complete else None, identity)
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
    assert pd.isna(value) if case not in {"supported", "holiday", "class"} else value == pytest.approx(0.001)
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
    assert set(columns) == {f"f_{name}" for name in BUY_VALUE_FEATURES} and quiet.loc["2024-06-11", columns].eq(0).all()
    print(
        f"SANITY: supported historical ratios survive missing current cap; all {len(columns)} source-complete quiet intensities are zero without an action denominator."
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
            **dict(base, owner_cik="0002", is_director=1, is_officer=0),
            accession_number="director",
            filing_date="2024-06-06",
            transaction_date="2024-06-05",
            officer_title="",
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
        "buy_value_mcap_120d": 0.0025,
        "buy_value_mcap_180d": 0.0025,
        "c_suite_buy_mcap_180d": 0.0015,
        "management_buy_mcap_180d": 0.001,
        "ten_pct_buy_mcap_180d": 0.0,
    }
    for field, value in expected.items():
        assert panel.loc["2024-06-10", "f_ic_insider_" + field] == pytest.approx(value)
    assert panel.loc["2024-06-10", "f_ic_insider_net_buy_ratio_180d"] == pytest.approx(0.0)
    print(
        "SANITY: the three population selectors sum individual action ratios (C-suite 0.0015 + management 0.001 = total 0.0025); "
        "gross buys and the raw-USD net ratio retain their definitions."
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


@pytest.mark.parametrize("case", ["known", "absent"])
def test_action_cap_requires_the_saved_security_basis_and_cap(case: str) -> None:
    # Saved price-part legs: an unsupported-units ticker (APTV/CRH) has neither F nor mcap_pit.
    idx = pd.bdate_range("2024-06-03", "2024-06-14")
    close = pd.DataFrame(100.0, index=idx, columns=["AAA"])
    saved = 1.0 if case == "known" else np.nan
    frames = make_frames(
        idx,
        _peers(),
        close_split=close,
        mcap_pit=pd.DataFrame(1e9 * saved, index=idx, columns=["AAA"]),
        security_basis=pd.DataFrame(saved, index=pd.date_range("2024-01-01", idx.max()), columns=["AAA"]),
    )
    rows = [
        _source_row(),
        _txn(accession_number="unit-control", filing_date="2024-06-05", transaction_date="2024-06-04", shares=10_000, shares_owned_after=110_000),
    ]
    panel = build_insider_feature_panel(
        frames, _frame(rows), df_valuation_close=close, coverage=InsiderCoverage.from_availability(None, idx.max())
    ).set_index("date")
    value = panel.loc["2024-06-05", "f_ic_insider_buy_value_mcap_60d"]
    assert value == pytest.approx(0.001) if case == "known" else pd.isna(value)
    assert panel.loc["2024-06-04", "f_ic_insider_buy_value_mcap_60d"] == 0
    assert panel.loc["2024-06-05", "f_ic_insider_distinct_buyers_120d"] == 1
    print(
        f"SANITY: {case} saved basis/cap: $1M on a $1B saved cap gives .001; without a saved F and cap the value stays NULL and the buyer still counts."
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


def test_interval_sum_is_exact_when_overlapping_lots_expire() -> None:
    """Known truth: days with no active lot sum to exactly 0, so a zero-holdings denominator never reads as positive."""
    values = np.array([0.1, 0.2, 1234567.891, 0.3, 77.7])
    start = np.array([0, 1, 2, 3, 3])
    stop = np.array([5, 4, 6, 5, 4])
    out = _accumulate_intervals(values, start, stop, np.zeros(5, dtype=int), (9, 1))[:, 0]
    assert out[6:].tolist() == [0.0, 0.0]
    assert out[5] == pytest.approx(1234567.891)
    assert out[3] == pytest.approx(values.sum())
    print("sanity: interval sums return to exactly 0 after the last expiry; no float residue reaches the denominator test")
