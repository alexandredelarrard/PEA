"""Known-truth purchase_pct_prior: window net trades over insider holdings `days` calendar days earlier."""

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.quotes import terminal_basis_factor
from src.data_aggregate.utils.institutionals.insider_positions import cumulative_positions, position_units
from src.data_aggregate.utils.institutionals.insider_quality import clean_transactions
from tests.data_aggregate.insider_pit_reference import _NO_SPLIT_FIXTURE, basis_frame, row, source_frame

IDX = pd.date_range("2022-09-01", "2024-04-30")


def _ratio(rows: list[dict], disputes: pd.DataFrame | None = None, split: str | None = None, idx: pd.DatetimeIndex = IDX) -> pd.Series:
    df_clean, diagnostics = clean_transactions(source_frame(rows))
    if split is None:
        factor = terminal_basis_factor(basis_frame(_NO_SPLIT_FIXTURE, df_clean["ticker"], idx.max()))
    else:
        df_basis = pd.DataFrame({"AAA": np.where(idx < pd.Timestamp(split), 1.0, 10.0)}, index=idx)
        factor = terminal_basis_factor(df_basis)
    df_trades = position_units(df_clean, factor)
    df_flows = diagnostics["other_flows"]
    df_flows = position_units(df_flows.assign(transaction_day=df_flows["transaction_day"].fillna(df_flows["day"])), factor)
    none = pd.DataFrame(columns=["ticker", "owner_cik"]) if disputes is None else disputes
    return cumulative_positions(df_trades, df_flows, idx, 60, none)["AAA"]


def _cast() -> list[dict]:
    holder = row(
        accession_number="h",
        owner_cik="3",
        transaction_code="F",
        shares=10,
        shares_owned_after=5000,
        filing_date="2022-10-01",
        transaction_date="2022-09-29",
    )
    seller = row(
        accession_number="s",
        owner_cik="2",
        transaction_code="S",
        shares=100,
        shares_owned_after=900,
        filing_date="2022-10-02",
        transaction_date="2022-09-30",
    )
    buyer = row(accession_number="b", owner_cik="1", shares=1000, shares_owned_after=11000)
    return [holder, seller, buyer, _buyer_prior()]


def _buyer_prior(filing_date: str = "2022-09-28") -> dict:
    """The buyer's direct holdings snapshot of 100 shares, public long before the purchase."""
    return row(
        accession_number="bp",
        owner_cik="1",
        security_type="nonderiv_holding",
        transaction_code=None,
        shares=None,
        price_per_share=None,
        shares_owned_after=100,
        filing_date=filing_date,
        transaction_date=None,
    )


def test_buyer_seller_and_non_trader_holder_share_the_lagged_denominator() -> None:
    values = _ratio(_cast())
    # Holdings 60 days before 2024-02-15 (2023-12-17); more than a year of rows exists: non-trader 5,000 + seller 900 + buyer's prior 100; the buyer's own 11,000 is not yet public.
    assert abs(values.loc["2024-02-15"] - 1000 / 6000) < 1e-12
    print(
        f"SANITY: window net +1000 over holdings 5000+900+100 sixty days earlier = {values.loc['2024-02-15']:.6f} (hand 1000/6000 = {1000 / 6000:.6f})."
    )


def test_shift_uses_holdings_before_the_window_not_after_the_trade() -> None:
    values = _ratio(_cast())
    # 2024-03-12: the buyer's balance (11,000) became public on 2024-01-11, so holdings 60 days earlier (2024-01-12) include it,
    # but the trade has left the 60-day window: no active trade -> NaN.
    assert pd.isna(values.loc["2024-03-12"])
    print("SANITY: once the purchase leaves the 60-day window the cell is NaN, and the denominator is the snapshot 60 days before the cell.")


def test_latest_row_may_be_a_withholding_row() -> None:
    rows = _cast()
    rows.append(
        row(
            accession_number="f",
            owner_cik="3",
            transaction_code="F",
            shares=20,
            shares_owned_after=4000,
            filing_date="2022-11-20",
            transaction_date="2022-11-18",
        )
    )
    values = _ratio(rows)
    assert abs(values.loc["2024-02-15"] - 1000 / 5000) < 1e-12
    print(f"SANITY: the holder's later F row resets its balance to 4000: {values.loc['2024-02-15']:.6f} = 1000/(4000+900+100).")


def test_holdings_snapshot_row_updates_the_balance() -> None:
    rows = _cast()
    rows.append(
        row(
            accession_number="snap",
            owner_cik="3",
            security_type="nonderiv_holding",
            transaction_code=None,
            shares=None,
            price_per_share=None,
            shares_owned_after=7000,
            filing_date="2022-11-20",
            transaction_date=None,
        )
    )
    values = _ratio(rows)
    assert abs(values.loc["2024-02-15"] - 1000 / 8000) < 1e-12
    print(f"SANITY: a direct holdings snapshot sets the balance to 7000: {values.loc['2024-02-15']:.6f} = 1000/(7000+900+100).")


def test_split_inside_the_window_keeps_one_unit_basis() -> None:
    values = _ratio(_cast(), split="2024-01-20")
    # All rows pre-date the split and are put in terminal (post-split) units x10: the ratio is unchanged.
    assert abs(values.loc["2024-02-15"] - 1000 / 6000) < 1e-12
    print(f"SANITY: a 10:1 split after every row rescales numerator and holdings alike: {values.loc['2024-02-15']:.6f}.")


def test_disputed_account_is_left_out_of_every_date_numerator_and_holdings() -> None:
    dispute = pd.DataFrame({"ticker": ["AAA"], "owner_cik": ["0000000002"]})
    values = _ratio(_cast(), disputes=dispute)
    assert abs(values.loc["2024-02-15"] - 1000 / 5100) < 1e-12
    print(f"SANITY: the disputed seller leaves the holdings (5000+100): {values.loc['2024-02-15']:.6f} = 1000/5100.")


def test_no_known_holdings_sixty_days_earlier_is_nan() -> None:
    values = _ratio([row(accession_number="b", owner_cik="1", shares=1000, shares_owned_after=11000)])
    # Holdings 60 days before 2024-01-12 do not exist yet: unknown, never an infinite ratio.
    assert values.loc["2024-01-12":"2024-03-09"].isna().all()
    print("SANITY: a ticker whose only balance is the buyer's own post-trade balance has no holdings 60 days earlier -> NaN.")


def test_holdings_are_unknown_before_a_year_of_balance_rows() -> None:
    holder = row(
        accession_number="h",
        owner_cik="3",
        transaction_code="F",
        shares=10,
        shares_owned_after=5000,
        filing_date="2023-10-01",
        transaction_date="2023-09-29",
    )
    values = _ratio([holder, _cast()[2], _buyer_prior("2023-10-01")])
    # 60 days before 2024-02-15 is 2023-12-17, under 365 days after the ticker's first balance row (2023-10-02): unknown.
    assert pd.isna(values.loc["2024-02-15"])
    print("SANITY: holdings need 365 days of balance rows on the ticker; with 77 days the ratio is NaN, not 1000/5000.")


def test_account_first_seen_in_the_window_is_left_out_of_the_numerator() -> None:
    newcomer_sale = row(
        accession_number="n",
        owner_cik="4",
        transaction_code="S",
        shares=3000,
        shares_owned_after=500,
        filing_date="2024-02-10",
        transaction_date="2024-02-08",
    )
    base = _ratio(_cast())
    values = _ratio([*_cast(), newcomer_sale])
    # Owner 4 has no balance 60 days earlier: its 3000-share sale is neither in the numerator nor (its 500) in the holdings.
    assert abs(values.loc["2024-02-15"] - 1000 / 6000) < 1e-12
    assert values.loc["2024-02-11":"2024-03-01"].dropna().ge(-1.0).all()
    sale_only = _ratio([_cast()[0], _cast()[1], _buyer_prior(), newcomer_sale])
    assert sale_only.loc["2024-02-11":"2024-02-20"].isna().all()
    print(
        f"SANITY: a newcomer's 3000-share sale changes nothing ({values.loc['2024-02-15']:.6f} = base {base.loc['2024-02-15']:.6f}); "
        f"min over the window {values.min():.4f} >= -1; with only that sale the ratio is NaN."
    )


def test_a_share_quantity_above_the_bound_is_excluded_not_wrapped() -> None:
    df_clean, diagnostics = clean_transactions(source_frame(_cast()))
    factor = terminal_basis_factor(basis_frame(_NO_SPLIT_FIXTURE, df_clean["ticker"], IDX.max()))
    df_trades = position_units(df_clean, factor)
    df_flows = diagnostics["other_flows"]
    df_flows = position_units(df_flows.assign(transaction_day=df_flows["transaction_day"].fillna(df_flows["day"])), factor)
    none = pd.DataFrame(columns=["ticker", "owner_cik"])
    seller = df_trades["owner_cik"].eq("0000000002")
    df_trades.loc[seller, "shares_owned_after"] = 2e13  # a filer's typo: int64 micro-shares wrap above ~9.2e12
    value = cumulative_positions(df_trades, df_flows, IDX, 60, none)["AAA"].loc["2024-02-15"]
    assert abs(value - 1000 / 5100) < 1e-12
    print(f"SANITY: a 2e13-share balance typo is excluded, not wrapped: the seller leaves the holdings and the ratio reads {value:.6f} = 1000/5100.")
