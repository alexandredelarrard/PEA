"""Known-truth ic_insider_conviction: the largest account increase, net window shares over the account's own prior balance."""

import logging

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.quotes import terminal_basis_factor
from src.data_aggregate.utils.institutionals.insider_features import CONVICTION_FEATURES, EMISSION, PEER_RELATIVE, InsiderCoverage
from src.data_aggregate.utils.institutionals.insider_positions import account_conviction, position_units
from src.data_aggregate.utils.institutionals.insider_quality import clean_transactions
from tests.conftest import make_frames
from tests.data_aggregate.insider_pit_reference import _NO_SPLIT_FIXTURE, basis_frame, build_prepared_insider_panel, row, source_frame

IDX = pd.date_range("2022-09-01", "2024-04-30")
DAY = pd.Timestamp("2024-02-15")


def _conviction(rows: list[dict], days: int = 60, disputes: pd.DataFrame | None = None, split: str | None = None) -> pd.Series:
    df_clean, diagnostics = clean_transactions(source_frame(rows))
    if split is None:
        factor = terminal_basis_factor(basis_frame(_NO_SPLIT_FIXTURE, df_clean["ticker"], IDX.max()))
    else:
        factor = terminal_basis_factor(pd.DataFrame({"AAA": np.where(IDX < pd.Timestamp(split), 1.0, 10.0)}, index=IDX))
    df_trades = position_units(df_clean, factor)
    df_flows = diagnostics["other_flows"]
    df_flows = position_units(df_flows.assign(transaction_day=df_flows["transaction_day"].fillna(df_flows["day"])), factor)
    none = pd.DataFrame(columns=["ticker", "owner_cik", "start", "stop"]) if disputes is None else disputes
    return account_conviction(df_trades, df_flows, IDX, days, none)["AAA"]


def _prior(owner: str, after: float) -> dict:
    """A withholding row that publishes the account's balance long before the window."""
    return row(
        accession_number=f"prior{owner}",
        owner_cik=owner,
        transaction_code="F",
        shares=1,
        shares_owned_after=after,
        filing_date="2023-06-01",
        transaction_date="2023-05-30",
    )


def _buy(owner: str, shares: float, after: float, filed: str, traded: str) -> dict:
    return row(accession_number=f"b{owner}", owner_cik=owner, shares=shares, shares_owned_after=after, filing_date=filed, transaction_date=traded)


def _cast() -> list[dict]:
    """Owner 1: 10,000 -> +1,000 (0.1); owner 2: 200 -> +100 (0.5); owner 3: 1,000 -> -400 (-0.4)."""
    seller = row(
        accession_number="s3",
        owner_cik="3",
        transaction_code="S",
        shares=400,
        shares_owned_after=600,
        filing_date="2024-01-13",
        transaction_date="2024-01-11",
    )
    return [
        _prior("1", 10_000),
        _prior("2", 200),
        _prior("3", 1_000),
        _buy("1", 1000, 11_000, "2024-01-10", "2024-01-08"),
        _buy("2", 100, 300, "2024-01-12", "2024-01-10"),
        seller,
    ]


def test_max_over_accounts_picks_the_largest_increase() -> None:
    values = _conviction(_cast())
    assert abs(values.loc[DAY] - 0.5) < 1e-12
    print(f"SANITY: accounts +0.10, +0.50 and -0.40 -> the maximum is {values.loc[DAY]:.4f} (hand 100/200 = 0.5), not the pooled 700/11200.")


def test_sell_only_ticker_is_negative() -> None:
    values = _conviction([_prior("3", 1_000), _cast()[5]])
    assert abs(values.loc[DAY] - (-0.4)) < 1e-12
    print(f"SANITY: a sell-only ticker reads {values.loc[DAY]:.4f} (hand -400/1000 = -0.4).")


def test_zero_prior_account_is_excluded_and_counted(caplog) -> None:
    new_account = _buy("9", 5000, 5000, "2024-01-12", "2024-01-10")
    with caplog.at_level(logging.INFO):
        alone = _conviction([new_account])
    assert pd.isna(alone.loc[DAY])
    assert any("left out" in message for message in caplog.messages)
    mixed = _conviction([*_cast(), new_account])
    assert abs(mixed.loc[DAY] - 0.5) < 1e-12
    print(
        f"SANITY: an account opening with its own purchase has no prior: alone -> NaN, beside the others the max stays {mixed.loc[DAY]:.4f}; the log counts it."
    )


def test_window_boundary_and_visibility_day() -> None:
    only = _conviction([_prior("1", 10_000), _cast()[3]])
    visible = pd.Timestamp("2024-01-10")  # the fixture source carries no publication lag: visible_from is the filing day
    last, gone = visible + pd.Timedelta(days=59), visible + pd.Timedelta(days=60)
    assert pd.isna(only.loc[visible - pd.Timedelta(days=1)]) and abs(only.loc[visible] - 0.1) < 1e-12
    assert abs(only.loc[last] - 0.1) < 1e-12 and pd.isna(only.loc[gone])
    print(f"SANITY: a purchase public on 2024-01-10 is invisible on 01-09, counts from 01-10 to {last.date()} and is gone on {gone.date()}.")


def test_prior_is_the_balance_public_before_the_window_not_after() -> None:
    later = row(
        accession_number="late",
        owner_cik="1",
        transaction_code="F",
        shares=1,
        shares_owned_after=50_000,
        filing_date="2024-01-20",
        transaction_date="2024-01-19",
    )
    values = _conviction([_prior("1", 10_000), _cast()[3], later])
    assert abs(values.loc[DAY] - 1000 / 10_000) < 1e-12
    print(f"SANITY: owner 1's later 50,000 balance (public after the window opened) leaves the prior at 10,000: {values.loc[DAY]:.4f}.")


def test_disputed_account_is_left_out() -> None:
    dispute = pd.DataFrame(
        {"ticker": ["AAA"], "owner_cik": ["0000000002"], "start": [pd.Timestamp("2022-10-01")], "stop": [pd.Timestamp("2024-06-01")]}
    )
    values = _conviction(_cast(), disputes=dispute)
    assert abs(values.loc[DAY] - 0.1) < 1e-12
    print(f"SANITY: with owner 2 disputed the maximum falls to owner 1's {values.loc[DAY]:.4f} (hand 1000/10000).")


def test_split_inside_the_window_cancels() -> None:
    values = _conviction(_cast(), split="2024-01-20")
    assert abs(values.loc[DAY] - 0.5) < 1e-12
    print(f"SANITY: a 10:1 split after every row rescales net and prior alike: {values.loc[DAY]:.4f} (hand 0.5).")


def test_longer_window_keeps_trades_longer() -> None:
    day = pd.Timestamp("2024-03-20")
    assert pd.isna(_conviction(_cast(), days=60).loc[day])
    assert abs(_conviction(_cast(), days=120).loc[day] - 0.5) < 1e-12
    print("SANITY: on 2024-03-20 the 60-day window has emptied (NaN) while the 120-day window still reads 0.5.")


def test_quiet_window_is_zero_and_the_unobserved_tail_is_nan() -> None:
    idx = pd.bdate_range("2024-01-01", "2024-09-30")
    close = pd.DataFrame(100.0, index=idx, columns=["AAA"])
    frames = make_frames(idx, {"AAA": {"AAA": 1.0}}, close_split=close)
    rows = [row(accession_number="p", filing_date="2024-01-02", transaction_date="2023-12-29")]
    panel = build_prepared_insider_panel(
        frames,
        pd.DataFrame(rows),
        shares_out_history=pd.DataFrame(1_000.0, index=idx, columns=["AAA"]),
        coverage=InsiderCoverage.from_availability(None, pd.Timestamp("2024-08-31")),
    ).set_index("date")
    name = "f_ic_insider_conviction_60d"
    assert panel.loc[pd.Timestamp("2024-07-31"), name] == 0.0
    assert panel.loc[panel.index > pd.Timestamp("2024-08-31"), name].isna().all()
    print("SANITY: conviction_60d is an observed 0 in a covered quiet window (2024-07-31) and NaN after the table frontier.")


def test_declarations_and_dropped_column() -> None:
    assert set(CONVICTION_FEATURES) == {"ic_insider_conviction_60d", "ic_insider_conviction_120d"}
    assert all(EMISSION[name] == "raw" for name in CONVICTION_FEATURES)
    assert EMISSION[PEER_RELATIVE] == "raw+peers"
    assert "ic_insider_purchase_pct_prior_180d" not in EMISSION
    print("SANITY: two conviction legs are raw, the pooled 120d ratio adds its peer z, the all-insider 180d ratio is gone.")
