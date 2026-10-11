"""Known-truth tests for the per-filing 13F value-unit repair.

Parsing math on a constructed fixture, per the testing rule: the truth here is arithmetic (a
filing is in dollars or in thousands and the market price says which), not an economic claim,
so a synthetic info table with a known answer is the right instrument.

The worked example is real: CIK 0000861177, MSFT, 2021-12-31 -- 42,212,390 shares against a
reported value of $14,196,871,048,000, an implied price of $336,320 where the actual close was
about $336.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.data_aggregate.utils.institutionals.value_basis import (
    ABSTAINED,
    DIVIDED,
    KEPT,
    MULTIPLIED,
    _period_close,
    as_traded_close,
    log_register,
    nullable_value_to_daily,
    repair_value_basis,
)

PERIOD = pd.Timestamp("2021-12-31")
CLOSE = 336.0


def _close_split(price: float = CLOSE) -> pd.DataFrame:
    """A wide (date x ticker) close frame straddling the period end.

    The period end 2021-12-31 IS in the index here, but the as-of lookup is what the code
    relies on: most quarter ends are not trading days.
    """
    idx = pd.date_range("2021-12-28", "2022-01-05", freq="D")
    return pd.DataFrame(
        {
            ticker: price
            for ticker in ("MSFT", "AAPL", "GOOG", "AMZN", "NVDA", "JPM", "BAC", "DIS", "GE", "META", "TMUS", "IBM", "ORCL", "CVX", "XOM")
        },
        index=idx,
    )


def _filing(cik: str, value_per_share: float, *, shares: float = 42_212_390.0, ticker: str = "MSFT", rows: int = 15) -> pd.DataFrame:
    """One filing, `rows` holdings, each implying `value_per_share`."""
    return pd.DataFrame(
        {
            "cik": cik,
            "period": PERIOD,
            "ticker": [
                ("MSFT", "AAPL", "GOOG", "AMZN", "NVDA", "JPM", "BAC", "DIS", "GE", "META", "TMUS", "IBM", "ORCL", "CVX", "XOM")[i % 15]
                for i in range(rows)
            ],
            "shares": [shares] * rows,
            "value_usd": [shares * value_per_share] * rows,
            "call_value": [1_000.0] * rows,
            "put_value": [500.0] * rows,
        }
    )


def test_a_filing_already_in_dollars_is_untouched_to_the_last_bit():
    """The 95.3% correct population must survive the repair BIT-IDENTICAL. Asserted directly,
    because a repair that quietly perturbs the majority is worse than no repair."""
    h = _filing("0000861177", CLOSE)
    repaired, register = repair_value_basis(h, _close_split())

    assert (repaired["value_basis_repaired"] == KEPT).all()
    for col in ("value_usd", "call_value", "put_value"):
        assert repaired[col].to_numpy().tobytes() == h[col].to_numpy().tobytes(), col
    assert float(register["median_ratio"].iloc[0]) == pytest.approx(1.0)

    print("\n=== SANITY CHECK: dollars filing untouched ===")
    print(
        f"  implied ${CLOSE:.2f} vs close ${CLOSE:.2f} -> ratio "
        f"{float(register['median_ratio'].iloc[0]):.4f}, factor "
        f"{register['factor'].iloc[0]}, flag KEPT, value bit-identical. Validated."
    )


def test_a_filing_in_thousands_is_divided_by_1000_on_every_value_leg():
    """The MSFT worked example: implied $336,320 against a $336 close."""
    h = _filing("0000861177", CLOSE * 1000.0)
    repaired, register = repair_value_basis(h, _close_split())

    assert (repaired["value_basis_repaired"] == DIVIDED).all()
    assert repaired["value_usd"].iloc[0] == pytest.approx(42_212_390.0 * CLOSE)
    # the unit is declared once for the whole info table, so the option legs move too
    assert repaired["call_value"].iloc[0] == pytest.approx(1.0)
    assert repaired["put_value"].iloc[0] == pytest.approx(0.5)

    print("\n=== SANITY CHECK: thousands filing divided by 1000 ===")
    print(
        f"  implied ${CLOSE * 1000:,.0f} vs close ${CLOSE:.2f} -> ratio "
        f"{float(register['median_ratio'].iloc[0]):,.0f}, factor 0.001. value_usd "
        f"{h['value_usd'].iloc[0]:.3e} -> {repaired['value_usd'].iloc[0]:.3e}; call_value "
        f"1000.0 -> {repaired['call_value'].iloc[0]}. All value legs moved. Validated."
    )


def test_a_filing_1000x_low_is_multiplied_by_1000():
    h = _filing("0000861177", CLOSE / 1000.0)
    repaired, register = repair_value_basis(h, _close_split())

    assert (repaired["value_basis_repaired"] == MULTIPLIED).all()
    assert repaired["value_usd"].iloc[0] == pytest.approx(42_212_390.0 * CLOSE)

    print("\n=== SANITY CHECK: 1000x-low filing multiplied by 1000 ===")
    print(
        f"  ratio {float(register['median_ratio'].iloc[0]):.2e} -> factor 1000, value_usd restored to {repaired['value_usd'].iloc[0]:.3e}. Validated."
    )


def test_integer_ratio_does_not_invent_a_split() -> None:
    h = _filing("0000861177", CLOSE * 50.0)
    repaired, register = repair_value_basis(h, _close_split())
    assert repaired["value_basis_repaired"].eq(ABSTAINED).all()
    assert repaired["value_usd"].eq(h["shares"] * CLOSE).all()
    assert repaired["shares"].equals(h["shares"])
    assert register["factor"].isna().all()
    print("SANITY: ratio 50 against an as-traded quote is unsupported; no imaginary split rescues it.")


def test_a_filing_at_a_non_integer_out_of_band_ratio_abstains():
    """2.5 is neither a power of 1000 nor a whole integer, so there is nothing to infer -- the
    unit factor stays unresolved; common dollars are separately market-estimated."""
    h = _filing("0000861177", CLOSE * 2.5)
    repaired, register = repair_value_basis(h, _close_split())

    assert (repaired["value_basis_repaired"] == ABSTAINED).all()
    assert repaired["value_usd"].eq(h["shares"] * CLOSE).all()
    assert repaired["call_value"].isna().all()
    assert not (repaired["value_usd"].fillna(-1) == 0).any()
    assert repaired["shares"].equals(h["shares"])
    assert register["factor"].isna().all()

    print("\n=== SANITY CHECK: non-integer out-of-band filing abstains ===")
    print(
        f"  ratio {float(register['median_ratio'].iloc[0]):.2f} is in no band and not a whole "
        f"integer -> factor NA, flag ABSTAINED, common dollars market-estimated, shares intact at "
        f"{repaired['shares'].iloc[0]:,.0f}. Validated."
    )


def test_the_decision_is_per_filing_not_per_table():
    """Three filings on one period, one of each kind. A table-wide rescale is wrong on the
    majority; only a per-filing decision gets all three right."""
    h = pd.concat([_filing("0000000001", CLOSE), _filing("0000000002", CLOSE * 1000.0), _filing("0000000003", CLOSE / 1000.0)], ignore_index=True)
    repaired, register = repair_value_basis(h, _close_split())

    got = repaired.groupby("cik")["value_basis_repaired"].first().to_dict()
    assert got == {"0000000001": KEPT, "0000000002": DIVIDED, "0000000003": MULTIPLIED}
    # every filing now agrees with the market
    assert repaired["value_usd"].round(2).nunique() == 1
    assert len(register) == 3

    print("\n=== SANITY CHECK: the decision is per filing ===")
    print(
        f"  three filings, one table, one period: {got}. After repair all three report the "
        f"same ${repaired['value_usd'].iloc[0]:.3e}, so they collapse onto one basis. "
        f"Validated."
    )


def test_one_wild_row_cannot_flip_a_filing_because_the_statistic_is_the_median():
    """A single mis-reported share count moves one row's implied price arbitrarily far. The
    median over the filing's slice is what makes that harmless."""
    h = _filing("0000861177", CLOSE, rows=5)
    h.loc[0, "shares"] = 1.0  # implied price explodes on this row alone

    repaired, register = repair_value_basis(h, _close_split())
    assert (repaired["value_basis_repaired"] == KEPT).all()
    assert repaired["value_usd"].iloc[0] == CLOSE
    assert repaired["shares"].iloc[0] == 1.0
    assert repaired["value_basis_price_mismatch"].iloc[0]

    print("\n=== SANITY CHECK: median resists one wild row ===")
    print(
        f"  1 of 5 rows given a share count of 1 (implied price "
        f"${h['value_usd'].iloc[0]:,.0f}). Median ratio still "
        f"{float(register['median_ratio'].iloc[0]):.4f} -> book factor KEPT; anomalous row uses "
        f"the market-value fallback with its reported one share preserved. Validated."
    )


@pytest.mark.parametrize("factor", [0.001, 1.0, 1000.0])
def test_position_price_outliers_reuse_missing_value_fallback_after_book_scaling(factor: float, caplog: pytest.LogCaptureFixture) -> None:
    h = _filing("0000861177", 100.0 / factor, shares=100.0)
    h.loc[:5, "value_usd"] *= pd.Series([1.11, 0.89, 1.1, 0.9, 1.09, 0.91])
    original = h.copy(deep=True)
    close = _close_split(100.0)
    got, register = repair_value_basis(h, close)
    assert register["factor"].iloc[0] == factor
    assert got.loc[:1, "value_usd"].eq(10_000.0).all()
    pd.testing.assert_series_equal(got.loc[2:, "value_usd"], h.loc[2:, "value_usd"] * factor)
    assert got["value_basis_price_mismatch"].tolist() == [True, True] + [False] * 13
    assert got["value_basis_reconstructed"].equals(got["value_basis_price_mismatch"])
    assert got.loc[:1, "value_basis_value_source"].eq("market_estimated").all()
    assert got.loc[2:, "value_basis_value_source"].eq("filed").all()
    pd.testing.assert_series_equal(got["shares"], h["shares"])
    pd.testing.assert_series_equal(got["call_value"], h["call_value"] * factor)
    assert register["estimated_price_mismatch_rows"].iloc[0] == 2
    with caplog.at_level(logging.INFO):
        log_register(register, float(h["value_usd"].sum()))
    assert "position price fallback" in caplog.text
    assert "reported shares preserved" in caplog.text
    pd.testing.assert_frame_equal(h, original)
    pd.testing.assert_frame_equal(got, repair_value_basis(got, close)[0])
    close.loc[close.index > PERIOD] *= 1000
    pd.testing.assert_frame_equal(got, repair_value_basis(h, close)[0])
    print(
        f"SANITY: factor {factor:g} is frozen before both high/low position fallbacks; inclusive 10% boundaries, shares, option dollars, source and past observations survive."
    )


def test_a_quarter_end_off_the_trading_calendar_still_resolves_a_price():
    """2021-12-31 aside, most quarter ends are weekends. An exact-match price lookup returns
    NaN there, every row abstains, and the repair silently nulls the whole quarter."""
    close = _close_split()
    close = close.drop(index=pd.Timestamp("2021-12-31"))  # period end is now a non-trading day
    repaired, register = repair_value_basis(_filing("0000861177", CLOSE), close)

    assert (repaired["value_basis_repaired"] == KEPT).all(), "the as-of lookup failed and the quarter abstained"

    print("\n=== SANITY CHECK: as-of price lookup ===")
    print(
        f"  2021-12-31 removed from the price index; the repair reads back to 2021-12-30 and "
        f"still scores ratio {float(register['median_ratio'].iloc[0]):.4f} -> KEPT. "
        f"An exact lookup would have nulled the quarter. Validated."
    )


def test_no_close_split_degrades_instead_of_nulling_every_value():
    """D5: absent price data is a degrade, not an error and not a mass nulling."""
    h = _filing("0000861177", CLOSE)
    repaired, register = repair_value_basis(h, None)

    assert (repaired["value_basis_repaired"] == KEPT).all()
    assert repaired["value_usd"].equals(h["value_usd"])
    assert register["status"].eq("source_only_no_price").all()

    print("\n=== SANITY CHECK: no close_split degrades ===")
    print(
        "  close_split=None -> every row KEPT and value left as filed, register source-only. The builder degrades rather than nulling the table. Validated."
    )


def test_unit_changing_factor_requires_distinct_securities() -> None:
    h = _filing("0000861177", CLOSE * 1000.0, rows=20).assign(ticker="MSFT")
    got, register = repair_value_basis(h, _close_split())
    assert got["value_usd"].eq(h["shares"] * CLOSE).all()
    assert got["shares"].equals(h["shares"])
    print("SANITY: duplicate rows of one security cannot authorize a book-wide unit change.")


def test_filing_dates_in_one_quarter_do_not_share_a_unit_factor() -> None:
    early = _filing("0000861177", CLOSE * 1000.0, rows=3).assign(ticker=["MSFT", "AAPL", "GOOG"], filing_date=pd.Timestamp("2022-02-14"))
    late = _filing("0000861177", CLOSE / 1000.0, rows=3).assign(ticker=["MSFT", "AAPL", "GOOG"], filing_date=pd.Timestamp("2022-03-01"))
    close = _close_split().assign(GOOG=CLOSE)
    got, register = repair_value_basis(pd.concat([early, late], ignore_index=True), close, min_securities=3)
    assert got.loc[:2, "value_basis_repaired"].eq(DIVIDED).all()
    assert got.loc[3:, "value_basis_repaired"].eq(MULTIPLIED).all()
    assert len(register) == 2
    print("SANITY: the same manager and quarter can carry separate filing-unit decisions.")


def test_share_value_swap_is_reconstructed_after_scaling() -> None:
    h = _filing("0000861177", CLOSE * 1000.0, rows=5).assign(ticker=["MSFT", "AAPL", "GOOG", "AMZN", "NVDA"])
    h.loc[4, "value_usd"] = h.loc[4, "shares"]
    close = _close_split().assign(GOOG=CLOSE, AMZN=CLOSE, NVDA=CLOSE)
    got, _ = repair_value_basis(h, close, min_securities=3)
    assert got["value_usd"].eq(h["shares"] * CLOSE).all()
    assert got["shares"].equals(h["shares"])
    print("SANITY: a swapped value is excluded from evidence and is not scaled twice.")


def test_mixed_units_are_not_hidden_by_the_median() -> None:
    h = _filing("0000861177", CLOSE, rows=5).assign(ticker=["MSFT", "AAPL", "GOOG", "AMZN", "NVDA"])
    h.loc[3:, "value_usd"] *= 1000.0
    got, _ = repair_value_basis(h, _close_split().assign(GOOG=CLOSE, AMZN=CLOSE, NVDA=CLOSE))
    assert got["value_usd"].eq(h["shares"] * CLOSE).all()
    assert got["shares"].equals(h["shares"])
    print("SANITY: 60% agreement cannot justify a unit factor; common dollars use tagged market estimates.")


def test_repair_is_idempotent_and_input_is_unchanged() -> None:
    h = _filing("0000861177", CLOSE * 1000.0, rows=3).assign(ticker=["MSFT", "AAPL", "GOOG"])
    original = h.copy(deep=True)
    once, _ = repair_value_basis(h, _close_split().assign(GOOG=CLOSE), min_securities=3)
    twice, _ = repair_value_basis(once, _close_split().assign(GOOG=CLOSE), min_securities=3)
    pd.testing.assert_frame_equal(once, twice)
    pd.testing.assert_frame_equal(h, original)
    print("SANITY: input source evidence is untouched and repeat normalization changes nothing.")


@pytest.mark.parametrize(
    "ticker,price,future_ratio,level",
    [
        ("NVDA", 522.20, 40.0, 1.0),
        ("AAPL", 561.02, 28.0, 1.0),
        ("GE", 94.47, 1.0, 1.605093),
    ],
)
def test_as_traded_price_restores_known_split_and_level_bases(ticker: str, price: float, future_ratio: float, level: float) -> None:
    close = pd.DataFrame({ticker: [price / future_ratio / level]}, index=[PERIOD])
    events = pd.DataFrame({"ticker": [ticker], "date": [pd.Timestamp("2022-07-01")], "value": [future_ratio]})
    got = as_traded_close(close, pd.DataFrame(level, index=close.index, columns=close.columns), events)
    assert got is not None
    assert got[ticker].iloc[0] == pytest.approx(price)
    assert close[ticker].iloc[0] == pytest.approx(price / future_ratio / level)
    print(f"SANITY: {ticker} as-traded price restored to {price} without modifying the quote input.")


def test_future_basis_change_does_not_change_the_historical_reference() -> None:
    close = _close_split()
    level = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    no_events = pd.DataFrame(columns=["ticker", "date", "value"])
    before = as_traded_close(close, level, no_events)
    events = pd.DataFrame({"ticker": close.columns, "date": pd.Timestamp("2024-06-10"), "value": 10.0})
    after = as_traded_close(close / 10 / 1.6, level * 1.6, events)
    pd.testing.assert_frame_equal(before, after)
    print("SANITY: adding a future split/spinoff and its equivalent quote adjustment leaves old references unchanged.")


def test_old_quote_does_not_supply_unit_evidence() -> None:
    close = _close_split().loc[: pd.Timestamp("2021-12-29")]
    got, register = repair_value_basis(_filing("0000861177", CLOSE * 1000), close, max_price_age_days=1)
    assert register["status"].eq("source_only_no_price").all()
    assert got["value_basis_repaired"].eq(KEPT).all()
    print("SANITY: a stale price cannot authorize a unit change; its source-only status is explicit.")


def test_unknown_dollar_observation_invalidates_an_older_value() -> None:
    history = pd.DataFrame({"ticker": ["MSFT", "MSFT"], "as_of": pd.to_datetime(["2022-02-14", "2022-05-16"]), "value": [100.0, float("nan")]})
    grid = pd.DatetimeIndex(["2022-02-15", "2022-05-17"])
    got = nullable_value_to_daily(history, "value", grid)
    assert got["MSFT"].iloc[0] == 100
    assert pd.isna(got["MSFT"].iloc[1])
    print("SANITY: a new unknown dollar state does not forward-fill the previous quarter's known value.")


def test_default_unit_override_requires_fifteen_distinct_securities() -> None:
    h = _filing("0000861177", CLOSE * 1000, rows=14)
    got, register = repair_value_basis(h, _close_split())
    assert got["value_usd"].eq(h["shares"] * CLOSE).all()
    assert register["priced_securities"].iloc[0] == 14
    full, register = repair_value_basis(_filing("0000861177", CLOSE * 1000, rows=15), _close_split())
    assert full["value_basis_repaired"].eq(DIVIDED).all()
    print("SANITY: production gate abstains with 14 securities and permits a coherent 15-security unit override.")


def test_unknown_units_use_marked_market_values_without_inventing_a_factor() -> None:
    h = _filing("0000861177", CLOSE * 2.5)
    h.loc[0, "call_value"] = 0.0
    original = h.copy(deep=True)
    got, register = repair_value_basis(h, _close_split())
    assert got["value_usd"].eq(h["shares"] * CLOSE).all()
    assert got["value_basis_repaired"].eq(ABSTAINED).all()
    assert register["factor"].isna().all()
    assert got["value_basis_reconstructed"].all()
    assert got["value_basis_value_source"].eq("market_estimated").all()
    assert got["call_value"].iloc[0] == 0.0
    assert got["call_value"].iloc[1:].isna().all()
    assert register["estimated_rows"].iloc[0] == len(h)
    assert register["estimated_value"].iloc[0] == pytest.approx(float((h["shares"] * CLOSE).sum()))
    pd.testing.assert_frame_equal(h, original)
    twice, _ = repair_value_basis(got, _close_split())
    pd.testing.assert_frame_equal(twice, got)
    print(
        "SANITY: unknown book units retain their unresolved factor but all common values are separately marked market estimates; input/shares unchanged."
    )


def test_missing_values_estimate_only_with_valid_common_quantity_and_period_quote() -> None:
    h = _filing("0000861177", CLOSE).iloc[:6].copy()
    h["value_usd"] = float("nan")
    h["shares"] = [100.0, 100.0, 0.0, float("nan"), float("inf"), 100.0]
    h["position_type"] = ["common"] * 5 + ["debt"]
    close = _close_split().drop(columns=["AAPL"])
    got, _ = repair_value_basis(h, close)
    assert got["value_usd"].iloc[0] == 100 * CLOSE
    assert got["value_basis_value_source"].iloc[0] == "market_estimated"
    assert got["value_usd"].iloc[1:].isna().all()
    assert not got["value_basis_reconstructed"].iloc[1:].any()
    stale, _ = repair_value_basis(h, close.loc[:"2021-12-28"], max_price_age_days=1)
    assert stale["value_usd"].isna().all()
    future, _ = repair_value_basis(h, close.loc["2022-01-01":])
    assert future["value_usd"].isna().all()
    print("SANITY: missing/stale/future prices, zero/missing/infinite shares and non-common instruments cannot create market-estimated holdings.")


def test_market_fallback_is_not_unit_evidence_and_uses_no_future_price() -> None:
    h = _filing("0000861177", CLOSE * 1000, rows=14)
    got, register = repair_value_basis(h, _close_split())
    assert register["priced_securities"].iloc[0] == 14
    assert register["factor"].isna().all()
    assert got["value_usd"].eq(h["shares"] * CLOSE).all()
    close = _close_split()
    close.loc[close.index > PERIOD] *= 100
    future, later_register = repair_value_basis(h, close)
    pd.testing.assert_frame_equal(got, future)
    pd.testing.assert_frame_equal(register, later_register)
    print("SANITY: estimated values neither authorize a unit factor nor borrow a later market quote.")


def test_unresolved_option_values_use_their_own_underlying_quantities() -> None:
    h = _filing("0000861177", CLOSE * 2.5).assign(
        call_shares=3.0,
        put_shares=7.0,
        debt_prn=1000.0,
        debt_value=950.0,
    )
    original = h.copy(deep=True)
    got, register = repair_value_basis(h, _close_split())
    assert got["call_value"].eq(3 * CLOSE).all()
    assert got["put_value"].eq(7 * CLOSE).all()
    assert got["debt_value"].isna().all()
    assert got["value_basis_call_reconstructed"].all()
    assert got["value_basis_put_reconstructed"].all()
    assert got["value_basis_call_value_source"].eq("market_estimated").all()
    assert register["factor"].isna().all()
    assert register["estimated_call_rows"].iloc[0] == len(h)
    assert register["estimated_put_value"].iloc[0] == 7 * CLOSE * len(h)
    pd.testing.assert_frame_equal(h, original)
    pd.testing.assert_frame_equal(
        got[["shares", "call_shares", "put_shares", "debt_prn"]], original[["shares", "call_shares", "put_shares", "debt_prn"]]
    )
    twice, _ = repair_value_basis(got, _close_split())
    pd.testing.assert_frame_equal(got, twice)
    print(
        "SANITY: calls/puts use their own underlying quantities; equity price never prices debt principal; unresolved factors and original quantities survive."
    )


def test_option_estimates_reject_invalid_quantities_and_ineligible_quotes() -> None:
    h = _filing("0000861177", CLOSE, rows=6).assign(
        call_shares=[2.0, 2.0, 0.0, -1.0, float("nan"), float("inf")],
        put_shares=[2.0, 2.0, 0.0, -1.0, float("nan"), float("inf")],
        call_value=float("nan"),
        put_value=float("nan"),
    )
    close = _close_split().drop(columns=["AAPL"])
    got, _ = repair_value_basis(h, close)
    for column in ("call_value", "put_value"):
        assert got[column].iloc[0] == 2 * CLOSE
        assert got[column].iloc[1:].isna().all()
    for prices, kwargs in ((close.loc[:"2021-12-28"], {"max_price_age_days": 1}), (close.loc["2022-01-01":], {})):
        rejected, _ = repair_value_basis(h, prices, **kwargs)
        assert rejected[["call_value", "put_value"]].isna().all().all()
    close.loc[close.index > PERIOD] *= 1000
    later, _ = repair_value_basis(h, close)
    pd.testing.assert_frame_equal(got, later)
    print(
        "SANITY: option fallback rejects zero/negative/missing/infinite quantities and missing/stale/future quotes; later prices cannot revise old estimates."
    )


def test_option_only_position_and_known_book_scaling_remain_distinct() -> None:
    h = _filing("0000861177", CLOSE, rows=1).assign(
        shares=0.0,
        value_usd=0.0,
        call_shares=10.0,
        put_shares=20.0,
        call_value=float("nan"),
        put_value=float("nan"),
        position_type="call",
    )
    got, register = repair_value_basis(h, _close_split())
    assert got["call_value"].iloc[0] == 10 * CLOSE
    assert got["put_value"].iloc[0] == 20 * CLOSE
    assert got["value_usd"].iloc[0] == 0.0
    assert register["priced_securities"].iloc[0] == 0
    assert not got["value_basis_reconstructed"].any()
    unverified = h.assign(call_value=60_000.0, put_value=140_000.0)
    own_quantity, _ = repair_value_basis(unverified, _close_split())
    assert own_quantity["call_value"].iloc[0] == 10 * CLOSE
    assert own_quantity["put_value"].iloc[0] == 20 * CLOSE
    absent, _ = repair_value_basis(unverified.drop(columns=["call_shares", "put_shares"]), _close_split())
    assert absent[["call_value", "put_value"]].isna().all().all()
    zeros, _ = repair_value_basis(unverified.assign(call_value=0.0, put_value=0.0).drop(columns=["call_shares", "put_shares"]), _close_split())
    assert zeros[["call_value", "put_value"]].eq(0).all().all()
    supported = _filing("0000861177", CLOSE * 1000).assign(
        call_shares=3.0,
        put_shares=7.0,
        debt_prn=1000.0,
        call_value=3 * CLOSE * 1000,
        put_value=7 * CLOSE * 1000,
        debt_value=950_000.0,
    )
    scaled, _ = repair_value_basis(supported, _close_split())
    assert scaled["call_value"].eq(3 * CLOSE).all()
    assert scaled["put_value"].eq(7 * CLOSE).all()
    assert scaled["debt_value"].eq(950.0).all()
    assert not scaled[["value_basis_call_reconstructed", "value_basis_put_reconstructed"]].any().any()
    pd.testing.assert_frame_equal(scaled[["call_shares", "put_shares", "debt_prn"]], supported[["call_shares", "put_shares", "debt_prn"]])
    print(
        "SANITY: option-only positions need no common shares; options never vote; a known book factor scales debt/options once without quantity changes."
    )


@pytest.mark.parametrize("filed_scale", [1.0, 1000.0, 0.001])
def test_inverse_common_fallback_uses_normalized_dollars_without_voting(filed_scale: float) -> None:
    book = _filing("0000000001", CLOSE * filed_scale)
    extra = book.iloc[[0]].assign(ticker="ZERO", shares=0.0, value_usd=100 * CLOSE * filed_scale)
    h = pd.concat([book, extra], ignore_index=True)
    original = h.copy(deep=True)
    got, register = repair_value_basis(h, _close_split().assign(ZERO=CLOSE))
    assert got.loc[15, "shares"] == pytest.approx(100.0)
    assert got.loc[15, "value_usd"] == pytest.approx(100 * CLOSE)
    assert got.loc[15, "value_basis_shares_reconstructed"]
    assert got.loc[15, "value_basis_shares_source"] == "value_estimated"
    assert not got.loc[15, "value_basis_inverse_unit_unverified"]
    assert register.loc[0, "priced_securities"] == 15
    assert register.loc[0, "estimated_shares_rows"] == 1
    assert register.loc[0, "factor"] == pytest.approx(1 / filed_scale)
    pd.testing.assert_series_equal(got.loc[:14, "shares"], original.loc[:14, "shares"])
    pd.testing.assert_frame_equal(h, original)
    twice, _ = repair_value_basis(got, _close_split().assign(ZERO=CLOSE))
    pd.testing.assert_frame_equal(got, twice)
    print(
        "SANITY: inverse common fallback divides once-normalized dollars by the historical price; inferred quantity never votes and source is immutable."
    )


def test_inverse_unverified_units_exclusions_and_missing_value_logs(caplog: pytest.LogCaptureFixture) -> None:
    book = _filing("0000000001", CLOSE * 2.5)
    extra = pd.DataFrame(
        {
            "cik": "0000000001",
            "period": PERIOD,
            "ticker": ["ZERO", "NOQUOTE", "MISSING", "MSFT"],
            "shares": [0.0, 0.0, 10.0, 0.0],
            "value_usd": [10 * CLOSE, 10 * CLOSE, float("nan"), 0.0],
            "call_value": [0.0, 0.0, 0.0, 0.0],
            "put_value": 0.0,
        }
    )
    h = pd.concat([book, extra], ignore_index=True)
    got, register = repair_value_basis(h, _close_split().assign(ZERO=CLOSE, MISSING=CLOSE))
    assert got.loc[15, "shares"] == 10
    assert got.loc[15, "value_usd"] == 10 * CLOSE
    assert got.loc[15, "value_basis_inverse_unit_unverified"]
    assert register["factor"].isna().all()
    assert got.loc[16, "value_basis_excluded"]
    assert pd.isna(got.loc[16, "shares"]) and pd.isna(got.loc[16, "value_usd"])
    assert got.loc[17, "value_usd"] == 10 * CLOSE
    assert got.loc[17, "value_basis_reconstructed"]
    assert not got.loc[18, "value_basis_excluded"]
    assert got.loc[18, "shares"] == 0 and got.loc[18, "value_usd"] == 0
    with caplog.at_level(logging.INFO):
        log_register(register, float(h.value_usd.sum()))
    assert "inverse" in caplog.text
    assert "unit unverified" in caplog.text
    assert "excluded" in caplog.text
    assert "missing filed value" in caplog.text
    assert register.loc[0, "excluded_rows"] == 1
    assert register.loc[0, "inverse_unit_unverified_rows"] == 1
    assert register.loc[0, "estimated_missing_value_rows"] == 1
    print(
        "SANITY: unresolved units remain explicit; inverse shares are estimates, missing dollars are reconstructed, invalid records are marked/null and structural zeros survive; run logs identify every path."
    )


def test_common_exclusion_keeps_supported_siblings_and_inverse_provenance() -> None:
    book = _filing("0000000001", CLOSE)
    extra = book.iloc[[0]].assign(
        ticker="NOQUOTE",
        shares=0.0,
        value_usd=100.0,
        call_shares=5.0,
        call_value=100.0,
        debt_prn=500.0,
        debt_value=50.0,
    )
    h = pd.concat([book, extra], ignore_index=True)
    got, register = repair_value_basis(h, _close_split())
    assert got.loc[15, "value_basis_common_excluded"]
    assert not got.loc[15, "value_basis_excluded"]
    assert pd.isna(got.loc[15, "shares"]) and pd.isna(got.loc[15, "value_usd"])
    assert got.loc[15, "call_shares"] == 5 and got.loc[15, "call_value"] == 100
    assert got.loc[15, "debt_prn"] == 500 and got.loc[15, "debt_value"] == 50
    assert register.loc[0, "common_excluded_rows"] == 1
    assert register.loc[0, "excluded_rows"] == 0
    h["call_shares"] = pd.Series([pd.NA] * len(h), dtype="Float64")
    nullable_sibling, _ = repair_value_basis(h, _close_split())
    assert nullable_sibling.loc[15, "call_value"] == 100
    assert not nullable_sibling.loc[15, "value_basis_excluded"]

    one = book.iloc[[0]].assign(ticker="ONE", shares=0.0, value_usd=10.0)
    inverse, _ = repair_value_basis(one, _close_split().assign(ONE=1.0))
    assert inverse["shares"].iloc[0] == 10
    assert inverse["value_basis_shares_reconstructed"].iloc[0]
    assert not inverse["value_basis_reconstructed"].iloc[0]
    assert inverse["value_basis_value_source"].iloc[0] == "filed"

    missing, _ = repair_value_basis(book.drop(columns="value_usd"), _close_split())
    assert missing["value_usd"].eq(book["shares"] * CLOSE).all()
    assert missing["value_basis_reconstructed"].all()
    nullable = one.assign(shares=pd.Series([pd.NA], index=one.index, dtype="Float64"))
    excluded, _ = repair_value_basis(nullable, None)
    assert excluded["value_basis_excluded"].iloc[0]
    assert pd.isna(excluded["shares"].iloc[0])
    unknown = nullable.assign(value_usd=pd.Series([pd.NA], index=nullable.index, dtype="Float64"))
    priced_unknown, _ = repair_value_basis(unknown, _close_split().assign(ONE=1.0))
    assert pd.isna(priced_unknown["value_usd"].iloc[0])
    assert not priced_unknown["value_basis_reconstructed"].iloc[0]
    no_sibling = one.assign(ticker="NOQUOTE", call_value=100.0, put_value=0.0)
    unsupported, _ = repair_value_basis(no_sibling, _close_split())
    assert pd.isna(unsupported["call_value"].iloc[0])
    assert unsupported["value_basis_excluded"].iloc[0]
    print(
        "SANITY: unavailable common legs stay unknown, supported siblings survive, unsupported money cannot defeat exclusion; $1 inverse provenance and absent/nullable inputs are handled."
    )


def test_same_day_accessions_have_independent_book_unit_factors() -> None:
    original = _filing("0000861177", CLOSE).assign(accession_number="original", filing_date=pd.Timestamp("2022-02-14"))
    amendment = _filing("0000861177", CLOSE * 1000).assign(accession_number="amendment", filing_date=pd.Timestamp("2022-02-14"))
    corrected, register = repair_value_basis(pd.concat([original, amendment], ignore_index=True), _close_split())
    factors = register.set_index("accession_number").factor
    assert factors.to_dict() == {"original": 1.0, "amendment": 0.001}
    assert corrected.loc[corrected.accession_number.eq("original"), "call_value"].eq(1000).all()
    assert corrected.loc[corrected.accession_number.eq("amendment"), "call_value"].eq(1).all()
    assert corrected.value_usd.eq(original.value_usd.iloc[0]).all()
    print("SANITY: same manager, quarter and filing day; accession-specific unit factors remain separate before amendment resolution.")


def test_shared_as_traded_quote_restores_split_and_level() -> None:
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    close = pd.DataFrame({"AAA": [25.0, 25.0, 50.0]}, index=idx)
    level = pd.DataFrame(2.0, index=idx, columns=["AAA"])
    splits = pd.DataFrame({"ticker": ["AAA"], "date": [idx[2]], "value": [2.0]})
    quotes = as_traded_close(close, level, splits)
    assert quotes["AAA"].tolist() == [100.0, 100.0, 100.0]
    found = _period_close(quotes, pd.Series(pd.to_datetime(["2019-12-31", "2020-01-04", "2020-01-20"])), 7)
    assert pd.isna(found.loc[(pd.Timestamp("2019-12-31"), "AAA")])
    assert found.loc[(pd.Timestamp("2020-01-04"), "AAA")] == 100
    assert pd.isna(found.loc[(pd.Timestamp("2020-01-20"), "AAA")])
    print("SANITY: shared quote restores only genuine split and level units; no forward or stale quote enters cleaning.")
