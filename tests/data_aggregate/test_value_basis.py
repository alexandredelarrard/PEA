"""Known-truth tests for the per-filing 13F value-unit repair.

Parsing math on a constructed fixture, per the testing rule: the truth here is arithmetic (a
filing is in dollars or in thousands and the market price says which), not an economic claim,
so a synthetic info table with a known answer is the right instrument.

The worked example is real: CIK 0000861177, MSFT, 2021-12-31 -- 42,212,390 shares against a
reported value of $14,196,871,048,000, an implied price of $336,320 where the actual close was
about $336.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_aggregate.utils.institutionals.value_basis import (
    ABSTAINED,
    DIVIDED,
    KEPT,
    MULTIPLIED,
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
    return pd.DataFrame({"MSFT": price, "AAPL": price}, index=idx)


def _filing(cik: str, value_per_share: float, *, shares: float = 42_212_390.0, ticker: str = "MSFT", rows: int = 3) -> pd.DataFrame:
    """One filing, `rows` holdings, each implying `value_per_share`."""
    return pd.DataFrame(
        {
            "cik": cik,
            "period": PERIOD,
            "ticker": ticker,
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
        f"  ratio {float(register['median_ratio'].iloc[0]):.2e} -> factor 1000, "
        f"value_usd restored to {repaired['value_usd'].iloc[0]:.3e}. Validated."
    )


def test_a_filing_at_a_whole_integer_ratio_is_kept_as_filed():
    """50 is a whole integer, which is the spinoff adjustment `close_split` carries and a 13F
    does not -- a real gap, not a units error, so the value is kept as filed rather than nulled."""
    h = _filing("0000861177", CLOSE * 50.0)
    repaired, register = repair_value_basis(h, _close_split())

    assert (repaired["value_basis_repaired"] == KEPT).all()
    assert repaired["value_usd"].astype(float).equals(h["value_usd"].astype(float))
    assert repaired["call_value"].notna().all()
    # shares are untouched: a value-unit error says nothing about the share count, and the
    # share-based features must keep this filer
    assert repaired["shares"].equals(h["shares"])
    assert (register["factor"] == 1.0).all()

    print("\n=== SANITY CHECK: whole-integer ratio kept as filed ===")
    print(
        f"  ratio {float(register['median_ratio'].iloc[0]):.1f} is a whole integer -> "
        f"factor 1.0, flag KEPT, value_usd left at {repaired['value_usd'].iloc[0]:.3e}, "
        f"shares intact at {repaired['shares'].iloc[0]:,.0f}. Validated."
    )


def test_a_filing_at_a_non_integer_out_of_band_ratio_abstains():
    """2.5 is neither a power of 1000 nor a whole integer, so there is nothing to infer -- the
    row is flagged and nulled, never dropped."""
    h = _filing("0000861177", CLOSE * 2.5)
    repaired, register = repair_value_basis(h, _close_split())

    assert (repaired["value_basis_repaired"] == ABSTAINED).all()
    assert repaired["value_usd"].isna().all()
    assert repaired["call_value"].isna().all()
    assert not (repaired["value_usd"].fillna(-1) == 0).any()
    assert repaired["shares"].equals(h["shares"])
    assert register["factor"].isna().all()

    print("\n=== SANITY CHECK: non-integer out-of-band filing abstains ===")
    print(
        f"  ratio {float(register['median_ratio'].iloc[0]):.2f} is in no band and not a whole "
        f"integer -> factor NA, flag ABSTAINED, value_usd NaN (not 0.0), shares intact at "
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

    print("\n=== SANITY CHECK: median resists one wild row ===")
    print(
        f"  1 of 5 rows given a share count of 1 (implied price "
        f"${h['value_usd'].iloc[0]:,.0f}). Median ratio still "
        f"{float(register['median_ratio'].iloc[0]):.4f} -> KEPT. A row-wise rule would have "
        f"rescaled it. Validated."
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
    assert register.empty

    print("\n=== SANITY CHECK: no close_split degrades ===")
    print(
        "  close_split=None -> every row KEPT and value left as filed, register empty. "
        "The builder degrades rather than nulling the table. Validated."
    )
