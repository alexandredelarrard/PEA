"""Known-truth tests for the 13F book cleaning redesign: line-level repair, consensus prices, book acceptance by coverage and
unproven absence. Each case is a constructed filing whose right answer is arithmetic (a price, a share count, a cover total)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.sec_thirteen import superinvestor_metrics as metrics
from src.data_aggregate.utils.sec_thirteen.holdings_clean import clean_filings
from src.data_aggregate.utils.sec_thirteen.value_basis import BOOK_CEILING_USD, consensus_prices, repair_value_basis, unrealistic_lines
from tests.data_aggregate.test_superinvestor_features import _book, _compute, _snapshot, _value
from tests.data_aggregate.test_value_basis import CLOSE, PERIOD, _close_split, _filing


def _with_cover(h: pd.DataFrame, cover_raw: float) -> pd.DataFrame:
    """Raw filed lines in the stored unit plus a cover total in the same unit (the declared total's own unit is aligned by a power of ten)."""
    return h.assign(raw_value_usd=h["value_usd"], declared_value_total=cover_raw)


def test_a_swapped_line_is_repaired_from_the_cover_total_and_never_becomes_billions() -> None:
    """META-like line: the filer put dollars in the shares column, so value ~ shares. Rebuilding value = shares x quote would give $23bn
    inside a book whose own cover total is $60M; the other leg (shares = value / quote) is the only plausible one."""
    h = _filing("0000861177", CLOSE, shares=100.0, rows=15)
    swapped_shares = 69_650_056.0
    h.loc[14, ["shares", "value_usd"]] = [swapped_shares, swapped_shares - 31.0]  # BDX-style: dollars rounded to thousands, off by a few units
    h = _with_cover(h, float(h["value_usd"].sum()))
    got, register = repair_value_basis(h, _close_split())
    line = got.loc[14]
    assert line["value_usd"] == pytest.approx(swapped_shares - 31.0)
    assert line["shares"] == pytest.approx((swapped_shares - 31.0) / CLOSE)
    assert line["value_basis_shares_reconstructed"] and not line["value_basis_reconstructed"]
    assert got["value_usd"].max() < 1e8
    assert bool(register["book_ok"].iloc[0])
    print(f"SANITY: swapped line keeps ${line['value_usd']:,.0f} and gets {line['shares']:,.0f} shares; no position above $1e8 in a $70M-class book.")


def test_a_line_outside_the_band_takes_the_quote_when_the_value_is_plausible() -> None:
    h = _filing("0000861177", CLOSE, rows=15)
    h.loc[3, "value_usd"] = h.loc[3, "shares"] * CLOSE * 6.0
    h = _with_cover(h, float(h["value_usd"].sum()) * 1.0)
    got, _ = repair_value_basis(h, _close_split())
    assert got.loc[3, "value_usd"] == pytest.approx(h.loc[3, "shares"] * CLOSE)
    assert got.loc[3, "shares"] == h.loc[3, "shares"]
    print("SANITY: a line 6x its quote is rebuilt as shares x price_pit; shares stay as filed.")


def test_a_missing_value_line_never_voids_the_book() -> None:
    h = _filing("0000861177", CLOSE, rows=15)
    h.loc[[1, 2], "value_usd"] = np.nan
    h = h.assign(ticker=["MSFT", "NOQ1", "NOQ2"] + list(h["ticker"].iloc[3:]))  # two unpriced lines lose their value for good
    got, register = repair_value_basis(h, _close_split())
    assert got["value_usd"].isna().sum() == 2
    assert register["kept_rows"].iloc[0] == 13 and register["row_coverage"].iloc[0] == pytest.approx(13 / 15)
    assert bool(register["book_ok"].iloc[0])
    print("SANITY: 2 of 15 lines unknown (87% kept): the book is accepted with its 13 known lines.")


def test_a_book_below_the_coverage_floor_is_rejected_for_weights_only() -> None:
    h = _filing("0000861177", CLOSE, rows=15)
    h.loc[:5, "value_usd"] = np.nan
    h = h.assign(ticker=[f"NOQ{i}" for i in range(6)] + list(h["ticker"].iloc[6:]))
    _, register = repair_value_basis(h, _close_split())
    assert register["row_coverage"].iloc[0] == pytest.approx(9 / 15)
    assert not bool(register["book_ok"].iloc[0])
    print("SANITY: 9 of 15 lines known (60% < 80%): the book is flagged not valued; its shares still count for holders.")


def test_consensus_price_prices_a_security_the_price_table_does_not_cover() -> None:
    def book(cik: str, value_per_share: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cik": cik,
                "period": PERIOD,
                "ticker": None,
                "cusip": ["OFFUNIV01", "OFFUNIV02"],
                "position_type": "common",
                "shares": [100.0, 200.0],
                "value_usd": [100.0 * value_per_share, 200.0 * value_per_share],
            }
        )

    cleaned = pd.concat([book(f"{i:010d}", 50.0) for i in range(3)], ignore_index=True)
    price = consensus_prices(cleaned, 3)
    assert price.loc[(PERIOD, "OFFUNIV01")] == 50.0
    assert consensus_prices(cleaned, 4).empty, "three filers cannot satisfy a four-filer floor"

    wrong = book("0000000009", 50.0 * 1000)
    got, _ = repair_value_basis(wrong, _close_split(), consensus=price)
    assert got["value_usd"].tolist() == [100.0 * 50.0, 200.0 * 50.0]
    assert got["value_basis_reference"].eq("consensus").all()
    print("SANITY: three filers at $50 give a $50 consensus; a fourth filer 1000x high is rebuilt from it, tagged consensus.")


def test_unit_is_voted_against_consensus_for_a_book_with_no_priced_lines() -> None:
    rows = 6
    cusips = [f"OFFUNIV{i:02d}" for i in range(rows)]
    consensus = pd.Series(50.0, index=pd.MultiIndex.from_product([[PERIOD], cusips], names=["period", "cusip"]))
    h = pd.DataFrame(
        {
            "cik": "0000000001",
            "period": PERIOD,
            "ticker": None,
            "cusip": cusips,
            "position_type": "common",
            "shares": 100.0,
            "value_usd": 50.0 * 100.0 * 1000.0,
        }
    )
    got, register = repair_value_basis(h, None, consensus=consensus)
    assert register["factor"].iloc[0] == 0.001 and register["status"].iloc[0] == "consensus_checked"
    assert got["value_usd"].eq(5000.0).all()
    print("SANITY: six unpriced securities all 1000x their consensus move the whole book by 1/1000.")


def test_a_book_that_cannot_prove_absence_lists_holdings_but_leaves_the_rest_unknown() -> None:
    """A confidential or combination filing is no longer dropped: what it lists is held, what it omits is unknown, never a known zero."""
    complete = _book("1", "2024-06-30", {"AAA": (10.0, 100.0), "BBB": (10.0, 100.0)})
    unproven = _book("2", "2024-06-30", {"AAA": (10.0, 100.0)})
    unproven = replace(unproven, support=replace(unproven.support, zero_eligible=False, comparison_eligible=False))
    history = _compute(_snapshot([complete, unproven]))
    assert _value(history, "holders", "AAA") == 2
    assert _value(history, "holders", "BBB") == 1, "the unproven book's omission of BBB is not a known non-holder"
    only = _compute(_snapshot([unproven]))
    assert np.isnan(_value(only, "holders", "BBB")), "with only the unproven book, BBB is unknown rather than zero"
    print("SANITY: a combination/confidential book counts AAA as held and leaves BBB unknown; a complete book still proves BBB's absence.")


def test_book_total_uses_the_known_lines_when_enough_are_valued() -> None:
    universe = pd.Index(["AAA", "BBB", "CCC", "DDD", "EEE"], dtype=object)
    full = _book("1", "2024-06-30", {t: (10.0, 100.0) for t in universe})
    holey = _book("1", "2024-09-30", {t: (10.0, 100.0) for t in universe})
    holey.df_positions.loc[0, "value_usd"] = np.nan
    thin = _book("1", "2024-12-31", {t: (10.0, 100.0) for t in universe})
    thin.df_positions.loc[:1, "value_usd"] = np.nan
    table = metrics.summarize_books([full, holey, thin], universe, 0.8)
    assert table.total.tolist()[:2] == [500.0, 400.0]
    assert np.isnan(table.total[2])
    assert table.coverage.tolist() == pytest.approx([1.0, 0.8, 0.6])
    print("SANITY: 4 of 5 valued lines (80%) keep a $400 total; 3 of 5 (60%) leave the total unknown.")


def test_one_ceiling_rule_excludes_unrealistic_lines_and_never_stops_the_build() -> None:
    index = pd.MultiIndex.from_tuples([(PERIOD, "AAA")], names=["period", "ticker"])
    quarter = pd.Series(1e9, index=index)
    base = pd.DataFrame({"cik": "1", "period": PERIOD, "accession_number": "a", "ticker": "AAA", "position_type": "common", "value_usd": [5e8]})
    assert not unrealistic_lines(base, {"mcap": quarter}).any()
    assert unrealistic_lines(base.assign(value_usd=2e9), {"mcap": quarter}).all(), "above 110% of the issuer's cap"
    assert unrealistic_lines(base.assign(ticker="ZZZ", value_usd=1e22), {"mcap": quarter}).all(), "an unpriced 1e22 line"
    huge = pd.concat([base.assign(ticker=f"T{i}", value_usd=BOOK_CEILING_USD * 0.4) for i in range(4)], ignore_index=True)
    gone = unrealistic_lines(huge, None)
    assert gone.sum() == 2 and huge.loc[~gone, "value_usd"].sum() <= BOOK_CEILING_USD, "a $19tn book loses lines until it fits"
    print("SANITY: a line above 110% of its cap or $5tn, and the lines that keep a book above $12tn, are excluded; nothing raises.")


def test_filings_with_an_impossible_period_and_all_bond_lines_leave_the_book() -> None:
    filings = pd.DataFrame(
        {
            "accession_number": ["ok", "typo", "early"],
            "period": ["2024-06-30", "3006-09-30", "1987-03-31"],
            "filing_date": ["2024-08-01"] * 3,
            "parsed_row_count": [3, 1, 1],
        }
    )
    holdings = pd.DataFrame(
        {
            "accession_number": ["ok", "ok", "ok", "typo", "early"],
            "period": ["2024-06-30"] * 3 + ["3006-09-30", "1987-03-31"],
            "filing_date": ["2024-08-01"] * 5,
            "cusip": list("abcde"),
            "position_type": ["common", "debt", "common", "common", "common"],
            "debt_prn": [0.0, 5e6, 7.0, 0.0, 0.0],
            "debt_value": [0.0, 5e6, 7.0, 0.0, 0.0],
        }
    )
    got_filings, got_holdings = clean_filings(filings, holdings)
    assert got_filings["accession_number"].tolist() == ["ok"] and got_filings["parsed_row_count"].tolist() == [2]
    assert got_holdings["cusip"].tolist() == ["a", "c"] and got_holdings["debt_prn"].eq(0.0).all() and len(got_holdings) == 2
    print("SANITY: periods 3006 and 1987 drop their filings, the bond line leaves, the surviving manifest counts 2 stored rows and bond legs read 0.")
