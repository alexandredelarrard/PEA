# ruff: noqa: N806
"""Step 2 — Dividend / shareholder-yield features.

Checks: features are point-in-time; non-payers get a real 0 yield; rising
dividends -> positive growth; shareholder_yield adds buyback yield (up for
repurchasers, down for issuers); the built panel exposes the expected f_*
columns. (The dividend extractor's own parser, `_extract_dividends`, is
covered in test_da_capex_and_dividend_consolidation.py.)
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.data_aggregate.utils.fundamentals.dividend_features import (
    _cagr,
    _calendar_prior,
    _dividend_fields,
    _ttm_dividends,
    build_dividend_feature_panel,
)
from src.data_aggregate.utils.fundamentals.inputs import FundamentalHistory
from src.data_aggregate.utils.price.frames import PriceFrames
from src.utils.window import Offset, shift_date
from tests.data_aggregate.fundamental_pit_reference import known_no_share_events, legacy_capital_frames
from tests.support.config import fixture_config

_YOY = 252


def _published_cash(history: pd.DataFrame) -> pd.DataFrame:
    """These math fixtures are already public; consumer buffer checks use the actual step."""
    return history.assign(available_on=history["date"])


def _synth():
    dates = pd.bdate_range("2021-01-04", periods=3 * 252)  # ~3y
    payers = ["P0", "P1", "P2", "P3"]
    nonpayers = ["N0", "N1", "N2", "N3"]
    tickers = payers + nonpayers
    rng = np.random.default_rng(0)
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (len(dates), len(tickers))), axis=0), index=dates, columns=tickers)

    # quarterly dividends; P0 GROWS its payout each year, others flat
    ex_dates = dates[::63]
    rows = []
    for i, d in enumerate(ex_dates):
        year = i // 4
        for p in payers:
            amt = 0.5 * (1.15**year if p == "P0" else 1.0)  # P0 grows 15%/yr
            rows.append({"date": d, "ticker": p, "dividends": amt})
    div_hist = pd.DataFrame(rows)

    # fundamentals: sharesOutstanding — P0 buys back, N0 issues, rest flat
    frows = []
    for q, aso in enumerate(dates[::63]):
        for t in tickers:
            base = 1e9
            shares = base * (0.9 ** (q / 4) if t == "P0" else 1.1 ** (q / 4) if t == "N0" else 1.0)
            frows.append({"ticker": t, "as_of": aso, "sharesOutstanding": shares})
    fund = pd.DataFrame(frows)
    return dates, tickers, close, div_hist, fund


def test_dividend_fields_economics_and_pit():
    dates, tickers, close, div_hist, fund = _synth()
    F = _dividend_fields(
        _published_cash(div_hist), legacy_capital_frames(fund, close), FundamentalHistory(raw_history=fund, share_basis=known_no_share_events())
    )
    for k in ("dividend_yield", "dividend_growth", "dividend_payer", "shareholder_yield"):
        assert k in F, f"missing {k}"

    t = dates[-1]
    # non-payer -> 0 yield, flag 0; payer -> positive yield, flag 1
    assert F["dividend_yield"].loc[t, "N0"] == 0.0
    assert F["dividend_payer"].loc[t, "N0"] == 0.0
    assert F["dividend_yield"].loc[t, "P0"] > 0
    assert F["dividend_payer"].loc[t, "P0"] == 1.0
    # rising payout -> positive growth for P0, ~0 for a flat payer
    assert F["dividend_growth"].loc[t, "P0"] > 0.05
    assert abs(F["dividend_growth"].loc[t, "P1"]) < 0.02
    # shareholder yield: buyback firm > its dividend yield; issuer < its div yield
    assert F["shareholder_yield"].loc[t, "P0"] > F["dividend_yield"].loc[t, "P0"]
    assert F["shareholder_yield"].loc[t, "N0"] < F["dividend_yield"].loc[t, "N0"]

    # point-in-time: perturb everything AFTER t -> value at t unchanged
    div2 = pd.concat([div_hist, pd.DataFrame([{"date": dates[-1], "ticker": "P0", "dividends": 99.0}])], ignore_index=True)
    tm = dates[len(dates) // 2]
    F2 = _dividend_fields(
        _published_cash(div2), legacy_capital_frames(fund, close), FundamentalHistory(raw_history=fund, share_basis=known_no_share_events())
    )
    assert np.isclose(F["dividend_yield"].loc[tm, "P0"], F2["dividend_yield"].loc[tm, "P0"])

    print("\n=== SANITY CHECK: dividend economics + point-in-time ===")
    print(
        f"  P0 yield={F['dividend_yield'].loc[t, 'P0']:.3f} growth={F['dividend_growth'].loc[t, 'P0']:.2f} "
        f"shareholder={F['shareholder_yield'].loc[t, 'P0']:.3f} (>div, buyback); "
        f"N0 yield=0, shareholder<0 (issuer). Future dividend didn't change past value. Validated."
    )


def _synth_reconcile():
    """~6y so the 5y CAGR is defined. Three names:
    A      — pays via BOTH sources (per-share ex-dates AND `dividendsPaid`), grows 10%/yr,
    B_ONLY — pays via source B ONLY (no ex-date history; yfinance missed it), flat,
    N      — never pays."""
    dates = pd.bdate_range("2018-01-01", periods=6 * 252 + 20)
    tickers = ["A", "B_ONLY", "N"]
    close = pd.DataFrame(100.0, index=dates, columns=tickers)
    ex = dates[::63]  # quarterly ex-dates
    rows = [{"date": d, "ticker": "A", "dividends": 1.0 * (1.10 ** (i // 4))} for i, d in enumerate(ex)]  # A grows 10%/yr per share
    div_hist = pd.DataFrame(rows)
    frows = []
    for q, aso in enumerate(dates[::63]):
        ttm = 4.0 * (1.10 ** (q // 4))  # A's TTM per-share ~= 4 * quarterly
        frows += [
            {"ticker": "A", "as_of": aso, "sharesOutstanding": 1000.0, "dividendsPaid": ttm * 1000.0, "netIncome": 8000.0, "freeCashflow": 12000.0},
            {"ticker": "B_ONLY", "as_of": aso, "sharesOutstanding": 1000.0, "dividendsPaid": 2000.0, "netIncome": 8000.0, "freeCashflow": 1000.0},
            {"ticker": "N", "as_of": aso, "sharesOutstanding": 1000.0, "dividendsPaid": 0.0, "netIncome": 8000.0, "freeCashflow": 5000.0},
        ]
    return dates, close, div_hist, pd.DataFrame(frows)


def test_reconcile_sources_5y_growth_payout_coverage():
    dates, close, div_hist, fund = _synth_reconcile()
    F = _dividend_fields(
        _published_cash(div_hist), legacy_capital_frames(fund, close), FundamentalHistory(raw_history=fund, share_basis=known_no_share_events())
    )
    t = dates[-1]
    for k in ("dividend_growth_5y", "dividend_payout_ratio", "dividend_coverage"):
        assert k in F, f"missing {k}"

    # RECONCILIATION: A's per-share (source A) yield agrees with its source-B yield
    # (dividendsPaid/mcap) — same cash two ways.
    src_b_yield_A = 4.0 * (1.10**5) * 1000.0 / (1000.0 * 100.0)  # dividendsPaid / mcap
    assert abs(F["dividend_yield"].loc[t, "A"] - src_b_yield_A) < 0.03
    # RECONCILIATION gap-fill: B_ONLY has NO ex-dates, yield comes from source B = 2000/100000
    assert abs(F["dividend_yield"].loc[t, "B_ONLY"] - 0.02) < 1e-6
    assert F["dividend_payer"].loc[t, "B_ONLY"] == 1.0  # counted as a payer via source B
    assert F["dividend_yield"].loc[t, "N"] == 0.0  # true non-payer -> real 0

    # 5-YEAR GROWTH: A grew per-share ~10%/yr -> 5y CAGR ~= 0.10
    assert abs(F["dividend_growth_5y"].loc[t, "A"] - 0.10) < 0.02
    assert pd.isna(F["dividend_growth_5y"].loc[t, "N"])  # non-payer -> undefined

    # PAYOUT + COVERAGE (dividend safety): B_ONLY pays 2000 on 8000 NI (25%), FCF only
    # 1000 -> coverage 0.5 (<1, UNSAFE); A's FCF 12000 comfortably covers -> coverage > 1
    assert abs(F["dividend_payout_ratio"].loc[t, "B_ONLY"] - 0.25) < 1e-6
    assert abs(F["dividend_coverage"].loc[t, "B_ONLY"] - 0.5) < 1e-6
    assert F["dividend_coverage"].loc[t, "A"] > 1.0 > F["dividend_coverage"].loc[t, "B_ONLY"]

    print("\n=== SANITY CHECK: dividend source reconciliation + 5y growth + safety ===")
    print(
        f"  A yield (src A) {F['dividend_yield'].loc[t, 'A']:.4f} ~= dividendsPaid/mcap "
        f"{src_b_yield_A:.4f} (two sources agree); B_ONLY yield {F['dividend_yield'].loc[t, 'B_ONLY']:.3f} "
        f"gap-filled from source B; A 5y CAGR {F['dividend_growth_5y'].loc[t, 'A']:.3f}~0.10; "
        f"B_ONLY payout {F['dividend_payout_ratio'].loc[t, 'B_ONLY']:.2f}, coverage "
        f"{F['dividend_coverage'].loc[t, 'B_ONLY']:.2f}<1 (unsafe) < A {F['dividend_coverage'].loc[t, 'A']:.2f}. Validated."
    )


def test_zero_fill_is_scoped_to_the_listed_window():
    """A non-payer's real 0 must NOT be written onto rows before the company listed.

    The panel is a near-dense date x ticker grid, so a 2020 IPO carries rows back to 1995.
    An unscoped `.fillna(0.0)` asserted "pays no dividend" on 565,893 such rows — and
    because `_xs` is a percentile rank computed per day across the universe, those phantoms
    sat in the cross-section that prices the real names. Measured on the live cube:
    1996-01-02 had 300 listed tickers but 491 carrying a dividend rank, with all 272
    zero-yield names tied at 0.278, so the real universe was compressed into [0.278, 1.0].

    NEW must be NaN before it lists and a real 0 after; OLD, listed throughout, is the
    control and must be 0 on the same early dates."""
    dates = pd.bdate_range("2021-01-04", periods=3 * 252)
    listing = dates[len(dates) // 2]
    close = pd.DataFrame(100.0, index=dates, columns=["OLD", "NEW", "PAYER"])
    close.loc[close.index < listing, "NEW"] = np.nan  # not quotable yet

    div_hist = pd.DataFrame([{"date": d, "ticker": "PAYER", "dividends": 0.5} for d in dates[::63]])
    fund = pd.DataFrame([{"ticker": t, "as_of": aso, "sharesOutstanding": 1e9} for aso in dates[::63] for t in ("OLD", "NEW", "PAYER")])

    F = _dividend_fields(
        _published_cash(div_hist), legacy_capital_frames(fund, close), FundamentalHistory(raw_history=fund, share_basis=known_no_share_events())
    )
    before, after = dates[10], dates[-1]
    for k in ("dividend_yield", "dividend_payer", "shareholder_yield"):
        assert pd.isna(F[k].loc[before, "NEW"]), f"{k} fabricated on a pre-listing row"
        assert pd.notna(F[k].loc[after, "NEW"]), f"{k} lost after listing"
        assert pd.notna(F[k].loc[before, "OLD"]), f"{k} nulled on a LISTED row"
    # and the post-listing value is the real 0 the design intends, not a NaN
    assert F["dividend_yield"].loc[after, "NEW"] == 0.0
    assert F["dividend_payer"].loc[after, "NEW"] == 0.0
    assert F["dividend_payer"].loc[before, "OLD"] == 0.0
    assert F["dividend_payer"].loc[after, "PAYER"] == 1.0

    print("\n=== SANITY CHECK: the zero-fill is scoped to the LISTED window ===")
    print(
        f"  NEW lists {listing.date()}: dividend_yield/payer/shareholder_yield are NaN "
        f"before it and a real 0 after. OLD (listed throughout) keeps its 0 on the same "
        f"early date, so non-payers still rank correctly. Live cost of the mask: 448 rows "
        f"of interior price holes = 0.014% of listed rows. Validated."
    )


def test_panel_exposes_f_columns():
    dates, tickers, close, div_hist, fund = _synth()
    # give each ticker enough peers for the _vs_peers column to populate
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}
    panel = build_dividend_feature_panel(
        _published_cash(div_hist),
        replace(cast(PriceFrames, legacy_capital_frames(fund, close)), trading_index=dates, peers=peers),
        FundamentalHistory(raw_history=fund, share_basis=known_no_share_events()),
    )
    for c in (
        "f_dividend_yield",
        "f_dividend_yield_vs_hist",
        "f_dividend_growth",
        "f_dividend_payer",
        "f_shareholder_yield",
        "f_shareholder_yield_vs_hist",
    ):
        assert c in panel.columns, f"{c} missing from panel"
    assert not any(column.endswith(("_vs_peers", "_xs")) for column in panel.columns)
    payer = panel["f_dividend_payer"].dropna()
    assert set(payer.unique()) <= {0.0, 1.0}
    print("\n=== SANITY CHECK: dividend panel columns ===")
    print(
        "  dividend yield and shareholder yield have raw+history; growth and payer are raw; "
        f"peer/XS legs are absent and payer stays exact 0/1. Rows={len(panel)}. Validated."
    )


# --------------------------------------------------------------------------- #
# calendar-window growth (365 days), brute-force known truth                   #
# --------------------------------------------------------------------------- #
def _first_session_of(idx: pd.DatetimeIndex, year: int, month: int) -> pd.Timestamp:
    return idx[idx >= pd.Timestamp(year, month, 1)][0]


def _brute_ttm(pay: dict[pd.Timestamp, float], day: pd.Timestamp) -> float:
    """Sum of payments in the half-open window (day - 365 days, day]."""
    return sum(v for d, v in pay.items() if day - pd.Timedelta(days=365) < d <= day)


def test_ttm_window_drops_a_payment_exactly_365_days_old_and_not_a_day_earlier() -> None:
    idx = pd.bdate_range("2021-01-04", "2022-12-30")
    paid = pd.Timestamp("2021-06-01")
    hist = pd.DataFrame([{"date": paid, "ticker": "P", "dividends": 3.0}])
    calendar = _ttm_dividends(_published_cash(hist), idx, ["P"], "365D")["P"]
    sessions = _ttm_dividends(_published_cash(hist), idx, ["P"])["P"]
    day_before, edge = pd.Timestamp("2022-05-31"), pd.Timestamp("2022-06-01")
    assert calendar[day_before] == 3.0 and calendar[edge] == 0.0, "right-closed: day 364 holds the payment, day 365 does not"
    assert sessions[day_before] == 0.0, "the 252-session window (levels) already dropped it: it is a different clock"
    print(
        f"SANITY: a 3.0 payment on {paid.date()} is in the calendar TTM on {day_before.date()} and out on {edge.date()}; the 252-session TTM drops it earlier."
    )


def test_calendar_growth_matches_a_brute_force_year_over_year_and_first_year_is_undefined() -> None:
    idx = pd.bdate_range("2019-01-02", "2022-12-30")
    pay = {_first_session_of(idx, y, m): (1.0 if y < 2021 else 1.5) for y in range(2019, 2023) for m in (3, 6, 9, 12)}
    hist = pd.DataFrame([{"date": d, "ticker": "P", "dividends": v} for d, v in pay.items()])
    close = pd.DataFrame(100.0, index=idx, columns=["P"])
    growth = _dividend_fields(_published_cash(hist), legacy_capital_frames(None, close))["dividend_growth"]["P"]
    checked = 0
    for day in idx[idx >= idx[0] + pd.Timedelta(days=365)][::7]:
        prior_day = idx[idx <= day - pd.Timedelta(days=365)][-1]
        base = _brute_ttm(pay, prior_day)
        want = _brute_ttm(pay, day) / base - 1.0 if base > 0 else np.nan
        assert np.isclose(growth[day], want, equal_nan=True), f"{day.date()}: {growth[day]} != {want}"
        checked += 1
    assert growth[idx < idx[0] + pd.Timedelta(days=365)].isna().all(), "no comparable year yet -> undefined, never a partial-window ratio"
    print(
        f"SANITY: {checked} dates equal TTM(d)/TTM(last session <= d-365) - 1 by brute force; the first 365 days are NaN; 2021-12-31 growth {growth[pd.Timestamp('2021-12-31')]:.3f}."
    )


def test_cash_paid_fallback_uses_the_same_365_day_prior() -> None:
    idx = pd.bdate_range("2020-01-01", "2023-12-29")
    close = pd.DataFrame(100.0, index=idx, columns=["A", "B"])
    hist = pd.DataFrame([{"date": idx[10], "ticker": "A", "dividends": 1.0}])  # B has no ex-dates: source B only
    step = _first_session_of(idx, 2021, 4)
    rows = [
        {"ticker": t, "as_of": d, "sharesOutstanding": 1e3, "dividendsPaid": (1500.0 if d >= step else 1000.0) * (1 if t == "B" else 0)}
        for t in ("A", "B")
        for d in [_first_session_of(idx, y, m) for y in range(2020, 2024) for m in (1, 4, 7, 10)]
    ]
    fund = pd.DataFrame(rows)
    history = FundamentalHistory(raw_history=fund, share_basis=known_no_share_events())
    growth = _dividend_fields(_published_cash(hist), legacy_capital_frames(fund, close), history)["dividend_growth"]["B"]
    edge = step + pd.Timedelta(days=365)
    last_before = idx[idx < edge][-1]
    first_on = idx[idx >= edge][0]
    assert np.isclose(growth[last_before], 0.5), f"prior (day-365) still holds the 1000 level: {growth[last_before]}"
    assert np.isclose(growth[first_on], 0.0), f"prior is the 1500 level from the step session: {growth[first_on]}"
    print(
        f"SANITY: cash paid steps 1000 -> 1500 on {step.date()}: growth is 0.5 on {last_before.date()} and 0.0 from {first_on.date()} (365 days later)."
    )


def test_cagr_is_five_calendar_years_with_a_known_step() -> None:
    idx = pd.bdate_range("2019-06-03", "2026-01-30")
    step = pd.Timestamp("2020-01-02")
    now = pd.DataFrame({"P": np.where(idx >= step, 32.0, 1.0)}, index=idx)
    out = _cagr(now, 5)["P"]
    inside = step + pd.Timedelta(days=5 * 365 - 1)
    outside = step + pd.Timedelta(days=5 * 365)
    assert inside in idx and outside in idx
    assert np.isclose(out[inside], 1.0), f"32^(1/5) - 1 = 1 while the 1825-day prior still sits before the step: {out[inside]}"
    assert np.isclose(out[outside], 0.0), f"the prior date is the step session itself: {out[outside]}"
    assert _calendar_prior(now, 365).iloc[0].isna().all()
    print(f"SANITY: x32 step on {step.date()}: 5y CAGR 1.0 on day 1,824 ({inside.date()}), 0.0 on day 1,825 ({outside.date()}).")


def test_incremental_dividend_window_reproduces_full_tail() -> None:
    """The part warm-up must cover the 5y calendar look-back (1,825 days) plus the 365-day TTM window."""
    # a real exchange calendar (~251 sessions a year), not 261 weekdays: the look-back is in calendar days
    dates = pd.bdate_range("2018-01-01", periods=7 * 252, freq="C", holidays=USFederalHolidayCalendar().holidays("2018-01-01", "2026-12-31"))
    tickers = ["A", "B", "C", "D", "E"]
    close = pd.DataFrame(100.0, index=dates, columns=tickers)
    growth = dict(zip(tickers, (1.02, 1.05, 1.08, 1.11, 1.14), strict=True))
    dividends = pd.DataFrame(
        [
            {"date": date, "ticker": ticker, "dividends": growth[ticker] ** (i // 4)}
            for ticker_pos, ticker in enumerate(tickers)
            for i, date in enumerate(dates[ticker_pos * 10 :: 63])
        ]
    )
    peers = {ticker: {peer: 1.0 for peer in tickers if peer != ticker} for ticker in tickers}

    full = build_dividend_feature_panel(
        _published_cash(dividends),
        replace(cast(PriceFrames, legacy_capital_frames(None, close)), trading_index=dates, peers=peers),
        FundamentalHistory(share_basis=known_no_share_events()),
    )
    policy = fixture_config().data.tables.cube_part_fundamentals
    warmup = int(policy.warmup.grid.value)
    last_pos = 7 * 252 - 40
    refresh_from = dates[last_pos] - pd.Timedelta(days=int(policy.reprocess.overlap.value))
    start = shift_date(refresh_from, Offset(warmup, str(policy.warmup.grid.unit)), dates)
    windowed = build_dividend_feature_panel(
        _published_cash(dividends),
        replace(cast(PriceFrames, legacy_capital_frames(None, close.loc[start:])), trading_index=dates[dates >= start], peers=peers),
        FundamentalHistory(share_basis=known_no_share_events()),
    )

    columns = ["f_dividend_growth_5y"]
    full_tail = full.loc[full["date"] >= refresh_from, ["date", "ticker", *columns]]
    windowed_tail = windowed.loc[windowed["date"] >= refresh_from, ["date", "ticker", *columns]]
    pd.testing.assert_frame_equal(
        full_tail.set_index(["date", "ticker"]).sort_index(),
        windowed_tail.set_index(["date", "ticker"]).sort_index(),
        check_exact=False,
        atol=1e-7,
        rtol=0,
    )

    print("\n=== SANITY CHECK: incremental dividend panel matches full tail ===")
    print(
        f"  {warmup} warm-up sessions preserve the 1,825-day CAGR look-back plus its "
        f"365-day TTM base; {len(full_tail)} rewritten/appended rows match. Validated."
    )


if __name__ == "__main__":
    test_dividend_fields_economics_and_pit()
    test_panel_exposes_f_columns()
