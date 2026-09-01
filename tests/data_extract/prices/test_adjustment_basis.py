"""
test_adjustment_basis.py  (tests/data_extract/prices/)
--------------------------------------------------------------------------------------------
The adjustment-basis contract, pinned.

WHY THIS FILE EXISTS: before 2026-09-01, `tests/` contained ZERO occurrences of
`auto_adjust`, `adj_close`, `unadjusted` or any split-ratio assertion. The market-cap defect
survived for years because nothing checked -- not because the check was hard. Every assertion
below would have failed on the pre-fix table.

Two kinds of test here, deliberately:
  * SYNTHETIC known-truth for the RULES (the macro basis pin, the forward-return contract,
    and that the validator's invariants actually FIRE) -- no DB, no network;
  * REAL DATA for the economic facts (AAPL 106.26, KO 24.98, AMZN's exact identity), because
    "is the stored series split-adjusted-only" is a question about the stored series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.common.prices import forward_compound, forward_return
from src.data_store.schema import Tables

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def store():
    from src.context import get_config_context
    _, context = get_config_context("./configs", use_cache=False, save=False)
    return context.store


def _series(store, ticker: str) -> pd.DataFrame:
    df = store.load(Tables.prices, columns=["ticker", "date", "close_split", "close_total"],
                    where={"ticker": ticker}, optional=True)
    if df is None or df.empty:
        pytest.skip(f"`prices` has no rows for {ticker}")
    return df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 1. the stored basis is SPLIT-adjusted, not dividend-adjusted                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ticker,day,split_px,total_px", [
    # AAPL 2020-07-31: Yahoo `Close` 106.26 == Sharadar `price` 106.26 on two independent
    # vendors; `Adj Close` 102.795 is what the repo used to store as `close`.
    ("AAPL", "2020-07-31", 106.26, 102.795),
    # KO 2004-02-27: a 22-year dividend history, so D(d) = 0.512 -- the largest gap in the
    # sample and the clearest possible discriminator.
    ("KO", "2004-02-27", 24.98, 12.790),
])
def test_close_split_is_the_split_adjusted_quote(store, ticker, day, split_px, total_px):
    """`close_split` must be the SPLIT-ADJUSTED quote and `close_total` the total-return one.

    Getting this backwards is not a cosmetic error: `close_split x sharesbas` is the market
    cap identity, and on `close_total` the dividend factor D(d) survives into the product --
    median 0.618 in 2003, i.e. every value denominator 38% too low and monotone in FUTURE
    dividends."""
    row = _series(store, ticker)
    row = row[pd.to_datetime(row["date"]).dt.strftime("%Y-%m-%d") == day]
    if row.empty:
        pytest.skip(f"{ticker} has no bar on {day}")
    got_split = float(row["close_split"].iloc[0])
    got_total = float(row["close_total"].iloc[0])

    assert got_split == pytest.approx(split_px, abs=0.01), (
        f"{ticker} {day}: close_split {got_split} != {split_px}. If this reads {total_px} "
        f"the two columns are SWAPPED and market cap is on the dividend-adjusted basis.")
    assert got_total == pytest.approx(total_px, abs=0.01)

    print(f"\n=== SANITY CHECK: {ticker} {day} basis ===")
    print(f"  close_split {got_split} (== Sharadar `price`, split-adjusted only)")
    print(f"  close_total {got_total} (== old `close`; D = {got_total / got_split:.4f})")
    print("  Validated: the level basis and the return basis are distinct and correctly "
          "assigned.")


def test_a_non_payer_has_identical_bases(store):
    """AMZN pays no dividend, so `close_split == close_total` EXACTLY on every row.

    The cleanest regression guard in the suite: it is exact equality, not a tolerance, so any
    stray adjustment applied to one column and not the other shows up immediately."""
    df = _series(store, "AMZN").dropna(subset=["close_split", "close_total"])
    identical = (df["close_split"] == df["close_total"]).sum()
    assert identical == len(df), (
        f"AMZN: {len(df) - identical} of {len(df)} rows differ between the two bases, but "
        f"AMZN has never paid a dividend -- so something adjusted one column and not the "
        f"other.")

    print("\n=== SANITY CHECK: non-payer identity ===")
    print(f"  AMZN: {identical:,}/{len(df):,} rows have close_split == close_total EXACTLY.")
    print("  Validated: the dividend leg is the ONLY difference between the two columns.")


def test_the_dividend_factor_is_monotone_and_terminates_at_one(store):
    """`close_total / close_split` = D(d), the product of (1 - div/price) over ex-dates AFTER
    d. So it must RISE monotonically toward the present and reach 1.0 on the last row -- no
    future dividends remain to discount.

    This encodes the definition of D(d) as a test. A violation means the two columns came
    from different responses or different vintages."""
    df = _series(store, "KO").dropna(subset=["close_split", "close_total"])
    df = df[df["close_split"] > 0]
    d = (df["close_total"] / df["close_split"]).to_numpy()

    # A tolerance, not exact monotonicity: each ratio is a quotient of two rounded prices, so
    # neighbouring days wobble in the 4th decimal. The SHAPE is what is being asserted.
    drops = int((np.diff(d) < -1e-3).sum())
    assert drops == 0, f"D(d) falls on {drops} day(s) -- it must be non-decreasing in date"
    assert d[-1] == pytest.approx(1.0, abs=1e-6), (
        f"D(last row) = {d[-1]}, must be exactly 1.0 -- no dividends remain after the last "
        f"bar, so the two bases must coincide there")

    print("\n=== SANITY CHECK: D(d) shape on KO ===")
    print(f"  first {d[0]:.4f} -> last {d[-1]:.6f}; 0 decreases over {len(d):,} rows.")
    print("  Validated: close_total is close_split discounted by FUTURE dividends only.")


# --------------------------------------------------------------------------- #
# 2. the macro leg must NOT follow the equity leg off total return            #
# --------------------------------------------------------------------------- #
def test_the_macro_leg_is_pinned_to_total_return():
    """⚠ THE TRAP THIS FILE EXISTS TO PIN. `download_ohlcv` is SHARED, and the macro caller
    needs the opposite basis from the equity one.

    `SPY` is stored under the series name `equity_tr`: it is the L/S benchmark leg and what
    `beta_market` and `fwd_market` are measured against inside EVERY label. `XLE` (`energy`)
    pays ~3%. A bare `auto_adjust=False` flip -- the obvious way to give the equity leg its
    split-adjusted close -- would silently convert both to PRICE returns and corrupt every
    beta and every label in the book, with no error anywhere."""
    import src.data_extract.utils.prices.fetch_macro as fm
    from src.constants.constants_price import MACRO_PRICE_SERIES

    seen: dict = {}

    def _spy(tickers, since, until, *a, **kw):
        seen.update(kw)
        idx = pd.date_range("2024-01-02", periods=3, freq="B")
        return pd.concat([pd.DataFrame({"date": idx, "ticker": sym,
                                        "close_total": [100.0, 101.0, 102.0]})
                          for sym in MACRO_PRICE_SERIES], ignore_index=True)

    original = fm.download_ohlcv
    try:
        fm.download_ohlcv = _spy
        ctx = type("Ctx", (), {"log": type("L", (), {"info": lambda *a, **k: None,
                                                     "warning": lambda *a, **k: None})()})()
        fm._fetch_price_leg(ctx, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-05"))
    finally:
        fm.download_ohlcv = original

    assert seen.get("auto_adjust") is True, (
        f"the macro leg called download_ohlcv with auto_adjust={seen.get('auto_adjust')!r}. "
        f"It MUST be True: SPY is stored as `equity_tr` and every label is measured against "
        f"it, so a price-return SPY corrupts beta_market and fwd_market everywhere.")

    print("\n=== SANITY CHECK: macro basis pin ===")
    print(f"  _fetch_price_leg -> download_ohlcv(auto_adjust={seen.get('auto_adjust')}, "
          f"actions={seen.get('actions')})")
    print("  Validated: the equity leg's auto_adjust=False cannot drag the benchmark with "
          "it.")


# --------------------------------------------------------------------------- #
# 3. the label formula: compounded returns, not a price ratio                 #
# --------------------------------------------------------------------------- #
def test_forward_compound_and_forward_return_diverge_exactly_on_the_dividend():
    """The two agree for a NON-PAYER and diverge for a PAYER -- which is precisely why the
    labels had to move from one to the other.

    `targets.compute_epsilon` used `forward_return(close, h)` = `close.shift(-h)/close - 1`,
    a literal PRICE ratio, while every factor leg it was differenced against was already
    log-compounded from returns. Two defects in one line: mixed bases, and a label that
    systematically under-measures dividend payers. Synthetic known-truth, so the expected
    answer is arithmetic rather than assumed."""
    idx = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(7)
    price_ret = pd.DataFrame({"X": rng.normal(0.0003, 0.01, len(idx))}, index=idx)

    # a NON-PAYER: the price path IS the total-return path
    close_np = (1 + price_ret).cumprod() * 100.0
    h = 20
    a = forward_compound(price_ret, h)["X"]
    b = forward_return(close_np, h)["X"]
    both = a.notna() & b.notna()
    assert np.allclose(a[both], b[both], atol=1e-9), "they must agree when there is no dividend"

    # a PAYER: 0.5% quarterly, so the total-return path outruns the price path
    div = pd.Series(0.0, index=idx)
    div.iloc[::63] = 0.005
    total_ret = price_ret["X"] + div
    close_payer = (1 + price_ret["X"]).cumprod() * 100.0        # PRICE path only
    a_pay = forward_compound(total_ret.to_frame("X"), h)["X"]
    b_pay = forward_return(close_payer.to_frame("X"), h)["X"]
    valid = a_pay.notna() & b_pay.notna()
    gap = (a_pay[valid] - b_pay[valid]).mean()

    assert gap > 1e-4, (
        f"the compounded TOTAL return must exceed the PRICE ratio for a payer; gap {gap:.6f}")

    print("\n=== SANITY CHECK: label formula ===")
    print(f"  non-payer: forward_compound == forward_return to 1e-9 over "
          f"{int(both.sum())} windows")
    print(f"  payer (0.5%/qtr): forward_compound exceeds the price ratio by "
          f"{gap:+.4%} per {h}-day window on average")
    print("  Validated: the label change is exactly the dividends, and nothing else.")


# --------------------------------------------------------------------------- #
# 4. THE GATE MUST FIRE -- a gate that has never failed has never been tested #
# --------------------------------------------------------------------------- #
def test_the_invariants_fire_on_a_deliberately_corrupted_ticker():
    """Corrupt one ticker's `close_split` by 2x and confirm invariants 1 and 2 both catch it.

    A validator nobody has seen fail is a validator nobody knows works. This is the same
    2x an unapplied split produces -- exactly the live MNST shape -- so it is not an
    artificial input either."""
    from src.validate.prices import invariant_market_cap, invariant_price_vintage

    idx = pd.bdate_range("2024-01-31", periods=12, freq="QE")
    clean = pd.DataFrame({
        "ticker": ["GOOD"] * len(idx) + ["BAD"] * len(idx),
        "date": list(idx) * 2,
        "close_split": [100.0] * len(idx) * 2,
        "price": [100.0] * len(idx) * 2,
        "sharesOutstanding": [1e9] * len(idx) * 2,
        "sharesbas": [1e9] * len(idx) * 2,
        "marketcap": [1e11] * len(idx) * 2,
    })
    assert invariant_market_cap(clean).failed == 0, "the clean panel must pass"
    assert invariant_price_vintage(clean).failed == 0

    corrupt = clean.copy()
    bad_rows = corrupt["ticker"] == "BAD"
    corrupt.loc[bad_rows, "close_split"] *= 2.0        # an unapplied 2:1 split

    mcap = invariant_market_cap(corrupt)
    vintage = invariant_price_vintage(corrupt)

    assert mcap.failed == len(idx), f"invariant 1 caught {mcap.failed} of {len(idx)} rows"
    assert set(mcap.failing_tickers) == {"BAD"}, "GOOD must not be implicated"
    assert mcap.failing_tickers["BAD"]["median_ratio"] == pytest.approx(2.0)
    assert vintage.failed == len(idx) and set(vintage.failing_tickers) == {"BAD"}

    print("\n=== SANITY CHECK: the gate fires ===")
    print(f"  clean panel        -> invariant 1: 0 failures, invariant 2: 0 failures")
    print(f"  BAD close_split x2 -> invariant 1: {mcap.failed} rows "
          f"(median ratio {mcap.failing_tickers['BAD']['median_ratio']}), "
          f"invariant 2: {vintage.failed} rows")
    print(f"  GOOD is untouched in both: {sorted(mcap.failing_tickers)}")
    print("  Validated: both invariants detect an unapplied split and cluster it to the one "
          "ticker responsible.")
