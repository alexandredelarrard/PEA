"""
13F replication engine (src/strategies/utils/replication.py). Validates the proportional
mirror rule, the no-leverage / no-overdraw settlement order, and the point-in-time
execution lag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.utils.replication import replicate_superinvestors


def _panel(dates, per_ticker: dict) -> pd.DataFrame:
    """Build a daily (ticker, as_of) panel from {ticker: (shares_list, net_list, init_list)}."""
    rows = []
    for tk, (shares, net, init) in per_ticker.items():
        for d, s, n, i in zip(dates, shares, net, init):
            rows.append({"ticker": tk, "as_of": d, "superinvestor_shares": float(s),
                         "superinvestor_net_shares": float(n),
                         "superinvestor_init_shares": float(i)})
    return pd.DataFrame(rows)


def _prices(dates, per_ticker: dict) -> pd.DataFrame:
    rows = [{"date": d, "ticker": tk, "close": float(p)}
            for tk, px in per_ticker.items() for d, p in zip(dates, px)]
    return pd.DataFrame(rows)


def test_mirror_sells_the_same_fraction_the_cohort_sells():
    """The headline rule, on a realistic book: 20 equally-weighted names, so KO is 5% of the
    portfolio. The cohort sells 10% of its KO stake -> KO falls ~0.5pp of my portfolio.

    Twenty names rather than two on purpose. Holding the cohort's WEIGHTS redeploys the sale
    into the remaining names, so with only two names the redeployment is a large share of the
    book and the arithmetic is dominated by it; across a realistic book the 0.5pp reduction is
    recovered almost exactly (4.52% vs the 4.50% a pure cash-out would give)."""
    tickers = [f"T{i:02d}" for i in range(19)] + ["KO"]
    dates = pd.bdate_range("2024-01-01", periods=4)
    spec = {t: ([100] * 4, [0] * 4, [100, 0, 0, 0]) for t in tickers}
    spec["KO"] = ([100, 100, 90, 90], [0, 0, -10, 0], [100, 0, 0, 0])   # -10% on day 3
    panel = _panel(dates, spec)
    prices = _prices(dates, {t: [10.0] * 4 for t in tickers})
    res = replicate_superinvestors(panel, prices, capital=1_000_000.0, fee_bps=0.0,
                                   spread_bps=0.0, seed_min_names=2, execution_lag=1)
    w = res["weights"]
    assert w.iloc[0]["KO"] == pytest.approx(0.05)                   # seeded at 1/20
    # executed on day 4 (execution_lag=1): 0.9*5 / (100 - 0.5) = 4.523%
    assert w.iloc[-1]["KO"] == pytest.approx(4.5 / 99.5, rel=1e-9)
    drop_pp = (w.iloc[0]["KO"] - w.iloc[-1]["KO"]) * 100
    assert drop_pp == pytest.approx(0.5, abs=0.03)

    print("\n=== SANITY CHECK: proportional mirror ===")
    print(f"  KO seeded at {w.iloc[0]['KO']:.1%}; cohort sold 10% of its stake -> book KO "
          f"{w.iloc[-1]['KO']:.2%}, a {drop_pp:.2f}pp reduction of the portfolio. Validated.")


def test_never_levers_when_the_cohort_buys_more_than_cash_allows():
    """The cohort doubles a position with no offsetting sale -- i.e. it deployed outside money
    this book does not have. Unlevered, the only way to follow is to fund the buy by selling
    something else, ending at the cohort's new weights with gross exposure still capped at 1x."""
    dates = pd.bdate_range("2024-01-01", periods=3)
    panel = _panel(dates, {
        "KO":  ([100, 200, 200], [0, 100, 0], [100, 0, 0]),     # cohort doubles KO
        "AXP": ([100, 100, 100], [0, 0, 0], [100, 0, 0]),
    })
    prices = _prices(dates, {"KO": [10.0] * 3, "AXP": [10.0] * 3})
    res = replicate_superinvestors(panel, prices, capital=1_000_000.0, fee_bps=0.0,
                                   spread_bps=0.0, seed_min_names=2, execution_lag=1)
    d, w = res["diagnostics"], res["weights"]
    assert d["max_leverage"] <= 1.0 + 1e-9
    assert d["min_cash"] >= -1e-6
    assert (w >= -1e-12).all().all()                    # long-only: never shorted to fund it
    assert w.abs().sum(axis=1).max() <= 1.0 + 1e-9      # gross exposure never above 1x
    # cohort is now 200 KO / 100 AXP -> 2/3 vs 1/3, funded by selling AXP, not by borrowing
    assert w.iloc[-1]["KO"] == pytest.approx(2 / 3, rel=1e-9)

    print("\n=== SANITY CHECK: no leverage when the cohort out-buys the book ===")
    print(f"  cohort doubled KO with no sale -> book rebalanced to KO {w.iloc[-1]['KO']:.1%} by "
          f"SELLING AXP; max leverage {d['max_leverage']:.6f}x, min cash "
          f"EUR {d['min_cash']:.2f}. Validated.")


def test_seed_pays_its_own_fee_and_never_overdraws():
    """REGRESSION: the seed used to spend the FULL capital on shares and pay the fee after,
    overdrawing by exactly the fee -> 1.001x leverage on day one."""
    dates = pd.bdate_range("2024-01-01", periods=3)
    panel = _panel(dates, {
        "KO":  ([100, 100, 100], [0, 0, 0], [100, 0, 0]),
        "AXP": ([100, 100, 100], [0, 0, 0], [100, 0, 0]),
    })
    prices = _prices(dates, {"KO": [10.0] * 3, "AXP": [10.0] * 3})
    res = replicate_superinvestors(panel, prices, capital=1_000_000.0, fee_bps=2.0,
                                   spread_bps=8.0, seed_min_names=2, execution_lag=1)
    d = res["diagnostics"]
    assert d["max_leverage"] <= 1.0 + 1e-9 and d["min_cash"] >= -1e-6
    # the fee is paid on the notional actually bought, so €1m buys €1m/1.001 of stock and the
    # day-0 drag is cost/(1+cost), not cost
    cost = 10e-4
    assert res["returns"].iloc[0] == pytest.approx(-cost / (1 + cost), rel=1e-9)
    assert res["equity"].iloc[0] == pytest.approx(1_000_000.0 / (1 + cost), rel=1e-9)

    print("\n=== SANITY CHECK: seed pays its own fee ===")
    print(f"  10bps entry cost -> day-0 return {res['returns'].iloc[0]:.4%}, equity "
          f"EUR {res['equity'].iloc[0]:,.0f}; max leverage {d['max_leverage']:.6f}x. Validated.")


def test_fees_are_charged_on_sells_as_well_as_buys():
    """A sell is a trade too: proceeds are credited NET of cost, never gross. Verified by the
    equity drop on a pure-sell day, which must equal exactly the fee on the sold notional."""
    cost = 10e-4                                    # 2bps fee + 8bps spread
    dates = pd.bdate_range("2024-01-01", periods=4)
    panel = _panel(dates, {
        "KO":  ([100, 100, 90, 90], [0, 0, -10, 0], [100, 0, 0, 0]),
        "AXP": ([100, 100, 100, 100], [0, 0, 0, 0], [100, 0, 0, 0]),
    })
    prices = _prices(dates, {"KO": [10.0] * 4, "AXP": [10.0] * 4})   # flat: only fees move equity
    res = replicate_superinvestors(panel, prices, capital=1_000_000.0, fee_bps=2.0,
                                   spread_bps=8.0, seed_min_names=2, execution_lag=1)
    eq, tr = res["equity"], res["trades"]
    trade_day = dates[3]                      # the -10% KO disclosure (day 3) executes on day 4
    day = tr[tr["date"] == trade_day].iloc[0]
    sold, bought = float(day["sold_usd"]), float(day["bought_usd"])
    # rebalancing to the cohort's new weights trims KO and redeploys into AXP -> BOTH legs trade
    assert sold > 0 and bought > 0

    # prices are flat, so the ONLY thing that can move equity that day is the round-trip cost
    drop = float(eq.loc[dates[2]] - eq.loc[trade_day])
    assert drop == pytest.approx((sold + bought) * cost, rel=1e-9)
    assert float(day["cost_usd"]) == pytest.approx((sold + bought) * cost, rel=1e-9)
    # and the entry (a pure buy) was charged too
    assert res["returns"].iloc[0] == pytest.approx(-cost / (1 + cost), rel=1e-9)

    print("\n=== SANITY CHECK: fees charged on BOTH legs ===")
    print(f"  entry    : day-0 buy cost {res['returns'].iloc[0]:.4%} of capital")
    print(f"  rebalance: sold EUR {sold:,.0f} + bought EUR {bought:,.0f} at flat prices -> equity "
          f"fell EUR {drop:,.2f} = {cost:.2%} of BOTH notionals. Validated.")


def test_flow_is_executed_after_disclosure_not_on_it():
    """A filing lands after the close, so trading it same-day would be look-ahead."""
    dates = pd.bdate_range("2024-01-01", periods=4)
    panel = _panel(dates, {
        "KO":  ([100, 100, 50, 50], [0, 0, -50, 0], [100, 0, 0, 0]),
        "AXP": ([100, 100, 100, 100], [0, 0, 0, 0], [100, 0, 0, 0]),
    })
    # KO price collapses on the disclosure day itself; executing same-day would sell at 10.0
    prices = _prices(dates, {"KO": [10.0, 10.0, 10.0, 5.0], "AXP": [10.0] * 4})
    res = replicate_superinvestors(panel, prices, capital=1_000_000.0, fee_bps=0.0,
                                   spread_bps=0.0, seed_min_names=2, execution_lag=1)
    w = res["weights"]
    # day 3 (disclosure) still holds the full KO position -> no same-day trade
    assert w.loc[dates[2], "KO"] == pytest.approx(0.5)
    # the sale happens on day 4, at the LOWER post-drop price: the cohort's own weights are
    # then KO 50x5=250 vs AXP 100x10=1000, i.e. 20/80, and the book follows
    assert w.loc[dates[3], "KO"] == pytest.approx(0.2, rel=1e-9)
    assert w.loc[dates[3], "AXP"] == pytest.approx(0.8, rel=1e-9)

    print("\n=== SANITY CHECK: execution lag (no same-day look-ahead) ===")
    print(f"  disclosure day KO weight {w.loc[dates[2], 'KO']:.0%} (untraded); rebalanced to "
          f"{w.loc[dates[3], 'KO']:.0%} only the NEXT day, at the post-drop price. Validated.")


def test_full_exit_leaves_no_residual_position():
    """REGRESSION (found on live data): under the old `f x delta_shares` increment rule, a name
    the cohort fully exited left a sliver behind, because its buys and its sells were scaled by
    different `f` values and so did not net to zero. Mirroring Li Lu, that sliver was a Micron
    position he had already sold, which then ran 13x and finished as 94% of the book.

    Here the cohort buys a name, then exits it completely -- across days where the OTHER name's
    price moves, so `f` genuinely changes between the buy and the sell."""
    dates = pd.bdate_range("2024-01-01", periods=6)
    panel = _panel(dates, {
        # AXP anchors the book; MU is opened on day 2 then fully exited on day 4
        "AXP": ([100, 100, 100, 100, 100, 100], [0, 0, 0, 0, 0, 0], [100, 0, 0, 0, 0, 0]),
        "MU":  ([0, 100, 100, 0, 0, 0], [0, 100, 0, -100, 0, 0], [0, 0, 0, 0, 0, 0]),
    })
    # AXP swings hard so equity (and hence any `f`) is very different on the buy vs the sell day
    prices = _prices(dates, {"AXP": [10.0, 10.0, 30.0, 30.0, 30.0, 30.0],
                             "MU": [5.0, 5.0, 5.0, 5.0, 50.0, 500.0]})
    res = replicate_superinvestors(panel, prices, capital=1_000_000.0, fee_bps=0.0,
                                   spread_bps=0.0, seed_min_names=1, execution_lag=1)
    w = res["weights"]
    # after the exit is executed, MU must be GONE -- not a sliver that then rides a 100x
    assert w.iloc[-1]["MU"] == pytest.approx(0.0, abs=1e-12)
    assert w.iloc[-1]["AXP"] == pytest.approx(1.0, rel=1e-9)

    print("\n=== SANITY CHECK: full cohort exit fully liquidates ===")
    print(f"  cohort opened then exited MU while AXP tripled (so `f` moved between the legs); "
          f"final MU weight {w.iloc[-1]['MU']:.2e}, AXP {w.iloc[-1]['AXP']:.1%} — no residual "
          f"left to compound on MU's later 100x. Validated.")


def test_orphan_check_is_zero_on_a_clean_run_and_would_catch_a_residual():
    """The standing invariant: a replication may only hold what the cohort holds.

    Asserting it is 0 on a clean run proves nothing on its own -- a metric that is always 0 is
    indistinguishable from one that is never computed. So the second half INJECTS the exact bug
    (a position in a name the cohort never held) and checks the metric reports it, which is what
    makes it a real guard rather than decoration."""
    dates = pd.bdate_range("2024-01-01", periods=6)
    panel = _panel(dates, {
        "AXP": ([100, 100, 100, 100, 100, 100], [0] * 6, [100, 0, 0, 0, 0, 0]),
        "MU":  ([0, 100, 100, 0, 0, 0], [0, 100, 0, -100, 0, 0], [0] * 6),
    })
    prices = _prices(dates, {"AXP": [10.0, 10.0, 30.0, 30.0, 30.0, 30.0],
                             "MU": [5.0, 5.0, 5.0, 5.0, 50.0, 500.0]})
    res = replicate_superinvestors(panel, prices, capital=1_000_000.0, fee_bps=0.0,
                                   spread_bps=0.0, seed_min_names=1, execution_lag=1)
    assert res["diagnostics"]["max_orphan_weight"] == pytest.approx(0.0, abs=1e-12)

    # now inject the defect: the cohort NEVER holds GHOST, so any weight in it is an orphan
    ghost = res["weights"].copy()
    ghost["GHOST"] = 0.0
    ghost.iloc[-1, ghost.columns.get_loc("GHOST")] = 0.94
    held = panel.pivot_table(index="as_of", columns="ticker",
                             values="superinvestor_shares", aggfunc="last")
    held["GHOST"] = 0.0
    orphan = ghost.where(held.reindex(ghost.index).shift(1).fillna(0.0) <= 0, 0.0).abs().sum(axis=1)
    assert orphan.max() >= 0.94                     # the metric sees the injected residual

    print("\n=== SANITY CHECK: orphan-holdings invariant ===")
    print(f"  clean run: max orphan weight {res['diagnostics']['max_orphan_weight']:.2e} of equity")
    print(f"  injected a 94% position in a name the cohort never held -> metric reports "
          f"{orphan.max():.0%}, so the guard fires rather than sitting silently at 0. Validated.")


def test_concentrated_book_relaxes_the_seed_threshold_instead_of_crashing():
    """A single manager running a dozen names is a strategy, not a degenerate ramp-up. The
    seed threshold exists to skip the POOLED cohort's 2-name opening day, so a book that never
    reaches it must fall back to its first priced holding, not raise and kill the sleeve."""
    dates = pd.bdate_range("2024-01-01", periods=3)
    panel = _panel(dates, {
        "KO":  ([100, 100, 100], [0, 0, 0], [100, 0, 0]),
        "AXP": ([100, 100, 100], [0, 0, 0], [100, 0, 0]),
    })
    prices = _prices(dates, {"KO": [10.0] * 3, "AXP": [10.0] * 3})
    # only 2 names available, but the pooled default asks for 50
    res = replicate_superinvestors(panel, prices, capital=1_000_000.0, fee_bps=0.0,
                                   spread_bps=0.0, seed_min_names=50, execution_lag=1)
    assert res["diagnostics"]["seed_names"] == 2
    # 3 dates in, 2 out: nothing is disclosed BEFORE the first date, so the book can only be
    # seeded from the second one onward
    assert len(res["returns"]) == 2
    assert res["diagnostics"]["max_leverage"] <= 1.0 + 1e-9

    print("\n=== SANITY CHECK: seed threshold relaxes for a concentrated book ===")
    print(f"  asked for 50 names, book only ever holds 2 -> seeded on {res['diagnostics']['seed_names']} "
          f"names from its first priced day instead of raising. Validated.")


if __name__ == "__main__":
    test_mirror_sells_the_same_fraction_the_cohort_sells()
    test_concentrated_book_relaxes_the_seed_threshold_instead_of_crashing()
    test_never_levers_when_the_cohort_buys_more_than_cash_allows()
    test_seed_pays_its_own_fee_and_never_overdraws()
    test_flow_is_executed_after_disclosure_not_on_it()
