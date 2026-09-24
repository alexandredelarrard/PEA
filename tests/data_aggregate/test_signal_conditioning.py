"""The daily price-conditioning layer (`ic_sig_*`, registry section 7) -- part-2-validation.md
10.6 assigns this file V6, V7, V8, B1 and L8.

    B1  the panel MOVES on a day with no filing, while the disclosure features hold flat
    V8  `_age_days` is NaN before the first event and increases by exactly 1 per trading day
    L8  no future prices: rebuilding with the price frame truncated at d reproduces day d
    V6  NaN, never 0, before a ticker's first event
    V7  monotone behaviour between events

B1 is the defining check of the whole two-layer architecture (research report acceptance test
#13): without it the cube's informed-capital families are step functions that only move four
times a year, and nothing else in the suite would notice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.institutionals.signal_conditioning import (
    COST_ANCHOR_WINDOW,
    EMISSION,
    EXCURSION_LOOKBACK,
    build_signal_conditioning_panel,
)
from tests.conftest import make_frames

IDX = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=520))
TICKERS = ["A", "B", "C", "D"]
PEERS = {t: {p: 1.0 for p in TICKERS if p != t} for t in TICKERS}


def _close(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({t: 100 * np.cumprod(1 + rng.normal(0.0004, 0.015, len(IDX))) for t in TICKERS}, index=IDX)


def _events(dates: dict[str, list[int]]) -> dict[str, pd.DataFrame]:
    """`{ticker: [grid positions]}` -> the sink's event frames for all three families."""
    rows = [{"ticker": t, "date": IDX[i], "value": 1e5, "shares": 100.0} for t, positions in dates.items() for i in positions]
    frame = pd.DataFrame(rows)
    return {"insider": frame, "super": frame[["ticker", "date"]], "act": frame.iloc[:1]}


def test_age_days_is_nan_before_the_first_event_then_counts_trading_days():
    """V8 + V6. The event day is 0, each later trading day adds exactly 1, and everything
    before the ticker's first event is NaN -- never 0, which would read as "an event happened
    today" on a name that has never had one."""
    close = _close()
    panel = build_signal_conditioning_panel(make_frames(IDX, PEERS, close_total=close, close_split=close), _events({"A": [100, 300]}))
    a = panel[panel["ticker"] == "A"].set_index("date").sort_index()
    age = a["f_ic_sig_insider_age_days"]
    assert age.loc[IDX[:100]].isna().all(), "age carried a value before the first event"
    assert age.loc[IDX[100]] == 0.0
    assert age.loc[IDX[101]] == 1.0 and age.loc[IDX[150]] == 50.0
    assert age.loc[IDX[299]] == 199.0
    assert age.loc[IDX[300]] == 0.0, "the second event did not reset the clock"
    # exactly +1 per trading day between the two events
    between = age.loc[IDX[100] : IDX[299]]
    assert (between.diff().dropna() == 1.0).all()
    # a ticker with NO event of the family is NaN throughout
    b = panel[panel["ticker"] == "B"]["f_ic_sig_insider_age_days"]
    assert b.isna().all()
    print("\n=== SANITY CHECK: V8/V6 ic_sig_*_age_days ===")
    print(
        f"  A: NaN for 100 days, 0 on the event, +1 per trading day to 199, resets to 0 on "
        f"the second event; B (no event) NaN throughout ({len(b)} rows). Validated."
    )


def test_the_panel_moves_on_every_day_with_no_filing():
    """B1 -- THE defining check (report acceptance test #13). One event, then a 30-trading-day
    span with nothing filed at all: every conditioning leg must change on every one of those
    days. A step-function feature would be flat and is what this proves the layer replaces."""
    close = _close(seed=5)
    panel = build_signal_conditioning_panel(
        make_frames(IDX, PEERS, close_total=close, close_split=close, sector_ret=close.pct_change().rolling(5).mean().bfill()),
        _events({"A": [100], "B": [90]}),
    )
    span = IDX[150:180]  # 30 trading days, no event anywhere
    a = panel[(panel["ticker"] == "A") & panel["date"].isin(span)].set_index("date").sort_index()
    daily = ("f_ic_sig_insider_age_days", "f_ic_sig_insider_ret_since", "f_ic_sig_insider_resid_ret_since", "f_ic_sig_insider_price_vs_buy")
    moved = {}
    for col in daily:
        d = a[col].diff().dropna()
        moved[col] = int((d != 0).sum())
        assert moved[col] == len(d), f"{col} was flat on {len(d) - moved[col]} of {len(d)} days"
    print("\n=== SANITY CHECK: B1 -- the panel moves without a filing ===")
    print(
        f"  {span[0].date()} -> {span[-1].date()}, zero events: "
        + ", ".join(f"{c.removeprefix('f_ic_sig_')} moved {n}/{len(span) - 1} days" for c, n in moved.items())
        + ". The two-layer architecture EXISTS, not just described. Validated."
    )


def test_insider_frontier_masks_only_the_insider_conditioning_family():
    """An unavailable insider tail must not erase independently covered families."""
    close = _close(seed=7)
    frontier = IDX[150]
    panel = build_signal_conditioning_panel(
        make_frames(IDX, PEERS, close_total=close, close_split=close),
        _events({"A": [100]}),
        frontiers={"insider": frontier},
    )
    tail = panel[panel["date"] > frontier]
    insider_cols = [column for column in panel if column.startswith("f_ic_sig_insider_")]
    assert tail[insider_cols].isna().to_numpy().all()
    assert tail["f_ic_sig_super_age_days"].notna().any()
    print(
        f"SANITY: all {len(insider_cols)} insider-conditioning legs are NaN after "
        f"{frontier.date()}, while the independently covered superinvestor age remains "
        "populated."
    )


def test_no_future_prices_reach_a_conditioning_value():
    """L8. Rebuild with the price frame truncated at day d; every value on or before d must be
    identical. A leak would show up as the truncated run disagreeing."""
    close = _close(seed=9)
    events = _events({"A": [100], "B": [40], "C": [220]})
    full = build_signal_conditioning_panel(make_frames(IDX, PEERS, close_total=close, close_split=close), events)
    cut = 300
    truncated = build_signal_conditioning_panel(make_frames(IDX[:cut], PEERS, close_total=close.iloc[:cut], close_split=close.iloc[:cut]), events)
    cols = [c for c in full.columns if c.startswith("f_")]
    a = full[full["date"] <= IDX[cut - 1]].set_index(["date", "ticker"])[cols].sort_index()
    b = truncated.set_index(["date", "ticker"])[cols].sort_index()
    common = a.index.intersection(b.index)
    diff = (a.loc[common] - b.loc[common]).abs()
    worst = float(np.nanmax(diff.to_numpy())) if len(common) else 0.0
    # the `_xs` legs are per-date ranks over whatever tickers exist that day, so they are
    # unchanged too: truncating DATES cannot change a within-date cross-section.
    assert worst < 1e-9, f"truncating the future changed a past value by {worst}"
    print("\n=== SANITY CHECK: L8 -- no future prices in the conditioning layer ===")
    print(
        f"  rebuilt with prices truncated at {IDX[cut - 1].date()}: max |diff| over "
        f"{len(common):,} shared (date, ticker) cells x {len(cols)} legs = {worst:.2e}. "
        "Validated."
    )


def test_excursions_are_signed_capped_and_anchored():
    """#81/#82. The run-up is >= 0 and the drawdown <= 0 by construction, because both are
    measured against the START of the window they search -- and the cap moves BOTH ends, so an
    event 380 days old is scored over the trailing 252 days against the price 252 days ago.
    Capping only the search would divide a trailing minimum by a price from 380 days back and
    report a favourable "max adverse excursion" on any name that had since doubled."""
    close = _close(seed=13)
    panel = build_signal_conditioning_panel(make_frames(IDX, PEERS, close_total=close, close_split=close), _events({"A": [20]}))
    a = panel[panel["ticker"] == "A"].set_index("date").sort_index()
    runup, dd = a["f_ic_sig_insider_max_runup_since_buy"], a["f_ic_sig_insider_max_dd_since_buy"]
    assert (runup.dropna() >= -1e-12).all(), "a run-up went negative"
    assert (dd.dropna() <= 1e-12).all(), "a drawdown went positive"
    assert runup.loc[IDX[:20]].isna().all() and dd.loc[IDX[:20]].isna().all()
    assert abs(runup.loc[IDX[20]]) < 1e-12 and abs(dd.loc[IDX[20]]) < 1e-12
    # inside the cap: the running extreme since the anchor. 1e-6, not 1e-9 -- the panel legs
    # are cast to float32 by `build_peer_relative_panel`, so ~1e-8 is exact agreement here.
    at = 200
    seg = close["A"].iloc[20 : at + 1]
    assert abs(runup.loc[IDX[at]] - (seg.max() / close["A"].iloc[20] - 1)) < 1e-6
    # past the cap: the trailing 252-day extreme against the price 252 days ago, not against
    # the 380-day-old anchor
    at = 400
    start = at - EXCURSION_LOOKBACK + 1
    window = close["A"].iloc[start : at + 1]
    assert abs(runup.loc[IDX[at]] - (window.max() / close["A"].iloc[start] - 1)) < 1e-6
    assert abs(dd.loc[IDX[at]] - (window.min() / close["A"].iloc[start] - 1)) < 1e-6
    print("\n=== SANITY CHECK: capped excursions since the last insider buy ===")
    print(
        f"  run-up >= 0 and drawdown <= 0 always; both 0 on the event day and NaN before it; "
        f"at age 180 the extreme is taken since the anchor, at age 380 over the trailing "
        f"{EXCURSION_LOOKBACK} days against that window's own opening price "
        f"(runup {runup.loc[IDX[at]]:+.3f} / dd {dd.loc[IDX[at]]:+.3f}). Validated."
    )


def test_the_cost_anchor_is_restated_across_a_split():
    """#80. A Form 4 carries the price AS TRADED and `close_split` is restated to today's
    basis, so a 10-for-1 split between the purchase and today reads as a 90% collapse unless
    the filed side is restated. Two purchases of 100 shares at $1,000 each, a 10:1 split, and
    a post-split price of $100: the insider is exactly flat, not 90% under water."""
    close = pd.DataFrame({t: 100.0 for t in TICKERS}, index=IDX)  # today's (post-split) basis
    events = {"insider": pd.DataFrame([{"ticker": "A", "date": IDX[400], "value": 100_000.0, "shares": 100.0}])}
    splits = pd.DataFrame({"ticker": ["A"], "date": [IDX[450]], "ratio": [10.0]})

    naive = build_signal_conditioning_panel(make_frames(IDX, PEERS, close_total=close, close_split=close), events, splits=None)
    fixed = build_signal_conditioning_panel(make_frames(IDX, PEERS, close_total=close, close_split=close), events, splits=splits)
    day = IDX[460]  # after the split, inside the 126d window
    got_naive = naive.loc[(naive["ticker"] == "A") & (naive["date"] == day), "f_ic_sig_insider_price_vs_buy"].iloc[0]
    got = fixed.loc[(fixed["ticker"] == "A") & (fixed["date"] == day), "f_ic_sig_insider_price_vs_buy"].iloc[0]
    assert abs(got_naive - (100 / 1000 - 1)) < 1e-9  # -90%, the fabricated number
    assert abs(got) < 1e-9, f"restated anchor still wrong: {got}"
    print("\n=== SANITY CHECK: the insider cost anchor across a split ===")
    print(
        f"  100 shares at $1,000, then 10-for-1, now $100: unrestated reads "
        f"{got_naive:+.1%} (a split detector), restated reads {got:+.4f} -- flat, which is "
        f"the truth. Window {COST_ANCHOR_WINDOW} trading days. Validated."
    )


def test_every_declared_feature_is_emitted():
    close = _close(seed=21)
    panel = build_signal_conditioning_panel(
        make_frames(
            IDX, PEERS, close_total=close, close_split=close, sector_ret=close.pct_change().rolling(5).mean().bfill(), ret=close.pct_change()
        ),
        _events({"A": [100, 300], "B": [40], "C": [220], "D": [10, 250]}),
    )
    expected = set()
    for name, mode in EMISSION.items():
        expected.add(f"f_{name}")
        if mode == "raw+xs":
            expected.add(f"f_{name}_xs")
    emitted = {c for c in panel.columns if c.startswith("f_")}
    assert emitted == expected, f"missing {sorted(expected - emitted)}; " f"undeclared {sorted(emitted - expected)}"
    for col in sorted(emitted):
        assert panel[col].notna().any(), f"{col} is declared, emitted and ALL-NaN"
    print("\n=== SANITY CHECK: ic_sig_* declared vs emitted ===")
    print(
        f"  {len(EMISSION)} features -> {len(emitted)} legs, exact match with the EMISSION "
        f"map, every one carrying non-null cells (an existence assertion alone passes on a "
        f"dead column). Validated."
    )
