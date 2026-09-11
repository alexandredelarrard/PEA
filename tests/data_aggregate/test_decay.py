"""Phase 2.1b — the exponential event-decay transform and the panel emission map.

`decay_events` is the machinery every SPARSE informed-capital signal depends on: an event flag
is 0 on ~99.9% of ticker-days, a peer basket of all-zeros has zero dispersion, and
`peer_relative` returns NaN for zero dispersion by policy -- so an undecayed flag is an ABSENT
feature, not a weak one. These tests pin the three properties that are decisions rather than
details (trading-day clock, event stacking, NaN-before-first-event) plus the half-life itself.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.institutionals.decay import decay_events


def _grid(n: int = 400, start: str = "2020-01-01") -> pd.DatetimeIndex:
    """A trading-day grid: business days, which is what the cube's date axis is."""
    return pd.bdate_range(start, periods=n)


def _events(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "ticker", "magnitude"])


# --------------------------------------------------------------------------- half-life

def test_halflife_is_exact_in_trading_days():
    """A single unit event reads exactly 0.5 after `halflife` TRADING days and 0.25 after two.

    This is the definition of the transform, so it is checked to floating-point tolerance
    rather than approximately. The offsets are counted on the grid, never in calendar days --
    the two differ by ~40% over a quarter, and the point of the trading-day clock is that
    `halflife=63` means one quarter of TRADING regardless of where the holidays fall.
    """
    idx, hl = _grid(300), 63
    out = decay_events(_events([(idx[10], "AAA", 1.0)]), idx, hl, magnitude_col="magnitude")
    assert out.loc[idx[10], "AAA"] == pytest.approx(1.0)
    assert out.loc[idx[10 + hl], "AAA"] == pytest.approx(0.5)
    assert out.loc[idx[10 + 2 * hl], "AAA"] == pytest.approx(0.25)
    assert out.loc[idx[10 + 3 * hl], "AAA"] == pytest.approx(0.125)


def test_magnitude_scales_the_whole_path():
    idx, hl = _grid(200), 21
    one = decay_events(_events([(idx[5], "AAA", 1.0)]), idx, hl, magnitude_col="magnitude")
    ten = decay_events(_events([(idx[5], "AAA", 10.0)]), idx, hl, magnitude_col="magnitude")
    assert np.allclose(ten["AAA"].to_numpy() / 10.0, one["AAA"].to_numpy(), equal_nan=True)


def test_default_magnitude_is_one_when_no_column_is_given():
    """The count-style default: "a 13D was filed" carries no size, and must weigh 1.0."""
    idx = _grid(100)
    ev = pd.DataFrame({"date": [idx[3]], "ticker": ["AAA"]})
    assert decay_events(ev, idx, 21).loc[idx[3], "AAA"] == pytest.approx(1.0)


# ------------------------------------------------------------------- V6: NaN, never 0

def test_v6_nan_before_the_first_event_never_zero():
    """No 13D ever filed here, versus one filed and fully decayed, are DIFFERENT facts.

    Zero-filling the first lets the model read a never-targeted name as a recovered one, so the
    pre-event region must be NaN and the post-event region must never return to NaN."""
    idx, first = _grid(300), 100
    out = decay_events(_events([(idx[first], "AAA", 1.0)]), idx, 21, magnitude_col="magnitude")
    before, after = out["AAA"].iloc[:first], out["AAA"].iloc[first:]
    assert before.isna().all(), "pre-event region must be entirely NaN"
    assert (before == 0).sum() == 0, "pre-event region must contain no zeros"
    assert after.notna().all(), "the series is dense from the first event onward"


def test_v6_is_per_ticker_not_global():
    """A late-arriving ticker keeps its own NaN prefix; another ticker's event must not end it."""
    idx = _grid(300)
    out = decay_events(_events([(idx[10], "AAA", 1.0), (idx[200], "BBB", 1.0)]),
                       idx, 21, magnitude_col="magnitude")
    assert out["BBB"].iloc[:200].isna().all()
    assert out["BBB"].iloc[200:].notna().all()
    assert out["AAA"].iloc[10:].notna().all()


def test_v6_survives_underflow_to_zero():
    """A value decayed below float resolution is still a NUMBER, not a NaN: the mask is built
    from the event history, not from `value > 0`. With halflife=1 over 400 days the tail
    underflows to exactly 0.0, which is the case that would break an `out > 0` mask.

    1,200 days at halflife=1 is ~1,200 halvings; float64 goes subnormal below ~2.2e-308 and
    reaches exactly 0.0 below ~5e-324, i.e. after ~1,080. A 400-day grid gets to ~1e-120,
    which is small but emphatically not zero -- the test would pass vacuously."""
    idx = _grid(1200)
    out = decay_events(_events([(idx[0], "AAA", 1.0)]), idx, 1, magnitude_col="magnitude")
    tail = out["AAA"].iloc[-50:]
    assert (tail == 0.0).all(), "expected genuine underflow in this regime"
    assert tail.notna().all(), "underflowed days must stay 0.0, not become NaN"


# ------------------------------------------------------- V7: monotone decay + stacking

def test_v7_strictly_decreasing_with_no_new_event():
    idx = _grid(300)
    out = decay_events(_events([(idx[10], "AAA", 1.0)]), idx, 63, magnitude_col="magnitude")
    path = out["AAA"].iloc[10:150].to_numpy()
    assert np.all(np.diff(path) < 0), "no new event -> the series must strictly decrease"


def test_events_stack_rather_than_overwrite():
    """Two 13D amendments a month apart must SUM. A last-event-wins rule would read an
    escalating campaign -- the actual signal -- as no news at all."""
    idx, hl = _grid(300), 63
    a, b = idx[10], idx[31]                      # ~one month of trading days apart
    both = decay_events(_events([(a, "AAA", 1.0), (b, "AAA", 1.0)]), idx, hl,
                        magnitude_col="magnitude")
    only_a = decay_events(_events([(a, "AAA", 1.0)]), idx, hl, magnitude_col="magnitude")
    only_b = decay_events(_events([(b, "AAA", 1.0)]), idx, hl, magnitude_col="magnitude")
    on_b = both.loc[b, "AAA"]
    assert on_b > only_a.loc[b, "AAA"] and on_b > only_b.loc[b, "AAA"]
    # and it is exactly additive, not merely larger
    assert on_b == pytest.approx(only_a.loc[b, "AAA"] + only_b.loc[b, "AAA"])


def test_two_events_on_the_same_day_sum():
    """`np.add.at` rather than fancy-index assignment: plain assignment keeps only the last."""
    idx = _grid(100)
    out = decay_events(_events([(idx[5], "AAA", 1.0), (idx[5], "AAA", 2.0)]), idx, 21,
                       magnitude_col="magnitude")
    assert out.loc[idx[5], "AAA"] == pytest.approx(3.0)


# ------------------------------------------------------------------- point-in-time edges

def test_a_non_trading_day_event_lands_on_the_next_trading_day():
    """Rounding a weekend filing BACKWARDS onto Friday would be a look-ahead: Saturday's news
    could not be acted on until Monday."""
    idx = _grid(60, start="2024-01-01")                    # business days from a Monday
    saturday = pd.Timestamp("2024-01-13")
    monday = pd.Timestamp("2024-01-15")
    out = decay_events(_events([(saturday, "AAA", 1.0)]), idx, 21, magnitude_col="magnitude")
    assert out.loc[monday, "AAA"] == pytest.approx(1.0)
    assert out["AAA"].loc[:pd.Timestamp("2024-01-12")].isna().all()


def test_events_after_the_grid_are_dropped_not_clamped():
    """Clamping a future filing onto the last day would inject information the grid has not
    reached. The ticker simply has no event yet, so its whole column stays NaN."""
    idx = _grid(50)
    out = decay_events(_events([(idx[-1] + pd.Timedelta(days=90), "AAA", 1.0)]), idx, 21,
                       magnitude_col="magnitude")
    assert out.empty or "AAA" not in out.columns or out["AAA"].isna().all()


def test_nan_magnitude_keeps_the_event_at_unit_weight():
    """An event whose SIZE is unknown still happened; dropping the row would erase it."""
    idx = _grid(100)
    out = decay_events(_events([(idx[5], "AAA", np.nan)]), idx, 21, magnitude_col="magnitude")
    assert out.loc[idx[5], "AAA"] == pytest.approx(1.0)


def test_empty_and_invalid_inputs():
    idx = _grid(50)
    assert decay_events(_events([]), idx, 21, magnitude_col="magnitude").empty
    assert decay_events(None, idx, 21).empty
    with pytest.raises(ValueError, match="halflife"):
        decay_events(_events([(idx[0], "AAA", 1.0)]), idx, 0, magnitude_col="magnitude")


# --------------------------------------------------------------------------- performance

def test_full_grid_decays_in_seconds_not_minutes():
    """500 tickers x ~15 trading years. The recursion is O(days x tickers); the rejected
    per-event outer product is O(events x days x tickers) and does not fit in memory."""
    idx = _grid(3800)
    rng = np.random.default_rng(0)
    tickers = [f"T{i:03d}" for i in range(500)]
    ev = pd.DataFrame({
        "date": idx[rng.integers(0, len(idx), 20_000)],
        "ticker": rng.choice(tickers, 20_000),
        "magnitude": rng.uniform(0.5, 5.0, 20_000)})
    t0 = time.perf_counter()
    out = decay_events(ev, idx, 63, magnitude_col="magnitude")
    elapsed = time.perf_counter() - t0
    assert out.shape == (3800, 500)
    assert elapsed < 15.0, f"decay took {elapsed:.1f}s on the full grid"


# ------------------------------------------------------------------- panel emission map

def _fields_and_peers():
    idx = _grid(40)
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    rng = np.random.default_rng(7)
    fdf = pd.DataFrame(rng.normal(size=(len(idx), len(tickers))), index=idx, columns=tickers)
    # `peer_relative` expects {ticker: {peer: weight}} -- a weighted basket, self excluded.
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}
    return {"sig": fdf}, peers


def test_default_emission_is_unchanged_two_legs_no_raw():
    """The regression guard for D13: every one of the thirteen existing builders passes no map
    and must keep its exact columns."""
    fields, peers = _fields_and_peers()
    cols = set(build_peer_relative_panel(fields, peers).columns)
    assert cols == {"date", "ticker", "f_sig_vs_peers", "f_sig_xs"}


def test_raw_emits_one_column_not_three():
    fields, peers = _fields_and_peers()
    cols = set(build_peer_relative_panel(fields, peers, emission={"sig": "raw"}).columns)
    assert cols == {"date", "ticker", "f_sig"}


def test_raw_plus_xs_and_raw_plus_peers():
    fields, peers = _fields_and_peers()
    xs = set(build_peer_relative_panel(fields, peers, emission={"sig": "raw+xs"}).columns)
    pr = set(build_peer_relative_panel(fields, peers, emission={"sig": "raw+peers"}).columns)
    assert xs == {"date", "ticker", "f_sig", "f_sig_xs"}
    assert pr == {"date", "ticker", "f_sig", "f_sig_vs_peers"}


def test_the_raw_leg_carries_the_untouched_value():
    """Not winsorized, not clipped: that treatment belongs to the z-score. A conviction weight
    is a quantity in its own units and trimming it destroys what it was emitted for."""
    fields, peers = _fields_and_peers()
    fields["sig"].iloc[0, 0] = 999.0                      # a value winsorize_xs would trim
    panel = build_peer_relative_panel(fields, peers, emission={"sig": "raw"})
    got = panel.loc[panel["ticker"] == "AAA", "f_sig"].iloc[0]
    assert got == pytest.approx(999.0)
    assert panel["f_sig"].dtype == np.float32          # float32 like every other leg


def test_a_field_left_out_of_the_map_keeps_the_default():
    fields, peers = _fields_and_peers()
    fields["other"] = fields["sig"] * 2
    cols = set(build_peer_relative_panel(fields, peers, emission={"sig": "raw"}).columns)
    assert cols == {"date", "ticker", "f_sig", "f_other_vs_peers", "f_other_xs"}


def test_a_bad_or_stray_emission_declaration_raises():
    """A silent no-op here is exactly the drift the parts registry was built to end."""
    fields, peers = _fields_and_peers()
    with pytest.raises(ValueError, match="emission mode"):
        build_peer_relative_panel(fields, peers, emission={"sig": "peers"})
    with pytest.raises(KeyError, match="not present"):
        build_peer_relative_panel(fields, peers, emission={"nope": "raw"})


def test_decayed_signal_survives_peer_relativization_where_a_flag_does_not():
    """The reason the transform exists, end to end.

    Same events, two encodings. As a 0/1 FLAG the panel is ~all-zeros, every peer basket has
    zero dispersion, and `peer_relative` returns NaN by policy -- so the feature is absent, not
    weak. DECAYED, the tickers sit at different points on their own decay curves, the basket has
    real dispersion every day, and the feature exists.

    Every ticker gets an event on purpose: `peer_relative` needs `min_peers=3` peers WITH DATA
    on the date, so a universe where only some names are ever eventful would come back NaN for
    a reason that has nothing to do with decay, and the comparison would prove nothing.
    """
    idx = _grid(250)
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}
    ev = _events([(idx[10 + 12 * i], t, 1.0) for i, t in enumerate(tickers)])

    flag = pd.DataFrame(0.0, index=idx, columns=tickers)
    for d, t, _ in ev.itertuples(index=False):
        flag.loc[d, t] = 1.0
    flag_nan = build_peer_relative_panel({"x": flag}, peers)["f_x_vs_peers"].isna().mean()

    decayed = decay_events(ev, idx, 63, magnitude_col="magnitude").reindex(columns=tickers)
    dec_nan = build_peer_relative_panel({"x": decayed}, peers)["f_x_vs_peers"].isna().mean()

    assert flag_nan > 0.95, f"expected the flag to be ~all-NaN, got {flag_nan:.3f}"
    assert dec_nan < 0.35, f"decay must recover coverage, got {dec_nan:.3f}"
