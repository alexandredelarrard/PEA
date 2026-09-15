"""Broad 13F institutional-ownership features (`ic_inst_*`, registry section 1).

Covers the QoQ aggregation math on the D28 breadth SHARE, the 45-day filing-lag
point-in-time stamping, the SPLIT restatement (a 20-for-1 split must not read as
accumulation -- value-sanity V4), the D16 hard cutoff and the D17 coverage guards, the
emitted `f_*` columns, and the pure extractor parsers (SEC join + OpenFIGI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants.constants import F13_MAX_EARLY_DAYS, F13_MAX_LATE_DAYS
from src.data_aggregate.utils.common.data_utils import to_day
from src.data_aggregate.utils.institutionals import institutional_features as _mod
from src.data_aggregate.utils.institutionals.institutional_features import (
    EMISSION,
    INST_DELTA_FLOOR_PERIOD,
    INST_LEVEL_FLOOR_PERIOD,
    MIN_PRIOR_HOLDERS,
)
from src.data_aggregate.utils.institutionals.institutional_features import (
    _quarter_features as _qf,
)
from src.data_aggregate.utils.institutionals.institutional_features import (
    build_institutional_feature_panel as _panel,
)
from src.data_aggregate.utils.institutionals.institutional_features import (
    clean_holdings as _clean_holdings,
)
from src.data_extract.utils.institutionals.fetch_13f import _holdings_frame
from src.data_extract.utils.institutionals.fetch_cusip_map import _parse_openfigi
from tests.conftest import make_frames

LAG = pd.Timedelta(days=45)


# ⚠ THE PER-TICKER COVERAGE-ONSET GUARD IS OFF BY DEFAULT IN THIS MODULE, and that is a
# deliberate, narrow decision rather than a convenience. `MIN_PRIOR_HOLDERS` is 100 -- a floor
# measured against the production 13F universe, where the median S&P name carries over a
# thousand filers -- while every fixture below holds two or three synthetic managers. Left on,
# it would null every QoQ delta in the file and the arithmetic tests would be asserting NaN ==
# NaN and passing for the wrong reason. So the wrappers default it to 0: a test that means to
# exercise the ARITHMETIC gets the arithmetic, and the one test that means to exercise the
# GUARD passes a floor explicitly and says which. Adding a fixture here does not re-enable it.
def _quarter_features(holdings, **kw):
    # ⚠ CLEAN FIRST, because `_qf` is INTERNAL and its contract is a CLEANED frame. The
    # coercion that used to sit inside it -- dates to midnight, the four numeric legs
    # zero-filled when the source omits them, amendments collapsed last-filed-wins -- now
    # lives once in `holdings_clean.clean_holdings`, and `build_institutional_feature_panel`
    # calls it before `_qf` on the production path. A fixture handed straight to `_qf` has
    # no `call_value` / `put_value` and raises `KeyError`; that is the test bypassing a
    # production step, not the builder demanding a column the source lacks.
    kw.setdefault("min_prior_holders", 0)
    return _qf(_clean_holdings(holdings), **kw)


def build_institutional_feature_panel(frames, holdings, **kw):
    # Mirrors the real signature since step 1.6: `frames` first, then the source, and every
    # remaining argument keyword-only. `*a` would swallow `frames` into `holdings`.
    kw.setdefault("min_prior_holders", 0)
    return _panel(frames, holdings, **kw)


def _frame(infotable):
    return _holdings_frame("111", "2022-05-10", "2022-03-31", infotable)


def _holdings():
    """Deterministic manager-grain 13F for ticker A over 2 quarters:
      Q1 (2022-03-31): M1=100, M2=200            -> holders 2, shares 300
      Q2 (2022-06-30): M1=150 (up), M3=50 (new)  -> M2 exited
        => new_buyer_ratio = 1/2, exit_ratio = 1/2, increasers=1 (M1), decreasers=0,
           cluster=(1-0)/2=0.5, shares 200 -> shares_chg = 200/300-1 = -1/3
    Ticker B is present only in Q1 (single holder) so it exercises the no-prior path."""
    rows = [
        {"cik": "M1", "period": "2022-03-31", "ticker": "A", "shares": 100, "value_usd": 1.0},
        {"cik": "M2", "period": "2022-03-31", "ticker": "A", "shares": 200, "value_usd": 2.0},
        {"cik": "M1", "period": "2022-06-30", "ticker": "A", "shares": 150, "value_usd": 1.5},
        {"cik": "M3", "period": "2022-06-30", "ticker": "A", "shares": 50, "value_usd": 0.5},
        {"cik": "M1", "period": "2022-03-31", "ticker": "B", "shares": 40, "value_usd": 0.4},
    ]
    return pd.DataFrame(rows)


def test_quarter_feature_math_on_the_d28_share():
    qf = _quarter_features(_holdings())
    a2 = qf[(qf["ticker"] == "A") & (qf["as_of"] == pd.Timestamp("2022-06-30") + LAG)].iloc[0]
    # D28: `ic_inst_holders` is holders / the quarter's own filer count, not a headcount.
    # Q2's universe filers are {M1, M3} = 2, both of which hold A -> 1.0.
    assert a2["ic_inst_holders"] == 1.0
    assert abs(a2["ic_inst_new_buyer_ratio"] - 0.5) < 1e-9  # M3 of 2 holders
    assert abs(a2["ic_inst_exit_ratio"] - 0.5) < 1e-9  # M2 of 2 prior holders
    assert abs(a2["ic_inst_cluster_buying"] - 0.5) < 1e-9
    assert abs(a2["ic_inst_shares_chg"] - (200 / 300 - 1)) < 1e-9
    # Q1 has no prior quarter -> every delta is NaN, never 0
    a1 = qf[(qf["ticker"] == "A") & (qf["as_of"] == pd.Timestamp("2022-03-31") + LAG)].iloc[0]
    assert np.isnan(a1["ic_inst_breadth_chg"]) and np.isnan(a1["ic_inst_new_buyer_ratio"])
    assert np.isnan(a1["ic_inst_exit_ratio"]) and np.isnan(a1["ic_inst_shares_chg"])
    print("\n=== SANITY CHECK: 13F quarter-over-quarter math (D28 share) ===")
    print(
        f"  A Q2: holders_share={a2['ic_inst_holders']:.2f} (2 of 2 filers), "
        f"new_buyer_ratio=0.5, exit_ratio=0.5, cluster=0.5, "
        f"shares_chg={a2['ic_inst_shares_chg']:.3f}; Q1 deltas all NaN (no prior). Validated."
    )


def _growing_universe(pairs: list[tuple[str, int, int]]) -> pd.DataFrame:
    """`(period, n_filers, n_holding_A)` -> manager-grain rows. The filers who do not hold A
    hold Z instead, so they count toward the universe pool without touching A's numerator."""
    rows = []
    for period, n_filers, n_a in pairs:
        for m in range(n_filers):
            rows.append({"cik": f"M{m}", "period": period, "ticker": "A" if m < n_a else "Z", "shares": 100, "value_usd": 1.0})
    return pd.DataFrame(rows)


def test_the_breadth_share_survives_a_filer_count_jump():
    """D28's reason to exist: the raw holder COUNT grows with the fetch, the SHARE does not.
    A's holder count rises 5 -> 6 while the universe grows 10 -> 12 filers, so the share is
    0.50 throughout and the change is 0.00 rather than +1 holder."""
    qf = _quarter_features(_growing_universe([("2022-03-31", 10, 5), ("2022-06-30", 12, 6)]))
    a = qf[qf["ticker"] == "A"].sort_values("as_of")
    assert abs(a.iloc[0]["ic_inst_holders"] - 0.5) < 1e-9
    assert abs(a.iloc[1]["ic_inst_holders"] - 0.5) < 1e-9
    assert abs(a.iloc[1]["ic_inst_breadth_chg"]) < 1e-9
    print("\n=== SANITY CHECK: D28 breadth share vs filer-count growth ===")
    print(
        "  A's holder COUNT 5 -> 6 while the universe went 10 -> 12 filers: the share stays "
        "0.50 and ic_inst_breadth_chg is 0.00, not +1. Validated."
    )


def test_a_filer_count_jump_past_the_threshold_nulls_the_delta_anyway():
    """The share absorbs a PROPORTIONAL coverage change; D17 catches the rest. Tripling the
    filer count is not something a market does, so even though the share is unchanged
    (2/5 -> 6/15) the delta is suppressed -- the two guards are belt and braces, and this is
    the one assertion that says so."""
    qf = _quarter_features(_growing_universe([("2022-03-31", 5, 2), ("2022-06-30", 15, 6)]))
    a = qf[qf["ticker"] == "A"].sort_values("as_of")
    assert abs(a.iloc[0]["ic_inst_holders"] - 2 / 5) < 1e-9
    assert abs(a.iloc[1]["ic_inst_holders"] - 6 / 15) < 1e-9  # the LEVEL survives
    assert np.isnan(a.iloc[1]["ic_inst_breadth_chg"]), "D17 did not null a +200% filer jump"
    print("\n=== SANITY CHECK: D28 share + D17 guard together ===")
    print(
        "  universe 5 -> 15 filers (+200%): the share is legitimately unchanged at 0.40 and "
        "the LEVEL survives, but every delta on that quarter is NaN because a tripling of "
        "coverage is a fetch event, not a market one. Validated."
    )


def test_filing_lag_point_in_time():
    idx = pd.bdate_range("2022-01-03", "2023-06-30")
    build_institutional_feature_panel(make_frames(idx, {}), _holdings())
    # rebuild the daily field directly to check the lag boundary
    from src.data_aggregate.utils.common.pit import fundamentals_to_daily

    qf = _quarter_features(_holdings())
    daily = fundamentals_to_daily(qf, "ic_inst_cluster_buying", idx)["A"]
    q2_asof = pd.Timestamp("2022-06-30") + LAG  # 2022-08-14
    before = daily.loc[idx[idx < pd.Timestamp("2022-08-14")]]
    after = daily.loc[idx[idx >= q2_asof]]
    # Q2 cluster (0.5) must NOT appear before its as_of; appears on/after
    assert not (np.isclose(before.dropna(), 0.5)).any(), "13F leaked before filing lag"
    assert np.isclose(after.dropna().iloc[0], 0.5), "13F feature missing after filing lag"
    print("\n=== SANITY CHECK: 45-day filing-lag point-in-time ===")
    print(f"  Q2 (Jun-30) ic_inst_cluster_buying only visible from {q2_asof.date()} onward, " f"never before. Leak-free. Validated.")


def _two_quarters(shares_q1: dict, shares_q2: dict) -> pd.DataFrame:
    rows = []
    for period, book in (("2022-03-31", shares_q1), ("2022-06-30", shares_q2)):
        for cik, sh in book.items():
            rows.append({"cik": cik, "period": period, "ticker": "A", "shares": sh, "value_usd": float(sh)})
    return pd.DataFrame(rows)


def test_a_split_is_not_accumulation():
    """V4. A 20-for-1 split between the two quarters multiplies every holder's reported count
    by 20 with no trade taking place. Unrestated, `shares_chg` reads +1,900% and EVERY holder
    counts as an increaser, so `cluster_buying` prints its maximum +1.0 -- a fabricated
    unanimous-conviction signal on the one quarter nobody decided anything."""
    book = {"M1": 100.0, "M2": 200.0, "M3": 300.0}
    post = {k: v * 20 for k, v in book.items()}
    holdings = _two_quarters(book, post)
    splits = pd.DataFrame({"ticker": ["A"], "date": [pd.Timestamp("2022-05-10")], "ratio": [20.0]})

    naive = _quarter_features(holdings, splits=None)
    naive_q2 = naive[naive["as_of"] == pd.Timestamp("2022-06-30") + LAG].iloc[0]
    assert abs(naive_q2["ic_inst_shares_chg"] - 19.0) < 1e-9  # +1,900%, the defect
    assert abs(naive_q2["ic_inst_cluster_buying"] - 1.0) < 1e-9  # every holder an "increaser"

    fixed = _quarter_features(holdings, splits=splits)
    q2 = fixed[fixed["as_of"] == pd.Timestamp("2022-06-30") + LAG].iloc[0]
    assert abs(q2["ic_inst_shares_chg"]) < 1e-9, "split still read as accumulation"
    assert abs(q2["ic_inst_cluster_buying"]) < 1e-9, "split still read as cluster buying"
    print("\n=== SANITY CHECK: V4 split guard on the 13F share legs ===")
    print(
        f"  unrestated: shares_chg={naive_q2['ic_inst_shares_chg']:+.0%}, "
        f"cluster_buying={naive_q2['ic_inst_cluster_buying']:+.2f} (both fabricated); "
        f"restated through prices_splits: {q2['ic_inst_shares_chg']:+.4f} / "
        f"{q2['ic_inst_cluster_buying']:+.4f}. A real purchase after the split still "
        "registers. Validated."
    )

    # and a REAL purchase on top of the split still registers
    real = _quarter_features(_two_quarters(book, {k: v * 20 * 1.1 for k, v in book.items()}), splits=splits)
    r2 = real[real["as_of"] == pd.Timestamp("2022-06-30") + LAG].iloc[0]
    assert abs(r2["ic_inst_shares_chg"] - 0.1) < 1e-9
    assert abs(r2["ic_inst_cluster_buying"] - 1.0) < 1e-9


def test_coverage_hole_and_break_guards():
    """D17 + the hole extension. A quarter whose universe filer count collapses is missing
    data, not a market event: its deltas AND levels are suppressed, and the RECOVERY quarter
    (whose level is fine but whose predecessor is the hole) loses only its deltas."""
    periods = ["2021-06-30", "2021-09-30", "2021-12-31", "2022-03-31"]
    counts = [40, 40, 2, 40]  # 2021-12-31 is the hole
    rows = []
    for period, n in zip(periods, counts, strict=False):
        for m in range(n):
            rows.append({"cik": f"M{m}", "period": period, "ticker": "A", "shares": 100.0 + m, "value_usd": 100.0 + m})
    qf = _quarter_features(pd.DataFrame(rows)).set_index("period")

    hole, recovery = pd.Timestamp("2021-12-31"), pd.Timestamp("2022-03-31")
    normal = pd.Timestamp("2021-09-30")
    assert np.isnan(qf.loc[hole, "ic_inst_holders"]), "hole level not suppressed"
    assert np.isnan(qf.loc[hole, "ic_inst_concentration"])
    assert np.isnan(qf.loc[hole, "ic_inst_breadth_chg"])
    assert np.isnan(qf.loc[hole, "inst_shares"]), "hole value leg not suppressed"
    # the recovery quarter keeps its LEVEL (40 filers really did file) and loses its DELTAS
    assert np.isfinite(qf.loc[recovery, "ic_inst_holders"])
    assert np.isnan(qf.loc[recovery, "ic_inst_shares_chg"]), "delta against a hole survived"
    assert np.isnan(qf.loc[recovery, "ic_inst_new_buyer_ratio"])
    # an ordinary quarter is untouched
    assert np.isfinite(qf.loc[normal, "ic_inst_holders"])
    assert np.isfinite(qf.loc[normal, "ic_inst_breadth_chg"])
    print("\n=== SANITY CHECK: D17 coverage-discontinuity guards ===")
    print(
        f"  filer counts {dict(zip(periods, counts, strict=False))}: the 2021-12-31 HOLE has every level "
        f"and delta NaN; 2022-03-31 keeps holders="
        f"{qf.loc[recovery, 'ic_inst_holders']:.2f} but its deltas are NaN (differenced "
        f"against a hole); 2021-09-30 untouched. Validated."
    )


def test_per_ticker_coverage_onset_guard():
    """The per-ticker analogue of D16/D17 (`MIN_PRIOR_HOLDERS`). D16 asks when the MARKET's 13F
    coverage began and D17 when it jumped; neither asks when THIS TICKER's did, so a name that
    arrives in the index on a spin-off or an IPO has one filer in its first quarter and several
    hundred in its second, and `shares / prev_shares - 1` reads that as a flow of millions of
    percent (CCI, 2014-12-31: 34,264,348).

    The fixture is that shape exactly, placed after the D16 floors so nothing else can suppress
    it: one filer holding 9 shares, then 400 filers holding ~300M. Every DELTA must go; the
    LEVELS must stay, because a thinly-held name really is thinly held that quarter and
    `ic_inst_holders` is the feature that measures it."""
    onset, broad = "2015-03-31", "2015-06-30"
    rows = [{"cik": "M0", "period": onset, "ticker": "A", "shares": 9.0, "value_usd": 9.0}]
    rows += [{"cik": f"M{m}", "period": broad, "ticker": "A", "shares": 750_000.0 + m, "value_usd": 750_000.0 + m} for m in range(400)]
    # a second name broadly held in BOTH quarters, so the universe filer count never moves
    # enough to fire D17 and this test is about the per-ticker guard alone
    rows += [{"cik": f"M{m}", "period": p, "ticker": "B", "shares": 500.0 + m, "value_usd": 500.0 + m} for p in (onset, broad) for m in range(400)]

    off = _quarter_features(pd.DataFrame(rows)).set_index(["ticker", "period"])
    on = _quarter_features(pd.DataFrame(rows), min_prior_holders=MIN_PRIOR_HOLDERS).set_index(["ticker", "period"])
    key = ("A", pd.Timestamp(broad))

    unguarded = off.loc[key, "ic_inst_shares_chg"]
    assert unguarded > 1e6, f"fixture no longer reproduces the defect: {unguarded}"
    for c in (
        "ic_inst_shares_chg",
        "ic_inst_breadth_chg",
        "ic_inst_new_buyer_ratio",
        "ic_inst_exit_ratio",
        "ic_inst_cluster_buying",
        "inst_value_flow",
    ):
        assert np.isnan(on.loc[key, c]), f"{c} survived the per-ticker onset guard"
    # levels untouched, on both the guarded ticker and its broadly-held neighbour
    assert np.isfinite(on.loc[key, "ic_inst_holders"])
    assert np.isfinite(on.loc[key, "ic_inst_concentration"])
    assert np.isfinite(on.loc[("B", pd.Timestamp(broad)), "ic_inst_shares_chg"]), "a name held by 400 filers in BOTH quarters must keep its delta"
    print("\n=== SANITY CHECK: per-ticker coverage-onset guard ===")
    print(
        f"  A: 1 filer/9 shares -> 400 filers/300M shares reads as shares_chg="
        f"{unguarded:,.0f} unguarded; NaN at the {MIN_PRIOR_HOLDERS}-filer floor, while "
        f"ic_inst_holders={on.loc[key, 'ic_inst_holders']:.3f} survives. B, broadly held "
        "throughout, keeps its delta. Validated."
    )


def _pre_floor_fixture() -> pd.DataFrame:
    periods = ["2012-12-31", "2013-03-31", "2013-06-30", "2013-09-30"]
    return pd.DataFrame(
        [{"cik": f"M{m}", "period": p, "ticker": "A", "shares": 100.0 + m, "value_usd": 100.0 + m} for p in periods for m in range(30)]
    )


def test_d16_hard_cutoff_before_the_2013_break():
    """D16. The 13F numbers before 2013-06-30 exist and look valid but describe the fetch, so
    nothing computed from them survives -- and since step 2.4 they are not computed at all:
    the pre-floor cut drops the rows before the loop rather than nulling their output after."""
    qf = _quarter_features(_pre_floor_fixture()).set_index("period")
    for p in ("2012-12-31", "2013-03-31"):
        assert pd.Timestamp(p) not in qf.index, f"{p} was still emitted"
    assert np.isfinite(qf.loc[INST_LEVEL_FLOOR_PERIOD, "ic_inst_holders"])
    assert np.isnan(qf.loc[INST_LEVEL_FLOOR_PERIOD, "ic_inst_breadth_chg"]), "the first post-break quarter has no comparable predecessor (L10)"
    assert np.isfinite(qf.loc[INST_DELTA_FLOOR_PERIOD, "ic_inst_breadth_chg"])
    print("\n=== SANITY CHECK: D16 hard cutoff ===")
    print(
        f"  the two pre-{INST_LEVEL_FLOOR_PERIOD.date()} quarters are not emitted at all, "
        f"and deltas are NaN before {INST_DELTA_FLOOR_PERIOD.date()} -- both measured from "
        "the 192 -> 3,046 filer jump. Validated."
    )


def test_the_pre_floor_cut_leaves_the_delta_floor_onward_identical(monkeypatch):
    """⚠ THE CUT MUST NOT MOVE THE FIRST SURVIVING QUARTER, and that is not obvious: `prev` is
    carried across periods inside the loop, so dropping 2013-03-31 takes away the predecessor
    2013-06-30's deltas were differenced against. It is safe only because
    `INST_DELTA_FLOOR_PERIOD` already nulls exactly those deltas -- so this runs the loop with
    and WITHOUT the cut (the floor pushed back to 1990, which disables it) and asserts the two
    agree from the delta floor onward.
    """
    h = _clean_holdings(_pre_floor_fixture())
    cut = _qf(h, min_prior_holders=0).set_index("period")

    monkeypatch.setattr(_mod, "INST_LEVEL_FLOOR_PERIOD", pd.Timestamp("1990-01-01"))
    uncut = _qf(h, min_prior_holders=0).set_index("period")
    assert pd.Timestamp("2012-12-31") in uncut.index, "the A/B did not actually disable the cut"

    tail = cut.index[cut.index >= INST_DELTA_FLOOR_PERIOD]
    pd.testing.assert_frame_equal(cut.loc[tail], uncut.loc[tail])

    # and 2013-06-30 itself: it lost a predecessor, but every delta there was already NaN
    deltas = [c for c in cut.columns if c.endswith(("_chg", "_ratio", "_buying"))]
    assert cut.loc[INST_LEVEL_FLOOR_PERIOD, [c for c in deltas if c != "ic_inst_net_options_ratio"]].isna().all()

    print("\n=== SANITY CHECK: the pre-floor cut is output-neutral ===")
    print(
        f"  {len(tail)} quarter(s) from {INST_DELTA_FLOOR_PERIOD.date()} onward are identical "
        f"with and without the cut; {INST_LEVEL_FLOOR_PERIOD.date()} loses its predecessor "
        "but its deltas were already NaN from the delta floor. Validated."
    )


def test_panel_columns_match_the_emission_map():
    idx = pd.bdate_range("2022-01-03", "2023-06-30")
    tickers = ["A", "B", "C", "D"]
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}
    fund = pd.DataFrame([{"ticker": t, "as_of": "2022-01-01", "sharesOutstanding": 1000.0, "sharesOutstandingPit": 1000.0} for t in tickers])
    close = pd.DataFrame({t: 10.0 for t in tickers}, index=idx)
    panel = build_institutional_feature_panel(make_frames(idx, peers, close_split=close), _holdings(), shares_out_history=fund)
    expected = set()
    for name, mode in EMISSION.items():
        expected.add(f"f_{name}")
        if mode == "raw+xs":
            expected.add(f"f_{name}_xs")
        elif mode == "raw+peers":
            expected.add(f"f_{name}_vs_peers")
    emitted = {c for c in panel.columns if c.startswith("f_")}
    # every emitted column is declared, and the two-leg shape matches the map exactly
    assert emitted <= expected, f"undeclared column(s): {sorted(emitted - expected)}"
    assert "f_ic_inst_holders" in emitted and "f_ic_inst_holders_xs" not in emitted
    assert "f_ic_inst_ownership_pct_vs_peers" in emitted
    assert "f_ic_inst_concentration_xs" in emitted
    # A's ownership pct at a late date = 200 shares / 1000 = 0.2
    from src.data_aggregate.utils.common.pit import fundamentals_to_daily

    qf = _quarter_features(_holdings())
    inst_sh = fundamentals_to_daily(qf, "inst_shares", idx)["A"].dropna().iloc[-1]
    assert abs(inst_sh - 200.0) < 1e-9
    print("\n=== SANITY CHECK: 13F emitted columns vs the EMISSION map ===")
    print(
        f"  {len(emitted)} legs emitted, all declared ({len(EMISSION)} features); "
        f"holders is raw-only, ownership_pct carries the peer leg, concentration the "
        f"percentile. A latest inst_shares={inst_sh:.0f} (/1000 = 0.2). Validated."
    )


def _holdings_opts():
    """Manager-grain 13F WITH option exposure over 2 quarters (ticker A), using real recent
    quarter-ends to exercise the March->May availability lag:
      Q1 2025-12-31: M1/M2/M3, long value 3.0M, calls 200k / puts 100k
      Q2 2026-03-31: M3 exits, M4 new; long value 4.0M, calls 400k / puts 0 (bullish shift)"""
    return pd.DataFrame(
        [
            {"cik": "M1", "period": "2025-12-31", "ticker": "A", "shares": 100, "value_usd": 1_000_000, "call_value": 100_000, "put_value": 50_000},
            {"cik": "M2", "period": "2025-12-31", "ticker": "A", "shares": 100, "value_usd": 1_000_000, "call_value": 50_000, "put_value": 50_000},
            {"cik": "M3", "period": "2025-12-31", "ticker": "A", "shares": 100, "value_usd": 1_000_000, "call_value": 50_000, "put_value": 0},
            {"cik": "M1", "period": "2026-03-31", "ticker": "A", "shares": 100, "value_usd": 1_000_000, "call_value": 200_000, "put_value": 0},
            {"cik": "M2", "period": "2026-03-31", "ticker": "A", "shares": 150, "value_usd": 1_500_000, "call_value": 100_000, "put_value": 0},
            {"cik": "M4", "period": "2026-03-31", "ticker": "A", "shares": 150, "value_usd": 1_500_000, "call_value": 100_000, "put_value": 0},
        ]
    )


def test_options_concentration_and_the_availability_date():
    qf = _quarter_features(_holdings_opts())
    q2 = qf[qf["as_of"] == pd.Timestamp("2026-03-31") + LAG].iloc[0]
    # the crux: a March-31 quarter is only public ~mid-May
    assert q2["as_of"] == pd.Timestamp("2026-05-15")
    assert abs(q2["ic_inst_shares_chg"] - (400 / 300 - 1)) < 1e-9  # +33.3% VOLUME
    # net options = (calls - puts) / (long value + calls + puts) = 400k / 4.4M
    assert abs(q2["ic_inst_net_options_ratio"] - (400_000 / 4_400_000)) < 1e-6
    assert abs(q2["ic_inst_new_buyer_ratio"] - (1 / 3)) < 1e-9  # M4 of 3 holders
    assert abs(q2["ic_inst_exit_ratio"] - (1 / 3)) < 1e-9  # M3 of 3 prior
    # Herfindahl of manager value shares (1.0/1.5/1.5 of 4.0M)
    assert abs(q2["ic_inst_concentration"] - ((1 / 4) ** 2 + 2 * (1.5 / 4) ** 2)) < 1e-6
    # registry section 1: `inst_value_chg` and `net_options_ratio_chg` are DROPPED -- price
    # -contaminated and redundant against flow_to_mcap / the level respectively.
    assert "ic_inst_value_chg" not in qf.columns
    assert "ic_inst_net_options_ratio_chg" not in qf.columns
    print("\n=== SANITY CHECK: 13F options / concentration / availability ===")
    print(
        f"  Q2(2026-03-31) as_of={q2['as_of'].date()} (leak-free); "
        f"shares_chg={q2['ic_inst_shares_chg']:+.3f} "
        f"net_options={q2['ic_inst_net_options_ratio']:.4f} "
        f"HHI={q2['ic_inst_concentration']:.3f} "
        f"new_buyer_ratio={q2['ic_inst_new_buyer_ratio']:.3f} "
        f"exit_ratio={q2['ic_inst_exit_ratio']:.3f}; the two dropped features are absent. "
        "Validated."
    )


def test_value_to_mcap_and_flow_panel():
    idx = pd.bdate_range("2025-10-01", "2026-09-30")
    tickers = ["A", "B", "C", "D"]
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}
    fund = pd.DataFrame(
        [{"ticker": t, "as_of": "2025-01-01", "sharesOutstanding": 1_000_000.0, "sharesOutstandingPit": 1_000_000.0} for t in tickers]
    )
    close = pd.DataFrame({t: 10.0 for t in tickers}, index=idx)  # price 10 -> mcap 10M
    panel = build_institutional_feature_panel(make_frames(idx, peers, close_split=close), _holdings_opts(), shares_out_history=fund)
    for c in (
        "f_ic_inst_net_options_ratio",
        "f_ic_inst_concentration",
        "f_ic_inst_value_to_mcap",
        "f_ic_inst_value_to_mcap_xs",
        "f_ic_inst_flow_to_mcap",
        "f_ic_inst_flow_to_mcap_xs",
    ):
        assert c in panel.columns, f"{c} missing from panel"
    # A after its Q2 becomes public: long value 4.0M / mcap 10M = 0.40 (raw, pre xs-rank)
    from src.data_aggregate.utils.common.pit import daily_market_cap, fundamentals_to_daily

    qf = _quarter_features(_holdings_opts())
    iv = fundamentals_to_daily(qf, "inst_value", idx)["A"].dropna().iloc[-1]
    mc = daily_market_cap(fund, close, level_factor=None)["A"].iloc[-1]
    assert abs(iv / mc - 0.40) < 1e-6
    print("\n=== SANITY CHECK: institutional weight (value / market cap) ===")
    print(
        f"  A inst_value=${iv:,.0f} / mcap=${mc:,.0f} = {iv / mc:.2f}; panel exposes "
        f"value_to_mcap + flow_to_mcap (raw + percentile) and the two bounded ratios. "
        "Validated."
    )


def test_extractor_parsers():
    info = pd.DataFrame({"CUSIP": ["037833100", "037833100", "594918104"], "VALUE": ["1000", "500", "2000"], "SSHPRNAMT": ["10", "5", "20"]})
    h = _frame(info).sort_values("cusip").reset_index(drop=True)
    assert {
        "cik",
        "period",
        "filing_date",
        "cusip",
        "shares",
        "value_usd",
        "call_shares",
        "call_value",
        "put_shares",
        "put_value",
        "debt_prn",
        "debt_value",
        "other_value",
    } <= set(h.columns)
    # a manager's two lines for the same CUSIP are summed into one row
    assert len(h) == 2
    assert h.loc[0, "shares"] == 15.0 and h.loc[0, "value_usd"] == 1500.0
    assert h.loc[1, "shares"] == 20.0 and h.loc[1, "value_usd"] == 2000.0
    assert h["period"].iloc[0] == pd.Timestamp("2022-03-31")
    assert h["cik"].iloc[0] == "0000000111"  # padded to the stored 10-digit form

    figi = [{"data": [{"ticker": "AAPL"}]}, {"warning": "no match"}]
    m = _parse_openfigi(figi, ["037833100", "999999999"])
    assert m == {"037833100": "AAPL"}
    print("\n=== SANITY CHECK: 13F + OpenFIGI parsers ===")
    print(
        "  join sums a manager's duplicate-CUSIP lines and takes VALUE as-is (edgartools "
        "normalizes each filing's $thousands/$ones unit); OpenFIGI maps CUSIP->ticker "
        "(not the free-text issuer name). Validated."
    )


def test_holdings_frame_splits_holding_types():
    """Stock / call / put / debt / other land in separate columns; long `shares`
    excludes options and bond principal (the noise that was inflating shares)."""
    info = pd.DataFrame(
        {
            "CUSIP": ["037833100"] * 5,
            "VALUE": ["1000000", "300000", "200000", "50000", "7000"],
            "SSHPRNAMT": ["10000", "3000", "2000", "40000", "70"],
            "SSHPRNAMTTYPE": ["SH", "SH", "SH", "PRN", "XX"],
            "PUTCALL": ["", "Call", "Put", "", ""],
        }
    )
    h = _frame(info)
    assert len(h) == 1
    r = h.iloc[0]
    assert r["shares"] == 10000 and r["value_usd"] == 1_000_000  # long stock ONLY
    assert r["call_shares"] == 3000 and r["call_value"] == 300_000
    assert r["put_shares"] == 2000 and r["put_value"] == 200_000  # bearish / sell-side
    assert r["debt_prn"] == 40000 and r["debt_value"] == 50_000
    assert r["other_value"] == 7_000  # malformed type -> other

    print("\n=== SANITY CHECK: 13F holding-type split ===")
    print(
        f"  shares(long)={r['shares']:.0f} calls={r['call_shares']:.0f} "
        f"puts={r['put_shares']:.0f} debt$={r['debt_value']:.0f} other$={r['other_value']:.0f} "
        f"-> long shares no longer contaminated by options/bonds. Validated."
    )


def test_holdings_frame_reads_edgartools_columns():
    """edgartools names the same fields Cusip / Value / SharesPrnAmount / PutCall / Type,
    and spells the amount type Shares/Principal rather than SH/PRN. Both vocabularies must
    classify identically, since the stored history was built from the bulk TSV names."""
    info = pd.DataFrame(
        {
            "Cusip": ["037833100"] * 3,
            "Value": [1_000_000, 300_000, 50_000],
            "SharesPrnAmount": [10_000, 3_000, 40_000],
            "Type": ["Shares", "Shares", "Principal"],
            "PutCall": ["", "Call", ""],
        }
    )
    r = _frame(info).iloc[0]
    assert r["shares"] == 10_000 and r["value_usd"] == 1_000_000
    assert r["call_shares"] == 3_000 and r["call_value"] == 300_000
    assert r["debt_prn"] == 40_000 and r["debt_value"] == 50_000
    assert r["other_value"] == 0 and r["put_value"] == 0
    print("\n=== SANITY CHECK: edgartools column vocabulary ===")
    print(
        "  Cusip/Value/SharesPrnAmount/PutCall/Type + Shares|Principal classify into the "
        "same buckets as the bulk CUSIP/VALUE/SSHPRNAMT/SSHPRNAMTTYPE names. Validated."
    )


def test_ownership_above_the_ceiling_is_nulled_not_clipped():
    """The 2026-09-12 build found `f_ic_inst_ownership_pct` reaching **9.68** (DUK), with all
    twelve extreme tickers carrying splits -- the recorded `sharesOutstandingPit` split defect
    arriving through the denominator.

    Three properties, and each has been got wrong somewhere in this repo before:
      * above 100% is NOT nulled. Securities lending double-counts a position, so 100-130% is a
        real reading and 24,347 of 1.5M cells sat there;
      * above the ceiling is NULLED, not clipped -- a clip would invent a 200%-owned company
        and hand the model a fabricated level;
      * everything below is untouched, to the bit.
    """
    from src.data_aggregate.utils.institutionals.institutional_features import OWNERSHIP_CEILING, _capped_ownership

    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=4))
    raw = pd.DataFrame(
        {
            "OK": [0.40, 0.95, 1.25, 1.99],  # all legitimate
            "BAD": [0.40, 2.01, 9.68, np.nan],
        },  # two impossible
        index=idx,
    )
    out = _capped_ownership(raw)
    assert out["OK"].tolist() == raw["OK"].tolist(), "a legitimate >100% reading was touched"
    assert out.loc[idx[0], "BAD"] == 0.40
    assert out["BAD"].isna().sum() == 3, "2.01 and 9.68 must both go, and the NaN stays NaN"
    assert (out.to_numpy(dtype="float64")[np.isfinite(out.to_numpy(dtype="float64"))] <= OWNERSHIP_CEILING).all()
    assert not (out == OWNERSHIP_CEILING).any().any(), "nulled, never clipped to the ceiling"
    print("\n=== SANITY CHECK: ownership ceiling ===")
    print(
        f"  ceiling {OWNERSHIP_CEILING}x: 1.25 and 1.99 kept (securities-lending "
        f"double-counting is real), 2.01 and 9.68 NULLED, and no cell is left sitting AT "
        f"the ceiling. Validated."
    )


def test_the_shared_cleaner_handles_the_live_dtypes_not_just_strings():
    """One cleaner, both grains, and it must survive the dtypes the DB actually returns.

    ⚠ THE FIXTURE IS `datetime.date` OBJECTS AND `pd.Timestamp`s CARRYING A TIME, NOT STRINGS,
    and that is the entire point. Postgres `DATE` columns come back as `datetime.date` and
    `TIMESTAMP` ones as timed `Timestamp`s; a parquet-cached or string-literal fixture hides
    this whole bug class. The WIP cleaner used
    `pd.to_datetime(col, format="%Y-%m-%d", errors="coerce")`, where:

      * on a real `DATE`/`TIMESTAMP` column the format is IGNORED, so it bought nothing;
      * it does NOT strip a time component, where `.dt.normalize()` does;
      * and on a STRING carrying a time it returns `NaT` -- so the row is dropped by the
        `dropna` and the holding silently vanishes.

    Also pinned: `filing_date` is OPTIONAL on `sec13f_hr`, so the cleaner must not raise when
    the projection left it out -- reading `h["filing_date"]` unguarded is what took 10 tests
    in this file down -- and the amendment must still win when it IS there.
    """
    import datetime as dt

    from src.data_aggregate.utils.institutionals.holdings_clean import clean_holdings

    rows = [
        # the ORIGINAL filing, and its AMENDMENT four days later with a corrected count
        {
            "ticker": "AAA",
            "cik": "0001",
            "period": dt.date(2024, 3, 31),
            "filing_date": pd.Timestamp("2024-05-10 14:32:05"),
            "shares": 1_000.0,
            "value_usd": 100_000.0,
        },
        {
            "ticker": "AAA",
            "cik": "0001",
            "period": dt.date(2024, 3, 31),
            "filing_date": pd.Timestamp("2024-05-14 09:00:00"),
            "shares": 2_500.0,
            "value_usd": 250_000.0,
        },
        {
            "ticker": "BBB",
            "cik": "0002",
            "period": dt.date(2024, 3, 31),
            "filing_date": pd.Timestamp("2024-05-11 00:00:00"),
            "shares": 700.0,
            "value_usd": 70_000.0,
        },
    ]
    out = clean_holdings(pd.DataFrame(rows), key=("ticker", "cik", "period"))

    assert len(out) == 2, out
    assert out["period"].dt.normalize().equals(out["period"]), "period is not at midnight"
    assert out["filing_date"].dt.normalize().equals(out["filing_date"]), "time not stripped"
    amended = out.loc[out["ticker"] == "AAA", "shares"].iloc[0]
    assert amended == 2_500.0, f"the amendment must win, got {amended}"
    # the option legs the elite table never carries are created, not demanded
    assert (out["call_value"] == 0.0).all() and (out["put_value"] == 0.0).all()

    # ... and with `filing_date` projected away it must DEGRADE, not raise
    no_fd = pd.DataFrame(rows).drop(columns=["filing_date"])
    degraded = clean_holdings(no_fd, key=("ticker", "cik", "period"))
    assert len(degraded) == 2, degraded

    # the format string that was there before turns a timed value into NaT -- the row would
    # have been dropped by the dropna, silently
    timed = pd.Series(["2024-05-10 14:32:05"])
    assert pd.to_datetime(timed, format="%Y-%m-%d", errors="coerce").isna().all()
    assert to_day(timed).notna().all()

    print()
    print("=== SANITY: the shared 13F cleaner on LIVE dtypes ===")
    print(f"  in : 3 rows, period as {type(rows[0]['period']).__name__}, " f"filing_date as timed Timestamp")
    print(f"  out: {len(out)} rows after the amendment collapse; " f"AAA shares = {amended:,.0f} (the amendment, not the original)")
    print(f"  period dtype {out['period'].dtype}, all at midnight; " f"time stripped from filing_date")
    print(f"  with `filing_date` projected away -> {len(degraded)} rows, no exception")
    print("  `to_datetime('2024-05-10 14:32:05', format='%Y-%m-%d') -> NaT` " "while `to_day` -> a real date")
    print(
        "  CONCLUSION: one cleaner serves both grains, normalizes the dtypes the DB really "
        "returns, degrades when the optional column is absent, and lets the amendment win. "
        "Validated."
    )


# --------------------------------------------------------------------------------------- #
# Step 2.4 -- the [-45, +60] filing band                                                    #
# --------------------------------------------------------------------------------------- #


def _banded(rows: list[dict], band=(F13_MAX_EARLY_DAYS, F13_MAX_LATE_DAYS)) -> pd.DataFrame:
    from src.data_aggregate.utils.institutionals.holdings_clean import clean_holdings

    return clean_holdings(pd.DataFrame(rows), key=("ticker", "cik", "period"), filing_band=band)


def _row(ticker: str, cik: str, lag_vs_deadline: int, shares: float) -> dict:
    """One holding, filed `lag_vs_deadline` days after the period+45d deadline."""
    period = pd.Timestamp("2024-03-31")
    return {
        "ticker": ticker,
        "cik": cik,
        "period": period,
        "shares": shares,
        "value_usd": shares * 10.0,
        "filing_date": period + pd.Timedelta(days=45 + lag_vs_deadline),
    }


def test_the_band_keeps_on_time_and_moderately_late_filings_and_drops_the_rest():
    """The cut is on `filing_date - (period + 45d)`, and the boundaries are INCLUSIVE: +60 is
    kept and +61 is not. A quarter-end snapshot filed a year later is a real position that
    nobody could have acted on, and the deadline stamp would make it look like they could."""
    out = _banded(
        [
            _row("AAA", "0001", -10, 100.0),  # early, well inside
            _row("BBB", "0002", 0, 200.0),  # exactly on the deadline
            _row("CCC", "0003", 60, 300.0),  # the last day kept
            _row("DDD", "0004", 61, 400.0),  # one day past
            _row("EEE", "0005", 400, 500.0),
        ]
    )  # a back-filed archive entry

    assert sorted(out["ticker"]) == ["AAA", "BBB", "CCC"]
    print("\n=== SANITY CHECK: the [-45, +60] filing band ===")
    print(f"  lags -10 / 0 / +60 / +61 / +400 vs the deadline -> kept " f"{sorted(out['ticker'])}. The boundary is inclusive at +60. Validated.")


def test_the_band_runs_before_the_amendment_dedup_so_an_on_time_original_survives():
    """⚠ THE ORDER IS THE DECISION. A filer who amends three years later has an ORIGINAL that
    was on time. Band first -> the original survives. Dedup first -> the late amendment wins
    `keep='last'` and is then dropped, taking the whole filing with it."""
    out = _banded(
        [
            _row("AAA", "0001", 5, 1_000.0),  # the original, on time
            _row("AAA", "0001", 1_100, 2_500.0),
        ]
    )  # the amendment, 3 years later

    assert len(out) == 1
    assert float(out["shares"].iloc[0]) == 1_000.0, "the late amendment won and then the band deleted the whole filing"
    print("\n=== SANITY CHECK: band before dedup ===")
    print(
        f"  original (+5d) + amendment (+1,100d) -> {len(out)} row at "
        f"{float(out['shares'].iloc[0]):,.0f} shares, i.e. the ORIGINAL. Dedup-first would "
        "have left 0 rows for this filer. Validated."
    )


def test_the_early_half_of_the_band_is_non_binding_on_the_live_table():
    """`F13_MAX_EARLY_DAYS` has never dropped a row -- ZERO rows in `sec13f_hr` have
    `filing_date < period` at all. It is kept as an assertion against a future extraction bug
    that stamps a filing before its own period, and this pins that it would in fact fire."""
    out = _banded(
        [
            _row("AAA", "0001", -44, 100.0),  # a day inside the floor
            _row("BBB", "0002", -46, 200.0),
        ]
    )  # filed before the period itself
    assert sorted(out["ticker"]) == ["AAA"]
    print("\n=== SANITY CHECK: the early floor is an assertion, not work ===")
    print("  -44d kept, -46d dropped. On the live table the drop count is 0 of 23,801,899 " "rows, so this half is non-binding today. Validated.")


def test_no_band_and_no_filing_date_both_degrade_instead_of_dropping():
    """D5. `filing_band=None` is the elite table's path (it has not been measured yet), and a
    table with no `filing_date` column cannot be banded at all -- neither may silently delete
    holdings."""
    rows = [_row("AAA", "0001", 5, 100.0), _row("BBB", "0002", 900, 200.0)]
    assert len(_banded(rows, band=None)) == 2
    no_fd = [{k: v for k, v in r.items() if k != "filing_date"} for r in rows]
    assert len(_banded(no_fd)) == 2
    print("\n=== SANITY CHECK: the band degrades ===")
    print("  band=None -> 2 rows (the elite table's opt-out); no `filing_date` column -> " "2 rows and a logged note, never a raise. Validated.")
