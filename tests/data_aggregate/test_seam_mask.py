"""
test_seam_mask.py  (tests/data_aggregate/)
-------------------------------------------------------------------------------------------
The `null_ret` SEAM MASK: a registered entry leaves both price legs exactly as the vendor
published them, so `close_total` sits on two different BASES either side of the bar and every
rolling window that STRADDLES it mixes them. `measure_seams` re-measures the register against
`close_total`; `mask_seam_windows` nulls each feature over its own lookback.

SYNTHETIC known-truth, no DB and no network -- this is window arithmetic on a closed formula,
not an economic property, so the repo's "feature tests use real data" rule points the other way
(same reasoning as `test_level_basis.py`). The fixture is DHR's real registered event, and the
spans asserted below are the ones measured on the live panel first:

    peer_mom_63   pinned at rank 0.9978 through s+62, reads 0.1891 at s+63
    mom_12_1      plateaus at 0.9934 from s+21, holds 0.9457 at s+251, craters 0.2928 at s+252

The assertions are therefore EXACT, not approximate: one cell short on either side is a
failure, because the entire claim of this change is that the window arithmetic is right. A
window sitting wholly on one side of the seam is internally consistent and must survive
untouched, which is the property that makes this a mask and not a ticker blacklist.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.common.level_basis import (
    NULL_RET_STEP_TOL, mask_seam_windows, measure_seams)
from src.data_aggregate.utils.common.prices import (
    MOMENTUM_SEAM_WINDOW, momentum_characteristic)
from src.data_aggregate.utils.common.xs import xs_rank_pct
from src.data_aggregate.utils.momentum.features import (
    SEAM_EWMA_SPAN_MULTIPLE, SEAM_WINDOWS, build_feature_panel, compute_raw_features)

#: DHR's registered event -- the largest of the six. `close_total` re-bases at the FTV spinoff
#: and `pct_change()` reads the seam between the two bases as a +61.218% return.
SEAM_RET = 0.61218
#: Position of the seam bar inside the fixture index. Chosen so the LONGEST mask (`mom_12_1`,
#: reaching s+251) lands entirely inside the frame AND entirely past every feature's warm-up,
#: so a masked cell is never confusable with a warm-up NaN.
SEAM_POS = 300
N_BARS = 700
TICKERS = ("SEAM", "CTRL", "CTRL2")


def _index() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.bdate_range("2014-01-01", periods=N_BARS), name="date")


def _base(index: pd.DatetimeIndex) -> np.ndarray:
    """A positive series with real two-sided variation, so no feature is NaN for want of
    dispersion (a flat series makes RSI undefined and every MA ratio exactly 0)."""
    i = np.arange(index.size, dtype="float64")
    return 100.0 * (1.0003 ** i) * (1.0 + 0.01 * np.sin(i))


def _frames(seam: bool = True, step: float = SEAM_RET) -> dict[str, pd.DataFrame]:
    """A `close_total` / `close_split` / OHLC / volume fixture carrying ONE basis change.

    The seam bar's own move is flattened to zero before the step is applied, so the observed
    one-bar ratio is EXACTLY `1 + step` and `measure_seams` compares against `expect_ret`
    without the underlying wiggle eating the tolerance.
    """
    index = _index()
    base = _base(index)
    close = pd.DataFrame({t: base.copy() for t in TICKERS}, index=index)
    if seam:
        col = close["SEAM"].to_numpy().copy()
        col[SEAM_POS] = col[SEAM_POS - 1]          # flatten the bar's real move
        col[SEAM_POS:] = col[SEAM_POS:] * (1.0 + step)
        close["SEAM"] = col
    return {
        "close_total": close,
        "close_split": close.copy(),               # the DD / LDOS / VTR shape: both legs step
        "open": close.shift(1).bfill() * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "volume": pd.DataFrame(1.0e6, index=index, columns=list(TICKERS)),
        "sector_ret": pd.DataFrame(0.0004, index=index, columns=list(TICKERS)),
    }


def _register(date: pd.Timestamp, expect: float = SEAM_RET) -> dict:
    return {"null_ret": {"SEAM": [{"date": str(date.date()), "expect_ret": expect}]}}


def _raw(frames: dict, seams: dict | None) -> dict:
    return compute_raw_features(
        frames["close_total"], frames["open"], frames["sector_ret"],
        close_split=frames["close_split"], high=frames["high"], low=frames["low"],
        volume=frames["volume"], seams=seams)


# --------------------------------------------------------------------------- #
# mask_seam_windows -- the window arithmetic, on its own                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("back,skip", [(5, 0), (21, 0), (63, 0), (252, 21), (43, 1), (1, 0)])
def test_the_masked_span_is_exactly_the_straddling_window(back, skip):
    """`[s+skip, s+back-1]`, both ends inclusive, and NOTHING outside it."""
    index = _index()
    frame = pd.DataFrame(1.0, index=index, columns=list(TICKERS))
    when = index[SEAM_POS]
    out = mask_seam_windows(frame, {"SEAM": [when]}, back, skip)

    masked = np.flatnonzero(out["SEAM"].isna().to_numpy())
    expected = np.arange(SEAM_POS + skip, SEAM_POS + back)
    assert masked.tolist() == expected.tolist()
    # the other columns are bit-identical -- the mask is per TICKER, not per date
    assert out["CTRL"].equals(frame["CTRL"])
    assert out["CTRL2"].equals(frame["CTRL2"])

    print(f"\n=== SANITY CHECK: mask span for (back={back}, skip={skip}) ===")
    print(f"  masked positions {masked[0]}..{masked[-1]} = [s+{skip}, s+{back - 1}] "
          f"({masked.size} bars); CTRL/CTRL2 bit-identical. Validated.")


def test_a_seam_outside_the_frame_and_an_absent_ticker_are_no_ops():
    index = _index()
    frame = pd.DataFrame(1.0, index=index, columns=list(TICKERS))
    off_calendar = mask_seam_windows(frame, {"SEAM": [pd.Timestamp("1990-01-02")]}, 252, 21)
    unknown = mask_seam_windows(frame, {"NOPE": [index[SEAM_POS]]}, 252, 21)
    empty = mask_seam_windows(frame, {}, 252, 21)

    for out in (off_calendar, unknown, empty):
        assert out.equals(frame)
    # and it returns a COPY, so a caller cannot be mutated from under itself
    empty.iloc[0, 0] = 99.0
    assert frame.iloc[0, 0] == 1.0

    print("\n=== SANITY CHECK: no-op cases ===")
    print("  seam off-calendar, unknown ticker and empty dict all return the frame "
          "unchanged, as a copy. Validated.")


def test_the_mask_runs_off_the_end_of_the_frame_without_raising():
    """A seam near the last bar masks to the end and stops there -- an incremental build's
    window can easily end mid-mask."""
    index = _index()
    frame = pd.DataFrame(1.0, index=index, columns=list(TICKERS))
    out = mask_seam_windows(frame, {"SEAM": [index[-10]]}, 252, 21)

    masked = np.flatnonzero(out["SEAM"].isna().to_numpy())
    assert masked.tolist() == list(range(N_BARS - 10 + 21, N_BARS))
    print("\n=== SANITY CHECK: mask clipped at the frame edge ===")
    print(f"  seam at bar {N_BARS - 10} of {N_BARS}: masked {masked.size} bars to the end, "
          "no raise. Validated.")


# --------------------------------------------------------------------------- #
# measure_seams -- the re-measurement contract                                 #
# --------------------------------------------------------------------------- #
def test_a_registered_seam_is_measured_on_close_total():
    frames = _frames()
    when = frames["close_total"].index[SEAM_POS]
    logged: list[str] = []
    seams = measure_seams(frames["close_total"], _register(when),
                          lambda m, *a: logged.append(m % a))

    assert seams == {"SEAM": [when]}
    assert any("MEASURED" in m for m in logged), logged

    print("\n=== SANITY CHECK: measure_seams reads close_total, not ret ===")
    print(f"  SEAM {when.date()}: +{SEAM_RET * 100:.2f}% one-bar basis change measured from "
          "close_total. Validated.")


def test_measuring_on_ret_would_find_nothing_which_is_why_it_reads_close_total():
    """The load-bearing negative. By the time a feature step runs, `apply_null_ret` has already
    set the seam cell of `ret` to NaN, so re-measuring THERE finds no usable bar pair and skips
    every entry -- the mask would silently never apply and still pass a happy-path test."""
    frames = _frames()
    when = frames["close_total"].index[SEAM_POS]
    ret = frames["close_total"].pct_change(fill_method=None)
    ret.at[when, "SEAM"] = np.nan                 # what apply_null_ret already did upstream

    logged: list[str] = []
    on_ret = measure_seams(ret, _register(when), lambda m, *a: logged.append(m % a))
    on_close = measure_seams(frames["close_total"], _register(when), lambda *a: None)

    assert on_ret == {}
    assert on_close == {"SEAM": [when]}
    assert any("SKIPPED" in m for m in logged), logged

    print("\n=== SANITY CHECK: ret is already nulled, close_total is not ===")
    print(f"  measured on ret -> {on_ret} (mask would be a no-op); "
          f"measured on close_total -> 1 seam. Validated.")


def test_a_seam_whose_defect_is_gone_is_skipped_and_logged():
    """The stale-register contract, inherited from `apply_null_ret`: an entry whose observed
    step no longer matches is never masked on faith."""
    frames = _frames(step=0.012)                  # a real +1.2% day where the seam used to be
    when = frames["close_total"].index[SEAM_POS]
    logged: list[str] = []
    seams = measure_seams(frames["close_total"], _register(when),
                          lambda m, *a: logged.append(m % a))

    assert seams == {}
    assert any("GONE or CHANGED" in m for m in logged), logged

    print("\n=== SANITY CHECK: a stale seam entry is refused ===")
    print(f"  register expects {SEAM_RET:+.5f}, observed +0.01200 -> SKIPPED, nothing masked. "
          "Validated.")


def test_the_seam_band_is_wide_enough_for_float_noise_and_no_wider():
    """Reuses `NULL_RET_STEP_TOL`, the register's own band, so the mask cannot drift away from
    the repair it follows."""
    edge = (1.0 + SEAM_RET) * (1.0 + NULL_RET_STEP_TOL * 0.9) - 1.0
    outside = (1.0 + SEAM_RET) * (1.0 + NULL_RET_STEP_TOL * 1.1) - 1.0
    index = _index()
    when = index[SEAM_POS]

    inside = measure_seams(_frames(step=edge)["close_total"], _register(when), lambda *a: None)
    refused = measure_seams(_frames(step=outside)["close_total"], _register(when),
                            lambda *a: None)

    assert inside == {"SEAM": [when]}
    assert refused == {}

    print("\n=== SANITY CHECK: the seam re-measurement band ===")
    print(f"  tol {NULL_RET_STEP_TOL:.4f} of the STEP: {edge:+.5f} matches, "
          f"{outside:+.5f} refused. Validated.")


def test_a_seam_outside_the_loaded_window_is_skipped():
    """An incremental build whose window starts after the seam must not raise, and must not
    mask a date it never loaded."""
    frames = _frames()
    when = frames["close_total"].index[SEAM_POS]
    narrow = frames["close_total"].iloc[SEAM_POS + 5:]

    logged: list[str] = []
    assert measure_seams(narrow, _register(when), lambda m, *a: logged.append(m % a)) == {}
    assert any("outside this build's window" in m for m in logged), logged

    first_bar = frames["close_total"].iloc[:1]
    assert measure_seams(first_bar, _register(frames["close_total"].index[0]),
                         lambda *a: None) == {}

    print("\n=== SANITY CHECK: seams outside the window ===")
    print("  window starting after the seam -> SKIPPED; a seam on the frame's first bar has "
          "no step to measure -> SKIPPED. Validated.")


# --------------------------------------------------------------------------- #
# momentum_characteristic -- the shared primitive all three consumers call     #
# --------------------------------------------------------------------------- #
def test_momentum_characteristic_masks_s21_to_s251_and_nothing_else():
    frames = _frames()
    close = frames["close_total"]
    when = close.index[SEAM_POS]

    plain = momentum_characteristic(close)
    masked = momentum_characteristic(close, seams={"SEAM": [when]})

    newly = np.flatnonzero((masked["SEAM"].isna() & plain["SEAM"].notna()).to_numpy())
    assert newly.tolist() == list(range(SEAM_POS + 21, SEAM_POS + 252))
    assert MOMENTUM_SEAM_WINDOW == (252, 21)
    # everything outside the span, and every other ticker, is bit-identical
    assert masked["CTRL"].equals(plain["CTRL"])
    outside = np.r_[0:SEAM_POS + 21, SEAM_POS + 252:N_BARS]
    assert masked["SEAM"].iloc[outside].equals(plain["SEAM"].iloc[outside])

    print("\n=== SANITY CHECK: momentum_characteristic seam mask ===")
    print(f"  masked bars {newly[0]}..{newly[-1]} = [s+21, s+251] ({newly.size} bars); "
          "outside the span and on CTRL, bit-identical. Validated.")


def test_seams_none_is_bit_identical_to_the_pre_mask_behaviour():
    """The back-compat guarantee: every existing call site and fixture passes nothing."""
    frames = _frames()
    close = frames["close_total"]

    assert momentum_characteristic(close).equals(
        momentum_characteristic(close, seams=None))
    assert momentum_characteristic(close).equals(
        close.shift(21) / close.shift(252) - 1.0)

    a, b = _raw(frames, None), _raw(frames, {})
    assert set(a) == set(b)
    for name in a:
        assert a[name].equals(b[name]), name

    print("\n=== SANITY CHECK: seams=None back-compat ===")
    print(f"  {len(a)} raw features bit-identical between seams=None and seams={{}}; "
          "momentum_characteristic reproduces the bare expression. Validated.")


# --------------------------------------------------------------------------- #
# compute_raw_features -- every masked feature, its exact span                 #
# --------------------------------------------------------------------------- #
def test_every_seam_window_feature_is_masked_over_exactly_its_own_lookback():
    frames = _frames()
    when = frames["close_total"].index[SEAM_POS]
    plain = _raw(frames, None)
    masked = _raw(frames, {"SEAM": [when]})

    lines = []
    for name, (back, skip) in SEAM_WINDOWS.items():
        assert name in plain, f"{name} is in SEAM_WINDOWS but not produced"
        newly = np.flatnonzero(
            (masked[name]["SEAM"].isna() & plain[name]["SEAM"].notna()).to_numpy())
        expected = list(range(SEAM_POS + skip, SEAM_POS + back))
        assert newly.tolist() == expected, (
            f"{name}: masked {newly.tolist()[:3]}..{newly.tolist()[-3:]}, "
            f"expected s+{skip}..s+{back - 1}")
        lines.append(f"  {name:<14} (back={back:>3}, skip={skip}) -> "
                     f"s+{skip}..s+{back - 1} = {newly.size} bars")

    # the features NOT in SEAM_WINDOWS must be bit-identical, seam ticker included
    untouched = [n for n in plain if n not in SEAM_WINDOWS]
    for name in untouched:
        assert masked[name].equals(plain[name]), name

    print("\n=== SANITY CHECK: per-feature seam spans ===")
    print("\n".join(lines))
    print(f"  {len(untouched)} unmasked features bit-identical (ret-derived, seasonal and the "
          f"whole volume family): {', '.join(sorted(untouched))}. Validated.")


def test_the_control_tickers_never_move():
    """The negative property. A mask that widened to the whole cross-section would still pass
    every span assertion above."""
    frames = _frames()
    when = frames["close_total"].index[SEAM_POS]
    plain = _raw(frames, None)
    masked = _raw(frames, {"SEAM": [when]})

    for name in plain:
        for ticker in ("CTRL", "CTRL2"):
            assert masked[name][ticker].equals(plain[name][ticker]), f"{name}/{ticker}"

    print("\n=== SANITY CHECK: the mask is confined to the seam ticker ===")
    print(f"  CTRL and CTRL2 bit-identical across all {len(plain)} raw features. Validated.")


def test_the_ewma_multiple_is_what_the_technical_windows_are_built_from():
    """`macd`'s slow span is 26 and Wilder's `n` is 14; the `+1` is the `.shift(1)`."""
    assert SEAM_WINDOWS["macd"] == (SEAM_EWMA_SPAN_MULTIPLE * 26 + 1, 1)
    assert SEAM_WINDOWS["macd_hist"] == SEAM_WINDOWS["macd"]
    assert SEAM_WINDOWS["rsi_14"] == (SEAM_EWMA_SPAN_MULTIPLE * 14 + 1, 1)
    assert SEAM_WINDOWS["atr_14"] == SEAM_WINDOWS["rsi_14"]

    weight_wilder = (1.0 - 1.0 / 14) ** (SEAM_EWMA_SPAN_MULTIPLE * 14)
    weight_macd = (1.0 - 2.0 / 27) ** (SEAM_EWMA_SPAN_MULTIPLE * 26)
    assert weight_wilder < 0.05 and weight_macd < 0.01

    print("\n=== SANITY CHECK: the EWMA cut-off is a decay argument ===")
    print(f"  at {SEAM_EWMA_SPAN_MULTIPLE} spans the seam bar's residual weight is "
          f"{weight_wilder:.2%} (Wilder n=14) and {weight_macd:.2%} (MACD slow span 26). "
          "Validated.")


# --------------------------------------------------------------------------- #
# the ordering: masked BEFORE the cross-sectional standardization              #
# --------------------------------------------------------------------------- #
def test_the_mask_lands_before_ranking_so_other_names_ranks_are_unaffected():
    """The whole reason the mask sits inside `compute_raw_features`. Masking AFTER the rank
    would leave the fabricated value in the cross-section, where it takes the top rank and
    shifts every other name's percentile. The test of that is exact: with the seam masked, the
    OTHER tickers' ranks must equal what they are when the seam ticker is not in the universe
    at all.
    """
    frames = _frames()
    when = frames["close_total"].index[SEAM_POS]
    seams = {"SEAM": [when]}
    affected = frames["close_total"].index[SEAM_POS + 21:SEAM_POS + 252]

    masked = _raw(frames, seams)
    dropped = _raw({k: (v.drop(columns=["SEAM"]) if "SEAM" in v.columns else v)
                    for k, v in frames.items()}, None)

    with_seam = xs_rank_pct(masked["mom_12_1"]).loc[affected, ["CTRL", "CTRL2"]]
    without = xs_rank_pct(dropped["mom_12_1"]).loc[affected, ["CTRL", "CTRL2"]]
    assert with_seam.equals(without)

    # and the counter-factual: ranking BEFORE masking does move them
    naive = xs_rank_pct(_raw(frames, None)["mom_12_1"]).loc[affected, ["CTRL", "CTRL2"]]
    assert not naive.equals(without)

    print("\n=== SANITY CHECK: mask precedes the cross-sectional rank ===")
    print(f"  over {len(affected)} affected dates, CTRL/CTRL2 mom_12_1 ranks with the seam "
          "MASKED == their ranks with the seam ticker dropped entirely.")
    print(f"  unmasked, the same ranks differ (e.g. {naive.iloc[0, 0]:.4f} vs "
          f"{without.iloc[0, 0]:.4f}) -- the fabricated value had displaced them. Validated.")


# --------------------------------------------------------------------------- #
# incremental safety                                                           #
# --------------------------------------------------------------------------- #
def test_an_incremental_window_masks_the_same_trading_DAYS_not_the_same_positions():
    """`mask_seam_windows` is positional, so a trimmed window shifts every index position --
    the property that has to hold is that the same DATES come out masked."""
    frames = _frames()
    close = frames["close_total"]
    when = close.index[SEAM_POS]
    cut = SEAM_POS - 40                            # a window that starts INSIDE the warm-up

    full = momentum_characteristic(close, seams={"SEAM": [when]})
    trimmed = momentum_characteristic(close.iloc[cut:], seams={"SEAM": [when]})

    full_masked = set(full.index[full["SEAM"].isna()])
    trimmed_masked = set(trimmed.index[trimmed["SEAM"].isna()])
    span = set(close.index[SEAM_POS + 21:SEAM_POS + 252])
    # inside the trimmed window the two agree exactly on the seam span
    assert trimmed_masked & span == span
    assert full_masked & span == span

    print("\n=== SANITY CHECK: the mask is date-stable under trimming ===")
    print(f"  window trimmed to start {SEAM_POS - cut} bars before the seam: all "
          f"{len(span)} dates of [s+21, s+251] masked in BOTH the full and the trimmed "
          "build. Validated.")


def test_every_part_warmup_exceeds_the_longest_mask_reach():
    """Why a seam BEFORE an incremental window's load start needs no special handling. Such a
    seam is skipped by `measure_seams` (the date is not in the loaded index), and that is safe
    only because the mask could not have reached an emitted row anyway: the first emitted row
    sits a full warm-up after the load start, and the warm-up is longer than the reach.
    """
    from src.data_aggregate.utils.common.parts import part_for
    from src.data_store.schema import Tables

    reach = max(back for back, _ in SEAM_WINDOWS.values()) - 1
    momentum = part_for(Tables.cube_part_momentum).warmup_trading_days
    targets = part_for(Tables.cube_part_targets).warmup_trading_days

    assert reach == 251
    assert momentum > reach, (momentum, reach)
    assert targets > reach, (targets, reach)

    print("\n=== SANITY CHECK: warm-up exceeds the mask reach ===")
    print(f"  longest mask reach s+{reach} (mom_12_1); cube_part_momentum warm-up "
          f"{momentum} trading days, cube_part_targets {targets}. A seam before the load "
          "start cannot contaminate an emitted row. Validated.")


def test_the_stored_panel_carries_the_mask_through_build_feature_panel():
    """End to end on the long panel the step actually writes."""
    frames = _frames()
    when = frames["close_total"].index[SEAM_POS]
    panel = build_feature_panel(
        frames["close_total"], frames["open"], frames["sector_ret"], method="rank",
        high=frames["high"], low=frames["low"], volume=frames["volume"],
        close_split=frames["close_split"], seams={"SEAM": [when]})

    seam_rows = panel[panel["ticker"] == "SEAM"].set_index("date").sort_index()
    span = frames["close_total"].index[SEAM_POS + 21:SEAM_POS + 252]
    assert seam_rows.loc[span, "mom_12_1"].isna().all()
    # the bar either side of the span still carries a value
    assert pd.notna(seam_rows.loc[frames["close_total"].index[SEAM_POS + 20], "mom_12_1"])
    assert pd.notna(seam_rows.loc[frames["close_total"].index[SEAM_POS + 252], "mom_12_1"])

    print("\n=== SANITY CHECK: the mask survives into the stored long panel ===")
    print(f"  SEAM mom_12_1 is NaN for all {len(span)} bars in [s+21, s+251] and non-null at "
          "s+20 and s+252. Validated.")
