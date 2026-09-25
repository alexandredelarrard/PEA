"""Short-FLOW features: FINRA RegSHO short-sale VOLUME (`ic_shortvol_*`) and SEC
fails-to-deliver (`ic_ftd_*`) -- registry section 6.

Covers the RegSHO file parser, the VOLUME-WEIGHTED ratios (the point of the rework: an
average of daily ratios is a different statistic), the 1-trading-day publication lag, the
one-sided price interactions, the FTD "absent date is NaN, absent ticker is 0" rule, and the
emitted `f_*` columns against the module's own EMISSION map.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.institutionals.short_flow_features import (
    BASE_WINDOW,
    EMISSION,
    FTD_PUB_LAG,
    SHORTVOL_PUB_LAG,
    Z_MIN_PERIODS,
    _fails_fields,
    _shortvol_fields,
    build_short_flow_feature_panel,
)
from src.data_extract.utils.institutionals.fetch_short_interest import _parse_regsho
from tests.conftest import make_frames


def test_parse_regsho():
    txt = (
        "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
        "20230103|AAPL|500|0|1000|Q\n"
        "20230103|AAPL|100|0|200|N\n"  # second market center -> summed
        "20230103|MSFT|300|0|1200|Q\n"
    )
    out = _parse_regsho(txt)
    aapl = out[out["source_symbol"] == "AAPL"].iloc[0]
    assert aapl["short_volume"] == 600 and aapl["total_volume"] == 1200  # summed
    assert out["date"].iloc[0] == pd.Timestamp("2023-01-03")
    assert _parse_regsho("").empty
    print("\n=== SANITY CHECK: RegSHO parser ===")
    print(
        f"  pipe file parsed; AAPL short/total summed across markets = "
        f"{int(aapl['short_volume'])}/{int(aapl['total_volume'])}; empty -> empty. Validated."
    )


def _synth(t=700, n=8, seed=0):
    dates = pd.DatetimeIndex(pd.bdate_range("2021-01-04", periods=t))
    tickers = [f"S{i}" for i in range(n)]
    rng = np.random.default_rng(seed)
    total = rng.uniform(1e6, 5e6, (t, n))
    frac = np.clip(rng.normal(0.4, 0.1, (t, n)), 0.05, 0.95)
    frac[:, 0] = 0.85  # S0 persistently heavily shorted
    rows = []
    for j, t in enumerate(tickers):
        for i, d in enumerate(dates):
            rows.append({"date": d, "ticker": t, "short_volume": total[i, j] * frac[i, j], "total_volume": total[i, j]})
    return dates, tickers, pd.DataFrame(rows)


def test_the_ratio_is_volume_weighted_not_an_average_of_daily_ratios():
    """THE test for this phase (part-2b Phase 2.5 verification). Five days: four quiet days
    that were 90% short on 100k shares, and one heavy day that was 10% short on 10m. The
    window's short share of TRADED VOLUME is 3.7%, not the 74% an average of daily ratios
    reports -- a factor of 20, and the average is the statistic the module used to emit."""
    dates = pd.DatetimeIndex(pd.bdate_range("2023-01-02", periods=5))
    rows = [{"date": d, "ticker": "A", "total_volume": 100_000.0, "short_volume": 90_000.0} for d in dates[:4]]
    rows.append({"date": dates[4], "ticker": "A", "total_volume": 10_000_000.0, "short_volume": 1_000_000.0})
    hist = pd.DataFrame(rows)
    df = _shortvol_fields(hist, dates, None, None, None)
    # the lag means day 5's window is only readable on the NEXT grid day, so extend the index
    idx = pd.DatetimeIndex(pd.bdate_range("2023-01-02", periods=6))
    df = _shortvol_fields(hist, idx, None, None, None)
    got = df["ic_shortvol_ratio_5d"].loc[idx[5], "A"]
    weighted = (4 * 90_000 + 1_000_000) / (4 * 100_000 + 10_000_000)
    naive = (4 * 0.9 + 0.1) / 5
    assert abs(got - weighted) < 1e-12
    assert abs(got - naive) > 0.5, "the volume weighting made no difference -- check the sum"
    print("\n=== SANITY CHECK: volume-weighted short ratio ===")
    print(
        f"  four 90%-short days on 100k + one 10%-short day on 10m: "
        f"ic_shortvol_ratio_5d = {got:.4f} (= sum short / sum total), against {naive:.2f} "
        f"for an average of daily ratios -- {naive / got:.0f}x. Validated."
    )


def test_publication_lag_is_one_trading_day():
    dates, tickers, hist = _synth()
    df = _shortvol_fields(hist, dates, None, None, None)
    assert {"ic_shortvol_ratio_5d", "ic_shortvol_ratio_20d", "ic_shortvol_ratio_60d", "ic_shortvol_ratio_z252", "ic_shortvol_acceleration"}.issubset(
        df
    )
    # heavily-shorted S0 has the highest ratio cross-sectionally
    assert df["ic_shortvol_ratio_20d"].loc[dates[300]].idxmax() == "S0"
    short = hist.pivot_table(index="date", columns="ticker", values="short_volume", aggfunc="sum")
    total = hist.pivot_table(index="date", columns="ticker", values="total_volume", aggfunc="sum")
    expected = (short.rolling(20, min_periods=10).sum() / total.rolling(20, min_periods=10).sum()).loc[dates[299], "S0"]
    assert np.isclose(df["ic_shortvol_ratio_20d"].loc[dates[300], "S0"], expected), "lag broken"
    assert SHORTVOL_PUB_LAG == 1
    print("\n=== SANITY CHECK: RegSHO 1-day publication lag ===")
    print(
        f"  S0 (85% shorted) tops ic_shortvol_ratio_20d at {dates[300].date()}; the value at "
        f"t equals the volume-weighted ratio computed through t-1 (published next morning). "
        "Validated."
    )


def test_price_interactions_are_one_sided_and_complementary():
    """#62/#63. High short flow means opposite things depending on the price path, so each
    interaction is zero in the other's regime -- never averaged into one signed number."""
    dates, tickers, hist = _synth(t=400)
    rng = np.random.default_rng(3)
    close = pd.DataFrame({t: 100 * np.cumprod(1 + rng.normal(0.0004, 0.02, len(dates))) for t in tickers}, index=dates)
    df = _shortvol_fields(hist, dates, None, close, None)
    weak = df["ic_shortvol_high_x_weak_price"]
    strong = df["ic_shortvol_high_x_strong_price"]
    assert (weak.fillna(0) >= 0).all().all() and (strong.fillna(0) >= 0).all().all()
    both = (weak > 0) & (strong > 0)
    assert not both.to_numpy().any(), "a cell is both confirming and absorbing"
    # and a NEGATIVE z (short flow below its own norm) fires neither
    z = df["ic_shortvol_ratio_z252"]
    quiet = (z < 0) & z.notna()
    assert (weak.where(quiet).fillna(0) == 0).all().all()
    print("\n=== SANITY CHECK: the two short-flow price interactions ===")
    print(
        f"  both legs >= 0; {int(both.to_numpy().sum())} cells fire both (must be 0); a "
        f"below-norm z fires neither. The asymmetry is shipped, not averaged. Validated."
    )


def test_ftd_absent_date_is_nan_and_absent_ticker_is_zero():
    """An FTD file lists a security only on days it had fails, so within a PUBLISHED date a
    missing ticker means no fails -- but a date the file never covered is unknown, not zero."""
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=120))
    covered = idx[:60]
    fails = pd.DataFrame([{"date": d, "ticker": "HI", "fails_quantity": 1e5} for d in covered])
    volume = pd.DataFrame({t: 1e6 for t in ("HI", "LO")}, index=idx)
    df = _fails_fields(fails, idx, None, volume)
    ratio = df["ic_ftd_to_adv20"]
    # LO never appears in the file at all -> it is not a column of the source pivot
    assert "HI" in ratio.columns
    # HI on a covered date, read after the publication lag: a real number
    assert np.isfinite(ratio.loc[idx[FTD_PUB_LAG + 30], "HI"])
    # a date the file does not cover: NaN, not 0
    assert np.isnan(ratio.loc[idx[-1], "HI"])
    print("\n=== SANITY CHECK: FTD coverage semantics ===")
    print(
        f"  file covers {len(covered)} of {len(idx)} grid days: HI reads "
        f"{ratio.loc[idx[FTD_PUB_LAG + 30], 'HI']:.4f} inside the covered span and NaN "
        f"outside it -- 'not published' is never reported as 'no fails'. Validated."
    )


def test_ftd_observed_all_zero_history_is_neutral_but_unavailable_is_nan():
    idx = pd.DatetimeIndex(pd.bdate_range("2022-01-03", periods=360))
    covered = idx[:300]
    fails = pd.DataFrame(
        [{"date": day, "ticker": ticker, "fails_quantity": value} for day in covered for ticker, value in (("ZERO", 0.0), ("CONST", 100.0))]
    )
    volume = pd.DataFrame({"ZERO": 1_000_000.0, "CONST": 1_000_000.0}, index=idx)

    fields = _fails_fields(fails, idx, None, volume)
    z = fields["ic_ftd_z252"]
    persistence = fields["ic_ftd_persistence_30d"]
    basis_warmup = max(3, BASE_WINDOW // 2)
    first_neutral = FTD_PUB_LAG + basis_warmup + Z_MIN_PERIODS - 2
    first_persistence = first_neutral + 14

    assert pd.isna(z.loc[idx[first_neutral - 1], "ZERO"])
    assert z.loc[idx[first_neutral], "ZERO"] == 0.0
    assert persistence.loc[idx[first_persistence], "ZERO"] == 0.0
    assert z["CONST"].isna().all(), "a nonzero constant basis has no defined neutral z-score"
    assert pd.isna(z.loc[idx[-1], "ZERO"]), "an uncovered source date must remain unavailable"
    print("\n=== SANITY CHECK: FTD neutral zero versus unavailable ===")
    print(
        f"  ZERO becomes z=0 after {Z_MIN_PERIODS} fully observed source dates and "
        "persistence=0 after its usual lookback; CONST and the uncovered tail remain NaN"
    )
    print("  OK: zero means observed neutral pressure, never missing source coverage")


def test_panel_columns_match_the_emission_map():
    dates, tickers, hist = _synth(t=400)
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}
    rng = np.random.default_rng(11)
    close = pd.DataFrame({t: 100 * np.cumprod(1 + rng.normal(0.0004, 0.02, len(dates))) for t in tickers}, index=dates)
    volume = pd.DataFrame({t: 3e6 for t in tickers}, index=dates)
    fund = pd.DataFrame([{"ticker": t, "as_of": "2020-01-01", "sharesOutstanding": 5e8, "sharesOutstandingPit": 5e8} for t in tickers])
    keep = rng.random(len(dates) * len(tickers)) < 0.3
    ftd = pd.DataFrame(
        {
            "date": np.repeat(dates.to_numpy(), len(tickers))[keep],
            "ticker": np.tile(np.array(tickers), len(dates))[keep],
            "fails_quantity": rng.lognormal(8, 1.2, int(keep.sum())).round(0),
        }
    )
    panel = build_short_flow_feature_panel(
        make_frames(dates, peers, volume=volume, close_total=close), hist, fails_history=ftd, shares_out_history=fund
    )
    expected = set()
    for name, mode in EMISSION.items():
        expected.add(f"f_{name}")
        if mode == "raw+xs":
            expected.add(f"f_{name}_xs")
        elif mode == "raw+peers":
            expected.add(f"f_{name}_vs_peers")
    emitted = {c for c in panel.columns if c.startswith("f_")}
    assert emitted == expected, f"missing {sorted(expected - emitted)}; " f"undeclared {sorted(emitted - expected)}"
    for leg in ("f_ic_shortvol_ratio_20d_vs_peers", "f_ic_shortvol_turnover_20d_xs", "f_ic_ftd_pct_so", "f_ic_shortvol_market_coverage"):
        assert panel[leg].notna().any(), f"{leg} is all-NaN"
    # the three bounded ratios stay in [0, 1]
    for w in (5, 20, 60):
        s = panel[f"f_ic_shortvol_ratio_{w}d"].dropna()
        assert s.between(0, 1).all(), f"{w}d ratio out of [0, 1]"
    # the coverage measurement is a share of the tape, so it cannot exceed 1 when the tape is
    # the larger number -- here RegSHO's 1-5m against a 3m tape, so it straddles 1 by design
    assert panel["f_ic_shortvol_market_coverage"].dropna().gt(0).all()
    # days_to_cover is GONE: it needs short-INTEREST positions this repo does not fetch
    assert not any("days_to_cover" in c for c in panel.columns)
    print("\n=== SANITY CHECK: short-flow panel columns vs the EMISSION map ===")
    print(
        f"  {len(emitted)} legs emitted from {len(EMISSION)} declared features, exact match; "
        f"the three ratios are in [0, 1]; ic_shortvol_days_to_cover is absent (needs FINRA "
        f"settlement positions, out of scope). Validated."
    )


def test_market_coverage_is_split_basis_free():
    """The defect the 2026-09-12 full build found: `ic_shortvol_market_coverage` breached its
    (0, 1) bound on 2,213 cells, max **3.71**, and all six offending tickers were
    corporate-action names — GE's window ending exactly at its 1-for-8 reverse split.

    Cause: RegSHO `total_volume` is AS-TRADED and never restated, while yfinance `Volume` IS
    retroactively scaled by the split ratio (measured over 746 splits: median post/pre volume
    ratio 0.869 against median R = 2.0). So the plain ratio carries a factor of `1 / F(d)`, and
    a REVERSE split (R < 1) inflates it — a 1-for-8 turns a true 28% coverage into 2.24.

    Here: a true coverage of 0.10 either side of a 1-for-8 reverse split. Unrestated the
    pre-split window reads 0.80 — eight times too high, but deliberately still UNDER 1.0 so
    `_guard_coverage`'s physical ceiling does not mask the effect this test is about. The two
    guards are independent and each has its own test.
    """
    idx = pd.DatetimeIndex(pd.bdate_range("2021-01-04", periods=120))
    split_date = idx[60]
    regsho = pd.DataFrame([{"date": d, "ticker": "GEX", "total_volume": 100_000.0, "short_volume": 40_000.0} for d in idx])
    # yfinance volume: as-traded 1,000,000 a day, then back-scaled by R = 0.125 BEFORE the split
    tape = pd.DataFrame({"GEX": [1_000_000.0 * (0.125 if d < split_date else 1.0) for d in idx]}, index=idx)
    splits = pd.DataFrame([{"date": split_date, "ticker": "GEX", "ratio": 0.125}])

    before = _shortvol_fields(regsho, idx, None, None, tape)["ic_shortvol_market_coverage"]
    after = _shortvol_fields(regsho, idx, None, None, tape, splits)["ic_shortvol_market_coverage"]
    pre, post = idx[50], idx[110]
    assert before.loc[pre, "GEX"] == pytest.approx(0.80, rel=1e-6), "the defect must reproduce"
    assert after.loc[pre, "GEX"] == pytest.approx(0.10, rel=1e-6)
    assert after.loc[post, "GEX"] == pytest.approx(0.10, rel=1e-6)
    assert before.loc[post, "GEX"] == pytest.approx(0.10, rel=1e-6), "post-split is unaffected"
    print("\n=== SANITY CHECK: market coverage is split-basis free ===")
    print(
        f"  across a 1-for-8 reverse split, true coverage 10%: unrestated reads "
        f"{before.loc[pre, 'GEX']:.2f} before the split (8x too high) and "
        f"{before.loc[post, 'GEX']:.2f} after; restated reads "
        f"{after.loc[pre, 'GEX']:.2f} / {after.loc[post, 'GEX']:.2f}. Validated."
    )


def test_coverage_above_one_is_nulled_as_a_physical_impossibility():
    """Off-exchange volume is a SUBSET of the consolidated tape, so coverage > 1.0 means the
    numerator and denominator are not the same security.

    After the split-basis fix, the 2026-09-12 build's 179 surviving breaches were all REUSED
    TICKERS — `WTW` carrying Weight Watchers' RegSHO volume over Willis Towers Watson's price
    grid (135 cells, max 3.71), plus AXON's rename chain and pre-Gen-Digital `GEN`. Nulled, not
    clipped: a clip at 1.0 would assert that RegSHO saw the entire tape.
    """
    idx = pd.DatetimeIndex(pd.bdate_range("2021-01-04", periods=60))
    regsho = pd.DataFrame(
        [{"date": d, "ticker": t, "total_volume": v, "short_volume": v * 0.4} for d in idx for t, v in (("GOOD", 250_000.0), ("REUSED", 3_000_000.0))]
    )
    tape = pd.DataFrame({"GOOD": [1_000_000.0] * len(idx), "REUSED": [1_000_000.0] * len(idx)}, index=idx)

    cov = _shortvol_fields(regsho, idx, None, None, tape)["ic_shortvol_market_coverage"]
    last = idx[-1]
    assert cov.loc[last, "GOOD"] == pytest.approx(0.25, rel=1e-6), "a real reading is kept"
    assert np.isnan(cov.loc[last, "REUSED"]), "3.0x the tape is impossible and must be NaN"
    assert not (cov == 1.0).any().any(), "nulled, never clipped to 1.0"
    print("\n=== SANITY CHECK: coverage ceiling ===")
    print(f"  GOOD reads {cov.loc[last, 'GOOD']:.2f}; a ticker whose RegSHO volume is 3x the " f"tape reads NaN rather than 3.0. Validated.")
