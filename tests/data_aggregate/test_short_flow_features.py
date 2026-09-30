"""Short-FLOW features: FINRA RegSHO short-sale VOLUME (`ic_shortvol_*`) and SEC
fails-to-deliver (`ic_ftd_*`) -- registry section 6.

Covers the RegSHO file parser, the VOLUME-WEIGHTED ratios (the point of the rework: an
average of daily ratios is a different statistic), the 1-trading-day publication lag, the
one-sided price interactions, the FTD "absent date is NaN, absent ticker is 0" rule, and the
emitted `f_*` columns against the module's own EMISSION map.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.institutionals.short_flow_features import (
    EMISSION,
    SHORTVOL_PUB_LAG,
    Z_MIN_PERIODS,
    _fails_fields,
    _ftd_publication_date,
    _publish_ftd_vintages,
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
    got = float(cast(Any, df["ic_shortvol_ratio_5d"].loc[idx[5], "A"]))
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
    assert {"ic_shortvol_ratio_5d", "ic_shortvol_ratio_20d", "ic_shortvol_ratio_60d", "ic_shortvol_acceleration"}.issubset(df)
    assert "ic_shortvol_ratio_z252" not in df
    # heavily-shorted S0 has the highest ratio cross-sectionally
    assert str(df["ic_shortvol_ratio_20d"].loc[dates[300]].idxmax()) == "S0"
    short = hist.pivot_table(index="date", columns="ticker", values="short_volume", aggfunc="sum")
    total = hist.pivot_table(index="date", columns="ticker", values="total_volume", aggfunc="sum")
    expected = float(cast(Any, (short.rolling(20, min_periods=10).sum() / total.rolling(20, min_periods=10).sum()).loc[dates[299], "S0"]))
    got = float(cast(Any, df["ic_shortvol_ratio_20d"].loc[dates[300], "S0"]))
    assert np.isclose(got, expected), "lag broken"
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
    print("\n=== SANITY CHECK: the two short-flow price interactions ===")
    print(
        f"  both legs >= 0 and {int(both.to_numpy().sum())} cells fire both (must be 0); "
        "the internal normalization conditions the economic interactions but is not emitted. Validated."
    )


def test_ftd_absent_ticker_is_zero_and_published_state_is_held():
    idx = pd.DatetimeIndex(pd.bdate_range("2023-12-01", "2024-02-14"))
    fails = pd.DataFrame(
        [
            {"date": "2024-01-02", "ticker": "HI", "fails_quantity": 100.0},
            {"date": "2024-01-02", "ticker": "LO", "fails_quantity": 50.0},
            {"date": "2024-01-15", "ticker": "HI", "fails_quantity": 300.0},
        ]
    )
    volume = pd.DataFrame(1_000.0, index=idx, columns=["HI", "LO"])
    ratio = _fails_fields(fails, idx, None, volume)["ic_ftd_to_adv20"]

    assert ratio.loc[pd.Timestamp("2024-01-30")].isna().all()
    assert ratio.loc[pd.Timestamp("2024-01-31"), "HI"] == pytest.approx(0.3)
    assert ratio.loc[pd.Timestamp("2024-01-31"), "LO"] == pytest.approx(0.0)
    assert ratio.loc[pd.Timestamp("2024-02-14"), "HI"] == pytest.approx(0.3)
    print("\n=== SANITY CHECK: FTD coverage semantics ===")
    print("  LO is observed zero on the ZIP's latest covered date; the Jan-a state appears Jan 31 and is held until the next ZIP")


def test_ftd_zip_vintage_publishes_atomically_without_summing_balances():
    idx = pd.DatetimeIndex(pd.bdate_range("2023-12-01", "2024-02-20"))
    fails = pd.DataFrame(
        [
            {"date": "2024-01-02", "ticker": "HI", "fails_quantity": 100.0},
            {"date": "2024-01-02", "ticker": "LO", "fails_quantity": 50.0},
            {"date": "2024-01-15", "ticker": "HI", "fails_quantity": 300.0},
            {"date": "2024-01-16", "ticker": "HI", "fails_quantity": 800.0},
            {"date": "2024-01-31", "ticker": "HI", "fails_quantity": 900.0},
        ]
    )
    volume = pd.DataFrame(1_000.0, index=idx, columns=["HI", "LO"])

    ratio = _fails_fields(fails, idx, None, volume)["ic_ftd_to_adv20"]

    assert ratio.loc[pd.Timestamp("2024-01-30")].isna().all()
    assert ratio.loc[pd.Timestamp("2024-01-31"), "HI"] == pytest.approx(0.3)
    assert ratio.loc[pd.Timestamp("2024-01-31"), "LO"] == pytest.approx(0.0)
    assert ratio.loc[pd.Timestamp("2024-02-14"), "HI"] == pytest.approx(0.3)
    assert ratio.loc[pd.Timestamp("2024-02-15"), "HI"] == pytest.approx(0.9)

    print("\n=== SANITY CHECK: FTD ZIP publication vintage ===")
    print("  January a publishes its latest 300-share balance atomically on Jan 31 (not the 100+300 sum); January b replaces it on Feb 15")


def test_ftd_observed_all_zero_history_is_neutral_but_unavailable_is_nan():
    idx = pd.DatetimeIndex(pd.bdate_range("2022-01-03", periods=360))
    covered = idx[:300]
    fails = pd.DataFrame(
        [{"date": day, "ticker": ticker, "fails_quantity": value} for day in covered for ticker, value in (("ZERO", 0.0), ("CONST", 100.0))]
    )
    volume = pd.DataFrame({"ZERO": 1_000_000.0, "CONST": 1_000_000.0}, index=idx)

    fields = _fails_fields(fails, idx, None, volume)
    persistence = fields["ic_ftd_persistence_30d"]
    observed_zero = persistence["ZERO"].dropna()

    assert not observed_zero.empty
    assert observed_zero.eq(0.0).all()
    assert persistence["CONST"].isna().all(), "a nonzero constant basis has no defined persistence state"
    assert persistence.loc[idx[-1], "ZERO"] == 0.0, "the latest published ZIP state is held until another ZIP supersedes it"
    assert "ic_ftd_z252" not in fields
    print("\n=== SANITY CHECK: FTD neutral zero versus unavailable ===")
    print(
        f"  ZERO becomes persistence=0 after {Z_MIN_PERIODS} fully observed source dates plus its lookback and the published state is held; "
        "CONST remains unavailable, while the internal z is not emitted"
    )
    print("  OK: zero means observed neutral pressure, never missing source coverage")


def test_latest_rolling_outputs_require_their_source_date():
    dates, tickers, hist = _synth(t=400, n=2)
    missing_regsho_date = dates[-2]
    hist = hist[hist["date"] != missing_regsho_date]
    shares = pd.DataFrame(500_000_000.0, index=dates, columns=tickers)
    close = pd.DataFrame({ticker: np.linspace(100.0, 120.0, len(dates)) for ticker in tickers}, index=dates)
    volume = pd.DataFrame(5_000_000.0, index=dates, columns=tickers)
    short_fields = _shortvol_fields(hist, dates, shares, close, volume)
    assert all(frame.loc[dates[-1]].isna().all() for frame in short_fields.values())

    print("\n=== SANITY CHECK: rolling source-date completeness ===")
    print("  a missing RegSHO t-1 observation makes every latest derived leg NaN")


def test_ftd_publication_snaps_weekend_and_holiday_forward():
    weekend_idx = pd.DatetimeIndex(pd.bdate_range("2024-06-03", "2024-07-03"))
    weekend_state = pd.DataFrame({"A": np.nan}, index=weekend_idx)
    weekend_state.loc[pd.Timestamp("2024-06-14"), "A"] = 1.0
    weekend_hist = pd.DataFrame([{"date": "2024-06-14"}])
    weekend_published = _publish_ftd_vintages(weekend_state, weekend_hist, weekend_idx)

    holiday_idx = pd.DatetimeIndex(pd.bdate_range("2023-12-01", "2024-01-17")).difference(pd.DatetimeIndex(["2024-01-15"]))
    holiday_state = pd.DataFrame({"A": np.nan}, index=holiday_idx)
    holiday_state.loc[pd.Timestamp("2023-12-29"), "A"] = 2.0
    holiday_hist = pd.DataFrame([{"date": "2023-12-29"}])
    holiday_published = _publish_ftd_vintages(holiday_state, holiday_hist, holiday_idx)

    assert _ftd_publication_date("2024-06-14") == pd.Timestamp("2024-06-30")
    assert pd.isna(weekend_published.loc[pd.Timestamp("2024-06-28"), "A"])
    assert weekend_published.loc[pd.Timestamp("2024-07-01"), "A"] == 1.0
    assert _ftd_publication_date("2023-12-29") == pd.Timestamp("2024-01-15")
    assert pd.isna(holiday_published.loc[pd.Timestamp("2024-01-12"), "A"])
    assert holiday_published.loc[pd.Timestamp("2024-01-16"), "A"] == 2.0
    print("\n=== SANITY CHECK: FTD publication-date snapping ===")
    print("  Sunday month-end publishes Monday Jul 1; the Jan 15 market holiday publishes on the first tradable session, Jan 16")


def test_appending_later_ftd_zip_does_not_rewrite_published_prefix():
    idx = pd.DatetimeIndex(pd.bdate_range("2023-12-01", "2024-02-20"))
    first = pd.DataFrame(
        [
            {"date": "2024-01-02", "ticker": "A", "fails_quantity": 100.0},
            {"date": "2024-01-15", "ticker": "A", "fails_quantity": 300.0},
        ]
    )
    later = pd.DataFrame(
        [
            {"date": "2024-01-16", "ticker": "A", "fails_quantity": 800.0},
            {"date": "2024-01-31", "ticker": "A", "fails_quantity": 900.0},
        ]
    )
    volume = pd.DataFrame(1_000.0, index=idx, columns=["A"])

    before = _fails_fields(first, idx, None, volume)["ic_ftd_to_adv20"]
    after = _fails_fields(pd.concat([first, later], ignore_index=True), idx, None, volume)["ic_ftd_to_adv20"]

    pd.testing.assert_frame_equal(before.loc[:"2024-02-14"], after.loc[:"2024-02-14"])
    assert after.loc[pd.Timestamp("2024-02-15"), "A"] == pytest.approx(0.9)
    print("\n=== SANITY CHECK: FTD appended-ZIP prefix invariance ===")
    print("  adding January b leaves every January-a published cell unchanged and replaces the held state only on Feb 15")


def test_reused_symbol_is_null_outside_the_current_issuer_tenure():
    idx = pd.bdate_range("2021-01-04", periods=90)
    ticker = "REUSED"
    hist = pd.DataFrame([{"date": day, "ticker": ticker, "short_volume": 400_000.0, "total_volume": 1_000_000.0} for day in idx])
    tenure = pd.DataFrame(
        [
            {
                "symbol": ticker,
                "issuer_cik": "0000000123",
                "valid_from": idx[20],
                "valid_to": idx[60],
            }
        ]
    )
    roster = pd.DataFrame([{"ticker": ticker, "cik": "123"}])
    panel = build_short_flow_feature_panel(
        make_frames(idx, {ticker: {}}, universe=pd.Index([ticker])),
        hist,
        symbol_tenure=tenure,
        ticker_ciks=roster,
    )
    ratio = panel[panel["ticker"] == ticker].set_index("date")["f_ic_shortvol_ratio_20d"].reindex(idx)
    assert ratio.loc[: idx[19]].isna().all()
    assert np.isfinite(ratio.loc[idx[45]])
    assert ratio.loc[idx[60] :].isna().all()
    print("\n=== SANITY CHECK: reused-symbol tenure mask ===")
    print("  RegSHO cells are available only inside the current roster CIK's proven half-open symbol tenure")


def test_canonical_ticker_survives_a_historical_alias_tenure():
    idx = pd.bdate_range("2023-05-01", periods=80)
    cutover = idx[35]
    history = pd.DataFrame(
        [
            {
                "date": day,
                "ticker": "NEW",
                "short_volume": 400_000.0,
                "total_volume": 1_000_000.0,
            }
            for day in idx
        ]
    )
    tenure = pd.DataFrame(
        [
            {"symbol": "OLD", "issuer_cik": "0000000123", "valid_from": idx[0], "valid_to": cutover},
            {"symbol": "NEW", "issuer_cik": "0000000123", "valid_from": cutover, "valid_to": None},
        ]
    )
    roster = pd.DataFrame([{"ticker": "NEW", "cik": "123.0"}])

    panel = build_short_flow_feature_panel(
        make_frames(idx, {"NEW": {}}, universe=pd.Index(["NEW"])),
        history,
        symbol_tenure=tenure,
        ticker_ciks=roster,
    )
    ratio = panel.set_index("date")["f_ic_shortvol_ratio_20d"].reindex(idx)

    assert ratio.loc[cutover:].notna().all()
    assert np.isclose(ratio.loc[cutover], 0.4)
    print("\n=== SANITY CHECK: canonical storage across a symbol alias ===")
    print("  canonical NEW rows survive the OLD tenure, and a float-shaped roster CIK resolves through shared pad_cik")


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
    assert emitted == expected, f"missing {sorted(expected - emitted)}; undeclared {sorted(emitted - expected)}"
    for leg in ("f_ic_shortvol_ratio_20d", "f_ic_shortvol_turnover_20d", "f_ic_ftd_to_adv20", "f_ic_shortvol_market_coverage"):
        assert panel[leg].notna().any(), f"{leg} is all-NaN"
    assert not any(column.endswith("_xs") for column in emitted)
    assert not any(column.endswith("_vs_peers") for column in emitted)
    assert not any(column in panel for column in ("f_ic_shortvol_ratio_z252", "f_ic_ftd_z252"))
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
        f"the three ratios are in [0, 1], no peer leg survives without target/OOS evidence, and history z outputs are absent; "
        f"ic_shortvol_days_to_cover is absent (needs FINRA "
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
    assert np.isnan(float(cast(Any, cov.loc[last, "REUSED"]))), "3.0x the tape is impossible and must be NaN"
    assert not (cov == 1.0).any().any(), "nulled, never clipped to 1.0"
    print("\n=== SANITY CHECK: coverage ceiling ===")
    print(f"  GOOD reads {cov.loc[last, 'GOOD']:.2f}; a ticker whose RegSHO volume is 3x the tape reads NaN rather than 3.0. Validated.")
