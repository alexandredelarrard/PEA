"""Unit tests for the Part 2 Phase 2.8 check suite.

Every test builds a panel that violates exactly ONE property and asserts the corresponding
check fails, plus a clean panel that passes. A check suite whose own failure path is untested
is the thing it is supposed to prevent: `f_ic_act_percent_of_class_vs_peers` shipped declared,
emitted and 0.0% non-null over 3.9M rows, and an existence assertion passed on it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validate import institutionals as iv


def _panel(features: dict[str, list[float]], dates: pd.DatetimeIndex | None = None,
           tickers: tuple[str, ...] = ("AAA",)) -> pd.DataFrame:
    n = len(next(iter(features.values())))
    dates = dates if dates is not None else pd.bdate_range("2025-01-01", periods=n)
    frames = []
    for ticker in tickers:
        frame = pd.DataFrame({"date": dates[:n], "ticker": ticker})
        for name, values in features.items():
            frame[name] = values
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_split_leg_separates_the_three_emission_legs():
    assert iv.split_leg("f_ic_bo_percent_of_class") == ("ic_bo_percent_of_class", "raw")
    assert iv.split_leg("f_ic_bo_percent_of_class_xs") == ("ic_bo_percent_of_class", "xs")
    assert iv.split_leg("f_ic_inst_ownership_pct_vs_peers") == ("ic_inst_ownership_pct",
                                                                "vs_peers")


def test_declared_bounds_fail_on_an_out_of_range_raw_leg():
    """V1-V3. `ic_inst_ownership_pct` above 1.05 is the share-count defect, not a big holder."""
    clean = _panel({"f_ic_inst_ownership_pct": [0.1, 0.5, 1.02]})
    assert iv.value_declared_bounds(clean).status == iv.PASS

    dirty = _panel({"f_ic_inst_ownership_pct": [0.1, 0.5, 4.0]})
    result = iv.value_declared_bounds(dirty)
    assert result.status == iv.FAIL and result.blocking
    assert result.detail[0]["out_of_range"] == 1


def test_group_summing_would_breach_the_percent_bound():
    """R3 restated as V1-V3: a 4-co-filer group at 10% each sums to 40% and stays in range, but
    a 15% group over 7 filers summed exceeds 100 -- which is why the bound is the regression
    guard and the source-side ratio is the evidence."""
    dirty = _panel({"f_ic_act_percent_of_class": [15.0, 105.0, 220.0]})
    result = iv.value_declared_bounds(dirty)
    assert result.status == iv.FAIL
    assert result.detail[0]["out_of_range"] == 2


def test_peer_z_rejects_inf_and_over_clip():
    clean = _panel({"f_ic_inst_ownership_pct_vs_peers": [-8.0, 0.0, 7.99]})
    assert iv.value_peer_z(clean).status == iv.PASS

    for bad in ([np.inf, 0.0, 1.0], [0.0, 0.0, 9.5]):
        result = iv.value_peer_z(_panel({"f_ic_inst_ownership_pct_vs_peers": bad}))
        assert result.status == iv.FAIL and result.blocking


def test_xs_leg_must_be_a_unit_percentile():
    assert iv.value_xs_unit(_panel({"f_ic_bo_holder_count_xs": [0.0, 0.5, 1.0]})).status == iv.PASS
    result = iv.value_xs_unit(_panel({"f_ic_bo_holder_count_xs": [0.0, 1.4, 1.0]}))
    assert result.status == iv.FAIL and result.detail[0]["out_of_unit"] == 1


def test_availability_floor_catches_a_pre_floor_value():
    """L1/L9. The floor is the source's own earliest date, so a feature carrying a value before
    it is reading something that was not public."""
    dates = pd.bdate_range("2024-12-01", periods=30)
    panel = _panel({"f_ic_bo_percent_of_class": [1.0] * 30}, dates=dates)
    floors = {"f_ic_bo_": pd.Timestamp("2024-12-17")}

    result = iv.leak_availability(panel, floors)
    assert result.status == iv.FAIL and result.blocking
    assert result.detail[0]["cells_before_floor"] > 0

    masked = panel.copy()
    masked.loc[masked["date"] < "2024-12-17", "f_ic_bo_percent_of_class"] = np.nan
    assert iv.leak_availability(masked, floors).status == iv.PASS


def test_availability_check_ignores_a_family_with_no_measured_floor():
    """A source that is absent contributes no floor, and an unfloored family must not be
    scored as passing -- it is simply not in the table."""
    panel = _panel({"f_ic_bo_percent_of_class": [1.0, 1.0, 1.0]})
    result = iv.leak_availability(panel, {})
    assert result.status == iv.PASS and result.detail == []


def test_first_period_delta_must_be_nan_on_the_first_13f_deadline():
    """L10 / D17. 2013-06-30 has no prior quarter; a value there is a universe-wide phantom."""
    dates = pd.bdate_range("2013-08-12", periods=6)
    panel = _panel({"f_ic_inst_breadth_chg": [0.4] * 6}, dates=dates, tickers=("AAA", "BBB"))
    result = iv.leak_first_period_delta(panel)
    assert result.status == iv.FAIL and result.blocking

    clean = panel.copy()
    clean.loc[clean["date"] <= "2013-08-14", "f_ic_inst_breadth_chg"] = np.nan
    assert iv.leak_first_period_delta(clean).status == iv.PASS


def test_13f_lag_fails_on_a_cohort_step_that_is_not_a_deadline():
    """L2 as a STEP-date test: a forward-filled panel is non-null on period+44d by design, so
    "NaN before the deadline" is untestable. What must not happen is a step on the wrong day."""
    dates = pd.bdate_range("2025-04-01", periods=60)
    periods = pd.Series([pd.Timestamp("2025-03-31")])  # deadline 2025-05-15
    tickers = tuple(f"T{i}" for i in range(10))

    early = [1.0 if d < pd.Timestamp("2025-05-02") else 2.0 for d in dates]
    result = iv.leak_13f_lag(_panel({"f_ic_inst_holders": early}, dates=dates, tickers=tickers),
                             periods)
    assert result.status == iv.FAIL and result.blocking
    assert result.detail[0]["first_off_deadline"] == "2025-05-02"

    late = [1.0 if d < pd.Timestamp("2025-05-15") else 2.0 for d in dates]
    assert iv.leak_13f_lag(_panel({"f_ic_inst_holders": late}, dates=dates, tickers=tickers),
                           periods).status == iv.PASS


def test_13f_lag_does_not_fail_a_price_scaled_feature():
    """A `*_to_mcap` leg moves every day because its denominator is a close. It has no step
    dates to place, so calling that a leak would be wrong -- it is reported as daily."""
    dates = pd.bdate_range("2025-04-01", periods=60)
    rng = np.random.default_rng(5)
    walk = (1.0 + rng.normal(0, 0.01, 60)).cumprod().tolist()
    result = iv.leak_13f_lag(
        _panel({"f_ic_inst_value_to_mcap": walk}, dates=dates,
               tickers=tuple(f"T{i}" for i in range(10))),
        pd.Series([pd.Timestamp("2025-03-31")]))
    assert result.status == iv.PASS
    assert result.detail[0]["verdict"].startswith("daily")


def test_event_date_stamping_passes_by_construction_when_the_column_is_unread():
    """L4. `sec_13d`/`sec_13g` never project `date_of_event`, so stamping on it is impossible
    -- a stronger proof than a sampled test, and the projection is the evidence."""
    unread = iv.leak_event_date_stamping("L4", "sec_13d", ["ticker", "filing_date"],
                                         "date_of_event")
    assert unread.status == iv.PASS

    read = iv.leak_event_date_stamping("L3", "insider_transactions",
                                       ["ticker", "filing_date", "transaction_date"],
                                       "transaction_date")
    assert read.status == iv.REPORT, "reading it is legitimate; it just is not proof"


def test_declared_vs_emitted_catches_the_all_nan_column():
    """D1. The 2026-08 failure mode: declared, emitted, and never non-null."""
    declared = {"ownership_features": {"ic_bo_percent_of_class": "raw+peers"}}
    panel = _panel({"f_ic_bo_percent_of_class": [1.0, 2.0, 3.0],
                    "f_ic_bo_percent_of_class_vs_peers": [np.nan] * 3})
    result = iv.check_declared_vs_emitted(panel, declared)
    assert result.status == iv.FAIL and result.blocking
    assert result.detail[0]["problem"] == "emitted, 0 non-null cells"


def test_declared_vs_emitted_catches_a_missing_column_and_reports_an_undeclared_one():
    declared = {"ownership_features": {"ic_bo_percent_of_class": "raw+xs"}}
    missing = iv.check_declared_vs_emitted(
        _panel({"f_ic_bo_percent_of_class": [1.0, 2.0, 3.0]}), declared)
    assert missing.status == iv.FAIL
    assert missing.detail[0]["feature"] == "f_ic_bo_percent_of_class_xs"

    extra = iv.check_declared_vs_emitted(
        _panel({"f_ic_bo_percent_of_class": [1.0, 2.0, 3.0],
                "f_ic_bo_percent_of_class_xs": [0.1, 0.2, 0.3],
                "f_ic_shortvol_ratio_5d": [0.1, 0.2, 0.3]}), declared)
    assert extra.status == iv.REPORT
    assert extra.detail[0]["feature"] == "f_ic_shortvol_ratio_5d"


def test_coverage_floor_is_applied_inside_the_availability_window():
    """G2, the FAMILY window. A feature scored over all history when its source only starts
    part-way through is cut for a property of the calendar.

    ⚠ Uses `f_ic_bo_holder_count`, which has no `FEATURE_FLOORS` override -- the four
    mandate-gated numerics have one, and this test is about the family fallback."""
    dates = pd.bdate_range("2024-01-01", periods=300)
    values = [np.nan if d < pd.Timestamp("2024-12-17") else 1.0 for d in dates]
    panel = _panel({"f_ic_bo_holder_count": values}, dates=dates)

    _, g2_all_history = iv.gate_coverage(panel, family_floors={})
    _, g2_windowed = iv.gate_coverage(
        panel, family_floors={"f_ic_bo_": pd.Timestamp("2024-12-17")})
    assert g2_windowed.status == iv.PASS
    assert g2_all_history.status == iv.PASS, "its own first non-null is the fallback window"

    sparse = _panel({"f_ic_bo_holder_count":
                     [1.0 if i % 50 == 0 else np.nan for i in range(300)]}, dates=dates)
    _, dropped = iv.gate_coverage(sparse, family_floors={"f_ic_bo_": dates[0]})
    assert dropped.detail[0]["feature"] == "f_ic_bo_holder_count"
    assert dropped.detail[0]["pct_non_null_in_window"] == pytest.approx(2.0, abs=0.01)


def test_coverage_sheet_carries_a_range_for_every_leg_bounded_or_not():
    """G1. The sheet is the only place an UNBOUNDED leg gets range evidence -- `DECLARED_BOUNDS`
    is silent about z-legs, `_xs` percentiles and every feature nobody declared a bound for, so
    without these columns most of the panel shipped with no distribution ever looked at."""
    values = [0.0] * 50 + list(np.linspace(0.1, 10.0, 50))
    sheet, _ = iv.gate_coverage(_panel({"f_ic_inst_holders": values,
                                        "f_ic_inst_holders_vs_peers": values}))
    row = {r["feature"]: r for r in sheet.detail}["f_ic_inst_holders_vs_peers"]
    assert row["min"] == 0.0 and row["max"] == 10.0
    assert row["p50"] == pytest.approx(0.0, abs=0.11), "the median sits in the zero mass"
    assert row["p99"] == pytest.approx(9.9, abs=0.2)
    assert row["pct_zero"] == pytest.approx(50.0, abs=0.01)
    print("\n=== SANITY CHECK: G1 range columns ===")
    print(f"  an UNBOUNDED peer leg reports min={row['min']}, p1={row['p1']}, p50={row['p50']}, "
          f"p99={row['p99']}, max={row['max']}, {row['pct_zero']}% exact zeros. Validated.")


def test_degenerate_leg_is_caught_below_the_all_nan_line():
    """V11. D1 catches the all-NaN column; the column that is 0.0 everywhere passes D1, passes
    its declared bound, passes the z cap and carries zero information. Both shapes -- exactly
    constant and 99.9% one value -- have to fail, and a fat-tailed-but-live leg must not."""
    n = 2000
    constant = _panel({"f_ic_inst_holders": [0.0] * n})
    assert iv.value_degenerate_legs(constant).status == iv.FAIL

    almost = [0.0] * (n - 1) + [3.0]                       # 99.95% one value
    result = iv.value_degenerate_legs(_panel({"f_ic_inst_holders": almost}))
    assert result.status == iv.FAIL and result.blocking
    assert result.detail[0]["problem"] == "> 99.9% one value"
    assert result.detail[0]["share"] == pytest.approx(99.95, abs=0.01)

    # a SPARSE event feature is not degenerate: 98% zeros is what a class-S decay leg looks like
    sparse = [0.0] * 1960 + list(np.linspace(0.1, 1.0, 40))
    assert iv.value_degenerate_legs(_panel({"f_ic_inst_holders": sparse})).status == iv.PASS
    # neither is a constant column with too few cells to judge
    assert iv.value_degenerate_legs(_panel({"f_ic_inst_holders": [1.0] * 50})).status == iv.PASS
    print("\n=== SANITY CHECK: V11 degenerate legs ===")
    print("  all-0.0 and 99.95%-one-value both FAIL; a 98%-zero sparse event leg and a "
          "50-cell column both PASS. Validated.")


def test_saturation_reports_clip_shoulder_and_tie_mass():
    rng = np.random.default_rng(11)
    n = 4000
    z = rng.normal(0, 3.5, n)
    z[:400] = 8.0  # 10% at the clip -> "investigate"
    panel = _panel({"f_ic_inst_ownership_pct": np.abs(rng.normal(0.4, 0.1, n)).tolist(),
                    "f_ic_inst_ownership_pct_vs_peers": z.tolist(),
                    "f_ic_inst_holders_xs": ([0.5] * 3000 + [0.9] * 1000)})
    c1, c2, c3, c4, c5 = iv.saturation_profile(panel)
    assert c1.status == iv.FAIL and c1.detail[0]["band"] == "investigate"
    assert c2.detail[0]["shoulder_ratio"] is not None
    assert c3.status == iv.PASS
    assert c4.detail and "p99.9_over_p50" in c4.detail[0]
    assert c5.detail[0]["pct_at_mode"] == pytest.approx(75.0, abs=0.1)


def test_degenerate_basket_rate_fires_when_the_peer_leg_is_mostly_nan():
    """C3. A raw value that exists while its peer leg does not means the basket could not rank
    it -- above 20% the feature is peer-unrankable and belongs on `raw+xs`. This is exactly
    what `f_ic_act_percent_of_class_vs_peers` did at 100%."""
    panel = _panel({"f_ic_inst_ownership_pct": [0.3] * 10,
                    "f_ic_inst_ownership_pct_vs_peers": [1.0] + [np.nan] * 9})
    _, _, c3, _, _ = iv.saturation_profile(panel)
    assert c3.status == iv.FAIL
    assert c3.detail[0]["pct_raw_present_peer_nan"] == pytest.approx(90.0)


def test_decay_check_flags_a_zero_before_the_first_event():
    """V6/V7. Zero-filling a class-S feature makes "no event has ever happened" indistinguishable
    from "an event happened with zero magnitude"."""
    n = 60
    decayed = [0.0] * 20 + list(np.exp(-np.arange(n - 20) / 10.0))
    result = iv.value_decay_behaviour(_panel({"f_ic_act_campaign_intensity": decayed},
                                             tickers=tuple(f"T{i}" for i in range(12))))
    assert result.status == iv.FAIL and result.blocking
    assert result.detail[0]["zeros_before_first_event"] > 0

    clean = [np.nan] * 20 + list(np.exp(-np.arange(n - 20) / 10.0))
    ok = iv.value_decay_behaviour(_panel({"f_ic_act_campaign_intensity": clean},
                                         tickers=tuple(f"T{i}" for i in range(12))))
    assert ok.status == iv.REPORT and ok.detail[0]["pct_decreasing"] > 90


def test_redundancy_matrix_finds_the_duplicate_pair():
    rng = np.random.default_rng(3)
    base = rng.normal(size=2000)
    panel = _panel({"f_ic_inst_a": base.tolist(),
                    "f_ic_inst_b": (base * 2.0 + 0.001).tolist(),
                    "f_ic_inst_c": rng.normal(size=2000).tolist()})
    result = iv.gate_redundancy(panel)
    assert result.detail and result.detail[0]["spearman"] == pytest.approx(1.0, abs=1e-6)
    assert result.detail[0]["flagged"] is True


def test_report_markdown_carries_the_skip_register():
    report = iv.InstitutionalsReport(
        checks=[iv.CheckResult("B1", "behavioural", "The panel moves with no filing", iv.SKIP,
                               measured="Phase 2.6")],
        rows=10, tickers=1, features=1, columns=3,
        date_min=pd.Timestamp("2020-01-01"), date_max=pd.Timestamp("2020-01-10"))
    text = report.to_markdown()
    assert "B1" in text and "SKIP" in text and "1 check(s) not runnable yet" in text
    assert report.blocking_failures == []


def test_skip_register_names_a_phase_for_every_entry():
    """An absent check must say WHO owes it. "Not implemented" with no owner is how B1 -- the
    defining check of the two-layer architecture -- would quietly never run."""
    assert len(iv.SKIPPED) >= 10
    for check_id, group, title, reason in iv.SKIPPED:
        assert check_id and group and title and reason
        # The owner may be a phase, a test module, a document -- or the BUILD's run log, which
        # is where V5's three restatement counters report. A log line is a named, re-readable
        # owner; "not implemented" with nothing after it is not.
        assert any(k in reason for k in ("Phase", "tests/", "unit test", "README", "Needs",
                                         "Synthetic", "Implemented", "`--deep`",
                                         "run log")), check_id


def test_off_universe_tickers_fail_the_scope_check():
    """V12. The part carried 503 tickers against 491 in prices/betas/governance on the
    2026-09-11 build, because `PanelMerger.to_long` is an OUTER-aligned concat and the step
    dropped its `_grid` marker unused. The row count was the small half of it: `_xs` is a
    same-day percentile over the cross-section, so an off-universe column silently changed the
    denominator of every `_xs` leg in the part."""
    panel = _panel({"f_ic_inst_holders": [1.0, 2.0, 3.0]}, tickers=("AAA", "ZZZ"))
    clean = iv.value_universe_scope(panel, {"AAA", "ZZZ"})
    assert clean.status == iv.PASS

    dirty = iv.value_universe_scope(panel, {"AAA"})
    assert dirty.status == iv.FAIL and dirty.blocking
    assert dirty.detail == [{"ticker": "ZZZ", "rows": 3}]

    # and it SKIPs rather than passing when there is nothing to compare against -- a check
    # that silently passes with no reference set is the failure it exists to catch
    assert iv.value_universe_scope(panel, None).status == iv.SKIP
    print("\n=== SANITY CHECK: V12 universe scope ===")
    print("  a ticker the price grid does not have FAILS (blocking) and is named with its row "
          "count; an absent reference set SKIPs instead of passing. Validated.")


def test_a_mandate_gated_numeric_is_scored_in_its_own_window():
    """G2. `FAMILY_SOURCES` keys on a prefix, but `f_ic_bo_` mixes event-shape features running
    to 1996 with NUMERIC ones that are ~0% filled before beneficial-ownership XML became
    mandatory on 2024-12-17. On the 2026-09-11 run those four read 0.24-4.72% and G2 listed
    them as low-coverage — a property of the calendar, not of the feature."""
    dates = pd.bdate_range("2020-01-01", periods=1500)
    mandate = pd.Timestamp("2024-12-17")
    values = [np.nan if d < mandate else 7.5 for d in dates]
    # the SIBLING has the same shape but no override, so it is scored over the whole panel
    panel = _panel({"f_ic_bo_percent_of_class": values,
                    "f_ic_bo_escalation_13g_to_13d": values}, dates=dates)
    floors = {"f_ic_bo_": dates[0]}

    sheet, dropped = iv.gate_coverage(panel, family_floors=floors)
    by = {r["feature"]: r for r in sheet.detail}
    assert by["f_ic_bo_percent_of_class"]["pct_non_null_in_window"] == 100.0, \
        "the mandate-gated numeric must be scored from 2024-12-17, where it is fully dense"
    sibling = by["f_ic_bo_escalation_13g_to_13d"]["pct_non_null_in_window"]
    assert sibling < 20.0, ("the same values scored over the family window read as sparse -- "
                            f"got {sibling}%")
    assert "ic_bo_percent_of_class" in iv.FEATURE_FLOORS
    assert dropped.status in (iv.PASS, iv.REPORT)
    print("\n=== SANITY CHECK: G2 per-feature availability window ===")
    print("  `f_ic_bo_percent_of_class` scores 100% inside its own 2024-12-17 window instead "
          "of ~1% from 1996; a sibling without an override keeps the family window. Validated.")


def test_r4_excludes_the_hole_quarters_it_would_otherwise_compare_across():
    """R4. D17 nulls the LEVEL on a hole quarter and the panel is forward-filled, so at that
    quarter's deadline the panel still carries the PRIOR quarter's share. Comparing it against
    the hole quarter's own source count is a cross-quarter comparison that must fail, and did:
    `FIX` / 2023-12-31 read 35 filers at source against an implied 29.85, where 0.064469 was
    byte-identical to the 2023-09-30 value and held flat from 2023-11-14 to 2024-05-14.

    The fixture is that shape: three healthy quarters and one collapsed to 2% of the pool, with
    a panel that carries each quarter's own share EXCEPT across the hole, where it holds the
    previous one. Excluding the hole must make R4 pass; drawing it must not be possible."""
    periods = ["2015-03-31", "2015-06-30", "2015-09-30", "2015-12-31"]
    pool = [400, 400, 8, 400]                      # 2015-09-30 is the hole
    held = [40, 44, 3, 52]                         # AAA's own filers that quarter
    rows = []
    for p, n_pool, n_held in zip(periods, pool, held):
        for m in range(n_pool):
            rows.append({"cik": f"M{m}", "period": pd.Timestamp(p), "ticker": "AAA"
                         if m < n_held else "BBB", "shares": 100.0})
    holdings = pd.DataFrame(rows)

    # the panel: each quarter's own share, except the hole, which holds the prior quarter's
    dates, values = [], []
    shares = [held[0] / pool[0], held[1] / pool[1], held[1] / pool[1], held[3] / pool[3]]
    for p, share in zip(periods, shares):
        deadline = pd.Timestamp(p) + pd.Timedelta(days=iv.F13_LAG_DAYS)
        for k in range(3):
            dates.append(deadline + pd.Timedelta(days=k))
            values.append(share)
    panel = pd.DataFrame({"date": dates, "ticker": "AAA", "f_ic_inst_holders": values})

    res = iv.reconcile_holder_count(panel, holdings, samples=50)
    drawn = {r["period"] for r in (res.detail or [])}
    assert "2015-09-30" not in drawn, \
        f"the hole quarter was drawn into the R4 sample: {sorted(drawn)}"
    assert res.status is iv.PASS, f"R4 should pass once the hole is excluded: {res.measured}"
    assert "hole quarter" in res.measured, "the exclusion must be reported, not silent"

    print("\n=== SANITY CHECK: R4 hole-quarter exclusion ===")
    print(f"  4 quarters, one collapsed 400 -> 8 filers. Drawn: {sorted(drawn)} -- the hole is "
          f"absent, and the remaining pairs reconcile ({res.measured}). Validated.")


# --------------------------------------------------------------------------- V13

def _filed(profile: dict[str, int]) -> pd.DataFrame:
    """A holdings frame carrying `rows` filings in each named month."""
    stamps = [pd.Timestamp(m) + pd.Timedelta(days=5) for m, n in profile.items() for _ in range(n)]
    return pd.DataFrame({"cik": "M1", "filing_date": stamps})


def _season_shape(years: range, heavy: tuple[int, int] = (250, 400),
                  tail: int = 8) -> dict[str, int]:
    """The measured `sec13f_hr` seasonal shape: two heavy months then a light tail month,
    repeating.

    The RATIOS are the live ones (2026-09-14: tail months 2.3k-28.7k rows against up to 492k
    for a heavy one -- the ~50x within-season swing that rules out an absolute floor); the
    magnitudes are scaled down 1000x on purpose. The check is scale-free, so the shape is the
    only part that carries meaning, and materialising the real counts builds a 13M-row frame
    to test arithmetic that never looks at the row count."""
    out: dict[str, int] = {}
    for y in years:
        for q, (m1, m2, m3) in enumerate(((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12))):
            out[f"{y}-{m1:02d}-01"] = heavy[0]
            out[f"{y}-{m2:02d}-01"] = heavy[1]
            out[f"{y}-{m3:02d}-01"] = tail
    return out


def test_v13_passes_on_a_healthy_13f_filing_axis():
    """The seasonal shape alone must not fire the check -- that is the false-positive the
    relative floor exists to avoid."""
    res = iv.value_filing_coverage("V13a", _filed(_season_shape(range(2020, 2025))), "sec13f_hr")
    assert res.status is iv.PASS, f"healthy seasonal shape scored {res.status}: {res.measured}"
    assert not res.detail

    print("\n=== SANITY CHECK: V13a on a healthy filing axis ===")
    print(f"  5y x 12 months, tail:heavy ratio 8:250:400 (the live ~50x within-season "
          f"swing): {res.measured}. No false fire. Validated.")


def test_v13_fails_naming_the_month_when_a_filing_season_is_lost():
    """The 2024-01/02 and 2025-06..08 defect, reproduced: whole months at zero."""
    profile = _season_shape(range(2020, 2025))
    for lost in ("2022-01-01", "2022-02-01"):
        profile[lost] = 0
    res = iv.value_filing_coverage("V13a", _filed(profile), "sec13f_hr")

    assert res.status is iv.FAIL and res.blocking
    named = {h["bucket"] for h in res.detail}
    assert named == {"2022-01-01", "2022-02-01"}, f"wrong months named: {named}"
    assert "2022-01-01" in res.measured, "the FAIL must name the months, not just count them"
    assert all(h["ratio"] == 0.0 for h in res.detail)

    print("\n=== SANITY CHECK: V13a on a lost filing season ===")
    print(f"  Removed 2022-01 and 2022-02 (the shape of the real 2024-01/02 gap). "
          f"{res.measured}; detail names exactly {sorted(named)}. Validated.")


def test_v13_catches_a_single_lost_month_which_is_why_the_grain_is_the_month():
    """A season loses ONE of its two heavy months. At month grain the ratio is 0.00; the
    season total only falls to ~0.73 and would sail past the floor. This is the measured
    reason `sec13f_hr` is scored monthly rather than by season."""
    profile = _season_shape(range(2020, 2025))
    profile["2022-04-01"] = 0
    res = iv.value_filing_coverage("V13a", _filed(profile), "sec13f_hr")
    assert res.status is iv.FAIL
    assert [h["bucket"] for h in res.detail] == ["2022-04-01"]

    season_ratio = (400 + 8) / (250 + 400 + 8)
    assert season_ratio > iv.FILING_COVERAGE_FLOOR

    print("\n=== SANITY CHECK: V13a month grain vs season grain ===")
    print(f"  One heavy month lost. Month grain: ratio 0.0 -> FAIL ({res.measured}). "
          f"Season grain would read {season_ratio:.2f}, above the {iv.FILING_COVERAGE_FLOOR:.0%} "
          f"floor, and MISS it. The finer grain is load-bearing. Validated.")


def test_v13_season_grain_absorbs_the_elite_tables_filing_timing_noise():
    """`sec13f_manager_holdings` is scored by SEASON because its month axis false-fires.

    At ~75-106 managers, which of a season's two heavy months a manager files in is timing
    noise. Measured 2026-09-14, month grain fires on 2012-04, 2016-04, 2019-04, 2019-10 and
    2021-01 (ratios 0.026-0.157 on 4-34 rows) while every one of those SEASONS is complete --
    five false holes. Season grain fires on none. Reproduced here: one season's managers
    nearly all file in month 2 instead of month 1."""
    profile = {m: (0 if pd.Timestamp(m).month % 3 == 0 else 100)
               for m in _season_shape(range(2018, 2025))}
    profile["2021-01-01"], profile["2021-02-01"] = 3, 197      # the season total is intact

    frame = _filed(profile)
    by_season = iv.value_filing_coverage("V13b", frame, "sec13f_manager_holdings")
    by_month = iv.value_filing_coverage("V13x", frame, "a_table_with_no_declared_grain")

    assert by_season.status is iv.PASS, f"season grain false-fired: {by_season.measured}"
    assert by_month.status is iv.FAIL, "the month grain is supposed to false-fire here"
    assert [h["bucket"] for h in by_month.detail] == ["2021-01-01"]

    print("\n=== SANITY CHECK: V13b season grain absorbs filing-timing noise ===")
    print(f"  A season's filings shift 3/197 across its two heavy months, total unchanged. "
          f"Month grain: {by_month.status} on 2021-01 -- a hole that is not one. "
          f"Season grain: {by_season.status} ({by_season.measured}). This is the measured "
          f"reason the elite table is scored by season. Validated.")


def test_v13_skips_a_calendar_position_the_table_never_fills():
    """The elite table's 59 light tail months are empty by nature, not by loss. Where a
    calendar position's median is 0 there is no expectation to score against, so the bucket is
    SKIPPED rather than failed -- at either grain. This is why the empty tails were never the
    reason for the season grain."""
    profile = {m: (0 if pd.Timestamp(m).month % 3 == 0 else 100)
               for m in _season_shape(range(2018, 2025))}
    res = iv.value_filing_coverage("V13x", _filed(profile), "a_table_with_no_declared_grain")

    assert res.status is iv.PASS, f"a never-filled calendar position was scored: {res.measured}"
    n_empty = sum(1 for m, n in profile.items() if n == 0)

    print("\n=== SANITY CHECK: V13 skips never-filled calendar positions ===")
    print(f"  {n_empty} tail months empty by nature; month grain still {res.status} "
          f"({res.measured}) because their position median is 0. Validated.")


def test_v13_ignores_the_partial_first_and_last_buckets():
    """`filing_date`'s min and max land mid-bucket, so both end buckets are partial by
    construction. Scoring them would fail every run for ever."""
    profile = _season_shape(range(2020, 2025))
    first, last = min(profile), max(profile)
    profile[first] = profile[last] = 1          # both partial, ratio ~0.000004
    res = iv.value_filing_coverage("V13a", _filed(profile), "sec13f_hr")

    assert res.status is iv.PASS, f"a partial boundary bucket was scored: {res.measured}"

    print("\n=== SANITY CHECK: V13a boundary handling ===")
    print(f"  {first} and {last} cut to 1 row each. {res.measured} -- both dropped as partial, "
          f"so the check does not fail on its own window edges. Validated.")
