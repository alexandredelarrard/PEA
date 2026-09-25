"""
tests/validate/test_checks.py
--------------------------------------------------------------------------------------------
ONE test per check function in `src/validate/checks/`, and that is the declared ceiling: nine.

Each test plants a defect whose answer is known by construction and asserts that the check
finds THAT defect and nothing else -- because the failure mode this whole package is built
against is a check that returns a clean result having measured nothing. A test that only
asserts "returns a CheckResult" reproduces the bug it is supposed to catch.

Every test prints its conclusion, so `pytest -q -s` reads as a short report rather than a row
of dots. Offline throughout: a real `DataStore` on in-memory SQLite (`sqlite_store`), so the
store facade under test is the production one.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.constants.constants import INSUFFICIENT_HISTORY_TICKERS
from src.data_store.schema import Tables
from src.validate.checks import (
    check_bounds,
    check_catalogue,
    check_clip,
    check_coverage,
    check_grain,
    check_leakage,
    check_profile,
    check_redundancy,
    check_timeseries,
)
from src.validate.spec import UndeclaredTableError

#: The part the synthetic frames borrow their grain from: `pk == (date, ticker)`, `date_col ==
#: date`, which is the shape every `cube_part_*` table has. Using a REGISTERED table means the
#: checks resolve the real declared pk rather than one the test invented.
PART = Tables.cube_part_momentum

SESSIONS = pd.bdate_range("2024-01-01", periods=40)


class _Ctx:
    """The two attributes every check touches on a `Context`."""

    def __init__(self, store, config):
        self.store, self.config = store, config


def _config(**defaults):
    """The REAL `configs/validate.yml`, with `defaults` overridden per test.

    Loading the shipped file is deliberate: it makes every one of these tests a check that
    the config still carries every key `TableSpec` requires."""
    config = OmegaConf.create({"validate": OmegaConf.load("configs/validate.yml")["validate"], "data_extract": {"redundant_ticks": []}})
    for key, value in defaults.items():
        config.validate.defaults[key] = value
    return config


def _panel(tickers, sessions=SESSIONS, **columns) -> pd.DataFrame:
    """A (date x ticker) panel with the given feature columns, each a callable of (i, ticker)."""
    rows = []
    for ticker in tickers:
        for i, day in enumerate(sessions):
            row = {"date": day, "ticker": ticker}
            row.update({name: fn(i, ticker) for name, fn in columns.items()})
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- #
# grain                                                                                   #
# --------------------------------------------------------------------------------------- #
def test_grain_finds_a_duplicated_declared_key(sqlite_store, capsys):
    """A second row on one (date, ticker) is a broken grain: score 10, and named."""
    frame = _panel(["AAA", "BBB"], f_x=lambda i, t: float(i))
    doubled = pd.concat([frame, frame.iloc[[7]]], ignore_index=True)
    # `save` upserts on the pk and would silently collapse the duplicate -- the whole point
    # is a table that HAS one, so it goes in as a raw append.
    doubled.to_sql("cube_part_momentum", sqlite_store.engine, index=False)
    context = _Ctx(sqlite_store, _config())

    result = check_grain(context, PART, config=context.config)

    assert result.status == "fail"
    assert result.worst_score == 10
    assert result.metrics["rows"] == len(doubled)
    assert result.metrics["distinct_keys"] == len(frame)
    assert result.metrics["duplicate_rows"] == 1
    example = result.findings[0].evidence["examples"][0]
    assert example["ticker"] == frame.iloc[7]["ticker"] and example["rows"] == 2
    with capsys.disabled():
        print(
            f"\n  grain: {result.metrics['rows']} rows over "
            f"{result.metrics['distinct_keys']} distinct (date, ticker) keys -> score "
            f"{result.worst_score}, duplicate named as "
            f"{example['ticker']} {pd.Timestamp(example['date']).date()}"
        )


# --------------------------------------------------------------------------------------- #
# coverage                                                                                #
# --------------------------------------------------------------------------------------- #
def test_coverage_files_the_hole_and_not_the_declared_exclusions(sqlite_store, capsys):
    """THE D-08 REGRESSION GUARD, plus the defect coverage is actually for.

    A universe exclusion absent from the table must not be a defect -- it is absent by
    declaration, and filing it is the false positive `momentum/_scripts/08` recorded. A
    traded session missing INSIDE a ticker's own span must be.
    """
    excluded = sorted(INSUFFICIENT_HISTORY_TICKERS)[0]
    roster = ["AAA", "BBB", excluded]
    sqlite_store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": roster}))
    sqlite_store.save(Tables.prices, _panel(roster, close=lambda i, t: 10.0 + i))

    # AAA complete; BBB missing three sessions in the middle of its own span.
    panel = _panel(["AAA", "BBB"], f_x=lambda i, t: float(i))
    holed = panel.drop(panel[(panel["ticker"] == "BBB") & panel["date"].isin(SESSIONS[20:23])].index)
    sqlite_store.save(PART, holed)
    context = _Ctx(sqlite_store, _config(universe_expected=2))

    result = check_coverage(context, PART, config=context.config)

    named = {f.ticker for f in result.findings if f.score >= 4}
    assert result.metrics["universe"] == 2, "the exclusion must never enter the universe"
    assert excluded not in result.metrics["absent"] and excluded not in named
    assert named == {"BBB"}, f"expected only BBB, got {named}"
    assert result.status == "fail"
    assert result.metrics["interior"]["min"] == pytest.approx(37 / 40)

    # ... and with the hole filled, the same table passes WITH the exclusion still reported.
    sqlite_store.replace(PART, panel)
    clean = check_coverage(context, PART, config=context.config)
    assert clean.status == "pass"
    assert [f.score for f in clean.findings] == [1, 1], "info only: exclusions and depth"
    info = clean.findings[0].evidence
    with capsys.disabled():
        print(
            f"\n  coverage: BBB at {result.metrics['interior']['min']:.1%} of its own traded "
            f"sessions -> filed; {excluded} absent and NOT filed "
            f"({len(info['exclusions'])} declared exclusions reported as info); "
            f"hole filled -> {clean.status.upper()}"
        )


# --------------------------------------------------------------------------------------- #
# profile                                                                                 #
# --------------------------------------------------------------------------------------- #
def test_profile_finds_the_dead_constant_and_infinite_legs(sqlite_store, capsys):
    """The 65-of-179 defect: a column that is null everywhere, and its two cousins.

    `f_dead` is all-NULL, which is the case a dtype filter drops -- pandas reads it back as
    `object`, so this test is also the guard on `frame._is_leg`.
    """
    frame = _panel(
        ["AAA", "BBB"],
        f_good=lambda i, t: float(i) + (0.5 if t == "BBB" else 0.0),
        f_dead=lambda i, t: None,
        f_const=lambda i, t: 1.0,
        f_inf=lambda i, t: (np.inf if i == 3 else float(i)),
    )
    sqlite_store.save(PART, frame)
    context = _Ctx(sqlite_store, _config())

    result = check_profile(context, PART, config=context.config, group=2)

    assert result.status == "fail"
    assert result.metrics["dead"] == ["f_dead"], result.metrics["dead"]
    assert result.metrics["constant"] == ["f_const"]
    assert result.metrics["infinite"] == ["f_inf"]
    by_field = {f.field: f.score for f in result.findings}
    assert by_field == {"f_dead": 10, "f_const": 7, "f_inf": 9}
    good = result.metrics["stats"]["f_good"]
    assert good["n_ok"] == len(frame) and good["n_distinct"] == 80
    assert good["min"] == 0.0 and good["max"] == pytest.approx(39.5)
    # `redundancy` reuses this as its centring constant, so it has to be the real mean.
    assert result.metrics["mean"]["f_good"] == pytest.approx(frame["f_good"].mean())
    with capsys.disabled():
        print(
            f"\n  profile: 4 legs -> dead {result.metrics['dead']}, constant "
            f"{result.metrics['constant']}, infinite {result.metrics['infinite']}; "
            f"f_good n_ok={good['n_ok']} n_distinct={good['n_distinct']} "
            f"p50={good['p50']}"
        )


# --------------------------------------------------------------------------------------- #
# bounds                                                                                  #
# --------------------------------------------------------------------------------------- #
def test_bounds_abstains_undeclared_and_names_the_worst_violation(sqlite_store, capsys):
    """Exit 3 with no declaration; with one, the count AND one (ticker, date, value)."""
    frame = _panel(["AAA", "BBB"], f_pct=lambda i, t: (140.0 if (t == "BBB" and i == 11) else -3.0 if (t == "AAA" and i == 2) else float(i)))
    sqlite_store.save(PART, frame)
    context = _Ctx(sqlite_store, _config())

    # 1. `cube_part_momentum` declares no bounds -> the check must refuse to answer.
    with pytest.raises(UndeclaredTableError) as undeclared:
        check_bounds(context, PART, config=context.config)
    assert "bounds" in str(undeclared.value)

    # 2. with a declaration, the violations are counted and the worst one is named.
    result = check_bounds(context, PART, config=context.config, bounds={"f_pct": (0.0, 100.0)})
    assert result.status == "fail"
    finding = result.findings[0]
    assert finding.score == 9 and finding.field == "f_pct"
    assert finding.evidence["n_violations"] == 2
    assert finding.evidence["worst"]["ticker"] == "BBB"
    assert finding.evidence["worst"]["value"] == pytest.approx(140.0)
    with capsys.disabled():
        print(
            f"\n  bounds: undeclared -> ABSTAIN (exit 3); declared [0, 100] -> "
            f"{finding.evidence['n_violations']} violations, worst "
            f"{finding.evidence['worst']['ticker']} "
            f"{pd.Timestamp(finding.evidence['worst']['date']).date()} = "
            f"{finding.evidence['worst']['value']}"
        )


# --------------------------------------------------------------------------------------- #
# redundancy                                                                              #
# --------------------------------------------------------------------------------------- #
def test_redundancy_matches_pandas_and_names_the_duplicated_leg(sqlite_store, capsys):
    """The streaming co-moment accumulator against the reference implementation.

    `DataFrame.corr()` is pairwise-complete by construction, so it is exactly what the
    accumulator claims to reproduce -- on a frame with holes punched in different places per
    column, which is the only case where a naive "drop rows with any NaN" would disagree.
    One pair is also EXACTLY duplicated, and must come back as score 10 rather than as a
    correlation that happens to round to 1.
    """
    rng = np.random.default_rng(20260921)
    # Four names, so the pairs clear `redundancy._MIN_PAIRWISE_N` -- an r of 1.00 over a
    # handful of shared rows is arithmetic, and the check declines to report it.
    n = len(SESSIONS) * 4
    base = rng.normal(size=n)
    frame = _panel(["AAA", "BBB", "CCC", "DDD"], f_x=lambda i, t: 0.0)
    frame["f_x"] = base
    frame["f_copy"] = base  # the same column, twice
    frame["f_near"] = base * 3.0 + 0.5 + rng.normal(scale=0.01, size=n)
    frame["f_free"] = rng.normal(size=n)
    # Holes in DIFFERENT places per column -- this is what makes it a pairwise-complete test.
    frame.loc[frame.index[:7], "f_near"] = np.nan
    frame.loc[frame.index[10:18], "f_free"] = np.nan
    frame.loc[frame.index[20:24], "f_x"] = np.nan
    sqlite_store.save(PART, frame)
    context = _Ctx(sqlite_store, _config())

    result = check_redundancy(context, PART, config=context.config, chunksize=17)  # many chunks, so the accumulation is exercised

    legs = ["f_copy", "f_free", "f_near", "f_x"]
    reference = frame[legs].corr()
    got = {(p["a"], p["b"]): p for p in result.metrics["top_pairs"]}
    for (a, b), pair in got.items():
        assert pair["r"] == pytest.approx(reference.loc[a, b], abs=1e-10), (a, b)

    identical = [p for p in result.metrics["top_pairs"] if p["identical"]]
    assert len(identical) == 1 and {identical[0]["a"], identical[0]["b"]} == {"f_x", "f_copy"}
    assert identical[0]["n"] == identical[0]["exact_equal"] == int(frame["f_x"].notna().sum())
    assert max(f.score for f in result.findings) == 10
    assert ("f_free", "f_x") not in got and ("f_x", "f_free") not in got
    with capsys.disabled():
        print(
            f"\n  redundancy: {result.metrics['pairs_tested']} pairs, every r equal to "
            f"pandas' pairwise-complete corr to 1e-10; f_x ~ f_copy identical on all "
            f"{identical[0]['n']} shared rows -> score 10; the independent pair not filed"
        )


# --------------------------------------------------------------------------------------- #
# clip                                                                                    #
# --------------------------------------------------------------------------------------- #
def test_clip_finds_the_clip_mass_and_the_plateau(sqlite_store, capsys):
    """A peer-z leg pinned to its clip, and a rank leg that is mostly one tie."""
    # 12 names so a cross-section clears min_tickers_xs; a third of each sits on the clip,
    # and the `_xs` leg gives 8 of 12 names the same rank on every date.
    names = [f"T{i:02d}" for i in range(12)]  # `min_tickers_xs` is lowered to 10 below
    frame = _panel(
        names,
        f_a_vs_peers=lambda i, t: 8.0 if int(t[1:]) < 4 else float(int(t[1:])) / 10.0,
        f_a_xs=lambda i, t: 0.5 if int(t[1:]) < 8 else float(int(t[1:])) / 100.0,
        f_b_vs_peers=lambda i, t: float(int(t[1:])) / 10.0,
    )
    sqlite_store.save(PART, frame)
    config = _config(min_tickers_xs=10)
    config.validate.tables[PART.name] = {"xs_suffix": "_xs", "peer_suffix": "_vs_peers", "clip_peer": 8.0}
    context = _Ctx(sqlite_store, config)

    result = check_clip(context, PART, config=config)

    assert result.metrics["peer_over_limit"] == ["f_a_vs_peers"], "f_b is nowhere near the clip"
    assert result.metrics["tie_over_limit"] == ["f_a_xs"]
    assert result.metrics["peer"]["f_a_vs_peers"]["share"] == pytest.approx(4 / 12)
    assert result.metrics["peer"]["f_b_vs_peers"]["share"] == 0.0
    assert result.metrics["tie"]["f_a_xs"]["mean_modal_share"] == pytest.approx(8 / 12)
    assert result.status == "fail"
    # ... and a table declaring no suffix convention must refuse to answer at all.
    del config.validate.tables[PART.name]
    with pytest.raises(UndeclaredTableError):
        check_clip(context, PART, config=config)
    with capsys.disabled():
        print(
            f"\n  clip: f_a_vs_peers {result.metrics['peer']['f_a_vs_peers']['share']:.1%} on "
            f"the +/-8 clip, f_b_vs_peers 0.0%; f_a_xs modal share "
            f"{result.metrics['tie']['f_a_xs']['mean_modal_share']:.1%}; "
            f"no declared suffix -> ABSTAIN"
        )


# --------------------------------------------------------------------------------------- #
# catalogue                                                                               #
# --------------------------------------------------------------------------------------- #
def test_catalogue_asserts_both_directions(sqlite_store, tmp_path, capsys):
    """The dead catalogued name scores higher than the undocumented live one, and both fire."""
    frame = _panel(["AAA"], f_live=lambda i, t: float(i), f_undocumented=lambda i, t: 1.0 * i)
    sqlite_store.save(PART, frame)
    context = _Ctx(sqlite_store, _config())

    path = tmp_path / "catalogue.json"
    path.write_text(
        json.dumps({"f_live": "a described, live leg", "f_renamed_away": "described, but no such column", "f_blank": ""}), encoding="utf-8"
    )

    result = check_catalogue(context, PART, config=context.config, catalogue=path)

    by_field = {f.field: f.score for f in result.findings}
    assert by_field["f_renamed_away"] == 9, "a catalogued name with no column is the _sec bug"
    assert by_field["f_undocumented"] == 4
    assert "f_live" not in by_field
    assert result.metrics["catalogued_not_live"] == ["f_blank", "f_renamed_away"]
    assert result.metrics["live_not_catalogued"] == ["f_undocumented"]
    # ... a catalogue written in a different naming convention is ONE problem, not 2N. Both
    # of this repo's catalogue files key on `ic_act_percent_of_class` against a live
    # `f_ic_act_percent_of_class`, which asserted blind reads as 216 defects and no matches.
    shifted = tmp_path / "shifted.json"
    shifted.write_text(json.dumps({"live": "same leg, no f_ prefix", "undocumented": "likewise"}), encoding="utf-8")
    convention = check_catalogue(context, PART, config=context.config, catalogue=shifted)
    assert convention.status == "abstain"
    assert "prefixing them with `f_`" in convention.reason
    # ... and with no catalogue at all the check must not invent one.
    with pytest.raises(UndeclaredTableError):
        check_catalogue(context, PART, config=context.config)
    with capsys.disabled():
        print(
            "\n  catalogue: both directions -- f_renamed_away catalogued but absent "
            "(score 9), f_undocumented live but undescribed (score 4), f_live clean; "
            "no --catalogue -> ABSTAIN"
        )


# --------------------------------------------------------------------------------------- #
# timeseries                                                                              #
# --------------------------------------------------------------------------------------- #
def test_timeseries_separates_a_hole_a_freeze_and_a_jump(sqlite_store, capsys):
    """Three kernels, three planted defects, and the FROZEN gate asserted in both directions.

    The jump leg is the load-bearing one: it must clear BOTH gates while the smooth legs
    clear neither -- the z gate alone flagged 296,926 steps on the live part, because a
    near-zero MAD makes every real move astronomical. And FROZEN must refuse to run on a
    table that declares no `daily_legs`: `cube_part_momentum` is daily and every leg in it is
    a cross-sectional percentile, where a name holding its rank holds its value."""

    def hole(i, ticker):  # a 6-row gap, AAA only
        return np.nan if (ticker == "AAA" and 15 <= i < 21) else round(0.01 * i, 4)

    def jump(i, ticker):  # one +500 spike, AAA only
        return float(i) + (500.0 if (ticker == "AAA" and i == 20) else 0.0)

    frame = _panel(["AAA", "BBB"], f_hole=hole, f_jump=jump, f_flat=lambda i, t: 5.0)
    sqlite_store.save(PART, frame)
    context = _Ctx(sqlite_store, _config())

    result = check_timeseries(context, PART, config=context.config)

    by_field = {f.field: f for f in result.findings}
    assert result.status == "fail"
    assert set(result.metrics["legs_with_holes"]) == {"f_hole"}
    assert set(result.metrics["legs_with_jumps"]) == {"f_jump"}, "a smooth series must clear neither gate"
    assert by_field["f_hole"].evidence["worst"]["days"] == 6
    assert by_field["f_hole"].ticker == "AAA"
    # Jumps are INFO by design: reported and ranked, never the reason a run is red.
    assert by_field["f_jump"].score == 3
    assert by_field["f_jump"].evidence["worst"]["change_over_span"] >= 0.5
    # ... and f_flat never moves, yet nothing is filed: momentum declares no `daily_legs`.
    assert result.metrics["n_frozen"] == 0 and result.scope["frozen_legs"] == 0
    assert "ABSTAINED" in result.scope["frozen_scope"] and "ABSTAINED" in result.reason

    # -- the same kernel on a table that DOES declare which legs are rebuilt daily -------- #
    daily = _panel(
        ["AAA"],
        f_ic_stall_to_mcap=lambda i, t: 0.5,  # flat, interior
        f_ic_rank_to_mcap=lambda i, t: round(0.5 + 0.01 * i, 4) if i < 8 else 1.0,
        f_ic_quarterly_holding=lambda i, t: 7.0,
    )  # flat, NOT a daily leg
    sqlite_store.save(Tables.cube_part_institutionals, daily)
    frozen = check_timeseries(context, Tables.cube_part_institutionals, config=context.config)

    scored = {f.field: f for f in frozen.findings}
    assert set(frozen.metrics["legs_with_frozen"]) == {"f_ic_stall_to_mcap"}, "a leg the builder does not rebuild daily is allowed to hold its value"
    assert scored["f_ic_stall_to_mcap"].score == 7
    # A run pinned at the series' own extreme is a saturation, not a stalled input: measured
    # on the live part, AAPL held `dollar_volume_63 == 1.0` for 2,071 sessions because it was
    # the largest dollar volume in the universe on every one of them.
    assert set(frozen.metrics["legs_saturated"]) == {"f_ic_rank_to_mcap"}
    assert scored["f_ic_rank_to_mcap"].score == 3
    assert scored["f_ic_rank_to_mcap"].evidence["worst"]["days"] == len(SESSIONS) - 8

    # -- conditional sparsity and one exact source-ineligible interval ----------------- #
    conditional = _panel(
        ["SMCI"],
        f_ic_sig_insider_age_days=lambda i, t: 200.0 if 6 <= i < 12 else 10.0,
        f_ic_sig_insider_price_vs_buy=lambda i, t: (np.nan if 6 <= i < 12 or 22 <= i < 28 else float(i)),
        f_ic_shortvol_ratio_5d=lambda i, t: np.nan if 15 <= i < 21 else float(i),
        f_ic_shortvol_ratio_20d=lambda i, t: np.nan if 15 <= i < 24 else float(i),
    )
    sqlite_store.replace(Tables.cube_part_institutionals, conditional)
    context.config.validate.tables.cube_part_institutionals.known_ineligible = [
        {
            "ticker": "SMCI",
            "fields": ["f_ic_shortvol_"],
            "start": str(SESSIONS[15].date()),
            "end": str(SESSIONS[20].date()),
            "reason": "known source suspension in this synthetic interval",
        },
        {
            "ticker": "SMCI",
            "fields": ["f_ic_shortvol_ratio_20d"],
            "start": str(SESSIONS[15].date()),
            "end": str(SESSIONS[23].date()),
            "reason": "20-session source gap plus warm-up after the source resumed",
        },
    ]
    classified = check_timeseries(context, Tables.cube_part_institutionals, config=context.config)
    assert classified.status == "fail", "the active-anchor hole must remain actionable"
    assert classified.metrics["legs_with_holes"] == ["f_ic_sig_insider_price_vs_buy"]
    assert classified.metrics["n_holes"] == 1
    assert classified.metrics["n_explained_holes"] == 3
    explained_fields = {row["leg"] for row in classified.metrics["worst_explained_holes"]}
    assert explained_fields == {
        "f_ic_sig_insider_price_vs_buy",
        "f_ic_shortvol_ratio_20d",
        "f_ic_shortvol_ratio_5d",
    }
    with capsys.disabled():
        worst_jump = by_field["f_jump"].evidence["worst"]
        print("")
        print(
            f"  timeseries: f_hole 6-row gap (score 6), f_jump z={worst_jump['z']:,.1f} "
            f"at {worst_jump['change_over_span']:.2f}x its p1-p99 span (score 3, info); "
            f"FROZEN ABSTAINS on momentum (no daily_legs) and on institutionals files "
            f"f_ic_stall_to_mcap at 7, f_ic_rank_to_mcap (pinned at its own max) at 3, "
            f"and the non-daily leg not at all; conditional holes: one anchor-active "
            f"price_vs_buy gap still fails, while one anchor-free gap and the exact SMCI "
            f"source interval and its field-specific rolling warm-up are explained "
            f"(never zero-filled)"
        )


# --------------------------------------------------------------------------------------- #
# leakage                                                                                 #
# --------------------------------------------------------------------------------------- #
def test_leakage_catches_a_label_that_reaches_the_last_price(sqlite_store, capsys):
    """Both halves: the horizon ladder must recede, and no feature may precede its source.

    The horizon half is asserted in both directions -- clean first, then the SAME frame with
    one horizon pushed to the last price session -- because a check that cannot fail is not
    evidence that the table is clean."""
    prices = _panel(["AAA", "BBB"], close=lambda i, t: 100.0 + i)[["date", "ticker", "close"]]
    prices.to_sql("prices", sqlite_store.engine, index=False)
    last_session = SESSIONS[-1]

    def label(offset):  # non-null until `offset` sessions from the end
        return lambda i, t: float(i) if i < len(SESSIONS) - offset else np.nan

    # TWO families at the SAME three horizons, which is the live shape: cube_part_targets
    # carries target_rank_*, target_zscore_* and target_epsilon_*. Legs at one horizon end on
    # one day BECAUSE they are one horizon, and comparing across families reported six leaks
    # on the live part where there were none.
    clean = _panel(
        ["AAA", "BBB"], f_ret_h30=label(5), f_ret_h60=label(10), f_ret_h90=label(15), f_vol_h30=label(5), f_vol_h60=label(10), f_vol_h90=label(15)
    )
    sqlite_store.save(Tables.cube_part_targets, clean)
    context = _Ctx(sqlite_store, _config())

    ok = check_leakage(context, Tables.cube_part_targets, config=context.config)
    ladder = [e for e in ok.metrics["horizon_ladder"] if e["family"] == "f_ret"]
    assert ok.status == "pass", ok.findings
    assert ok.metrics["horizon_families"] == ["f_ret", "f_vol"]
    assert [e["horizon_days"] for e in ladder] == [30, 60, 90]
    # 5 sessions apart, asserted in the CALENDAR days the ladder reports -- a business-day
    # grid makes those two different numbers, and the check must not silently mean sessions.
    step = (SESSIONS[34] - SESSIONS[29]).days
    assert all(e["recedes_by"] == step for e in ladder[1:])
    assert all(e["days_behind_last_price"] > 0 for e in ladder)
    assert "pit_sources" in ok.reason, "cube_part_targets declares none -- say so, do not pass"

    # ... and the same check on a frame whose 90-day label reaches the last price session.
    leaked = clean.copy()
    leaked.loc[leaked["date"] == last_session, "f_ret_h90"] = 1.0
    sqlite_store.replace(Tables.cube_part_targets, leaked)
    bad = check_leakage(context, Tables.cube_part_targets, config=context.config)
    assert bad.status == "fail" and bad.worst_score == 10
    assert {f.field for f in bad.findings} == {"f_ret_h90"}, "f_vol_h90 is clean and must not be dragged in by its sibling family"
    assert len(bad.findings) == 2, "it reaches the last price AND stops receding"

    # -- the point-in-time half, on a table that declares its sources --------------------- #
    # FTD absence is an observed zero once the source is globally live. BBB deliberately has
    # no event at all: requiring its own first event would manufacture a leak on the correct 0.
    ftd = pd.DataFrame({"ticker": ["AAA"], "date": [SESSIONS[10]], "fails_quantity": [1.0]})
    ftd.to_sql("sec_fails_to_deliver", sqlite_store.engine, index=False)
    short = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "date": [SESSIONS[10], SESSIONS[0]],
            "short_volume": [1.0, 2.0],
            "total_volume": [10.0, 20.0],
        }
    )
    short.to_sql("sec_short_interest", sqlite_store.engine, index=False)

    early_panel = _panel(
        ["AAA", "BBB"],
        f_ic_ftd_pct_so=lambda i, t: float(i),
        f_ic_shortvol_ratio_5d=lambda i, t: float(i),
    )
    sqlite_store.save(Tables.cube_part_institutionals, early_panel)
    global_early = check_leakage(context, Tables.cube_part_institutionals, config=context.config)
    ftd_leak = next(f for f in global_early.findings if f.field == "f_ic_ftd_pct_so")
    assert ftd_leak.evidence["n_tickers"] == 2
    assert ftd_leak.evidence["worst"]["availability_mode"] == "global_observed_zero"

    panel = _panel(
        ["AAA", "BBB"],
        f_ic_ftd_pct_so=lambda i, t: 0.0 if i >= 10 else np.nan,
        f_ic_shortvol_ratio_5d=lambda i, t: float(i),
    )
    sqlite_store.replace(Tables.cube_part_institutionals, panel)

    pit = check_leakage(context, Tables.cube_part_institutionals, config=context.config)
    assert "f_ic_ftd_pct_so" not in {finding.field for finding in pit.findings}
    leak = next(f for f in pit.findings if f.field == "f_ic_shortvol_ratio_5d")
    assert pit.status == "fail" and leak.score == 10
    assert leak.ticker == "AAA" and leak.evidence["n_tickers"] == 1
    assert leak.evidence["worst"]["lead_days"] == (SESSIONS[10] - SESSIONS[0]).days
    assert leak.evidence["worst"]["availability_mode"] == "per_ticker_event"
    ftd_sheet = next(row for row in pit.metrics["pit_sheet"] if row.get("leg") == "f_ic_ftd_pct_so")
    assert ftd_sheet["availability_mode"] == "global_observed_zero"
    assert ftd_sheet["tickers_without_source"] == 0
    assert "pit" in pit.scope["halves_run"]
    with capsys.disabled():
        print(
            f"\n  leakage: clean ladder h30 -> h60 -> h90 receding 5 sessions each and all "
            f"behind {last_session.date()} (step {step}d) -> PASS; the same frame with h90 pushed to "
            f"{last_session.date()} -> 2 findings at score 10; PIT permits BBB's observed "
            f"FTD zero after global source onset despite no BBB event, while still catching "
            f"AAA short volume {leak.evidence['worst']['lead_days']} days before its own "
            f"RegSHO event"
        )
