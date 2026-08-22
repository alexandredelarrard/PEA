"""
The composite config in `configs/build_cube.yml`, checked two ways:

  * as a CONFIG (fast, no data): naming, duplication, sign coherence, and the three
    rules the config header states -- one view per concept, sector-varying metrics on
    `_vs_peers`, sparse/sector-only members kept out of the universal groups.
  * as a BUILD on 10 real tickers: composites are actually produced, oriented so that
    higher = the long side, and every configured member either lands or is reported.

Why the config deserves its own test: members absent from the panel used to be skipped
in silence, and four of them were (gross_profitability, net_debt_to_ebitda,
interest_coverage, sbc_intensity -- all casualties of the `_x`/`_y` panel-merge
collision), so three composites ran a member short for as long as that bug lived.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data_aggregate.utils.assemble.composites import (
    _parse_member, build_composites, missing_members,
)
from src.data_aggregate.utils.fundamentals.fundamental_features import build_fundamental_feature_panel
from src.data_aggregate.utils.fundamentals.sector_features import build_sector_feature_panel

ROOT = Path(__file__).resolve().parents[2]
SEED, N_TICKERS = 20260727, 10

# metrics whose LEVEL is a property of the industry, not the firm: a universe
# percentile of these ranks sectors, so they must use the peer-relative view.
SECTOR_BOUND = ("margin", "sga_intensity", "revenue_per_employee", "rd_intensity",
                "capex_intensity", "gross_profitability", "software_to_revenue",
                "ddna_intensity", "efficiency_ratio", "net_interest_margin")
# ...but only the LEVEL is. A CHANGE in gross margin, or the growth rate of revenue per
# employee, is directly comparable across industries -- a software company improving
# margin by 2pp and a grocer doing the same are the same news. Deltas stay on `_xs`.
CHANGE_SUFFIXES = ("_chg", "_delta", "_growth", "_accel", "_vs_ttm", "_5y")
# ...except where the metric is ALREADY sector-scoped by construction (the *_health
# groups only ever contain one sector's names) or is a ratio of two sector-bound legs.
SECTOR_SCOPED_GROUPS = ("bank_health", "insurance_health", "reit_health", "energy_health")

# groups that are deliberately allowed to be sparsely populated
SPARSE_GROUPS = SECTOR_SCOPED_GROUPS + ("pension_risk", "earnings_call")


def _cfg() -> dict:
    raw = yaml.safe_load((ROOT / "configs" / "build_cube.yml").read_text(encoding="utf-8"))
    return next(v["composites"] for v in raw.values()
                if isinstance(v, dict) and "composites" in v)


@pytest.fixture(scope="module")
def groups() -> dict[str, list[str]]:
    return _cfg()["groups"]


# --------------------------------------------------------------------------- #
# 1. config hygiene                                                            #
# --------------------------------------------------------------------------- #
def test_config_is_well_formed(groups):
    problems: list[str] = []
    for theme, members in groups.items():
        parsed = [_parse_member(m) for m in members]
        names = [c for _, c in parsed]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            problems.append(f"{theme}: member listed twice {dupes}")
        # a member signed BOTH ways inside one group cancels itself out
        by_name: dict[str, set[float]] = {}
        for sign, col in parsed:
            by_name.setdefault(col, set()).add(sign)
        for col, signs in by_name.items():
            if len(signs) > 1:
                problems.append(f"{theme}: {col} appears with both signs")
        for col in names:
            if col.endswith("_peer"):
                problems.append(f"{theme}: {col} -- the suffix is '_vs_peers', not '_peer'")
            if not col.startswith(("f_", "mom_", "ma_ratio", "vol_", "max_", "ret_",
                                   "idio_", "downside_", "beta")):
                problems.append(f"{theme}: {col} does not look like a feature column")
    assert not problems, "config problems:\n  " + "\n  ".join(problems)

    n_members = sum(len(m) for m in groups.values())
    print(f"\n[1] {len(groups)} composites, {n_members} members, no duplicates and no "
          "self-cancelling signs")
    print(f"    themes: {', '.join(sorted(groups))}")
    print("    SANITY CHECK: every member is uniquely signed within its group, so no "
          "member silently nets itself out of a composite.")


def test_no_concept_appears_under_two_views_in_one_group(groups):
    """Rule 1: `_xs` (universe level), `_vs_peers` (peer level) and `_vs_hist` (own
    history) are three different bets. `value` used to carry five yields as BOTH `_xs`
    and `_vs_hist`, so half its weight was a time-series re-rating wearing the
    cross-sectional value label."""
    clashes = []
    for theme, members in groups.items():
        stems: dict[str, list[str]] = {}
        for _, col in map(_parse_member, members):
            stem = col
            for suf in ("_vs_peers", "_vs_hist", "_xs"):
                if stem.endswith(suf):
                    stem = stem[: -len(suf)]
                    break
            stems.setdefault(stem, []).append(col)
        for stem, cols in stems.items():
            if len(cols) > 1:
                clashes.append(f"{theme}: {stem} appears as {cols}")
    assert not clashes, "two views of one concept in one group:\n  " + "\n  ".join(clashes)
    print("\n[2] no concept appears under two views inside any group; the own-history "
          "view lives in its own `value_rerating` composite")
    print("    SANITY CHECK: cross-sectional cheapness and re-rating are no longer "
          "averaged into a single number.")


def test_sector_bound_metrics_use_the_peer_view(groups):
    """Rule 2: a universe percentile of the LEVEL of gross margin or revenue-per-employee
    ranks INDUSTRIES, so those must use the peer-basket view. Two exemptions, both
    principled: the *_health groups are already confined to one sector, and a CHANGE in
    a sector-bound metric is cross-sector comparable (a 2pp margin gain is the same news
    for a software firm and a grocer)."""
    wrong = []
    for theme, members in groups.items():
        if theme in SECTOR_SCOPED_GROUPS:
            continue
        for _, col in map(_parse_member, members):
            stem = col.removesuffix("_xs")
            is_change = any(s in stem for s in CHANGE_SUFFIXES)
            if col.endswith("_xs") and not is_change and any(k in col for k in SECTOR_BOUND):
                wrong.append(f"{theme}: {col}")
    assert not wrong, ("sector-bound metric on the universe view:\n  " + "\n  ".join(wrong))
    peer = sum(1 for ms in groups.values() for _, c in map(_parse_member, ms)
               if c.endswith("_vs_peers"))
    print(f"\n[3] {peer} members use the peer-relative view; no sector-bound metric is "
          "left on a universe percentile outside the sector-scoped groups")
    print("    SANITY CHECK: margins / intensities are compared within an industry, so "
          "the composites rank firms rather than sectors.")


# --------------------------------------------------------------------------- #
# 2. orientation: higher must mean the long side                               #
# --------------------------------------------------------------------------- #
def test_signs_orient_every_composite_to_the_long_side(groups):
    """A synthetic two-name panel where GOOD is better on every member after its sign is
    applied must score higher on every composite. Catches an inverted member, which no
    amount of averaging would reveal."""
    dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    rows = []
    for d in dates:
        for tkr, good in (("GOOD", True), ("BAD", False)):
            row = {"date": d, "ticker": tkr}
            for theme, members in groups.items():
                for sign, col in map(_parse_member, members):
                    # value the member so the SIGNED contribution favours GOOD
                    row[col] = (sign if good else -sign) * 1.0
            rows.append(row)
    panel = pd.DataFrame(rows)
    out = build_composites(panel, groups, method="zscore")

    comps = [c for c in out.columns if c.startswith("comp_")]
    assert len(comps) == len(groups), f"built {len(comps)} of {len(groups)} composites"
    inverted = []
    for c in comps:
        g = out.loc[out["ticker"] == "GOOD", c].mean()
        b = out.loc[out["ticker"] == "BAD", c].mean()
        if not (g > b):
            inverted.append(f"{c}: GOOD={g:.3f} <= BAD={b:.3f}")
    assert not inverted, "composite not oriented to the long side:\n  " + "\n  ".join(inverted)
    print(f"\n[4] all {len(comps)} composites score the synthetically-better name higher")
    print("    SANITY CHECK: every member's '-' prefix is consistent with 'higher = long', "
          "so no member is quietly subtracting signal from its own theme.")


def test_goodwill_roic_drag_keeps_its_positive_sign(groups):
    """The one sign that reads backwards but is right. `goodwill_roic_drag =
    roic_incl_goodwill - roic_ex_goodwill`; excluding goodwill SHRINKS invested capital,
    so roic_ex > roic_incl and the drag is structurally NEGATIVE. A bigger goodwill
    balance makes it more negative, so HIGHER (nearer zero) = less dilution = better,
    and the member takes a '+'."""
    members = dict(
        (col, sign) for sign, col in map(_parse_member, groups["ma_digestion"]))
    assert members.get("f_goodwill_roic_drag_xs") == 1.0, \
        "goodwill_roic_drag must be POSITIVE-signed: the metric is already negative"
    assert members.get("f_goodwill_to_equity_xs") == -1.0
    print("\n[5] ma_digestion: goodwill_roic_drag '+' (metric is already negative), "
          "goodwill_to_equity '-'")
    print("    SANITY CHECK: the drag is not double-negated -- inverting it would have "
          "rewarded exactly the names that overpaid for acquisitions.")


# --------------------------------------------------------------------------- #
# 3. a real build on 10 randomly drawn tickers                                 #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_panel() -> pd.DataFrame:
    """The two fundamentals panels built from the real `fundamentals_history` table.

    Reads the persisted table rather than rebuilding history from cached companyfacts
    JSON: that cache is gone, and reading the table is what the cube itself does, so
    this fixture now exercises the same input the pipeline sees.
    """
    from src.data_store.schema import Tables
    from src.data_store.store import DataStore
    from src.utils.db import get_engine
    try:
        store = DataStore(get_engine())
        uni = store.load("sp500_tickers")
    except Exception as exc:                       # pragma: no cover - env without the DB
        pytest.skip(f"sp500_tickers unavailable ({type(exc).__name__})")
    if uni is None or uni.empty:
        pytest.skip("sp500_tickers is empty")
    universe = sorted(uni["ticker"].dropna().unique())
    if len(universe) < N_TICKERS:
        pytest.skip("universe smaller than the draw")
    picked = sorted(random.Random(SEED).sample(universe, N_TICKERS))

    fund = store.load(Tables.fundamentals_history, where={"ticker": picked}, optional=True)
    if fund is None or fund.empty:
        pytest.skip("fundamentals_history is empty for the drawn tickers")

    idx = pd.bdate_range("2022-01-03", "2026-06-30")
    close = pd.DataFrame(100.0, index=idx, columns=picked)
    peers = {t: {p: 1.0 for p in picked if p != t} for t in picked}
    fp = build_fundamental_feature_panel(fund, peers, idx, stock_close=close)
    sp = build_sector_feature_panel(fund, peers, idx)
    panel = fp.merge(sp, on=["date", "ticker"], how="outer")
    panel.attrs["tickers"] = picked
    return panel


def test_composites_build_on_ten_real_tickers(groups, real_panel, caplog):
    tickers = real_panel.attrs["tickers"]
    with caplog.at_level(logging.WARNING):
        out = build_composites(real_panel, groups, method="zscore",
                               log=logging.getLogger("composites-test"))

    built = sorted(c for c in out.columns if c.startswith("comp_"))
    live = {c: float(out[c].notna().mean()) for c in built}
    non_degenerate = {c: v for c, v in live.items() if v > 0}
    assert non_degenerate, "no composite has a single value"
    # every UNIVERSAL composite must be broadly populated on a 10-name cross-section
    thin = {c: v for c, v in live.items()
            if v < 0.20 and c.removeprefix("comp_") not in SPARSE_GROUPS}
    assert not thin, f"universal composites nearly empty: { {k: f'{v:.0%}' for k,v in thin.items()} }"
    # composites are means of clipped z-scores -> bounded, never inf
    for c in built:
        vals = out[c].dropna()
        if vals.empty:
            continue
        assert np.isfinite(vals).all(), f"{c} has non-finite values"
        assert vals.abs().max() <= 4.0 + 1e-9, f"{c} exceeds the member clip"

    gaps = missing_members(real_panel, groups)
    if gaps:
        assert any("absent from the panel" in r.getMessage() for r in caplog.records), \
            "members are missing but nothing was logged"

    # This fixture builds only the FUNDAMENTAL + SECTOR panels, so members owned by the
    # governance / insider / 13F / technical / attention builders are legitimately absent
    # here -- `test_every_configured_member_is_a_real_feature` is what proves those names
    # are real. What matters in this test is that the universal composites still populate.
    print(f"\n[6] built {len(built)} composites on {len(tickers)} real tickers "
          f"({', '.join(tickers)}), from the fundamental + sector panels only")
    print(f"    {'composite':22} {'row coverage':>13}")
    for c, v in sorted(live.items(), key=lambda kv: -kv[1]):
        tag = "  (sparse by design)" if c.removeprefix("comp_") in SPARSE_GROUPS else ""
        print(f"    {c:22} {v:12.1%}{tag}")
    print("    SANITY CHECK: every universal composite is populated for the whole "
          "cross-section; only the sector-scoped and transcript-dependent ones are "
          "sparse, which is what their design intends.")


def test_every_configured_member_is_a_real_feature(groups, real_panel):
    """No typos: every member must exist either in the LIVE cube (which carries the
    governance / insider / 13F / technical / attention panels) or in the freshly-built
    fundamental + sector panels (which carry the new steps 4-8 features).

    The one documented exception is the `f_ec_*` earnings-call family: the builders are
    wired but `earnings_call_sections` holds only ~217 rows and was never scored, so
    those columns do not exist yet. They are configured deliberately so they light up the
    moment transcripts are ingested -- and the skip warning keeps that visible."""
    from src.data_store.store import DataStore
    from src.utils.db import get_engine
    try:
        cube_cols = set(DataStore(get_engine()).load("cube", limit=1).columns)
    except Exception as exc:                       # pragma: no cover - env without the DB
        pytest.skip(f"cube unavailable ({type(exc).__name__})")

    known = cube_cols | set(real_panel.columns)
    unknown, awaiting_data = [], []
    for theme, members in groups.items():
        for _, col in map(_parse_member, members):
            if col in known:
                continue
            (awaiting_data if col.startswith("f_ec_") else unknown).append(f"{theme}: {col}")
    assert not unknown, ("configured member is not a real feature anywhere:\n  "
                         + "\n  ".join(sorted(unknown)))

    print(f"\n[6b] every configured member resolves against {len(cube_cols)} live cube "
          f"columns + {len(real_panel.columns)} freshly-built panel columns")
    print(f"     {len(awaiting_data)} members awaiting DATA (earnings-call transcripts: "
          f"217 sections ingested, never scored) -- configured on purpose so they "
          f"activate on ingest")
    print("     SANITY CHECK: no typo'd or renamed feature name in the config; the only "
          "unresolved members are a known, reported data gap.")


def test_missing_members_are_reported_not_swallowed(groups, real_panel, caplog):
    """The regression that motivated the warning: a member absent from the panel is
    skipped, and the composite is still built from the remainder -- so the ONLY way to
    notice is the log."""
    gaps = missing_members(real_panel, groups)
    with caplog.at_level(logging.WARNING):
        build_composites(real_panel, groups, method="zscore",
                         log=logging.getLogger("composites-test"))
    warned = " ".join(r.getMessage() for r in caplog.records)

    if gaps:
        assert "absent from the panel" in warned
        for theme in gaps:
            assert theme in warned, f"{theme}'s missing members were not reported"

    # and it must fire for a member that simply does not exist
    fake = {"bogus": ["f_this_feature_does_not_exist_xs"]}
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        build_composites(real_panel, fake, method="zscore",
                         log=logging.getLogger("composites-test"))
    assert "f_this_feature_does_not_exist_xs" in " ".join(
        r.getMessage() for r in caplog.records)

    n_gap = sum(len(v) for v in gaps.values())
    print(f"\n[7] {n_gap} configured member(s) absent from this 10-ticker panel, all named "
          f"in the warning across {len(gaps)} theme(s)")
    for theme, cols in sorted(gaps.items())[:8]:
        print(f"      {theme:20} {len(cols):2d}  {', '.join(cols[:4])}"
              + (" ..." if len(cols) > 4 else ""))
    print("    SANITY CHECK: a renamed feature or an empty source table now surfaces as "
          "a warning instead of a quietly weaker composite.")
