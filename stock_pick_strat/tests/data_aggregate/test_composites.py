"""Tests for the thematic composite-signal builder (composites.build_composites):
sign inversion, NaN-tolerant averaging, and the ADDITIVE guarantee (raw features
are kept -> no information is lost).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.data_aggregate.utils.composites import build_composites, _parse_member

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _panel():
    dates = pd.bdate_range("2020-01-01", periods=1)
    n = 5
    tickers = [f"T{i}" for i in range(n)]
    return pd.DataFrame({
        "date": [dates[0]] * n,
        "ticker": tickers,
        "f_val": [0.1, 0.3, 0.5, 0.7, 0.9],           # higher = better
        "f_bad": [0.9, 0.7, 0.5, 0.3, 0.1],           # perfectly anti-correlated with f_val
        "f_sparse": [1.0, np.nan, np.nan, np.nan, 5.0],  # mostly missing
    })


def test_composites_sign_inversion_nan_tolerant_and_additive():
    panel = _panel()
    groups = {
        "value": ["f_val"],
        "risk": ["-f_bad"],                            # inverted -> equals z(f_val)
        "mix": ["f_val", "-f_bad", "f_sparse"],
    }
    out = build_composites(panel, groups, method="zscore")

    # ADDITIVE: every raw feature is still there (no information lost)
    assert {"f_val", "f_bad", "f_sparse"}.issubset(out.columns)
    # composites added
    assert {"comp_value", "comp_risk", "comp_mix"}.issubset(out.columns)

    # f_bad = 1 - f_val -> z(-f_bad) == z(f_val), so inverting recovers the value sign
    assert np.allclose(out["comp_value"].to_numpy(), out["comp_risk"].to_numpy(), atol=1e-9)

    # NaN-tolerant: comp_mix is defined for EVERY row even though f_sparse is mostly NaN
    assert out["comp_mix"].notna().all()

    # input not mutated
    assert "comp_value" not in panel.columns

    print("\n=== SANITY CHECK: composite signals ===")
    print(f"  comp_value == comp_risk (inverted anti-correlated member): "
          f"{np.allclose(out['comp_value'], out['comp_risk'])}")
    print(f"  comp_mix non-null on all rows despite a mostly-NaN member; raw features kept. Validated.")


# ---------------------------------------------------------------------------
# eps_beat composite (Option 1 beat-propensity score) — validated against the
# REAL configs so a sign/membership edit can't drift silently.
# ---------------------------------------------------------------------------

# raw member values, ordered T0 (most disappoint-prone) -> T4 (most beat-prone).
# Members flagged '-' in the config are anti-aligned here on purpose, so that
# AFTER the config's sign inversion every member reads "higher = more beat-prone".
_EPS_BEAT_RAW: dict[str, list[float]] = {
    "f_eps_surprise_4q_avg_xs":    [-20.0, -10.0,  0.0,  10.0,  20.0],  # +  surprise persistence
    "f_accruals_xs":               [  0.9,   0.7,  0.5,   0.3,   0.1],  # -  low accruals = quality
    "f_earnings_quality_xs":       [  0.5,  0.75,  1.0,  1.25,   1.5],  # +  cash-backed earnings
    "f_y_rev_growth_xs":           [ -0.1,   0.0, 0.05,  0.10,   0.2],  # +  top-line momentum
    "f_margin_expansion_delta_xs": [-0.03, -0.01,  0.0,  0.01,  0.03],  # +  operating leverage
    "f_q_margin_vs_ttm_xs":        [-0.02, -0.01,  0.0,  0.01,  0.02],  # +  margin inflecting up
    "f_eps_expectation_growth_xs": [ 0.30,  0.20, 0.10,  0.05,   0.0],  # -  the bar (low = easy beat)
}


def _load_eps_beat_group() -> list[str]:
    cfg = OmegaConf.load(_CONFIG_DIR / "build_cube.yml")
    return list(cfg.build_cube.composites.groups.eps_beat)


def test_eps_beat_group_membership_and_sign_priors():
    """The config must carry exactly the agreed members with the two
    'lower = better' ones (accruals, the expectation bar) inverted.

    Membership grew from six to seven with `q_margin_vs_ttm` (latest-quarter margin
    minus the TTM margin = margin inflecting up, a direct beat tell). `accruals` stays
    despite also sitting in accounting_quality: low accruals genuinely belong to a
    beat-propensity thesis, and this membership was agreed explicitly."""
    group = _load_eps_beat_group()
    signs = {col: sign for sign, col in (_parse_member(m) for m in group)}

    assert set(signs) == set(_EPS_BEAT_RAW), "eps_beat membership drifted from the agreed set"
    # orientation priors: higher raw value -> more beat-prone
    assert signs["f_accruals_xs"] == -1.0                # high accruals -> disappoint
    assert signs["f_eps_expectation_growth_xs"] == -1.0  # high bar -> harder to beat
    for pos in ("f_eps_surprise_4q_avg_xs", "f_earnings_quality_xs", "f_y_rev_growth_xs",
                "f_margin_expansion_delta_xs", "f_q_margin_vs_ttm_xs"):
        assert signs[pos] == 1.0

    print("\n=== SANITY CHECK: eps_beat membership & sign priors ===")
    print(f"  {len(signs)} members present; inverted = {[c for c,s in signs.items() if s < 0]}")
    print("  accruals & expectation-bar inverted, the other five positive. Validated.")


def test_eps_beat_score_orientation_is_monotonic():
    """Feeding the REAL config group a gradient from disappoint-prone (T0) to
    beat-prone (T4) must yield a strictly increasing comp_eps_beat, symmetric
    around ~0 (equal-weight z-scores), and defined even with a missing member."""
    tickers = [f"T{i}" for i in range(5)]
    d = pd.Timestamp("2020-01-02")
    panel = pd.DataFrame({"date": [d] * 5, "ticker": tickers, **_EPS_BEAT_RAW})
    # knock a hole in one member for one name -> NaN-tolerant mean must still score it
    panel.loc[panel["ticker"] == "T2", "f_earnings_quality_xs"] = np.nan

    out = build_composites(panel, {"eps_beat": _load_eps_beat_group()}, method="zscore")
    score = out.set_index("ticker")["comp_eps_beat"]

    assert score.notna().all()                                   # every name scored
    diffs = score.reindex(tickers).diff().dropna()
    assert (diffs > 0).all(), f"comp_eps_beat not monotone: {score.to_dict()}"
    assert score["T4"] > 0 > score["T0"]                         # beat-prone long, disappoint short

    print("\n=== SANITY CHECK: eps_beat_score orientation ===")
    for t in tickers:
        print(f"  {t}: comp_eps_beat = {score[t]:+.3f}")
    print(f"  strictly increasing T0->T4 (most beat-prone highest); "
          f"top {score['T4']:+.2f} > 0 > bottom {score['T0']:+.2f}, even with a NaN member. Validated.")


def test_composite_model_inputs_are_never_sign_inverted():
    """A composite fed to the model must not carry an INVERTED monotone constraint — the
    tree must never be free to flip a prior the composite already encodes (comp_eps_beat is
    built so that higher = more beat-prone, so a -1 there would invert the signal).

    This used to assert `comp_eps_beat in cfg.inputs.columns` against `modellling.yml`, but
    the feature set moved to per-member blocks (`configs/models/{lgbm,linear,
    random_forest}_modelling.yml`) and `inputs:` no longer exists — so the test failed on a
    missing config key rather than on anything about eps_beat.

    It now checks the WIRING INVARIANT rather than the SELECTION decision: which features
    the model trains on is a modelling choice that legitimately changes with every
    feature-pruning experiment, and as of this config NO `comp_*` composite is an input at
    all (the members run purely on `f_*` columns). Pinning selection here would just break
    on the next experiment; pinning the sign catches a genuine mistake if one is re-added.
    """
    models = _CONFIG_DIR / "models"
    positive_priors = {"comp_eps_beat"}          # built so higher = stronger signal
    wired: dict[str, dict] = {}
    for member, fname in (("lgbm", "lgbm_modelling.yml"),
                          ("linear", "linear_modelling.yml"),
                          ("random_forest", "random_forest_modelling.yml")):
        path = models / fname
        if not path.exists():
            continue
        blk = OmegaConf.load(path)[member]
        cols = [c for c in list(blk.get("columns") or []) if c.startswith("comp_")]
        mono_cfg = blk.get("monotonic")
        mono = ({k: v for d in (mono_cfg.get("features") or []) for k, v in dict(d).items()}
                if mono_cfg is not None else {})
        for col in cols:
            if col in positive_priors:
                assert mono.get(col, 1) == 1, (
                    f"{member}: {col} carries monotone {mono.get(col)}; the composite is "
                    "built so higher = stronger, so an inverted constraint flips it")
        if cols:
            wired[member] = {c: mono.get(c) for c in cols}

    print("\n=== SANITY CHECK: composite model-input sign priors ===")
    print(f"  composites wired into a member: {wired or 'none (members use f_* only)'}")
    print("  no positively-oriented composite carries an inverted monotone constraint. "
          "Validated.")


def test_composites_disabled_and_missing_members_are_safe():
    panel = _panel()
    # no groups -> unchanged
    assert build_composites(panel, {}).equals(panel)
    # a group whose members are all absent -> no column created, no crash
    out = build_composites(panel, {"ghost": ["f_does_not_exist", "-f_also_missing"]})
    assert "comp_ghost" not in out.columns
    assert {"f_val", "f_bad"}.issubset(out.columns)

    print("\n=== SANITY CHECK: composites degrade safely ===")
    print("  empty groups -> panel unchanged; all-missing group -> skipped, no crash. Validated.")
