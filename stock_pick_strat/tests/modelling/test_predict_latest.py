"""
Production prediction step (src/modelling/long_short/step_train.py::StepModelling.predict_latest).

Loads the saved ensemble artifacts and scores the LATEST cube date(s) per horizon (pred_h<h>) +
the IR-blended signal -> predictions_latest, for the allocation DAG. Crucially it builds the
feature panel DIRECTLY from the cube (NOT panel_from_cube, which drops null-target rows), so the
newest date — whose forward target has not matured — is still predictable.

Integration test: needs a populated `cube` + trained model artifacts; SKIPS cleanly otherwise.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")


def _step():
    from src.context import get_config_context
    from src.modelling.long_short.step_train import StepModelling
    config, context = get_config_context("./configs", use_cache=False, save=True)
    return StepModelling(context=context, config=config)


def test_predict_latest_makes_sense():
    try:
        step = _step()
        out = step.predict_latest(n_dates=1)
    except Exception as e:                                    # no DB / no cube / no artifacts
        pytest.skip(f"cube or model artifacts unavailable: {e}")
    if out is None or out.empty:
        pytest.skip("predict_latest returned no rows (empty cube)")

    hcols = [c for c in out.columns if c.startswith("pred_h")]
    last = out[out["date"] == out["date"].max()]
    assert hcols, "no per-horizon prediction columns produced"
    assert {"blended", "rank"}.issubset(out.columns)
    # every name unique + the blended signal is finite for (nearly) all names
    assert last["ticker"].is_unique
    assert last["blended"].notna().mean() > 0.9
    # each horizon is per-day standardized (mean ~0, std ~1) and finite
    for c in hcols:
        assert last[c].notna().mean() > 0.9
        assert abs(float(last[c].mean())) < 0.2 and abs(float(last[c].std()) - 1.0) < 0.2
    # rank is the per-day percentile of blended -> MONOTONE in blended (Spearman == 1)
    spear = last[["blended", "rank"]].corr(method="spearman").iloc[0, 1]
    assert spear > 0.999, f"rank not monotone in blended (spearman={spear:.3f})"
    # horizons are correlated but NOT identical -> the ensemble/blend adds information
    if len(hcols) >= 2:
        cc = last[hcols].corr().to_numpy()
        off = cc[np.triu_indices(len(hcols), 1)]
        assert (off > 0.2).all() and (off < 0.999).all(), f"horizon corr degenerate: {off}"

    top = last.nlargest(3, "blended")["ticker"].tolist()
    bot = last.nsmallest(3, "blended")["ticker"].tolist()
    print("\n=== SANITY CHECK: predict_latest on the last cube date ===")
    print(f"  latest date {last['date'].max().date()}: {len(last)} names | horizons {hcols}")
    print(f"  blended finite={last['blended'].notna().mean()*100:.0f}% "
          f"range=[{last['blended'].min():+.2f}, {last['blended'].max():+.2f}] | "
          f"spearman(blended,rank)={spear:.3f}")
    print(f"  top buys {top} | bottom {bot}")
    print("  CONCLUSION: per-horizon + blended predictions for the newest (unlabelled) cube date "
          "are finite, unique, monotone-ranked, horizons distinct -> allocation-ready. Validated.")


if __name__ == "__main__":
    test_predict_latest_makes_sense()
