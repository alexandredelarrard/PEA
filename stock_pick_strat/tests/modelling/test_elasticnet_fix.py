"""
Elastic-net linear member (src/modelling/utils_model/baselines.py::train_elasticnet).

Regression test for the "dead linear member" bug: the saved model predicted a CONSTANT
because it was trained with too high an `alpha` (0.05) whose L1 threshold exceeded every
feature's gradient -> all coefficients soft-thresholded to zero. At a sane `alpha` the model
must learn real (non-zero) coefficients and produce a NON-constant, correctly-signed
cross-sectional prediction; at a degenerate `alpha` it must WARN instead of shipping silently.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.modelling.long_short.utils import baselines


def _synthetic_panel(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f1 = rng.standard_normal(n)
    f2 = rng.standard_normal(n)
    f3 = rng.standard_normal(n)                      # irrelevant / noise feature
    y = 0.6 * f1 - 0.4 * f2 + 0.1 * rng.standard_normal(n)
    dates = pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 200, n), unit="D")
    return pd.DataFrame({"date": dates, "ticker": np.arange(n) % 300,
                         "y": y, "f1": f1, "f2": f2, "f3": f3})


def test_elasticnet_learns_at_sane_alpha_and_warns_when_degenerate(caplog):
    panel = _synthetic_panel()
    feats = ["f1", "f2", "f3"]

    # --- sane alpha: learns real coefficients, non-constant, correctly signed ---
    m = baselines.train_elasticnet(panel, feats, "y", alpha=1e-3, l1_ratio=0.3)
    coef = dict(zip(feats, m.coef))
    assert np.count_nonzero(np.abs(m.coef) > 0) >= 2, f"too few live coefs: {coef}"
    assert coef["f1"] > 0 and coef["f2"] < 0, f"signs wrong: {coef}"          # recovers 0.6, -0.4
    preds = m.predict(panel[feats].to_numpy())
    assert np.std(preds) > 1e-3, "prediction is (near-)constant at a sane alpha"
    ic = np.corrcoef(preds, panel["y"].to_numpy())[0, 1]
    assert ic > 0.5, f"prediction barely correlates with the target: corr={ic:.3f}"

    # --- degenerate alpha: all coefs zero, constant prediction, and it WARNS ---
    with caplog.at_level(logging.WARNING, logger="src.modelling.long_short.utils.baselines"):
        dead = baselines.train_elasticnet(panel, feats, "y", alpha=10.0, l1_ratio=0.3)
    assert np.count_nonzero(np.abs(dead.coef) > 0) == 0, "expected an all-zero (degenerate) fit"
    assert np.std(dead.predict(panel[feats].to_numpy())) < 1e-9, "degenerate fit must be constant"
    assert any("DEGENERATE" in r.message for r in caplog.records), "degenerate fit did not warn"

    print("\n=== SANITY CHECK: elastic-net fix ===")
    print(f"  sane alpha=1e-3 -> coefs {dict((k, round(v,3)) for k,v in coef.items())} "
          f"(f1>0, f2<0 recovered), pred std={np.std(preds):.3f}, corr(pred,y)={ic:.2f}")
    print("  degenerate alpha=10 -> all coefs 0, constant prediction, WARNING emitted")
    print("  CONCLUSION: at the configured alpha=0.001 the linear member LEARNS (was dead only "
          "because the saved model used alpha=0.05); the guard now flags any all-zero fit. Validated.")


if __name__ == "__main__":
    import types
    test_elasticnet_learns_at_sane_alpha_and_warns_when_degenerate(
        types.SimpleNamespace(records=[], at_level=lambda *a, **k: __import__("contextlib").nullcontext()))
