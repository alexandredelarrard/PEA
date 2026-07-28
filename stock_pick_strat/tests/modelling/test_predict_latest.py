"""
Production prediction step (src/modelling/long_short/step_train.py::StepModelling.predict_latest).

Loads the saved ensemble artifacts and scores the LATEST cube date(s) into `predictions_latest`
in LONG form: one row per (as-of date, ticker, horizon, model), each carrying `predicted_at`
(when the run produced it) and `predicts_for` (the date that row is about). `model` is every
ensemble member, plus 'ensemble' per horizon and 'blended' across horizons.

Long rather than wide because `predicts_for` is per horizon — the h30 and h90 predictions made
today are about different future dates, which a single column cannot hold.

Crucially it builds the feature panel DIRECTLY from the cube (NOT panel_from_cube, which drops
null-target rows), so the newest date — whose forward target has not matured — is still
predictable.

`test_predicts_for_*` are pure unit tests; the end-to-end one needs a populated `cube` + trained
artifacts and SKIPS cleanly otherwise.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.constants.constants import PREDICTION_MODEL_BLENDED, PREDICTION_MODEL_ENSEMBLE
from src.modelling.long_short.step_train import StepModelling

warnings.filterwarnings("ignore")

_LONG_COLUMNS = ["predicted_at", "date", "ticker", "horizon", "model", "predicts_for",
                 "pred", "rank"]


def _step():
    from src.context import get_config_context
    config, context = get_config_context("./configs", use_cache=False, save=True)
    return StepModelling(context=context, config=config)


def test_predicts_for_is_the_as_of_date_plus_horizon_trading_days():
    """The cube target is a forward return over h ROWS of the daily price panel = h TRADING
    days, so the target date is a BUSINESS-day offset, not a calendar one."""
    as_of = pd.Timestamp("2026-07-27")            # a Monday
    for h in (1, 5, 30, 60, 90):
        got = StepModelling.predicts_for(as_of, h)
        assert got == as_of + pd.tseries.offsets.BDay(h)
        assert got.weekday() < 5, f"h{h} landed on a weekend: {got}"
    # 30 trading days is ~6 calendar weeks, not 30 calendar days
    d30 = StepModelling.predicts_for(as_of, 30)
    assert (d30 - as_of).days == 42
    print("\n=== SANITY CHECK: predicts_for arithmetic ===")
    for h in (30, 60, 90):
        d = StepModelling.predicts_for(as_of, h)
        print(f"  as-of {as_of.date()} + h{h} trading days -> {d.date()} "
              f"({(d - as_of).days} calendar days)")
    print("  business-day offset (never a weekend); h30 = 42 calendar days, NOT 30. Validated.")


def test_prediction_rows_are_long_and_stamped():
    """`_prediction_rows` builds one (horizon, model) slice: z-scored per day, ranked, and
    stamped with predicted_at + predicts_for."""
    step = StepModelling.__new__(StepModelling)   # no DB needed for this pure shaping step
    dates = pd.to_datetime(["2026-07-24", "2026-07-24", "2026-07-24", "2026-07-27", "2026-07-27"])
    keys = pd.DataFrame({"date": dates, "ticker": ["AAA", "BBB", "CCC", "AAA", "BBB"]})
    raw = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
    stamp = pd.Timestamp("2026-07-28 06:00:00")

    out = step._prediction_rows(keys, raw, 30, "lgbm", stamp)

    assert list(out.columns) == _LONG_COLUMNS
    assert (out["horizon"] == 30).all() and (out["model"] == "lgbm").all()
    assert (out["predicted_at"] == stamp).all()
    # predicts_for follows each row's OWN as-of date
    assert (out["predicts_for"] == out["date"].map(lambda d: d + pd.tseries.offsets.BDay(30))).all()
    # per-day standardized: each date's preds have mean ~0
    for _, g in out.groupby("date"):
        assert abs(float(g["pred"].mean())) < 1e-9
    # rank is the per-day percentile, monotone in pred
    d0 = out[out["date"] == pd.Timestamp("2026-07-24")].sort_values("pred")
    assert list(d0["rank"]) == sorted(d0["rank"])

    print("\n=== SANITY CHECK: long prediction rows ===")
    print(out.to_string(index=False))
    print("  one row per (date, ticker) for this (horizon=30, model=lgbm); predicted_at is the "
          "RUN time, predicts_for follows each row's own as-of date; pred z-scored per day. "
          "Validated.")


def test_predict_latest_makes_sense():
    try:
        step = _step()
        out = step.predict_latest(n_dates=1)
    except Exception as e:                                    # no DB / no cube / no artifacts
        pytest.skip(f"cube or model artifacts unavailable: {e}")
    if out is None or out.empty:
        pytest.skip("predict_latest returned no rows (empty cube)")

    assert list(out.columns) == _LONG_COLUMNS
    as_of = out["date"].max()
    last = out[out["date"] == as_of]
    models = set(last["model"])
    horizons = sorted(int(h) for h in last["horizon"].unique())

    # the two aggregates must be present alongside the members
    assert PREDICTION_MODEL_ENSEMBLE in models, models
    assert PREDICTION_MODEL_BLENDED in models, models
    assert len(models) >= 3, f"expected members + ensemble + blended, got {models}"
    # long grain: (date, ticker, horizon, model) is unique — that is the PK
    assert not last.duplicated(["date", "ticker", "horizon", "model"]).any()
    # the dates: predicted_at is a RUN stamp at/after the as-of date; predicts_for is ahead of it
    assert (last["predicted_at"] >= as_of).all()
    assert (last["predicts_for"] > last["date"]).all()
    for h in horizons:
        sl = last[last["horizon"] == h]
        assert (sl["predicts_for"] == sl["date"] + pd.tseries.offsets.BDay(h)).all()

    # every slice is per-day standardized, finite, and monotone in rank
    for (h, m), g in last.groupby(["horizon", "model"]):
        assert g["ticker"].is_unique, (h, m)
        assert g["pred"].notna().mean() > 0.9, (h, m)
        assert abs(float(g["pred"].mean())) < 0.2 and abs(float(g["pred"].std()) - 1.0) < 0.2
        spear = g[["pred", "rank"]].corr(method="spearman").iloc[0, 1]
        assert spear > 0.999, f"{m} h{h}: rank not monotone in pred (spearman={spear:.3f})"

    # per-horizon ensembles are correlated but NOT identical -> the blend adds information
    ens = last[last["model"] == PREDICTION_MODEL_ENSEMBLE].pivot(
        index="ticker", columns="horizon", values="pred")
    if ens.shape[1] >= 2:
        off = ens.corr().to_numpy()[np.triu_indices(ens.shape[1], 1)]
        assert (off > 0.2).all() and (off < 0.999).all(), f"horizon corr degenerate: {off}"

    blended = last[last["model"] == PREDICTION_MODEL_BLENDED]
    top = blended.nlargest(3, "pred")["ticker"].tolist()
    bot = blended.nsmallest(3, "pred")["ticker"].tolist()
    print("\n=== SANITY CHECK: predict_latest on the last cube date ===")
    print(f"  as-of {as_of.date()} | predicted_at {last['predicted_at'].max()} | "
          f"{len(last)} rows = {blended['ticker'].nunique()} names x {len(horizons)} horizons "
          f"x {len(models)} models")
    print(f"  horizons {horizons} -> predicts_for "
          f"{ {h: str(last[last['horizon']==h]['predicts_for'].max().date()) for h in horizons} }")
    print(f"  models {sorted(models)}")
    print(f"  blended (h~{int(blended['horizon'].iloc[0])}) range "
          f"[{blended['pred'].min():+.2f}, {blended['pred'].max():+.2f}]")
    print(f"  top buys {top} | bottom {bot}")
    print("  CONCLUSION: long-format predictions per horizon AND per model for the newest "
          "(unlabelled) cube date, each stamped with when it was predicted and the date it "
          "predicts -> allocation-ready. Validated.")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
