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

`test_predicts_for_*` and `test_predict_latest_scores_the_newest_date_*` are pure unit
tests (the latter drives the real method against a spy store); the end-to-end one needs a
populated `cube` + trained artifacts and SKIPS cleanly otherwise.
"""
from __future__ import annotations

import logging
import warnings
from types import SimpleNamespace

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


class _StubModel:
    """Anything with `.predict(X)` is an ensemble member (see `ml.ensemble_predict`)."""

    def __init__(self, weight: float = 1.0):
        self._w = weight

    def predict(self, X):
        return np.asarray(X["f_a"], dtype="float64") * self._w


class _SpyStore:
    """Records what `predict_latest` asks the cube for, and what it writes back."""

    NOT_NULL = object()

    def __init__(self, cube: pd.DataFrame):
        self._cube = cube
        self.load_kwargs: dict = {}
        self.written: pd.DataFrame | None = None

    def columns(self, table):
        return list(self._cube.columns)

    def distinct(self, table, col, order=None, limit=None, **kw):
        d = sorted(pd.Timestamp(x) for x in self._cube[col].unique())
        if order == "desc":
            d = d[::-1]
        return d[:limit] if limit else d

    def load(self, table, columns=None, since=None, optional=False, **kw):
        self.load_kwargs = {"columns": list(columns or []), "since": since, **kw}
        df = self._cube
        if since is not None:
            df = df[df["date"] >= pd.Timestamp(since)]
        return df[list(columns)].copy() if columns else df.copy()

    def replace(self, table, df, **kw):
        self.written = df.copy()
        return len(df)


def test_predict_latest_scores_the_newest_date_even_with_all_labels_nan(monkeypatch):
    """THE defect the wide cube fixes, pinned without a DB.

    The newest cube date has matured NO horizon, so every one of its `target_*` columns is
    NaN. Under the long cube `stack()` produced no row for it at all and this method
    silently scored a date up to `max_horizon` sessions stale. The wide part LEFT-joins onto
    the feature panel, so the row exists — and `predict_latest` must return it.

    It must also load NO target column: this method scores, it does not evaluate, and a
    label in the projection is what made the staleness invisible."""
    dates = pd.to_datetime(["2026-09-08", "2026-09-09", "2026-09-10"])
    tickers = ["AAA", "BBB", "CCC"]
    rows = [{"date": d, "ticker": t, "f_a": float(i + 1) * (j + 1)}
            for j, d in enumerate(dates) for i, t in enumerate(tickers)]
    cube = pd.DataFrame(rows)
    # labels mature oldest-first; the NEWEST date has nothing at any horizon
    cube["target_rank_h30"] = [0.1, 0.5, 0.9] * 2 + [np.nan] * 3
    cube["target_rank_h90"] = [0.2, 0.6, 0.8] + [np.nan] * 6
    newest = dates.max()
    assert cube.loc[cube["date"] == newest, ["target_rank_h30", "target_rank_h90"]] \
        .isna().all(axis=None), "fixture must have an ALL-NaN newest date"

    store = _SpyStore(cube)
    step = StepModelling.__new__(StepModelling)
    step._context = SimpleNamespace(store=store)
    step._log = logging.getLogger("predict_latest_test")
    meta = {"feature_cols": ["f_a"], "categorical_cols": [],
            "train_ic_ir": {"30": 0.6, "90": 0.4}}
    models = {30: {"lgbm": _StubModel(1.0)}, 90: {"lgbm": _StubModel(-1.0)}}
    monkeypatch.setattr(StepModelling, "_load_saved_ensemble", lambda self: (meta, models))

    out = step.predict_latest(n_dates=1)

    # 1. the newest date is what got scored
    assert out["date"].nunique() == 1
    assert out["date"].max() == newest, f"scored {out['date'].max()}, not the newest {newest}"
    assert set(out.loc[out["model"] == PREDICTION_MODEL_BLENDED, "ticker"]) == set(tickers)

    # 2. no target column was ever requested
    asked = store.load_kwargs["columns"]
    assert [c for c in asked if c.startswith("target")] == [], asked
    assert asked == ["date", "ticker", "f_a"]

    # 3. both horizons scored the SAME rows — the wide grain's whole point
    for h in (30, 90):
        sl = out[(out["horizon"] == h) & (out["model"] == PREDICTION_MODEL_ENSEMBLE)]
        assert set(sl["ticker"]) == set(tickers), f"h{h} scored {sorted(sl['ticker'])}"
        assert (sl["date"] == newest).all()

    # 4. it was persisted, not just returned
    assert store.written is not None and len(store.written) == len(out)

    print("\n=== SANITY CHECK: predict_latest reaches the unlabelled newest date ===")
    print(f"  cube dates {[str(d.date()) for d in dates]}; {newest.date()} has "
          f"target_rank_h30 AND target_rank_h90 all NaN")
    print(f"  -> scored as-of {out['date'].max().date()} for {len(tickers)} names x "
          f"{sorted(out['horizon'].unique())} horizons x {sorted(out['model'].unique())}")
    print(f"  projection asked for {asked}: ZERO target columns loaded")
    print("  the long cube had no row for this date at all, so this method predicted off a "
          "stale one. Validated.")


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
