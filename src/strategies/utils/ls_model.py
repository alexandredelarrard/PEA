"""
ls_model.py  (src/strategies/utils/ls_model.py)
-----------------------------------------------
Shared L/S MODEL signal builder: load the trained ensemble members (pickled transformers, see
src/modelling/utils/artifacts.py), project the cube (OOS window date >= train_end), score each
horizon's ensemble, blend across horizons →
per-name combined z-signal, and load the equity return / price panels. Used by BOTH the
market-neutral L/S sleeve (`step_ls`) and the long-only sleeve (`step_eq_long_only`) so the
model/signal is defined once and neither strategy depends on the other's step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.constants.constants_price import MACRO_MARKET_SERIES
from src.context import Context
from src.data_aggregate.utils.assemble.cube import panel_from_cube, target_column
from src.data_aggregate.utils.common import data_utils as du
from src.data_store.schema import Tables
from src.modelling.utils.artifacts import load_ensemble, models_dir, read_metadata
from src.modelling.utils.ensemble import blend_horizons, ensemble_predict, optimal_forecast_weights
from src.utils.macro import load_macro_series
from src.utils.universe import load_universe_tickers


@dataclass
class SignalBundle:
    signal: pd.DataFrame  # date x ticker combined cross-sectional z-signal
    stock_ret: pd.DataFrame  # date x equity daily returns (`prices` is equity-only now)
    spy_ret: pd.Series  # market benchmark daily return
    # date x ticker SPLIT-ADJUSTED prices (all tickers; for share blotters). Named for the
    # basis: this is the price a share is transacted at, NOT the dividend-reinvested path --
    # `stock_ret` is where the total return lives.
    close_split: pd.DataFrame
    backtest_start: pd.Timestamp
    end: pd.Timestamp
    train_ic: dict
    horizons: list


def _load_models(context: Context, config: DictConfig) -> tuple[dict, dict[int, dict[str, Any]], str, list[int]]:
    """metadata.json + every saved member of the horizons that run trained, among the configured
    cube horizons (missing members skipped). A member file of a horizon the metadata does not list
    belongs to an older run and is never loaded. Raises when nothing is trained or the artifacts
    predate the current pickle layout."""
    directory = models_dir(context, config)
    meta = read_metadata(directory)
    target_type = meta.get("target_type", config.strategy_ls.get("target_type", "rank"))
    model_types = list(meta.get("model_types") or [])
    trained = {int(h) for h in meta.get("horizons", [])}
    horizons = [int(h) for h in config.build_cube.targets.horizons if int(h) in trained]
    return meta, load_ensemble(directory, horizons, model_types), target_type, horizons


def _project_cube(context: Context, meta: dict, models: dict, target_type: str, start: pd.Timestamp, end) -> pd.DataFrame:
    """Project the cube to the model's features plus ONE label column per trained horizon.

    The cube is one row per (date, ticker) with the targets wide, so the horizons select
    COLUMNS, not rows -- there is no row filter to apply and the caller resolves each
    horizon's label with `target_column` as it loops."""
    store = context.store
    cube_cols = set(store.columns(Tables.cube))
    tcols = [target_column(target_type, int(h)) for h in models]
    missing = [c for c in tcols if c not in cube_cols]
    if missing:
        raise KeyError(
            f"Target columns {missing} not in cube; rebuild it with '{target_type}' "
            f"in build_cube.targets.labels and those horizons in "
            f"build_cube.targets.horizons."
        )

    want = list(dict.fromkeys(meta["feature_cols"] + meta.get("categorical_cols", [])))
    load_cols = list(dict.fromkeys(["date", "ticker"] + tcols + [c for c in want if c in cube_cols]))
    # bound parameters, not f-string interpolation: `date >= '{start.date()}'` was the one
    # query in the repo pasting a value straight into SQL
    panel = store.load(Tables.cube, columns=load_cols, since=start, until=end)
    if panel is None:
        raise RuntimeError(f"'{Tables.cube}' returned no modeling panel")
    return panel


def _returns(context: Context, config: DictConfig, cube_cfg: DictConfig, model_cfg: DictConfig, start: pd.Timestamp):
    buffer = int(2.2 * (int(model_cfg.get("beta_window", 63)) + int(model_cfg.get("vol_window", 63))) + 30)
    cutoff = start - pd.Timedelta(days=buffer)
    long = context.store.load(Tables.prices, since=cutoff, where={"ticker": load_universe_tickers(context)})
    if long is None:
        raise RuntimeError(f"'{Tables.prices}' returned no price frame")
    pivot = du.prices_long_to_multiindex(long)
    # TWO bases, two jobs. `rets` drives the P&L, so it is the buy-and-hold path; the frame
    # returned as `close` is used as `book_prices` -- the price a share is actually
    # transacted at -- so it is the split-adjusted quote. Returning `close_total` for both
    # would price the book on a series no share ever traded at.
    close = du.extract_field(pivot, "CloseSplit")
    # `prices` is read for the universe only (it also holds secondary share classes), and holds
    # no market/index/FX column -- the benchmark leg comes from `prices_macro` instead.
    rets = du.daily_returns(du.extract_field(pivot, "CloseTotal"))
    mkt_close = load_macro_series(context.store, MACRO_MARKET_SERIES, since=cutoff)
    if mkt_close is None:
        raise RuntimeError(
            f"'{Tables.prices_macro}' has no '{MACRO_MARKET_SERIES}' rows -> no benchmark for the L/S sleeve. Run `data_extract macro`."
        )
    mkt_ret = mkt_close.pct_change(fill_method=None).reindex(rets.index)
    return close, rets, mkt_ret


def build_signal(context: Context, config: DictConfig, end=None) -> SignalBundle:
    """Load the ensemble, project the OOS cube, score + blend horizons -> combined z-signal, and
    load the equity returns/prices. `config.strategy_ls` holds the model windows/blend params."""
    cube_cfg, model_cfg = config.build_cube, config.strategy_ls
    meta, models, target_type, _ = _load_models(context, config)
    start = pd.Timestamp(meta["train_end"])
    train_ic = {int(k): float(v) for k, v in meta.get("train_ic_ir", {}).items()}
    cube = _project_cube(context, meta, models, target_type, start, end)
    close, stock_ret, spy_ret = _returns(context, config, cube_cfg, model_cfg, start)
    end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp(cube["date"].max())

    blended = None
    for h, members in models.items():
        panel = panel_from_cube(
            cube,
            horizon=h,
            label_name=meta["label_column"],
            feature_cols=meta["feature_cols"] + meta.get("categorical_cols", []),
            target_type=target_type,
        )
        panel = panel[(panel["date"] >= start) & (panel["date"] <= end_ts)]
        if panel.empty:
            continue
        scores, _ = ensemble_predict(members, panel)
        df = panel[["date", "ticker"]].copy()
        df["z"] = pd.Series(scores.to_numpy(), index=panel.index)
        df["z"] = df.groupby("date")["z"].transform(lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else np.nan))
        blended = (
            df.rename(columns={"z": f"z_{h}"})
            if blended is None
            else blended.merge(df.rename(columns={"z": f"z_{h}"}), on=["date", "ticker"], how="outer")
        )

    zc = [f"z_{h}" for h in models if blended is not None and f"z_{h}" in blended.columns]
    if not zc:
        raise RuntimeError("build_signal: no horizon produced a signal in the OOS window.")
    assert blended is not None
    hs = [int(c.split("_")[1]) for c in zc]
    ir = {h: train_ic.get(h, np.nan) for h in hs}
    if str(model_cfg.get("blend", "ir")) == "equal":
        bw = {h: 1.0 / len(hs) for h in hs}
    else:
        bw = optimal_forecast_weights({h: blended[f"z_{h}"].to_numpy() for h in hs}, ir, shrink=float(model_cfg.get("blend_shrink", 0.5)))
    blended["combined"] = blend_horizons(blended[zc].to_numpy(), np.array([bw[h] for h in hs]))
    signal = blended.pivot(index="date", columns="ticker", values="combined")
    signal.index = pd.to_datetime(signal.index)
    return SignalBundle(
        signal=signal,
        stock_ret=stock_ret,
        spy_ret=spy_ret,
        close_split=close,
        backtest_start=start,
        end=end_ts,
        train_ic=train_ic,
        horizons=list(models),
    )
