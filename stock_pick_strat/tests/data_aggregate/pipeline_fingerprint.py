"""
Deterministic numeric FINGERPRINT of the extraction + aggregation pipeline.

A refactor must not change a single number. This module runs the real public entry
points of both modules over fixed inputs (cached SEC companyfacts + a seeded synthetic
price panel) and reduces every output frame to a hash, so "before" and "after" can be
compared exactly rather than by eyeballing tests.

Only PUBLIC entry points are called, never private helpers: the whole point of the
refactor is that helpers move, split and merge, so the fingerprint must be blind to how
the work is organised internally and sensitive only to what comes out.

Used by `test_refactor_regression.py`; regenerate the stored baseline with
    python -m tests.data_aggregate.pipeline_fingerprint  > /dev/null
which writes `pipeline_fingerprint_baseline.json` next to this file.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "sec_bulk_cache"
BASELINE = Path(__file__).with_name("pipeline_fingerprint_baseline.json")

SEED = 20260727
N_TICKERS = 12
START, END = "2019-01-02", "2026-06-30"


# --------------------------------------------------------------------------- #
# fixed inputs                                                                 #
# --------------------------------------------------------------------------- #
class MissingCompanyFactsCache(RuntimeError):
    """`data/sec_bulk_cache/companyfacts_CIK*.json` holds fewer than `N_TICKERS` filers, so
    this fingerprint's fixed inputs cannot be reconstructed. The cache is a multi-GB SEC
    download and is absent on most machines; `random.sample` used to raise a bare
    `ValueError: Sample larger than population` from deep inside the stdlib, which read as a
    broken test rather than a missing input. `test_refactor_regression.py` skips on this.
    The aggregation half is covered without the cache by `aggregate_fingerprint.py`."""


def sample_tickers() -> list[tuple[str, str, str, str]]:
    """(ticker, cik, sector, industry_group) for a seeded draw over the cached filers."""
    from src.data_store.store import DataStore
    from src.utils.db import get_engine
    uni = DataStore(get_engine()).load("sp500_tickers").dropna(subset=["cik", "ticker"])
    meta = {r.ticker: (r.cik, r.sector, r.industry_group) for r in uni.itertuples()}
    avail = sorted(t for t, (cik, _, _) in meta.items()
                   if (CACHE / f"companyfacts_CIK{cik}.json").exists())
    if len(avail) < N_TICKERS:
        raise MissingCompanyFactsCache(
            f"{len(avail)} cached companyfacts filers under {CACHE}, need {N_TICKERS}")
    picked = sorted(random.Random(SEED).sample(avail, N_TICKERS))
    return [(t, *meta[t]) for t in picked]


def synthetic_prices(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Seeded geometric random walks. A constant price panel would leave every
    momentum / volatility / valuation-vs-price feature degenerate, so the fingerprint
    would not notice a refactor breaking them."""
    idx = pd.bdate_range(START, END)
    rng = np.random.default_rng(SEED)
    steps = rng.normal(0.0004, 0.018, size=(len(idx), len(tickers)))
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=tickers)
    return {
        "close": close,
        "open": close.shift(1).bfill() * (1 + rng.normal(0, 0.002, close.shape)),
        "high": close * (1 + np.abs(rng.normal(0, 0.006, close.shape))),
        "low": close * (1 - np.abs(rng.normal(0, 0.006, close.shape))),
        "volume": pd.DataFrame(rng.lognormal(15, 0.4, close.shape), index=idx, columns=tickers),
    }


# --------------------------------------------------------------------------- #
# hashing                                                                      #
# --------------------------------------------------------------------------- #
def frame_digest(df: pd.DataFrame | None) -> dict:
    """Shape + column list + a per-column hash of the rounded values. Rounding to 10
    decimals absorbs nothing a refactor should produce (a pure reorganisation is
    bit-identical) while keeping the digest stable across platforms."""
    if df is None or len(df) == 0:
        return {"rows": 0, "cols": 0, "columns": [], "hash": "empty"}
    d = df.sort_index(axis=1)
    if {"date", "ticker"}.issubset(d.columns):
        d = d.sort_values(["date", "ticker"]).reset_index(drop=True)
    per_col: dict[str, str] = {}
    for c in d.columns:
        s = d[c]
        if pd.api.types.is_numeric_dtype(s):
            v = np.round(pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64"), 10)
            payload = np.nan_to_num(v, nan=-1.2345e300).tobytes()
        else:
            payload = "|".join(map(str, s.tolist())).encode()
        per_col[str(c)] = hashlib.md5(payload).hexdigest()[:16]
    return {
        "rows": int(len(d)), "cols": int(d.shape[1]),
        "columns": [str(c) for c in d.columns],
        "hash": hashlib.md5(json.dumps(per_col, sort_keys=True).encode()).hexdigest(),
        "per_column": per_col,
    }


# --------------------------------------------------------------------------- #
# the pipeline                                                                 #
# --------------------------------------------------------------------------- #
def compute() -> dict:
    import json as _json

    from src.data_aggregate.utils.target.betas import estimate_all_betas
    from src.data_aggregate.utils.assemble.composites import build_composites
    from src.data_aggregate.utils.target.factors import build_characteristics, momentum_characteristic
    from src.data_aggregate.utils.target.targets import build_targets_multi
    from src.data_aggregate.utils.momentum.features import build_feature_panel, compute_raw_features
    from src.data_aggregate.utils.fundamentals.fundamental_features import build_fundamental_feature_panel
    from src.data_aggregate.utils.fundamentals.intrinsic import intrinsic_value_daily
    from src.data_aggregate.utils.fundamentals.sector_features import (
        build_sector_feature_panel, compute_sector_kpis,
    )
    from src.data_extract.utils.fundamentals.fetch_fundamentals import build_ticker_history

    names = sample_tickers()
    tickers = [t for t, *_ in names]
    px = synthetic_prices(tickers)
    close, idx = px["close"], px["close"].index
    returns = close.pct_change()
    sector_ret = returns.rolling(5).mean().bfill()      # deterministic stand-in
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}

    out: dict[str, dict] = {}

    # ---- data_extract: SEC companyfacts -> point-in-time fundamentals ---- #
    frames = []
    for ticker, cik, sector, group in names:
        facts = _json.loads((CACHE / f"companyfacts_CIK{cik}.json").read_text(encoding="utf-8"))
        h = build_ticker_history(ticker, facts, sector, group)
        out[f"extract.build_ticker_history.{ticker}"] = frame_digest(h)
        frames.append(h)
    fund = pd.concat(frames, ignore_index=True)
    out["extract.fundamentals_history"] = frame_digest(fund)

    # ---- data_aggregate: every panel builder ---- #
    raw = compute_raw_features(close, px["open"], sector_ret, high=px["high"],
                               low=px["low"], volume=px["volume"],
                               seasonal_horizons=[21, 60])
    # {name: date x ticker} -> one wide frame so a change in ANY family shows up
    out["aggregate.compute_raw_features"] = frame_digest(
        pd.concat({k: v for k, v in sorted(raw.items())
                   if isinstance(v, pd.DataFrame)}, axis=1))
    out["aggregate.build_feature_panel"] = frame_digest(
        build_feature_panel(close, px["open"], sector_ret, high=px["high"],
                            low=px["low"], volume=px["volume"]))
    chars = build_characteristics(close, returns, fund)
    out["aggregate.build_characteristics"] = frame_digest(
        pd.concat({k: v for k, v in chars.items() if isinstance(v, pd.DataFrame)}, axis=1)
        if any(isinstance(v, pd.DataFrame) for v in chars.values()) else None)
    out["aggregate.intrinsic_value_daily"] = frame_digest(
        intrinsic_value_daily(fund, close, idx).get("yield"))
    out["aggregate.compute_sector_kpis"] = frame_digest(compute_sector_kpis(fund))
    fp = build_fundamental_feature_panel(fund, peers, idx, stock_close=close)
    out["aggregate.fundamental_panel"] = frame_digest(fp)
    sp = build_sector_feature_panel(fund, peers, idx)
    out["aggregate.sector_panel"] = frame_digest(sp)

    panel = fp.merge(sp, on=["date", "ticker"], how="outer")
    cfg = yaml.safe_load((ROOT / "configs" / "build_cube.yml").read_text(encoding="utf-8"))
    groups = next(v["composites"]["groups"] for v in cfg.values()
                  if isinstance(v, dict) and "composites" in v)
    comp = build_composites(panel, groups, method="zscore")
    out["aggregate.composites"] = frame_digest(comp[["date", "ticker"]
                                               + sorted(c for c in comp.columns
                                                        if c.startswith("comp_"))])

    # Market (available immediately) PLUS the momentum style factor, whose
    # close.shift(21)/close.shift(252) definition is NaN for its first 252 trading days.
    # That combination is deliberate: a single no-warm-up factor never exercised the
    # regressor-join path where one slow factor used to blank every other beta, so the
    # fingerprint was blind to it.
    factor_panel = pd.DataFrame({
        "market": returns.mean(axis=1),
        "momentum": momentum_characteristic(close).mean(axis=1),
    })
    betas = estimate_all_betas(returns, factor_panel, sector_ret)
    out["aggregate.betas"] = frame_digest(
        pd.concat({k: v for k, v in betas.items() if isinstance(v, pd.DataFrame)}, axis=1)
        if isinstance(betas, dict) else betas)

    # The LABEL itself, per horizon -- it was not fingerprinted at all, which left the
    # one output the model actually trains on unprotected.
    # `min_names` is lowered from the production 20 because this harness runs a
    # 12-name cross-section: at the default, `_apply_label` blanks every day for having
    # too few names and the label fingerprint would be a uniformly-empty frame that
    # silently proves nothing. The value only gates which DAYS survive, not how the
    # residual is computed, so the computation under test is unaffected.
    targets = build_targets_multi(
        close, returns, peers, betas, factor_panel, macro_cols=[],
        horizons=(30, 60, 90), labels=("rank", "zscore"), min_names=5,
        sector_groups={"sector": {t: "S" for t in tickers}})
    for horizon, by_label in targets.items():
        for label, frame in by_label.items():
            out[f"aggregate.target_{label}_h{horizon}"] = frame_digest(frame)

    out["_meta"] = {"tickers": tickers, "seed": SEED, "start": START, "end": END}
    return out


def main() -> None:
    fp = compute()
    BASELINE.write_text(json.dumps(fp, indent=1, sort_keys=True), encoding="utf-8")
    n = sum(1 for k in fp if not k.startswith("_"))
    print(f"wrote {BASELINE.name}: {n} fingerprinted outputs")
    for k in sorted(fp):
        if k.startswith("_"):
            continue
        d = fp[k]
        print(f"  {k:46} rows={d['rows']:6d} cols={d['cols']:4d} {d['hash'][:12]}")


if __name__ == "__main__":
    main()
