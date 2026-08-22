"""
Deterministic fingerprinting PRIMITIVES: reduce a frame to a hash, and build a seeded
synthetic price panel to feed a builder under test.

A refactor must not change a single number, and comparing hashes is how this repo proves
that rather than eyeballing tests. `aggregate_fingerprint.py` is the live consumer -- it
owns the fingerprinted output set, the real fundamentals input slice
(`aggregate_fingerprint_fundamentals.parquet`) and the gated baseline.

RETIRED: this module used to also fingerprint the EXTRACTION half, rebuilding
`fundamentals_history` from cached SEC companyfacts JSON. That substrate is gone (a
multi-GB download, absent on every machine) and the companyfacts extraction path it
called no longer exists, so `compute()`, `pipeline_fingerprint_baseline.json` and
`test_refactor_regression.py` were removed with the fundamentals rebuild -- see
reports/planning/active-tasks/2026-08-21-fundamentals-rebuild-plan.md. That guard had
also always covered strictly less than `aggregate_fingerprint.py` (9 of 13 panel
builders unfingerprinted). What remains here is the shared kernel.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

SEED = 20260727
START, END = "2019-01-02", "2026-06-30"


# --------------------------------------------------------------------------- #
# fixed inputs                                                                 #
# --------------------------------------------------------------------------- #
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
