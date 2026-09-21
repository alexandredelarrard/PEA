"""
redundancy.py  (src/validate/checks/redundancy.py)
--------------------------------------------------------------------------------------------
Every pair of legs, pairwise-complete Pearson, in ONE streaming pass -- plus the exact-equality
count, which is the stronger claim and comes out of the same pass.

WHY ALL PAIRS AND NOT A CURATED LIST. The research screen carried `PRIOR_G3`, thirteen pairs
chosen by family prefix. Measured against the full cross, that list is strictly less complete:
the all-pairs screen found pairs it does not contain, including
`f_ic_super_exit_after_top10 ~ f_ic_super_full_exits` at r=0.9965 and
`f_ic_sig_super_ret_since ~ f_ic_sig_super_resid_ret_since` at r=0.9917. A curated list can only
find the redundancy somebody already suspected, which is the redundancy least likely to be
there. 249 legs is 30,876 pairs and the accumulator below does them all for the price of one
read.

HOW ONE PASS DOES IT. Pairwise-complete means each pair's statistics come only from rows where
BOTH legs are present, and the naive way to get that is a frame per pair. Instead, per chunk,
with `M` the 0/1 presence mask and `X0` the values with NaN replaced by 0:

    N   += M.T @ M           # rows where both present, per pair
    Sx  += X0.T @ M          # sum of x_i over those rows
    Sxx += (X0*X0).T @ M     # sum of x_i^2 over those rows
    Sxy += X0.T @ X0         # sum of x_i*x_j over those rows

and then, exactly:

    r_ij = (N*Sxy - Sx*Sx.T) / sqrt((N*Sxx - Sx^2) * (N*Sxx.T - Sx.T^2))

This is not an approximation and not a sample: it is the same number `pandas.corr` would give
on the whole table, computed in four matrix products per chunk. The legs are mean-centred
first (means come from `profile`'s output when the run has one, otherwise from a first pass)
because `N*Sxy - Sx*Sx.T` is a difference of large numbers, and on a leg with a big mean and a
small variance that subtraction loses the variance entirely.

⚠ READS float64 FROM THE DB, NEVER THE float32 CACHE. `r` would survive float32 easily -- it is
the EXACT-EQUALITY count that would not. "These two columns are the same column under two
names" is the strongest finding this package can make, and a float32 round-trip manufactures
equalities that are not in the database. The count is only worth having if it is exact.

⚠ AND IT ABSTAINS RATHER THAN PRINT AN IMPOSSIBLE NUMBER. If any |r| exceeds 1 by more than
1e-9 the accumulation has lost precision somewhere, and every number in the result is suspect.
That is reported as an abstain with the offending pair named, never as a finding.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_store.schema import Table, resolve
from src.validate.frame import feature_columns
from src.validate.io import CHUNK_ROWS, OUT_DIR
from src.validate.result import CheckResult, Finding, full_table_only
from src.validate.spec import load_spec

log = logging.getLogger(__name__)

CHECK = "redundancy"

#: How far past 1.0 a correlation may land before the whole result is thrown out. Floating
#: point costs a few ULPs; anything more is lost precision, not rounding.
_R_TOLERANCE = 1e-9

#: Pairs named as findings, worst first. The count is in `metrics`.
_MAX_FINDINGS = 40

#: Pairs below this many shared rows are not reported: an r of 1.00 on eleven overlapping
#: observations is arithmetic, not evidence of a duplicated leg.
_MIN_PAIRWISE_N = 100


def _means(context: Context, table: Any, columns: list[str], out: str | Path | None,
           chunksize: int) -> tuple[np.ndarray, str]:
    """Centring constants for the accumulation -- see the module docstring.

    `profile` already computed these over the same table, so a run that profiled first pays
    nothing here. The fallback is a full streaming pass, which is correct but doubles the I/O,
    so the source is recorded in `scope`."""
    if out is not None:
        path = Path(out) / OUT_DIR / "profile.json"
        if path.exists():
            means = (json.loads(path.read_text(encoding="utf-8"))
                     .get("metrics", {}).get("mean") or {})
            if all(means.get(c) is not None for c in columns):
                return np.array([float(means[c]) for c in columns], dtype="float64"), str(path)

    total = np.zeros(len(columns), dtype="float64")
    count = np.zeros(len(columns), dtype="float64")
    for chunk in context.store.iter_load(table, columns=columns, chunksize=chunksize):
        values = chunk[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
        present = np.isfinite(values)
        total += np.where(present, values, 0.0).sum(axis=0)
        count += present.sum(axis=0)
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0), "streamed pass"


def _accumulate(context: Context, table: Any, columns: list[str], centre: np.ndarray,
                chunksize: int) -> dict[str, np.ndarray]:
    """The four co-moment matrices plus the exact-equality counts, in one pass."""
    width = len(columns)
    acc = {name: np.zeros((width, width), dtype="float64")
           for name in ("n", "sx", "sxx", "sxy")}
    acc["eq"] = np.zeros((width, width), dtype="float64")
    rows = 0
    for chunk in context.store.iter_load(table, columns=columns, chunksize=chunksize):
        raw = chunk[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
        present = np.isfinite(raw)
        mask = present.astype("float64")
        centred = np.where(present, raw - centre, 0.0)
        acc["n"] += mask.T @ mask
        acc["sx"] += centred.T @ mask
        acc["sxx"] += (centred * centred).T @ mask
        acc["sxy"] += centred.T @ centred
        # Exact equality is asked on the RAW values, not the centred ones: two legs equal
        # after subtracting two different means are not the same column. NaN == NaN is False
        # in numpy, so rows where either side is absent never count as equal, which is
        # exactly the pairwise-complete rule the rest of the pass uses.
        for index in range(width):
            acc["eq"][index] += (raw == raw[:, index][:, None]).sum(axis=0)
        rows += len(chunk)
        log.info("redundancy %s: %s rows accumulated", table, f"{rows:,}")
    acc["rows"] = np.array([rows], dtype="float64")
    return acc


def _correlations(acc: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """`(r, n)` from the accumulators. Undefined pairs come back as NaN, never as 0."""
    n, sx, sxx, sxy = acc["n"], acc["sx"], acc["sxx"], acc["sxy"]
    covariance = n * sxy - sx * sx.T
    variance_i = n * sxx - sx * sx
    variance_j = variance_i.T
    denominator = np.sqrt(variance_i * variance_j)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denominator > 0, covariance / denominator, np.nan)
    return r, n


def check_redundancy(context: Context, table: Table | str, *, config: DictConfig,
                     cache: Any = None, tickers: list[str] | None = None,
                     threshold: float | None = None, out: str | Path | None = None,
                     chunksize: int = CHUNK_ROWS, **kwargs: Any) -> CheckResult:
    """Every pair's pairwise-complete Pearson r and exact-equality count, in one pass."""
    spec_t = resolve(table)
    if (declined := full_table_only(CHECK, spec_t.name, tickers)) is not None:
        return declined
    spec = load_spec(config, spec_t, redundancy_r=threshold)
    columns = feature_columns(context, spec_t)
    if len(columns) < 2:
        return CheckResult.abstained(
            CHECK, spec_t.name,
            f"{len(columns)} numeric leg(s) -- a pair needs two")

    centre, centre_source = _means(context, spec_t, columns, out or cache, chunksize)
    acc = _accumulate(context, spec_t, columns, centre, chunksize)
    r, n = _correlations(acc)
    rows = int(acc["rows"][0])
    if rows == 0:
        return CheckResult.abstained(CHECK, spec_t.name, "the table is empty -- nothing to measure")

    # ⚠ An impossible correlation invalidates the whole matrix, not just its own cell.
    finite = np.isfinite(r)
    if finite.any() and np.nanmax(np.abs(r[finite])) > 1.0 + _R_TOLERANCE:
        flat = np.where(finite, np.abs(r), -np.inf).argmax()
        i, j = divmod(int(flat), len(columns))
        return CheckResult.abstained(
            CHECK, spec_t.name,
            f"the accumulation produced |r| = {abs(r[i, j]):.12f} for "
            f"({columns[i]}, {columns[j]}), which is impossible -- the co-moment sums have "
            f"lost precision and every number in this matrix is suspect")

    upper = np.triu(np.ones_like(r, dtype=bool), k=1)
    candidates = upper & finite & (np.abs(r) >= spec.redundancy_r) & (n >= _MIN_PAIRWISE_N)
    n_ok = np.diag(acc["n"])

    pairs = []
    for i, j in zip(*np.where(candidates)):
        i, j = int(i), int(j)
        shared = int(n[i, j])
        identical = int(acc["eq"][i, j]) == shared
        # Which leg survives: the one with more non-null coverage, and on a tie the shorter
        # name, which in this repo is the base leg rather than a derived view of it.
        keep, drop = ((columns[i], columns[j]) if (n_ok[i], -len(columns[i]))
                      >= (n_ok[j], -len(columns[j])) else (columns[j], columns[i]))
        pairs.append({"a": columns[i], "b": columns[j], "r": float(r[i, j]),
                      "n": shared, "exact_equal": int(acc["eq"][i, j]),
                      "identical": identical, "keep": keep, "drop": drop,
                      "n_ok_a": int(n_ok[i]), "n_ok_b": int(n_ok[j]),
                      "n_ok_keep": int(n_ok[i] if keep == columns[i] else n_ok[j])})
    pairs.sort(key=lambda p: (not p["identical"], -abs(p["r"])))

    findings: list[Finding] = []
    for pair in pairs[:_MAX_FINDINGS]:
        if pair["identical"]:
            score, observed = 10, (f"identical on all {pair['n']:,} rows where both are "
                                   f"present (r = {pair['r']:.6f})")
        elif abs(pair["r"]) >= 0.999:
            score, observed = 8, (f"r = {pair['r']:.6f} over {pair['n']:,} shared rows, "
                                  f"{pair['exact_equal']:,} of them exactly equal")
        else:
            score, observed = 6, (f"r = {pair['r']:.6f} over {pair['n']:,} shared rows")
        findings.append(Finding.at(
            score, field=pair["drop"],
            observed=f"{pair['a']} ~ {pair['b']}: {observed}",
            expected=f"|r| < {spec.redundancy_r} between two legs that claim to measure "
                     f"different things. Keep `{pair['keep']}` ({pair['n_ok_keep']:,} "
                     f"non-null) and drop `{pair['drop']}` -- UNLESS the two are related by "
                     f"an accounting identity (`ebt - ebit == -intexp` is one), in which case "
                     f"the correlation is a tautology and the defect is that both were built "
                     f"as features",
            **pair))

    scope = {"rows": rows, "tickers": None, "legs": len(columns),
             "pairs": int(upper.sum()), "threshold": spec.redundancy_r,
             "min_pairwise_n": _MIN_PAIRWISE_N, "centred_on": centre_source,
             "source": "db(float64)"}
    metrics = {
        "legs": len(columns), "pairs_tested": int((upper & finite).sum()),
        "pairs_over_threshold": len(pairs),
        "identical_pairs": sum(1 for p in pairs if p["identical"]),
        "top_pairs": pairs[:100],
        "max_abs_r": (float(np.nanmax(np.abs(np.where(upper & finite, r, np.nan))))
                      if (upper & finite).any() else None),
    }
    return CheckResult.measured(CHECK, spec_t.name, findings, scope=scope, metrics=metrics)
