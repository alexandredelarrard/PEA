"""
profile.py  (src/validate/checks/profile.py)
--------------------------------------------------------------------------------------------
One row of distributional facts per column, on every row of the table, in groups of 8.

WHY THIS IS THE CHECK THAT EARNS ITS KEEP. The 2026-09-04 cube audit found **65 of 179
features entirely null** -- not thin, not stale: zero non-null values, across every ticker and
every session -- and sixty-eight tests were green throughout, because every one of them
asserted on shape. A dead leg has the right dtype, the right row count and the right name. The
only thing that finds it is looking at the values.

Groups of 8 are the memory control: `cube_part_fundamentals` is 3,315,035 x 249 x 8B = 6.6 GB
whole and 210 MB eight columns at a time.

⚠ RUN `pull` FIRST ON A WIDE TABLE. One group is one pass over the table, so 249 legs is 32
passes: against Postgres that is hours (measured -- a single 9-column streamed pass over
3.3M rows runs minutes, and this check makes 32 of them), while against the parquet snapshot
each group is a columnar read of exactly the 8 columns it needs. The snapshot is what makes
this check runnable on the widest part, which is why `data-check.md` pulls before it sweeps.

⚠ THE CACHE IS float32 AND TWO OF THESE FINDINGS ARE DTYPE-SENSITIVE. A float64 value above
3.4e38 becomes `inf` in float32, and values agreeing to 1 part in 2^24 collapse to one. So any
leg this check is about to call DEAD, CONSTANT or INFINITE off the cache is re-read from the
DB in float64 first, in ONE extra pass over just those columns -- typically none of them, and
never more than a handful. A finding is a claim about the database, not about the snapshot.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.context import Context
from src.data_store.schema import Table, resolve
from src.validate.frame import as_ts, column_groups, feature_columns
from src.validate.io import cache_used, read_columns, read_meta
from src.validate.result import CheckResult, Finding, full_table_only
from src.validate.utils.outliers import count_mad_outliers, mad_center_scale

log = logging.getLogger(__name__)

CHECK = "profile"

#: Columns read at once. See the module docstring for the 6.6 GB this divides.
GROUP = 8

#: Modified-Z beyond which a point counts as an extreme value, the repo-wide constant from
#: `outliers.py`. Reported as a COUNT, never as a finding: an extreme value in a feature is
#: usually the feature working.
_MAD_THRESHOLD = 3.5

_MAX_FINDINGS = 40

#: Above this many legs, a DB-sourced run is enough passes to be worth warning about.
_WIDE = 32


def _stats(values: pd.Series, dates: pd.Series | None) -> dict[str, Any]:
    """Everything `data-check.md`'s per-field table asks for, from one pass over one column."""
    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    finite = np.isfinite(numeric.values)
    ok = numeric[finite]
    n = int(len(numeric))
    out: dict[str, Any] = {
        "n": n,
        "n_ok": int(len(ok)),
        "n_null": int(numeric.isna().sum()),
        # Values that are present but not numbers. `frame._is_leg` admits an all-null
        # `object` column because that is what an entirely-NULL double looks like to pandas;
        # this counts, on the FULL table, how wrong that guess was. Anything above zero is a
        # text column and gets no findings -- reporting "0 finite values" on a column of
        # strings would be a defect the check invented.
        "n_uncoercible": int(values.notna().sum() - numeric.notna().sum()),
        # NaN and Inf are counted apart: a NaN is missing data, an Inf is a division that
        # went wrong upstream and is still sitting in the column.
        "n_inf": int(np.isinf(numeric.values).sum()),
        "n_distinct": int(ok.nunique()),
    }
    if len(ok) == 0:
        out.update({k: None for k in ("min", "p01", "p50", "p99", "p999", "max", "mean", "sd",
                                      "mad_center", "mad_scale", "n_mad_outliers",
                                      "first_date", "last_date")})
        return out
    quantiles = ok.quantile([0.01, 0.5, 0.99, 0.999])
    center, scale = mad_center_scale(ok)
    out.update({
        "min": float(ok.min()), "p01": float(quantiles.loc[0.01]),
        "p50": float(quantiles.loc[0.5]), "p99": float(quantiles.loc[0.99]),
        "p999": float(quantiles.loc[0.999]), "max": float(ok.max()),
        "mean": float(ok.mean()), "sd": float(ok.std(ddof=1)) if len(ok) > 1 else 0.0,
        "mad_center": center, "mad_scale": scale,
        "n_mad_outliers": count_mad_outliers(ok, threshold=_MAD_THRESHOLD),
    })
    if dates is not None:
        seen = dates[numeric.notna().values]
        out["first_date"] = seen.min() if len(seen) else None
        out["last_date"] = seen.max() if len(seen) else None
    return out


def _suspect(stats: dict[str, Any]) -> bool:
    """Is this leg about to be called dead, constant or infinite? See the dtype note."""
    return stats["n_ok"] == 0 or stats["n_distinct"] <= 1 or stats["n_inf"] > 0


def check_profile(context: Context, table: Table | str, *, config: Any = None,
                  cache: Any = None, tickers: list[str] | None = None,
                  group: int = GROUP, **kwargs: Any) -> CheckResult:
    """Per-column distribution over the full table, and the dead/constant/infinite legs."""
    spec = resolve(table)
    if (declined := full_table_only(CHECK, spec.name, tickers)) is not None:
        return declined
    date_col = spec.date_col
    columns = feature_columns(context, spec)
    if not columns:
        return CheckResult.abstained(CHECK, spec.name,
                                     "the table has no numeric non-key column to profile")

    from_cache = cache_used(cache, spec)
    if not from_cache and len(columns) > _WIDE:
        log.warning("profile %s: %d legs = %d full passes over the DB and no snapshot in %s "
                    "-- run `python -m src validate pull -T %s -o %s` first",
                    spec.name, len(columns), -(-len(columns) // group), cache, spec.name, cache)
    meta = read_meta(cache) if from_cache else None
    float32_source = from_cache and (meta or {}).get("float_dtype") == "float32"

    # The ticker axis rides along with the first group -- one extra column on one read, so
    # the summary line can say how many names the numbers cover instead of `None tickers`.
    ticker_col = next((c for c in ("ticker", "symbol") if c in context.store.columns(spec)), None)
    n_tickers: int | None = None

    stats: dict[str, dict[str, Any]] = {}
    rows = 0
    for index, block in enumerate(column_groups(columns, group)):
        wanted = ([date_col] + block) if date_col else list(block)
        if index == 0 and ticker_col:
            wanted = wanted + [ticker_col]
        frame = read_columns(context, spec, wanted, cache=cache)
        if index == 0 and ticker_col:
            n_tickers = int(frame[ticker_col].nunique())
        rows = max(rows, len(frame))
        dates = as_ts(frame[date_col]) if date_col else None
        for column in block:
            stats[column] = _stats(frame[column], dates)
        log.info("profile %s: %d/%d columns", spec.name, len(stats), len(columns))

    # -- confirm the dtype-sensitive verdicts against float64, in one pass ---------------- #
    confirmed: list[str] = []
    if float32_source:
        suspects = [c for c, s in stats.items() if _suspect(s)]
        if suspects:
            log.info("profile %s: re-reading %d suspect leg(s) from the DB in float64",
                     spec.name, len(suspects))
            wanted = ([date_col] + suspects) if date_col else suspects
            frame = read_columns(context, spec, wanted, cache=None)
            dates = as_ts(frame[date_col]) if date_col else None
            for column in suspects:
                stats[column] = _stats(frame[column], dates)
            confirmed = suspects

    findings: list[Finding] = []
    text = [c for c, s in stats.items() if s["n_uncoercible"] > 0]
    legs = {c: s for c, s in stats.items() if c not in set(text)}
    dead = [c for c, s in legs.items() if s["n_ok"] == 0]
    constant = [c for c, s in legs.items() if s["n_ok"] > 0 and s["n_distinct"] == 1]
    infinite = [c for c, s in legs.items() if s["n_inf"] > 0]

    # `n_distinct == 0` and `n_ok == 0` are the same leg by construction -- `n_distinct`
    # counts finite values -- so a dead leg is filed once, not twice.
    for column in dead[:_MAX_FINDINGS]:
        findings.append(Finding.at(
            10, field=column,
            observed=f"0 finite values in {stats[column]['n']:,} rows "
                     f"({stats[column]['n_null']:,} null)",
            expected="a feature carries values; a column that is null everywhere is a "
                     "builder that never ran, and it is invisible to every shape test",
            **{k: stats[column][k] for k in ("n", "n_ok", "n_null", "n_inf")}))

    for column in constant[:_MAX_FINDINGS]:
        findings.append(Finding.at(
            7, field=column,
            observed=f"one distinct value ({stats[column]['min']!r}) over "
                     f"{stats[column]['n_ok']:,} finite rows",
            expected="a feature varies; a constant column carries no information and will "
                     "be dropped by any model, silently",
            value=stats[column]["min"], n_ok=stats[column]["n_ok"]))

    for column in infinite[:_MAX_FINDINGS]:
        findings.append(Finding.at(
            9, field=column,
            observed=f"{stats[column]['n_inf']:,} infinite values",
            expected="finite or null -- an Inf is a division by zero that survived into the "
                     "table, and it poisons every mean, sd and z-score computed over it",
            n_inf=stats[column]["n_inf"], n=stats[column]["n"]))

    first_date, last_date = (context.store.bounds(spec) if date_col else (None, None))
    scope = {"rows": rows, "tickers": n_tickers, "first_date": first_date, "last_date": last_date,
             "columns": len(columns), "group": group,
             "source": ("cache(float32)" if float32_source else "cache" if from_cache else "db"),
             "confirmed_from_db": confirmed}
    metrics = {
        "columns": len(columns),
        "dead": dead, "constant": constant, "infinite": infinite, "skipped_text": text,
        # `redundancy` centres on these rather than recomputing 249 means in its own pass.
        "mean": {c: s["mean"] for c, s in stats.items()},
        "stats": stats,
    }
    reason = ("" if not float32_source or not confirmed else
              f"{len(confirmed)} leg(s) flagged off the float32 snapshot were re-read from "
              f"the DB in float64 before filing")
    return CheckResult.measured(CHECK, spec.name, findings, scope=scope, metrics=metrics,
                                reason=reason)
