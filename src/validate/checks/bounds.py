"""
bounds.py  (src/validate/checks/bounds.py)
--------------------------------------------------------------------------------------------
The one check that asserts on MEANING, and therefore the one that cannot have a default.

A percent of a share class cannot exceed 100. A share of a board cannot exceed 1. Nothing in
the data says which of those two a column called `..._pct` is -- measured on
`cube_part_institutionals`, `f_ic_act_percent_of_class` runs 0 -> 99.99 (a percent) while
`f_ic_inst_ownership_pct` runs 5.2e-08 -> 1.97 (a fraction, despite the name, and one that
exceeds 1.0). A built-in rule for `_pct` would pass the first and fail the second for the
wrong reason.

So bounds are DECLARED per leg in `configs/validate.yml`, and a table that declares none makes
this check **ABSTAIN (exit 3)**, never pass. "0 legs out of bounds" from a table with no
declared bounds is the exact shape of a green run that measured nothing.

⚠ READS float64 FROM THE DB, NEVER THE float32 CACHE. This is a boundary test: `100.0` in
float32 is `100.0`, but a float64 value a few ULPs below a bound can round UP across it, and a
manufactured violation on a semantic bound is the most expensive kind of false positive here
-- it reads as a data-integrity break. The declared legs are a handful of columns, so the
direct read costs one narrow pass.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_store.schema import Table, resolve
from src.validate.frame import as_ts
from src.validate.io import read_columns
from src.validate.result import CheckResult, Finding, full_table_only
from src.validate.spec import load_spec

log = logging.getLogger(__name__)

CHECK = "bounds"

_WHY = ("no [lo, hi] is asserted for any leg of this table, and a bound is a claim about "
        "MEANING that no statistic can infer -- declare the legs whose range you can state "
        "from the filing, or accept that this table is unchecked")


def check_bounds(context: Context, table: Table | str, *, config: DictConfig,
                 cache: Any = None, tickers: list[str] | None = None,
                 bounds: dict[str, tuple[float, float]] | None = None,
                 **kwargs: Any) -> CheckResult:
    """Declared `[lo, hi]` per leg, with one named example per violation."""
    spec_t = resolve(table)
    if (declined := full_table_only(CHECK, spec_t.name, tickers)) is not None:
        return declined
    spec = load_spec(config, spec_t, bounds=bounds)
    declared: dict[str, tuple[float, float]] = spec.require("bounds", _WHY)

    live = set(context.store.columns(spec_t))
    date_col = spec_t.date_col
    ticker_col = next((c for c in ("ticker", "symbol") if c in live), None)

    findings: list[Finding] = []
    stale = sorted(c for c in declared if c not in live)
    for column in stale:
        findings.append(Finding.at(
            6, field=column,
            observed=f"`{column}` has a declared bound {list(declared[column])} but does not "
                     f"exist in {spec_t.name}",
            expected="a declared bound names a live column -- otherwise the leg it was "
                     "written for is renamed or gone, and it is silently unchecked",
            bound=list(declared[column])))

    testable = [c for c in declared if c in live]
    if not testable:
        return CheckResult.abstained(
            CHECK, spec_t.name,
            f"none of the {len(declared)} declared bound(s) names a live column "
            f"({', '.join(stale)}) -- the declaration is stale, so nothing was measured")

    keys = [c for c in (date_col, ticker_col) if c]
    # `cache=None`: see the module docstring. A boundary test reads the DB's own float64.
    frame = read_columns(context, spec_t, keys + testable, cache=None)
    rows = len(frame)
    if rows == 0:
        return CheckResult.abstained(CHECK, spec_t.name, "the table is empty -- nothing to measure")
    if date_col:
        frame[date_col] = as_ts(frame[date_col])

    metrics: dict[str, Any] = {"declared": {c: list(b) for c, b in declared.items()},
                              "stale": stale, "legs": {}}
    for column in testable:
        lo, hi = declared[column]
        values = pd.to_numeric(frame[column], errors="coerce").astype("float64")
        finite = np.isfinite(values.values)
        outside = finite & ((values.values < lo) | (values.values > hi))
        n_ok = int(finite.sum())
        n_bad = int(outside.sum())
        leg = {"lo": lo, "hi": hi, "n_ok": n_ok, "n_violations": n_bad,
               "share": round(n_bad / n_ok, 6) if n_ok else None,
               "min": float(values[finite].min()) if n_ok else None,
               "max": float(values[finite].max()) if n_ok else None}
        metrics["legs"][column] = leg
        if not n_bad:
            continue
        # The worst violator, named. A count says how much; one (ticker, date, value) says
        # where to look, which is the difference between a finding and a statistic.
        bad = values[outside]
        worst_idx = (bad - np.clip(bad, lo, hi)).abs().idxmax()
        example = {"value": float(values.loc[worst_idx])}
        if ticker_col:
            example["ticker"] = str(frame.loc[worst_idx, ticker_col])
        if date_col:
            example["date"] = frame.loc[worst_idx, date_col]
        leg["worst"] = example
        findings.append(Finding.at(
            9, field=column, ticker=example.get("ticker"),
            observed=f"{n_bad:,} of {n_ok:,} finite values fall outside [{lo}, {hi}] "
                     f"(range seen {leg['min']:.6g} -> {leg['max']:.6g}); worst "
                     f"{example.get('ticker', '?')} "
                     f"{pd.Timestamp(example['date']).date() if date_col else ''} "
                     f"= {example['value']:.6g}",
            expected=f"every value in [{lo}, {hi}] -- the bound is what the quantity MEANS, "
                     f"so a value outside it is a unit error, a wrong denominator or a "
                     f"mis-tagged source, not an extreme observation",
            n_violations=n_bad, n_ok=n_ok, share=leg["share"],
            min=leg["min"], max=leg["max"], bound=[lo, hi], worst=example))

    first_date, last_date = (context.store.bounds(spec_t) if date_col else (None, None))
    n_tickers = int(frame[ticker_col].nunique()) if ticker_col else None
    scope = {"rows": rows, "tickers": n_tickers, "first_date": first_date,
             "last_date": last_date, "legs_declared": len(declared),
             "legs_tested": len(testable), "source": "db(float64)"}
    return CheckResult.measured(CHECK, spec_t.name, findings, scope=scope, metrics=metrics)
