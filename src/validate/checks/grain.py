"""
grain.py  (src/validate/checks/grain.py)
--------------------------------------------------------------------------------------------
Is the table one row per declared key?

The key is `resolve(table).pk`, never inferred. Cube parts key on `(date, ticker)` and
`fundamentals_history_sec` on `(ticker, as_of, ...)`, and a check that assumes the first would
report a clean grain on the second by counting the wrong thing -- passing loudly while
measuring nothing. A table whose registry entry declares no pk therefore ABSTAINS.

⚠ READS float64 FROM THE DB, NEVER THE float32 CACHE. Duplicate detection is an exact-equality
test on the key, and while today's keys are a date and a string, the contract is the reason:
the moment a numeric column enters a pk, a float32 round-trip manufactures collisions that are
not in the table. `cache=` is accepted (every check takes it) and deliberately ignored.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.context import Context
from src.data_store.schema import Table, resolve
from src.validate.frame import key_columns
from src.validate.io import CHUNK_ROWS
from src.validate.result import CheckResult, Finding, full_table_only

log = logging.getLogger(__name__)

CHECK = "grain"

#: Duplicate keys named in the finding's evidence. The count is the measurement; the examples
#: are there so a reader can go and look at one.
_MAX_EXAMPLES = 10


def _ticker_column(keys: list[str], columns: list[str]) -> str | None:
    """The entity axis of the panel, if the table has one."""
    for candidate in ("ticker", "symbol"):
        if candidate in keys or candidate in columns:
            return candidate
    return None


def check_grain(context: Context, table: Table | str, *, config: Any = None,
                cache: Any = None, tickers: list[str] | None = None,
                chunksize: int = CHUNK_ROWS, **kwargs: Any) -> CheckResult:
    """Rows vs distinct declared-pk keys, nulls in a key column, and the panel's edges."""
    spec = resolve(table)
    if (declined := full_table_only(CHECK, spec.name, tickers)) is not None:
        return declined
    keys = list(key_columns(spec))
    if not keys:
        return CheckResult.abstained(
            CHECK, spec.name,
            "the registry declares no `pk` for this table, so there is no grain to test -- "
            "add one to src/data_store/schema.py rather than letting this check guess")

    live = context.store.columns(spec)
    missing = [k for k in keys if k not in live]
    if missing:
        # The declared key is not the key the table has. That is a finding in the registry,
        # not a measurement of the data, and nothing downstream of it would mean anything.
        return CheckResult.abstained(
            CHECK, spec.name,
            f"declared pk {keys} names column(s) {missing} the live table does not have "
            f"-- the registry and the database disagree about the grain")

    parts = list(context.store.iter_load(spec, columns=keys, chunksize=chunksize))
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=keys)

    rows = len(frame)
    if rows == 0:
        return CheckResult.abstained(CHECK, spec.name, "the table is empty -- nothing to measure")

    nulls = {k: int(frame[k].isna().sum()) for k in keys}
    distinct = int(len(frame.drop_duplicates(subset=keys)))
    duplicated = rows - distinct

    findings: list[Finding] = []
    if duplicated:
        counts = (frame[frame.duplicated(subset=keys, keep=False)]
                  .groupby(keys, dropna=False).size().sort_values(ascending=False))
        examples = [{**dict(zip(keys, key if isinstance(key, tuple) else (key,))),
                     "rows": int(n)} for key, n in counts.head(_MAX_EXAMPLES).items()]
        findings.append(Finding.at(
            10,
            observed=f"{duplicated:,} of {rows:,} rows share a key with another row "
                     f"({distinct:,} distinct keys, worst key repeats {int(counts.iloc[0])}x)",
            expected=f"exactly one row per declared key {tuple(keys)}",
            duplicate_rows=duplicated, distinct_keys=distinct, rows=rows,
            keys_repeated=int(len(counts)), examples=examples))

    for column, count in nulls.items():
        if count:
            findings.append(Finding.at(
                10, field=column,
                observed=f"{count:,} of {rows:,} rows carry NULL in key column `{column}`",
                expected="a key column is never NULL -- a NULL key cannot be joined to, "
                         "and every one of these rows is unaddressable",
                null_rows=count, rows=rows, share=round(count / rows, 6)))

    ticker_col = _ticker_column(keys, live)
    n_tickers = int(frame[ticker_col].nunique()) if ticker_col in frame.columns else None
    first_date, last_date = (context.store.bounds(spec) if spec.date_col else (None, None))

    scope = {"rows": rows, "tickers": n_tickers, "first_date": first_date,
             "last_date": last_date, "pk": keys, "source": "db(float64)"}
    metrics = {"rows": rows, "distinct_keys": distinct, "duplicate_rows": duplicated,
               "nulls_in_key": nulls, "tickers": n_tickers,
               "date_col": spec.date_col, "first_date": first_date, "last_date": last_date}
    return CheckResult.measured(CHECK, spec.name, findings, scope=scope, metrics=metrics)
