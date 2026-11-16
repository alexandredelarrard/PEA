"""Read exact scoped source cells without claiming whole-table or feature certification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.context import Context
from src.data_store.schema import Table
from src.validate.result import CheckResult, Finding, jsonable


@dataclass(frozen=True)
class SampleSelection:
    """A pinned projection and explicit row scope, independent of the window resolver."""

    columns: tuple[str, ...]
    where: dict[str, Any]
    since: str | None = None
    until: str | None = None
    date_col: str | None = None


def check_window_sample(context: Context, table: Table, *, selection: SampleSelection) -> CheckResult:
    """Check selected persisted keys, clocks and values; missing evidence abstains."""
    # Load only the pinned projection and scope, including known-empty markers.
    available = context.store.columns(table)
    required = set(table.pk) | set(selection.columns)
    missing = sorted(required - set(available))
    if missing:
        return CheckResult.abstained("window_sample", table.name, f"Missing sample columns: {missing}")
    if not selection.where:
        raise ValueError("A sample needs explicit key/partition scope; an unscoped table read is not a sample")
    if not set(table.pk) <= set(selection.columns):
        raise ValueError(f"{table.name}: the sample projection must include every primary-key column")
    clock = selection.date_col or table.date_col
    if (selection.since is not None or selection.until is not None) and clock not in selection.columns:
        raise ValueError(f"{table.name}: bounded samples must project their clock")
    where: dict[str, Any] | pd.DataFrame = selection.where
    if set(selection.where) == set(table.pk) and all(not isinstance(value, list | tuple | set) for value in selection.where.values()):
        where = pd.DataFrame([selection.where])
    df_rows = context.store.load(
        table,
        columns=selection.columns,
        where=where,
        since=selection.since,
        until=selection.until,
        date_col=clock,
        markers=True,
        optional=True,
        order_by=table.pk,
    )
    if df_rows is None or df_rows.empty:
        return CheckResult.abstained("window_sample", table.name, "No rows in the pinned sample scope")

    # Check grain and boundaries without calling a production timing resolver.
    findings: list[Finding] = []
    bad_keys = df_rows[list(table.pk)].isna().any(axis=1) | df_rows.duplicated(list(table.pk), keep=False)
    if bad_keys.any():
        findings.append(Finding.at(8, f"{int(bad_keys.sum())} invalid/duplicate sample keys", "Non-null unique primary keys"))
    if clock in df_rows.columns:
        dates = pd.to_datetime(df_rows[clock], errors="coerce")
        outside = dates.isna()
        if selection.since is not None:
            outside |= dates < pd.Timestamp(selection.since)
        if selection.until is not None:
            outside |= dates > pd.Timestamp(selection.until)
        if outside.any():
            findings.append(
                Finding.at(8, f"{int(outside.sum())} missing/out-of-scope clocks", "Every sampled row inside the declared inclusive interval")
            )

    # Preserve precision, actual nulls and source provenance for repeatable comparison.
    records = jsonable(df_rows.to_dict(orient="records"))
    identity = {
        "columns": list(df_rows.columns),
        "dtypes": [str(dtype) for dtype in df_rows.dtypes],
        "cells": [[f"{type(value).__name__}:{value!r}" for value in row] for row in df_rows.itertuples(index=False, name=None)],
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()
    return CheckResult.measured(
        "window_sample",
        table.name,
        findings,
        scope={"rows": len(df_rows), "where": jsonable(selection.where), "since": selection.since, "until": selection.until, "clock": clock},
        metrics={"sha256": digest, "nulls": df_rows.isna().sum().to_dict(), "records": records, "identity": identity},
        reason="Scoped persisted-cell evidence only; feature economics and actual producer writes require separate public-run checks.",
    )
