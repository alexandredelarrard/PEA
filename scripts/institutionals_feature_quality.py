"""Read-only quality gates for the institutionals feature refactor."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Project imports intentionally follow the repository-root path bootstrap.
# ruff: noqa: E402

from src.context import get_config_context
from src.data_store.schema import Table, Tables
from src.utils.config import read_config
from src.validate.checks.catalogue import _load as load_catalogue
from src.validate.checks.profile import _stats as profile_stats
from src.validate.checks.redundancy import _correlations
from src.validate.checks.timeseries import _frozen_legs, _hole_condition, _known_ineligible_reason, _measure
from src.validate.io import CHUNK_ROWS
from src.validate.result import jsonable
from src.validate.spec import load_spec

log = logging.getLogger(__name__)

TABLE = Tables.cube_part_institutionals
KEYS = list(TABLE.pk)
SOURCE_TABLES: tuple[Table, ...] = (
    Tables.cube_part_prices,
    Tables.prices_splits,
    Tables.fundamentals_history,
    Tables.sp500_tickers,
    Tables.sec13f_hr,
    Tables.sec13f_manager_holdings,
    Tables.superinvestor_roster,
    Tables.cusip_ticker_map,
    Tables.insider_transactions,
    Tables.insider_transactions_live,
    Tables.insider_transactions_live_coverage,
    Tables.short_interest,
    Tables.sec_fails_to_deliver,
    Tables.sec_13d,
    Tables.sec_13g,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _schema(path: Path) -> tuple[pq.ParquetFile, list[str], dict[str, str]]:
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    return parquet, schema.names, {field.name: str(field.type) for field in schema}


def _date_bounds(path: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None, int]:
    values = pd.to_datetime(pq.read_table(path, columns=[TABLE.date_col]).column(0).to_pandas(), errors="coerce")
    invalid = int(values.isna().sum())
    return (None, None, invalid) if values.empty or values.notna().sum() == 0 else (values.min(), values.max(), invalid)


def _same_date(left: Any, right: pd.Timestamp | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return pd.Timestamp(left).normalize() == right.normalize()


def _verified_snapshot(path: Path, metadata: dict[str, Any], *, as_of: pd.Timestamp | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    parquet, columns, dtypes = _schema(path)
    rows = int(parquet.metadata.num_rows)
    missing = [key for key in KEYS if key not in columns]
    first_date, last_date, invalid_dates = _date_bounds(path) if TABLE.date_col in columns else (None, None, rows)
    errors: list[str] = []
    if metadata.get("table") != TABLE.name:
        errors.append(f"metadata table is {metadata.get('table')!r}, expected {TABLE.name!r}")
    if int(metadata.get("rows", -1)) != rows:
        errors.append(f"metadata rows={metadata.get('rows')!r}, parquet rows={rows}")
    if list(metadata.get("columns") or []) != columns:
        errors.append("metadata columns do not exactly match parquet column order")
    if metadata.get("pk") is not None and list(metadata["pk"]) != KEYS:
        errors.append(f"metadata pk={metadata['pk']!r}, expected {KEYS!r}")
    if metadata.get("date_col") is not None and metadata["date_col"] != TABLE.date_col:
        errors.append(f"metadata date_col={metadata['date_col']!r}, expected {TABLE.date_col!r}")
    if missing:
        errors.append(f"parquet is missing keys {missing}")
    if invalid_dates:
        errors.append(f"parquet carries {invalid_dates} invalid/null dates")
    if metadata.get("first_date") is not None and not _same_date(metadata["first_date"], first_date):
        errors.append("metadata first_date does not match parquet")
    if metadata.get("last_date") is not None and not _same_date(metadata["last_date"], last_date):
        errors.append("metadata last_date does not match parquet")
    if metadata.get("dtypes") is not None and metadata["dtypes"] != dtypes:
        errors.append("metadata dtypes do not exactly match parquet schema")
    declared_hash = metadata.get("snapshot_sha256") or metadata.get("sha256")
    actual_hash = _sha256(path)
    if declared_hash is not None and declared_hash != actual_hash:
        errors.append("metadata snapshot hash does not match parquet bytes")
    if as_of is not None and last_date is not None and last_date.normalize() > as_of.normalize():
        errors.append(f"snapshot ends {last_date.date()} after requested as-of {as_of.date()}")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        **metadata,
        "table": TABLE.name,
        "rows": rows,
        "columns": columns,
        "dtypes": dtypes,
        "pk": KEYS,
        "date_col": TABLE.date_col,
        "first_date": first_date,
        "last_date": last_date,
        "snapshot_sha256": actual_hash,
        "snapshot_bytes": path.stat().st_size,
    }


def _source_metadata(store: Any, table: Table) -> dict[str, Any]:
    if not store.exists(table):
        return {"table": table.name, "exists": False, "rows": 0, "columns": [], "first_date": None, "last_date": None}
    columns = list(store.columns(table))
    first_date, last_date = store.bounds(table, table.date_col) if table.date_col and table.date_col in columns else (None, None)
    return {
        "table": table.name,
        "exists": True,
        "rows": int(store.row_count(table)),
        "columns": columns,
        "columns_sha256": hashlib.sha256(json.dumps(columns, separators=(",", ":")).encode()).hexdigest(),
        "date_col": table.date_col,
        "first_date": first_date,
        "last_date": last_date,
    }


def freeze_baseline(config: str, snapshot: Path, as_of: str, out: Path) -> dict[str, Any]:
    meta_path = snapshot.parent / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"snapshot metadata does not exist: {meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"snapshot metadata is not a JSON object: {meta_path}")
    cutoff = pd.Timestamp(as_of).normalize()
    verified = _verified_snapshot(snapshot, metadata, as_of=cutoff)

    _, context = get_config_context(config, use_cache=False, save=False)
    sources = [_source_metadata(context.store, table) for table in SOURCE_TABLES]
    out.mkdir(parents=True, exist_ok=True)
    frozen_path = out / "baseline.parquet"
    if not frozen_path.exists():
        shutil.copy2(snapshot, frozen_path)
    if _sha256(frozen_path) != verified["snapshot_sha256"]:
        raise OSError("existing baseline hash differs from source snapshot")
    frozen_meta = {**verified, "as_of": cutoff, "snapshot": frozen_path.name}
    _write_json(out / "baseline-meta.json", frozen_meta)
    manifest = {
        "status": "pass",
        "table": TABLE.name,
        "as_of": cutoff,
        "snapshot": {
            "source": str(snapshot.resolve()),
            "frozen": str(frozen_path.resolve()),
            "sha256": verified["snapshot_sha256"],
            "bytes": verified["snapshot_bytes"],
            "rows": verified["rows"],
            "columns": verified["columns"],
            "dtypes": verified["dtypes"],
            "first_date": verified["first_date"],
            "last_date": verified["last_date"],
        },
        "source_tables": sources,
        "source_contract": "DataStore exists/columns/row_count/bounds metadata only; no table values were read",
    }
    _write_json(out / "input-manifest.json", manifest)
    return manifest


def _keys(path: Path) -> tuple[pd.DataFrame, np.ndarray, int, int]:
    frame = pq.read_table(path, columns=KEYS).to_pandas()
    frame[TABLE.date_col] = pd.to_datetime(frame[TABLE.date_col], errors="coerce")
    nulls = int(frame[KEYS].isna().any(axis=1).sum())
    duplicates = int(frame.duplicated(KEYS, keep=False).sum())
    ordered = frame.assign(_row=np.arange(len(frame), dtype="int64")).sort_values(KEYS, kind="stable")
    return ordered[KEYS].reset_index(drop=True), ordered["_row"].to_numpy(), nulls, duplicates


def _column_values(path: Path, column: str, order: np.ndarray) -> tuple[pd.Series, np.ndarray]:
    values = pq.read_table(path, columns=[column]).column(0).combine_chunks()
    nulls = values.is_null().to_numpy(zero_copy_only=False)[order]
    return values.to_pandas().iloc[order].reset_index(drop=True), nulls


def compare_refactor(before: Path, after: Path, out: Path) -> dict[str, Any]:
    _, before_columns, before_dtypes = _schema(before)
    _, after_columns, after_dtypes = _schema(after)
    missing_keys = {"before": [k for k in KEYS if k not in before_columns], "after": [k for k in KEYS if k not in after_columns]}
    if any(missing_keys.values()):
        report = {
            "status": "fail",
            "table": TABLE.name,
            "checks": {
                "keys": {"pass": False, "missing": missing_keys},
                "schema": {"pass": set(before_columns) == set(after_columns)},
                "column_order": {"pass": before_columns == after_columns},
                "dtypes": {"pass": before_dtypes == after_dtypes},
                "nulls": {"pass": False, "reason": "cannot align rows without both keys"},
                "nonfinite": {"pass": False, "reason": "cannot align rows without both keys"},
                "values": {"pass": False, "reason": "cannot align rows without both keys"},
            },
        }
        _write_json(out, report)
        return report

    before_keys, before_order, before_null_keys, before_duplicate_keys = _keys(before)
    after_keys, after_order, after_null_keys, after_duplicate_keys = _keys(after)
    keys_equal = before_keys.equals(after_keys)
    schema_equal = set(before_columns) == set(after_columns)
    common = [column for column in before_columns if column in after_columns]
    dtype_diffs = {
        column: {"before": before_dtypes[column], "after": after_dtypes[column]} for column in common if before_dtypes[column] != after_dtypes[column]
    }
    null_diffs: dict[str, int] = {}
    nonfinite_diffs: dict[str, int] = {}
    value_diffs: dict[str, int] = {}
    if keys_equal:
        for column in common:
            if column in KEYS:
                continue
            left, left_null = _column_values(before, column, before_order)
            right, right_null = _column_values(after, column, after_order)
            null_diff = int(np.count_nonzero(left_null != right_null))
            if null_diff:
                null_diffs[column] = null_diff
            left_num, right_num = pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")
            numeric = left_num.notna().sum() + right_num.notna().sum() > 0
            if numeric:
                left_array, right_array = left_num.to_numpy(dtype="float64"), right_num.to_numpy(dtype="float64")
                left_class = np.select([np.isposinf(left_array), np.isneginf(left_array), np.isnan(left_array)], [1, -1, 2], default=0)
                right_class = np.select([np.isposinf(right_array), np.isneginf(right_array), np.isnan(right_array)], [1, -1, 2], default=0)
                finite_pair = np.isfinite(left_array) & np.isfinite(right_array) & ~left_null & ~right_null
                nonfinite_diff = int(np.count_nonzero((left_class != right_class) & ~left_null & ~right_null))
                value_diff = int(np.count_nonzero((left_array != right_array) & finite_pair))
            else:
                both = ~left_null & ~right_null
                nonfinite_diff = 0
                value_diff = int(np.count_nonzero((left.astype("string") != right.astype("string")).to_numpy() & both))
            if nonfinite_diff:
                nonfinite_diffs[column] = nonfinite_diff
            if value_diff:
                value_diffs[column] = value_diff

    checks = {
        "keys": {
            "pass": keys_equal and before_null_keys == after_null_keys == before_duplicate_keys == after_duplicate_keys == 0,
            "equal": keys_equal,
            "before_null_rows": before_null_keys,
            "after_null_rows": after_null_keys,
            "before_duplicate_rows": before_duplicate_keys,
            "after_duplicate_rows": after_duplicate_keys,
        },
        "schema": {
            "pass": schema_equal,
            "before_only": sorted(set(before_columns) - set(after_columns)),
            "after_only": sorted(set(after_columns) - set(before_columns)),
        },
        "column_order": {"pass": before_columns == after_columns, "before": before_columns, "after": after_columns},
        "dtypes": {"pass": not dtype_diffs and schema_equal, "differences": dtype_diffs},
        "nulls": {"pass": keys_equal and not null_diffs, "differences": null_diffs},
        "nonfinite": {"pass": keys_equal and not nonfinite_diffs, "differences": nonfinite_diffs},
        "values": {"pass": keys_equal and not value_diffs, "differences": value_diffs},
    }
    status = "pass" if all(check["pass"] for check in checks.values()) else "fail"
    report = {
        "status": status,
        "table": TABLE.name,
        "before": {"path": str(before.resolve()), "sha256": _sha256(before), "rows": len(before_keys)},
        "after": {"path": str(after.resolve()), "sha256": _sha256(after), "rows": len(after_keys)},
        "checks": checks,
    }
    _write_json(out, report)
    return report


def _result(check: str, status: str, *, scope: dict[str, Any], metrics: dict[str, Any], reason: str = "") -> dict[str, Any]:
    return {"check": check, "table": TABLE.name, "status": status, "reason": reason, "scope": scope, "metrics": metrics}


def _grain(path: Path, columns: list[str]) -> tuple[dict[str, Any], pd.DataFrame | None]:
    missing = [key for key in KEYS if key not in columns]
    if missing:
        return _result("grain", "fail", scope={"snapshot": str(path)}, metrics={"missing_keys": missing}), None
    keys, _, null_rows, duplicate_rows = _keys(path)
    invalid_dates = int(pd.to_datetime(keys[TABLE.date_col], errors="coerce").isna().sum())
    status = "pass" if not null_rows and not duplicate_rows and not invalid_dates else "fail"
    return (
        _result(
            "grain",
            status,
            scope={"rows": len(keys), "keys": KEYS},
            metrics={"null_key_rows": null_rows, "duplicate_key_rows": duplicate_rows, "invalid_date_rows": invalid_dates},
        ),
        keys,
    )


def _schema_check(path: Path, metadata: dict[str, Any], columns: list[str], dtypes: dict[str, str], rows: int) -> dict[str, Any]:
    expected_hash = metadata.get("snapshot_sha256") or metadata.get("sha256")
    actual_hash = _sha256(path)
    comparisons = {
        "table": metadata.get("table") == TABLE.name,
        "rows": metadata.get("rows") == rows,
        "columns": metadata.get("columns") == columns,
        "column_order": metadata.get("columns") == columns,
        "dtypes": metadata.get("dtypes") == dtypes,
        "pk": metadata.get("pk") == KEYS,
        "date_col": metadata.get("date_col") == TABLE.date_col,
        "snapshot_sha256": expected_hash == actual_hash,
    }
    return _result(
        "schema",
        "pass" if all(comparisons.values()) else "fail",
        scope={"snapshot": str(path.resolve()), "metadata_hash": expected_hash, "snapshot_hash": actual_hash},
        metrics={"comparisons": comparisons, "columns": columns, "dtypes": dtypes},
    )


def _catalogue_check(path: Path, columns: list[str]) -> dict[str, Any]:
    try:
        entries, split = load_catalogue(path, TABLE.name)
    except (OSError, ValueError, SyntaxError, json.JSONDecodeError) as exc:
        return _result("catalogue", "abstain", scope={"catalogue": str(path)}, metrics={}, reason=str(exc))
    live: dict[str, list[str]] = {}
    for column in columns:
        if column in KEYS:
            continue
        characteristic = split(column)[0] if callable(split) else column
        live.setdefault(characteristic, []).append(column)
    missing = sorted(set(entries) - set(live))
    undocumented = sorted(set(live) - set(entries))
    blank = sorted(name for name in set(entries) & set(live) if not str(entries[name] or "").strip())
    return _result(
        "catalogue",
        "pass" if not missing and not undocumented and not blank else "fail",
        scope={"catalogue": str(path.resolve()), "catalogue_sha256": _sha256(path), "split": callable(split)},
        metrics={
            "catalogued_not_candidate": missing,
            "candidate_not_catalogued": undocumented,
            "blank_descriptions": blank,
            "resolved_columns": live,
        },
    )


def _numeric_columns(path: Path) -> list[str]:
    schema = pq.ParquetFile(path).schema_arrow
    return [
        field.name
        for field in schema
        if field.name not in KEYS and (pa.types.is_integer(field.type) or pa.types.is_floating(field.type) or pa.types.is_decimal(field.type))
    ]


def _profile(path: Path, columns: list[str]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    numeric = _numeric_columns(path)
    non_numeric = sorted(set(columns) - set(KEYS) - set(numeric))
    stats: dict[str, dict[str, Any]] = {}
    for index, column in enumerate(numeric, start=1):
        values = pq.read_table(path, columns=[column]).column(0).to_pandas()
        stats[column] = profile_stats(values, None)
        if index % 20 == 0 or index == len(numeric):
            log.info("profile candidate: %d/%d columns", index, len(numeric))
    dead = [name for name, values in stats.items() if values["n_ok"] == 0]
    constant = [name for name, values in stats.items() if values["n_ok"] > 0 and values["n_distinct"] == 1]
    infinite = [name for name, values in stats.items() if values["n_inf"] > 0]
    failures = dead + constant + infinite + non_numeric
    result = _result(
        "profile",
        "abstain" if not numeric else "fail" if failures else "pass",
        scope={"rows": next(iter(stats.values()))["n"] if stats else 0, "numeric_columns": len(numeric)},
        metrics={"dead": dead, "constant": constant, "infinite": infinite, "non_numeric": non_numeric, "stats": stats},
        reason="no numeric feature column" if not numeric else "",
    )
    return result, stats


def _bounds(path: Path, declared: dict[str, tuple[float, float]], columns: list[str]) -> dict[str, Any]:
    if not declared:
        return _result("bounds", "abstain", scope={}, metrics={}, reason="no bounds declared for this table")
    missing = sorted(set(declared) - set(columns))
    violations: dict[str, dict[str, Any]] = {}
    observed: dict[str, dict[str, float | None]] = {}
    for column, (lower, upper) in declared.items():
        if column not in columns:
            continue
        values = pd.to_numeric(pq.read_table(path, columns=[column]).column(0).to_pandas(), errors="coerce").to_numpy(dtype="float64")
        finite = values[np.isfinite(values)]
        low = None if not finite.size else float(finite.min())
        high = None if not finite.size else float(finite.max())
        observed[column] = {"min": low, "max": high, "lower": lower, "upper": upper}
        below = int(np.count_nonzero(finite < lower))
        above = int(np.count_nonzero(finite > upper))
        if below or above:
            violations[column] = {"below": below, "above": above, **observed[column]}
    return _result(
        "bounds",
        "pass" if not missing and not violations else "fail",
        scope={"declared": len(declared)},
        metrics={"missing_declared_columns": missing, "violations": violations, "observed": observed},
    )


def _redundancy(path: Path, stats: dict[str, dict[str, Any]], threshold: float) -> dict[str, Any]:
    columns = list(stats)
    if len(columns) < 2:
        return _result("redundancy", "abstain", scope={"columns": len(columns)}, metrics={}, reason="fewer than two numeric feature columns")
    width = len(columns)
    centre = np.array([float(stats[column]["mean"] or 0.0) for column in columns], dtype="float64")
    acc = {name: np.zeros((width, width), dtype="float64") for name in ("n", "sx", "sxx", "sxy", "eq")}
    rows = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=CHUNK_ROWS, columns=columns):
        raw = batch.to_pandas()[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
        present = np.isfinite(raw)
        mask = present.astype("float64")
        centered = np.where(present, raw - centre, 0.0)
        acc["n"] += mask.T @ mask
        acc["sx"] += centered.T @ mask
        acc["sxx"] += (centered * centered).T @ mask
        acc["sxy"] += centered.T @ centered
        for index in range(width):
            acc["eq"][index] += ((raw == raw[:, index][:, None]) & present & present[:, index][:, None]).sum(axis=0)
        rows += len(raw)
    acc["rows"] = np.array([rows], dtype="float64")
    correlation, overlap = _correlations(acc)
    min_pair = min(100, rows)
    pairs: list[dict[str, Any]] = []
    for left in range(width):
        for right in range(left + 1, width):
            n = int(overlap[left, right])
            exact = n >= max(2, min_pair) and int(acc["eq"][left, right]) == n
            value = correlation[left, right]
            high = n >= min_pair and np.isfinite(value) and abs(value) >= threshold
            if exact or high:
                pairs.append(
                    {"left": columns[left], "right": columns[right], "n": n, "r": None if not np.isfinite(value) else float(value), "exact": exact}
                )
    pairs.sort(key=lambda item: (not item["exact"], -(abs(item["r"]) if item["r"] is not None else 0.0)))
    return _result(
        "redundancy",
        "fail" if pairs else "pass",
        scope={"rows": rows, "columns": width, "threshold": threshold, "min_pairwise_n": min_pair},
        metrics={"redundant_pairs": pairs[:40], "n_redundant_pairs": len(pairs)},
    )


def _timeseries(path: Path, stats: dict[str, dict[str, Any]], keys: pd.DataFrame | None, spec: Any) -> dict[str, Any]:
    if keys is None or keys.empty or not stats:
        return _result("timeseries", "abstain", scope={}, metrics={}, reason="no candidate time series to measure")
    columns = list(stats)
    frozen_legs, frozen_limit, frozen_scope = _frozen_legs(spec, columns)
    frozen_set = set(frozen_legs)
    holes: list[dict[str, Any]] = []
    explained: list[dict[str, Any]] = []
    frozen: list[dict[str, Any]] = []
    jumps: list[dict[str, Any]] = []
    missing_conditions: list[dict[str, str]] = []
    series = 0
    for index, column in enumerate(columns, start=1):
        wanted = [TABLE.date_col, "ticker", column]
        rule = spec.conditional_holes.get(column)
        if rule is not None and rule["active_field"] not in columns:
            missing_conditions.append({"field": column, "active_field": rule["active_field"]})
            continue
        if rule is not None and rule["active_field"] != column:
            wanted.append(rule["active_field"])
        frame = pq.read_table(path, columns=wanted).to_pandas()
        frame[TABLE.date_col] = pd.to_datetime(frame[TABLE.date_col], errors="coerce")
        frame = frame.sort_values(["ticker", TABLE.date_col], kind="stable")
        for ticker, group in frame.groupby("ticker", sort=False):
            series += 1
            found = _measure(
                group[column],
                pd.DatetimeIndex(group[TABLE.date_col]),
                spec,
                frozen_limit=frozen_limit if column in frozen_set else None,
                hole_eligible=_hole_condition(group, column, spec),
            )
            for row in found["hole"]:
                item = {"ticker": str(ticker), "field": column, **row}
                reason = _known_ineligible_reason(
                    spec,
                    ticker=str(ticker).upper(),
                    leg=column,
                    start=row["start"],
                    end=row["end"],
                )
                (explained if reason else holes).append({**item, **({"reason": reason} if reason else {})})
            explained.extend({"ticker": str(ticker), "field": column, **row} for row in found["explained"])
            frozen.extend({"ticker": str(ticker), "field": column, **row} for row in found["frozen"])
            jumps.extend({"ticker": str(ticker), "field": column, **row} for row in found["jump"])
        if index % 20 == 0 or index == len(columns):
            log.info("timeseries candidate: %d/%d columns", index, len(columns))
    stalled = [row for row in frozen if not row["at_extreme"]]
    saturated = [row for row in frozen if row["at_extreme"]]
    failures = bool(holes or stalled or missing_conditions)
    return _result(
        "timeseries",
        "fail" if failures else "pass",
        scope={
            "rows": len(keys),
            "tickers": int(keys["ticker"].nunique()),
            "dates": int(keys[TABLE.date_col].nunique()),
            "columns": len(columns),
            "series": series,
            "frozen_scope": frozen_scope,
        },
        metrics={
            "n_holes": len(holes),
            "n_explained_holes": len(explained),
            "n_frozen": len(stalled),
            "n_frozen_at_extreme": len(saturated),
            "n_jumps": len(jumps),
            "missing_condition_fields": missing_conditions,
            "worst_holes": sorted(holes, key=lambda row: -row["days"])[:40],
            "worst_frozen": sorted(stalled, key=lambda row: -row["days"])[:40],
            "worst_jumps": sorted(jumps, key=lambda row: -row["z"])[:40],
        },
        reason="" if frozen_limit is not None else f"FROZEN {frozen_scope}",
    )


def _leakage(keys: pd.DataFrame | None, metadata: dict[str, Any]) -> dict[str, Any]:
    if keys is None or keys.empty:
        return _result("leakage", "abstain", scope={}, metrics={}, reason="no candidate dates")
    last_date = pd.Timestamp(keys[TABLE.date_col].max()).normalize()
    as_of_raw = metadata.get("as_of")
    if as_of_raw is not None and last_date > pd.Timestamp(as_of_raw).normalize():
        return _result(
            "leakage",
            "fail",
            scope={"last_date": last_date, "as_of": as_of_raw},
            metrics={"future_candidate_dates": True},
            reason="candidate contains observations after its declared as-of date",
        )
    return _result(
        "leakage",
        "abstain",
        scope={"last_date": last_date, "as_of": as_of_raw},
        metrics={"future_candidate_dates": False if as_of_raw is not None else None},
        reason="a candidate parquet cannot prove source-publication causality or prefix invariance; validate those against frozen source inputs",
    )


def validate_candidate(config: str, snapshot: Path, metadata_path: Path, catalogue: Path, out: Path) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata is not a JSON object: {metadata_path}")
    parquet, columns, dtypes = _schema(snapshot)
    rows = int(parquet.metadata.num_rows)
    config_node = read_config(config)
    spec = load_spec(config_node, TABLE)

    checks: dict[str, dict[str, Any]] = {}
    checks["grain"], keys = _grain(snapshot, columns)
    checks["schema"] = _schema_check(snapshot, metadata, columns, dtypes, rows)
    checks["catalogue"] = _catalogue_check(catalogue, columns)
    checks["bounds"] = _bounds(snapshot, spec.bounds, columns)
    checks["profile"], stats = _profile(snapshot, columns)
    checks["redundancy"] = _redundancy(snapshot, stats, spec.redundancy_r)
    checks["timeseries"] = _timeseries(snapshot, stats, keys, spec)
    checks["leakage"] = _leakage(keys, metadata)

    out.mkdir(parents=True, exist_ok=True)
    for name, result in checks.items():
        _write_json(out / f"{name}.json", result)
    failed = sorted(name for name, result in checks.items() if result["status"] == "fail")
    abstained = sorted(name for name, result in checks.items() if result["status"] == "abstain")
    summary = {
        "status": "fail" if failed else "pass",
        "table": TABLE.name,
        "snapshot": str(snapshot.resolve()),
        "snapshot_sha256": _sha256(snapshot),
        "metadata": str(metadata_path.resolve()),
        "catalogue": str(catalogue.resolve()),
        "checks": {name: result["status"] for name, result in checks.items()},
        "failed": failed,
        "abstained": abstained,
        "source_contract": "candidate parquet + supplied metadata/catalogue/config only; no DataStore or live cube values/schema",
    }
    _write_json(out / "candidate-validation.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze-baseline")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--snapshot", type=Path, required=True)
    freeze.add_argument("--as-of", required=True)
    freeze.add_argument("--out", type=Path, required=True)

    compare = commands.add_parser("compare-refactor")
    compare.add_argument("--before", type=Path, required=True)
    compare.add_argument("--after", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)

    validate = commands.add_parser("validate-candidate")
    validate.add_argument("--config", required=True)
    validate.add_argument("--snapshot", type=Path, required=True)
    validate.add_argument("--metadata", type=Path, required=True)
    validate.add_argument("--catalogue", type=Path, required=True)
    validate.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "freeze-baseline":
            freeze_baseline(args.config, args.snapshot, args.as_of, args.out)
            log.info("PASS freeze-baseline -> %s", args.out)
            return 0
        if args.command == "compare-refactor":
            report = compare_refactor(args.before, args.after, args.out)
            log.info("%s compare-refactor -> %s", report["status"].upper(), args.out)
            return 0 if report["status"] == "pass" else 1
        summary = validate_candidate(args.config, args.snapshot, args.metadata, args.catalogue, args.out)
        log.info("%s validate-candidate -> %s", summary["status"].upper(), args.out)
        return 0 if summary["status"] == "pass" else 1
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        log.error("%s failed: %s", args.command, exc)
        return 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
