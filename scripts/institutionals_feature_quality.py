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

from scripts.cube_institutionals_catalogue import INSTITUTIONALS
from src.context import get_config_context
from src.data_aggregate.transformers.step_cube_institutionals import StepCubeInstitutionals
from src.data_store.schema import Table, Tables
from src.utils.config import read_config
from src.validate.checks.catalogue import _load as load_catalogue
from src.validate.checks.profile import _stats as profile_stats
from src.validate.checks.redundancy import _correlations
from src.validate.checks.timeseries import _frozen_legs, _hole_condition, _known_ineligible_reason, _measure
from src.validate.io import CHUNK_ROWS
from src.validate.result import jsonable
from src.validate.spec import load_spec
from src.validate.utils.outliers import modified_zscore

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
PROVISIONAL_PEERS = ("ic_inst_ownership_pct", "ic_shortvol_ratio_20d")
SUFFIXES = (("_vs_peers", "peer"), ("_xs", "cross_sectional"), ("_hist", "historical"))

# Explicit economics, not name heuristics. Import-time equality makes a newly catalogued
# characteristic fail loudly until somebody classifies it.
_EVENT = {
    "ic_inst_cluster_buying",
    "ic_super_new_top10",
    "ic_insider_cluster_buy_120d",
    "ic_act_initial_13d",
    "ic_act_repeat_activist",
    "ic_bo_new_holder",
    "ic_bo_escalation_13g_to_13d",
    "ic_bo_de_escalation_13d_to_13g",
    "ic_shortvol_high_x_weak_price",
    "ic_shortvol_high_x_strong_price",
}
_COUNT = {
    "ic_inst_holders",
    "ic_super_holders",
    "ic_super_top10_holders",
    "ic_super_quarters_held",
    "ic_super_initiations",
    "ic_super_full_exits",
    "ic_super_exit_after_top10",
    "ic_insider_distinct_buyers_120d",
    "ic_bo_holder_count",
}
_AGE = {"ic_sig_super_age_days", "ic_sig_insider_age_days", "ic_sig_act_age_days"}
_SUPPORT = {
    "ic_super_sp500_share",
    "ic_shortvol_market_coverage",
    "ic_xs_bullish_available_family_count",
    "ic_xs_bearish_available_family_count",
}
_BOUNDED = {
    "ic_inst_new_buyer_ratio",
    "ic_inst_exit_ratio",
    "ic_inst_concentration",
    "ic_inst_net_options_ratio",
    "ic_inst_ownership_pct",
    "ic_super_conviction_weight",
    "ic_super_max_conviction",
    "ic_insider_purchase_pct_prior",
    "ic_insider_net_buy_ratio_180d",
    "ic_ftd_persistence_30d",
}
_UNSCALED_LEVEL = {"ic_super_selection_score"}
_NORMALIZED = {
    "ic_act_amendment_intensity",
    "ic_ftd_pct_so",
    "ic_ftd_to_adv20",
    "ic_insider_buy_shares_so_180d",
    "ic_insider_buy_value_mcap_180d",
    "ic_insider_buy_value_mcap_60d",
    "ic_insider_ceo_buy_mcap_180d",
    "ic_insider_cfo_buy_mcap_180d",
    "ic_insider_director_buy_mcap_180d",
    "ic_insider_discretionary_sell_mcap_60d",
    "ic_insider_owner_surprise_120d",
    "ic_insider_planned_sell_mcap_60d",
    "ic_inst_breadth_chg",
    "ic_inst_shares_chg",
    "ic_inst_value_to_mcap",
    "ic_shortvol_acceleration",
    "ic_shortvol_ratio_20d",
    "ic_shortvol_ratio_5d",
    "ic_shortvol_ratio_60d",
    "ic_shortvol_turnover_20d",
    "ic_sig_act_resid_ret_since",
    "ic_sig_act_ret_since",
    "ic_sig_act_vol_scaled_move",
    "ic_sig_insider_max_dd_since_buy",
    "ic_sig_insider_max_runup_since_buy",
    "ic_sig_insider_price_vs_buy",
    "ic_sig_insider_resid_ret_since",
    "ic_sig_insider_ret_since",
    "ic_sig_insider_vol_scaled_move",
    "ic_sig_super_resid_ret_since",
    "ic_sig_super_ret_since",
    "ic_sig_super_vol_scaled_move",
    "ic_super_breadth_chg",
    "ic_super_conviction_chg",
    "ic_super_conviction_weight_yoy",
    "ic_super_holders_yoy",
    "ic_super_rank_jump",
    "ic_super_shares_chg",
}
CHARACTERISTIC_KIND = {
    **{name: "event" for name in _EVENT},
    **{name: "count" for name in _COUNT},
    **{name: "age" for name in _AGE},
    **{name: "support" for name in _SUPPORT},
    **{name: "bounded" for name in _BOUNDED},
    **{name: "unscaled-level" for name in _UNSCALED_LEVEL},
    **{name: "normalized" for name in _NORMALIZED},
}
if set(CHARACTERISTIC_KIND) != set(INSTITUTIONALS) or sum(map(len, (_EVENT, _COUNT, _AGE, _SUPPORT, _BOUNDED, _UNSCALED_LEVEL, _NORMALIZED))) != len(
    INSTITUTIONALS
):
    raise RuntimeError("institutionals taxonomy must explicitly classify every catalogued characteristic")

FAMILY_WARMUP_SESSIONS = {
    "broad_13f": 0,
    "elite_13f": 0,
    "insider": 126,
    "beneficial_ownership": 0,
    "short_flow": 252,
    "price_conditioning": 0,
    "cross_source": 0,
    "cross_source_control": 0,
}
ELIGIBILITY_FORMULA = "candidate_rows__first_supported_after_builder_warmup__last_supported_complete_through_v1"


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


def _load_manifest(path: Path, as_of: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest is not a JSON object: {path}")
    if payload.get("table") != TABLE.name:
        raise ValueError(f"manifest table is {payload.get('table')!r}, expected {TABLE.name!r}")
    if pd.Timestamp(payload.get("as_of")).normalize() != pd.Timestamp(as_of).normalize():
        raise ValueError(f"manifest as_of={payload.get('as_of')!r} does not match {as_of!r}")
    snapshot = payload.get("snapshot") or {}
    frozen = Path(str(snapshot.get("frozen", "")))
    if not frozen.is_file() or not snapshot.get("sha256"):
        raise ValueError("manifest must name an existing hash-bound frozen baseline")
    if _sha256(frozen) != snapshot["sha256"]:
        raise ValueError("manifest baseline hash does not match its frozen parquet")
    return payload


def _verify_source_identity(store: Any, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    recorded = {row.get("table"): row for row in manifest.get("source_tables") or []}
    expected = {table.name for table in SOURCE_TABLES}
    if set(recorded) != expected:
        raise ValueError(f"manifest source tables differ: missing={sorted(expected - set(recorded))}, extra={sorted(set(recorded) - expected)}")
    current = [_source_metadata(store, table) for table in SOURCE_TABLES]
    changed = [row["table"] for row in current if jsonable(row) != recorded[row["table"]]]
    if changed:
        raise ValueError(f"source metadata changed since baseline freeze: {changed}")
    return current


def _read_projected(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    parts = [batch.to_pandas() for batch in parquet.iter_batches(batch_size=CHUNK_ROWS, columns=list(columns))]
    return pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0] if parts else pd.DataFrame(columns=list(columns))


def _feature_parts(column: str) -> tuple[str, str]:
    body = column.removeprefix("f_")
    suffix = "raw"
    characteristic = body
    for ending, label in SUFFIXES:
        if body.endswith(ending):
            suffix, characteristic = label, body.removesuffix(ending)
            break
    if characteristic not in INSTITUTIONALS or characteristic not in CHARACTERISTIC_KIND:
        raise ValueError(f"no explicit institutionals taxonomy for {column!r} (characteristic {characteristic!r})")
    return characteristic, suffix


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


def build_candidate(
    config: str,
    manifest_path: Path,
    as_of: str,
    out_cache: Path,
    out: Path,
    *,
    compare_to: Path | None = None,
    comparison_out: Path | None = None,
) -> dict[str, Any]:
    cutoff = pd.Timestamp(as_of).normalize()
    manifest = _load_manifest(manifest_path, as_of)
    config_node, context = get_config_context(config, use_cache=False, save=False)
    sources = _verify_source_identity(context.store, manifest)
    panel, window = StepCubeInstitutionals(context=context, config=config_node).build_panel(full=True)
    if not window.is_full:
        raise ValueError("build_panel(full=True) returned a non-full PartWindow")
    if not set(KEYS).issubset(panel.columns):
        raise ValueError(f"candidate panel is missing keys {sorted(set(KEYS) - set(panel.columns))}")
    panel = panel.copy()
    panel[TABLE.date_col] = pd.to_datetime(panel[TABLE.date_col], errors="coerce").dt.normalize()
    if panel[KEYS].isna().any(axis=1).any():
        raise ValueError("untrimmed candidate panel contains null keys")
    panel = panel.loc[panel[TABLE.date_col] <= cutoff].reset_index(drop=True)
    null_keys = int(panel[KEYS].isna().any(axis=1).sum())
    duplicate_keys = int(panel.duplicated(KEYS, keep=False).sum())
    if panel.empty or null_keys or duplicate_keys:
        raise ValueError(f"candidate key assertion failed: rows={len(panel)}, null_key_rows={null_keys}, duplicate_key_rows={duplicate_keys}")
    if panel[TABLE.date_col].max() > cutoff:
        raise ValueError("candidate contains rows after its declared cutoff")

    out_cache.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_cache, index=False)
    parquet, columns, dtypes = _schema(out_cache)
    first_date, last_date, invalid_dates = _date_bounds(out_cache)
    if invalid_dates:
        raise ValueError(f"written candidate contains {invalid_dates} invalid dates")
    metadata = {
        "table": TABLE.name,
        "as_of": cutoff,
        "rows": int(parquet.metadata.num_rows),
        "columns": columns,
        "dtypes": dtypes,
        "pk": KEYS,
        "date_col": TABLE.date_col,
        "first_date": first_date,
        "last_date": last_date,
        "snapshot": str(out_cache.resolve()),
        "snapshot_sha256": _sha256(out_cache),
        "snapshot_bytes": out_cache.stat().st_size,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "source_tables": sources,
        "builder": "StepCubeInstitutionals.build_panel(full=True)",
        "write_contract": "parquet artifact only; StepCubeInstitutionals.run/write_part not called",
    }
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "candidate-metadata.json", metadata)
    comparison: dict[str, Any] | None = None
    if compare_to is not None:
        if comparison_out is None:
            raise ValueError("--comparison-out is required with --compare-to")
        comparison = compare_refactor(compare_to, out_cache, comparison_out)
    return {"status": "fail" if comparison and comparison["status"] == "fail" else "pass", "metadata": metadata, "comparison": comparison}


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


def _requested_decision(characteristic: str, suffix: str) -> str:
    decisions = {
        "raw": "keep_raw",
        "historical": "keep_historical_normalization",
        "cross_sectional": "remove_cross_sectional_normalization",
    }
    if suffix == "peer":
        return "provisional_peer_pending_diagnostics" if characteristic in PROVISIONAL_PEERS else "remove_peer_normalization"
    if suffix not in decisions:
        raise ValueError(f"no explicit decision for suffix {suffix!r}")
    return decisions[suffix]


def _peer_slice(frame: pd.DataFrame, raw: str, peer: str) -> dict[str, Any]:
    raw_values = pd.to_numeric(frame[raw], errors="coerce")
    peer_values = pd.to_numeric(frame[peer], errors="coerce")
    overlap = np.isfinite(raw_values) & np.isfinite(peer_values)
    raw_support = np.isfinite(raw_values)
    correlation = float(raw_values[overlap].corr(peer_values[overlap])) if overlap.sum() >= 2 else None
    concentrations: list[float] = []
    maxima: list[float] = []
    for _, group in frame.loc[overlap].assign(_peer=peer_values[overlap]).groupby("date", sort=False):
        absolute = group["_peer"].abs().to_numpy(dtype="float64")
        total = float(absolute.sum())
        if total > 0:
            weights = absolute / total
            concentrations.append(float(np.square(weights).sum()))
            maxima.append(float(weights.max()))
    return {
        "rows": len(frame),
        "raw_supported_rows": int(raw_support.sum()),
        "same_date_overlap_rows": int(overlap.sum()),
        "same_date_overlap_share": float(overlap.sum() / raw_support.sum()) if raw_support.sum() else None,
        "raw_peer_correlation": correlation,
        "abs_score_weight_hhi_proxy": float(np.mean(concentrations)) if concentrations else None,
        "abs_score_max_weight_proxy": float(np.mean(maxima)) if maxima else None,
    }


def _peer_diagnostic(snapshot: Path, characteristic: str, columns: list[str]) -> dict[str, Any]:
    raw, peer = f"f_{characteristic}", f"f_{characteristic}_vs_peers"
    if raw not in columns or peer not in columns:
        return {
            "characteristic": characteristic,
            "status": "abstain",
            "reason": f"candidate is missing {raw if raw not in columns else peer}",
        }
    frame = _read_projected(snapshot, [*KEYS, raw, peer])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.sort_values(KEYS, kind="stable")
    raw_change = pd.to_numeric(frame[raw], errors="coerce").groupby(frame["ticker"]).diff()
    peer_change = pd.to_numeric(frame[peer], errors="coerce").groupby(frame["ticker"]).diff()
    unchanged_jump = raw_change.eq(0) & peer_change.notna() & peer_change.ne(0)
    unique_dates = pd.Index(sorted(frame["date"].dropna().unique()))
    midpoint = unique_dates[len(unique_dates) // 2] if len(unique_dates) else pd.NaT
    early = _peer_slice(frame.loc[frame["date"] < midpoint], raw, peer) if pd.notna(midpoint) else {}
    late = _peer_slice(frame.loc[frame["date"] >= midpoint], raw, peer) if pd.notna(midpoint) else {}
    whole = _peer_slice(frame, raw, peer)
    return {
        "characteristic": characteristic,
        "status": "pass",
        "reason": "",
        **whole,
        "unchanged_subject_score_jumps": int(unchanged_jump.sum()),
        "unchanged_subject_score_jump_share": float(unchanged_jump.mean()) if len(frame) else None,
        "unchanged_subject_score_jump_p99": float(peer_change[unchanged_jump].abs().quantile(0.99)) if unchanged_jump.any() else None,
        "chronological_midpoint": midpoint,
        "early_overlap_share": early.get("same_date_overlap_share"),
        "late_overlap_share": late.get("same_date_overlap_share"),
        "early_correlation": early.get("raw_peer_correlation"),
        "late_correlation": late.get("raw_peer_correlation"),
        "early_hhi_proxy": early.get("abs_score_weight_hhi_proxy"),
        "late_hhi_proxy": late.get("abs_score_weight_hhi_proxy"),
    }


def _model_folds(snapshot: Path, features: list[str]) -> pd.DataFrame:
    dates = pd.DatetimeIndex(sorted(_read_projected(snapshot, [TABLE.date_col])[TABLE.date_col].dropna().unique()))
    groups = [pd.DatetimeIndex(group) for group in np.array_split(dates, min(5, len(dates))) if len(group)]
    if not groups:
        return pd.DataFrame()
    ends = pd.DatetimeIndex([group[-1] for group in groups])
    state = [
        {"fold": index + 1, "start": group[0], "end": group[-1], "rows": 0, "tickers": set(), "nonnull": 0, "cells": 0}
        for index, group in enumerate(groups)
    ]
    parquet = pq.ParquetFile(snapshot)
    wanted = [*KEYS, *features]
    for batch in parquet.iter_batches(batch_size=CHUNK_ROWS, columns=wanted):
        frame = batch.to_pandas()
        stamps = pd.to_datetime(frame[TABLE.date_col], errors="coerce")
        fold_ids = np.searchsorted(ends.to_numpy(), stamps.to_numpy(), side="left")
        for index, entry in enumerate(state):
            mask = fold_ids == index
            if not mask.any():
                continue
            part = frame.loc[mask]
            entry["rows"] += len(part)
            entry["tickers"].update(map(str, part["ticker"].dropna().unique()))
            entry["nonnull"] += int(part[features].notna().sum().sum()) if features else 0
            entry["cells"] += len(part) * len(features)
    return pd.DataFrame(
        [
            {
                "fold": entry["fold"],
                "start": entry["start"],
                "end": entry["end"],
                "rows": entry["rows"],
                "tickers": len(entry["tickers"]),
                "feature_cells": entry["cells"],
                "feature_nonnull_share": entry["nonnull"] / entry["cells"] if entry["cells"] else None,
            }
            for entry in state
        ]
    )


def taxonomy(config: str, snapshot: Path, manifest_path: Path, as_of: str, out: Path) -> dict[str, Any]:
    read_config(config)
    manifest = _load_manifest(manifest_path, as_of)
    _, candidate_columns, candidate_dtypes = _schema(snapshot)
    first_date, last_date, invalid_dates = _date_bounds(snapshot)
    if invalid_dates or (last_date is not None and last_date.normalize() > pd.Timestamp(as_of).normalize()):
        raise ValueError("candidate dates are invalid or beyond the taxonomy cutoff")
    baseline = manifest.get("snapshot") or {}
    baseline_columns = list(baseline.get("columns") or [])
    baseline_dtypes = dict(baseline.get("dtypes") or {})
    if not set(KEYS).issubset(baseline_columns):
        raise ValueError("manifest baseline schema is missing panel keys")

    decisions: list[dict[str, Any]] = []
    candidate_set = set(candidate_columns)
    for column in baseline_columns:
        if column in KEYS:
            continue
        characteristic, suffix = _feature_parts(column)
        replacement = f"f_{characteristic}"
        decisions.append(
            {
                "baseline_column": column,
                "characteristic": characteristic,
                "suffix": suffix,
                "family": INSTITUTIONALS[characteristic][0],
                "kind": CHARACTERISTIC_KIND[characteristic],
                "requested_decision": _requested_decision(characteristic, suffix),
                "decision_rule_id": "raw_plus_interpretable_history__peer_only_if_diagnostics_survive_v1",
                "final_presence": column in candidate_set,
                "exact_final_presence": column in candidate_set,
                "retained_raw_presence": replacement in candidate_set,
                "final_column": column if column in candidate_set else replacement if replacement in candidate_set else None,
            }
        )
    decision_frame = pd.DataFrame(decisions)
    if len(decision_frame) != len(baseline_columns) - len(KEYS):
        raise ValueError("baseline feature reconciliation is incomplete")

    schema_diff = {
        "table": TABLE.name,
        "as_of": as_of,
        "baseline_columns": baseline_columns,
        "candidate_columns": candidate_columns,
        "added": sorted(candidate_set - set(baseline_columns)),
        "removed": sorted(set(baseline_columns) - candidate_set),
        "column_order_equal": baseline_columns == candidate_columns,
        "dtype_changes": {
            column: {"baseline": baseline_dtypes[column], "candidate": candidate_dtypes[column]}
            for column in set(baseline_dtypes) & set(candidate_dtypes)
            if baseline_dtypes[column] != candidate_dtypes[column]
        },
        "snapshot_sha256": _sha256(snapshot),
        "manifest_sha256": _sha256(manifest_path),
        "first_date": first_date,
        "last_date": last_date,
        "reconciled_baseline_features": len(decision_frame),
    }
    peer_frame = pd.DataFrame([_peer_diagnostic(snapshot, characteristic, candidate_columns) for characteristic in PROVISIONAL_PEERS])
    fold_frame = _model_folds(snapshot, [column for column in candidate_columns if column not in KEYS])
    out.mkdir(parents=True, exist_ok=True)
    decision_frame.to_csv(out / "feature-decisions.csv", index=False)
    _write_json(out / "schema-diff.json", schema_diff)
    peer_frame.to_csv(out / "peer-diagnostics.csv", index=False)
    fold_frame.to_csv(out / "model-fold-diagnostics.csv", index=False)
    return {"status": "pass", "features": len(decision_frame), "peer_rows": len(peer_frame), "fold_rows": len(fold_frame)}


_COVERAGE_BUCKETS = ("100%", "70%-<100%", "50%-<70%", "30%-<50%", "<=30%", "no-support")


def _coverage_bucket(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "no-support"
    share = numerator / denominator
    if np.isclose(share, 1.0):
        return "100%"
    if share >= 0.7:
        return "70%-<100%"
    if share >= 0.5:
        return "50%-<70%"
    if share > 0.3:
        return "30%-<50%"
    return "<=30%"


def _analysis_frames(
    snapshot: Path, keys: pd.DataFrame, features: list[str], recent_sessions: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    dates = pd.DatetimeIndex(sorted(keys[TABLE.date_col].dropna().unique()))
    recent_dates = set(dates[-min(recent_sessions, len(dates)) :])
    recent_mask = keys[TABLE.date_col].isin(recent_dates).to_numpy()
    ticker_positions = {str(ticker): np.asarray(positions, dtype="int64") for ticker, positions in keys.groupby("ticker", sort=False).indices.items()}
    coverage_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    outlier_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    all_stats: dict[str, dict[str, Any]] = {}
    for index, feature in enumerate(features, start=1):
        characteristic, _ = _feature_parts(feature)
        family = INSTITUTIONALS[characteristic][0]
        values = pd.to_numeric(_read_projected(snapshot, [feature])[feature], errors="coerce")
        array = values.to_numpy(dtype="float64")
        present = np.isfinite(array)
        stats = profile_stats(values, None)
        all_stats[feature] = stats
        distribution_rows.append({"feature": feature, "characteristic": characteristic, "family": family, **stats})

        scores = modified_zscore(array)
        extreme_positions = np.flatnonzero(present & (scores > 3.5))
        for position in extreme_positions[np.argsort(scores[extreme_positions])[-10:][::-1]]:
            outlier_rows.append(
                {
                    "feature": feature,
                    "ticker": keys.iloc[position]["ticker"],
                    "date": keys.iloc[position][TABLE.date_col],
                    "value": float(array[position]),
                    "modified_z": float(scores[position]),
                }
            )

        for ticker, positions in ticker_positions.items():
            supported = positions[present[positions]]
            if not len(supported):
                coverage_rows.append(
                    {
                        "feature": feature,
                        "characteristic": characteristic,
                        "family": family,
                        "ticker": ticker,
                        "eligibility_formula_id": ELIGIBILITY_FORMULA,
                        "family_warmup_sessions": FAMILY_WARMUP_SESSIONS[family],
                        "observed_warmup_candidate_rows": None,
                        "first_support": None,
                        "complete_through": None,
                        "full_numerator": 0,
                        "full_eligible_denominator": 0,
                        "full_share": None,
                        "full_bucket": "no-support",
                        "recent_numerator": 0,
                        "recent_eligible_denominator": 0,
                        "recent_share": None,
                        "recent_bucket": "no-support",
                    }
                )
                continue
            first_support = keys.iloc[supported[0]][TABLE.date_col]
            complete_through = keys.iloc[supported[-1]][TABLE.date_col]
            group_dates = keys.iloc[positions][TABLE.date_col]
            eligible = positions[(group_dates >= first_support).to_numpy() & (group_dates <= complete_through).to_numpy()]
            recent_eligible = eligible[recent_mask[eligible]]
            full_numerator = int(present[eligible].sum())
            recent_numerator = int(present[recent_eligible].sum())
            full_denominator, recent_denominator = len(eligible), len(recent_eligible)
            coverage_rows.append(
                {
                    "feature": feature,
                    "characteristic": characteristic,
                    "family": family,
                    "ticker": ticker,
                    "eligibility_formula_id": ELIGIBILITY_FORMULA,
                    "family_warmup_sessions": FAMILY_WARMUP_SESSIONS[family],
                    "observed_warmup_candidate_rows": int(np.flatnonzero(positions == supported[0])[0]),
                    "first_support": first_support,
                    "complete_through": complete_through,
                    "full_numerator": full_numerator,
                    "full_eligible_denominator": full_denominator,
                    "full_share": full_numerator / full_denominator if full_denominator else None,
                    "full_bucket": _coverage_bucket(full_numerator, full_denominator),
                    "recent_numerator": recent_numerator,
                    "recent_eligible_denominator": recent_denominator,
                    "recent_share": recent_numerator / recent_denominator if recent_denominator else None,
                    "recent_bucket": _coverage_bucket(recent_numerator, recent_denominator),
                }
            )

        history_values = array[~recent_mask]
        recent_values = array[recent_mask]
        history_ok = history_values[np.isfinite(history_values)]
        recent_ok = recent_values[np.isfinite(recent_values)]
        history_sd = float(history_ok.std(ddof=1)) if len(history_ok) > 1 else 0.0
        history_mean = float(history_ok.mean()) if len(history_ok) else None
        recent_mean = float(recent_ok.mean()) if len(recent_ok) else None
        drift_rows.append(
            {
                "feature": feature,
                "history_n": len(history_ok),
                "recent_n": len(recent_ok),
                "history_mean": history_mean,
                "recent_mean": recent_mean,
                "history_null_share": float(1 - len(history_ok) / len(history_values)) if len(history_values) else None,
                "recent_null_share": float(1 - len(recent_ok) / len(recent_values)) if len(recent_values) else None,
                "mean_shift_history_sd": (
                    (recent_mean - history_mean) / history_sd if history_mean is not None and recent_mean is not None and history_sd > 0 else None
                ),
            }
        )
        for split, sample in (("history", history_values), ("recent", recent_values)):
            finite = sample[np.isfinite(sample)]
            split_rows.append(
                {
                    "feature": feature,
                    "split": split,
                    "rows": len(sample),
                    "n_finite": len(finite),
                    "null_share": float(1 - len(finite) / len(sample)) if len(sample) else None,
                    "mean": float(finite.mean()) if len(finite) else None,
                    "p50": float(np.median(finite)) if len(finite) else None,
                    "min": float(finite.min()) if len(finite) else None,
                    "max": float(finite.max()) if len(finite) else None,
                }
            )
        if index % 20 == 0 or index == len(features):
            log.info("analyze candidate: %d/%d columns", index, len(features))
    return (
        pd.DataFrame(coverage_rows),
        pd.DataFrame(distribution_rows),
        pd.DataFrame(outlier_rows, columns=["feature", "ticker", "date", "value", "modified_z"]),
        pd.DataFrame(drift_rows),
        pd.DataFrame(split_rows),
        all_stats,
    )


def _coverage_summary(coverage: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature, group in coverage.groupby("feature", sort=False):
        for horizon in ("full", "recent"):
            counts = group[f"{horizon}_bucket"].value_counts()
            for bucket in _COVERAGE_BUCKETS:
                rows.append({"feature": feature, "horizon": horizon, "bucket": bucket, "tickers": int(counts.get(bucket, 0))})
    return pd.DataFrame(rows)


def analyze(config: str, table: str, snapshot: Path, manifest_path: Path, as_of: str, recent_sessions: int, out: Path) -> dict[str, Any]:
    if table != TABLE.name:
        raise ValueError(f"this driver analyzes {TABLE.name}, not {table!r}")
    if recent_sessions <= 0:
        raise ValueError("--recent-sessions must be positive")
    manifest = _load_manifest(manifest_path, as_of)
    config_node = read_config(config)
    spec = load_spec(config_node, TABLE)
    _, columns, _ = _schema(snapshot)
    if not set(KEYS).issubset(columns):
        raise ValueError("candidate snapshot is missing panel keys")
    keys = _read_projected(snapshot, KEYS)
    keys[TABLE.date_col] = pd.to_datetime(keys[TABLE.date_col], errors="coerce").dt.normalize()
    if keys[KEYS].isna().any(axis=1).any() or keys.duplicated(KEYS).any():
        raise ValueError("candidate keys are null or duplicated")
    if keys[TABLE.date_col].max() > pd.Timestamp(as_of).normalize():
        raise ValueError("candidate contains dates after --as-of")
    features = _numeric_columns(snapshot)
    if not features:
        raise ValueError("candidate has no numeric institutional feature")

    coverage, distributions, outliers, drift, splits, stats = _analysis_frames(snapshot, keys, features, recent_sessions)
    coverage_summary = _coverage_summary(coverage)
    redundancy = _redundancy(snapshot, stats, spec.redundancy_r)
    leakage = _leakage(keys, {"as_of": manifest["as_of"]})
    out.mkdir(parents=True, exist_ok=True)
    frames = {
        "coverage.csv": coverage,
        "coverage-summary.csv": coverage_summary,
        "distributions.csv": distributions,
        "outliers.csv": outliers,
        "drift.csv": drift,
        "chronological-split-diagnostics.csv": splits,
    }
    for name, frame in frames.items():
        frame.to_csv(out / name, index=False)
    _write_json(out / "redundancy.json", redundancy)
    _write_json(out / "leakage.json", leakage)

    expected_rows = {name: len(frame) for name, frame in frames.items()}
    actual_rows = {name: len(pd.read_csv(out / name)) for name in frames}
    expected_artifacts = [*frames, "redundancy.json", "leakage.json"]
    missing = [name for name in expected_artifacts if not (out / name).is_file()]
    reconciliation = {
        "pass": not missing and expected_rows == actual_rows and len(coverage) == len(features) * keys["ticker"].nunique(),
        "expected_artifacts": expected_artifacts,
        "missing_artifacts": missing,
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
        "coverage_expected_rows": len(features) * int(keys["ticker"].nunique()),
        "coverage_actual_rows": len(coverage),
    }
    summary = {
        "status": "pass" if reconciliation["pass"] else "fail",
        "table": TABLE.name,
        "as_of": as_of,
        "recent_sessions": recent_sessions,
        "snapshot": str(snapshot.resolve()),
        "snapshot_sha256": _sha256(snapshot),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "rows": len(keys),
        "tickers": int(keys["ticker"].nunique()),
        "features": len(features),
        "eligibility": {
            "formula_id": ELIGIBILITY_FORMULA,
            "candidate_rows_only": True,
            "first_support": "first finite value per feature and ticker after the builder's own warmup",
            "observed_warmup": "candidate trading rows before first finite support, recorded per feature and ticker",
            "complete_through": "last finite supported candidate row per feature and ticker",
            "absent_aggregate_rows_eligible": False,
            "family_warmup_sessions": FAMILY_WARMUP_SESSIONS,
        },
        "leakage_status": leakage["status"],
        "artifact_reconciliation": reconciliation,
    }
    _write_json(out / "analysis-summary.json", summary)
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

    build = commands.add_parser("build-candidate")
    build.add_argument("--config", required=True)
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--as-of", required=True)
    build.add_argument("--out-cache", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--compare-to", type=Path)
    build.add_argument("--comparison-out", type=Path)

    classify = commands.add_parser("taxonomy")
    classify.add_argument("--config", required=True)
    classify.add_argument("--snapshot", type=Path, required=True)
    classify.add_argument("--manifest", type=Path, required=True)
    classify.add_argument("--as-of", required=True)
    classify.add_argument("--out", type=Path, required=True)

    analysis = commands.add_parser("analyze")
    analysis.add_argument("--config", required=True)
    analysis.add_argument("--table", required=True)
    analysis.add_argument("--snapshot", type=Path, required=True)
    analysis.add_argument("--manifest", type=Path, required=True)
    analysis.add_argument("--as-of", required=True)
    analysis.add_argument("--recent-sessions", type=int, required=True)
    analysis.add_argument("--out", type=Path, required=True)
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
        if args.command == "validate-candidate":
            summary = validate_candidate(args.config, args.snapshot, args.metadata, args.catalogue, args.out)
        elif args.command == "build-candidate":
            summary = build_candidate(
                args.config,
                args.manifest,
                args.as_of,
                args.out_cache,
                args.out,
                compare_to=args.compare_to,
                comparison_out=args.comparison_out,
            )
        elif args.command == "taxonomy":
            summary = taxonomy(args.config, args.snapshot, args.manifest, args.as_of, args.out)
        else:
            summary = analyze(args.config, args.table, args.snapshot, args.manifest, args.as_of, args.recent_sessions, args.out)
        log.info("%s %s -> %s", summary["status"].upper(), args.command, args.out)
        return 0 if summary["status"] == "pass" else 1
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        log.error("%s failed: %s", args.command, exc)
        return 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
