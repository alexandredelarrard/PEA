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
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Project imports intentionally follow the repository-root path bootstrap.
# ruff: noqa: E402

from scripts.cube_institutionals_catalogue import INSTITUTIONALS
from src.context import get_config_context
from src.data_aggregate.transformers.step_cube_institutionals import StepCubeInstitutionals
from src.data_aggregate.utils.institutionals.availability import InstitutionalAvailability
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
PEER_CANDIDATES = ("ic_inst_ownership_pct", "ic_shortvol_ratio_20d")
PROVISIONAL_PEERS: tuple[str, ...] = ()
SUFFIXES = (("_vs_peers", "peer"), ("_xs", "cross_sectional"), ("_hist", "historical"))
REMOVED_CHARACTERISTICS: dict[str, tuple[str, str]] = {
    "ic_inst_flow_to_mcap": ("broad_13f", "reported-flow proxy duplicated the economically retained holdings changes"),
    "ic_super_flow_to_mcap": ("elite_13f", "reported-flow proxy duplicated the economically retained conviction changes"),
    "ic_super_exit_after_top10": ("elite_13f", "0.9965 correlated with full exits and supported on fewer issuers"),
    "ic_insider_buy_shares_so_180d": (
        "insider",
        "0.9998 rank-correlated with the model-consumed buy-value-to-market-cap feature",
    ),
    "ic_shortvol_ratio_z252": ("short_flow", "internal rolling z-score is not an economically interpretable output"),
    "ic_ftd_z252": ("short_flow", "internal rolling z-score is retained only as a feature input"),
    "ic_ftd_pct_so": ("short_flow", "0.9947 rank-correlated with the model-consumed fails-to-ADV20 feature"),
    "ic_act_campaign_age_days": ("beneficial_ownership", "sparse open-ended age is not comparable across issuers"),
    "ic_bo_holder_count": (
        "beneficial_ownership",
        "misleading count name hid a filing-activity ratio without a stable source-complete denominator",
    ),
    "ic_act_purpose_board": ("beneficial_ownership", "fragile text-derived purpose flag"),
    "ic_act_purpose_strategic": ("beneficial_ownership", "fragile text-derived purpose flag"),
    "ic_xs_bullish_family_ratio": ("cross_source", "opaque normalization of raw available-family support"),
    "ic_xs_bearish_family_ratio": ("cross_source", "opaque normalization of raw available-family support"),
    "ic_xs_conflict_ratio": ("cross_source", "opaque normalization of raw available-family support"),
    "ic_xs_bullish_actor_count": ("cross_source", "redundant actor aggregation without stable economic scale"),
}

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
    "ic_insider_distinct_buyers_120d",
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
    "ic_ftd_to_adv20",
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
ELIGIBILITY_FORMULA = "configured_source_onset__ticker_price_rows__builder_warmup__candidate_end_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _peer_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"peer cache does not exist: {path}")
    return {
        "frozen": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _manifest_peer_path(manifest: dict[str, Any]) -> Path:
    recorded = manifest.get("peer_cache") or {}
    path = Path(str(recorded.get("frozen", "")))
    if not path.is_file() or not recorded.get("sha256"):
        raise ValueError("manifest must name an existing hash-bound frozen peer cache")
    if _sha256(path) != recorded["sha256"] or path.stat().st_size != int(recorded.get("bytes", -1)):
        raise ValueError("manifest peer cache identity does not match its frozen artifact")
    return path


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
    _manifest_peer_path(payload)
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


def _feature_parts(column: str, *, allow_removed: bool = False) -> tuple[str, str]:
    body = column.removeprefix("f_")
    suffix = "raw"
    characteristic = body
    for ending, label in SUFFIXES:
        if body.endswith(ending):
            suffix, characteristic = label, body.removesuffix(ending)
            break
    known = characteristic in INSTITUTIONALS and characteristic in CHARACTERISTIC_KIND
    if not known and not (allow_removed and characteristic in REMOVED_CHARACTERISTICS):
        raise ValueError(f"no explicit institutionals taxonomy for {column!r} (characteristic {characteristic!r})")
    return characteristic, suffix


def freeze_baseline(config: str, snapshot: Path, as_of: str, out: Path) -> dict[str, Any]:
    meta_path = snapshot.parent / "meta.json"
    if not meta_path.is_file() and snapshot.name == "baseline.parquet":
        meta_path = snapshot.parent / "baseline-meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"snapshot metadata does not exist: {meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"snapshot metadata is not a JSON object: {meta_path}")
    cutoff = pd.Timestamp(as_of).normalize()
    verified = _verified_snapshot(snapshot, metadata, as_of=cutoff)

    _, context = get_config_context(config, use_cache=False, save=False)
    out.mkdir(parents=True, exist_ok=True)
    frozen_path = out / "baseline.parquet"
    if not frozen_path.exists():
        shutil.copy2(snapshot, frozen_path)
    peer_source = Path(context.paths["SECTOR_PEERS_PATH"])
    frozen_peer = out / "peer-baskets.json"
    if not frozen_peer.exists():
        shutil.copy2(peer_source, frozen_peer)
    elif _sha256(peer_source) != _sha256(frozen_peer):
        raise ValueError("live peer cache changed after the baseline peer artifact was frozen")
    sources = [_source_metadata(context.store, table) for table in SOURCE_TABLES]
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
        "peer_cache": {"source": str(peer_source.resolve()), **_peer_metadata(frozen_peer)},
        "source_contract": "DataStore exists/columns/row_count/bounds metadata only plus immutable peer-cache bytes; no table values were read",
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
    peer_path = _manifest_peer_path(manifest)
    context.paths["SECTOR_PEERS_PATH"] = peer_path
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
        "peer_cache": manifest["peer_cache"],
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


def _redundancy_frame(path: Path, stats: dict[str, dict[str, Any]], threshold: float) -> tuple[dict[str, Any], pd.DataFrame]:
    columns = list(stats)
    if len(columns) < 2:
        result = _result("redundancy", "abstain", scope={"columns": len(columns)}, metrics={}, reason="fewer than two numeric feature columns")
        return result, pd.DataFrame()
    width = len(columns)
    centre = np.array([float(stats[column]["mean"] or 0.0) for column in columns], dtype="float64")
    acc = {name: np.zeros((width, width), dtype="float64") for name in ("n", "sx", "sxx", "sxy", "eq")}
    rows = 0
    parquet = pq.ParquetFile(path)
    stride = max(1, parquet.metadata.num_rows // 50_000)
    samples: list[np.ndarray] = []
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
        take = (rows + np.arange(len(raw))) % stride == 0
        if take.any():
            samples.append(raw[take])
        rows += len(raw)
    acc["rows"] = np.array([rows], dtype="float64")
    correlation, overlap = _correlations(acc)
    sample = np.concatenate(samples)[:50_000] if samples else np.empty((0, width))
    spearman = pd.DataFrame(sample, columns=columns).corr(method="spearman", min_periods=min(100, len(sample))).to_numpy()
    counts = np.diag(overlap)
    min_pair = min(10_000, rows)
    all_pairs: list[dict[str, Any]] = []
    for left in range(width):
        for right in range(left + 1, width):
            n = int(overlap[left, right])
            union = int(counts[left] + counts[right] - n)
            same_mask = int(rows - counts[left] - counts[right] + 2 * n)
            pearson_value = correlation[left, right]
            spearman_value = spearman[left, right]
            exact = n >= max(2, min_pair) and same_mask == rows and int(acc["eq"][left, right]) == n
            high = n >= min_pair and (
                (np.isfinite(pearson_value) and abs(pearson_value) >= threshold) or (np.isfinite(spearman_value) and abs(spearman_value) >= threshold)
            )
            all_pairs.append(
                {
                    "left": columns[left],
                    "right": columns[right],
                    "overlap_n": n,
                    "overlap_ratio": n / union if union else None,
                    "identical_null_mask_rate": same_mask / rows if rows else None,
                    "pearson": None if not np.isfinite(pearson_value) else float(pearson_value),
                    "spearman_sample": None if not np.isfinite(spearman_value) else float(spearman_value),
                    "spearman_sample_n": len(sample),
                    "exact": exact,
                    "flagged": exact or high,
                    "disposition": "review" if exact or high else "keep",
                }
            )
    all_pairs.sort(key=lambda item: (not item["flagged"], -(abs(item["pearson"]) if item["pearson"] is not None else 0.0)))
    flagged = [item for item in all_pairs if item["flagged"]]
    result = _result(
        "redundancy",
        "fail" if flagged else "pass",
        scope={"rows": rows, "columns": width, "threshold": threshold, "min_pairwise_n": min_pair, "spearman_sample_n": len(sample)},
        metrics={"redundant_pairs": flagged[:40], "n_redundant_pairs": len(flagged)},
    )
    return result, pd.DataFrame(all_pairs)


def _redundancy(path: Path, stats: dict[str, dict[str, Any]], threshold: float) -> dict[str, Any]:
    return _redundancy_frame(path, stats, threshold)[0]


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
    holes_by_field = pd.Series([row["field"] for row in holes], dtype="string").value_counts().to_dict()
    explained_by_field = pd.Series([row["field"] for row in explained], dtype="string").value_counts().to_dict()
    frozen_by_field = pd.Series([row["field"] for row in stalled], dtype="string").value_counts().to_dict()
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
            "holes_by_field": holes_by_field,
            "explained_holes_by_field": explained_by_field,
            "frozen_by_field": frozen_by_field,
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
            "status": "removed" if raw in columns else "abstain",
            "reason": (
                "peer leg removed because the frozen feature-only candidate cannot supply the approved target/OOS retention evidence"
                if raw in columns
                else f"candidate is missing {raw}"
            ),
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
        "status": "abstain",
        "reason": "candidate contains no outcome target; support diagnostics alone cannot pass the OOS retention gate",
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
                "status": "abstain",
                "target_status": "no_target_in_candidate_snapshot",
                "diagnostic_scope": "feature_support_only",
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
    for column in candidate_columns:
        if column not in KEYS:
            _feature_parts(column)
    for column in baseline_columns:
        if column in KEYS:
            continue
        characteristic, suffix = _feature_parts(column, allow_removed=True)
        removed = REMOVED_CHARACTERISTICS.get(characteristic)
        replacement = None if removed else f"f_{characteristic}"
        decisions.append(
            {
                "baseline_column": column,
                "characteristic": characteristic,
                "suffix": suffix,
                "family": removed[0] if removed else INSTITUTIONALS[characteristic][0],
                "kind": "removed" if removed else CHARACTERISTIC_KIND[characteristic],
                "requested_decision": "remove_characteristic" if removed else _requested_decision(characteristic, suffix),
                "removal_reason": removed[1] if removed else None,
                "decision_rule_id": "raw_plus_interpretable_history__peer_only_if_diagnostics_survive_v1",
                "final_presence": column in candidate_set,
                "exact_final_presence": column in candidate_set,
                "retained_raw_presence": replacement in candidate_set if replacement else False,
                "final_column": column if column in candidate_set else replacement if replacement and replacement in candidate_set else None,
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
    peer_frame = pd.DataFrame([_peer_diagnostic(snapshot, characteristic, candidate_columns) for characteristic in PEER_CANDIDATES])
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


def _sample_even(values: np.ndarray, limit: int = 50_000) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if len(finite) <= limit:
        return finite
    return finite[np.linspace(0, len(finite) - 1, limit, dtype="int64")]


def _period_metrics(values: np.ndarray, mask: np.ndarray, dates: pd.Series) -> dict[str, Any]:
    sample = values[mask]
    finite = sample[np.isfinite(sample)]
    date_values = dates[mask].reset_index(drop=True)
    daily_variance = pd.Series(sample).groupby(date_values).var().dropna()
    q25, median, q75 = np.quantile(finite, [0.25, 0.5, 0.75]) if len(finite) else (np.nan, np.nan, np.nan)
    return {
        "rows": len(sample),
        "n_finite": len(finite),
        "nonnull_share": len(finite) / len(sample) if len(sample) else None,
        "zero_share": float(np.mean(finite == 0.0)) if len(finite) else None,
        "median": None if not np.isfinite(median) else float(median),
        "iqr": None if not np.isfinite(q75 - q25) else float(q75 - q25),
        "cross_sectional_variance": float(daily_variance.median()) if len(daily_variance) else None,
        "finite": finite,
    }


def _known_ineligible_mask(spec: Any, ticker: str, feature: str, dates: pd.Series) -> np.ndarray:
    eligible = np.ones(len(dates), dtype=bool)
    for rule in spec.known_ineligible:
        if ticker != rule["ticker"] or not any(feature == field or feature.startswith(field) for field in rule["fields"]):
            continue
        eligible &= ~dates.between(rule["start"], rule["end"], inclusive="both").to_numpy()
    return eligible


def _feature_eligibility_start(
    availability: InstitutionalAvailability,
    characteristic: str,
    family: str,
) -> pd.Timestamp:
    """Configured source onset for coverage; never infer it from the feature's own values."""
    if family == "broad_13f":
        dependencies = ((Tables.sec13f_hr, None),)
    elif family == "elite_13f":
        dependencies = ((Tables.sec13f_manager_holdings, None),)
    elif family == "insider":
        field = "is_10b5_1" if characteristic in {"ic_insider_discretionary_sell_mcap_60d", "ic_insider_planned_sell_mcap_60d"} else None
        dependencies = ((Tables.insider_transactions, field),)
    elif family == "short_flow":
        dependencies = ((Tables.sec_fails_to_deliver, None),) if characteristic.startswith("ic_ftd_") else ((Tables.short_interest, None),)
    elif family == "beneficial_ownership":
        dependencies = ((Tables.sec_13d, None),) if characteristic.startswith("ic_act_") else ((Tables.sec_13d, None), (Tables.sec_13g, None))
    elif family == "price_conditioning":
        if characteristic.startswith("ic_sig_super_"):
            dependencies = ((Tables.sec13f_manager_holdings, None),)
        elif characteristic.startswith("ic_sig_insider_"):
            dependencies = ((Tables.insider_transactions, None),)
        else:
            dependencies = ((Tables.sec_13d, None),)
    elif family == "cross_source_control":
        return min(
            availability.source_date(table)
            for table in (
                Tables.sec13f_hr,
                Tables.sec13f_manager_holdings,
                Tables.insider_transactions,
                Tables.short_interest,
                Tables.sec_13d,
            )
        )
    else:
        raise KeyError(f"No coverage dependency declaration for {characteristic!r} ({family!r})")
    starts = [availability.source_date(table, field) for table, field in dependencies]
    override = availability.derived_features.get(characteristic)
    if override is not None:
        starts.append(override)
    return max(starts)


def _analysis_frames(
    snapshot: Path,
    keys: pd.DataFrame,
    features: list[str],
    recent_sessions: int,
    spec: Any,
    availability: InstitutionalAvailability,
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
        source_start = _feature_eligibility_start(availability, characteristic, family)
        values = pd.to_numeric(_read_projected(snapshot, [feature])[feature], errors="coerce")
        array = values.to_numpy(dtype="float64")
        present = np.isfinite(array)
        stats = profile_stats(values, None)
        all_stats[feature] = stats
        finite = array[present]
        lower, upper = spec.bounds.get(feature, (None, None))
        bound_breaches = int(((finite < lower) if lower is not None else np.zeros(len(finite), dtype=bool)).sum())
        bound_breaches += int(((finite > upper) if upper is not None else np.zeros(len(finite), dtype=bool)).sum())
        distribution_rows.append(
            {
                "feature": feature,
                "characteristic": characteristic,
                "family": family,
                **stats,
                "finite_rate": len(finite) / len(array),
                "null_rate": 1 - len(finite) / len(array),
                "zero_rate": float(np.mean(finite == 0.0)) if len(finite) else None,
                "p05": float(np.quantile(finite, 0.05)) if len(finite) else None,
                "p95": float(np.quantile(finite, 0.95)) if len(finite) else None,
                "bound_lower": lower,
                "bound_upper": upper,
                "bound_breaches": bound_breaches,
            }
        )

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
                    "family": family,
                    "characteristic": characteristic,
                    "bound_violation": bool((lower is not None and array[position] < lower) or (upper is not None and array[position] > upper)),
                }
            )

        active = None
        rule = spec.conditional_holes.get(feature)
        if rule is not None:
            active_values = pd.to_numeric(_read_projected(snapshot, [rule["active_field"]])[rule["active_field"]], errors="coerce").to_numpy(
                dtype="float64"
            )
            active = np.isfinite(active_values)
            if "min_value" in rule:
                active &= active_values >= float(rule["min_value"])
            if "max_value" in rule:
                active &= active_values <= float(rule["max_value"])

        for ticker, positions in ticker_positions.items():
            group_dates = keys.iloc[positions][TABLE.date_col]
            source_eligible = positions[(group_dates >= source_start).to_numpy()]
            warmup = FAMILY_WARMUP_SESSIONS[family]
            eligible = source_eligible[min(warmup, len(source_eligible)) :]
            if len(eligible):
                eligible_dates = keys.iloc[eligible][TABLE.date_col]
                keep = _known_ineligible_mask(spec, ticker, feature, eligible_dates)
                if active is not None:
                    keep &= active[eligible]
                eligible = eligible[keep]
            if not len(eligible):
                coverage_rows.append(
                    {
                        "feature": feature,
                        "characteristic": characteristic,
                        "family": family,
                        "ticker": ticker,
                        "eligibility_formula_id": ELIGIBILITY_FORMULA,
                        "family_warmup_sessions": FAMILY_WARMUP_SESSIONS[family],
                        "configured_source_start": source_start,
                        "warmup_candidate_rows": warmup,
                        "eligibility_start": None,
                        "eligibility_end": None,
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
                    "configured_source_start": source_start,
                    "warmup_candidate_rows": warmup,
                    "eligibility_start": keys.iloc[eligible[0]][TABLE.date_col],
                    "eligibility_end": keys.iloc[eligible[-1]][TABLE.date_col],
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

        key_dates = keys[TABLE.date_col]
        for window in sorted({126, recent_sessions}):
            recent_window_dates = dates[-min(window, len(dates)) :]
            recent_window = key_dates.isin(recent_window_dates).to_numpy()
            history_dates = dates[max(0, len(dates) - window - 5 * 252) : max(0, len(dates) - window)]
            history_window = key_dates.isin(history_dates).to_numpy()
            history_metrics = _period_metrics(array, history_window, key_dates)
            recent_metrics = _period_metrics(array, recent_window, key_dates)
            h_sample, r_sample = _sample_even(history_metrics.pop("finite")), _sample_even(recent_metrics.pop("finite"))
            distance = float(ks_2samp(h_sample, r_sample, method="asymp").statistic) if len(h_sample) and len(r_sample) else None
            coverage_ratio = (
                recent_metrics["nonnull_share"] / history_metrics["nonnull_share"]
                if history_metrics["nonnull_share"] not in (None, 0) and recent_metrics["nonnull_share"] is not None
                else None
            )
            variance_ratio = (
                recent_metrics["cross_sectional_variance"] / history_metrics["cross_sectional_variance"]
                if history_metrics["cross_sectional_variance"] not in (None, 0) and recent_metrics["cross_sectional_variance"] is not None
                else None
            )
            flagged = bool(
                (coverage_ratio is not None and coverage_ratio < 0.75)
                or (variance_ratio is not None and variance_ratio < 0.20)
                or (distance is not None and distance > 0.35)
            )
            drift_rows.append(
                {
                    "feature": feature,
                    "recent_sessions": window,
                    **{f"history_{name}": value for name, value in history_metrics.items()},
                    **{f"recent_{name}": value for name, value in recent_metrics.items()},
                    "coverage_ratio": coverage_ratio,
                    "variance_ratio": variance_ratio,
                    "distribution_distance_ks": distance,
                    "flagged": flagged,
                    "disposition": "investigate" if flagged else "pass",
                }
            )

        for fold, fold_dates in enumerate(np.array_split(dates, 5), start=1):
            fold_mask = key_dates.isin(fold_dates).to_numpy()
            metrics = _period_metrics(array, fold_mask, key_dates)
            metrics.pop("finite")
            split_rows.append(
                {
                    "feature": feature,
                    "fold": fold,
                    "start": pd.Timestamp(fold_dates[0]) if len(fold_dates) else None,
                    "end": pd.Timestamp(fold_dates[-1]) if len(fold_dates) else None,
                    **metrics,
                    "target_status": "abstained:no_target_in_candidate_snapshot",
                }
            )
        if index % 20 == 0 or index == len(features):
            log.info("analyze candidate: %d/%d columns", index, len(features))
    return (
        pd.DataFrame(coverage_rows),
        pd.DataFrame(distribution_rows),
        pd.DataFrame(
            outlier_rows,
            columns=["feature", "ticker", "date", "value", "modified_z", "family", "characteristic", "bound_violation"],
        ),
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
    availability = InstitutionalAvailability.from_config(config_node)
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

    coverage, distributions, outliers, drift, splits, stats = _analysis_frames(
        snapshot,
        keys,
        features,
        recent_sessions,
        spec,
        availability,
    )
    coverage_summary = _coverage_summary(coverage)
    redundancy, redundancy_frame = _redundancy_frame(snapshot, stats, spec.redundancy_r)
    leakage = _leakage(keys, {"as_of": manifest["as_of"]})
    coverage_wide: dict[str, pd.DataFrame] = {}
    for horizon in ("full", "recent"):
        wide = coverage_summary[coverage_summary["horizon"].eq(horizon)].pivot(index="feature", columns="bucket", values="tickers").reset_index()
        for bucket in _COVERAGE_BUCKETS:
            if bucket not in wide:
                wide[bucket] = 0
        wide["eligible_tickers"] = wide[list(_COVERAGE_BUCKETS[:-1])].sum(axis=1)
        wide["reconciled"] = wide[list(_COVERAGE_BUCKETS)].sum(axis=1).eq(int(keys["ticker"].nunique()))
        coverage_wide[horizon] = wide[["feature", *_COVERAGE_BUCKETS, "eligible_tickers", "reconciled"]]
    item4 = pd.DataFrame(
        [
            {
                "feature": name,
                "status": "removed",
                "sample_rows": 0,
                "reason": "fragile keyword extraction removed before final schema; no text-derived predictor retained",
            }
            for name in ("ic_act_purpose_board", "ic_act_purpose_strategic")
        ]
    )
    out.mkdir(parents=True, exist_ok=True)
    frames = {
        "coverage.csv": coverage,
        "coverage-summary.csv": coverage_summary,
        "coverage-full.csv": coverage_wide["full"],
        "coverage-recent252.csv": coverage_wide["recent"],
        "distributions.csv": distributions,
        "outliers.csv": outliers,
        "drift.csv": drift,
        "model-fold-diagnostics.csv": splits,
        "redundancy.csv": redundancy_frame,
        "item4-text-audit.csv": item4,
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
            "source_start": "configured source/derived-feature availability; never inferred from candidate values",
            "ticker_rows": "candidate price-grid rows on or after source onset",
            "warmup": "family warmup is removed from the denominator before scoring",
            "complete_through": "candidate end; trailing nulls remain eligible and reduce coverage",
            "never_finite": "eligible never-finite tickers score zero coverage rather than no-support",
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
