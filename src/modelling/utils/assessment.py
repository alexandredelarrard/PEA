"""Assessment arithmetic with fixed populations and deterministic feature elimination."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from yaml.nodes import MappingNode

from src.modelling.utils.metrics import daily_ic_series


def json_value(value: Any) -> Any:
    """JSON evidence uses null for unavailable numbers, never non-standard NaN/Infinity tokens."""
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    """Atomically replace one run-owned evidence file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(json_value(payload), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def frame_digest(frame: pd.DataFrame, columns: list[str]) -> str:
    """Order-independent normalized row-hash SHA256; detects data changes before reusing fits."""
    names = list(dict.fromkeys(["date", "ticker", *sorted(columns)]))
    normalized = frame[names].copy(deep=False)
    for column in names:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = normalized[column].astype("datetime64[ns]")
        elif column != "ticker" and str(normalized[column].dtype) != "float32":
            normalized[column] = pd.to_numeric(normalized[column], errors="raise").astype("float32")
    rows = pd.util.hash_pandas_object(normalized, index=False).to_numpy(dtype="uint64")
    digest = hashlib.sha256(json.dumps(names).encode())
    digest.update(np.sort(rows).tobytes())
    return digest.hexdigest()


def set_horizon_features(text: str, family: str, field: str, values: list[str], horizon: int = 60) -> str:
    """Change one horizon feature entry while preserving unrelated YAML text and comments."""
    root = yaml.compose(text)
    if not isinstance(root, MappingNode):
        raise ValueError("Model YAML must contain a family mapping")
    family_entry = next(((key, value) for key, value in root.value if key.value == family), None)
    if family_entry is None or not isinstance(family_entry[1], MappingNode):
        raise ValueError(f"Missing model YAML family {family}")
    family_key, family_value = family_entry
    existing = next(((key, value) for key, value in family_value.value if key.value == field), None)
    indentation = (existing[0].start_mark.column if existing else family_key.start_mark.column + 2) + 2
    pad = " " * indentation
    entry = f"{pad}{horizon}: []\n" if not values else f"{pad}{horizon}:\n" + "".join(f"{pad}  - {json.dumps(name)}\n" for name in values)
    if existing is not None:
        if not isinstance(existing[1], MappingNode) or existing[1].flow_style:
            raise ValueError("Horizon overrides must use a block-style YAML mapping")
        old = next(((key, value) for key, value in existing[1].value if str(key.value) == str(horizon)), None)
        if old is not None:
            start = text.rfind("\n", 0, old[0].start_mark.index) + 1
            end = _yaml_line_boundary(text, old[1].end_mark.index)
            return text[:start] + entry + text[end:]
        position = _yaml_line_boundary(text, existing[1].end_mark.index)
        return text[:position] + entry + text[position:]
    position = _yaml_line_boundary(text, family_value.end_mark.index)
    prefix = "" if position == 0 or text[position - 1] == "\n" else "\n"
    return text[:position] + prefix + " " * (indentation - 2) + field + ":\n" + entry + text[position:]


def _yaml_line_boundary(text: str, index: int) -> int:
    line_start = text.rfind("\n", 0, index) + 1
    if not text[line_start:index].strip():
        return line_start
    following = text.find("\n", index)
    return following + 1 if following >= 0 else len(text)


def audit_features(
    chunks: Iterable[pd.DataFrame],
    features: list[str],
    all_columns: list[str],
    categoricals: list[str],
    excluded: Mapping[str, str],
    sample_budget: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Exact streamed eligibility/support plus an explicitly sampled distribution diagnostic."""
    statistics = {
        feature: {
            "observed": 0,
            "parse_errors": 0,
            "nonfinite": 0,
            "min": float("inf"),
            "max": -float("inf"),
            "first": None,
            "last": None,
            "dtypes": set(),
            "bad_examples": [],
        }
        for feature in features
    }
    populations: Counter = Counter()
    counts: Counter = Counter()
    dates: Counter = Counter()
    samples = pd.DataFrame()
    total_rows = 0
    for chunk in chunks:
        if chunk[["date", "ticker"]].isna().any().any() or chunk.duplicated(["date", "ticker"]).any():
            raise ValueError("Null or duplicate cube keys in an audit chunk")
        total_rows += len(chunk)
        df_dates = pd.to_datetime(chunk["date"])
        dates.update(df_dates.value_counts().to_dict())
        sector = chunk["sector"].astype("str").fillna("Unknown") if "sector" in chunk else pd.Series("Unknown", index=chunk.index)
        identifiers = pd.MultiIndex.from_arrays([df_dates.dt.year, sector])
        codes, groups = pd.factorize(identifiers)
        sizes = np.bincount(codes)
        populations.update({tuple(group): int(count) for group, count in zip(groups, sizes, strict=True)})
        priority = pd.util.hash_pandas_object(chunk[["date", "ticker"]], index=False).to_numpy(dtype="uint64") ^ np.uint64(seed)
        take = np.argpartition(priority, min(sample_budget, len(chunk)) - 1)[:sample_budget] if len(chunk) > sample_budget else np.arange(len(chunk))
        sample_data: dict[str, Any] = {
            "date": df_dates.iloc[take].to_numpy(),
            "ticker": chunk["ticker"].iloc[take].to_numpy(),
            "_sample_priority": priority[take],
        }
        for feature in features:
            stats = statistics[feature]
            raw = chunk[feature]
            numeric = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            malformed = raw.notna().to_numpy() & np.isnan(numeric)
            with np.errstate(over="ignore", invalid="ignore"):
                normalized = numeric.astype("float32")
            finite = np.isfinite(normalized)
            stats["dtypes"].add(str(raw.dtype))
            stats["parse_errors"] += int(malformed.sum())
            stats["nonfinite"] += int((~np.isnan(numeric) & ~finite).sum())
            stats["observed"] += int(finite.sum())
            if finite.any():
                stats["min"] = min(stats["min"], float(normalized[finite].min()))
                stats["max"] = max(stats["max"], float(normalized[finite].max()))
                first, last = str(df_dates.loc[finite].min().date()), str(df_dates.loc[finite].max().date())
                stats["first"] = min(stats["first"] or first, first)
                stats["last"] = max(stats["last"] or last, last)
            bad = np.flatnonzero(malformed | (~np.isnan(numeric) & ~finite))[:3]
            stats["bad_examples"] = (
                stats["bad_examples"]
                + [
                    {"date": str(df_dates.iloc[index].date()), "ticker": str(chunk["ticker"].iloc[index]), "value": str(raw.iloc[index])}
                    for index in bad
                ]
            )[:3]
            observed = np.bincount(codes, weights=finite, minlength=len(groups))
            counts.update({(feature, *group): int(count) for group, count in zip(groups, observed, strict=True)})
            sample_data[feature] = normalized[take]
        samples = pd.concat([samples, pd.DataFrame(sample_data)], ignore_index=True).nsmallest(sample_budget, "_sample_priority")
    ledger = []
    for column in all_columns:
        stats = statistics.get(column)
        reason = excluded.get(column)
        if stats is not None:
            stats["dtypes"] = sorted(stats["dtypes"])
            reason = (
                "unresolved_numeric_text"
                if stats["parse_errors"]
                else "nonfinite_model_input"
                if stats["nonfinite"]
                else "all_null_development"
                if not stats["observed"]
                else "constant_development"
                if stats["min"] == stats["max"]
                else reason
            )
        ledger.append(
            {
                "feature": column,
                "role": "excluded" if reason else "categorical" if column in categoricals else "numeric",
                "reason": reason,
                **(stats or {}),
            }
        )
    coverage = [
        {"feature": feature, "year": int(year), "sector": str(sector), "observed": count, "rows": populations[(year, sector)]}
        for (feature, year, sector), count in counts.items()
    ]
    report = {
        "rows": total_rows,
        "date_counts": [{"date": str(date.date()), "rows": count} for date, count in sorted(dates.items())],
        "features": ledger,
        "coverage": coverage,
        "sample_rows": len(samples),
        "sample_policy": "lowest seeded date/ticker hash; distributions only, never training input",
    }
    return report, samples.drop(columns="_sample_priority", errors="ignore")


def summarize_ic(values: pd.Series, horizon: int) -> dict[str, Any]:
    """Summarize a pooled daily IC series using the existing population-std/horizon convention."""
    finite = values.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    mean = float(finite.mean()) if len(finite) else float("nan")
    std = float(finite.std()) if len(finite) else float("nan")
    ratio = mean / std * np.sqrt(252 / horizon) if std > 0 else float("nan")
    return {"mean_ic": mean, "ic_std": std, "ic_ir": float(ratio), "n_days": len(finite)}


def assess_candidate(expected: pd.DataFrame, predictions: pd.DataFrame, horizon: int, label_name: str = "y") -> tuple[dict[str, Any], pd.Series]:
    """Pair on fixed keys; a candidate cannot improve by dropping predictions or undefined IC days."""
    keys = ["date", "ticker"]
    if expected.duplicated(keys).any() or predictions.duplicated(keys).any():
        raise ValueError("Duplicate assessment prediction/population keys")
    extra = predictions[keys].merge(expected[keys], on=keys, how="left", indicator=True)
    if (extra["_merge"] == "left_only").any():
        raise ValueError("Unexpected out-of-fold prediction keys")
    valid_target = np.isfinite(expected[label_name].to_numpy(dtype=float))
    supported = expected.loc[valid_target, keys + [label_name]].copy()
    breadth = supported.groupby("date")[label_name].agg(["size", "nunique"])
    dates = breadth.index[(breadth["size"] >= 3) & (breadth["nunique"] >= 3)]
    supported = supported.loc[supported["date"].isin(dates)]
    joined = supported.merge(predictions[keys + ["pred"]], on=keys, how="left", validate="one_to_one", indicator=True)
    missing = int((joined["_merge"] != "both").sum())
    nonfinite = int((~np.isfinite(joined["pred"].to_numpy(dtype=float))).sum())
    series = daily_ic_series(joined, label_name)
    undefined = len(dates) - len(series)
    summary = summarize_ic(series, horizon)
    eligible = not nonfinite and not undefined and np.isfinite(summary["ic_ir"])
    summary.update(
        {
            "eligible": bool(eligible),
            "objective": summary["ic_ir"] if eligible else None,
            "expected_rows": len(supported),
            "expected_dates": len(dates),
            "missing_prediction_rows": missing,
            "nonfinite_prediction_rows": nonfinite,
            "undefined_dates": undefined,
            "common_excluded_dates": expected["date"].nunique() - len(dates),
            "common_invalid_target_rows": int((~valid_target).sum()),
        }
    )
    return summary, series


def prune_pool(features: list[str], importance: Mapping[str, float], fraction: float) -> tuple[list[str], list[str], str]:
    """Remove exact sampled zeros first, otherwise the lowest ceil(fraction * N), retaining one input."""
    if not 0 < fraction < 1 or not features:
        raise ValueError("Pruning needs a non-empty pool and a fraction in (0, 1)")
    if len(features) != len(set(features)):
        raise ValueError("Duplicate feature pool")
    if any(feature not in importance or not np.isfinite(importance[feature]) or importance[feature] < 0 for feature in features):
        raise ValueError("Missing or invalid feature importance")
    ranked = sorted(features, key=lambda feature: (importance[feature], feature))
    zeros = [feature for feature in ranked if importance[feature] == 0]
    count = min(len(features) - 1, len(zeros) if zeros else ceil(fraction * len(features)))
    removed = ranked[:count]
    return [feature for feature in features if feature not in set(removed)], removed, "zero_importance" if zeros else "bottom_fraction"


def balanced_sample_positions(panel: pd.DataFrame, budget: int, seed: int) -> np.ndarray:
    """Sample equal row counts per date where supported, independently of candidate features."""
    if budget <= 0 or budget >= len(panel):
        return np.arange(len(panel))
    groups = list(panel.groupby("date", sort=True).indices.values())
    sizes = np.array([len(group) for group in groups])
    quotas = np.minimum(sizes, budget // len(groups))
    remaining = budget - int(quotas.sum())
    rng = np.random.default_rng(seed)
    while remaining:
        available = np.flatnonzero(quotas < sizes)
        additions = np.minimum(sizes[available] - quotas[available], max(1, remaining // len(available)))
        if additions.sum() > remaining:
            quotas[rng.permutation(available)[:remaining]] += 1
            break
        quotas[available] += additions
        remaining -= int(additions.sum())
    selected = [rng.choice(group, int(quota), replace=False) for group, quota in zip(groups, quotas, strict=True) if quota]
    return np.sort(np.concatenate(selected))
