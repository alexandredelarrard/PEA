"""Shared assessment identities, finite-number summaries and deterministic sampling."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(json_value(payload), handle, indent=2, allow_nan=False)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def frame_digest(frame: pd.DataFrame, columns: list[str], *, downcast: bool = True) -> str:
    """Order-independent input identity; scoring can retain prediction-relevant float64 values."""
    names = list(dict.fromkeys(["date", "ticker", *sorted(columns)]))
    normalized = frame[names].copy(deep=False)
    for column in names:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = normalized[column].astype("datetime64[ns]")
        elif column != "ticker":
            normalized[column] = pd.to_numeric(normalized[column], errors="raise").astype("float32" if downcast else "float64")
    rows = pd.util.hash_pandas_object(normalized, index=False).to_numpy(dtype="uint64")
    digest = hashlib.sha256(json.dumps(names).encode())
    digest.update(np.sort(rows).tobytes())
    return digest.hexdigest()


def summarize_ic(values: pd.Series, horizon: int) -> dict[str, Any]:
    """Summarize a pooled daily IC series using the existing population-std/horizon convention."""
    finite = values.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    mean = float(finite.mean()) if len(finite) else float("nan")
    std = float(finite.std()) if len(finite) else float("nan")
    ratio = mean / std * np.sqrt(252 / horizon) if std > 0 else float("nan")
    return {"mean_ic": mean, "ic_std": std, "ic_ir": float(ratio), "n_days": len(finite)}


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
