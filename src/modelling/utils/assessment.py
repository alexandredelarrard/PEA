"""Assessment evidence serialization, safe publication and deterministic sampling."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

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


@contextmanager
def assessment_workspace(run: Path, logger: logging.Logger) -> Iterator[Path]:
    """Serialize same-run refreshes and allocate a fresh same-filesystem analysis directory."""
    lock = run / ".assessment.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write("Assessment in progress; inspect interrupted publication before removing this lock.\n")
    workspace = None
    try:
        workspace = Path(tempfile.mkdtemp(prefix=".assessment-", dir=run))
        yield workspace
    finally:
        if workspace is not None:
            _discard_assessment_directory(workspace, run, logger)
        lock.unlink(missing_ok=True)


def _discard_assessment_directory(directory: Path, run: Path, logger: logging.Logger) -> None:
    """Remove owned scratch output without turning a completed publication into a failure."""
    if directory.resolve().parent != run.resolve():
        raise ValueError("Assessment scratch directory escaped its run")
    if directory.exists():
        try:
            shutil.rmtree(directory)
        except OSError as exc:
            logger.warning("Assessment scratch cleanup failed; retained %s: %s", directory, exc)


def _replace(source: Path, destination: Path) -> None:
    """Allow a bounded delay for Windows scanners to release a just-written directory."""
    for attempt in range(5):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1 * 2**attempt)


def publish_assessment(run: Path, workspace: Path, state: dict[str, Any], logger: logging.Logger) -> Path:
    """Replace a complete analysis/report with bounded backup and rollback; mark completion last."""
    backup = run / f".assessment-backup-{uuid4().hex}"
    backup.mkdir()
    previous, published = [], []
    try:
        for name in ("assessment", "report.html"):
            destination = run / name
            if destination.exists():
                _replace(destination, backup / name)
                previous.append(name)
        _replace(workspace / "report.html", run / "report.html")
        published.append("report.html")
        _replace(workspace, run / "assessment")
        published.append("assessment")
        report = run / "report.html"
        state["complete"] = True
        write_json(run / "assessment/metadata.json", state)
        write_json(run.parent / "latest.json", {"run_id": state["run_id"], "directory": run.name})
    except BaseException:
        for name in reversed(published):
            destination = run / name
            if destination.is_dir():
                _replace(destination, run / f".assessment-failed-{uuid4().hex}")
            else:
                destination.unlink(missing_ok=True)
        for name in previous:
            _replace(backup / name, run / name)
        raise
    finally:
        if backup.exists() and not any(backup.iterdir()):
            backup.rmdir()
    _discard_assessment_directory(backup, run, logger)
    return report


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
