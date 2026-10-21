"""Select complete dated modelling runs and validate their saved fitting contract."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def validate_run_id(run_id: str) -> str:
    """Accept one portable filename component, excluding reserved Windows names."""
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", run_id):
        raise ValueError("run_id must be one safe alphanumeric filename component")
    if re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", run_id, flags=re.IGNORECASE):
        raise ValueError("run_id cannot be a reserved Windows filename")
    return run_id


def new_run_id() -> str:
    """Create an automatic UTC identity for a fresh training invocation."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")


def contained_path(directory: Path, relative: str) -> Path:
    """Resolve a nonempty relative path without leaving its owning directory."""
    root = Path(directory).resolve()
    path = Path(relative)
    resolved = (root / path).resolve()
    if path.is_absolute() or resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"Artifact directory/path must be relative and cannot escape {root}")
    return resolved


def runs_directory(context: Any, config: DictConfig) -> Path:
    """Resolve the configured run root relative to the application data store."""
    directory = Path(str(OmegaConf.select(config, "model.assessment.output_dir", default="output/modelling_backtest")))
    return directory if directory.is_absolute() else Path(context.paths["DATA_STORE"]) / directory


def run_directory(context: Any, config: DictConfig, run_id: str) -> Path:
    """Resolve an exact safe run identity without creating it."""
    return contained_path(runs_directory(context, config), validate_run_id(run_id))


def read_fitting_configuration(directory: Path) -> dict[str, Any]:
    """Read saved allowlisted modelling controls, excluding runtime/provider state."""
    snapshot = json.loads((directory / "training/configuration.json").read_text(encoding="utf-8"))
    keys = {"seed", "train", "model", "linear", "lgbm", "random_forest"}
    if not isinstance(snapshot, dict) or set(snapshot) != keys or not isinstance(snapshot["model"], dict):
        raise ValueError("Malformed saved fitting configuration")
    if any(snapshot[key] is not None and not isinstance(snapshot[key], dict) for key in keys - {"seed", "model"}):
        raise ValueError("Malformed saved fitting configuration block")
    return snapshot


def read_run(directory: Path, *, training_required: bool = True, assessed: bool = False) -> dict[str, Any]:
    """Read one new-workflow run, refusing wrong identities and incomplete required stages."""
    state = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if state.get("run_id") != directory.name or state.get("workflow") != "fresh-assessment-v1":
        raise ValueError("Invalid or obsolete modelling run; create a new training run")
    snapshot = read_fitting_configuration(directory)
    model = snapshot["model"]
    horizons, families = state.get("horizons"), state.get("model_types")
    if not horizons or not families or len(set(horizons)) != len(horizons) or len(set(families)) != len(families):
        raise ValueError("Invalid horizon/family inventory")
    if families != model.get("ensemble") or horizons != model.get("horizons", horizons):
        raise ValueError("Saved run configuration differs from declared models")
    if state.get("target_type") != model.get("target_type", "rank") or state.get("label_column") != model.get("label_column"):
        raise ValueError("Saved run target contract differs from fitted configuration")
    if training_required:
        if state.get("training_complete") is not True:
            raise ValueError("Training artifacts are incomplete for this run")
        completed = datetime.fromisoformat(state["training_completed_at_utc"])
        if completed.tzinfo is None or completed.utcoffset() is None:
            raise ValueError("Training completion time must include its timezone")
        expected = list(range(1, int(model["cv"]["n_splits"]) + 1))
        if set(state.get("folds", {})) != {str(h) for h in horizons} or any(state["folds"][str(h)] != expected for h in horizons):
            raise ValueError("Incomplete declared fold inventory")
    if assessed:
        assessment = json.loads((directory / "assessment/metadata.json").read_text(encoding="utf-8"))
        if assessment.get("complete") is not True or assessment.get("run_id") != state["run_id"] or not (directory / "report.html").is_file():
            raise ValueError("This run requires a complete successful assessment")
    return state


def select_run(context: Any, config: DictConfig, run_id: str | None = None, *, assessed: bool = False) -> tuple[Path, dict[str, Any]]:
    """Pin an explicit run or the newest complete training by its immutable completion time."""
    if run_id is not None:
        directory = run_directory(context, config, run_id)
        return directory, read_run(directory, assessed=assessed)
    root = runs_directory(context, config)
    candidates = []
    for directory in root.iterdir() if root.is_dir() else []:
        if not directory.is_dir():
            continue
        try:
            validate_run_id(directory.name)
            if directory.resolve().parent != root.resolve():
                continue
            state = read_run(directory, assessed=assessed)
            completed = datetime.fromisoformat(state["training_completed_at_utc"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        candidates.append((completed, state["run_id"], directory, state))
    if not candidates:
        qualifier = "successfully assessed" if assessed else "complete"
        raise FileNotFoundError(f"No {qualifier} new-workflow training in {root}; run modelling train first")
    _, _, directory, state = max(candidates, key=lambda item: item[:2])
    return directory, state
