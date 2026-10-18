"""
artifacts.py  (src/modelling/utils/artifacts.py)
------------------------------------------------
Where trained members live and how they are read back. One pickle per (horizon, family) --
`model_h<h>_<family>.pkl`, the fitted transformer itself -- plus `metadata.json`, the
compatibility contract read by prediction, the L/S sleeves and the app. `artifact_format`
in the metadata names the pickle layout, so artifacts from an older layout fail with a
"retrain" message instead of an unpickling error deep inside a module that no longer exists.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import shutil
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.constants.constants import MODEL_GENERATION_ARTIFACT_FORMAT
from src.modelling.utils.assessment import write_json

ARTIFACT_FORMAT = "transformer-pickle-v1"
TRAINING_CONTRACT = "matured-fold-es-v1"
METADATA_FILE = "metadata.json"


def validate_run_id(run_id: str) -> str:
    """Accept one portable filename component, never a path or reserved Windows name."""
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", run_id):
        raise ValueError("run_id must be one safe alphanumeric filename component")
    if re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", run_id, flags=re.IGNORECASE):
        raise ValueError("run_id cannot be a reserved Windows filename")
    return run_id


def new_run_id() -> str:
    """UTC dated identity for a directly invoked training run."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")


def contained_path(directory: Path, relative: str) -> Path:
    """Resolve a nonempty relative artifact path within its declared directory."""
    root = Path(directory).resolve()
    path = Path(relative)
    resolved = (root / path).resolve()
    if path.is_absolute() or resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"Artifact directory/path must be relative and cannot escape {root}")
    return resolved


def run_directory(context: Any, config: DictConfig, run_id: str | None = None) -> Path:
    """Resolve a named assessment run without creating or selecting a different run."""
    identity = run_id if run_id is not None else OmegaConf.select(config, "model.assessment.run_id")
    validate_run_id(identity)
    raw = OmegaConf.select(config, "model.assessment.output_dir", default="output/modelling_backtest")
    directory = Path(str(raw))
    if not directory.is_absolute():
        directory = Path(context.paths["DATA_STORE"]) / directory
    return contained_path(directory, identity)


def read_run(directory: Path, *, training_required: bool = True) -> dict[str, Any]:
    """Read this exact run, refusing incomplete training as an assessment/refit input."""
    state = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
    if state.get("run_id") != directory.name or not state.get("horizons") or not state.get("model_types"):
        raise ValueError("Invalid or wrong-run modelling manifest")
    if training_required and state.get("training_complete") is not True:
        raise ValueError("Training artifacts are incomplete for this run")
    return state


def fitting_source_identity() -> dict[str, str]:
    """Hash the fitting/scoring implementation without requiring Git in a task container."""
    root = Path(__file__).resolve().parents[3]
    relative = [
        *[
            f"src/modelling/transformers/{name}"
            for name in ("__init__.py", "base.py", "lightgbm_model.py", "random_forest.py", "linear_regression.py")
        ],
        *[f"src/modelling/utils/{name}" for name in ("features.py", "panel.py", "cv.py", "ensemble.py", "metrics.py", "artifacts.py")],
        "src/modelling/steps/step_long_short.py",
    ]
    return {name: file_identity(root / name)["sha256"] for name in relative}


def file_identity(path: Path) -> dict[str, Any]:
    """Exact non-tabular artifact size and SHA-256."""
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def verify_inventory(directory: Path, inventory: dict[str, dict[str, Any]]) -> None:
    """Validate every declared byte identity before consuming any member."""
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("Missing artifact file inventory")
    for name, expected in inventory.items():
        if not isinstance(name, str) or not isinstance(expected, dict) or set(expected) != {"bytes", "sha256"}:
            raise ValueError("Malformed artifact file identity")
        if type(expected["bytes"]) is not int or expected["bytes"] < 0 or not isinstance(expected["sha256"], str):
            raise ValueError("Malformed artifact file identity")
        if not re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]):
            raise ValueError("Malformed artifact file hash")
        path = contained_path(directory, name)
        if not path.is_file():
            raise FileNotFoundError(f"Missing artifact file: {path}")
        if file_identity(path) != expected:
            raise ValueError(f"Artifact file identity changed: {path}")


def member_inventory(directory: Path, horizons: Iterable[int], families: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Inventory the exact configured ensemble, refusing missing or nested member filenames."""
    families = list(families)
    inventory = {}
    for horizon in horizons:
        for family in families:
            path = member_path(directory, int(horizon), family)
            if path.parent.resolve() != Path(directory).resolve():
                raise ValueError("A member filename must stay in its declared directory")
            inventory[path.name] = file_identity(path)
    if not inventory:
        raise ValueError("The saved ensemble must contain members")
    return inventory


def models_dir(context: Any, config: DictConfig) -> Path:
    """`model.models_dir` from the config: an absolute path as-is, a relative one under
    `context.paths["DATA_STORE"]`; `context.paths["MODELS_DIR"]` when the key is unset."""
    raw = config.get("model", {}).get("models_dir")
    if not raw:
        return Path(context.paths["MODELS_DIR"])
    path = Path(str(raw))
    return path if path.is_absolute() else Path(context.paths["DATA_STORE"]) / path


def member_path(directory: Path, horizon: int, family: str) -> Path:
    """File of one ensemble member."""
    return Path(directory) / f"model_h{int(horizon)}_{family}.pkl"


def clear_members(directory: Path) -> list[Path]:
    """Delete every member pickle (`model_h<h>_<family>.pkl`) of a previous run, so a horizon or
    family this run does not produce can never be loaded next to the new members. Returns the
    deleted files."""
    removed = sorted(Path(directory).glob("model_h*_*.pkl"))
    for path in removed:
        path.unlink()
    return removed


def write_metadata(directory: Path, meta: dict) -> Path:
    """Persist `metadata.json` (stamped with `artifact_format`), creating the directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / METADATA_FILE
    artifact_format = MODEL_GENERATION_ARTIFACT_FORMAT if "member_directory" in meta else ARTIFACT_FORMAT
    write_json(path, {**meta, "artifact_format": artifact_format, "training_contract": TRAINING_CONTRACT})
    return path


def read_metadata(directory: Path) -> dict:
    """`metadata.json` of a trained ensemble; raises when it is missing or was written by an
    older artifact layout (both mean: retrain before predicting)."""
    path = Path(directory) / METADATA_FILE
    if not path.exists():
        raise FileNotFoundError(f"No {METADATA_FILE} in {directory}; run `modelling train` / `full-train` first.")
    meta = json.loads(path.read_text())
    found = meta.get("artifact_format")
    if found not in (ARTIFACT_FORMAT, MODEL_GENERATION_ARTIFACT_FORMAT):
        raise RuntimeError(
            f"Model artifacts in {directory} use artifact_format={found!r}, this code reads {ARTIFACT_FORMAT!r}: "
            "retrain with `python -m src modelling train` / `full-train` before predicting."
        )
    if meta.get("training_contract") != TRAINING_CONTRACT:
        raise RuntimeError("Obsolete training/input contract; retrain with `modelling train` before predicting or full-train.")
    return meta


def selection_signature(config: DictConfig, features: list[str], horizons: list[int]) -> str:
    """Identify the fixed training inputs/parameters whose fold rounds may be reused."""
    shared = config.model
    payload = {
        "features": features,
        "horizons": horizons,
        "blocks": {key: OmegaConf.to_container(config[key], resolve=True) for key in ("linear", "lgbm", "random_forest", "train") if key in config},
        "shared": {
            key: OmegaConf.to_container(shared[key], resolve=True) if OmegaConf.is_config(shared.get(key)) else shared.get(key)
            for key in ("ensemble", "cv", "weight_decay", "target_type", "label_column")
        },
        "seed": config.get("seed"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def save_member(model: Any, path: Path) -> Path:
    """Pickle one fitted member (its own `__getstate__` drops context / config / logger)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_member(path: Path) -> Any:
    """Unpickle one member; the pickle names its own class, so no family registry is needed."""
    with Path(path).open("rb") as f:
        return pickle.load(f)


def load_ensemble(
    directory: Path, horizons: Iterable[int], families: Iterable[str], *, metadata: dict[str, Any] | None = None
) -> dict[int, dict[str, Any]]:
    """Load a pinned, verified manifest or an explicit optional-member legacy flat directory."""
    directory = Path(directory)
    horizons, families = list(horizons), list(families)
    if metadata is not None:
        generated = metadata.get("artifact_format") == MODEL_GENERATION_ARTIFACT_FORMAT
        if generated:
            relative = metadata.get("member_directory")
            if not isinstance(relative, str) or not relative:
                raise ValueError("Missing generation member directory")
            directory = contained_path(directory, relative)
        elif "member_directory" in metadata:
            raise ValueError("Generation member directory requires the generation artifact format")
        inventory = metadata.get("member_files")
        if generated or inventory is not None:
            declared_horizons = metadata.get("horizons", horizons)
            declared_families = metadata.get("model_types", families)
            expected = {member_path(directory, int(h), fam).name for h in declared_horizons for fam in declared_families}
            if not isinstance(inventory, dict) or set(inventory) != expected:
                raise ValueError("Incomplete model member inventory")
            requested = {member_path(directory, int(h), fam).name for h in horizons for fam in families}
            if not requested or not requested <= set(inventory):
                raise ValueError("Requested members are absent from the model inventory")
            verify_inventory(directory, inventory)
    models: dict[int, dict[str, Any]] = {}
    for h in horizons:
        if any(member_path(directory, int(h), fam).parent.resolve() != directory.resolve() for fam in families):
            raise ValueError("A member filename must stay in its declared directory")
        members = {fam: load_member(p) for fam in families if (p := member_path(directory, int(h), fam)).exists()}
        if members:
            models[int(h)] = members
    if not models:
        raise FileNotFoundError(f"No saved model files in {directory}.")
    return models


def publish_generation(directory: Path, snapshot: Path, run_id: str, attempt_id: str) -> dict[str, Any]:
    """Copy a verified full-history snapshot, then atomically publish one complete metadata view."""
    validate_run_id(run_id)
    validate_run_id(attempt_id)
    metadata = read_metadata(snapshot)
    if metadata.get("run_id") != run_id or metadata.get("full_history") is not True or "member_directory" in metadata:
        raise ValueError("Publication requires this run's flat full-history snapshot")
    inventory = metadata.get("member_files")
    expected = {member_path(snapshot, int(h), fam).name for h in metadata["horizons"] for fam in metadata["model_types"]}
    if not isinstance(inventory, dict) or set(inventory) != expected:
        raise ValueError("Incomplete model member inventory")
    verify_inventory(snapshot, inventory)
    relative = Path("generations") / run_id / attempt_id
    generation = contained_path(directory, str(relative))
    generation.mkdir(parents=True, exist_ok=False)
    for name in inventory:
        shutil.copyfile(contained_path(snapshot, name), contained_path(generation, name))
    verify_inventory(generation, inventory)
    published = {**metadata, "member_directory": relative.as_posix(), "full_history_attempt_id": attempt_id}
    write_metadata(directory, published)
    return {**published, "artifact_format": MODEL_GENERATION_ARTIFACT_FORMAT, "training_contract": TRAINING_CONTRACT}
