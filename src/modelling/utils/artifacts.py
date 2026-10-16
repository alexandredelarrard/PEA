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
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

ARTIFACT_FORMAT = "transformer-pickle-v1"
TRAINING_CONTRACT = "matured-fold-es-v1"
METADATA_FILE = "metadata.json"


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
    path.write_text(json.dumps({**meta, "artifact_format": ARTIFACT_FORMAT, "training_contract": TRAINING_CONTRACT}, indent=2))
    return path


def read_metadata(directory: Path) -> dict:
    """`metadata.json` of a trained ensemble; raises when it is missing or was written by an
    older artifact layout (both mean: retrain before predicting)."""
    path = Path(directory) / METADATA_FILE
    if not path.exists():
        raise FileNotFoundError(f"No {METADATA_FILE} in {directory}; run `modelling train` / `full-train` first.")
    meta = json.loads(path.read_text())
    found = meta.get("artifact_format")
    if found != ARTIFACT_FORMAT:
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


def load_ensemble(directory: Path, horizons: Iterable[int], families: Iterable[str]) -> dict[int, dict[str, Any]]:
    """`{horizon: {family: member}}` for every member file present, in the given horizon and
    family order (missing files are skipped). Raises when no member exists at all."""
    families = list(families)
    models: dict[int, dict[str, Any]] = {}
    for h in horizons:
        members = {fam: load_member(p) for fam in families if (p := member_path(directory, int(h), fam)).exists()}
        if members:
            models[int(h)] = members
    if not models:
        raise FileNotFoundError(f"No saved model files in {directory}.")
    return models
