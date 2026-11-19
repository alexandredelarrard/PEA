"""Load the bootstrap and stage owners without filesystem-dependent overrides."""

from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from src.utils.data_policy import validate_data_policy


def read_config(path: str | Path) -> DictConfig:
    """Load config.yml first and sorted stage/family files; each namespace has one owner."""
    folder = Path(path)
    main = folder / "config.yml"
    if not main.is_file():
        raise FileNotFoundError(f"Main config file not found: {main}")
    files = [main, *sorted((p for p in folder.rglob("*.yml") if p != main), key=lambda p: p.relative_to(folder).as_posix())]
    owners: dict[str, Path] = {}
    content: dict[str, Any] = {}
    for file in files:
        fragment = OmegaConf.load(file)
        if not isinstance(fragment, DictConfig):
            raise TypeError(f"Configuration must be a mapping: {file}")
        if not all(isinstance(key, str) for key in fragment):
            raise TypeError(f"Configuration namespaces must be strings: {file}")
        duplicates = set(owners).intersection(fragment)
        if duplicates:
            key = sorted(duplicates)[0]
            raise ValueError(f"Duplicate configuration namespace {key}: {owners[key]} and {file}")
        owners.update(dict.fromkeys(fragment, file))
        content.update(cast(dict[str, Any], OmegaConf.to_container(fragment, resolve=False)))
    config = OmegaConf.create(content)
    if "data" in config:
        validate_data_policy(cast(DictConfig, config))
    return cast(DictConfig, config)
