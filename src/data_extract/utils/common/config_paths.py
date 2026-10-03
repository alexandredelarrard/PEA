"""Canonical config-directory resolution for every `@cache`d loader in `data_extract`."""

from __future__ import annotations

from pathlib import Path

from src.constants.constants import DEFAULT_CONFIG_DIR


def resolve_config_dir(config_dir: str | None = None) -> str:
    """One canonical absolute path for a config directory (`None` means the default).

    `@cache`d loaders key on their argument, so resolving first makes every spelling of the same
    directory share one cache entry.
    """
    return str(Path(config_dir or DEFAULT_CONFIG_DIR).resolve())
