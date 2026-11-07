"""Canonical config-directory resolution for every `@cache`d loader in `data_extract`, and the real entries of a config blob."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.constants.constants import DEFAULT_CONFIG_DIR

#: A config key starting with this is inline documentation, not an entry.
_DOC_PREFIX = "_"


def resolve_config_dir(config_dir: str | None = None) -> str:
    """One canonical absolute path for a config directory (`None` means the default).

    `@cache`d loaders key on their argument, so resolving first makes every spelling of the same
    directory share one cache entry.
    """
    return str(Path(config_dir or DEFAULT_CONFIG_DIR).resolve())


def config_entries(blob: Mapping[str, Any]) -> dict[str, Any]:
    """The blob's real entries, its inline documentation keys dropped."""
    return {k: v for k, v in blob.items() if not k.startswith(_DOC_PREFIX)}
