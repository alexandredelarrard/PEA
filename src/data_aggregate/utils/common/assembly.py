"""Shared assembly policy: registered optional inputs and exact read projections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.constants.constants import PANEL_KEYS
from src.data_aggregate.utils.common.parts import PART_BY_NAME, CubePart
from src.data_store.store import DataStore


def _names(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{label} must be a YAML list")
    if any(not isinstance(name, str) or not name.strip() or name != name.strip() for name in value):
        raise ValueError(f"{label} contains a blank or invalid name")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} contains duplicate names")
    return tuple(value)


@dataclass(frozen=True)
class AssemblyPolicy:
    optional: frozenset[str]
    columns: dict[str, tuple[str, ...]]

    @classmethod
    def from_config(cls, build_cube: Any) -> AssemblyPolicy:
        raw = build_cube.get("assembly", {})
        if not isinstance(raw, Mapping):
            raise ValueError("build_cube.assembly must be a YAML mapping")
        optional = _names(raw.get("optional_parts", []), "assembly.optional_parts")
        selections = raw.get("columns", {})
        if not isinstance(selections, Mapping):
            raise ValueError("assembly.columns must be a YAML mapping")
        column_parts = _names(list(selections), "assembly.columns")
        unknown = (set(optional) | set(column_parts)) - PART_BY_NAME.keys()
        if unknown:
            raise ValueError(f"assembly contains unknown cube parts: {sorted(unknown)}")
        columns = {name: () if values is None else _names(values, f"assembly.columns.{name}") for name, values in selections.items()}
        return cls(frozenset(optional), columns)

    def projection(self, part: CubePart, actual: list[str]) -> list[str]:
        wanted = list(dict.fromkeys([*PANEL_KEYS, *(self.columns.get(part.name) or actual)]))
        missing = sorted(set(wanted) - set(actual))
        if missing:
            raise ValueError(f"{part.name} missing columns {missing}; run {part.command}")
        return wanted


def part_has_rows(store: DataStore, part: CubePart) -> bool:
    """Probe physical presence before validating requested optional payload columns."""
    if not store.exists(part.table):
        return False
    actual = store.columns(part.table)
    probe = store.load(part.table, columns=actual[:1], limit=1, optional=True)
    return probe is not None and not probe.empty
