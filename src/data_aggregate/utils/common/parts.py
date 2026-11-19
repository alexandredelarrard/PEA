"""Cube build ownership and minimum algorithm dependencies; timing policies live in data.yml."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.data_store.schema import Table, Tables, name_of

PartKind = Literal["prices", "features", "targets", "betas"]


@dataclass(frozen=True, slots=True)
class CubePart:
    """A build command and bounded daily-grid dependencies, separate from configured read context."""

    table: Table
    command: str
    kind: PartKind
    binding_lookbacks: tuple[tuple[str, int], ...] = ()

    @property
    def name(self) -> str:
        return self.table.name


CUBE_PARTS: tuple[CubePart, ...] = (
    CubePart(Tables.cube_part_prices, "build-prices", "prices"),
    CubePart(Tables.cube_part_targets, "build-target", "targets"),
    CubePart(Tables.cube_part_betas, "build-target", "betas"),
    CubePart(
        Tables.cube_part_fundamentals,
        "build-fundamentals",
        "features",
        (("fundamental", 1260), ("dividend", 1530), ("employee", 0), ("sector", 0), ("earnings", 0)),
    ),
    CubePart(Tables.cube_part_momentum, "build-momentum", "features", (("price", 1260),)),
    CubePart(Tables.cube_part_text, "build-text", "features", (("earnings_call_sentiment", 0), ("earnings_call_embedding", 0))),
    CubePart(Tables.cube_part_institutionals, "build-institutionals", "features", (("short_flow", 322), ("ownership", 126), ("insider", 0))),
    CubePart(Tables.cube_part_sec_thirteen, "build-sec-thirteen", "features", (("cluster", 0),)),
    CubePart(Tables.cube_part_governance, "build-governance", "features", (("governance", 1260),)),
)

FEATURE_PARTS: tuple[CubePart, ...] = tuple(part for part in CUBE_PARTS if part.kind == "features")
PART_BY_NAME: dict[str, CubePart] = {part.name: part for part in CUBE_PARTS}
PART_COMMANDS: tuple[str, ...] = tuple(dict.fromkeys(part.command for part in CUBE_PARTS))
TERMINAL_TABLES: tuple[Table, ...] = (Tables.cube, Tables.predictions, Tables.cube_signal, Tables.predictions_latest)


def part_for(table: Table | str) -> CubePart:
    """Resolve build ownership from the registered table identity."""
    return PART_BY_NAME[name_of(table)]
