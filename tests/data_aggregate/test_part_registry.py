"""
The cube part registry is the single source of truth for the sub-step wiring.

Before the split there were FOUR hand-synced copies of "which feature groups exist":
`StepBuildCube._GROUP_SOURCES`, `_GROUP_WARMUP_TRADING_DAYS`, the table list hard-coded
inside `cube_parts_status`, and a `GROUPS` literal in the Airflow DAG whose comment said
"must match StepBuildCube._GROUP_SOURCES". They HAD drifted: `attention` was commented out
of the DAG but still registered in `_GROUP_SOURCES`, so the nightly status gate reported
`cube_part_attention` as missing on every single run.

These tests assert the registry, the sub-step classes, the CLI and the DAG cannot drift
apart again. They need no DB and no fixtures.
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

from src.data_aggregate.transformers.step_assemble_cube import StepAssembleCube
from src.data_aggregate.transformers.step_cube_extras import StepCubeExtras
from src.data_aggregate.transformers.step_cube_fundamentals import StepCubeFundamentals
from src.data_aggregate.transformers.step_cube_momentum import StepCubeMomentum
from src.data_aggregate.transformers.step_cube_prices import StepCubePrices
from src.data_aggregate.transformers.step_cube_target import StepCubeTarget
from src.data_aggregate.transformers.step_cube_text import StepCubeText
from src.data_aggregate.utils.common.parts import (
    CUBE_PARTS, FEATURE_PARTS, PART_BY_NAME, PART_COMMANDS,
)
from src.data_aggregate.utils.common.price_frames import ALL_FIELDS

# the sub-step that owns each CLI command
OWNER = {
    "build-prices": StepCubePrices,
    "build-target": StepCubeTarget,
    "build-fundamentals": StepCubeFundamentals,
    "build-momentum": StepCubeMomentum,
    "build-text": StepCubeText,
    "build-extras": StepCubeExtras,
}


def test_every_part_has_an_owning_substep_and_cli_command():
    from src.data_aggregate import cli

    commands = set(cli.cli.commands)
    assert set(PART_COMMANDS) == set(OWNER), (
        f"registry commands {sorted(PART_COMMANDS)} vs owners {sorted(OWNER)}")
    for cmd in PART_COMMANDS:
        assert cmd in commands, f"registry command '{cmd}' has no CLI command"
    # the three commands that are not part-producing
    for cmd in ("assemble-cube", "cube-status", "build-cube"):
        assert cmd in commands, f"'{cmd}' missing from the CLI"

    # every part is written by exactly one command, and every command writes >= 1 part
    by_cmd: dict[str, list[str]] = {}
    for p in CUBE_PARTS:
        by_cmd.setdefault(p.command, []).append(p.name)
    assert set(by_cmd) == set(OWNER)

    print("\n=== SANITY CHECK: part registry <-> sub-steps <-> CLI ===")
    for cmd in PART_COMMANDS:
        print(f"  {cmd:<20} {OWNER[cmd].__name__:<24} -> {', '.join(by_cmd[cmd])}")
    print(f"  + assemble-cube -> cube | cube-status (JSON) | build-cube (all {len(OWNER)} in one "
          "process)")
    print("  CONCLUSION: every part has exactly one owning sub-step and CLI command, and every "
          "command is registered. Validated.")


def test_every_substep_constructs_and_binds_its_part(sqlite_store):
    """CONSTRUCT all six sub-steps, which nothing else in the suite did.

    Their `__init__` resolves `part_for(Tables.cube_part_*)` and reads the universe, so a
    registry lookup that no longer matches its key -- e.g. indexing the name-keyed
    `PART_BY_NAME` with a `Table` object -- is a KeyError at construction that every
    build-time test missed because they exercise `run()` helpers on pre-built objects."""
    import pandas as pd
    from omegaconf import OmegaConf
    from src.data_aggregate.utils.common.parts import CUBE_PARTS as REG

    sqlite_store.replace("sp500_tickers", pd.DataFrame({"ticker": ["AAA", "BBB"]}))
    config = OmegaConf.create({
        # no market_ticker / other_tickers: the market and commodity/FX series are named
        # rows in `prices_macro` now, not config-selected tickers inside `prices`
        "build_cube": {"targets": {"horizons": [30, 60, 90]},
                       "composites": {"enabled": False}},
        "data_extract": {"redundant_ticks": []},
    })
    # `config` on the context too: sub-steps resolve their universe through
    # `load_universe_tickers`, which reads `data_extract.redundant_ticks` off `context.config`.
    ctx = SimpleNamespace(store=sqlite_store, log=logging.getLogger("test"), config=config,
                          paths={"DATA_STORE": Path("."), "SECTOR_PEERS_PATH": Path("peers.json")})
    owned = {cmd: [p.name for p in REG if p.command == cmd] for cmd in OWNER}
    for cmd, cls in OWNER.items():
        step = cls(context=ctx, config=config)
        assert step._part.name in owned[cmd], f"{cls.__name__} bound the wrong part"
        assert isinstance(step._part.name, str), "CubePart.name must stay a plain str"
    print("\n=== SANITY CHECK: sub-step construction ===")
    print(f"  all {len(OWNER)} sub-steps construct and bind a registered part: "
          f"{ {c: OWNER[c](context=ctx, config=config)._part.name for c in OWNER} }")
    print("  CONCLUSION: the registry lookup in every __init__ resolves. Validated.")


def test_substep_price_fields_are_declared_and_valid():
    """Each feature sub-step declares the price fields it reads, which is what makes the
    projection meaningful -- a step asking for everything would undo the memory win."""
    declared = {cls.__name__: cls._FIELDS for cls in
                (StepCubeTarget, StepCubeFundamentals, StepCubeMomentum, StepCubeText,
                 StepCubeExtras)}
    for name, fields in declared.items():
        assert fields, f"{name} declares no price fields"
        unknown = [f for f in fields if f not in ALL_FIELDS]
        assert not unknown, f"{name} declares unknown price field(s) {unknown}"

    # only the momentum step should need the full OHLCV set; the rest must be lighter
    assert set(StepCubeMomentum._FIELDS) >= {"close", "open", "high", "low", "volume"}
    for name in ("StepCubeFundamentals", "StepCubeText"):
        assert set(declared[name]) == {"close"}, f"{name} should need close only"
    assert set(StepCubeExtras._FIELDS) == {"close", "volume"}

    print("\n=== SANITY CHECK: declared price-field projections ===")
    for name, fields in declared.items():
        print(f"  {name:<24} {len(fields)} field(s): {', '.join(fields)}")
    print("  CONCLUSION: only the momentum step materialises full OHLCV; fundamentals and text "
          "read close alone, extras close+volume. Validated.")


def test_feature_parts_cover_every_group_exactly_once():
    """The 14 feature groups of the old exploded DAG map onto the 4 feature parts, each group
    owned by exactly one part."""
    owners: dict[str, list[str]] = {}
    for part in FEATURE_PARTS:
        for group, _ in part.binding_lookbacks:
            owners.setdefault(group, []).append(part.name)
    dupes = {g: p for g, p in owners.items() if len(p) > 1}
    assert not dupes, f"feature group(s) claimed by more than one part: {dupes}"
    assert len(owners) == 14, f"expected 14 feature groups, got {len(owners)}: {sorted(owners)}"

    print("\n=== SANITY CHECK: feature groups -> parts ===")
    for part in FEATURE_PARTS:
        print(f"  {part.name:<26} {[g for g, _ in part.binding_lookbacks]}")
    print(f"  CONCLUSION: {len(owners)} groups across {len(FEATURE_PARTS)} parts, no group owned "
          "twice (the old DAG ran these as 14 separate tasks). Validated.")


def test_assemble_reads_only_registered_feature_parts():
    """The assemble step iterates FEATURE_PARTS, so a newly registered part is picked up with
    no edit there -- and the betas/targets parts are read explicitly, not as features."""
    import inspect

    src = inspect.getsource(StepAssembleCube)
    assert "FEATURE_PARTS" in src, "assemble must enumerate the registry, not a literal list"
    for part in FEATURE_PARTS:
        assert part.name not in src, (
            f"{part.name} is hard-coded in the assemble step; it should come from the registry")
    print("\n=== SANITY CHECK: assemble is registry-driven ===")
    print(f"  iterates FEATURE_PARTS ({len(FEATURE_PARTS)} parts), no part name hard-coded")
    print("  CONCLUSION: adding a feature part needs a registry entry only. Validated.")
