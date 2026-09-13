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
from src.data_aggregate.transformers.step_cube_institutionals import StepCubeInstitutionals
from src.data_aggregate.transformers.step_cube_fundamentals import StepCubeFundamentals
from src.data_aggregate.transformers.step_cube_governance import StepCubeGovernance
from src.data_aggregate.transformers.step_cube_momentum import StepCubeMomentum
from src.data_aggregate.transformers.step_cube_prices import StepCubePrices
from src.data_aggregate.transformers.step_cube_target import StepCubeTarget
from src.data_aggregate.transformers.step_cube_text import StepCubeText
from src.data_aggregate.utils.common.parts import (
    CUBE_PARTS, FEATURE_PARTS, PART_BY_NAME, PART_COMMANDS,
)
from src.constants.constants import DEFAULT_CONFIG_DIR
from src.data_aggregate.utils.common.price_frames import ALL_FIELDS

# the sub-step that owns each CLI command
OWNER = {
    "build-prices": StepCubePrices,
    "build-target": StepCubeTarget,
    "build-fundamentals": StepCubeFundamentals,
    "build-momentum": StepCubeMomentum,
    "build-text": StepCubeText,
    "build-institutionals": StepCubeInstitutionals,
    "build-governance": StepCubeGovernance,
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
    """CONSTRUCT every part-owning sub-step, which nothing else in the suite did.

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
        "build_cube": {"targets": {"horizons": [30, 60, 90]}},
        "data_extract": {"redundant_ticks": []},
    })
    # `config` on the context too: sub-steps resolve their universe through
    # `load_universe_tickers`, which reads `data_extract.redundant_ticks` off `context.config`.
    # `config_dir` too: `StepCubePrices.__init__` reads the Yahoo bug register from it, and
    # doing that at construction is deliberate -- an unapproved register must fail the step
    # before it loads 1.9M price rows, which means this test exercises that path.
    ctx = SimpleNamespace(store=sqlite_store, log=logging.getLogger("test"), config=config,
                          config_dir=Path(DEFAULT_CONFIG_DIR),
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
                 StepCubeInstitutionals, StepCubeGovernance)}
    for name, fields in declared.items():
        assert fields, f"{name} declares no price fields"
        unknown = [f for f in fields if f not in ALL_FIELDS]
        assert not unknown, f"{name} declares unknown price field(s) {unknown}"

    # only the momentum step should need the full OHLCV set; the rest must be lighter
    # momentum is the only step needing BOTH price bases: close_total for the returns,
    # close_split for the four features that pair with open/high/low/volume.
    assert set(StepCubeMomentum._FIELDS) >= {"close_split", "close_total", "open", "high",
                                             "low", "volume"}
    # ⚠ THE RULE IS ABOUT WHAT A STEP BUILDS, NOT ABOUT THE FIELD COUNT. A step that builds
    # only LEVELS must never take the total-return series -- a market cap or an EV computed on
    # it would compound every dividend ever paid into the level. It SHOULD take `level_factor`,
    # which is the other half of a correct level and is not a price at all.
    for name in ("StepCubeFundamentals", "StepCubeText"):
        assert "close_total" not in declared[name], (
            f"{name} builds LEVELS (market cap, EV, per-share ratios), so it must never take "
            f"the total-return series")
    for name in ("StepCubeFundamentals", "StepCubeText", "StepCubeInstitutionals"):
        assert not ({"open", "high", "low"} & set(declared[name])), (
            f"{name} does not build bars, so materialising the OHLC range is pure memory")
    assert set(StepCubeFundamentals._FIELDS) == {"close_split", "level_factor"}
    # ⚠ INSTITUTIONALS NOW TAKES BOTH BASES, and it is the second step (with momentum) that
    # legitimately needs them -- the price-conditioning layer added in Phase 2.6 is a family of
    # RETURNS measured from a disclosure date (`ic_sig_*_ret_since` and the two excursions),
    # while the 13F / insider families still scale dollars by a market cap. The discipline the
    # blanket ban used to enforce is enforced at the call site instead: `daily_market_cap` is
    # keyword-only in `level_factor` and documents that `close_total` reintroduces the defect,
    # and the step passes it `frames.close_split`. `sector_ret` makes the conditioning returns
    # sector-residual; `ret` is the persisted daily return the realized vol comes from.
    assert set(StepCubeInstitutionals._FIELDS) == {"close_split", "close_total", "volume",
                                                   "level_factor", "sector_ret", "ret"}
    # Governance is the third and last step allowed `close_total`: pay-vs-performance
    # differences pay growth against a trailing shareholder RETURN, and a return is exactly
    # what the total-return series is for. That is why the exemption above stays a named
    # 3-tuple rather than "every step but momentum": the rule is about what a step BUILDS,
    # not about how many fields it reads. `close_split` rides along because
    # `PriceFrames.skeleton()` keys the universe grid on it -- but NOT `level_factor`, and
    # that absence is the real assertion: without it no level (market cap, EV, per-share
    # ratio) can be computed here even by accident, which is what licenses the return series.
    assert set(StepCubeGovernance._FIELDS) == {"close_split", "close_total"}
    assert "level_factor" not in StepCubeGovernance._FIELDS, (
        "governance takes close_total, so it must not also hold the level factor -- the pair "
        "is what a market-cap/EV computation needs")

    print("\n=== SANITY CHECK: declared price-field projections ===")
    for name, fields in declared.items():
        print(f"  {name:<24} {len(fields)} field(s): {', '.join(fields)}")
    print("  CONCLUSION: only the momentum step materialises full OHLCV; fundamentals and text "
          "read close alone, governance the return series alone, institutionals both bases "
          "(levels on close_split, ic_sig_* returns on close_total). Validated.")


def test_feature_parts_cover_every_group_exactly_once():
    """The feature groups of the old exploded DAG map onto the feature parts, each group
    owned by exactly one part. The count is invariant to WHERE a group lives -- it was 14
    whether `governance` sat on the old `extras` part or on its own -- so a MOVE must never
    change it. Only a genuine add or delete may touch this number, and it must say which:

      14 -> 13   `attention` DELETED (dead code: defined, never called from `run()`), with the
                 `extras` -> `institutionals` rename.
      13 -> 15   Phase 2.6/2.7 ADDED two derived panels to `cube_part_institutionals` --
                 `conditioning` (`ic_sig_*`, look-back 252: the excursion cap) and
                 `cross_source` (`ic_xs_*`, look-back 126: the distinct-ACTOR window). The
                 same edit RENAMED `short_interest` -> `short_flow`, which is a move and does
                 not change the count; its look-back moved 103 -> 322 because the family
                 gained a 252-day self-history z and a 30-day persistence count.
      15 -> 16   `ownership` DECLARED. ⚠ NOT AN ADD -- the beneficial-ownership panel has been
                 merged into `cube_part_institutionals` since Phase 2.5 and was simply MISSING
                 from `binding_lookbacks`, so this test counted 15 while the part merged seven
                 panels. Its 378 (`ownership_features.HOLDER_ACTIVE_DAYS`, both the 13G holder
                 ffill limit and the rolling distinct-filer denominator) is the LONGEST bounded
                 look-back in the part, and the warm-up covering it was luck: 390 vs 378.
                 A count that can go down when a group is forgotten is the failure this test
                 exists to catch, and it did not catch this one -- an omission reads as "not
                 added yet", which is why the entry carries a comment saying otherwise."""
    owners: dict[str, list[str]] = {}
    for part in FEATURE_PARTS:
        for group, _ in part.binding_lookbacks:
            owners.setdefault(group, []).append(part.name)
    dupes = {g: p for g, p in owners.items() if len(p) > 1}
    assert not dupes, f"feature group(s) claimed by more than one part: {dupes}"
    assert len(owners) == 16, f"expected 16 feature groups, got {len(owners)}: {sorted(owners)}"

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
