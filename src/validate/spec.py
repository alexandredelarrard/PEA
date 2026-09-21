"""
spec.py  (src/validate/spec.py)
--------------------------------------------------------------------------------------------
Resolves what a check is allowed to assume about a table: `configs/validate.yml` defaults,
overlaid by that table's entry, overlaid by explicit kwargs.

WHY A PER-TABLE DECLARATION IS UNAVOIDABLE. Measured across six same-shaped `cube_part_*`
tables, the mean number of legs frozen inside a quarter runs from 0.1 of 28 on
`cube_part_momentum` to 66.4 of 103 on `cube_part_governance` -- and governance is CORRECT,
because it ffills quarterly source data onto a daily grid. Any built-in frozen-run default
fires on two thirds of governance and never on momentum. The same holds for clip conventions
(`_vs_peers` exists on some parts and not others) and for bounds.

So: a value a table has not declared is not defaulted, it is ABSENT, and the check that needs
it raises `UndeclaredTableError` and the CLI exits 3. Populate an entry only where a
measurement or an existing `_scripts/` constant supports the value -- an absent entry abstains
loudly, an invented one lies quietly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.data_store.schema import name_of


class UndeclaredTableError(LookupError):
    """A check needs a declaration the table does not carry. The CLI turns this into
    abstain (exit 3), never a pass."""

    def __init__(self, table: str, key: str, why: str) -> None:
        self.table, self.key = table, key
        super().__init__(f"{table} declares no `{key}` in configs/validate.yml -- {why}")


@dataclass(frozen=True)
class TableSpec:
    """Everything the checks may assume about one table, already merged.

    Every field is `None` when undeclared rather than carrying a built-in default, except the
    thresholds under `validate.defaults`, which are table-agnostic by construction (a Pearson
    r of 0.985 means the same thing on every table; a frozen run of 30 days does not)."""

    table: str
    # -- table-agnostic thresholds (always present, from `validate.defaults`) --------- #
    redundancy_r: float
    exact_equal_tol: float
    clip_limit_share: float
    tie_limit_share: float
    jump_z: float
    jump_span_frac: float
    jump_min_distinct: int
    hole_min_days: int
    frozen_min_days: int
    min_tickers_xs: int
    universe_expected: int
    coverage_min_share: float
    edge_days: int
    recent_sessions: int
    # -- per-table declarations (None == undeclared == the check abstains) ----------- #
    xs_suffix: str | None = None
    peer_suffix: str | None = None
    clip_peer: float | None = None
    cadence: str | None = None
    ffill_horizon_days: int | None = None
    daily_legs: tuple[str, ...] = ()
    pit_sources: dict[str, tuple[str, ...]] = field(default_factory=dict)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    declared: bool = False

    def require(self, key: str, why: str) -> Any:
        """The declared value of `key`, or `UndeclaredTableError`."""
        value = getattr(self, key, None)
        if value is None or (isinstance(value, (tuple, dict, list)) and not value):
            raise UndeclaredTableError(self.table, key, why)
        return value

    def is_daily_leg(self, column: str) -> bool:
        """A leg the builder re-computes EVERY session, so a flat run is a defect rather than
        a between-deadline hold. Patterns match as a prefix OR a suffix: the institutionals
        evidence is suffix-shaped (`_to_mcap`, `_ret_since` -- legs rescaled by a daily
        close), while a family that is daily end-to-end is named by its prefix."""
        return any(column.startswith(p) or column.endswith(p) for p in self.daily_legs)

    def view_suffixes(self) -> tuple[str, ...]:
        """The standardised-view suffixes this table declares, longest first so
        `_vs_peers` is stripped before a shorter suffix could match inside it."""
        return tuple(sorted((s for s in (self.xs_suffix, self.peer_suffix) if s),
                            key=len, reverse=True))


def _plain(node: Any) -> Any:
    return OmegaConf.to_container(node, resolve=True) if isinstance(node, DictConfig) else node


def load_spec(config: DictConfig, table: Any, **overrides: Any) -> TableSpec:
    """`defaults <- tables.<name> <- explicit kwargs`.

    An override of `None` is "not supplied on the command line" and never overwrites a
    declaration -- otherwise every unset CLI flag would silently un-declare the config.
    """
    name = name_of(table)
    root = config.get("validate") if hasattr(config, "get") else None
    if root is None:
        raise UndeclaredTableError(name, "validate", "configs/validate.yml is missing or unmerged")

    defaults = dict(_plain(root.get("defaults")) or {})
    tables = _plain(root.get("tables")) or {}
    declared = name in tables
    merged: dict[str, Any] = {**defaults, **(tables.get(name) or {})}
    merged.update({k: v for k, v in overrides.items() if v is not None})

    return TableSpec(
        table=name,
        redundancy_r=float(merged["redundancy_r"]),
        exact_equal_tol=float(merged["exact_equal_tol"]),
        clip_limit_share=float(merged["clip_limit_share"]),
        tie_limit_share=float(merged["tie_limit_share"]),
        jump_z=float(merged["jump_z"]),
        jump_span_frac=float(merged["jump_span_frac"]),
        jump_min_distinct=int(merged["jump_min_distinct"]),
        hole_min_days=int(merged["hole_min_days"]),
        frozen_min_days=int(merged["frozen_min_days"]),
        min_tickers_xs=int(merged["min_tickers_xs"]),
        universe_expected=int(merged["universe_expected"]),
        coverage_min_share=float(merged["coverage_min_share"]),
        edge_days=int(merged["edge_days"]),
        recent_sessions=int(merged["recent_sessions"]),
        xs_suffix=merged.get("xs_suffix"),
        peer_suffix=merged.get("peer_suffix"),
        clip_peer=None if merged.get("clip_peer") is None else float(merged["clip_peer"]),
        cadence=merged.get("cadence"),
        ffill_horizon_days=(None if merged.get("ffill_horizon_days") is None
                            else int(merged["ffill_horizon_days"])),
        daily_legs=tuple(merged.get("daily_legs") or ()),
        pit_sources={k: (v,) if isinstance(v, str) else tuple(v)
                     for k, v in (merged.get("pit_sources") or {}).items()},
        bounds={k: (float(v[0]), float(v[1]))
                for k, v in (merged.get("bounds") or {}).items()},
        declared=declared,
    )
