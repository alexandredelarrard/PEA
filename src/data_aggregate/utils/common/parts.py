"""
parts.py  (src/data_aggregate/utils/common/parts.py)
--------------------------------------------------
THE registry of cube part tables: one row per intermediate table a sub-step writes,
carrying its CLI sub-command, its kind, and its incremental warm-up.

This replaces three parallel dicts that had to be kept in sync by hand
(`StepBuildCube._GROUP_SOURCES`, `_GROUP_WARMUP_TRADING_DAYS` and the hard-coded table
list inside `cube_parts_status`), plus a fourth copy in the Airflow DAG whose comment
read "must match StepBuildCube._GROUP_SOURCES". They drifted: `attention` was commented
out of the DAG but still listed in `_GROUP_SOURCES`, so the status gate reported
`cube_part_attention` missing on every run. That panel has since been deleted outright;
the registry is why its removal needed one edit here rather than four in lockstep.

WARM-UPS. Each part is rebuilt incrementally -- read its latest date, recompute only a
warm-up-padded trailing window, append the rows after that date. `warmup_trading_days` is
the longest look-back the part's features compute ON THE DAILY PRICE GRID, plus a safety
buffer. Source tables (fundamentals / 13F / insider / def14a / ...) are read in FULL, so a
builder whose look-back is in FILING or QUARTER space needs ~no grid warm-up and is
floored at ~6 months.

`binding_lookbacks` records, per merged feature group, the look-back that actually binds.
It is DATA rather than a literal duplicated in the test, so
`tests/data_aggregate/test_part_registry.py` can assert every warm-up covers its members.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.data_store.schema import Table, Tables, name_of

# No "market" kind: `cube_part_market` existed only to keep the market/commodity/FX series out
# of the equity frame the cross-sectional ranks are built on. They live in `prices_macro` now.
PartKind = Literal["prices", "features", "targets", "betas"]


@dataclass(frozen=True, slots=True)
class CubePart:
    """One part's BUILD ORCHESTRATION: which CLI sub-command owns it and how far back an
    incremental run must warm up. Schema (name, PK, date column) lives on `table`, so this
    registry no longer re-declares any of it.

    Note `kind` here is aggregation semantics (it drives `FEATURE_PARTS` and the status gate's
    `never_behind` set) -- NOT `Table.kind`, which is DDL grouping.
    """
    table: Table
    command: str
    kind: PartKind
    warmup_trading_days: int
    # (merged feature group, its longest DAILY-grid look-back in trading days)
    binding_lookbacks: tuple[tuple[str, int], ...] = ()

    @property
    def name(self) -> str:
        return self.table.name


CUBE_PARTS: tuple[CubePart, ...] = (
    # The normalized price grid every other step reads instead of re-loading `prices`.
    # `ret` and `sector_ret` are PERSISTED, so a trailing recompute reproduces them exactly
    # (a trimmed window's first pct_change row would otherwise come back NaN). 260 days is
    # not a look-back: it keeps a year of context in `get_trading_days`'s interior-calendar
    # -hole warning, which is a diagnostic over history.
    CubePart(Tables.cube_part_prices, "build-prices", "prices", 520),

    # style momentum shift(252) + beta window(126); the forward horizon is added at the call
    # site, because targets look FORWARD and recent NaN labels MATURE between runs.
    CubePart(Tables.cube_part_targets, "build-target", "targets", 390),
    CubePart(Tables.cube_part_betas, "build-target", "betas", 390),

    CubePart(Tables.cube_part_fundamentals, "build-fundamentals", "features", 1320,
             (("fundamental", 1260),      # _self_history_z rolling(1260)
              ("dividend", 1260),         # 5y payout growth shift(5 * 252)
              ("employee", 252),          # YoY headcount / rev-per-employee shift(252)
              ("sector", 0),              # _yearly_lag over the FULL fundamentals history
              ("earnings", 0))),          # trailing-4Q rolling over REPORTED quarters
    CubePart(Tables.cube_part_momentum, "build-momentum", "features", 1320,
             (("price", 1260),)),         # seasonal_h*: close.shift(252 * seasonal_years=5)
    CubePart(Tables.cube_part_text, "build-text", "features", 130,
             (("earnings_call_sentiment", 0),   # QoQ over reported quarters
              ("earnings_call_embedding", 0))),  # QoQ embedding drift
    # `short_interest` is now the BINDING look-back at 103: the `attention` entry (63) went with
    # the attention panel itself. The warm-up stays 160 -- it still covers 103 with buffer, and
    # the price-conditioning look-backs that will be added to this part have to be re-checked
    # against it anyway. `test_part_registry.py` asserts every warm-up covers its members, so
    # that check is enforced rather than remembered.
    CubePart(Tables.cube_part_institutionals, "build-institutionals", "features", 160,
             (("short_interest", 103),    # short-vol rolling(63) + FTD shift(40)
              ("institutional", 0),       # QoQ vs the prior 13F period
              ("superinvestor", 0),
              ("insider", 0))),           # rolling('180D') over the FULL transaction calendar
    # Governance sources are all FILING-space (annual proxies, 8-K vote records), so every YoY
    # delta needs no grid warm-up at all. TWO legs bind on the daily grid, and the longer one
    # decides:
    #   * 252d -- the trailing shareholder return the pay-vs-performance family differences
    #     against pay growth (phase 4);
    #   * 1260d -- `self_history_z` behind `f_avg_board_tenure_vs_hist`, a trailing 5-year
    #     self-z taken on the DAILY frame (phase 3, D51).
    # ⚠ THE 1260 IS WHY THIS PART IS HEAVY, and under-declaring it was a live defect: with a
    # 390-day warm-up an incremental run gives `self_history_z` 390 days of context where a
    # full run gives it 1260, so the rolling mean and std differ and the incremental tail
    # silently disagrees with a rebuild. `min_periods=252` makes it produce a WRONG number
    # rather than a NaN, which is what hid it.
    # A cheaper fix exists and is deferred to phase 7: take the self-z on the FILING grain (a
    # trailing 5-FILING z) and ffill it, which needs no grid warm-up at all and is arguably the
    # better statistic -- a day-weighted mean of an annual series weights each proxy by how long
    # it happened to stay current. It would change the measured values, so it needs its
    # sign-stability screen re-run before it ships.
    CubePart(Tables.cube_part_governance, "build-governance", "features", 1260,
             (("governance", 1260),)),
)

FEATURE_PARTS: tuple[CubePart, ...] = tuple(p for p in CUBE_PARTS if p.kind == "features")
PART_BY_NAME: dict[str, CubePart] = {p.name: p for p in CUBE_PARTS}


# the ordered CLI sub-commands the DAG chains (deduplicated, registry order preserved)
PART_COMMANDS: tuple[str, ...] = tuple(dict.fromkeys(p.command for p in CUBE_PARTS))
# downstream tables reported alongside the parts by the status gate
TERMINAL_TABLES: tuple[Table, ...] = (Tables.cube, Tables.predictions, Tables.cube_signal,
                                      Tables.predictions_latest)


def part_for(table: Table | str) -> CubePart:
    """The build orchestration for a part table. Accepts the `Table` or its name -- `PART_BY_NAME`
    is keyed by NAME, so indexing it with a `Table` object raises KeyError."""
    return PART_BY_NAME[name_of(table)]
