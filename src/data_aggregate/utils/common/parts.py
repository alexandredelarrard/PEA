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

⚠ THAT LAST SENTENCE IS TRUE FOR A *BOUNDED* LOOK-BACK AND FALSE FOR AN UNBOUNDED ONE, and one
part had to stop relying on it. Reading the source in full puts every event in memory; it does
NOT put every event on the grid, because the grid starts at `window.since`. A feature that asks
"how many trading days since the last insider buy", "what has the price done since the last
13D", "is this holder new" or any expanding-window statistic is therefore computed from the
first event INSIDE the window, and no value of `warmup_trading_days` can fix it -- those
look-backs have no finite length. Worse, `decay.snap_to_grid` moves an event onto the first
trading day >= its date, so on a trimmed grid an event PREDATING the window lands on the
window's first day: a 2010 13D became a 2025 13D. Measured on `cube_part_institutionals`,
2026-09-12, over a 567-day incremental window: **76 of 127 columns drifted**, and the drift
reached the newest row, which is the only row an append writes. `f_ic_bo_new_holder` was wrong
on 490 of 491 tickers on the last date; `f_ic_sig_insider_age_days` by a median 674 days and up
to 4,412.

FIXED by `StepCubeInstitutionals._load_frames`, which takes no `since` at all: that part
COMPUTES over the full calendar on both paths and lets `write_part` slice the tail. It is
affordable there precisely because `_load_source` never trimmed the event tables anyway. Any
future part carrying an unbounded look-back needs the same treatment, not a bigger warm-up.

`binding_lookbacks` records, per merged feature group, the look-back that actually binds.
It is DATA rather than a literal duplicated in the test, so
`tests/data_aggregate/test_part_registry.py` can assert every warm-up covers its members.
⚠ It records the longest *bounded* one, so a green test is still not evidence that an
incremental build is correct -- L7 in `validate institutionals` is. And a group can go MISSING
without anything noticing: the beneficial-ownership panel had no entry until 2026-09-12, which
is why nothing flagged `HOLDER_ACTIVE_DAYS = 378` as the part's longest bounded look-back.
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
    # ⚠ 160 -> 390, and the two families that forced it are the ones Phase 2.5/2.6 added:
    #   * `short_flow` 322 -- a 252-day self-history z, then a 30-day persistence count on top
    #     of it, then the 40-day FTD publication shift. Its predecessor entry (103) described
    #     a module that no longer exists.
    #   * `conditioning` 252 -- the excursion cap on `ic_sig_insider_max_dd/runup_since_buy`
    #     (`build_cube.institutionals.excursion_lookback`).
    # 390 is the same warm-up targets/betas already use, which is not a coincidence: it is one
    # 252-day look-back plus the ~130-day buffer a trailing recompute needs to reproduce the
    # rolling statistic at the window's oldest rewritten date. `test_part_registry.py` asserts
    # every warm-up covers its members, so this is enforced rather than remembered.
    CubePart(Tables.cube_part_institutionals, "build-institutionals", "features", 390,
             (("short_flow", 322),        # z252 + persistence(30) + FTD shift(40)
              # `ownership_features.HOLDER_ACTIVE_DAYS`: both the 13G holder ffill(limit=)
              # and the rolling distinct-filer denominator. The LONGEST bounded look-back in
              # the part, and it was MISSING from this tuple entirely -- which is why
              # `test_part_registry.py` stayed green while the panel it describes was the one
              # `f_ic_bo_new_holder` went wrong in.
              ("ownership", 378),
              ("conditioning", 252),      # capped max-drawdown / run-up since the last buy
              ("institutional", 0),       # QoQ vs the prior 13F period
              ("superinvestor", 0),
              ("cross_source", 126),      # trailing distinct-ACTOR window
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
