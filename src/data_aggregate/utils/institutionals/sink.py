"""
sink.py  (src/data_aggregate/utils/institutionals/sink.py)
----------------------------------------------------------
THE CONTRACT BETWEEN THE FOUR SOURCE PANELS AND THE TWO DERIVED ONES.

The price-conditioning layer (`signal_conditioning.py`, registry section 7) needs each family's
EVENT DATES, and the cross-source layer (`cross_source_features.py`, section 8) needs a handful
of the family INTENSITY frames plus the identity of each ACTOR behind an event. Neither can
re-derive them:

  * the insider event dates are the output of a 2M-row scope-and-repair pass
    (`insider_quality.clean_transactions`), which is the most expensive read in the step;
  * the elite dates are the output of the per-manager availability join, which needs the whole
    `manager_quarter_state` / `public_state` pipeline;
  * re-running either would double the build, and re-reading `sec13f_hr` (21.7M rows) beside
    the accumulating panel is what OOM-killed the aggregation task before it was coarsened.

So each `build_*_feature_panel` takes an OPTIONAL sink and drops its events and its declared
signal frames into it on the way past. Optional matters: every existing caller and test passes
nothing and is unaffected, and a builder that is skipped (its source absent) simply contributes
nothing -- the derived panels then report which families they could not build rather than
inventing one.

WHY A SMALL EXPLICIT SET AND NOT "KEEP EVERYTHING". `keep_signals` filters against
`CROSS_SOURCE_INPUTS`, so the sink holds **7** value-plus-mask pairs, not the ~90 frames the
four builders
produce. A `(date x ticker)` float64 frame is ~31 MB on this universe; keeping all of them
would add 2.8 GB to a build whose whole point was to stop doing that. The alternative --
pivoting the 7 columns back out of the merged long panel -- costs a second full concat of a
~2 GB panel, which is worse.

⚠ THE SIGNAL NAMES ARE A DECLARATION, AND A WRONG ONE IS SILENT. If a feature is renamed and
this map is not, the sink simply never receives it and a family count quietly loses a family.
`cross_source_features` therefore REQUIRES at least `MIN_FAMILIES` inputs per direction and
logs exactly which resolved -- the "audit the gate before believing the gate" lesson from the
Phase 2.3 config sweep, which first ran a regex that would have condemned four live features.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

#: Registry section 8 #83 -- the four INDEPENDENT bullish actors, by the feature that measures
#: each one's intensity. One per family: the point of the count is that four unrelated parties
#: agree, so two features off the same source would double-count one opinion.
BULLISH_INPUTS: dict[str, str] = {
    "insider_buy": "ic_insider_buy_value_mcap_180d",
    "elite_conviction": "ic_super_conviction_chg",
    "activist_entry": "ic_act_initial_13d",
    "escalation_13g": "ic_bo_escalation_13g_to_13d",
}

#: Registry section 8 #85 -- the bearish side, which requires EXPLICIT negative actions. The
#: absence of buying is not a short signal (report section 3.6.2), so there is no
#: "no insider bought" member here, and every one of these is somebody doing something.
BEARISH_INPUTS: dict[str, str] = {
    "insider_net_sell": "ic_insider_net_buy_ratio_180d",
    "elite_exit": "ic_super_full_exits",
    "short_flow": "ic_shortvol_high_x_weak_price",
}

#: Members whose bearish direction is the negative of the source feature.
NEGATED_INPUTS: frozenset[str] = frozenset({"insider_net_sell"})

CROSS_SOURCE_INPUTS: frozenset[str] = frozenset(BULLISH_INPUTS.values()) | frozenset(BEARISH_INPUTS.values())


@dataclass(frozen=True)
class AvailableSignal:
    """A signal value and the independent fact that the cell is observable."""

    values: pd.DataFrame
    available: pd.DataFrame

    def __post_init__(self) -> None:
        if not self.values.index.equals(self.available.index) or not self.values.columns.equals(self.available.columns):
            raise ValueError("Signal values and availability mask must be aligned")
        if not all(pd.api.types.is_bool_dtype(dtype) for dtype in self.available.dtypes):
            raise TypeError("Signal availability must be boolean")
        outside = self.values.notna() & ~self.available
        if bool(outside.to_numpy().any()):
            raise ValueError("Signal has values outside its declared availability")
        unexplained = self.available & self.values.isna()
        if bool(unexplained.to_numpy().any()):
            raise ValueError("Signal has available-but-unexplained NaN cells; fix the dependency or cleaning mask")


@dataclass
class ConditioningSink:
    """What the source panels hand to the derived ones. Three dicts, one purpose each.

    `events[family]` -- `[ticker, date]`, the family's DISCLOSURE dates, whatever their
    direction. This is what the conditioning layer dates its price path from: "how has the
    stock moved since the last time this family said anything". For insiders it also carries
    `value` + `shares` AS FILED, which only the #80 cost anchor reads.

    `actors[family]` -- `[ticker, date, actor]`, the BULLISH subset, carrying the identity of
    whoever acted. A different set from `events` on purpose: holding a position is a
    disclosure, but only ADDING to it is a bullish act, and #84 counts distinct actors who
    acted. An actor id is unique only within its family, so the consumer namespaces it.

    `signals[name]` -- a value frame plus its explicit per-cell availability mask.
    """

    events: dict[str, pd.DataFrame] = field(default_factory=dict)
    actors: dict[str, pd.DataFrame] = field(default_factory=dict)
    signals: dict[str, AvailableSignal] = field(default_factory=dict)
    frontiers: dict[str, pd.Timestamp] = field(default_factory=dict)

    def set_frontier(self, family: str, complete_through: pd.Timestamp | None) -> None:
        """Record the inclusive date through which `family` was actually observed."""
        if complete_through is None or pd.isna(complete_through):
            return
        value = pd.Timestamp(complete_through)
        if value.tz is not None:
            value = value.tz_localize(None)
        self.frontiers[family] = value.normalize()

    def add_events(self, family: str, frame: pd.DataFrame | None) -> None:
        """Record `family`'s disclosure dates. A None/empty frame is recorded as nothing, so a
        consumer sees an ABSENT family rather than an empty one."""
        self._add(self.events, family, frame, ("ticker", "date", "value", "shares"))

    def add_actors(self, family: str, frame: pd.DataFrame | None) -> None:
        """Record `family`'s bullish acts with the actor behind each one."""
        self._add(self.actors, family, frame, ("ticker", "date", "actor"), need_actor=True)

    @staticmethod
    def _add(target: dict, family: str, frame: pd.DataFrame | None, columns: tuple[str, ...], need_actor: bool = False) -> None:
        if frame is None or frame.empty:
            return
        # ⚠ DUPLICATE NAMES ARE REJECTED HERE, at the contract point, because the symptom
        # surfaces nowhere near the cause. `insider_features` renamed `shares_n -> shares` on a
        # frame that already had a `shares` column; `.loc[:, keep]` happily returns BOTH, and
        # the build died 25 minutes later inside `signal_conditioning` on a `to_numeric` that
        # was handed a DataFrame. A sink that accepts an ambiguous column is not a contract.
        dupes = sorted({c for c in frame.columns if list(frame.columns).count(c) > 1})
        if dupes:
            raise ValueError(f"{family} frame has duplicate column(s) {dupes} -- project the " f"columns you want BEFORE renaming into them")
        keep = [c for c in columns if c in frame.columns]
        required = {"ticker", "date"} | ({"actor"} if need_actor else set())
        if not required.issubset(keep):
            raise KeyError(f"{family} needs {sorted(required)}; got {list(frame.columns)}")
        target[family] = frame.loc[:, keep].reset_index(drop=True)

    def add_signal(
        self,
        name: str,
        values: pd.DataFrame,
        available: pd.DataFrame,
    ) -> None:
        """Register one cross-source signal, rejecting ambiguity at the boundary."""
        if name not in CROSS_SOURCE_INPUTS:
            return
        if name in self.signals:
            raise ValueError(f"Duplicate cross-source signal {name!r}")
        self.signals[name] = AvailableSignal(values=values, available=available)

    def keep_signals(
        self,
        fields: dict[str, pd.DataFrame],
        availability: Mapping[str, pd.DataFrame] | None = None,
    ) -> None:
        """Keep declared frames; production builders provide explicit masks.

        The `frame.notna()` fallback preserves isolated builder/test callers that do not
        construct the derived cross-source panel. The production orchestrator always passes
        the YAML resolver and therefore never uses this fallback.
        """
        for name, frame in fields.items():
            if name in CROSS_SOURCE_INPUTS and frame is not None and not frame.empty:
                mask = availability.get(name) if availability is not None else frame.notna()
                if mask is None:
                    raise KeyError(f"Missing availability mask for cross-source signal {name!r}")
                self.add_signal(name, frame, mask)
