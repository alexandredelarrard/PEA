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
`CROSS_SOURCE_INPUTS`, so the sink holds **8** wide frames, not the ~90 the four builders
produce. A `(date x ticker)` float64 frame is ~31 MB on this universe; keeping all of them
would add 2.8 GB to a build whose whole point was to stop doing that. The alternative --
pivoting the 8 columns back out of the merged long panel -- costs a second full concat of a
~2 GB panel, which is worse.

⚠ THE SIGNAL NAMES ARE A DECLARATION, AND A WRONG ONE IS SILENT. If a feature is renamed and
this map is not, the sink simply never receives it and a family count quietly loses a family.
`cross_source_features` therefore REQUIRES at least `MIN_FAMILIES` inputs per direction and
logs exactly which resolved -- the "audit the gate before believing the gate" lesson from the
Phase 2.3 config sweep, which first ran a regex that would have condemned four live features.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

#: Registry section 8 #83 -- the four INDEPENDENT bullish actors, by the feature that measures
#: each one's intensity. One per family: the point of the count is that four unrelated parties
#: agree, so two features off the same source would double-count one opinion.
BULLISH_INPUTS: dict[str, str] = {
    "insider_buy":      "ic_insider_buy_value_mcap_180d",
    "elite_conviction": "ic_super_conviction_chg",
    "activist_entry":   "ic_act_initial_13d",
    "escalation_13g":   "ic_bo_escalation_13g_to_13d",
}

#: Registry section 8 #85 -- the bearish side, which requires EXPLICIT negative actions. The
#: absence of buying is not a short signal (report section 3.6.2), so there is no
#: "no insider bought" member here, and every one of these is somebody doing something.
BEARISH_INPUTS: dict[str, str] = {
    "insider_sell":       "ic_insider_discretionary_sell_mcap_60d",
    "elite_exit":         "ic_super_full_exits",
    "activist_reduction": "ic_act_delta_percent_class",
    "short_flow":         "ic_shortvol_high_x_weak_price",
}

#: Members whose BEARISH direction is the negative of the feature: a stake REDUCTION is a
#: negative change in percent-of-class, so the flag has to be taken on the sign-flipped value.
NEGATED_INPUTS: frozenset[str] = frozenset({"activist_reduction"})

CROSS_SOURCE_INPUTS: frozenset[str] = frozenset(BULLISH_INPUTS.values()) | frozenset(
    BEARISH_INPUTS.values())


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

    `signals[name]` -- a `(date x ticker)` wide frame for one of `CROSS_SOURCE_INPUTS`.
    """
    events: dict[str, pd.DataFrame] = field(default_factory=dict)
    actors: dict[str, pd.DataFrame] = field(default_factory=dict)
    signals: dict[str, pd.DataFrame] = field(default_factory=dict)

    def add_events(self, family: str, frame: pd.DataFrame | None) -> None:
        """Record `family`'s disclosure dates. A None/empty frame is recorded as nothing, so a
        consumer sees an ABSENT family rather than an empty one."""
        self._add(self.events, family, frame, ("ticker", "date", "value", "shares"))

    def add_actors(self, family: str, frame: pd.DataFrame | None) -> None:
        """Record `family`'s bullish acts with the actor behind each one."""
        self._add(self.actors, family, frame, ("ticker", "date", "actor"), need_actor=True)

    @staticmethod
    def _add(target: dict, family: str, frame: pd.DataFrame | None,
             columns: tuple[str, ...], need_actor: bool = False) -> None:
        if frame is None or frame.empty:
            return
        # ⚠ DUPLICATE NAMES ARE REJECTED HERE, at the contract point, because the symptom
        # surfaces nowhere near the cause. `insider_features` renamed `shares_n -> shares` on a
        # frame that already had a `shares` column; `.loc[:, keep]` happily returns BOTH, and
        # the build died 25 minutes later inside `signal_conditioning` on a `to_numeric` that
        # was handed a DataFrame. A sink that accepts an ambiguous column is not a contract.
        dupes = sorted({c for c in frame.columns if list(frame.columns).count(c) > 1})
        if dupes:
            raise ValueError(f"{family} frame has duplicate column(s) {dupes} -- project the "
                             f"columns you want BEFORE renaming into them")
        keep = [c for c in columns if c in frame.columns]
        required = {"ticker", "date"} | ({"actor"} if need_actor else set())
        if not required.issubset(keep):
            raise KeyError(f"{family} needs {sorted(required)}; got {list(frame.columns)}")
        target[family] = frame.loc[:, keep].reset_index(drop=True)

    def keep_signals(self, fields: dict[str, pd.DataFrame]) -> None:
        """Keep only the frames `CROSS_SOURCE_INPUTS` declares -- see the module docstring."""
        for name, frame in fields.items():
            if name in CROSS_SOURCE_INPUTS and frame is not None and not frame.empty:
                self.signals[name] = frame
