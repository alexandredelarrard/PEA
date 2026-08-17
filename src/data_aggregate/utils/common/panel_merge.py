"""
panel_merge.py  (src/data_aggregate/utils/common/panel_merge.py)
--------------------------------------------------------------
Accumulate feature panels on (date, ticker) with a HARD uniqueness guarantee on feature
names, then emit one long frame.

WHY THE GUARANTEE MATTERS. A plain `merge` on the keys silently renames BOTH sides to
`<name>_x` / `<name>_y`, and that is how 20 such columns once reached the live cube: the
fundamental and sector panels each emitted `interest_coverage`, `net_debt_to_ebitda`,
`gross_profitability`, `cash_conversion_cycle` and `sbc_intensity` under DIFFERENT
formulas, so which one a model saw depended on merge order. Exactly one panel must own
each feature name.

This replaces three near-copies of the same merge-and-log tail
(`StepBuildCube._merge_panel` + `_attach_panel` + `_merge_ec_panel`, plus three
hand-rolled repeats at individual call sites) and closes a real hole: the cross-part merge
in `assemble_cube_from_parts` used `how="outer"` OUTSIDE the guard, so a name owned by two
PARTS produced `_x`/`_y` with no error at all. Coarser parts make that likelier, so the
assemble step now merges through here too.

It is also cheaper. Chained `DataFrame.merge` re-copies the whole accumulator once per
panel; this holds the panels (date, ticker)-indexed and does a single `concat(axis=1)`, so
peak memory is the panels themselves rather than N partial copies of their union.
"""
from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

from src.constants.constants import PANEL_KEYS


class FeatureCollisionError(ValueError):
    """Two feature panels emit the same feature name -- merging would split it into
    pandas `_x` / `_y` columns instead of failing loudly."""


class PanelMerger:
    """Collect feature panels, then emit one long ['date', 'ticker', <features...>] frame."""

    def __init__(self, log: logging.Logger | None = None,
                 keys: Sequence[str] = tuple(PANEL_KEYS)) -> None:
        self._log = log or logging.getLogger(__name__)
        self._keys = list(keys)
        self._frames: list[pd.DataFrame] = []
        self._owner: dict[str, str] = {}          # feature name -> the label that produced it

    @property
    def feature_columns(self) -> list[str]:
        return list(self._owner)

    def add(self, panel: pd.DataFrame | None, label: str,
            empty_msg: str | None = None) -> int:
        """Register one panel; returns how many feature columns it contributed.

        An empty / None panel is logged and skipped (a source that has not been fetched
        yet is normal, not an error). A duplicate feature name raises, naming the panel
        that already owns it."""
        if panel is None or panel.empty:
            self._log.warning(empty_msg or f"No {label} features built.")
            return 0
        missing = [k for k in self._keys if k not in panel.columns]
        if missing:
            raise ValueError(f"{label} panel is missing the join key(s) {missing}")

        features = [c for c in panel.columns if c not in self._keys]
        clash = sorted(c for c in features if c in self._owner)
        if clash:
            owners = {c: self._owner[c] for c in clash}
            raise FeatureCollisionError(
                f"feature name(s) already in the cube panel: {clash} "
                f"(owned by {owners}; now also emitted by '{label}'). Give each feature a "
                "single owning panel (or rename it) -- merging would silently split it "
                "into _x / _y columns.")
        if not features:
            self._log.warning("%s panel carries no feature columns.", label)
            return 0

        for c in features:
            self._owner[c] = label
        indexed = panel.set_index(self._keys)
        self._frames.append(indexed[features] if len(features) != len(indexed.columns)
                            else indexed)
        cov = indexed[features].notna().any(axis=1).mean()
        self._log.info("Merged %s %s features (row coverage %.1f%%)",
                       len(features), label, 100 * cov)
        return len(features)

    def to_long(self) -> pd.DataFrame:
        """One outer-aligned concat -> long frame. Empty (keys only) if nothing was added."""
        if not self._frames:
            return pd.DataFrame(columns=self._keys)
        out = pd.concat(self._frames, axis=1).copy().reset_index()
        return out
