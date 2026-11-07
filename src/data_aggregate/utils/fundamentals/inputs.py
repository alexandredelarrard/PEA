"""Explicit financial histories, supplemental facts and existing feature settings."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data_aggregate.utils.common.xs import HIST_MIN_PERIODS, HIST_WINDOW
from src.utils.price.adjustments import ShareBasis


@dataclass(frozen=True, slots=True)
class FundamentalHistory:
    """Keep raw cash units separate from the proven share-comparison working view."""

    raw_history: pd.DataFrame | None = None
    normalized_share_history: pd.DataFrame | None = None
    share_basis: ShareBasis | None = None


@dataclass(frozen=True, slots=True)
class FundamentalSupplement:
    """Facts needed only during the fundamental calculation."""

    earnings_history: pd.DataFrame | None = None
    notes_num: pd.DataFrame | None = None


@dataclass(frozen=True, slots=True)
class FundamentalFeatureSettings:
    """Existing history, earnings and intrinsic-model controls."""

    history_window: int = HIST_WINDOW
    history_min_periods: int = HIST_MIN_PERIODS
    earnings_lags: int = 8
    earnings_scale_quarters: int = 8
    persistence_quarters: int = 8
    intrinsic_cfg: dict | None = None
