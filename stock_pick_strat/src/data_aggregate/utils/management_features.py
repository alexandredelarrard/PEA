"""
management_features.py  (src/data_aggregate/utils/management_features.py)
-------------------------------------------------------------------------
Peer-relative GOVERNANCE / OWNERSHIP / WORKFORCE features, built from the
management snapshot archive (fetch_management). These capture well-documented
"quality of the firm and its owners" premia when ranked against direct peers:

    insider_ownership       insider % held  (skin in the game)
    institutional_ownership institution % held  (smart-money backing / crowding)
    founder_led             founder among the officers (founder-premium literature)
    family_owned            family-controlled proxy (long-horizon outperformance)
    net_insider_buying      net insider buys over 6m  (insider-trading signal)
    ceo_age                 CEO age  (younger/founder dynamism, weak prior)

(revenue_per_employee lives in employee_features.py -- it now uses the FMP
historical headcount series rather than this current-only snapshot.)

DATA NOTE (same as analyst estimates): the snapshot has no free historical
archive, so every value is applied strictly point-in-time from its `as_of` and
only accrues coverage as fetch_management is run over time. Leak-free, but
~empty historically until the archive builds up; most useful right now for the
LIVE cross-sectional ranking of top firms vs peers.
"""

from __future__ import annotations
import pandas as pd

from src.data_aggregate.utils.factors import fundamentals_to_daily
from src.data_aggregate.utils.fundamental_features import build_peer_relative_panel


def _management_fields(mgmt_hist: pd.DataFrame, idx: pd.DatetimeIndex) -> dict:
    """Daily wide frames (date x ticker), point-in-time from each `as_of`."""
    F: dict[str, pd.DataFrame] = {}
    for src, name in [
        ("heldPercentInsiders", "insider_ownership"),
        ("heldPercentInstitutions", "institutional_ownership"),
        ("founder_present", "founder_led"),
        ("family_owned", "family_owned"),
        ("net_insider_buying", "net_insider_buying"),
        ("ceo_age", "ceo_age"),
    ]:
        f = fundamentals_to_daily(mgmt_hist, src, idx)
        if not f.empty and f.notna().any().any():
            F[name] = f
    return F


def build_management_feature_panel(
    management_history: pd.DataFrame | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Long-format management/ownership feature panel (`f_<name>_vs_peers`,
    `f_<name>_xs`). Empty if the snapshot history is unavailable."""
    if (management_history is None or management_history.empty
            or "as_of" not in management_history.columns):
        return pd.DataFrame(columns=["date", "ticker"])

    fields = _management_fields(management_history, trading_index)
    return build_peer_relative_panel(fields, peer_dict)
