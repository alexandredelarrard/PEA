"""
governance_features.py  (src/data_aggregate/utils/governance_features.py)
--------------------------------------------------------------------------
Peer-relative GOVERNANCE / EXECUTIVE-PAY features built from the LLM-extracted
DEF 14A proxy archive (`def14a_llm`, one row per annual proxy, keyed on the
filing date `as_of`). This fully replaces the retired EDGAR officer/insider regex
extraction: the signal here is the QUALITY and ALIGNMENT of the board and the CEO
pay package, which the governance-premium and pay-for-performance literature link
to forward returns. (Institutional ownership comes from the 13F panel; insider
ownership is the directors+officers-as-a-group figure from the proxy.)

Characteristics (all point-in-time from each proxy's `as_of`, so leak-free):

    ceo_pay_growth              YoY growth in CEO total compensation
    ceo_pay_vs_revenue_growth   CEO-pay growth MINUS TTM revenue growth -> the
                                pay-for-performance MISALIGNMENT signal (pay racing
                                ahead of the business = governance red flag / short)
    ceo_pay_ratio               CEO-to-median-employee pay ratio (excess-pay level)
    ceo_equity_pay_pct          share of CEO pay that is equity (alignment with owners)
    ceo_tenure                  years the CEO has led the firm (calendar year − ceo_since_year;
                                experience/stability vs entrenchment)
    pct_independent_directors   board independence
    pct_female_directors        board diversity
    board_size                  board size (bloat vs lean)
    avg_board_tenure            average director tenure (entrenchment vs freshness)
    say_on_pay_support          most recent say-on-pay approval % (shareholder assent)
    insider_ownership_pct       directors+officers ownership as a group (skin in the game)

DATA NOTE: the proxy archive accrues as fetch_def14a_llm runs over the universe;
until then this panel is sparse (empty -> skipped), exactly like the analyst and
management panels. Each value is applied strictly from its filing `as_of`.
"""

from __future__ import annotations

import pandas as pd

from src.data_aggregate.utils.factors import fundamentals_to_daily
from src.data_aggregate.utils.fundamental_features import (
    _fiscal_change_to_daily,
    _infer_yoy_periods,
    build_peer_relative_panel,
)

# def14a_llm level columns surfaced directly as peer-relative features
_LEVEL_FIELDS: list[tuple[str, str]] = [
    ("ceo_pay_ratio", "ceo_pay_ratio"),
    ("ceo_equity_pay_pct", "ceo_equity_pay_pct"),
    ("pct_independent_directors", "pct_independent_directors"),
    ("pct_female_directors", "pct_female_directors"),
    ("board_size", "board_size"),
    ("avg_board_tenure", "avg_board_tenure"),
    ("say_on_pay_support_pct", "say_on_pay_support"),
    ("insider_ownership_pct", "insider_ownership_pct"),
    # founder-CEO flag (DEF 14A `ceo_is_founder`, 1/0): founder-led firms behave
    # differently (long-termism, skin in the game); the model interacts it with
    # revenue growth (a separate feature) to capture "founder-led high-growth".
    ("ceo_is_founder", "founder_ceo"),
]


def _governance_fields(
    def14a_hist: pd.DataFrame,
    idx: pd.DatetimeIndex,
    fundamentals: pd.DataFrame | None,
) -> dict:
    """Daily wide frames (date x ticker), point-in-time from each proxy `as_of`."""
    F: dict[str, pd.DataFrame] = {}

    for src, name in _LEVEL_FIELDS:
        f = fundamentals_to_daily(def14a_hist, src, idx)
        if not f.empty and f.notna().any().any():
            F[name] = f

    # CEO tenure = years the CEO has led the firm at each date. Tenure accrues daily,
    # so it is the current calendar year MINUS the (point-in-time ffilled) `ceo_since_year`,
    # not a stale as_of snapshot. Guard bad extractions (start year in the future ->
    # negative); the downstream peer-relative winsorization clips any remaining outliers.
    since = fundamentals_to_daily(def14a_hist, "ceo_since_year", idx)
    if not since.empty and since.notna().any().any():
        years = pd.Series(idx.year, index=idx, dtype="float64")
        tenure = since.rsub(years, axis=0).where(lambda t: t >= 0)
        if tenure.notna().any().any():
            F["ceo_tenure"] = tenure

    # CEO total-comp growth (proxies are annual -> one filing per year -> periods=1)
    pay_growth = _fiscal_change_to_daily(def14a_hist, "ceo_total_comp", idx,
                                         kind="pct", periods=1)
    if pay_growth.notna().any().any():
        F["ceo_pay_growth"] = pay_growth
        # pay-for-performance misalignment: CEO pay growing faster than the business.
        if fundamentals is not None and not fundamentals.empty:
            rev_growth = _fiscal_change_to_daily(
                fundamentals, "totalRevenue", idx,
                kind="pct", periods=_infer_yoy_periods(fundamentals))
            if not rev_growth.empty and rev_growth.notna().any().any():
                cols = pay_growth.columns.intersection(rev_growth.columns)
                if len(cols) > 0:
                    F["ceo_pay_vs_revenue_growth"] = pay_growth[cols] - rev_growth[cols]
    return F


def build_governance_feature_panel(
    def14a_history: pd.DataFrame | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
    fundamentals_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Long-format governance / executive-pay feature panel (`f_<name>_vs_peers`,
    `f_<name>_xs`). Empty if the DEF 14A LLM archive is unavailable."""
    if (def14a_history is None or def14a_history.empty
            or "as_of" not in def14a_history.columns):
        return pd.DataFrame(columns=["date", "ticker"])

    fields = _governance_fields(def14a_history, trading_index, fundamentals_history)
    return build_peer_relative_panel(fields, peer_dict)
