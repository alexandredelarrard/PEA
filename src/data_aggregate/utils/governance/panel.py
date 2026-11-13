"""Point-in-time governance features from proxy, vote, pay, and board records."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.panel import PanelSettings, build_peer_relative_panel
from src.data_aggregate.utils.common.pit import (
    fiscal_change_to_daily,
    fundamentals_to_daily,
    infer_yoy_periods,
)
from src.data_aggregate.utils.common.typing import frame_column
from src.data_aggregate.utils.governance.def14a_impute import bridged_total
from src.data_aggregate.utils.governance.director_comp import (
    PEER_RELATIVE_FIELDS as DIRECTOR_PAY_PEER_FIELDS,
)
from src.data_aggregate.utils.governance.director_comp import (
    director_pay_fields,
)
from src.data_aggregate.utils.governance.directors import (
    PEER_RELATIVE_FIELDS as BOARD_QUALITY_PEER_FIELDS,
)
from src.data_aggregate.utils.governance.directors import (
    board_quality_fields,
)
from src.data_aggregate.utils.governance.names import ceo_identity_changed
from src.data_aggregate.utils.governance.pay_features import (
    PEER_RELATIVE_FIELDS as PAY_PEER_FIELDS,
)
from src.data_aggregate.utils.governance.pay_features import (
    SCT_GROWTH,
    UNSHIPPED_FIELDS,
    log_growth,
    pay_fields,
    prior_annual_leg,
)
from src.data_aggregate.utils.governance.provisions_features import (
    PEER_RELATIVE_FIELDS as PROVISION_PEER_FIELDS,
)
from src.data_aggregate.utils.governance.provisions_features import (
    provision_fields,
)
from src.data_aggregate.utils.governance.staleness import (
    LEVEL_MAX_AGE_DAYS,
    expire_stale,
    measurement_to_daily,
    source_date_column,
    stint_to_daily,
    ultimate_source_dates,
)
from src.data_aggregate.utils.governance.vote_dissent_features import (
    PEER_RELATIVE_FIELDS as VOTE_PEER_FIELDS,
)
from src.data_aggregate.utils.governance.vote_dissent_features import (
    vote_dissent_fields,
)

# def14a_llm level columns read off the proxy row. HOW each is ENCODED is decided by the three
# sets below, not by membership here.
_LEVEL_FIELDS: list[tuple[str, str]] = [
    ("ceo_pay_ratio", "ceo_pay_ratio"),
    ("ceo_equity_pay_pct", "ceo_equity_pay_pct"),
    ("pct_independent_directors", "pct_independent_directors"),
    ("pct_female_directors", "pct_female_directors"),
    ("board_size", "board_size"),
    ("avg_board_tenure", "avg_board_tenure"),
    ("insider_ownership_pct", "insider_ownership_pct"),
]

# Null impossible daily values after expansion. Gating before forward fill would hide a bad
# filing behind the prior value. Flags and unbounded dollar fields are deliberately excluded.
_DOMAIN: dict[str, tuple[float, float, bool]] = {
    "ceo_equity_pay_pct": (0.0, 1.0, True),
    "pct_independent_directors": (0.0, 1.0, True),
    "pct_female_directors": (0.0, 1.0, True),
    "board_size": (3.0, 60.0, True),
    "ceo_pay_ratio": (0.0, 1e5, True),  # Zero is valid for genuinely unpaid CEOs.
    "insider_ownership_pct": (0.0, 0.90, True),
}

# The ownership ceiling applies only to issuers that have ever disclosed a dual-class
# structure; a high single-class founder stake can be real.
_DOMAIN_ONLY_WHERE: dict[str, str] = {"insider_ownership_pct": "dual_class_shares"}
_RAW_DEF14A_FIELDS: list[tuple[str, str]] = [
    ("ceo_is_founder", "founder_ceo"),
]
_PEER_LEGACY = frozenset(
    {
        "board_size",
        "ceo_pay_ratio",
        "ceo_equity_pay_pct",
        "insider_ownership_pct",
    }
)
_VS_HIST_LEGACY = frozenset({"avg_board_tenure"})


def _nonnegative(frame: pd.DataFrame) -> pd.DataFrame:
    """A share or ratio below zero is not a measurement."""
    return frame.where(frame >= 0)


#: Variance-stabilising transform applied to a field before it is centred on its peers; fields not listed are used as is.
_PEER_TRANSFORM: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "ceo_pay_ratio": lambda frame: np.log1p(_nonnegative(frame)),
    "insider_ownership_pct": lambda frame: np.sqrt(_nonnegative(frame)),
    **dict.fromkeys(VOTE_PEER_FIELDS, lambda frame: np.sqrt(_nonnegative(frame))),
}

#: Every governance feature ships raw; the peer leg uses no input trim, and the blocks are laid out peers, history, raw.
_PANEL_SETTINGS = PanelSettings(peer_transform=_PEER_TRANSFORM, trim_peer_inputs=False, leg_order=("vs_peers", "vs_hist", "raw"))

#: Fixed bound on the CEO pay log growth (and on its gap to revenue log growth): log(20) is a 20x year, -log(20) a 95% cut.
PAY_LOG_GROWTH_BOUND: float = 3.0


def _bound_pay_growth[Shaped: (pd.Series, pd.DataFrame)](growth: Shaped) -> Shaped:
    """Clip a pay log growth (or its gap to revenue log growth) to +-`PAY_LOG_GROWTH_BOUND`."""
    return growth.clip(-PAY_LOG_GROWTH_BOUND, PAY_LOG_GROWTH_BOUND)


def _domain_condition(def14a_hist: pd.DataFrame, field: str, idx: pd.DatetimeIndex, tally: dict[str, int] | None = None) -> pd.DataFrame | None:
    """Return the same-filing condition for a conditional domain gate."""
    col = _DOMAIN_ONLY_WHERE.get(field)
    if col is None:
        return None
    if col not in def14a_hist.columns:
        if tally is not None:
            tally[f"⚠ domain gate on {field} lost its discriminator ({col}) -> applied unconditionally"] = 1
        return None
    qualifying = cast(pd.DataFrame, def14a_hist.loc[frame_column(def14a_hist, field).notna()]) if field in def14a_hist.columns else def14a_hist
    if qualifying.empty:
        return None
    if "ticker" in def14a_hist.columns:
        values = cast(pd.Series, pd.to_numeric(frame_column(def14a_hist, col), errors="coerce"))
        ever = cast(pd.Series, (values.fillna(0.0) > 0).groupby(frame_column(def14a_hist, "ticker")).max())
        qualifying = cast(pd.DataFrame, qualifying.assign(**{col: frame_column(qualifying, "ticker").map(ever).fillna(False).astype(float)}))
    cond = fundamentals_to_daily(qualifying, col, idx)
    return None if cond.empty else cond


def _gate(frame: pd.DataFrame, field: str, tally: dict[str, int] | None, condition: pd.DataFrame | None = None) -> pd.DataFrame:
    """Blank out-of-domain cells and add their counts to the build tally."""
    bound = _DOMAIN.get(field)
    if bound is None or frame.empty:
        return frame
    lo, hi, lo_inclusive = bound
    inside = (frame >= lo) if lo_inclusive else (frame > lo)
    inside &= frame <= hi
    bad = frame.notna() & ~inside
    if condition is not None:
        # Reindexed onto the value frame, so a ticker or date the condition does not cover
        # falls to NaN -> `== 1` is False -> the bound does NOT apply there. That is the
        # deliberate direction: an unqualified breach is exempted rather than blanked, because
        # the condition column is measured to be complete wherever the value exists.
        bad &= condition.reindex(index=frame.index, columns=frame.columns) == 1
    n_bad = int(bad.to_numpy().sum())
    if n_bad and tally is not None:
        tally[f"domain-gated: {field} outside {'[' if lo_inclusive else '('}{lo:g}, {hi:g}]"] = n_bad
        tally[f"domain-gated: {field} tickers"] = int(bad.any(axis=0).sum())
    return frame.mask(bad)


def _expire(frame: pd.DataFrame, history: pd.DataFrame, field: str, feature: str, tally: dict[str, int] | None) -> pd.DataFrame:
    """Apply the level horizon after domain gating and tally expired cells."""
    if frame is None or frame.empty:
        return frame
    before = int(frame.notna().to_numpy().sum())
    out = expire_stale(frame, history, field, max_age_days=LEVEL_MAX_AGE_DAYS, feature=feature)
    expired = before - int(out.notna().to_numpy().sum())
    if expired and tally is not None:
        tally[f"expired >{LEVEL_MAX_AGE_DAYS}d: {feature}"] = expired
        tally[f"expired >{LEVEL_MAX_AGE_DAYS}d: {feature} (of non-null)"] = before
    return out


# Every governance feature ships raw. Only `_PEER_LEGACY` gets a peer leg and only
# `_VS_HIST_LEGACY` gets self-history; binary and meaningful-zero features stay raw.
# The two pay-growth features are log ratios clipped to `PAY_LOG_GROWTH_BOUND`, the same on every date.
# `control_wedge` is already bounded in [0, 1].


def _ceo_pay_growth(def14a_hist: pd.DataFrame, idx: pd.DatetimeIndex, tally: dict[str, int] | None = None) -> pd.DataFrame:
    """Expand filing pay log growth (clipped to +-`PAY_LOG_GROWTH_BOUND`) point-in-time and null CEO-transition comparisons.

    A cross-filing change counts only between two filings 250-550 days apart (`pay_features` annual gate); the
    filing's own SCT pair is kept whatever the gap."""
    if "ceo_total_comp" not in def14a_hist.columns or "as_of" not in def14a_hist.columns:
        return pd.DataFrame(index=idx)
    keep = [
        c
        for c in ("ticker", "as_of", "ceo_total_comp", source_date_column("ceo_total_comp"), "ceo_name_proxy", SCT_GROWTH)
        if c in def14a_hist.columns
    ]
    d = cast(pd.DataFrame, def14a_hist[keep].copy())
    d["_bridged"] = bridged_total(def14a_hist)
    d["as_of"] = pd.to_datetime(d["as_of"], errors="coerce")
    d["ceo_total_comp"] = pd.to_numeric(d["ceo_total_comp"], errors="coerce")
    # The growth is knowable at the later filing, dated by the source of that filing's total.
    d[source_date_column("chg")] = ultimate_source_dates(d, "ceo_total_comp")
    d = d.dropna(subset=["ticker", "as_of"]).sort_values(["ticker", "as_of"])
    if d.empty:
        return pd.DataFrame(index=idx)
    # Log of the ratio of two positive totals: a prior year filed as $0 has no growth rate. The prior
    # is the previous filing only when it is about one year earlier, so a skipped proxy gives NaN
    # (a carry barrier), never a two-year change labelled as yearly growth.
    prior = prior_annual_leg(d, "ceo_total_comp")
    d["chg"] = np.log(d["ceo_total_comp"].where(d["ceo_total_comp"] > 0) / prior.where(prior > 0))
    # a bridged total's change is the bridge's slope, so neither change touching it is data
    d["chg"] = d["chg"].mask(d["_bridged"] | d.groupby("ticker", sort=False)["_bridged"].shift(1, fill_value=False).astype(bool))
    if SCT_GROWTH in d.columns:
        # The filing's own FY vs FY-1 pair wins over the cross-filing change.
        sct = cast(pd.Series, pd.to_numeric(d[SCT_GROWTH], errors="coerce"))
        d["chg"] = sct.where(sct.notna(), d["chg"])
    d["chg"] = _bound_pay_growth(d["chg"])
    names = d["ceo_name_proxy"] if "ceo_name_proxy" in d.columns else pd.Series(None, index=d.index, dtype="object")
    changed = ceo_identity_changed(names, d["ticker"])
    # Unknown identity does not prove a change, so it does not mask the growth rate.
    sub = d.assign(_turn=changed.fillna(0.0))
    if sub.empty:
        return pd.DataFrame(index=idx)
    growth = measurement_to_daily(sub, "chg", idx)
    turn = fundamentals_to_daily(sub, "_turn", idx)
    bad = growth.notna() & (turn.reindex(index=growth.index, columns=growth.columns) == 1.0)
    if tally is not None:
        tally["ceo_pay_growth: nulled across a CEO change"] = int(bad.to_numpy().sum())
        tally["ceo_pay_growth: tickers with a nulled transition"] = int(bad.any(axis=0).sum())
        tally["ceo_pay_growth: kept on an UNKNOWN CEO identity (filings)"] = int((changed.reindex(sub.index).isna() & sub["chg"].notna()).sum())
    # A growth rate becomes knowable at the later filing, represented by `sub`.
    return _expire(growth.mask(bad), sub[["ticker", "as_of", "chg", source_date_column("chg")]], "chg", "ceo_pay_growth", tally)


def _control_wedge(def14a_hist: pd.DataFrame, idx: pd.DatetimeIndex, tally: dict[str, int] | None = None) -> pd.DataFrame:
    """Expand same-filing voting power minus economic ownership point-in-time."""
    need = {"ticker", "as_of", "insider_ownership_pct", "insider_voting_pct", "dual_class_shares"}
    if not need.issubset(def14a_hist.columns):
        if tally is not None:
            missing = sorted(need - set(def14a_hist.columns))
            tally[f"⚠ control_wedge not built: {', '.join(missing)} absent"] = 1
        return pd.DataFrame()

    sub = def14a_hist.loc[:, sorted(need)].copy()
    dual = cast(pd.Series, pd.to_numeric(frame_column(sub, "dual_class_shares"), errors="coerce"))
    own = cast(pd.Series, pd.to_numeric(frame_column(sub, "insider_ownership_pct"), errors="coerce"))
    vote = cast(pd.Series, pd.to_numeric(frame_column(sub, "insider_voting_pct"), errors="coerce"))

    # Reuse the daily ownership bounds so the two paths cannot drift.
    lo, hi, _ = _DOMAIN["insider_ownership_pct"]
    own = own.where((own >= 0.0) & ((own <= hi) | (dual == 0)) & (own <= 1.0))
    vote = vote.where(vote.between(0.0, 1.0))
    # The wedge is as old as its oldest leg; the voting leg counts only where it was disclosed.
    legs = [ultimate_source_dates(def14a_hist, c) for c in ("insider_ownership_pct", "dual_class_shares")]
    legs.append(ultimate_source_dates(def14a_hist, "insider_voting_pct").where(vote.notna()))
    sub[source_date_column("control_wedge")] = pd.concat(legs, axis=1).min(axis=1)

    # single class -> voting equals ownership; unknown structure -> unknown wedge
    vote = vote.where(vote.notna(), own.where(dual == 0))
    wedge = (vote - own).where(dual.notna())

    swapped = int((wedge < 0).sum())
    if swapped and tally is not None:
        tally["⚠ control_wedge < 0 (voting/ownership legs swapped): filings"] = swapped
    sub["control_wedge"] = wedge.where(wedge >= 0)
    if not sub["control_wedge"].notna().any():
        return pd.DataFrame()

    daily = fundamentals_to_daily(sub, "control_wedge", idx)
    # Control structure is a level, aged from the filing that supplied both legs.
    return _expire(daily, sub[["ticker", "as_of", "control_wedge", source_date_column("control_wedge")]], "control_wedge", "control_wedge", tally)


def _def14a_raw_fields(def14a_hist: pd.DataFrame, idx: pd.DatetimeIndex, tally: dict[str, int] | None = None) -> dict[str, pd.DataFrame]:
    """Build raw-only DEF 14A fields."""
    out: dict[str, pd.DataFrame] = {}
    for src, name in _RAW_DEF14A_FIELDS:
        f = _gate(stint_to_daily(def14a_hist, src, idx), src, tally, _domain_condition(def14a_hist, src, idx, tally))
        f = _expire(f, def14a_hist, src, name, tally)
        if not f.empty and f.notna().any().any():
            out[name] = f
    return out


def _governance_fields(
    def14a_hist: pd.DataFrame,
    idx: pd.DatetimeIndex,
    fundamentals: pd.DataFrame | None,
    tally: dict[str, int] | None = None,
) -> dict:
    """Build point-in-time daily governance frames from proxy filings."""
    f_dict: dict[str, pd.DataFrame] = {}

    for src, name in _LEVEL_FIELDS:
        daily = stint_to_daily(def14a_hist, src, idx) if src.startswith("ceo_") else fundamentals_to_daily(def14a_hist, src, idx)
        f = _gate(daily, src, tally, _domain_condition(def14a_hist, src, idx, tally))
        f = _expire(f, def14a_hist, src, name, tally)
        if not f.empty and f.notna().any().any():
            f_dict[name] = f

    # Tenure accrues daily but still expires from the filing that disclosed `ceo_since_year`.
    since = stint_to_daily(def14a_hist, "ceo_since_year", idx)
    if not since.empty and since.notna().any().any():
        years = pd.Series([stamp.year for stamp in idx], index=idx, dtype="float64")
        tenure = since.rsub(years, axis=0).where(lambda t: t >= 0)
        tenure = _expire(tenure, def14a_hist, "ceo_since_year", "ceo_tenure", tally)
        if tenure.notna().any().any():
            f_dict["ceo_tenure"] = tenure

    # CEO total-comp growth (proxies are annual -> one filing per year -> periods=1), guarded
    # across a CEO change -- `_ceo_pay_growth` says why that is not `fiscal_change_to_daily`.
    pay_growth = _ceo_pay_growth(def14a_hist, idx, tally)
    if not pay_growth.empty and pay_growth.notna().any().any():
        f_dict["ceo_pay_growth"] = pay_growth
        # pay-for-performance misalignment: CEO pay growing faster than the business.
        if fundamentals is not None and not fundamentals.empty:
            rev_growth = log_growth(fiscal_change_to_daily(fundamentals, "totalRevenue", idx, kind="pct", periods=infer_yoy_periods(fundamentals)))
            if not rev_growth.empty and rev_growth.notna().any().any():
                cols = pay_growth.columns.intersection(rev_growth.columns)
                if len(cols) > 0:
                    # Pay growth is already expired; subtraction preserves that mask.
                    f_dict["ceo_pay_vs_revenue_growth"] = _bound_pay_growth(pay_growth[cols] - rev_growth[cols])

    # Voting power minus economic ownership has a meaningful zero and is already bounded.
    wedge = _control_wedge(def14a_hist, idx, tally)
    if not wedge.empty and wedge.notna().any().any():
        f_dict["control_wedge"] = wedge

    return f_dict


def build_governance_feature_panel(
    def14a_history: pd.DataFrame | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
    fundamentals: pd.DataFrame | None = None,
    votes: pd.DataFrame | None = None,
    exec_comp: pd.DataFrame | None = None,
    directors: pd.DataFrame | None = None,
    director_comp: pd.DataFrame | None = None,
    close_total: pd.DataFrame | None = None,
    availability: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build the long governance panel and per-family data-quality tallies.

    Every input is expected to have been through `clean.clean_governance_inputs`.
    """
    # `_gate` writes before any feature family, so the tally is owned here.
    tally: dict[str, int] = {}
    raw: dict[str, pd.DataFrame] = {}
    if def14a_history is not None and not def14a_history.empty and "as_of" in def14a_history.columns:
        computed = _governance_fields(def14a_history, trading_index, fundamentals, tally)
        computed.update(_def14a_raw_fields(def14a_history, trading_index, tally))
        # Raw is canonical; selected encodings are additive.
        raw.update(computed)

    # Executive-pay families own their child-table and return-series inputs.
    pay_frames, pay_tally = pay_fields(def14a_history, exec_comp, fundamentals, close_total, peer_dict, trading_index, availability)
    raw.update({k: v for k, v in pay_frames.items() if k not in UNSHIPPED_FIELDS})

    # Provision changes are differenced in filing space before daily expansion.
    prov_frames, prov_tally = provision_fields(def14a_history, trading_index)
    raw.update(prov_frames)

    # Per-person director sources supply quality, turnover, and compensation families.
    quality_frames, quality_tally = board_quality_fields(directors, trading_index)
    raw.update(quality_frames)

    dpay_frames, dpay_tally = director_pay_fields(director_comp, def14a_history, trading_index)
    raw.update(dpay_frames)

    vote_fields, vote_tally = vote_dissent_fields(votes, trading_index)
    tally.update(vote_tally)
    tally.update(pay_tally)
    tally.update(prov_tally)
    tally.update(quality_tally)
    tally.update(dpay_tally)

    # Vote fields ship raw; selected bounded levels also get a peer leg.
    raw.update(vote_fields)

    # Raw is canonical: the selected fields add a peer leg, `avg_board_tenure` a self-history leg.
    peer_fields = _PEER_LEGACY | PAY_PEER_FIELDS | PROVISION_PEER_FIELDS | BOARD_QUALITY_PEER_FIELDS | DIRECTOR_PAY_PEER_FIELDS | VOTE_PEER_FIELDS
    emission = {name: "raw+peers" if name in peer_fields else "raw+hist" if name in _VS_HIST_LEGACY else "raw" for name in raw}
    panel = build_peer_relative_panel(raw, peer_dict, emission=emission, settings=_PANEL_SETTINGS, availability=availability)
    return panel, tally
