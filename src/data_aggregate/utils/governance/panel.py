"""Point-in-time governance features from proxy, vote, pay, and board records."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from src.constants.constants import PANEL_KEYS
from src.data_aggregate.utils.common.panel import mask_to_availability, peer_relative
from src.data_aggregate.utils.common.pit import (
    fiscal_change_to_daily,
    fundamentals_to_daily,
    infer_yoy_periods,
)
from src.data_aggregate.utils.common.typing import frame_column
from src.data_aggregate.utils.common.xs import self_history_z, winsorize_xs
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
    pay_fields,
)
from src.data_aggregate.utils.governance.provisions_features import (
    PEER_RELATIVE_FIELDS as PROVISION_PEER_FIELDS,
)
from src.data_aggregate.utils.governance.provisions_features import (
    provision_fields,
)
from src.data_aggregate.utils.governance.staleness import LEVEL_MAX_AGE_DAYS, expire_stale
from src.data_aggregate.utils.governance.vote_dissent_features import (
    PEER_RELATIVE_FIELDS,
    vote_dissent_fields,
)

_SHARES_OUTSTANDING = "sharesOutstandingPit"

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
    "say_on_pay_support_pct": (0.0, 1.0, True),
    "board_size": (3.0, 60.0, True),
    "ceo_pay_ratio": (0.0, 1e5, True),  # Zero is valid for genuinely unpaid CEOs.
    "insider_ownership_pct": (0.0, 0.90, True),
}

# The ownership ceiling applies only to issuers that have ever disclosed a dual-class
# structure; a high single-class founder stake can be real.
_DOMAIN_ONLY_WHERE: dict[str, str] = {"insider_ownership_pct": "dual_class_shares"}
_RAW_DEF14A_FIELDS: list[tuple[str, str]] = [
    ("ceo_is_founder", "founder_ceo"),
    ("say_on_pay_support_pct", "say_on_pay_support"),
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
_RAW_ONLY_COMPUTED = frozenset({"ceo_pay_growth", "ceo_pay_vs_revenue_growth"})
_RAW_ONLY_BOUNDED = frozenset({"control_wedge"})


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
# `_RAW_ONLY_COMPUTED` is winsorized because percentage changes can explode from genuine
# near-zero compensation. `control_wedge` is already bounded in [0, 1].


def _ceo_pay_growth(def14a_hist: pd.DataFrame, idx: pd.DatetimeIndex, tally: dict[str, int] | None = None) -> pd.DataFrame:
    """Expand filing pay growth point-in-time and null CEO-transition comparisons."""
    if "ceo_total_comp" not in def14a_hist.columns or "as_of" not in def14a_hist.columns:
        return pd.DataFrame(index=idx)
    keep = [c for c in ("ticker", "as_of", "ceo_total_comp", "ceo_name_proxy") if c in def14a_hist.columns]
    d = cast(pd.DataFrame, def14a_hist[keep].copy())
    d["as_of"] = pd.to_datetime(d["as_of"], errors="coerce")
    d["ceo_total_comp"] = pd.to_numeric(d["ceo_total_comp"], errors="coerce")
    d = d.dropna(subset=["ticker", "as_of", "ceo_total_comp"]).sort_values(["ticker", "as_of"])
    if d.empty:
        return pd.DataFrame(index=idx)
    # `replace` on the infinities reproduces `fiscal_change_to_daily` exactly: a prior year
    # filed as $0 makes `pct_change` infinite, and an infinity is not a growth rate.
    d["chg"] = d.groupby("ticker", sort=False)["ceo_total_comp"].pct_change(periods=1).replace([np.inf, -np.inf], np.nan)
    names = d["ceo_name_proxy"] if "ceo_name_proxy" in d.columns else pd.Series(None, index=d.index, dtype="object")
    changed = ceo_identity_changed(names, d["ticker"])
    # Unknown identity does not prove a change, so it does not mask the growth rate.
    sub = d.loc[d["chg"].notna()].assign(_turn=changed.fillna(0.0))
    if sub.empty:
        return pd.DataFrame(index=idx)
    growth = fundamentals_to_daily(sub, "chg", idx)
    turn = fundamentals_to_daily(sub, "_turn", idx)
    bad = growth.notna() & (turn.reindex(index=growth.index, columns=growth.columns) == 1.0)
    if tally is not None:
        tally["ceo_pay_growth: nulled across a CEO change"] = int(bad.to_numpy().sum())
        tally["ceo_pay_growth: tickers with a nulled transition"] = int(bad.any(axis=0).sum())
        tally["ceo_pay_growth: kept on an UNKNOWN CEO identity (filings)"] = int(changed.reindex(sub.index).isna().sum())
    # A growth rate becomes knowable at the later filing, represented by `sub`.
    return _expire(growth.mask(bad), sub[["ticker", "as_of", "chg"]], "chg", "ceo_pay_growth", tally)


# The denominator must be point-in-time at the proxy date.
def economic_ownership(def14a_hist: pd.DataFrame, fundamentals: pd.DataFrame | None, tally: dict[str, int] | None = None) -> pd.DataFrame:
    """Replace ownership with same-filing shares/outstanding where both are available."""
    if fundamentals is None or fundamentals.empty:
        return def14a_hist
    need = {"ticker", "as_of", "insider_shares"}
    if not need.issubset(def14a_hist.columns):
        return def14a_hist
    if not {"ticker", "as_of", _SHARES_OUTSTANDING}.issubset(fundamentals.columns):
        if tally is not None:
            tally[f"⚠ economic ownership not computed: {_SHARES_OUTSTANDING} absent"] = 1
        return def14a_hist

    # merge_asof requires identical datetime resolution across the two sources.
    left = def14a_hist.copy()
    left["_as_of"] = pd.to_datetime(left["as_of"], errors="coerce").astype("datetime64[ns]")
    right = cast(
        pd.DataFrame,
        fundamentals[["ticker", "as_of", _SHARES_OUTSTANDING]]
        .copy()
        .assign(_as_of=lambda d: pd.to_datetime(d["as_of"], errors="coerce").astype("datetime64[ns]"))
        .drop(columns="as_of")
        .dropna(subset=["_as_of"]),
    )
    right = cast(pd.DataFrame, right.loc[frame_column(right, _SHARES_OUTSTANDING) > 0])
    if right.empty:
        return def14a_hist

    # Preserve row identity because merge_asof resets the index.
    left["_key"] = range(len(left))
    dated_left = cast(pd.DataFrame, left.dropna(subset=["_as_of"])).sort_values("_as_of")
    merged = pd.merge_asof(dated_left, right.sort_values("_as_of"), on="_as_of", by="ticker", direction="backward")

    shares_out = cast(pd.Series, pd.to_numeric(frame_column(merged, _SHARES_OUTSTANDING), errors="coerce"))
    insider = cast(pd.Series, pd.to_numeric(frame_column(merged, "insider_shares"), errors="coerce"))
    computed = insider / shares_out
    # a share of a whole: anything outside (0, 1] means the two counts are on different bases
    # (one class's shares over the total, or a count in thousands) and is not usable
    computed = computed.where((shares_out > 0) & (computed > 0) & (computed <= 1.0))

    # Economic ownership cannot exceed the disclosed voting share on a comparable basis.
    if "insider_voting_pct" in merged.columns:
        vote = cast(pd.Series, pd.to_numeric(frame_column(merged, "insider_voting_pct"), errors="coerce"))
        bad = computed.notna() & vote.notna() & (computed > vote + 1e-9)
        if tally is not None and int(bad.sum()):
            tally["insider_ownership_pct: computed value REJECTED (exceeds voting power)"] = int(bad.sum())
        computed = computed.where(~bad)

    by_key = pd.Series(computed.to_numpy(), index=frame_column(merged, "_key").to_numpy())
    filled = by_key.reindex(frame_column(left, "_key").to_numpy())
    filled.index = left.index

    out = left.drop(columns=["_as_of", "_key"])
    disclosed = cast(pd.Series, pd.to_numeric(frame_column(out, "insider_ownership_pct"), errors="coerce"))

    # Filed evidence wins; the computed percentage only fills a missing disclosure.
    fills = filled.notna() & disclosed.isna()
    n = int(fills.sum())
    if n:
        out["insider_ownership_pct"] = disclosed.where(~fills, filled)
        if tally is not None:
            tally["insider_ownership_pct: COMPUTED (no disclosed combined percentage)"] = n
    if tally is not None:
        kept = int((filled.notna() & disclosed.notna()).sum())
        if kept:
            tally["insider_ownership_pct: disclosed value PREFERRED over the computed one"] = kept
    return out


def repair_ownership_basis(def14a_hist: pd.DataFrame, tally: dict[str, int] | None = None) -> pd.DataFrame:
    """Blank dual-class ownership percentages that use a per-class denominator."""
    need = {"insider_ownership_pct", "insider_voting_pct", "dual_class_shares"}
    if not need.issubset(def14a_hist.columns):
        return def14a_hist

    out = def14a_hist.copy()
    dual_values = cast(pd.Series, pd.to_numeric(frame_column(out, "dual_class_shares"), errors="coerce"))
    dual = dual_values > 0
    if "ticker" in out.columns:
        # One missed flag can accompany the bad value, so corroborate structure across history.
        dual = dual | out["ticker"].map(dual.groupby(out["ticker"]).max()).fillna(False).astype(bool)
    vote = pd.to_numeric(out["insider_voting_pct"], errors="coerce")
    own = pd.to_numeric(out["insider_ownership_pct"], errors="coerce")
    per_class = dual & vote.notna() & own.notna() & (own > vote + 1e-9)

    n = int(per_class.sum())
    if n:
        out.loc[per_class, "insider_ownership_pct"] = pd.NA
        if "ceo_ownership_pct" in out.columns:
            # the CEO leg was read off the SAME table and the same columns, so it inherits the
            # basis error on those filings whether or not its own pair is comparable
            out.loc[per_class, "ceo_ownership_pct"] = pd.NA
        if tally is not None:
            tally["insider_ownership_pct: blanked (per-class basis, own > vote)"] = n
            tally["insider_ownership_pct: blanked (per-class basis) — tickers"] = (
                int(out.loc[per_class, "ticker"].nunique()) if "ticker" in out.columns else 0
            )
    return out


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
    return _expire(daily, sub[["ticker", "as_of", "control_wedge"]], "control_wedge", "control_wedge", tally)


def _def14a_raw_fields(def14a_hist: pd.DataFrame, idx: pd.DatetimeIndex, tally: dict[str, int] | None = None) -> dict[str, pd.DataFrame]:
    """Build raw-only DEF 14A fields."""
    out: dict[str, pd.DataFrame] = {}
    for src, name in _RAW_DEF14A_FIELDS:
        f = _gate(fundamentals_to_daily(def14a_hist, src, idx), src, tally, _domain_condition(def14a_hist, src, idx, tally))
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
        f = _gate(fundamentals_to_daily(def14a_hist, src, idx), src, tally, _domain_condition(def14a_hist, src, idx, tally))
        f = _expire(f, def14a_hist, src, name, tally)
        if not f.empty and f.notna().any().any():
            f_dict[name] = f

    # Tenure accrues daily but still expires from the filing that disclosed `ceo_since_year`.
    since = fundamentals_to_daily(def14a_hist, "ceo_since_year", idx)
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
            rev_growth = fiscal_change_to_daily(fundamentals, "totalRevenue", idx, kind="pct", periods=infer_yoy_periods(fundamentals))
            if not rev_growth.empty and rev_growth.notna().any().any():
                cols = pay_growth.columns.intersection(rev_growth.columns)
                if len(cols) > 0:
                    # Pay growth is already expired; subtraction preserves that mask.
                    f_dict["ceo_pay_vs_revenue_growth"] = pay_growth[cols] - rev_growth[cols]

    # Voting power minus economic ownership has a meaningful zero and is already bounded.
    wedge = _control_wedge(def14a_hist, idx, tally)
    if not wedge.empty and wedge.notna().any().any():
        f_dict["control_wedge"] = wedge

    # Raw percentage changes need their own cross-sectional tail guard.
    for name in _RAW_ONLY_COMPUTED & f_dict.keys():
        f_dict[name] = winsorize_xs(f_dict[name])
    return f_dict


def _stack(
    fields: dict[str, pd.DataFrame],
    suffix: str,
    availability: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Stack a {name: daily wide frame} dict to the long panel as `f_<name><suffix>`."""
    long_frames = []
    for name, fdf in fields.items():
        if fdf is None or fdf.empty:
            continue
        fdf = mask_to_availability(fdf.apply(pd.to_numeric, errors="coerce"), availability)
        if not fdf.notna().any().any():
            continue
        s = fdf.stack().astype("float32")
        s.index.set_names(PANEL_KEYS, inplace=True)
        long_frames.append(s.rename(f"f_{name}{suffix}"))
    if not long_frames:
        return pd.DataFrame(columns=PANEL_KEYS)
    return pd.concat(long_frames, axis=1).dropna(how="all").copy().reset_index()


def _peer_only(
    fields: dict[str, pd.DataFrame],
    peer_dict: dict,
    availability: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build peer z-scores without the cross-sectional percentile twin."""
    rel: dict[str, pd.DataFrame] = {}
    for name, fdf in fields.items():
        if fdf is None or fdf.empty:
            continue
        fdf = mask_to_availability(fdf.apply(pd.to_numeric, errors="coerce"), availability)
        if not fdf.notna().any().any():
            continue
        rel[name] = winsorize_xs(peer_relative(fdf, peer_dict))
    return _stack(rel, "_vs_peers", availability)


def build_governance_feature_panel(
    def14a_history: pd.DataFrame | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
    fundamentals_history: pd.DataFrame | None = None,
    votes: pd.DataFrame | None = None,
    exec_comp: pd.DataFrame | None = None,
    directors: pd.DataFrame | None = None,
    director_comp: pd.DataFrame | None = None,
    close_total: pd.DataFrame | None = None,
    availability: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build the long governance panel and per-family data-quality tallies."""
    # `_gate` writes before any feature family, so the tally is owned here.
    tally: dict[str, int] = {}
    legacy_peers: dict[str, pd.DataFrame] = {}
    legacy_hist: dict[str, pd.DataFrame] = {}
    raw: dict[str, pd.DataFrame] = {}
    if def14a_history is not None and not def14a_history.empty and "as_of" in def14a_history.columns:
        # Blank per-class ownership before computed economic ownership fills the gap.
        def14a_history = repair_ownership_basis(def14a_history, tally)
        def14a_history = economic_ownership(def14a_history, fundamentals_history, tally)
        computed = _governance_fields(def14a_history, trading_index, fundamentals_history, tally)
        computed.update(_def14a_raw_fields(def14a_history, trading_index, tally))
        # Raw is canonical; selected encodings are additive.
        raw.update(computed)
        legacy_peers = {k: v for k, v in computed.items() if k in _PEER_LEGACY}
        legacy_hist = {k: v for k, v in computed.items() if k in _VS_HIST_LEGACY}

    # Executive-pay families own their child-table and return-series inputs.
    pay_frames, pay_tally = pay_fields(def14a_history, exec_comp, fundamentals_history, close_total, peer_dict, trading_index, availability)
    raw.update(pay_frames)
    pay_peers = {k: v for k, v in pay_frames.items() if k in PAY_PEER_FIELDS}

    # Provision changes are differenced in filing space before daily expansion.
    prov_frames, prov_tally = provision_fields(def14a_history, trading_index)
    raw.update(prov_frames)
    prov_peers = {k: v for k, v in prov_frames.items() if k in PROVISION_PEER_FIELDS}

    # Per-person director sources supply quality, turnover, and compensation families.
    quality_frames, quality_tally = board_quality_fields(directors, trading_index)
    raw.update(quality_frames)
    quality_peers = {k: v for k, v in quality_frames.items() if k in BOARD_QUALITY_PEER_FIELDS}

    dpay_frames, dpay_tally = director_pay_fields(director_comp, def14a_history, trading_index)
    raw.update(dpay_frames)
    dpay_peers = {k: v for k, v in dpay_frames.items() if k in DIRECTOR_PAY_PEER_FIELDS}

    vote_fields, vote_tally = vote_dissent_fields(votes, trading_index)
    tally.update(vote_tally)
    tally.update(pay_tally)
    tally.update(prov_tally)
    tally.update(quality_tally)
    tally.update(dpay_tally)

    # Vote fields ship raw; selected bounded levels also get a peer leg.
    raw.update(vote_fields)
    peers_only = {k: v for k, v in vote_fields.items() if k in PEER_RELATIVE_FIELDS}

    parts = [
        _peer_only({**legacy_peers, **pay_peers, **prov_peers, **quality_peers, **dpay_peers, **peers_only}, peer_dict, availability),
        _stack({k: self_history_z(mask_to_availability(v, availability)) for k, v in legacy_hist.items()}, "_vs_hist", availability),
        _stack(raw, "", availability),
    ]
    parts = [p for p in parts if not p.empty and len(p.columns) > 2]
    if not parts:
        return pd.DataFrame(columns=PANEL_KEYS), tally
    out = parts[0]
    for nxt in parts[1:]:
        out = out.merge(nxt, on=PANEL_KEYS, how="outer")
    return out, tally
