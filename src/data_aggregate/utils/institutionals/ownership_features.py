"""Schedule 13D activist and Schedule 13G beneficial-owner event features.

Filings are canonicalized to one event per ``(ticker, accession_number, cusip)`` before event
features are built. ``filer_id`` uses the smallest reporting-person CIK, falling back to the
name, so amendments and transitions can be followed point in time. Joint-filer membership
resolution remains future data work.

``ic_bo_holder_count`` uses the same trailing activity window in numerator and denominator;
without explicit exits, a filer lapses after ``HOLDER_ACTIVE_DAYS``. Item 4 categories use
deterministic keyword matches. The four ``percent_of_class`` features were removed because
their source field is effectively unavailable before the December 2024 XML mandate and is too
recent for train/test/validation use; restoration requirements are recorded in ``wiki/TODO.md``.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.errors import _empty_panel
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.common.price_frames import PriceFrames
from src.data_aggregate.utils.institutionals.availability import InstitutionalAvailability
from src.data_aggregate.utils.institutionals.decay import days_since_last_true, decay_events, snap_to_grid
from src.data_store.schema import Tables

#: The columns `_canonicalize` reads off EITHER schedule without checking first. The
#: canonical-event key (`ticker`, `accession_number`, `cusip`) plus the only legal stamp
#: (`filing_date` -- `date_of_event` is never projected, which is what makes L4 structural)
#: plus the two ownership/identity legs.
_NEED = {"ticker", "accession_number", "cusip", "filing_date", "reporting_person_cik"}


def _absent(df: pd.DataFrame | None, need: set[str] | None = None) -> bool:
    """True when `df` cannot be built from: missing, empty, or short a required column.

    The three-part test is the D5 entry contract stated once. `need` is the set the builder
    dereferences unconditionally -- a column it only uses `if present` does NOT belong here,
    or an optional projection turns into an empty panel.
    """
    if df is None or df.empty:
        return True
    return bool(need) and not need.issubset(df.columns)


#: D28: 13G filing volume is not constant over 15 years, so the raw filer count is meaningless
#: on its own -- see the module docstring.
ACT_HALFLIFE_DEFAULT = 126.0
BO_HALFLIFE_DEFAULT = 126.0

#: How long a 13G filing keeps its filer counted as a holder, in TRADING days (~18 months).
#: The schema carries no "no longer holds >5%" signal, so activity has to stand in for
#: holding, and Rule 13d-2(b) sets the cadence it stands on: a 13G holder owed an annual
#: amendment within 45 days of year-end (quarterly since the 2024 amendments), so a still-live
#: >5% position refiles at least yearly and 18 months tolerates one late or skipped cycle.
#: Both sides of `holder_count` use this ONE window -- an ever-accumulating numerator over a
#: recent-flow denominator is not a share of anything, and measured that way the "share"
#: reached 1.23.
HOLDER_ACTIVE_DAYS = 378

_BOARD_RE = re.compile(r"board seat|board representation|board of directors|nominat|director representation", re.IGNORECASE)
_STRATEGIC_RE = re.compile(
    r"strategic alternative|strategic review|sale of the (?:issuer|company)|" r"business combination|merger|explore.{0,20}alternative", re.IGNORECASE
)

EMISSION: dict[str, str] = {
    "ic_act_initial_13d": "raw",  # decayed occurrence; 0.3% ties, no drift
    "ic_act_amendment_intensity": "raw",  # 0.2% ties
    "ic_act_campaign_age_days": "raw+xs",  # 2.9% ties AND a real drift -- see below
    "ic_act_repeat_activist": "raw",  # 2.2% ties, 36 tickers ever
    "ic_act_purpose_board": "raw",  # 0.0% ties
    "ic_act_purpose_strategic": "raw",  # 0.1% ties
    "ic_bo_holder_count": "raw",  # D28-normalized; 70.3% ties -> no xs leg
    "ic_bo_new_holder": "raw",  # 2.9% ties
    "ic_bo_escalation_13g_to_13d": "raw",
    "ic_bo_de_escalation_13d_to_13g": "raw",  # 23.4% ties
}

#: ⚠ `ic_act_campaign_age_days` IS THE ONE FEATURE THAT TAKES AN `_xs` LEG ON DRIFT RATHER THAN
#: ON ITS TIE FRACTION, and it is the same argument the plan flags for insider's
#: `days_since_last_buy`. Measured on the last grid day: 257 names, MEDIAN 3,879 trading days
#: and a max of 7,800 -- i.e. the typical "campaign age" is 15 years, because the feature keeps
#: counting long after the campaign ended and the count is mechanically small early in the
#: sample only because the history is short. The raw leg stays (a day count is a quantity in
#: its own units), but the within-date percentile is what makes 2003 comparable to 2026.


def _canonicalize(df: pd.DataFrame, text_col: str | None = None, has_amendment: bool = True) -> pd.DataFrame:
    """One row per `(ticker, accession_number, cusip)` -- see module docstring. `filer_id` is
    the group's identity for time-series tracking; `n_reporting_persons` is the co-filer count
    kept SEPARATE from any ownership number, exactly so nothing downstream is tempted to fold
    it back in."""
    cols = ["ticker", "accession_number", "cusip", "filing_date", "filer_id", "n_reporting_persons"]
    if has_amendment:
        cols.append("is_amendment")
    if text_col:
        cols.append("text")
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    d = df.copy()
    d["filing_date"] = pd.to_datetime(d["filing_date"], errors="coerce")
    d = d.dropna(subset=["ticker", "accession_number", "filing_date"])
    if d.empty:
        return pd.DataFrame(columns=cols)
    d["cusip"] = d["cusip"].fillna("") if "cusip" in d.columns else ""
    cik = d["reporting_person_cik"] if "reporting_person_cik" in d.columns else pd.Series(index=d.index, dtype=object)
    name = d["reporting_person_name"] if "reporting_person_name" in d.columns else pd.Series(index=d.index, dtype=object)
    cik = cik.astype(object).where(cik.notna() & (cik.astype(str).str.len() > 0), None)
    d["_filer_key"] = cik.where(cik.notna(), name)

    key = ["ticker", "accession_number", "cusip"]
    agg_map = {
        "filing_date": ("filing_date", "first"),
        "filer_id": ("_filer_key", lambda s: (s.dropna().sort_values().iloc[0] if s.notna().any() else None)),
        "n_reporting_persons": ("_filer_key", "nunique"),
    }
    if has_amendment:
        agg_map["is_amendment"] = ("is_amendment", "first")
    if text_col and text_col in d.columns:
        agg_map["text"] = (text_col, lambda s: next((x for x in s if isinstance(x, str) and x.strip()), None))
    return d.groupby(key, sort=False).agg(**agg_map).reset_index()


def _snap_to_grid(dates: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Snap each date onto the first trading day >= it -- see `decay.snap_to_grid`, which now
    owns the rule because the conditioning layer needs the same one."""
    return snap_to_grid(dates, idx)


def _days_since(bool_wide: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Trading days since the last True per column -- see `decay.days_since_last_true`."""
    return days_since_last_true(bool_wide.reindex(idx))


def _rolling_distinct(events: pd.DataFrame, idx: pd.DatetimeIndex, window: int) -> pd.Series:
    """Distinct `filer_id` with an event in the trailing `window` TRADING DAYS, as of each
    grid day.

    THIS IS THE D28 DENOMINATOR AND IT HAS TO BE POINT-IN-TIME. A per-CALENDAR-YEAR distinct
    count is not: the newest year is always PARTIAL, so it reads low (measured 2026-09-08: 102
    filers year-to-date against 140 for all of 2025 and 187 for 2024) and every value computed
    inside that year is revised downward as the year fills in -- an inflated `holder_count`
    exactly where the predictions are made. A trailing window is stable the day it is computed
    and never revised. An exact rolling distinct count, not a decayed one (a decayed distinct
    count is not a distinct count).

    It shares `window` with the numerator's activity window on purpose: a ratio of an
    ever-accumulating numerator to a recent-flow denominator is not a share of anything, and
    measured that way `holder_count` reached **1.23**."""
    ev = events.dropna(subset=["filing_date", "filer_id"])
    if ev.empty or len(idx) == 0:
        return pd.Series(0.0, index=idx)
    ev = ev.assign(_grid_date=_snap_to_grid(ev["filing_date"], idx)).dropna(subset=["_grid_date"])
    if ev.empty:
        return pd.Series(0.0, index=idx)
    # Work in GRID POSITIONS, so one constant in trading days drives both sides of the ratio.
    pos = idx.get_indexer(pd.DatetimeIndex(ev["_grid_date"]))
    order = np.argsort(pos, kind="stable")
    pos, filers = pos[order], ev["filer_id"].astype(str).to_numpy()[order]
    active: Counter = Counter()
    out = np.zeros(len(idx), dtype="float64")
    enter = leave = 0
    for k in range(len(idx)):
        while enter < len(pos) and pos[enter] <= k:
            active[filers[enter]] += 1
            enter += 1
        while leave < len(pos) and pos[leave] <= k - window:
            key = filers[leave]
            active[key] -= 1
            if active[key] <= 0:
                del active[key]
            leave += 1
        out[k] = len(active)
    return pd.Series(out, index=idx)


def _sum_over_filers(wide: pd.DataFrame) -> pd.DataFrame:
    """Collapse a `(ticker, filer_id)`-columned wide frame to `ticker`, NaN-preserving: a
    ticker with no active filer at all stays NaN (`min_count=1`) rather than reading as a
    measured zero."""
    return wide.T.groupby(level="ticker").sum(min_count=1).T


def _act_fields(canon: pd.DataFrame, idx: pd.DatetimeIndex, halflife: float) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if canon.empty:
        return out
    is_amend = canon["is_amendment"].fillna(0).astype(float).eq(1.0)
    initial, amend = canon[~is_amend], canon[is_amend]

    out["ic_act_initial_13d"] = decay_events(initial, idx, halflife, date_col="filing_date")
    out["ic_act_amendment_intensity"] = decay_events(amend, idx, halflife, date_col="filing_date")

    occ = initial.assign(_flag=1.0, _grid_date=_snap_to_grid(initial["filing_date"], idx))
    occ = occ.dropna(subset=["_grid_date"])
    wide = occ.pivot_table(index="_grid_date", columns="ticker", values="_flag", aggfunc="max")
    out["ic_act_campaign_age_days"] = _days_since(wide.reindex(idx).notna(), idx)

    initial_sorted = initial.dropna(subset=["filer_id"]).sort_values("filing_date")
    prior_campaigns = initial_sorted.groupby("filer_id").cumcount()
    out["ic_act_repeat_activist"] = decay_events(initial_sorted[prior_campaigns >= 3], idx, halflife, date_col="filing_date")

    if "text" in canon.columns:
        has_text = canon["text"].notna()
        board = canon[has_text & canon["text"].str.contains(_BOARD_RE, na=False)]
        strat = canon[has_text & canon["text"].str.contains(_STRATEGIC_RE, na=False)]
        out["ic_act_purpose_board"] = decay_events(board, idx, halflife, date_col="filing_date")
        out["ic_act_purpose_strategic"] = decay_events(strat, idx, halflife, date_col="filing_date")

    return out


def _bo_fields(
    canon: pd.DataFrame,
    idx: pd.DatetimeIndex,
    halflife: float,
    *,
    coverage: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if canon.empty:
        return out
    ce = canon.dropna(subset=["filer_id"])
    if ce.empty:
        return out

    occ = ce.assign(_flag=1.0, _grid_date=_snap_to_grid(ce["filing_date"], idx))
    occ = occ.dropna(subset=["_grid_date"])
    occ_wide = occ.pivot_table(index="_grid_date", columns=["ticker", "filer_id"], values="_flag", aggfunc="max")
    # `limit=` is the whole exit policy: a filer counts as a holder for HOLDER_ACTIVE_DAYS
    # after each filing, then lapses. See the constant.
    occ_wide = occ_wide.reindex(idx).ffill(limit=HOLDER_ACTIVE_DAYS)
    active_count = _sum_over_filers(occ_wide)
    if coverage is not None:
        active_count = active_count.reindex(index=idx, columns=coverage.columns).fillna(0.0).where(coverage)

    denom = _rolling_distinct(ce, idx, window=HOLDER_ACTIVE_DAYS)
    out["ic_bo_holder_count"] = active_count.divide(denom.where(denom > 0), axis=0)

    first_holder = ce.sort_values("filing_date").drop_duplicates(subset=["ticker", "filer_id"], keep="first")
    out["ic_bo_new_holder"] = decay_events(first_holder, idx, halflife, date_col="filing_date")

    return out


def _complete_active_window_mask(
    frames: PriceFrames,
    idx: pd.DatetimeIndex,
    *,
    source_start: pd.Timestamp,
    complete_through: pd.Timestamp | None,
) -> pd.DataFrame | None:
    """Eligibility for interpreting no 13G filing as an observed zero holder numerator."""
    if complete_through is None or pd.isna(complete_through):
        return None
    columns = pd.Index(sorted(map(str, frames.universe)), name="ticker")
    if frames.close_split is not None and not frames.close_split.empty:
        listed = frames.close_split.reindex(index=idx, columns=columns).notna()
    else:
        listed = pd.DataFrame(True, index=idx, columns=columns)

    # The active-filer state looks back HOLDER_ACTIVE_DAYS trading sessions. Before that many
    # sessions have elapsed from the source floor, a missing filer could have filed just before
    # observable history began, so it is unavailable rather than zero.
    start_pos = int(idx.searchsorted(pd.Timestamp(source_start).normalize(), side="left"))
    first_complete = start_pos + HOLDER_ACTIVE_DAYS - 1
    history_complete = pd.Series(False, index=idx)
    if first_complete < len(idx):
        history_complete.iloc[first_complete:] = True
    through = pd.Series(idx <= pd.Timestamp(complete_through).normalize(), index=idx)
    temporal = pd.DataFrame(
        np.broadcast_to((history_complete & through).to_numpy()[:, None], (len(idx), len(columns))).copy(),
        index=idx,
        columns=columns,
    )
    return InstitutionalAvailability.combine(temporal, listed)


def _cross_fields(canon_13d: pd.DataFrame, canon_13g: pd.DataFrame, idx: pd.DatetimeIndex, halflife: float) -> dict[str, pd.DataFrame]:
    """13G<->13D escalation, keyed on each filer's FIRST-EVER filing of each type per ticker
    (an amendment does not re-trigger the transition)."""
    out: dict[str, pd.DataFrame] = {}
    if canon_13d.empty or canon_13g.empty:
        return out
    first_d = (
        canon_13d.dropna(subset=["filer_id"])
        .sort_values("filing_date")
        .drop_duplicates(subset=["ticker", "filer_id"], keep="first")[["ticker", "filer_id", "filing_date"]]
    )
    first_g = (
        canon_13g.dropna(subset=["filer_id"])
        .sort_values("filing_date")
        .drop_duplicates(subset=["ticker", "filer_id"], keep="first")[["ticker", "filer_id", "filing_date"]]
    )
    merged = first_d.merge(first_g, on=["ticker", "filer_id"], suffixes=("_d", "_g"))
    esc = merged[merged["filing_date_g"] < merged["filing_date_d"]].rename(columns={"filing_date_d": "filing_date"})[["ticker", "filing_date"]]
    deesc = merged[merged["filing_date_d"] < merged["filing_date_g"]].rename(columns={"filing_date_g": "filing_date"})[["ticker", "filing_date"]]
    out["ic_bo_escalation_13g_to_13d"] = decay_events(esc, idx, halflife, date_col="filing_date")
    out["ic_bo_de_escalation_13d_to_13g"] = decay_events(deesc, idx, halflife, date_col="filing_date")
    return out


def build_ownership_feature_panel(
    frames: PriceFrames,
    sec_13d: pd.DataFrame | None,
    sec_13g: pd.DataFrame | None,
    *,
    decay_halflife_act: float = ACT_HALFLIFE_DEFAULT,  # 6month default value
    decay_halflife_bo: float = BO_HALFLIFE_DEFAULT,
    availability: InstitutionalAvailability | None = None,
    complete_through_13d: pd.Timestamp | None = None,
    complete_through_13g: pd.Timestamp | None = None,
    sink=None,
) -> pd.DataFrame:
    """Long-format beneficial-ownership panel (`f_<name>` per `EMISSION`). Empty when neither
    source has usable rows.

    `sink` is the optional `ConditioningSink`. The `act` family hands it EVERY 13D filing as
    its event dates (an amendment restates a live campaign, so it is news and the conditioning
    clock should restart on it) but only the INITIAL filings as bullish acts, since an
    amendment can as easily disclose a sale.

    ⚠ `frames` RATHER THAN TWO UNPACKED FIELDS. `peer_dict` and `trading_index` were all read
    off one `PriceFrames` at the call site. Collapsing them is not about the basis here -- this
    builder reads no wide price frame -- but about arity: three of the old parameters were one
    object at every call site, and unpacking them at 39 of those is what let them drift apart.

    ⚠ NO `frames.require(...)`: this builder dereferences no optional wide frame at all.
    `trading_index` and `peers` are non-Optional fields of `PriceFrames`, so requiring them
    would assert something the type already guarantees.

    The non-frame arguments are KEYWORD-ONLY. A positional slip between two same-typed
    `pd.DataFrame | None` neighbours is a silent wrong-frame bug that reads as a plausible
    call; the keyword form makes it unrepresentable.
    """

    peer_dict = frames.peers
    trading_index = frames.trading_index

    # D5 entry guard, PER LEG. The two channels are independent fetchers -- a universe with
    # 13G coverage and no 13D still builds the `ic_bo_*` half -- so a leg that cannot be used
    # is nulled rather than failing the whole panel. `_NEED` is what `_canonicalize`
    # dereferences unconditionally; `is_amendment` and the Item 4 text are NOT in it, because
    # they are 13D-only and it already builds its column list around their absence.
    sec_13d = None if _absent(sec_13d, _NEED) else sec_13d
    sec_13g = None if _absent(sec_13g, _NEED) else sec_13g
    if sec_13d is None and sec_13g is None:
        return _empty_panel()

    idx = pd.DatetimeIndex(trading_index).normalize().unique().sort_values()
    if idx.empty:
        return pd.DataFrame(columns=["date", "ticker"])

    canon_13d = _canonicalize(sec_13d, text_col="item4_purpose_of_transaction", has_amendment=True)
    canon_13g = _canonicalize(sec_13g, has_amendment=False)
    if canon_13d.empty and canon_13g.empty:
        return pd.DataFrame(columns=["date", "ticker"])

    if availability is not None:
        g_start = availability.source_date(Tables.sec_13g)
    else:
        g_start = pd.to_datetime(canon_13g.get("filing_date"), errors="coerce").min()
    bo_coverage = (
        _complete_active_window_mask(
            frames,
            idx,
            source_start=pd.Timestamp(g_start),
            complete_through=complete_through_13g,
        )
        if pd.notna(g_start)
        else None
    )

    fields: dict[str, pd.DataFrame] = {}
    fields.update(_act_fields(canon_13d, idx, decay_halflife_act))
    fields.update(_bo_fields(canon_13g, idx, decay_halflife_bo, coverage=bo_coverage))
    fields.update(_cross_fields(canon_13d, canon_13g, idx, decay_halflife_bo))

    for name in list(fields):
        if fields[name] is None or fields[name].empty:
            fields.pop(name)
    if not fields:
        return pd.DataFrame(columns=["date", "ticker"])

    if sink is not None and not canon_13d.empty:
        sink.set_frontier("act", complete_through_13d)
        sink.add_events("act", canon_13d[["ticker", "filing_date"]].rename(columns={"filing_date": "date"}).drop_duplicates())
        initial = canon_13d[~canon_13d["is_amendment"].fillna(0).astype(float).eq(1.0)]
        sink.add_actors(
            "act",
            initial.dropna(subset=["filer_id"])
            .assign(actor=lambda d: d["filer_id"])[["ticker", "filing_date", "actor"]]
            .rename(columns={"filing_date": "date"}),
        )
    if sink is not None:
        columns = pd.Index(sorted(map(str, frames.universe)), name="ticker")
        if frames.close_split is not None and not frames.close_split.empty:
            listed = frames.close_split.reindex(index=idx, columns=columns).notna()
        else:
            listed = pd.DataFrame(True, index=idx, columns=columns)
        signal_fields = dict(fields)
        signal_masks: dict[str, pd.DataFrame] = {}
        if "ic_act_initial_13d" in fields:
            raw = fields["ic_act_initial_13d"].reindex(index=idx, columns=columns)
            mask = (
                availability.source_mask(
                    Tables.sec_13d,
                    idx,
                    columns,
                    requirements=(listed,),
                )
                if availability is not None
                else raw.notna()
            )
            signal_masks["ic_act_initial_13d"] = mask
            signal_fields["ic_act_initial_13d"] = raw.fillna(0.0).where(mask)
        if "ic_bo_escalation_13g_to_13d" in fields:
            raw = fields["ic_bo_escalation_13g_to_13d"].reindex(index=idx, columns=columns)
            if availability is not None:
                g_mask = InstitutionalAvailability.date_mask(
                    idx,
                    columns,
                    availability.source_date(Tables.sec_13g),
                )
                mask = availability.source_mask(
                    Tables.sec_13d,
                    idx,
                    columns,
                    requirements=(g_mask, listed),
                )
            else:
                mask = raw.notna()
            signal_masks["ic_bo_escalation_13g_to_13d"] = mask
            signal_fields["ic_bo_escalation_13g_to_13d"] = raw.fillna(0.0).where(mask)
        sink.keep_signals(signal_fields, signal_masks)

    emission = {k: v for k, v in EMISSION.items() if k in fields}
    return build_peer_relative_panel(fields, peer_dict, emission=emission)
