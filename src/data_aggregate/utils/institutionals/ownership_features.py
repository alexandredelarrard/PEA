"""Schedule 13D activist and Schedule 13G beneficial-owner event features.

Filings are canonicalized to one event per ``(ticker, accession_number, cusip)`` before event
features are built. ``filer_id`` uses the smallest reporting-person CIK, falling back to the
name, so amendments and transitions can be followed point in time. Joint-filer membership
resolution remains future data work.

The normalized holder-count proxy and four ``percent_of_class`` features were removed because
they lack a stable, source-complete denominator and
their source field is effectively unavailable before the December 2024 XML mandate and is too
recent for train/test/validation use; restoration requirements are recorded in ``wiki/TODO.md``.
"""

from __future__ import annotations

import pandas as pd

from src.data_aggregate.utils.common.errors import _empty_panel
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.common.price_frames import PriceFrames
from src.data_aggregate.utils.institutionals.availability import InstitutionalAvailability
from src.data_aggregate.utils.institutionals.decay import decay_events
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

EMISSION: dict[str, str] = {
    "ic_act_initial_13d": "raw",  # decayed occurrence; 0.3% ties, no drift
    "ic_act_amendment_intensity": "raw",  # 0.2% ties
    "ic_act_repeat_activist": "raw",  # 2.2% ties, 36 tickers ever
    "ic_bo_new_holder": "raw",  # 2.9% ties
    "ic_bo_escalation_13g_to_13d": "raw",
    "ic_bo_de_escalation_13d_to_13g": "raw",  # 23.4% ties
}


def _canonicalize(df: pd.DataFrame | None, has_amendment: bool = True) -> pd.DataFrame:
    """One row per `(ticker, accession_number, cusip)` -- see module docstring. `filer_id` is
    the group's identity for time-series tracking; `n_reporting_persons` is the co-filer count
    kept SEPARATE from any ownership number, exactly so nothing downstream is tempted to fold
    it back in."""
    cols = ["ticker", "accession_number", "cusip", "filing_date", "filer_id", "n_reporting_persons"]
    if has_amendment:
        cols.append("is_amendment")
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
        "filer_id": ("_filer_key", lambda s: s.dropna().sort_values().iloc[0] if s.notna().any() else None),
        "n_reporting_persons": ("_filer_key", "nunique"),
    }
    if has_amendment:
        agg_map["is_amendment"] = ("is_amendment", "first")
    return d.groupby(key, sort=False).agg(**agg_map).reset_index()


def _act_fields(canon: pd.DataFrame, idx: pd.DatetimeIndex, halflife: float) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if canon.empty:
        return out
    is_amend = canon["is_amendment"].fillna(0).astype(float).eq(1.0)
    initial, amend = canon[~is_amend], canon[is_amend]

    out["ic_act_initial_13d"] = decay_events(initial, idx, halflife, date_col="filing_date")
    out["ic_act_amendment_intensity"] = decay_events(amend, idx, halflife, date_col="filing_date")

    initial_sorted = initial.dropna(subset=["filer_id"]).sort_values("filing_date")
    prior_campaigns = initial_sorted.groupby("filer_id").cumcount()
    out["ic_act_repeat_activist"] = decay_events(initial_sorted[prior_campaigns >= 3], idx, halflife, date_col="filing_date")

    return out


def _bo_fields(
    canon: pd.DataFrame,
    idx: pd.DatetimeIndex,
    halflife: float,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if canon.empty:
        return out
    ce = canon.dropna(subset=["filer_id"])
    if ce.empty:
        return out

    first_holder = ce.sort_values("filing_date").drop_duplicates(subset=["ticker", "filer_id"], keep="first")
    out["ic_bo_new_holder"] = decay_events(first_holder, idx, halflife, date_col="filing_date")

    return out


def _complete_source_mask(
    frames: PriceFrames,
    idx: pd.DatetimeIndex,
    *,
    source_start: pd.Timestamp,
    complete_through: pd.Timestamp | None,
) -> pd.DataFrame | None:
    """Cells for which source absence is proven rather than merely unobserved."""
    if complete_through is None or pd.isna(complete_through):
        return None
    columns = pd.Index(sorted(map(str, frames.universe)), name="ticker")
    listed = (
        frames.close_split.reindex(index=idx, columns=columns).notna()
        if frames.close_split is not None and not frames.close_split.empty
        else pd.DataFrame(True, index=idx, columns=columns)
    )
    return InstitutionalAvailability.combine(
        InstitutionalAvailability.date_mask(idx, columns, source_start),
        InstitutionalAvailability.through_mask(idx, columns, pd.Timestamp(complete_through)),
        listed,
    )


def _known_13g_identity_mask(canon: pd.DataFrame, idx: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """A holder state is unknowable after a filing whose holder has no usable identity."""
    mask = pd.DataFrame(True, index=idx, columns=columns)
    unknown = canon[canon["filer_id"].isna()]
    for ticker, rows in unknown.groupby("ticker"):
        if ticker in mask.columns:
            first = pd.to_datetime(rows["filing_date"], errors="coerce").min()
            if pd.notna(first):
                mask.loc[idx >= pd.Timestamp(first).normalize(), ticker] = False
    return mask


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
    # dereferences unconditionally; `is_amendment` is NOT in it because it is 13D-only and
    # `_canonicalize` already builds its column list around its absence.
    sec_13d = None if _absent(sec_13d, _NEED) else sec_13d
    sec_13g = None if _absent(sec_13g, _NEED) else sec_13g
    if sec_13d is None and sec_13g is None:
        return _empty_panel()

    idx = pd.DatetimeIndex(trading_index).normalize().unique().sort_values()
    if idx.empty:
        return pd.DataFrame(columns=["date", "ticker"])

    canon_13d = _canonicalize(sec_13d, has_amendment=True)
    canon_13g = _canonicalize(sec_13g, has_amendment=False)
    if canon_13d.empty and canon_13g.empty:
        return pd.DataFrame(columns=["date", "ticker"])

    if availability is not None:
        g_start = availability.source_date(Tables.sec_13g)
    else:
        g_start = pd.to_datetime(canon_13g["filing_date"], errors="coerce").min()
    columns = pd.Index(sorted(map(str, frames.universe)), name="ticker")
    d_start = pd.to_datetime(canon_13d["filing_date"], errors="coerce").min()
    act_coverage = (
        _complete_source_mask(
            frames,
            idx,
            source_start=availability.source_date(Tables.sec_13d) if availability is not None else pd.Timestamp(d_start),
            complete_through=complete_through_13d,
        )
        if pd.notna(d_start)
        else None
    )
    bo_source_coverage = (
        _complete_source_mask(
            frames,
            idx,
            source_start=pd.Timestamp(g_start),
            complete_through=complete_through_13g,
        )
        if pd.notna(g_start)
        else None
    )
    bo_identity = _known_13g_identity_mask(canon_13g, idx, columns)
    bo_mask = InstitutionalAvailability.combine(bo_source_coverage, bo_identity) if bo_source_coverage is not None else bo_identity
    cross_complete = act_coverage is not None and bo_source_coverage is not None

    fields: dict[str, pd.DataFrame] = {}
    fields.update(_act_fields(canon_13d, idx, decay_halflife_act))
    fields.update(_bo_fields(canon_13g, idx, decay_halflife_bo))
    fields.update(_cross_fields(canon_13d, canon_13g, idx, decay_halflife_bo))

    cross_names = {"ic_bo_escalation_13g_to_13d", "ic_bo_de_escalation_13d_to_13g"}
    cross_masks: dict[str, pd.DataFrame] = {}
    for name in cross_names & fields.keys():
        observed = fields[name].reindex(index=idx, columns=columns).notna()
        # An unidentified holder prevents an absence/zero claim, but cannot erase a later
        # transition proven by the same known filer in both schedules. Source frontiers still
        # bound that positive state independently, including when only one frontier is known.
        cross_masks[name] = InstitutionalAvailability.combine(
            bo_identity | observed,
            *(mask for mask in (act_coverage, bo_source_coverage) if mask is not None),
        )
    for name, frame in fields.items():
        mask = act_coverage if name.startswith("ic_act_") else cross_masks[name] if name in cross_names else bo_mask
        if mask is not None:
            fields[name] = frame.reindex(index=idx, columns=columns).where(mask)

    for name in list(fields):
        if fields[name] is None or fields[name].empty:
            fields.pop(name)
    if not fields:
        return pd.DataFrame(columns=["date", "ticker"])

    if sink is not None and not canon_13d.empty:
        sink.set_frontier("act", complete_through_13d)
        observed_13d = canon_13d
        if complete_through_13d is not None and pd.notna(complete_through_13d):
            observed_13d = observed_13d[observed_13d["filing_date"] <= pd.Timestamp(complete_through_13d)]
        sink.add_events("act", observed_13d[["ticker", "filing_date"]].rename(columns={"filing_date": "date"}).drop_duplicates())
    if sink is not None:
        signal_fields = dict(fields)
        signal_masks: dict[str, pd.DataFrame] = {}
        if "ic_act_initial_13d" in fields:
            raw = fields["ic_act_initial_13d"].reindex(index=idx, columns=columns)
            mask = act_coverage if act_coverage is not None else raw.notna()
            signal_masks["ic_act_initial_13d"] = mask
            signal_fields["ic_act_initial_13d"] = raw.fillna(0.0).where(mask) if act_coverage is not None else raw
        if "ic_bo_escalation_13g_to_13d" in fields:
            raw = fields["ic_bo_escalation_13g_to_13d"].reindex(index=idx, columns=columns)
            mask = cross_masks["ic_bo_escalation_13g_to_13d"] if cross_complete else raw.notna()
            signal_masks["ic_bo_escalation_13g_to_13d"] = mask
            signal_fields["ic_bo_escalation_13g_to_13d"] = raw.fillna(0.0).where(mask) if cross_complete else raw
        sink.keep_signals(signal_fields, signal_masks)

    emission = {name: EMISSION[name] for name in fields}
    return build_peer_relative_panel(fields, peer_dict, emission=emission, availability=frames.availability)
