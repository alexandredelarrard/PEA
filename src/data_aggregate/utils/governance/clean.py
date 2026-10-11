"""The single governance cleaning stage: every fill, carry, repair and reconciliation of the
raw proxy inputs runs here, in a fixed order, before any feature is built."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

import pandas as pd

from src.data_aggregate.utils.common.typing import frame_column
from src.data_aggregate.utils.governance.def14a_impute import (
    FILL_CROSS_SOURCE,
    ensure_provenance,
    fill_kind_column,
    impute_def14a,
    impute_exec_comp,
    write_with_provenance,
)
from src.data_aggregate.utils.governance.director_comp import impute_director_comp
from src.data_aggregate.utils.governance.directors import (
    DERIVED_AGGREGATES,
    SOURCE_DERIVED,
    board_aggregates,
    fill_director_attributes,
    finalize_board_source,
    merge_board_aggregates,
)

_SHARES_OUTSTANDING = "sharesOutstandingPit"


@dataclass(frozen=True)
class GovernanceInputs:
    """The governance source frames; `cleaned` is True once the stage has run on them."""

    def14a: pd.DataFrame | None
    exec_comp: pd.DataFrame | None = None
    directors: pd.DataFrame | None = None
    director_comp: pd.DataFrame | None = None
    votes: pd.DataFrame | None = None
    ownership: pd.DataFrame | None = None
    cleaned: bool = False


def _present(df: pd.DataFrame | None) -> bool:
    return df is not None and not df.empty


def _record(stats: dict[str, int], stage: str, part: dict[str, int]) -> None:
    stats.update({f"{stage}|{k}": int(v) for k, v in part.items()})


def clean_governance_inputs(raw: GovernanceInputs, fundamentals: pd.DataFrame | None) -> tuple[GovernanceInputs, dict[str, int]]:
    """Clean the raw governance inputs. Returns `(cleaned bundle, stats keyed "<stage>|<rule>")`.

    Order: child fills, board merge, `impute_def14a`, board provenance, ownership basis repair,
    economic ownership. A bundle already cleaned is returned unchanged with empty stats.
    """
    if raw.cleaned:
        return raw, {}
    stats: dict[str, int] = {}

    exec_comp, directors, director_comp = raw.exec_comp, raw.directors, raw.director_comp
    if _present(exec_comp):
        exec_comp, part = impute_exec_comp(cast(pd.DataFrame, exec_comp))
        _record(stats, "exec_comp", part)
    if _present(directors):
        directors, part = fill_director_attributes(cast(pd.DataFrame, directors))
        _record(stats, "directors", part)
    if _present(director_comp):
        director_comp, part = impute_director_comp(cast(pd.DataFrame, director_comp))
        _record(stats, "director_comp", part)

    df = raw.def14a
    if _present(df):
        frame = cast(pd.DataFrame, df)
        if _present(directors):
            frame, part = merge_board_aggregates(frame, board_aggregates(cast(pd.DataFrame, directors)))
            _record(stats, "board_merge", part)
            _board_fill_kinds(frame)
        frame, part = impute_def14a(frame)
        _record(stats, "impute_def14a", part)
        frame, part = finalize_board_source(frame)
        _record(stats, "board_source", part)
        if "as_of" in frame.columns:
            # Blank per-class ownership before computed economic ownership fills the gap.
            tally: dict[str, int] = {}
            frame = repair_ownership_basis(frame, tally)
            frame = economic_ownership(frame, fundamentals, tally)
            _record(stats, "ownership", tally)
        df = frame

    cleaned = replace(raw, def14a=df, exec_comp=exec_comp, directors=directors, director_comp=director_comp, cleaned=True)
    return cleaned, stats


def _board_fill_kinds(frame: pd.DataFrame) -> None:
    """Board averages derived from the director rows are a cross-source fill."""
    for field in DERIVED_AGGREGATES:
        if field in frame.columns and f"{field}_source" in frame.columns:
            ensure_provenance(frame, field)
            frame.loc[frame[f"{field}_source"] == SOURCE_DERIVED, fill_kind_column(field)] = FILL_CROSS_SOURCE


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
        out["insider_ownership_pct"] = disclosed
        # dated at the proxy that disclosed the share count
        write_with_provenance(out, fills, "insider_ownership_pct", filled, FILL_CROSS_SOURCE, pd.to_datetime(out["as_of"], errors="coerce"))
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
        write_with_provenance(out, per_class, "insider_ownership_pct", None, None, None)
        if "ceo_ownership_pct" in out.columns:
            # the CEO leg was read off the SAME table and the same columns, so it inherits the
            # basis error on those filings whether or not its own pair is comparable
            write_with_provenance(out, per_class, "ceo_ownership_pct", None, None, None)
        if tally is not None:
            tally["insider_ownership_pct: blanked (per-class basis, own > vote)"] = n
            tally["insider_ownership_pct: blanked (per-class basis) — tickers"] = (
                int(out.loc[per_class, "ticker"].nunique()) if "ticker" in out.columns else 0
            )
    return out
