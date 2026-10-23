"""Point-in-time roster eligibility, concentration scores and existing manager weighting."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_aggregate.utils.sec_thirteen.superinvestor_roster import roster_as_of

logger = logging.getLogger(__name__)


#: The three concentration measures that agreed in sign when regressed on forward manager
#: excess return -- fewer positions, lower effective-N and higher top-10 weight all predict
#: better returns. Averaging their percentiles rather than picking one is deliberate: they
#: are 0.9+ correlated with each other, so the mean is a de-noised version of any single leg.
#:
#: ⚠ `turnover`, `survival` and `filing_lag_days` are DELIBERATELY ABSENT. All three are
#: sticky enough to be usable and all three came out with the OPPOSITE sign to the research
#: report's recommendation; until that contradiction is settled they are not safe in a score.
#: `filing_lag_days` is near-degenerate besides (25/50/75 percentiles 41/45/45 days, with a
#: 1,311-day back-filing tail that makes its rank IC untrustworthy).
_LEGS: tuple[tuple[str, bool], ...] = (
    ("n_positions", False),  # ascending=False -> fewest positions ranks highest
    ("eff_n", False),
    ("top10_weight", True),
)

#: Columns `manager_concentration_score` needs on `state` beyond the three legs.
_KEYS = ("cik", "period", "avail")


def roster_members_on(
    period: pd.Timestamp,
    *,
    history: pd.DataFrame | None,
    first: pd.Timestamp,
    config_dir: Path,
    cache: dict[pd.Timestamp, set[str]],
) -> set[str]:
    """Reuse the existing roster lookup at the first supported snapshot or later."""
    key = max(pd.Timestamp(period), first)
    if key not in cache:
        cache[key] = roster_as_of(history, key, config_dir=config_dir)
    return cache[key]


def select_managers(
    state: pd.DataFrame,
    *,
    roster_at: Callable[[pd.Timestamp], set[str]],
    mode: str,
    k: int,
    min_quarters: int,
    min_positions: int,
) -> pd.Series:
    """Apply the existing eligibility, concentration score and weighting policy."""
    eligible = eligibility(state, roster_at, min_quarters=min_quarters, min_positions=min_positions)
    df_scored = manager_concentration_score(state, eligible=eligible)
    weights = elite_weight(df_scored, mode=mode, k=k)
    df_diagnostics = selection_diagnostics(weights, state)
    if not df_diagnostics.empty:
        df_settled = df_diagnostics[df_diagnostics["n_public"] >= k]
        logger.info(
            "elite selection (%s, k=%s): %s of %s manager-quarters eligible, live set %s-%s managers, median churn %s per filing date",
            mode,
            k,
            int(eligible.sum()),
            len(eligible),
            int(df_settled["n_selected"].min()) if len(df_settled) else 0,
            int(df_settled["n_selected"].max()) if len(df_settled) else 0,
            int(df_settled["churn"].median()) if len(df_settled) else 0,
        )
    return weights


def _empty(name: str, dtype: str) -> pd.Series:
    idx = pd.MultiIndex.from_arrays([[], []], names=["cik", "period"])
    return pd.Series(dtype=dtype, index=idx, name=name)


def _latest_per_avail(state: pd.DataFrame) -> pd.DataFrame:
    """Keep each manager's highest public period when availability dates coincide."""
    return state.sort_values(["cik", "avail", "period"]).drop_duplicates(["cik", "avail"], keep="last")


def _wide(state: pd.DataFrame, column: str, grid: pd.DatetimeIndex, ciks: pd.Index) -> pd.DataFrame:
    """Carry each manager's public value across the availability grid."""
    return state.pivot(index="avail", columns="cik", values=column).reindex(index=grid, columns=ciks).ffill()


def manager_concentration_score(state: pd.DataFrame, eligible: pd.Series | None = None) -> pd.DataFrame:
    """Rank whole-book concentration across the then-public eligible manager states."""
    need = set(_KEYS) | {leg for leg, _ in _LEGS}
    if state.empty or not need.issubset(state.columns):
        return pd.DataFrame({"score": _empty("score", "float64"), "n_public": _empty("n_public", "float64")})
    st = state.dropna(subset=["avail"]).sort_values(["cik", "period"]).copy()
    if eligible is not None:
        keep = eligible.reindex(pd.MultiIndex.from_frame(st[["cik", "period"]]))
        st = st[keep.fillna(False).to_numpy()]
    if st.empty:
        return pd.DataFrame({"score": _empty("score", "float64"), "n_public": _empty("n_public", "float64")})

    grid = pd.DatetimeIndex(sorted(st["avail"].unique()))
    ciks = pd.Index(sorted(st["cik"].unique()), name="cik")
    pub = _latest_per_avail(st)
    total, live = None, None
    for leg, ascending in _LEGS:
        wide = _wide(pub, leg, grid, ciks)
        live = wide.notna() if live is None else (live & wide.notna())
        pct = wide.rank(axis=1, pct=True, ascending=ascending)
        total = pct if total is None else total + pct
    assert total is not None and live is not None
    score = (total / float(len(_LEGS))).where(live)
    n_public = live.sum(axis=1).astype("float64")

    r = grid.get_indexer(pd.Index(st["avail"]))
    c = ciks.get_indexer(pd.Index(st["cik"]))
    out = pd.DataFrame({"score": score.to_numpy()[r, c], "n_public": n_public.to_numpy()[r]}, index=pd.MultiIndex.from_frame(st[["cik", "period"]]))
    logger.info(
        "concentration score: %s manager-quarters over %s availability dates, pool %s-%s managers",
        len(out),
        len(grid),
        int(n_public.min()),
        int(n_public.max()),
    )
    return out


def eligibility(state: pd.DataFrame, roster_at: Callable[[pd.Timestamp], set[str]], min_quarters: int = 4, min_positions: int = 0) -> pd.Series:
    """Apply historical roster, recorded prior observations and optional position eligibility."""
    idx = pd.MultiIndex.from_frame(state[["cik", "period"]])
    if state.empty:
        return pd.Series(dtype="bool", index=idx, name="eligible")
    st = state.sort_values(["cik", "period"]).copy()
    st["n_prior"] = st.groupby("cik").cumcount()
    listed = {p: roster_at(pd.Timestamp(p)) for p in st["period"].unique()}
    on_roster = np.fromiter((c in listed[p] for c, p in zip(st["cik"], st["period"], strict=False)), dtype=bool, count=len(st))
    npos = pd.to_numeric(st.get("n_index_positions", pd.Series(0, index=st.index)), errors="coerce").fillna(0)
    ok = pd.Series(
        on_roster & (st["n_prior"] >= min_quarters).to_numpy() & (npos >= min_positions).to_numpy(),
        index=pd.MultiIndex.from_frame(st[["cik", "period"]]),
        name="eligible",
    )
    return ok.reindex(idx).fillna(False)


def elite_weight(scored: pd.DataFrame, mode: str = "top_k", k: int = 15) -> pd.Series:
    """Apply the existing mean-percentile cutoff or continuous weights."""
    if scored.empty:
        return _empty("sel", "float64")
    score = scored["score"].astype("float64")
    if mode == "continuous":
        return score.fillna(0.0).rename("sel")
    if mode != "top_k":
        raise ValueError(f"unknown selection mode {mode!r}: expected 'top_k' or 'continuous'")
    n = scored["n_public"].astype("float64")
    cut = (n - k + 1).clip(lower=1) / n.where(n > 0)
    return (score >= cut - 1e-12).astype("float64").where(score.notna(), 0.0).rename("sel")


def selection_diagnostics(sel: pd.Series, state: pd.DataFrame) -> pd.DataFrame:
    """Count public managers, selected managers and churn on the availability grid."""
    cols = ["date", "n_public", "n_selected", "churn"]
    if sel.empty or state.empty or "avail" not in state.columns:
        return pd.DataFrame(columns=cols)
    st = state.dropna(subset=["avail"]).copy()
    st["sel"] = sel.reindex(pd.MultiIndex.from_frame(st[["cik", "period"]])).to_numpy()
    grid = pd.DatetimeIndex(sorted(st["avail"].unique()))
    ciks = pd.Index(sorted(st["cik"].unique()), name="cik")
    live = _wide(_latest_per_avail(st), "sel", grid, ciks)
    chosen = live.fillna(0.0) > 0
    # ⚠ `fill_value` AND `astype(bool)`, not `.shift(1).fillna(False)`. A shifted boolean
    # frame comes back as OBJECT dtype holding Python bools, and `~` on object dtype is
    # integer bitwise inversion -- `~True` is -2, which is truthy, so every date reported
    # its full selected set as churn.
    prev = chosen.shift(1, fill_value=False).astype(bool)
    return pd.DataFrame(
        {
            "date": grid,
            "n_public": live.notna().sum(axis=1).to_numpy(),
            "n_selected": chosen.sum(axis=1).to_numpy(),
            "churn": ((chosen & ~prev).sum(axis=1) + (~chosen & prev).sum(axis=1)).to_numpy(),
        }
    )
