"""
manager_selection.py  (src/data_aggregate/utils/institutionals/manager_selection.py)
----------------------------------------------------------------------------------------------
WHICH elite managers count, and how much, at every point in time. Produces the `sel(m, q)`
series `superinvestor_features` multiplies every aggregate by; nothing here is a feature.

WHY CONCENTRATION AND NOT PERFORMANCE. Trailing manager performance does not predict forward
manager performance -- measured over 63 managers and 2,595 manager-quarters, the trailing-12q
to forward-4q rank IC is -0.001 (t -0.01), so a "top 15 by returns" selector ranks noise.
Concentration is a near-permanent manager trait (own rank IC 0.97 quarter to quarter), needs
no return data, and is computable from the filing itself with zero look-ahead.

⚠ HOW MUCH THIS SELECTOR IS WORTH, MEASURED, AND IT IS FAR LESS THAN THE PLAN CLAIMED. The
+3.41%/yr top-15 consensus basket behind decision D3b reproduces here to within 0.5pp
(+2.98%) -- but ONLY on the basis it was measured, which applies today's roster to 2013 and
forms portfolios from filings that were not yet public. Removing each bias in turn, over 58
quarterly rebalances against the equal-weight universe:

    correction applied                                   top-15 basket    vs bottom-15
    none (Finding 4(d)'s basis)                                +2.98%        +5.34pp
    + only filings public at the rebalance                     +0.94%        +3.27pp
    + the roster Dataroma published at q                       +0.50%           --
    + both, whole-book concentration  <- what ships            -0.32%        +1.82pp
    + both, S&P500-slice concentration                         -1.25%        +1.05pp

So the basket does NOT beat the index once the survivorship and the look-ahead are gone,
and that must not be dressed up: two thirds of the headline was manager survivorship and
the rest was trading on unfiled 13Fs. What DOES survive is the SPREAD -- the least
concentrated managers' consensus sits at -2.1% to -2.4%/yr in every corrected cell, so
concentration still separates managers by ~2pp/yr. A spread is what a cross-sectional cube
feature needs; an absolute excess is what a standalone basket would need, and only the
first claim is supported.

THE BASIS QUESTION THE PLAN LEFT OPEN IS SETTLED, AND IT INVERTS WITH THE BIAS. On the
biased basis the S&P500 slice looks better (top-15 +2.98% vs +2.23% whole-book); on the
corrected basis the whole book wins on both legs (-0.32% vs -1.25%, spread +1.82pp vs
+1.05pp). Whole book is also the coherent choice, because the eligibility gate and the
score then measure the same portfolio -- see `eligibility`.

⚠ THE POOL IS THE ROSTER AS DATAROMA PUBLISHED IT AT `q`, NOT TODAY'S (plan D20). Only 18 of
the 50 managers Dataroma listed in 2013 are still listed, and the cull is performance-driven
-- it removed Sequoia, Fairholme, Wintergreen, Arlington Value, RBS Partners and Tilson,
several of which are exactly the concentrated managers this selector ranks highest. Applying
today's roster to 2013 is survivorship bias CORRELATED WITH THE SELECTION CRITERION, which is
the worst kind. Measured on the live table the two pools differ by a mean 9.8 managers per
quarter (max 16), in both directions: today's roster drops the 19 culled managers that have a
book AND adds managers Dataroma had not yet discovered.

⚠ THE SCORE IS RANKED ON THE AVAILABILITY GRID, NOT WITHIN THE PERIOD. Ranking manager `m`'s
`q` filing against every other `q` filing uses filings that are not public when `m`'s lands:
16.5% of manager-quarters are filed after the 45-day deadline, so a within-period percentile
moves when a late filer is added, and the selection is then not reproducible from the data a
reader had on the day. Each manager is ranked against the LATEST PUBLIC state of every other
manager at `avail(m, q)`, and the weight is frozen to that filing -- the decision made when
it landed, which is the only decision a reader could have made.

THE CONSEQUENCE OF FREEZING: the live selected set is not exactly `k`. Managers file on
different days, so at any date the set mixes decisions taken at each manager's own last
filing. Concentration's 0.97 stickiness keeps the drift small; `selection_diagnostics`
reports it and the test asserts it stays in band.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

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
    ("n_positions", False),      # ascending=False -> fewest positions ranks highest
    ("eff_n", False),
    ("top10_weight", True),
)

#: Columns `manager_concentration_score` needs on `state` beyond the three legs.
_KEYS = ("cik", "period", "avail")


def _empty(name: str, dtype: str) -> pd.Series:
    idx = pd.MultiIndex.from_arrays([[], []], names=["cik", "period"])
    return pd.Series(dtype=dtype, index=idx, name=name)


def _latest_per_avail(state: pd.DataFrame) -> pd.DataFrame:
    """One row per `(cik, avail)` -- the highest period a manager made public on that date.

    Two filings can share an availability date (an amendment filed alongside the original,
    or two periods caught up on in one go). Resolving that by frame order would make the
    whole selection order-coupled, which is the defect `manager_stock_conviction`'s rank
    tie-break already had to fix once."""
    return (state.sort_values(["cik", "avail", "period"])
            .drop_duplicates(["cik", "avail"], keep="last"))


def _wide(state: pd.DataFrame, column: str, grid: pd.DatetimeIndex,
          ciks: pd.Index) -> pd.DataFrame:
    """`column` per `(availability date x manager)`, forward-filled -- each manager's latest
    PUBLIC value at every date on the grid. The same point-in-time join `_effective` performs
    for the ticker aggregates, applied to the manager axis instead."""
    return (state.pivot(index="avail", columns="cik", values=column)
            .reindex(index=grid, columns=ciks).ffill())


def manager_concentration_score(state: pd.DataFrame,
                                eligible: pd.Series | None = None) -> pd.DataFrame:
    """Point-in-time concentration per `(cik, period)`.

    `state` is `manager_quarter_state` output with an `avail` column -- the date the filing
    became public (`public_state` computes it). `eligible` is a boolean per `(cik, period)`;
    ineligible manager-quarters are dropped from the RANKING as well as from the result, so a
    manager with two quarters of history cannot push an established one down the ladder.

    ⚠ RETURNS TWO COLUMNS, NOT THE BARE PERCENTILE THE PLAN SPECIFIED.

    * ``score`` -- percentile in (0, 1], 1 = most concentrated, taken across the managers
      public at `avail(m, q)` and read off at that manager's own availability date.
    * ``n_public`` -- how many managers that percentile was taken across.

    `n_public` is not decoration: the eligible pool grows from ~40 managers in 2013 to ~63
    today, so "the top 15" is the 63rd percentile in one era and the 76th in another. A
    percentile alone cannot express a fixed-`k` cut, and hard-coding one threshold would
    silently select 15 managers early and 24 late.
    """
    need = set(_KEYS) | {leg for leg, _ in _LEGS}
    if state.empty or not need.issubset(state.columns):
        return pd.DataFrame({"score": _empty("score", "float64"),
                             "n_public": _empty("n_public", "float64")})
    st = state.dropna(subset=["avail"]).sort_values(["cik", "period"]).copy()
    if eligible is not None:
        keep = eligible.reindex(pd.MultiIndex.from_frame(st[["cik", "period"]]))
        st = st[keep.fillna(False).to_numpy()]
    if st.empty:
        return pd.DataFrame({"score": _empty("score", "float64"),
                             "n_public": _empty("n_public", "float64")})

    grid = pd.DatetimeIndex(sorted(st["avail"].unique()))
    ciks = pd.Index(sorted(st["cik"].unique()), name="cik")
    pub = _latest_per_avail(st)
    total, live = None, None
    for leg, ascending in _LEGS:
        wide = _wide(pub, leg, grid, ciks)
        live = wide.notna() if live is None else (live & wide.notna())
        pct = wide.rank(axis=1, pct=True, ascending=ascending)
        total = pct if total is None else total + pct
    score = (total / float(len(_LEGS))).where(live)
    n_public = live.sum(axis=1).astype("float64")

    r = grid.get_indexer(st["avail"])
    c = ciks.get_indexer(st["cik"])
    out = pd.DataFrame(
        {"score": score.to_numpy()[r, c], "n_public": n_public.to_numpy()[r]},
        index=pd.MultiIndex.from_frame(st[["cik", "period"]]))
    logger.info("concentration score: %s manager-quarters over %s availability dates, "
                "pool %s-%s managers", len(out), len(grid),
                int(n_public.min()), int(n_public.max()))
    return out


def eligibility(state: pd.DataFrame, roster_at: Callable[[pd.Timestamp], set[str]],
                min_quarters: int = 4, min_positions: int = 0) -> pd.Series:
    """Boolean per `(cik, period)`: may this manager-quarter be scored at all?

    ⚠ `min_positions` DEFAULTS TO 0, NOT TO THE 3 THE PLAN PROPOSED, AND THE REASON IS
    MEASURED. Gating on the INDEX sleeve while scoring on the WHOLE BOOK is incoherent --
    it drops a manager for holding two universe names and then ranks them on two hundred --
    and on the corrected pool (roster as of `q`, filings public at the rebalance) it costs
    1.3pp a year: the top-15 consensus basket runs -0.27%/yr with the gate off and -1.60%
    with it on, and the top-15-minus-bottom-15 spread collapses from +1.97pp to +0.63pp.
    Set it above 0 only alongside a slice-basis score, where the gate and the score at least
    measure the same portfolio.

    Three gates, all point-in-time:

    * **on the roster at `q`** -- `roster_at(q)` is `superinvestor_roster.roster_as_of`, the
      most recent Dataroma snapshot at or before `q`. ⚠ The snapshot series starts
      2013-01-01, so `roster_as_of` returns the EMPTY set for 2011-2012 and would zero the
      whole pre-2013 panel; the caller is expected to floor `q` at the first snapshot date.
      Extrapolating the 2013 roster backwards is a compromise, but it is the least biased one
      available -- the alternative is today's roster, which is the bias this table exists to
      remove.
    * **`min_quarters` of prior filings** -- a concentration estimate from two quarters is
      noise. Counted on the manager's OWN prior periods in this table, not on calendar time,
      so a manager who skipped a year is not credited with it.
    * **`min_positions` index positions** -- off by default, see the warning above. The
      argument for it is that a manager holding two universe names contributes almost
      nothing to a consensus basket; the measurement says that argument is wrong, because
      those managers are the concentrated ones.
    """
    idx = pd.MultiIndex.from_frame(state[["cik", "period"]])
    if state.empty:
        return pd.Series(dtype="bool", index=idx, name="eligible")
    st = state.sort_values(["cik", "period"]).copy()
    st["n_prior"] = st.groupby("cik").cumcount()
    listed = {p: roster_at(pd.Timestamp(p)) for p in st["period"].unique()}
    on_roster = np.fromiter((c in listed[p] for c, p in zip(st["cik"], st["period"])),
                            dtype=bool, count=len(st))
    npos = pd.to_numeric(st.get("n_index_positions", pd.Series(0, index=st.index)),
                         errors="coerce").fillna(0)
    ok = pd.Series(on_roster
                   & (st["n_prior"] >= min_quarters).to_numpy()
                   & (npos >= min_positions).to_numpy(),
                   index=pd.MultiIndex.from_frame(st[["cik", "period"]]), name="eligible")
    return ok.reindex(idx).fillna(False)


def elite_weight(scored: pd.DataFrame, mode: str = "top_k", k: int = 15) -> pd.Series:
    """`sel(m, q)` in [0, 1] from `manager_concentration_score`'s output.

    * ``top_k`` -- 1.0 for the `k` highest-scoring managers public at that availability date,
      0.0 otherwise. Keeps the aggregate interpretable ("how many of the k most concentrated
      managers hold this name") and is the form the consensus-basket measurement was run in.
      ⚠ IT ALSO CUTS COVERAGE: a name held only by managers outside the top `k` has no
      counted holder, so the panel's first-appearance mask leaves it out entirely rather
      than at 0. Discarding ~70% of the pool is the point, but it is not free.
    * ``continuous`` -- the percentile itself. Discards nothing and keeps the gradient, which
      is what a boosted model would rather have; the cost is that the aggregate stops being a
      headcount and its scale drifts with the pool.

    ⚠ PREFER `continuous` ON THE EVIDENCE. `top_k` asserts a cliff between manager `k` and
    `k+1`; what was measured is a gradient -- the corrected top-15 basket returns -0.32%/yr
    and the selector's whole value is the ~1.8pp spread over the least-concentrated control.
    A hard cut bets on a boundary the data does not show.

    The `top_k` cut is `score >= (n_public - k + 1) / n_public`, which is the top `k` of that
    date's pool exactly. Using one fixed percentile instead would select 15 managers when the
    pool is 40 and 24 when it is 63.
    """
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
    """Per availability date: how many managers are public, how many are selected, and how
    much the selected set churned since the previous date.

    Concentration is 97% rank-sticky, so a membership change above ~4 names between
    consecutive filing dates is a bug rather than a signal. This is the frame the test
    asserts on and the one to read when the selector behaves oddly."""
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
    return pd.DataFrame({
        "date": grid,
        "n_public": live.notna().sum(axis=1).to_numpy(),
        "n_selected": chosen.sum(axis=1).to_numpy(),
        "churn": ((chosen & ~prev).sum(axis=1) + (~chosen & prev).sum(axis=1)).to_numpy(),
    })
