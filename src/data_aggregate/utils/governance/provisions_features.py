"""
provisions_features.py  (src/data_aggregate/utils/governance/provisions_features.py)
-----------------------------------------------------------------------------------
The last three governance families, all off `def14a_llm` alone:

  * **8 — entrenchment-provision TRANSITIONS.** Thirteen 0->1 / 1->0 flags on seven bylaw and
    board-leadership columns, plus the two aggregate counts and the board-independence delta.
  * **9 — board busyness.** `avg_other_public_boards`, a column nothing has ever consumed.
  * **10 — the auditor block.** Change, tenure and big-4, on the CANONICAL firm from
    `auditors.py` rather than the raw string.

⚠ **CHANGES, NOT LEVELS** (GPT §11). A classified board is a fact about a company; *adopting*
one is an act by a board, and it is the act that carries information. So the seven provision
columns ship as transitions and only two levels ship at all (`board_busyness`,
`ceo_is_board_chair`).

⚠ **A SILENT YEAR IS NOT A CHANGE.** `poison_pill` and `majority_voting` are tri-state by
design -- NULL when the proxy says nothing -- because inferring FALSE from silence made
`poison_pill` read TRUE for 0.1% of rows and flipped `majority_voting` 21.2% year-over-year on a
bylaw that does not change. `_transitions` therefore drops the NULL rows BEFORE it diffs, which
both stops a silent year manufacturing a transition and stops it hiding one.

⚠ **NO INTERACTION FEATURES** (D16). GPT §13's five products are not built; every leg of every
one of them ships separately, and the tree models find the products themselves. The one leg that
existed *only* inside a product -- `ceo_is_board_chair`, X05's left half -- ships here as a plain
level. There must be no column with `_x_` in its name, and a test asserts it.

MEASURED 2026-09-08 on the live archive (12,343 proxies, 488 tickers) -- the numbers that
decided three things the plan had specified differently. Each is recorded next to the set it
changed:

  * `poison_pill_added` fires **zero** times, so it is not exported (`_MIN_EVENTS`);
  * two thirds of `board_busyness_delta_1y`'s pairs had an INTERPOLATED leg, so both deltas
    are now gated on provenance (`_DISCLOSED_ONLY_DELTAS`);
  * nothing in this module earns a peer leg (`PEER_RELATIVE_FIELDS`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.pit import fundamentals_to_daily
from src.data_aggregate.utils.governance.auditors import (
    BIG4, canonical_auditor_series, unrecognised_auditor_names,
)
from src.data_aggregate.utils.governance.def14a_impute import DELTA_PROVENANCE_COLUMNS
from src.data_aggregate.utils.governance.staleness import expire_event_fields

#: The seven tri-state / boolean provision columns the transition detector runs over. Every one
#: is stored 1.0 / 0.0 / NULL by `flatten._bnum`, and NULL means the proxy was silent.
PROVISION_COLUMNS: tuple[str, ...] = (
    "classified_board", "dual_class_shares", "poison_pill", "majority_voting",
    "ceo_is_board_chair", "independent_chair", "lead_independent_director",
)

#: exported flag -> (source column, direction). The detector computes BOTH directions on all
#: seven columns generically; only these thirteen are exported.
#:
#: ⚠ There is deliberately no `dual_class_removed`: GPT does not ask for it, and the two
#: directions are not symmetric evidence -- an unwind is usually a recapitalisation retiring a
#: share class rather than a board improving its own governance.
TRANSITION_FLAGS: dict[str, tuple[str, str]] = {
    # deterioration
    "classified_board_added": ("classified_board", "up"),
    "poison_pill_added": ("poison_pill", "up"),
    "dual_class_added": ("dual_class_shares", "up"),
    "majority_voting_removed": ("majority_voting", "down"),
    "ceo_became_board_chair": ("ceo_is_board_chair", "up"),
    "independent_chair_lost": ("independent_chair", "down"),
    "lead_independent_director_lost": ("lead_independent_director", "down"),
    # improvement
    "classified_board_removed": ("classified_board", "down"),
    "poison_pill_removed": ("poison_pill", "down"),
    "majority_voting_added": ("majority_voting", "up"),
    "ceo_stopped_being_chair": ("ceo_is_board_chair", "down"),
    "independent_chair_added": ("independent_chair", "up"),
    "lead_independent_director_added": ("lead_independent_director", "up"),
}

DETERIORATION_FLAGS: tuple[str, ...] = (
    "classified_board_added", "poison_pill_added", "dual_class_added",
    "majority_voting_removed", "ceo_became_board_chair", "independent_chair_lost",
    "lead_independent_director_lost",
)
IMPROVEMENT_FLAGS: tuple[str, ...] = (
    "classified_board_removed", "poison_pill_removed", "majority_voting_added",
    "ceo_stopped_being_chair", "independent_chair_added", "lead_independent_director_added",
)

#: ⚠ A TRANSITION FLAG THAT NEVER FIRES IS NOT EXPORTED, and this is a measured correction to
#: the plan rather than a tidy-up. Phase 5 was written expecting `poison_pill_added` to be
#: "near-empty, and that is the honest answer": a transition needs two disclosed observations of
#: a field disclosed 20% of the time. Measured on the live archive it is not near-empty, it is
#: **exactly empty** -- 1,585 adjacent disclosed pairs of `poison_pill`, of which **0 are 0->1
#: and 23 are 1->0**. So the column would ship as a literal constant 0.0, which is worse than
#: absent: `poison_pill_added == 0` reads as "this firm adopted no rights plan" when the
#: truthful statement is "the 80% of proxies that say nothing still say nothing".
#:
#: The gate counts fired events at the FILING grain, so the column REAPPEARS by itself the first
#: time a filer in this universe adopts a pill between two disclosing proxies. The count is
#: logged either way, which is what the plan actually asked for.
#:
#: ⚠ A dropped flag is still summed into its aggregate count. Adding a constant 0.0 changes no
#: total, and keeping it in preserves the count's COVERAGE -- dropping it from the sum would
#: shrink the `min_count=1` population for no gain.
_MIN_EVENTS = 1

#: ⚠ DELTAS GATED ON PROVENANCE, the second measured correction. Both these columns are in
#: `impute_def14a.INTERP`, and that module's own rule is the one being honoured here: *never
#: linearly interpolate a level whose YoY change is itself a feature*, because the delta then
#: measures the FILL's slope rather than the company's change. D23 knowingly took that trade for
#: `avg_other_public_boards`; phase 2 §5 required the share to be measured before shipping.
#:
#: Measured 2026-09-08, share of adjacent pairs with at least one INTERPOLATED leg:
#:     avg_other_public_boards      65.5%  (6,556 of 10,004)   <- the majority, not a corner
#:     pct_independent_directors    20.7%  (2,329 of 11,224)
#:
#: 65.5% is not a trade worth taking: a linearly-filled segment has a CONSTANT first difference,
#: so two thirds of `board_busyness_delta_1y` would have been one number repeated. Both deltas
#: are therefore computed only where BOTH legs were disclosed, which `impute_def14a` records in
#: `<column>_imputed`. The cost is coverage and it is reported in the tally.
#:
#: The LEVELS keep the fill. A carried board average is a defensible estimate of a standing
#: fact; its year-over-year change is not.
#:
#: The list itself lives in `def14a_impute.DELTA_PROVENANCE_COLUMNS`, next to the code that
#: stamps the flags, so there is one list rather than two that can drift apart.
_DISCLOSED_ONLY_DELTAS: tuple[str, ...] = DELTA_PROVENANCE_COLUMNS

#: A YoY pair must be two filings roughly one year apart -- the same window and the same reason
#: as `pay_features._ANNUAL_GAP_DAYS`. It gates the two `_delta_1y` fields, whose NAMES claim a
#: one-year change; it deliberately does NOT gate the transitions, which compare the two nearest
#: KNOWN observations however far apart they are, because a provision change is news whenever it
#: becomes visible. Measured cost, among the pairs that already cleared the provenance gate:
#: 106 of 8,895 board-independence pairs (1.2%) and 38 of 3,448 busyness pairs (1.1%).
_ANNUAL_GAP_DAYS: tuple[float, float] = (250.0, 550.0)

#: The board-independence drop that fires `board_independence_drop_10pp`, in fraction units.
#:
#: ⚠ Read it for what it is. `pct_independent_directors` is a fraction, so -0.10 is one
#: independent seat lost off a ten-person board -- which is why it fires on **10.2%** of
#: adjacent pairs (1,143 of 11,224), and on **8.8%** of the shipped feature's non-null cells,
#: rather than on the handful a "governance collapse" reading would suggest. It is a
#: one-director event on a typical board, not a rare cliff.
_INDEPENDENCE_DROP = -0.10

#: ⚠ AND WHY THE THRESHOLD NEEDS A TOLERANCE. `0.80 - 0.90` is `-0.09999999999999998` in binary
#: floating point, so a board going from 90% to 80% independent -- the textbook ten-point drop,
#: and two numbers a proxy states exactly -- does NOT satisfy `<= -0.10`. The flag would have
#: silently missed the very case it is named after. One part in a billion is far below any
#: meaningful change in a board fraction (a 1,000-member board would move it by 0.001), so the
#: tolerance cannot admit a case that is not genuinely at or past the threshold.
_DROP_EPS = 1e-9

#: Days per year, for a tenure measured from a start DATE rather than a start year.
_DAYS_PER_YEAR = 365.25

#: A disclosed `auditor_since_year` outside this range is an extraction error, not a tenure.
#: The observed maximum disclosed tenure is 120 years, which is plausible for a US industrial
#: audited since the 1900s, so the band is wide on purpose and only rejects the impossible.
_TENURE_YEAR_BAND: tuple[int, int] = (1850, 2100)

#: ⚠ EMPTY, and for the SECOND phase running that is a measured result rather than an omission
#: (phase 3's four vote families did earn twelve peer legs -- the yardstick separates).
#: Phase 5 specified nine of these fields as "peer-panelled". The precondition a peer z needs is
#: that a peer norm EXISTS -- the phase-3 floor is 7.7% between-sector variance share, which
#: `insider_ownership_pct` scores, against 9.0% for `profitMargins` -- and nothing here reaches
#: it. Measured 2026-09-08 on the built frames:
#:
#:     field                            between-sector   peer-basket R^2   peer z is NaN
#:     board_busyness_delta_1y                    5.9%              0.6%           43.1%
#:     ceo_is_board_chair                         5.1%              1.9%            5.5%
#:     board_busyness                             3.5%              0.7%            0.6%
#:     board_independence_delta_1y                2.8%              0.3%            2.5%
#:     auditor_tenure                             2.7%              0.3%            0.3%
#:     governance_deterioration_count             2.4%              0.3%           45.8%
#:     governance_improvement_count               2.4%              0.3%           36.3%
#:     net_governance_change                      2.3%              0.3%           20.4%
#:     -- the floor the same rule kept in phase 3 --
#:     insider_ownership_pct (kept)               7.7%               n/a
#:     board_size (kept)                         12.4%             10.0%
#:     profitMargins (reference)                  9.0%             13.6%
#:
#: THE THIRD COLUMN IS THE DAMNING ONE. It is the share of non-null cells whose peer z comes back
#: NaN because the basket had no dispersion at all -- every peer reporting the same value. On the
#: two counts that is 46% and 36% of the panel, which is what "a peer z of a mostly-zero integer
#: count" means arithmetically: most baskets are seven zeros, and a firm's one event divided by
#: the standard deviation of seven zeros is not a comparison.
#:
#: Three separate reasons converge, and they are worth keeping apart because each disqualifies a
#: different group:
#:
#:   * THE COUNTS AND THE DELTAS ARE ALREADY DIFFERENCES with a meaningful zero, and that zero
#:     IS the thesis -- exactly the argument that kept `ceo_pay_growth` raw in phase 3. A board
#:     that lost its independent chair inside a sector where three peers also did is still a
#:     board that lost its independent chair; a peer z encodes it as "normal".
#:   * `ceo_is_board_chair` IS BINARY, so its peer z is the z-score of a Bernoulli draw over ~7
#:     peers and reads "how many of my peers ALSO have a combined chair" -- the same degeneracy
#:     that moved `founder_ceo` to raw-only in phase 3.
#:   * `board_busyness` AND `auditor_tenure` are genuine continuous levels and simply have no
#:     sector norm: who audits you, and how many other boards your directors sit on, are
#:     firm-level facts.
PEER_RELATIVE_FIELDS: frozenset[str] = frozenset()

#: Fields that ship as a raw 1/0 flag: no peer leg (which `PEER_RELATIVE_FIELDS` already
#: settles) and, more importantly, no assumed SIGN.
#:
#: ⚠ UNLIKE the other two family modules, this set is NOT a subset of `EVENT_FIELDS`. There the
#: invariant "a flag records an event and must expire with it" holds because every flag they emit
#: is an act. Here three are 0/1 descriptions of a standing STATE -- `auditor_is_big4`,
#: `auditor_tenure_censored`, `ceo_is_board_chair` -- so they are raw in ENCODING and levels in
#: TIME. Expiring them would delete a fact that is still true.
#:
#: ⚠ `auditor_changed` is the one that needs saying out loud, because GPT's objection to it
#: stands and was overridden rather than answered (D32): without Item 4.01's reasons -- was the
#: auditor dismissed, did it resign, was there an accounting disagreement -- a change carries no
#: direction. It ships as an unsigned flag. `sec_8k` DOES carry the 4.01 rows and reading their
#: narratives is the obvious follow-up; it is not in this plan.
RAW_FLAG_FIELDS: frozenset[str] = frozenset(
    set(TRANSITION_FLAGS)
    | {"board_independence_drop_10pp", "auditor_changed", "auditor_is_big4",
       "auditor_tenure_censored", "ceo_is_board_chair"}
)

#: Fields describing an EVENT, which expire 548 days after the filing that disclosed them (D21).
#:
#: The five LEVELS are excluded and each for a stated reason: `board_busyness` and
#: `ceo_is_board_chair` are standing facts between proxies; `auditor_tenure` ACCRUES daily off a
#: start date, so ageing it out would delete a number that is more current than the filing;
#: `auditor_is_big4` and `auditor_tenure_censored` describe that standing level rather than an
#: act.
#:
#: ⚠ The three COUNTS are declared events but are not aged directly -- they are summed FROM the
#: already-expired flags, so they inherit the mask exactly and cannot disagree with their own
#: components. `test_provisions_features` asserts that inheritance.
EVENT_FIELDS: frozenset[str] = frozenset(
    set(TRANSITION_FLAGS)
    | {"governance_deterioration_count", "governance_improvement_count",
       "net_governance_change", "board_independence_delta_1y",
       "board_independence_drop_10pp", "board_busyness_delta_1y", "auditor_changed"}
)

#: Every field this module can emit -- the exhaustiveness anchor. Phase 4 learned why it is
#: needed: `pay_up_stock_down` sat in two classification sets while no code path produced it,
#: and the two fields that WERE produced sat in neither and silently skipped their expiry.
ALL_FIELDS: frozenset[str] = frozenset(
    set(TRANSITION_FLAGS)
    | {"governance_deterioration_count", "governance_improvement_count",
       "net_governance_change", "board_independence_delta_1y",
       "board_independence_drop_10pp",
       "board_busyness", "board_busyness_delta_1y",
       "ceo_is_board_chair",
       "auditor_changed", "auditor_tenure", "auditor_tenure_censored", "auditor_is_big4"}
)


# --------------------------------------------------------------------------- #
# primitives                                                                   #
# --------------------------------------------------------------------------- #
def _prepared(hist: pd.DataFrame, cols: list[str]) -> pd.DataFrame | None:
    """`hist` narrowed to `[ticker, as_of, *cols]`, dated and sorted.

    Sorting here rather than trusting the caller: every primitive below reads `shift(1)` as
    "the previous filing", and that is only true on a chronologically ordered frame.
    """
    have = [c for c in cols if c in hist.columns]
    if not have or "ticker" not in hist.columns or "as_of" not in hist.columns:
        return None
    h = hist[["ticker", "as_of", *have]].copy()
    h["as_of"] = pd.to_datetime(h["as_of"], errors="coerce")
    h = h.dropna(subset=["ticker", "as_of"])
    return h.sort_values(["ticker", "as_of"]).reset_index(drop=True) if not h.empty else None


def _transitions(hist: pd.DataFrame, field: str) -> pd.DataFrame | None:
    """`[ticker, as_of, <field>__up, <field>__down]`, stamped on the LATER filing's `as_of`.

    One row per pair of ADJACENT KNOWN observations -- not one row per detected change, which is
    what the plan's sketch said. The difference matters twice over:

      * a quiet year has to read 0.0, not NaN, or "no change was disclosed" is indistinguishable
        from "we have never seen this provision"; and
      * the aggregate counts are `min_count=1` sums over these flags, so without the zeros a
        count could only ever be positive or NaN and never the 0 that describes most firms.

    ⚠ THE NULL ROWS GO BEFORE THE DIFF (GPT §11). Dropping them first is what makes this
    tri-state-safe: a silent proxy year neither manufactures a transition nor conceals one,
    because the comparison falls through to the two nearest years that actually spoke. The pair
    may therefore span more than a year, and for a provision that is correct -- the change became
    public at the later filing, whenever the earlier one was.
    """
    h = _prepared(hist, [field])
    if h is None or field not in h.columns:
        return None
    h[field] = pd.to_numeric(h[field], errors="coerce")
    h = h.dropna(subset=[field])
    if h.empty:
        return None
    prev = h.groupby("ticker", sort=False)[field].shift(1)
    paired = prev.notna()
    if not bool(paired.any()):
        return None
    delta = h[field] - prev
    out = pd.DataFrame({
        "ticker": h["ticker"],
        "as_of": h["as_of"],
        f"{field}__up": (delta > 0).astype("float64"),
        f"{field}__down": (delta < 0).astype("float64"),
    })
    return out[paired.to_numpy()].reset_index(drop=True)


def _annual_delta(hist: pd.DataFrame, field: str, tally: dict[str, int]) -> pd.DataFrame | None:
    """`[ticker, as_of, <field>]` holding the ONE-YEAR change, on DISCLOSED legs only.

    Three gates, in the order they reject:

      1. both legs disclosed -- `<field>_imputed` from `impute_def14a` marks a leg the temporal
         fill invented, and a difference between two invented points is the fill's slope;
      2. the two filings are 250-550 days apart, so a pair straddling a missing proxy is not
         relabelled a one-year change;
      3. the result is stamped on the LATER filing, the date the change became knowable.

    Every rejection is counted into `tally`, because the coverage this costs is the thing a
    reader needs in order to judge the gate.
    """
    flag = f"{field}_imputed"
    h = _prepared(hist, [field, flag])
    if h is None or field not in h.columns:
        return None
    h[field] = pd.to_numeric(h[field], errors="coerce")
    h = h.dropna(subset=[field])
    if h.empty:
        return None
    # An absent provenance column means "nothing was recorded as imputed", not "everything was":
    # `impute_def14a` stamps it for exactly the columns in `_DISCLOSED_ONLY_DELTAS`, and a
    # synthetic frame that never went through impute is clean by construction.
    if field in _DISCLOSED_ONLY_DELTAS and flag in h.columns:
        imputed = pd.to_numeric(h[flag], errors="coerce").fillna(0.0) > 0
    else:
        imputed = pd.Series(False, index=h.index)

    g = h.groupby("ticker", sort=False)
    prev = g[field].shift(1)
    gap = g["as_of"].diff().dt.days
    paired = prev.notna()
    if not bool(paired.any()):
        return None

    # ⚠ BOTH legs, and `fill_value=False` rather than `.fillna(False)`: a groupby-shift of a
    # boolean Series returns OBJECT dtype, and `~` on a Python `bool` inside an object Series is
    # integer inversion -- `~True` is -2, which is truthy, so the earlier leg's check silently
    # passed for every row. Caught because the rejection count came out exactly equal to the
    # number of interpolated CELLS, when one interpolated cell should invalidate up to two pairs.
    prev_imputed = imputed.groupby(h["ticker"], sort=False).shift(1, fill_value=False)
    clean = ~imputed & ~prev_imputed.astype(bool)
    annual = gap.between(*_ANNUAL_GAP_DAYS)
    keep = paired & clean & annual

    tally[f"{field} delta: adjacent pairs"] = int(paired.sum())
    tally[f"{field} delta: rejected (a leg was interpolated)"] = int((paired & ~clean).sum())
    tally[f"{field} delta: rejected (gap outside 250-550d)"] = int(
        (paired & clean & ~annual).sum())
    if not bool(keep.any()):
        return None
    out = pd.DataFrame({"ticker": h["ticker"], "as_of": h["as_of"],
                        field: (h[field] - prev).where(keep)})
    return out[keep.to_numpy()].reset_index(drop=True)


def _sum_min_count(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Element-wise sum over daily frames with `min_count=1`: an all-NaN cell stays NaN.

    Written out rather than `sum(frames)` because a plain `+` propagates NaN from ANY frame, so
    a ticker missing one provision would lose its whole count -- and `fillna(0).sum()` goes the
    other way and turns a firm with no provision history at all into a confident zero. The union
    of columns keeps a ticker only one source knows about.
    """
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    cols = frames[0].columns
    for f in frames[1:]:
        cols = cols.union(f.columns)
    total = pd.DataFrame(0.0, index=frames[0].index, columns=cols)
    seen = pd.DataFrame(0.0, index=frames[0].index, columns=cols)
    for f in frames:
        g = f.reindex(columns=cols)
        total = total.add(g.fillna(0.0), fill_value=0.0)
        seen = seen.add(g.notna().astype("float64"), fill_value=0.0)
    return total.where(seen > 0)


def _flag_frame(hist: pd.DataFrame, name: str, values: pd.Series) -> pd.DataFrame:
    """`[ticker, as_of, <name>]` -- a per-filing series re-keyed under its EXPORTED name.

    `expire_stale` measures a cell's age against the `as_of` of the filing that produced it, and
    it finds that filing by looking `name` up in the history frame it is handed. So every flag
    needs a history whose column is called what the FEATURE is called; this builds it.
    """
    return pd.DataFrame({"ticker": hist["ticker"].to_numpy(),
                         "as_of": hist["as_of"].to_numpy(),
                         name: pd.to_numeric(values, errors="coerce").to_numpy()})


def _expire(frames: dict[str, pd.DataFrame], hist: pd.DataFrame, tally: dict[str, int],
            sources: dict[str, str] | None = None) -> dict[str, pd.DataFrame]:
    """`expire_event_fields` plus the tally lines, which every caller here wants together."""
    capped, stats = expire_event_fields(frames, hist, EVENT_FIELDS, sources=sources)
    for name, (expired, before) in stats.items():
        if expired:
            tally[f"expired >548d: {name}"] = expired
            tally[f"non-null before expiry: {name}"] = before
    return capped


# --------------------------------------------------------------------------- #
# family 8 -- provision transitions                                            #
# --------------------------------------------------------------------------- #
def _provision_transitions(
    hist: pd.DataFrame, idx: pd.DatetimeIndex, tally: dict[str, int],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """(exported flag frames, ALL flag frames incl. the never-fired ones the counts still use).

    The second return value is what the aggregates are built from -- see `_MIN_EVENTS` for why a
    flag can be summed but not shipped.
    """
    exported: dict[str, pd.DataFrame] = {}
    for_counts: dict[str, pd.DataFrame] = {}

    detected = {col: _transitions(hist, col) for col in PROVISION_COLUMNS}
    for flag, (col, direction) in TRANSITION_FLAGS.items():
        trans = detected.get(col)
        if trans is None:
            tally[f"skipped: no adjacent disclosed pair of {col}"] = 1
            continue
        events = trans[f"{col}__{direction}"]
        fired = int((events > 0).sum())
        tally[f"events: {flag}"] = fired
        h = _flag_frame(trans, flag, events)
        daily = fundamentals_to_daily(h, flag, idx)
        if daily.empty or not daily.notna().any().any():
            continue
        frame = _expire({flag: daily}, h, tally)[flag]
        for_counts[flag] = frame
        if fired < _MIN_EVENTS:
            # A constant 0.0 column, measured rather than assumed. See `_MIN_EVENTS`.
            tally[f"NOT exported (0 events, constant column): {flag}"] = 1
            continue
        exported[flag] = frame
    return exported, for_counts


def _aggregates(for_counts: dict[str, pd.DataFrame],
                tally: dict[str, int]) -> dict[str, pd.DataFrame]:
    """The two counts and their net. ONE EVENT, ONE POINT -- no estimated weights (GPT §11
    GOV03): the literature has no agreed severity ordering over these provisions, and inventing
    one would bury a judgement inside a number that looks like a count."""
    det = [for_counts[f] for f in DETERIORATION_FLAGS if f in for_counts]
    imp = [for_counts[f] for f in IMPROVEMENT_FLAGS if f in for_counts]
    tally["deterioration count: members summed"] = len(det)
    tally["improvement count: members summed"] = len(imp)

    out: dict[str, pd.DataFrame] = {}
    d = _sum_min_count(det)
    i = _sum_min_count(imp)
    if not d.empty:
        out["governance_deterioration_count"] = d
    if not i.empty:
        out["governance_improvement_count"] = i
    if not d.empty and not i.empty:
        cols = d.columns.intersection(i.columns)
        if len(cols) > 0:
            # Higher = improving. Both sums come from the same seven source columns, so they are
            # non-null together and the difference does not silently drop a firm.
            out["net_governance_change"] = i[cols] - d[cols]
    return {k: v for k, v in out.items() if v.notna().any().any()}


def _independence_delta(hist: pd.DataFrame, idx: pd.DatetimeIndex,
                        tally: dict[str, int]) -> dict[str, pd.DataFrame]:
    """`board_independence_delta_1y` and the -10pp flag derived from it.

    `pct_independent_directors` keeps shipping as a LEVEL from `panel.py` (D3, a live
    `comp_governance` member); this is additive to it.
    """
    field = "pct_independent_directors"
    d = _annual_delta(hist, field, tally)
    if d is None:
        tally["skipped: no disclosed annual pair of pct_independent_directors"] = 1
        return {}
    h = d.rename(columns={field: "board_independence_delta_1y"})
    daily = fundamentals_to_daily(h, "board_independence_delta_1y", idx)
    if daily.empty or not daily.notna().any().any():
        return {}
    # NaN-preserving: `daily <= x` is False on NaN, which would turn "unknown" into "no drop".
    # `_DROP_EPS` is what makes an exactly-ten-point drop fire -- see the constant.
    drop = (daily <= _INDEPENDENCE_DROP + _DROP_EPS).astype("float64").where(daily.notna())
    frames = {"board_independence_delta_1y": daily, "board_independence_drop_10pp": drop}
    capped = _expire(frames, h, tally,
                     sources={k: "board_independence_delta_1y" for k in frames})
    return {k: v for k, v in capped.items() if not v.empty and v.notna().any().any()}


# --------------------------------------------------------------------------- #
# family 9 -- board busyness                                                   #
# --------------------------------------------------------------------------- #
def _busyness(hist: pd.DataFrame, idx: pd.DatetimeIndex,
              tally: dict[str, int]) -> dict[str, pd.DataFrame]:
    """`board_busyness` (a level, fill kept) and its delta (disclosed legs only).

    ⚠ NO `busy_board = 1` THRESHOLD (GPT §12 closing note). The academic measure is the FRACTION
    OF OUTSIDE DIRECTORS holding three or more seats; `avg_other_public_boards` is a board
    AVERAGE, a different statistic, and a cutoff on it would claim a definition it does not
    have. The per-director detail the real measure needs sits in
    `def14a_directors.other_public_company_boards` -- phase 6, explicitly not here.
    """
    field = "avg_other_public_boards"
    out: dict[str, pd.DataFrame] = {}
    if field not in hist.columns:
        tally["skipped: no avg_other_public_boards -> no board busyness"] = 1
        return out
    level = fundamentals_to_daily(hist, field, idx)
    if level.empty or not level.notna().any().any():
        tally["skipped: avg_other_public_boards all null -> no board busyness"] = 1
        return out
    out["board_busyness"] = level

    d = _annual_delta(hist, field, tally)
    if d is None:
        tally["skipped: no disclosed annual pair of avg_other_public_boards"] = 1
        return out
    h = d.rename(columns={field: "board_busyness_delta_1y"})
    delta = fundamentals_to_daily(h, "board_busyness_delta_1y", idx)
    if delta.empty or not delta.notna().any().any():
        return out
    out["board_busyness_delta_1y"] = _expire(
        {"board_busyness_delta_1y": delta}, h, tally)["board_busyness_delta_1y"]
    return out


# --------------------------------------------------------------------------- #
# family 10 -- the auditor block                                               #
# --------------------------------------------------------------------------- #
def _auditor_history(hist: pd.DataFrame, tally: dict[str, int]) -> pd.DataFrame | None:
    """Per filing: the canonical firm, whether it changed, whether it is big-4, and the start
    date the tenure accrues from -- plus the flag saying which BASIS that start date came from.

    ⚠ THE CANONICAL FIRM IS THE WHOLE POINT. 336 tickers show a changing raw `auditor_name`
    string against 165 showing a changing FIRM, so 51% of apparent auditor changes are the filer
    respelling its own auditor's name. An un-normalised flag would be half noise.

    TWO BASES FOR TENURE, NEVER MIXED SILENTLY:
      * DISCLOSED -- `auditor_since_year`, filled on only 14.2% of rows (1,758), mean disclosed
        tenure 23.4 years, max 120;
      * CENSORED -- the run-length of the canonical firm inside our own archive, computable
        wherever `auditor_name` is (85.1%) but left-censored by the archive's own start, and
        severely so at a 23-year mean tenure: the measured censored mean is **10.7 years against
        the disclosed 23.4**, i.e. the fallback understates by more than half.

    `auditor_tenure_censored` is 1.0 on the second basis. A censored 12 and a disclosed 12 are
    different claims, and the flag is what keeps them distinguishable downstream.
    """
    h = _prepared(hist, ["auditor_name", "auditor_since_year"])
    if h is None or "auditor_name" not in h.columns:
        return None
    unknown = unrecognised_auditor_names(h["auditor_name"])
    if unknown:
        # A NEW unrecognised spelling means `AUDITOR_ALIASES` needs an entry, not that a company
        # hired a mystery firm. `other` is a legitimate value too, so this is the only thing
        # that makes the difference visible.
        tally["auditor_name strings NOT in AUDITOR_ALIASES (-> other)"] = sum(unknown.values())
    h["firm"] = canonical_auditor_series(h["auditor_name"])
    # The rows naming NO auditor go BEFORE the change detection, for the same reason the NULLs
    # go before the diff in `_transitions`: the 14.9% of proxies that name no auditor are silent,
    # not evidence of a new one, and comparing across them would report two changes (to a mystery
    # firm and back) where there were none.
    h = h[h["firm"].notna()].reset_index(drop=True)
    if h.empty:
        return None

    g = h.groupby("ticker", sort=False)
    prev = g["firm"].shift(1)
    paired = prev.notna()
    changed = h["firm"] != prev
    h["auditor_changed"] = (changed & paired).astype("float64").where(paired)
    h["auditor_is_big4"] = h["firm"].isin(BIG4).astype("float64")

    # The RUN: consecutive filings naming the same firm. Point-in-time by construction -- a
    # run's start is a filing already public on every date the run covers. Keyed on the group
    # ordinal so two tickers cannot share a run id.
    breaks = (changed & paired).astype("int64")
    h["_run"] = g.ngroup().astype("str") + "|" + breaks.groupby(h["ticker"]).cumsum().astype("str")
    start = h.groupby("_run", sort=False)["as_of"].transform("min")

    since = pd.to_numeric(h["auditor_since_year"], errors="coerce") \
        if "auditor_since_year" in h.columns else pd.Series(np.nan, index=h.index)
    lo, hi = _TENURE_YEAR_BAND
    since = since.where(since.between(lo, hi))
    disclosed_start = pd.to_datetime(
        since.map(lambda y: f"{int(y):04d}-01-01" if pd.notna(y) else None), errors="coerce")
    # Disclosed where it exists, the archive run-length otherwise. `combine_first` keeps the
    # disclosed leg wherever it survived the sanity band.
    tenure_start = disclosed_start.combine_first(start)
    h["auditor_tenure_censored"] = disclosed_start.isna().astype("float64")
    h["_tenure_start_ord"] = tenure_start.map(
        lambda t: float(t.toordinal()) if pd.notna(t) else np.nan)

    tally["auditor: filings with a canonical firm"] = len(h)
    tally["auditor: changes detected (canonical firm)"] = int(
        h["auditor_changed"].fillna(0.0).sum())
    tally["auditor: tenure on the DISCLOSED basis"] = int(disclosed_start.notna().sum())
    tally["auditor: tenure on the CENSORED archive basis"] = int(disclosed_start.isna().sum())
    return h


def _auditor_fields(hist: pd.DataFrame, idx: pd.DatetimeIndex,
                    tally: dict[str, int]) -> dict[str, pd.DataFrame]:
    h = _auditor_history(hist, tally)
    if h is None:
        tally["skipped: no auditor_name -> no auditor block"] = 1
        return {}

    out: dict[str, pd.DataFrame] = {}
    for name in ("auditor_is_big4", "auditor_tenure_censored"):
        f = fundamentals_to_daily(h, name, idx)
        if not f.empty and f.notna().any().any():
            out[name] = f

    # TENURE ACCRUES DAILY off the start date, exactly as `panel._governance_fields` accrues
    # `ceo_tenure` off `ceo_since_year`: a snapshot taken at the proxy would under-report by up
    # to a year, and the accrual needs no new information.
    start = fundamentals_to_daily(h, "_tenure_start_ord", idx)
    if not start.empty and start.notna().any().any():
        today = pd.Series(idx.map(pd.Timestamp.toordinal), index=idx, dtype="float64")
        tenure = start.rsub(today, axis=0).div(_DAYS_PER_YEAR).where(lambda t: t >= 0)
        if tenure.notna().any().any():
            out["auditor_tenure"] = tenure

    changed = fundamentals_to_daily(h, "auditor_changed", idx)
    if not changed.empty and changed.notna().any().any():
        out["auditor_changed"] = _expire({"auditor_changed": changed}, h, tally)["auditor_changed"]
    return out


# --------------------------------------------------------------------------- #
# entry point                                                                  #
# --------------------------------------------------------------------------- #
def provision_fields(
    def14a: pd.DataFrame | None,
    idx: pd.DatetimeIndex,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """(daily wide frames keyed by feature name, data-quality tallies).

    `def14a` is expected to have been through `impute_def14a` already -- the caller owns the
    clean-on-read, as it does for the pay families -- and that matters here beyond coverage:
    `_annual_delta` reads the `<column>_imputed` provenance columns that function stamps.

    EACH FAMILY DEGRADES INDEPENDENTLY. A half-built archive missing `auditor_name` still gets
    its provision transitions; one missing `avg_other_public_boards` still gets the auditor
    block. Every skip is recorded with its reason, so "no feature" never looks like "feature
    built and empty".

    NO PEER LEG AND NO INTERACTION is emitted from here -- see `PEER_RELATIVE_FIELDS` for the
    measurements behind the first and D16 for the second.
    """
    tally: dict[str, int] = {}
    if def14a is None or def14a.empty or "as_of" not in def14a.columns:
        tally["skipped: no def14a -> no provision, busyness or auditor families"] = 1
        return {}, tally

    frames: dict[str, pd.DataFrame] = {}
    exported, for_counts = _provision_transitions(def14a, idx, tally)
    frames.update(exported)
    frames.update(_aggregates(for_counts, tally))
    frames.update(_independence_delta(def14a, idx, tally))
    frames.update(_busyness(def14a, idx, tally))
    frames.update(_auditor_fields(def14a, idx, tally))

    # X05's left leg, shipped as a plain LEVEL now that the product is gone (D16). It is
    # deliberately NOT `independent_chair`, which GPT §14 rules out of scope: "is the chair
    # independent" and "does the CEO hold the chair" are different questions, and only the
    # second is the entrenchment leg the dropped interaction needed. Coverage 99.3% after
    # impute, 99.7% in the modern era.
    duality = fundamentals_to_daily(def14a, "ceo_is_board_chair", idx)
    if not duality.empty and duality.notna().any().any():
        frames["ceo_is_board_chair"] = duality

    undeclared = set(frames) - ALL_FIELDS
    if undeclared:
        raise AssertionError(
            f"provision fields not declared in ALL_FIELDS: {sorted(undeclared)}")
    return frames, tally
