"""
panel.py  (src/data_aggregate/utils/governance/panel.py)
--------------------------------------------------------
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

SECOND SOURCE, second time basis. `vote_dissent_features` adds the four Item 5.07 dissent
families from `sec_8k_votes` -- say-on-pay, board-wide, CEO-specific and auditor. Those are
REVEALED shareholder opinions rather than company-reported characteristics, and they are
EVENTS: each expires 548 days after its meeting (D21), where every level above persists
because a board does not stop having a size between proxies.

⚠ THE TWO SOURCES ARE ONE YEAR APART. `say_on_pay_support` above comes from the PROXY, which
discloses the PRIOR year's meeting result; `sop_dissent` comes from the 8-K filed days after
the meeting itself. Measured over 4,379 overlapping company-years, the two agree within 2pp on
93.0% of them (median |gap| 0.0031) ONCE the proxy is shifted back a year -- so they share a
denominator and differ only in latency, and the 8-K leg is the point-in-time one. Compared at
the same `as_of` they look like they disagree by 60 points (JPM 2023: 89% vs 31%), which is
the lag, not a basis conflict.
"""

from __future__ import annotations

import pandas as pd

from src.data_aggregate.utils.common.pit import (
    fiscal_change_to_daily,
    fundamentals_to_daily,
    infer_yoy_periods,
)
from src.data_aggregate.utils.common.xs import self_history_z, winsorize_xs
from src.data_aggregate.utils.common.panel import peer_relative
from src.data_aggregate.utils.governance.director_comp import (
    PEER_RELATIVE_FIELDS as DIRECTOR_PAY_PEER_FIELDS,
    director_pay_fields,
)
from src.data_aggregate.utils.governance.directors import (
    PEER_RELATIVE_FIELDS as BOARD_QUALITY_PEER_FIELDS,
    board_quality_fields,
)
from src.data_aggregate.utils.governance.pay_features import (
    PEER_RELATIVE_FIELDS as PAY_PEER_FIELDS,
    pay_fields,
)
from src.data_aggregate.utils.governance.provisions_features import (
    PEER_RELATIVE_FIELDS as PROVISION_PEER_FIELDS,
    provision_fields,
)
from src.data_aggregate.utils.governance.vote_dissent_features import (
    PEER_RELATIVE_FIELDS, vote_dissent_fields,
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

#: ⚠ `founder_ceo` LEFT `_LEVEL_FIELDS` and now ships RAW as `f_founder_ceo`, overriding D3.
#: It is a 1/0 flag, and both peer-relative encodings were degenerate on it: the peer z divides
#: by the standard deviation of a Bernoulli draw over ~7 peers, so `f_founder_ceo_vs_peers`
#: read as "how many of my peers are ALSO founder-led" and survived on only 239 tickers / 32%
#: of cells; and `rank(pct=True)` of a binary is a two-valued affine rescale whose scale wobbles
#: with the day's base rate. Measured sector share 5.8% -- firm-specific, so there is no peer
#: norm to standardize against either. Founder-led firms behave differently (long-termism, skin
#: in the game) and the raw indicator says so directly.
#:
#: ⚠ THIS MOVES A LIVE COMPOSITE. `f_founder_ceo_xs` was a member of `governance` in
#: `configs/build_cube.yml`, which averages eight [0, 1] percentile ranks; `f_founder_ceo`
#: replaces it there. A raw 0/1 has a WIDER spread than a binary's rank encoding (which
#: compresses toward the base rate), so founder-CEO now carries more weight inside that
#: composite than it did. Authorised explicitly, not a silent consequence.
_RAW_DEF14A_FIELDS: list[tuple[str, str]] = [
    ("ceo_is_founder", "founder_ceo"),
    # `say_on_pay_support` is a FRACTION with an absolute meaning -- 0.60 is a near-revolt at
    # any firm in any sector, and re-ranking it per day throws that away. Sector share 2.8%,
    # the lowest in the panel: there is no peer norm to standardize against either. ⚠ It is
    # also, per D44, a ONE-YEAR-LAGGED copy of the new raw `f_sop_dissent`, so the two are
    # near-duplicates offset in time rather than independent evidence.
    ("say_on_pay_support_pct", "say_on_pay_support"),
]

#: ⚠ THE ENCODING RULE FOR THE LEGACY BLOCK, measured 2026-09-08. Every governance feature
#: ships RAW. On top of that, exactly two encodings survived scrutiny, and `_xs` survived none.
#:
#: `_xs` IS GONE ENTIRELY. It is `rank(axis=1, pct=True)`, a per-DATE monotone map, so the
#: per-date Spearman correlation with the raw level is 1.0000 -- a within-date model cannot tell
#: them apart. Its one legitimate consumer was the `governance` composite in
#: `configs/build_cube.yml`, which averaged eight percentile ranks on a common [0, 1] scale;
#: that composite was deleted as carrying no predictive power, and a grep then found ZERO
#: remaining references to any of the eight `_xs` legs.
#:
#: `_vs_peers` SURVIVES ON FOUR, on between-sector variance share (a peer norm has to exist
#: before standardizing against one is meaningful; `profitMargins` scores 9.0% and
#: `totalRevenue` 8.1% on the same measure):
#:     board_size 12.4%  ceo_pay_ratio 12.2%  ceo_equity_pay_pct 8.6%
#:     insider_ownership_pct 7.7% -- the lowest of the four, kept because it is the most
#:     SIGN-STABLE signal in the whole panel (IC +0.0182 then +0.0138 across halves).
#: Dropped for `avg_board_tenure` (peer IC flips -0.0077 -> +0.0082), for
#: `pct_independent_directors` (4.0% sector share), and for `ceo_tenure` /
#: `pct_female_directors` -- whose IC flips sign under ALL THREE encodings, which also
#: disqualified their monotone constraints in `configs/models/lgbm_modelling.yml`.
_PEER_LEGACY: frozenset[str] = frozenset({
    "board_size", "ceo_pay_ratio", "ceo_equity_pay_pct", "insider_ownership_pct",
})

#: `_vs_hist` SURVIVES ON EXACTLY ONE. A trailing 5-year self-z asks "is this firm unusual by
#: its OWN standards", and its precondition is within-firm movement. `avg_board_tenure` is the
#: only governance field where it beat both alternatives AND held its sign across both halves
#: of the sample: **+0.0128 (2011-18) then +0.0270 (2019-26)**, while raw (-0.0120 -> +0.0128)
#: and peer (-0.0077 -> +0.0082) each flip. That is economically the right shape too -- "this
#: board is more entrenched than it has been in five years" is a deterioration signal, where
#: the absolute 7.4 years is a firm characteristic.
#:
#: ⚠ Two candidates were REJECTED after looking like winners, and the reasons are recorded so
#: they are not re-proposed:
#:   * `ceo_pay_ratio` scored IC 0.0133 on `_vs_hist` -- measured on ZERO first-half dates,
#:     because Dodd-Frank pay-ratio disclosure only begins FY2017, so its self-history exists
#:     solely from 2019 and the result has no out-of-sample at all;
#:   * `insider_ownership_pct` has a zero trailing std on 23.3% of firm-days and its
#:     `_vs_hist` INVERTS the raw sign (+0.0098/+0.0188 raw vs -0.0000/-0.0069) -- a
#:     self-history view of a near-constant holding is noise.
_VS_HIST_LEGACY: frozenset[str] = frozenset({"avg_board_tenure"})

#: Fields computed inside `_governance_fields` that ship RAW ONLY -- no peer z, no percentile.
#:
#: ⚠ BOTH ARE DIFFERENCES WITH A MEANINGFUL ZERO, and that zero IS the thesis: above 0 means
#: CEO pay is outpacing the business. Measured 2026-09-08, peer-relativizing INVERTS that flag:
#: 50.6% of `ceo_pay_vs_revenue_growth` cells are positive, and **29.3% of those get a negative
#: peer z** -- a firm overpaying by +2% inside a sector averaging +5% is encoded as "well
#: aligned" while pay outruns revenue. For `ceo_pay_growth` it is 63.0% positive and **36.9%
#: inverted**. Neither is remotely sector-driven (2.9% between-sector variance share each,
#: against 9.0% for `profitMargins`) -- the subtraction already removed the sector component.
#: `_xs` is gentler (it preserves the ordering, so only 5.4% / 20.3% cross the midpoint) but
#: still discards the zero, and with the `governance` composite deleted nothing needs the
#: bounded scale that forced a percentile encoding here.
_RAW_ONLY_COMPUTED: frozenset[str] = frozenset({
    "ceo_pay_growth", "ceo_pay_vs_revenue_growth",
})


def _def14a_raw_fields(def14a_hist: pd.DataFrame,
                       idx: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """DEF 14A fields that ship RAW -- no peer z, no percentile rank (`_RAW_DEF14A_FIELDS`)."""
    out: dict[str, pd.DataFrame] = {}
    for src, name in _RAW_DEF14A_FIELDS:
        f = fundamentals_to_daily(def14a_hist, src, idx)
        if not f.empty and f.notna().any().any():
            out[name] = f
    return out


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
    pay_growth = fiscal_change_to_daily(def14a_hist, "ceo_total_comp", idx,
                                         kind="pct", periods=1)
    if pay_growth.notna().any().any():
        F["ceo_pay_growth"] = pay_growth
        # pay-for-performance misalignment: CEO pay growing faster than the business.
        if fundamentals is not None and not fundamentals.empty:
            rev_growth = fiscal_change_to_daily(
                fundamentals, "totalRevenue", idx,
                kind="pct", periods=infer_yoy_periods(fundamentals))
            if not rev_growth.empty and rev_growth.notna().any().any():
                cols = pay_growth.columns.intersection(rev_growth.columns)
                if len(cols) > 0:
                    F["ceo_pay_vs_revenue_growth"] = pay_growth[cols] - rev_growth[cols]
    return F


def _stack(fields: dict[str, pd.DataFrame], suffix: str) -> pd.DataFrame:
    """Stack a {name: daily wide frame} dict to the long panel as `f_<name><suffix>`."""
    long_frames = []
    for name, fdf in fields.items():
        if fdf is None or fdf.empty:
            continue
        fdf = fdf.apply(pd.to_numeric, errors="coerce")
        if not fdf.notna().any().any():
            continue
        s = fdf.stack().astype("float32")
        s.index.set_names(["date", "ticker"], inplace=True)
        long_frames.append(s.rename(f"f_{name}{suffix}"))
    if not long_frames:
        return pd.DataFrame(columns=["date", "ticker"])
    return pd.concat(long_frames, axis=1).copy().reset_index()


def _peer_only(fields: dict[str, pd.DataFrame], peer_dict: dict) -> pd.DataFrame:
    """`f_<name>_vs_peers` ONLY -- the peer z without the `_xs` percentile twin.

    `build_peer_relative_panel` emits both legs and is shared by thirteen builders, so it is
    not the place to express a per-family choice; this is the same peer arithmetic (winsorized
    inputs, dispersion-floored, clipped, then trimmed cross-sectionally) with the second leg
    left off. Kept deliberately thin so the two stay in step.
    """
    rel: dict[str, pd.DataFrame] = {}
    for name, fdf in fields.items():
        if fdf is None or fdf.empty:
            continue
        fdf = fdf.apply(pd.to_numeric, errors="coerce")
        if not fdf.notna().any().any():
            continue
        rel[name] = winsorize_xs(peer_relative(fdf, peer_dict))
    return _stack(rel, "_vs_peers")


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
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Long-format governance feature panel plus the data-quality tallies of every family.

    EVERY field ships RAW as `f_<name>`. On top of that a measured minority also gets
    `f_<name>_vs_peers`, and exactly one gets `f_<name>_vs_hist`; NO field gets `_xs`, which
    was retired on 2026-09-08 (see the encoding rule above `_PEER_LEGACY`).

    THE SOURCES ARE INDEPENDENT: the proxy archive supplies the structural levels and the pay
    families, its per-NEO child the exact CPS denominator, `close_total` the performance leg,
    and `sec_8k_votes` the four dissent families. Any of them may be absent -- a database
    part-way through its first fetch has some and not others -- so none is allowed to
    short-circuit the rest, and an empty panel is returned only when they are all missing.
    """
    legacy_peers: dict[str, pd.DataFrame] = {}
    legacy_hist: dict[str, pd.DataFrame] = {}
    raw: dict[str, pd.DataFrame] = {}
    if (def14a_history is not None and not def14a_history.empty
            and "as_of" in def14a_history.columns):
        computed = _governance_fields(def14a_history, trading_index, fundamentals_history)
        computed.update(_def14a_raw_fields(def14a_history, trading_index))
        # EVERY legacy field ships raw; the two surviving encodings are additive on top.
        raw.update(computed)
        legacy_peers = {k: v for k, v in computed.items() if k in _PEER_LEGACY}
        legacy_hist = {k: v for k, v in computed.items() if k in _VS_HIST_LEGACY}

    # The three EXECUTIVE-PAY families. They read the same proxy archive as the block above,
    # but they need two sources it does not (the per-NEO child table for the exact CPS
    # denominator, and the total-return series for the performance leg), so they live in their
    # own module and declare their own encoding sets.
    pay_frames, pay_tally = pay_fields(
        def14a_history, exec_comp, fundamentals_history, close_total,
        peer_dict, trading_index)
    raw.update(pay_frames)
    pay_peers = {k: v for k, v in pay_frames.items() if k in PAY_PEER_FIELDS}

    # Provisions, board busyness and the auditor block. Same source as the legacy levels above,
    # but the unit is a CHANGE rather than a state (D16/GPT §11), so the module reads the archive
    # in filing space and diffs it there instead of on the daily grid.
    prov_frames, prov_tally = provision_fields(def14a_history, trading_index)
    raw.update(prov_frames)
    prov_peers = {k: v for k, v in prov_frames.items() if k in PROVISION_PEER_FIELDS}

    # The two PER-PERSON children (phase 6). `def14a_directors` supplies what a board AVERAGE
    # cannot express -- turnover, entrenchment, dispersion, the ISS overboarded share -- and
    # `def14a_director_comp` a pay family that no other panel reads. ⚠ The same directors frame
    # has ALREADY repaired the board averages upstream, in the step, before `impute_def14a` ran
    # (D35): that is a change to existing features' PROVENANCE, not a new family, and it is why
    # `board_busyness_delta_1y` rejects 22.5% of pairs here where phase 5 rejected 65.5%.
    quality_frames, quality_tally = board_quality_fields(directors, trading_index)
    raw.update(quality_frames)
    quality_peers = {k: v for k, v in quality_frames.items()
                     if k in BOARD_QUALITY_PEER_FIELDS}

    dpay_frames, dpay_tally = director_pay_fields(director_comp, def14a_history, trading_index)
    raw.update(dpay_frames)
    dpay_peers = {k: v for k, v in dpay_frames.items() if k in DIRECTOR_PAY_PEER_FIELDS}

    vote_fields, tally = vote_dissent_fields(votes, trading_index)
    tally.update(pay_tally)
    tally.update(prov_tally)
    tally.update(quality_tally)
    tally.update(dpay_tally)
    # EVERY vote field ships raw; only the bounded levels also get a peer leg, and none gets
    # `_xs` (see `PEER_RELATIVE_FIELDS` for the measurements behind both halves of that).
    raw.update(vote_fields)
    peers_only = {k: v for k, v in vote_fields.items() if k in PEER_RELATIVE_FIELDS}

    parts = [_peer_only({**legacy_peers, **pay_peers, **prov_peers, **quality_peers,
                         **dpay_peers, **peers_only}, peer_dict),
             _stack({k: self_history_z(v) for k, v in legacy_hist.items()}, "_vs_hist"),
             _stack(raw, "")]
    parts = [p for p in parts if not p.empty and len(p.columns) > 2]
    if not parts:
        return pd.DataFrame(columns=["date", "ticker"]), tally
    out = parts[0]
    for nxt in parts[1:]:
        out = out.merge(nxt, on=["date", "ticker"], how="outer")
    return out, tally
