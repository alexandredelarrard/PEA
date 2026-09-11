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

import numpy as np
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
from src.data_aggregate.utils.governance.names import ceo_identity_changed
from src.data_aggregate.utils.governance.staleness import LEVEL_MAX_AGE_DAYS, expire_stale
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

#: THE VALIDITY GATE. The range each proxy LEVEL is allowed to occupy, as
#: `(lo, hi, lo_inclusive)`, keyed on the SOURCE column in `def14a_llm`. Measured full-table
#: on `cube_part_governance` (3,031,768 rows) on 2026-09-08; the breach that put each entry
#: here is recorded beside it, so every bound is evidence rather than taste.
#:
#: ⚠ NaN, NOT CLIP. A 4.716 equity share means the DENOMINATOR is wrong -- RCL's 2021 proxy
#: reports $3,042,000 of stock awards against a $645,000 total -- and clipping it to 1.0 would
#: publish "100% of this CEO's pay was equity" as though it had been measured. The one
#: deliberate exception in the whole governance panel is `director_cash_fee_pct`, whose breach
#: is 1.001-1.093 from a summation artefact and which IS clipped; that lives in
#: `director_comp.py` beside the code that computes it.
#:
#: ⚠ APPLIED AFTER `fundamentals_to_daily`, NOT BEFORE, and the order is the point. Gating the
#: filing row first would let the pivot's forward-fill carry the PREVIOUS proxy's value across
#: the bad filing's whole coverage period -- substituting stale data for a detected defect.
#: Gating the daily frame leaves a genuine hole, which is the honest answer: a proxy reporting
#: 14 independent directors on 11 seats tells us nothing about that year.
#:
#: ⚠ NO ENTRY CURRENTLY USES AN EXCLUSIVE LOWER BOUND (`lo_inclusive=False`). The flag is kept
#: because `ceo_pay_ratio` needed it until the TSLA evidence below overturned that, and the
#: mechanism is cheaper to keep than to re-derive; anything added here should default to
#: INCLUSIVE unless a measurement says the endpoint is impossible.
#:
#: ⚠ THE FLAGS ARE DELIBERATELY ABSENT. `ceo_is_founder` and the provision transitions are
#: 1/0, and a closed [0, 1] bound would admit 0.5 while catching nothing -- the audit's
#: `02_domains.py` FLAG block already passes on all 25 of them.
_DOMAIN: dict[str, tuple[float, float, bool]] = {
    # Shares of a whole. Both breaches are NEGATIVE at one end and > 1 at the other, which is
    # a broken denominator in both directions, not a rounding artefact.
    "ceo_equity_pay_pct":        (0.0, 1.0, True),   # 4,409 cells / 8 tickers: -0.011 .. 4.716
    "pct_independent_directors": (0.0, 1.0, True),   # 1,992 / 4: 1.006 .. 1.273 (ABT 2012-03-15
                                                     # reads 1.273 -- 14 independent on 11 seats)
    "pct_female_directors":      (0.0, 1.0, True),   # 0 cells today; inert, by definition true
    "say_on_pay_support_pct":    (0.0, 1.0, True),   # 0 cells today; catches a %-vs-fraction
                                                     # regression on the one RAW fraction
    # A board has seats. GPN 2001-08-31 and MKC 1998-02-17 each report a ONE-director board.
    # ⚠ The upper bound is INVENTED, not measured: the observed max is 33 and 60 fires on
    # nothing. It exists to catch a future 10x parse error, not to trim today's distribution.
    "board_size":                (3.0, 60.0, True),  # 792 / 3 below 3; 0 above 60
    # ⚠ ZERO IS ADMITTED, AND THAT IS A MEASURED DECISION, NOT AN OVERSIGHT. A first cut made
    # this bound exclusive on the reasoning that "a ratio needs a positive numerator". The data
    # refutes it: all 1,064 cells at exactly 0 are **TSLA**, whose proxies report Musk's total
    # comp as 0 with `ceo_salary` ALSO 0, consistently across five filings (2021, 2022, 2023,
    # 2024, 2025) and with `median_employee_pay` populated and plausible each time. He genuinely
    # takes no compensation. A CEO who is paid nothing is a real and informative governance
    # observation, and an exclusive bound blanked nothing here BUT real values.
    #
    # A domain gate may only reject the IMPOSSIBLE. Zero is possible; NEGATIVE is not, and the
    # lower bound catches that (EQT 2009 reports `ceo_total_comp = -8,920,166`).
    #
    # ⚠ THE ZEROS THAT *ARE* DEFECTS NEED A DIFFERENT INSTRUMENT, and it is phase 2's. Ford
    # 2011 reads 0 between $13.6M and $15.5M; AIZ 2009 reads 0 before $6.4M; COHR 2013 reads a
    # 0 total against a $628,000 SALARY. None of those is detectable from the value's own
    # range -- each needs the total compared against its components and against the same CEO's
    # neighbouring years, which is exactly what the post-write sanity step does. Discriminating
    # them HERE would mean reimplementing that comparison in a range check.
    #
    # ⚠ THE UPPER BOUND CANNOT CUT A WELL-PAID CEO, because this column is a MULTIPLE and not
    # a dollar amount. Measured over 981,463 cells / 491 tickers: p1 = 2.5, p50 = 183,
    # p99 = 1,965. A CEO paid $12M against a $60k median employee lands at 200. And the gap
    # around the bar is enormous -- the highest value BELOW it is TSLA's 18,043, then SBUX
    # 6,666 / WELL 6,569 / AMZN 6,474, against a bar of 100,000. Nothing whatsoever lies
    # between 18,043 and GOOGL's 197,274, so any bar in that window is equivalent today and
    # the exact choice of 1e5 is inert against real data (~6,500 is about the highest ratio an
    # S&P 500 filer has ever disclosed).
    #
    # ⚠ NO DOLLAR-DENOMINATED FIELD IS BOUNDED ANYWHERE IN THIS DICT. `ceo_total_comp`,
    # `median_employee_pay`, `log_ceo_total_comp` and `log_median_director_pay` have no upper
    # bound at all, deliberately: pay levels are unbounded above and a cap on one would be
    # exactly the kind of invented ceiling this gate exists to avoid.
    #
    # ⚠ THE UPPER BOUND CATCHES A DIFFERENT DEFECT AND IS NOT INERT: 501 cells / 1 ticker
    # (GOOGL 2018 and 2019) where `ceo_pay_ratio` == `median_employee_pay` EXACTLY (197,274
    # and 246,804) -- the extraction read the second leg of the disclosed "1 to 197,274"
    # ratio. That is D4 and phase 2 repairs it at source; this bound is the backstop. The
    # highest pay ratio ever disclosed by an S&P 500 filer is ~6,500, so 1e5 is 15x clear of
    # real data.
    #
    # ⚠ SUB-1 RATIOS ARE LEFT ALONE ON PURPOSE. 5,494 further cells across 7 tickers lie in
    # (0, 1) -- XYZ 1.46e-05, EQT 8.64e-06, SMCI 0.095, ABNB 0.65, TTWO 0.75 -- and every one
    # of them is `ceo_total_comp / median_employee_pay` computed CORRECTLY off a
    # `ceo_total_comp` of $1 or $2.75 that is really a SALARY line. The ratio is not the
    # broken leg, and nulling it here would hide the leg that is. That is phase 2's fix.
    "ceo_pay_ratio":             (0.0, 1e5, True),   # 0 KEPT (TSLA, real); 501 above 1e5
    # D12. >= 90% insider ownership is impossible AS ECONOMICS for a company with a public
    # float, and 98.3% of the 59 filings above the bar are dual-class -- the extraction took
    # the "% of total voting power" column. Insiders hold ~14% of Meta and control ~61% of its
    # votes; the archive says 99.8%.
    #
    # ⚠ THE 50-90% BAND STAYS LIVE, and this bound is CONDITIONAL -- see `_DOMAIN_ONLY_WHERE`.
    # Band alone is not the defect; band PLUS dual-class is, and LVS 2005 is the proof.
    #
    # ⚠ THIS BOUND HAS OUTLIVED ITS CAUSE AND STAYS ANYWAY. The 1,027 dual-class filings were
    # re-extracted into SEPARATE ownership and voting fields (`insider_ownership_pct` /
    # `insider_voting_pct`), so the swap this catches should no longer be produced -- and the
    # bound remains as the permanent backstop, because "the extraction was fixed once" is not
    # a property a later regression respects. `_control_wedge` is the feature that difference
    # became; this is the gate that stops the difference being read as the level.
    "insider_ownership_pct":     (0.0, 0.90, True),  # 14,689 / 16: 0.901 .. 1.0000
}

#: ⚠ THE ONE CONDITIONAL BOUND, and the condition is the whole finding rather than a caveat on
#: it. `field -> the def14a_llm column that must be TRUE before the bound applies.`
#:
#: Of the 59 filings above the 0.90 insider-ownership bar, **58 carry `dual_class_shares = 1`
#: and exactly one does not**: LVS 2005-04-29 at 0.91, where Sheldon Adelson genuinely held
#: ~88% of a single-class company after the IPO. An unconditional band would blank that one
#: real observation to catch the other 58, and raising the bar instead is a worse trade -- 0.92
#: would spare LVS but also leak ABNB 0.901, GOOGL 0.905, NKE 0.916 and DASH 0.918 straight
#: through, four defects to save one value.
#:
#: ⚠ THE CONDITION IS EVALUATED PER TICKER, NOT PER FILING, and that is the whole reason it
#: works. The flag and the value fail together: a per-class ownership figure is produced exactly
#: when the model did not read the table as per-class, and that same miss writes
#: `dual_class_shares = 0`. A per-filing condition is therefore switched off by the very defect
#: it gates. Measured on the live archive after the 2026-09-09 re-extraction, four filings remain
#: above the bar: EL 2003 (dual = 1, caught either way), LVS 2005 (single class in 23 of 23
#: filings, correctly spared) and **UHS 2024-04-04 at 0.9996 and REGN 2024-04-25 at 0.9740, both
#: flagged single class on that one filing while 27 of 31 and 26 of 31 of their own filings
#: disclose dual class**. Promoting the condition to "this filer ever disclosed dual class"
#: blanks those two and still leaves LVS standing.
#:
#: The discriminator is available exactly where it is needed: `dual_class_shares` is populated
#: on 12,335 of 12,343 filings and on **every** filing that carries an `insider_ownership_pct`
#: at all (measured 2026-09-08: zero rows with an ownership value and a null flag).
#:
#: ⚠ FAILS CLOSED. If the condition column is absent from the history handed in, the bound is
#: applied UNCONDITIONALLY -- an impossible value with no way to excuse it is still impossible.
_DOMAIN_ONLY_WHERE: dict[str, str] = {"insider_ownership_pct": "dual_class_shares"}


def _domain_condition(def14a_hist: pd.DataFrame, field: str, idx: pd.DatetimeIndex,
                      tally: dict[str, int] | None = None) -> pd.DataFrame | None:
    """The daily truth frame gating `_DOMAIN[field]`, or None when the bound is unconditional.

    Read through the SAME `fundamentals_to_daily` pivot as the value it qualifies, so the
    condition inherits the value's point-in-time discipline rather than needing its own.

    ⚠ SUBSET TO THE ROWS THAT CARRY THE VALUE FIRST, and this is not a refinement -- it is the
    difference between the gate working and leaking. `fundamentals_to_daily` forward-fills each
    column INDEPENDENTLY, so on a date where the value is stale and the condition is not, the
    two come from DIFFERENT FILINGS and the condition no longer describes the value it is
    qualifying.

    Measured on the live archive: UHS files `insider_ownership_pct = 0.996, dual_class_shares =
    1` in 1999, then `NULL, 0` in 2000. Unsubsetted, dates in 2000-2001 show the 1999 ownership
    against the 2000 flag -- and **515 cells of UHS's 0.996 sailed through the gate**. Taking
    the condition off the rows where the value is present makes it the flag of the filing whose
    value is actually on screen, because `fundamentals_to_daily` shows the last NON-NULL value.
    Verified: the leak falls from 766 cells to LVS's 251 intended ones.

    ⚠ THE COLUMN IS AVAILABLE IN PRODUCTION AND THAT IS NOT AN ACCIDENT. `def14a_llm` is
    deliberately ABSENT from `sources.SOURCE_COLUMNS`, so it loads in FULL (the table is small
    and a projection saves nothing), and `provisions_features` already reads
    `dual_class_shares` off this same frame to build `f_dual_class_added`. If a projection is
    ever added for `def14a_llm`, this column has to be in it.
    A missing condition is therefore a REGRESSION rather than a normal state, so it is tallied
    -- the bound still applies (fails closed), but the build says out loud that it stopped
    discriminating.
    """
    col = _DOMAIN_ONLY_WHERE.get(field)
    if col is None:
        return None
    if col not in def14a_hist.columns:
        if tally is not None:
            tally[f"⚠ domain gate on {field} lost its discriminator ({col}) "
                  f"-> applied unconditionally"] = 1
        return None
    qualifying = (def14a_hist[def14a_hist[field].notna()]
                  if field in def14a_hist.columns else def14a_hist)
    if qualifying.empty:
        return None
    if "ticker" in def14a_hist.columns:
        ever = ((pd.to_numeric(def14a_hist[col], errors="coerce").fillna(0.0) > 0)
                .groupby(def14a_hist["ticker"]).max())
        qualifying = qualifying.assign(
            **{col: qualifying["ticker"].map(ever).fillna(False).astype(float)})
    cond = fundamentals_to_daily(qualifying, col, idx)
    return None if cond.empty else cond


def _gate(frame: pd.DataFrame, field: str, tally: dict[str, int] | None,
          condition: pd.DataFrame | None = None) -> pd.DataFrame:
    """Blank the cells of `frame` that lie outside `_DOMAIN[field]`, and COUNT them.

    A field with no `_DOMAIN` entry is returned untouched -- the gate is opt-in, so adding a
    level to `_LEVEL_FIELDS` never silently acquires a bound nobody chose.

    `condition`, when given, RESTRICTS the bound to the cells where it is true; everything else
    is exempt. Only `insider_ownership_pct` uses it -- see `_DOMAIN_ONLY_WHERE` for the one
    filing that makes the difference.

    Never a silent drop: the cell count and the affected-ticker count go into the tally the
    panel already returns and the step already logs, so the cost of the gate is visible in the
    build output rather than inferred from a coverage diff three weeks later.
    """
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
        bad &= (condition.reindex(index=frame.index, columns=frame.columns) == 1)
    n_bad = int(bad.to_numpy().sum())
    if n_bad and tally is not None:
        tally[f"domain-gated: {field} outside "
              f"{'[' if lo_inclusive else '('}{lo:g}, {hi:g}]"] = n_bad
        tally[f"domain-gated: {field} tickers"] = int(bad.any(axis=0).sum())
    return frame.mask(bad)


def _expire(frame: pd.DataFrame, history: pd.DataFrame, field: str, feature: str,
            tally: dict[str, int] | None) -> pd.DataFrame:
    """Apply the LEVEL staleness horizon to one daily frame and COUNT what it removed.

    `field` is the column in `history` whose `as_of` dates the cell; `feature` is the emitted
    name, which is what `LEVEL_MAX_AGE_DAYS` is keyed on. The horizon is passed explicitly
    rather than left to `horizon_for` so that a feature added to `_LEVEL_FIELDS` without being
    added to `LEVEL_HORIZON_FIELDS` still gets a bound -- the opposite default from `_gate`
    above, because a missing DOMAIN is an unmeasured range while a missing HORIZON is an
    unbounded forward-fill, and the second is the defect this whole phase exists to close.

    ⚠ AFTER `_gate`, and the order is what makes the tally readable: a cell already blanked as
    out-of-domain must not also be counted as expired, or the two costs double-count.
    """
    if frame is None or frame.empty:
        return frame
    before = int(frame.notna().to_numpy().sum())
    out = expire_stale(frame, history, field, max_age_days=LEVEL_MAX_AGE_DAYS,
                       feature=feature)
    expired = before - int(out.notna().to_numpy().sum())
    if expired and tally is not None:
        tally[f"expired >{LEVEL_MAX_AGE_DAYS}d: {feature}"] = expired
        tally[f"expired >{LEVEL_MAX_AGE_DAYS}d: {feature} (of non-null)"] = before
    return out


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
#:
#: ⚠ RAW DOES NOT MEAN UNBOUNDED, and until 2026-09-08 it did. These are the only two
#: governance columns with no encoding at all -- no peer z, no self-history leg, hence no +-8
#: clip -- and a PCT change over a legitimately tiny CEO package reached **280,621,540x**
#: (GOOGL: Page's real $1 -> Pichai's real $280,621,552). Measured full-table on
#: `cube_part_governance` (2,455,297 non-null cells, 485 tickers): the **4,534 cells above
#: 100x, on 16 tickers -- 0.18% of the column** -- hold **99.999999992%** of the feature's
#: entire sum of squares. A model fitted on this column would be fitting sixteen tickers.
#:
#: The membership of this set now DRIVES a cross-sectional 1%/99% trim at the end of
#: `_governance_fields`, so the constant is live rather than documentary. Measured on the same
#: table, clipping each date to its own cross-section:
#:
#:                                    raw           trimmed     cells moved
#:     ceo_pay_growth       max  280,621,540           39.59   56,090 (2.28%)
#:                           sd    2,854,545            1.62
#:     ceo_pay_vs_rev_gro.  max  280,621,540           39.64   55,472 (2.28%)
#:                          min         -212.44         -4.01
#:
#: ⚠ IT IS WINSORIZATION AND NOT A VALIDITY GATE, on purpose. 22 of the 30 filings with
#: `ceo_total_comp <= 1` are internally consistent and most are REAL (Jobs' $1 salary 2007-11,
#: Page, Pandit, Musk's $0, Zelnick's $0). The inputs are fine; the FUNCTIONAL FORM is what
#: breaks, so the treatment belongs on the output and not on the input. The input-side half of
#: the fix is the CEO-identity guard in `_ceo_pay_growth`, which removes ratios that are not
#: quantities at all.
#:
#: ⚠ THE TRIM IS APPLIED PER SHIPPED COLUMN, not to `ceo_pay_growth` and then inherited:
#: the difference carries its own tail from the REVENUE leg (raw min -212.44, which no bound on
#: the pay leg can reach) and has to be trimmed on its own cross-section.
_RAW_ONLY_COMPUTED: frozenset[str] = frozenset({
    "ceo_pay_growth", "ceo_pay_vs_revenue_growth",
})

#: Panel-computed fields that ship RAW and are NATURALLY BOUNDED -- so they need no trim and are
#: deliberately NOT in `_RAW_ONLY_COMPUTED`, whose membership DRIVES the 1%/99% winsorization.
#:
#: `control_wedge` is insider VOTING power minus insider ECONOMIC ownership. Both legs are
#: fractions in [0, 1] and the negative side is clamped away where it is built, so the feature
#: cannot leave [0, 1] and there is nothing for a percentile trim to do. Winsorizing it would
#: only blur the one thing it exists to measure: how far a dual-class structure separates
#: control from capital.
#:
#: The set exists because `reports/validate/governance/_scripts/07_catalogue.py` builds its
#: "declared" universe from each module's field registries, so a feature computed inside
#: `_governance_fields` with NO registry entry reads as live-but-undeclared. Measured
#: 2026-09-09: `f_control_wedge` did exactly that on its first rebuild -- the feature was
#: correct, the exhaustiveness contract simply had no record of it.
_RAW_ONLY_BOUNDED: frozenset[str] = frozenset({"control_wedge"})


def _ceo_pay_growth(def14a_hist: pd.DataFrame, idx: pd.DatetimeIndex,
                    tally: dict[str, int] | None = None) -> pd.DataFrame:
    """`ceo_total_comp` growth per filing, NULLED across a CEO CHANGE, then expanded PIT.

    ⚠ WHY THIS IS NO LONGER `fiscal_change_to_daily`. The arithmetic is identical -- a
    per-ticker `pct_change` over one filing, pivoted on `as_of`, forward-filled onto the
    trading grid -- but the guard has to be applied on the DAILY grid, and it is only sound
    there if the flag is expanded from EXACTLY the rows the value is expanded from. Pivoting
    both legs out of one subset buys that. A flag pivoted from the full archive would
    forward-fill over its own gaps and present a LATER filing's flag against an EARLIER
    filing's growth -- the desync that leaked 515 cells past the phase-0 insider-ownership
    gate before it was found.

    ⚠ A CEO CHANGE IS NOT PAY GROWTH, and that is the finding. The worst cell in the
    feature -- GOOGL, 251 trading days at 280,621,551x -- is Larry Page's real $1 followed by
    Sundar Pichai's real $280,621,552. Both numbers are correct; their RATIO is not a quantity,
    because the two packages belong to two different people. Measured on the live archive:
    **920 of 7,975 filing-level growth observations (11.5%) span a transition**, expanding to
    **259,177 daily cells on 409 tickers -- 12.50% of the column's coverage**. That is a
    materially larger cut than the plan expected and it is recorded rather than softened; the
    justification is unchanged, that growth across two people is not a number.

    ⚠ IT IS NOT A SUBSTITUTE FOR THE TRIM. The guard alone takes the maximum from
    280,621,551 only to 28,095,226 and the standard deviation from 3,105,939 to 366,440,
    because ten of the twelve worst cells are a real package following a prior year filed as $0
    or $1 by the SAME person: SMCI 2025 28,095,226x off Charles Liang's $1, C 2012 14,857,102x
    off Vikram Pandit's $1, EQT 2021 7,526,514x off Toby Rice's $1, AAPL 2013 4,174,991x off
    Tim Cook's $1, and TSLA 2019 45,753x off Musk's $49,920 where BOTH figures are right. Only
    the cross-sectional trim bounds those, which is why phase 1 ships both halves.

    ⚠ AN UNKNOWN IDENTITY KEEPS THE VALUE HERE, where `pay_features._comp_history` nulls
    it. The asymmetry is deliberate: that field is the guarded one and can afford to be strict,
    this is the legacy leg whose tail the trim already bounds. The population is currently
    **zero** -- every filing carrying a `ceo_total_comp` also carries a resolvable
    `ceo_name_proxy` on both sides -- and the tally makes the policy visible if that changes.
    """
    if "ceo_total_comp" not in def14a_hist.columns or "as_of" not in def14a_hist.columns:
        return pd.DataFrame(index=idx)
    keep = [c for c in ("ticker", "as_of", "ceo_total_comp", "ceo_name_proxy")
            if c in def14a_hist.columns]
    d = def14a_hist[keep].copy()
    d["as_of"] = pd.to_datetime(d["as_of"], errors="coerce")
    d["ceo_total_comp"] = pd.to_numeric(d["ceo_total_comp"], errors="coerce")
    d = d.dropna(subset=["ticker", "as_of", "ceo_total_comp"]).sort_values(["ticker", "as_of"])
    if d.empty:
        return pd.DataFrame(index=idx)
    # `replace` on the infinities reproduces `fiscal_change_to_daily` exactly: a prior year
    # filed as $0 makes `pct_change` infinite, and an infinity is not a growth rate.
    d["chg"] = (d.groupby("ticker", sort=False)["ceo_total_comp"].pct_change(periods=1)
                .replace([np.inf, -np.inf], np.nan))
    names = (d["ceo_name_proxy"] if "ceo_name_proxy" in d.columns
             else pd.Series(None, index=d.index, dtype="object"))
    changed = ceo_identity_changed(names, d["ticker"])
    # 0.0 means "do not mask", which is ALSO the unknown-identity policy above -- so filling
    # the unknowns here cannot substitute a STALE flag for a missing one: every row of `sub`
    # carries an explicit 0 or 1, so the two pivots below have identical shape and the flag on
    # any date always belongs to the same filing as the growth beside it.
    sub = d.loc[d["chg"].notna()].assign(_turn=changed.fillna(0.0))
    if sub.empty:
        return pd.DataFrame(index=idx)
    growth = fundamentals_to_daily(sub, "chg", idx)
    turn = fundamentals_to_daily(sub, "_turn", idx)
    bad = growth.notna() & (turn.reindex(index=growth.index, columns=growth.columns) == 1.0)
    if tally is not None:
        tally["ceo_pay_growth: nulled across a CEO change"] = int(bad.to_numpy().sum())
        tally["ceo_pay_growth: tickers with a nulled transition"] = int(bad.any(axis=0).sum())
        tally["ceo_pay_growth: kept on an UNKNOWN CEO identity (filings)"] = int(
            changed.reindex(sub.index).isna().sum())
    # ⚠ EXPIRED AGAINST `sub`, NOT `d`. A growth rate belongs to the LATER of the two filings
    # it differences, and `sub` is exactly the rows where that difference exists -- so its
    # `as_of` is the date the number became knowable. Passing `d` would date a 2014-vs-2013
    # growth to whichever 2013 row happened to survive the dropna, aging it by a whole cycle.
    return _expire(growth.mask(bad), sub[["ticker", "as_of", "chg"]], "chg",
                   "ceo_pay_growth", tally)


#: The share-count column the economic ownership percentage is divided by. `sharesOutstandingPit`
#: rather than `sharesOutstanding` because the denominator has to be the count that was KNOWABLE
#: at the proxy's own date; measured 2026-09-09 it is populated on 51,504 of 51,504 rows over all
#: 491 tickers from 1995-09-01, so the join loses nothing.
_SHARES_OUTSTANDING = "sharesOutstandingPit"


def economic_ownership(def14a_hist: pd.DataFrame,
                       fundamentals: pd.DataFrame | None,
                       tally: dict[str, int] | None = None) -> pd.DataFrame:
    """COMPUTE `insider_ownership_pct` from the group's filed share count over shares outstanding.

    ⚠ FOR A DUAL-CLASS FILER THE ECONOMIC PERCENTAGE IS NOT A DISCLOSED FACT, and that is the
    finding that resolves D12 rather than a caveat on it. Alphabet's 2026 ownership table has

        Name | Class A Shares | Class A % | Class B Shares | Class B % | Total Voting Power %

    and no combined column anywhere. Every percentage in it is either per-class or a voting
    figure, so there is no cell an extraction could read that means "share of the company".
    Re-extracting under a two-column schema confirmed this the expensive way: the model returned
    per-class percentages (GOOGL 0.905-0.952 against voting 0.539-0.596) because that is all the
    filing contains.

    ⚠ BOTH LEGS OF THIS RATIO ARE FILED EVIDENCE, which is what separates it from an accounting
    identity used as a fallback. The numerator is the group's own disclosed share count, exact to
    the share (GOOGL 772,937,064; META 343,379,929), and the denominator is the filer's own
    reported shares outstanding. Only the DIVISION is ours, and it reproduces the figures the
    audit expected and the extraction could not reach:

        GOOGL  772,937,064 / 12,116,000,000 = 6.38%   (expected ~6%)
        META   343,379,929 /  2,504,000,000 = 13.7%   (expected ~14%)

    ⚠ AS-OF JOINED IN FILING SPACE, NOT DIVIDED ON THE DAILY GRID. `fundamentals_to_daily`
    forward-fills each column independently, so dividing two daily frames would put a stale
    proxy's share count over a fresh quarter's shares outstanding and let the value drift with
    every buyback between proxies. Pairing them on the proxy's own `as_of` keeps the number the
    snapshot the filer would have printed, and the PIT discipline then comes from expanding ONE
    finished column.

    Returns `def14a_hist` with `insider_ownership_pct` REPLACED where the computation is
    possible; the extracted value survives everywhere else (a single-class filer usually prints
    a real combined percentage, and that is a disclosed fact worth preferring nothing over).
    """
    if fundamentals is None or fundamentals.empty:
        return def14a_hist
    need = {"ticker", "as_of", "insider_shares"}
    if not need.issubset(def14a_hist.columns):
        return def14a_hist
    if not {"ticker", "as_of", _SHARES_OUTSTANDING}.issubset(fundamentals.columns):
        if tally is not None:
            tally[f"⚠ economic ownership not computed: {_SHARES_OUTSTANDING} absent"] = 1
        return def14a_hist

    # ⚠ BOTH KEYS FORCED TO datetime64[ns], because `merge_asof` REFUSES a resolution mismatch
    # with `MergeError: incompatible merge keys dtype('<M8[us]') and dtype('<M8[ms]')` -- and
    # the two frames come from different places, so they genuinely differ: the proxy archive
    # arrives as microseconds and the fundamentals slice as milliseconds. `pd.to_datetime` does
    # NOT normalise the unit, so this crashed the whole panel build until it was pinned.
    left = def14a_hist.copy()
    left["_as_of"] = pd.to_datetime(left["as_of"], errors="coerce").astype("datetime64[ns]")
    right = (fundamentals[["ticker", "as_of", _SHARES_OUTSTANDING]].copy()
             .assign(_as_of=lambda d: pd.to_datetime(d["as_of"], errors="coerce")
                     .astype("datetime64[ns]"))
             .drop(columns="as_of").dropna(subset=["_as_of"]))
    right = right[right[_SHARES_OUTSTANDING] > 0]
    if right.empty:
        return def14a_hist

    # ⚠ AN EXPLICIT ROW KEY, because `merge_asof` RESETS THE INDEX. A first cut aligned the
    # result back with `reindex` on the original labels against a merge output indexed 0..n-1,
    # which silently handed GOOGL's row META's percentage and META's row GOOGL's -- the values
    # were individually plausible and the tally counted the right number of them, so nothing
    # but a named-ticker probe would have caught it.
    left["_key"] = range(len(left))
    merged = pd.merge_asof(
        left.dropna(subset=["_as_of"]).sort_values("_as_of"),
        right.sort_values("_as_of"),
        on="_as_of", by="ticker", direction="backward")

    shares_out = pd.to_numeric(merged[_SHARES_OUTSTANDING], errors="coerce")
    insider = pd.to_numeric(merged["insider_shares"], errors="coerce")
    computed = insider / shares_out
    # a share of a whole: anything outside (0, 1] means the two counts are on different bases
    # (one class's shares over the total, or a count in thousands) and is not usable
    computed = computed.where((shares_out > 0) & (computed > 0) & (computed <= 1.0))

    # ⚠ CORROBORATED AGAINST THE VOTING LEG WHERE THERE IS ONE. An insider group holding
    # super-voting shares cannot control a smaller share of the votes than of the equity, so a
    # computed economic stake ABOVE the disclosed voting percentage is not a measurement of
    # anything -- it means the denominator and the numerator are on different bases.
    if "insider_voting_pct" in merged.columns:
        vote = pd.to_numeric(merged["insider_voting_pct"], errors="coerce")
        bad = computed.notna() & vote.notna() & (computed > vote + 1e-9)
        if tally is not None and int(bad.sum()):
            tally["insider_ownership_pct: computed value REJECTED (exceeds voting power)"] = int(
                bad.sum())
        computed = computed.where(~bad)

    by_key = pd.Series(computed.to_numpy(), index=merged["_key"].to_numpy())
    filled = by_key.reindex(left["_key"].to_numpy())
    filled.index = left.index

    out = left.drop(columns=["_as_of", "_key"])
    disclosed = pd.to_numeric(out["insider_ownership_pct"], errors="coerce")

    # ⚠ THE DISCLOSED PERCENTAGE WINS. THE COMPUTED ONE ONLY FILLS A HOLE, and getting this
    # precedence backwards was a real bug in the first cut of this function, caught by
    # cross-checking the computation against the filers' own arithmetic on the 49 SINGLE-class
    # filings where both exist. It agrees to a **median 0.0004 (0.04 percentage points)** and
    # lands within 1pp on 40 of 49 -- but the 9 disagreements reach 4x, and they are a defect
    # in `sharesOutstandingPit`, not in the division:
    #
    #     APH 2021-04-12   insider 14,885,716 / so_pit 149,788,355.5 = 9.94%, disclosed 2.50%
    #
    # Amphenol had ~598M shares then and has split 2:1 twice since, so its stored point-in-time
    # count is out by 4x. Bucketing the 49 pairs by whether a split intervened isolates it:
    # with NO later split the computation is right 19 of 19, and with one it is right 21 of 30.
    # (Rescaling by the cumulative split factor was tried and is WORSE overall -- 27 of 49
    # against 40 -- so the column is genuinely point-in-time and the failures are per-ticker,
    # not a uniform basis rule.)
    #
    # So the derivation is used ONLY where the filer discloses nothing, which is exactly the
    # dual-class hole it exists to fill and where no alternative exists. Where the filer does
    # print a percentage, that is filed evidence and it is preferred over our arithmetic.
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


def repair_ownership_basis(def14a_hist: pd.DataFrame,
                           tally: dict[str, int] | None = None) -> pd.DataFrame:
    """Blank `insider_ownership_pct` / `ceo_ownership_pct` where the value is a PER-CLASS
    percentage rather than an economic stake, using the voting leg as the discriminator.

    ⚠ THE DENOMINATOR IS THE SECOND HALF OF D12, AND EXTRACTION CANNOT FIX IT. Alphabet's 2026
    ownership table has the columns

        Name | Class A Shares | Class A % | Class B Shares | Class B % | Total Voting Power %

    and NO combined economic column. Larry Page's 46.5% is 389,051,160 shares over Class B's
    ~837M, not over the ~12bn shares outstanding, which would be ~3%. The figure is literally
    correct for the column it came from and is wrong by a factor of ~15 as economics -- and for
    this filer shape an insider economic percentage is **not a disclosed fact**, only a
    computable one (`insider_shares` carries the exact numerator for that).

    ⚠ THE TEST IS FREE, DETERMINISTIC AND DOES NOT NEED A THRESHOLD. An insider group holding
    SUPER-VOTING shares controls at least as large a share of the votes as of the equity, so on
    a dual-class filer `voting >= ownership` is an identity. When the stored ownership figure
    EXCEEDS the voting figure, the two are not on the same denominator, and the only way that
    happens is the ownership leg being a per-class percentage. Measured on the re-extracted
    sample: GOOGL reports ownership 0.905-0.952 against voting 0.539-0.596 across six filings
    -- an impossible ordering that the per-class reading explains exactly.

    Restricted to dual-class filers on purpose: for a single-class filer the two legs are the
    same quantity by construction, so the comparison carries no information and any inequality
    there is noise rather than a basis error.

    ⚠ BLANKED, NOT RESCALED. Rescaling would need the shares outstanding of each class at the
    filing's own date, which is a different data source and a different phase; publishing a
    computed guess in a column whose provenance says "as disclosed" is how the original defect
    happened. Unknown is the honest value, and `insider_voting_pct` -- which IS on a single
    unambiguous basis -- carries the control signal in the meantime.
    """
    need = {"insider_ownership_pct", "insider_voting_pct", "dual_class_shares"}
    if not need.issubset(def14a_hist.columns):
        return def14a_hist

    out = def14a_hist.copy()
    dual = pd.to_numeric(out["dual_class_shares"], errors="coerce") > 0
    if "ticker" in out.columns:
        # ⚠ CORROBORATE ACROSS THE FILER'S OWN HISTORY, do not trust the single filing's flag.
        # The flag and the value fail TOGETHER: a per-class ownership figure is produced exactly
        # when the model did not recognise the table as per-class, and that same miss sets
        # `dual_class_shares = 0`. Keying the repair on the filing's own flag therefore disables
        # it on precisely the filings it exists to catch. Measured after the 2026-09-09
        # re-extraction: UHS 2024-04-04 (own 0.9996 / vote 0.9080) and CCL escaped the repair on
        # their own flag while 27 of UHS's 31 filings and every other CCL filing disclose dual
        # class. Promoting to "this filer ever disclosed dual class" catches both and costs
        # nothing: LVS discloses single class in 23 of 23 filings, so its real 0.91 still stands.
        dual = dual | out["ticker"].map(
            dual.groupby(out["ticker"]).max()).fillna(False).astype(bool)
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
            tally["insider_ownership_pct: blanked (per-class basis) — tickers"] = int(
                out.loc[per_class, "ticker"].nunique()) if "ticker" in out.columns else 0
    return out


def _control_wedge(def14a_hist: pd.DataFrame, idx: pd.DatetimeIndex,
                   tally: dict[str, int] | None = None) -> pd.DataFrame:
    """Insider VOTING power minus insider ECONOMIC ownership, per filing, then expanded PIT.

    ⚠ THIS IS THE NUMBER THE D12 DEFECT WAS ACCIDENTALLY MEASURING. The extraction used to be
    told three times to read the percent-of-class column and never the voting-power column
    beside it, and on 59 filings it read the voting column anyway -- giving UHS 100% and META
    99.8% "insider ownership". Once both columns are extracted into their own fields, their
    DIFFERENCE is a governance fact in its own right: a founder controlling 61% of the votes on
    14% of the equity bears one seventh of the cost of a value-destroying decision they can
    impose unilaterally, which is a different incentive structure from one holding 14% of both.

    ⚠ COMPUTED IN FILING SPACE, NOT ON THE DAILY GRID, and that is the load-bearing choice
    rather than a style preference. `fundamentals_to_daily` forward-fills each column
    INDEPENDENTLY, so differencing two daily frames would subtract a stale filing's ownership
    from a fresh filing's voting power on every date between two proxies -- the same desync
    that leaked 515 cells of UHS's 0.996 past the phase-0 gate before `_domain_condition` was
    made to subset first. Differencing the two legs on the SAME ROW makes that impossible by
    construction, and one `fundamentals_to_daily` at the end carries the result PIT.

    ⚠ THE SINGLE-CLASS LEG IS A DERIVATION AND IS LABELLED AS ONE. A company with one class of
    stock has one vote per share, so an insider group's percent of class IS its percent of
    voting power and the wedge is exactly 0. The extraction does not write that: a null voting
    leg means "this filing's ownership table printed no voting-power column", which is the
    ordinary single-class case. Inferring the zero here rather than in `flatten.py` keeps the
    stored table pure extraction, and it is what makes the feature readable across MIXED
    VINTAGES -- only the 1,027 dual-class filings were re-extracted under the two-column
    schema, so for a single-class filing a null voting leg carries no information either way.
    `dual_class_shares` null (8 of 12,343 filings) yields a null wedge rather than a zero:
    unknown share structure is not evidence of one share, one vote.

    ⚠ A NEGATIVE WEDGE IS BLANKED, AND IT IS THE MIRRORED-CONFUSION DETECTOR. An insider group
    holding super-voting shares cannot have less voting power than economic ownership, so a
    negative difference means the model put the ownership number in the voting field or the
    reverse -- exactly the risk that asking for both columns introduces. The count reaches the
    build log rather than being silently clipped to zero.
    """
    need = {"ticker", "as_of", "insider_ownership_pct", "insider_voting_pct",
            "dual_class_shares"}
    if not need.issubset(def14a_hist.columns):
        if tally is not None:
            missing = sorted(need - set(def14a_hist.columns))
            tally[f"⚠ control_wedge not built: {', '.join(missing)} absent"] = 1
        return pd.DataFrame()

    sub = def14a_hist.loc[:, sorted(need)].copy()
    dual = pd.to_numeric(sub["dual_class_shares"], errors="coerce")
    own = pd.to_numeric(sub["insider_ownership_pct"], errors="coerce")
    vote = pd.to_numeric(sub["insider_voting_pct"], errors="coerce")

    # The SAME bounds the daily gate applies, read from `_DOMAIN` so the two cannot drift, and
    # with the same conditionality: the 0.90 ceiling is a dual-class-only bound (LVS 2005 at
    # 0.91 is a real single-class holding), while a share of a whole is a fraction either way.
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
    # ⚠ EXPIRED AGAINST `sub`, ON THE LEVEL HORIZON. A control structure is a standing fact
    # between proxies, not a dated event -- so it takes the 1,095-day level clock like the
    # other structural levels, aged on the `as_of` of the filing whose two columns produced it.
    return _expire(daily, sub[["ticker", "as_of", "control_wedge"]], "control_wedge",
                   "control_wedge", tally)


def _def14a_raw_fields(def14a_hist: pd.DataFrame,
                       idx: pd.DatetimeIndex,
                       tally: dict[str, int] | None = None) -> dict[str, pd.DataFrame]:
    """DEF 14A fields that ship RAW -- no peer z, no percentile rank (`_RAW_DEF14A_FIELDS`).

    `tally` is optional so the two callers that only want the frames (both tests) stay valid;
    the panel passes the real one so `_gate`'s cost reaches the build log.
    """
    out: dict[str, pd.DataFrame] = {}
    for src, name in _RAW_DEF14A_FIELDS:
        f = _gate(fundamentals_to_daily(def14a_hist, src, idx), src, tally,
                  _domain_condition(def14a_hist, src, idx, tally))
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
    """Daily wide frames (date x ticker), point-in-time from each proxy `as_of`.

    Every level passes through `_gate` before it is stacked, so an out-of-domain proxy value
    can reach neither the raw leg nor the `_vs_peers` leg built from the same frame.
    """
    F: dict[str, pd.DataFrame] = {}

    for src, name in _LEVEL_FIELDS:
        f = _gate(fundamentals_to_daily(def14a_hist, src, idx), src, tally,
                  _domain_condition(def14a_hist, src, idx, tally))
        f = _expire(f, def14a_hist, src, name, tally)
        if not f.empty and f.notna().any().any():
            F[name] = f

    # CEO tenure = years the CEO has led the firm at each date. Tenure accrues daily,
    # so it is the current calendar year MINUS the (point-in-time ffilled) `ceo_since_year`,
    # not a stale as_of snapshot. Guard bad extractions (start year in the future ->
    # negative); the downstream peer-relative winsorization clips any remaining outliers.
    #
    # ⚠ IT STILL EXPIRES, and on `ceo_since_year`'s filing date rather than on the tenure
    # value's own freshness -- the two are different things and only the first is knowable. A
    # tenure that accrues daily off a 2006 proxy is not a fresh measurement of a long tenure,
    # it is an unverified assumption that the same person is still in the job: `ceo_since_year`
    # reached 7,301 days (20.0 years) stale, and over that span a CEO change is the base case.
    since = fundamentals_to_daily(def14a_hist, "ceo_since_year", idx)
    if not since.empty and since.notna().any().any():
        years = pd.Series(idx.year, index=idx, dtype="float64")
        tenure = since.rsub(years, axis=0).where(lambda t: t >= 0)
        tenure = _expire(tenure, def14a_hist, "ceo_since_year", "ceo_tenure", tally)
        if tenure.notna().any().any():
            F["ceo_tenure"] = tenure

    # CEO total-comp growth (proxies are annual -> one filing per year -> periods=1), guarded
    # across a CEO change -- `_ceo_pay_growth` says why that is not `fiscal_change_to_daily`.
    pay_growth = _ceo_pay_growth(def14a_hist, idx, tally)
    if not pay_growth.empty and pay_growth.notna().any().any():
        F["ceo_pay_growth"] = pay_growth
        # pay-for-performance misalignment: CEO pay growing faster than the business.
        if fundamentals is not None and not fundamentals.empty:
            rev_growth = fiscal_change_to_daily(
                fundamentals, "totalRevenue", idx,
                kind="pct", periods=infer_yoy_periods(fundamentals))
            if not rev_growth.empty and rev_growth.notna().any().any():
                cols = pay_growth.columns.intersection(rev_growth.columns)
                if len(cols) > 0:
                    # ⚠ NOT EXPIRED AGAIN. `pay_growth` arrives already expired, and a
                    # subtraction propagates its NaNs -- so the pay leg's 1,095-day horizon
                    # already bounds this feature. Re-expiring it here would be a second
                    # identical mask whose tally entry double-counted the same cells; the
                    # revenue leg carries its own fundamentals cadence, which is not
                    # governance's to bound.
                    F["ceo_pay_vs_revenue_growth"] = pay_growth[cols] - rev_growth[cols]

    # Control versus economics: the voting-power leg minus the ownership leg. Ships RAW only —
    # it is a difference with a MEANINGFUL ZERO (zero is one share, one vote), and the panel's
    # own encoding rule says a difference whose zero is the thesis must not be re-ranked or
    # peer-standardised, which is the same argument that keeps `ceo_pay_vs_revenue_growth` raw.
    # It is deliberately absent from `_RAW_ONLY_COMPUTED` too: both legs are fractions, so the
    # difference is bounded in [0, 1] by construction and has no tail to winsorize.
    wedge = _control_wedge(def14a_hist, idx, tally)
    if not wedge.empty and wedge.notna().any().any():
        F["control_wedge"] = wedge

    # THE LAST THING THAT HAPPENS TO THE TWO UNENCODED COLUMNS. Everything else in `F` reaches
    # a bound downstream -- the peer leg winsorizes its own inputs, the self-history leg clips
    # -- so these two are the only ones that would otherwise ship a raw tail. See
    # `_RAW_ONLY_COMPUTED` for the measurement and for why the trim is per column.
    for name in _RAW_ONLY_COMPUTED & F.keys():
        F[name] = winsorize_xs(F[name])
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
    # ⚠ The tally is created HERE, not by the first family that happens to build one, because
    # `_gate` writes into it before any family runs and its counts are the visible cost of the
    # validity gate.
    tally: dict[str, int] = {}
    legacy_peers: dict[str, pd.DataFrame] = {}
    legacy_hist: dict[str, pd.DataFrame] = {}
    raw: dict[str, pd.DataFrame] = {}
    if (def14a_history is not None and not def14a_history.empty
            and "as_of" in def14a_history.columns):
        # ⚠ BEFORE ANY FAMILY READS THE ARCHIVE, and that ordering is the phase-0 lesson rather
        # than a preference: a value corrected here cannot reach the raw leg, the peer leg or
        # `_control_wedge` in its uncorrected form, whereas a per-field fix would have to be
        # repeated in three places and would be forgotten in one of them.
        # ⚠ REPAIR FIRST, THEN COMPUTE, and the order is load-bearing rather than incidental.
        # `economic_ownership` PREFERS a disclosed percentage and only fills a hole, so a
        # per-class value that reached the table would be preferred over the computed one and
        # then blanked by the repair -- losing coverage the filed share count could have
        # supplied. Blanking first turns that per-class value into exactly the hole the
        # computation exists to fill: GOOGL goes 0.922 (per-class) -> NULL -> 0.064 (economic),
        # where the other order gives 0.922 -> 0.922 -> NULL.
        def14a_history = repair_ownership_basis(def14a_history, tally)
        def14a_history = economic_ownership(def14a_history, fundamentals_history, tally)
        computed = _governance_fields(def14a_history, trading_index, fundamentals_history,
                                      tally)
        computed.update(_def14a_raw_fields(def14a_history, trading_index, tally))
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

    vote_fields, vote_tally = vote_dissent_fields(votes, trading_index)
    tally.update(vote_tally)
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
