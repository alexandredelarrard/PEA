"""
director_comp.py  (src/data_aggregate/utils/governance/director_comp.py)
-------------------------------------------------------------------------
The NON-EMPLOYEE DIRECTOR Compensation Table, Item 402(k): `def14a_director_comp`, 80,252
rows over 482 tickers and 7,944 filings, read by nothing in the cube before this module.

WHAT IT ADDS THAT THE CEO FAMILIES CANNOT. Executive pay measures what the board GRANTS; this
measures what the board TAKES, and the ratio between the two is a power-concentration measure —
`ceo_to_director_pay_ratio` has a p10-p90 spread of 19.2x to 79.5x around a median of 43.5x,
the dispersion of a real signal rather than of a rounding artefact.

⚠ ITS COVERAGE IS A REGIME STAIRCASE, NOT AN EXTRACTION GAP, and quoting the flat 64% headline
is reading a pre-2006 average of a post-2006 disclosure. Item 402(k) created the Director
Compensation Table in the same 2006 Reg S-K overhaul that created the SCT, so the table starts
at the 2008 proxy season by regulation:

    era      | 95-99 | 00-04 | 05-09 | 10-14 | 15-19 | 20-26
    coverage |  5.7% |  5.4% | 52.5% | 84.9% | 89.2% | 91.3%

That is **kind A** under D25: left NaN before the regime, never filled, and NEVER date-filtered
in code — no `if year < 2006` branch exists or may be added. In the era the model trades it is
~90% present.

⚠ IT DOES NOT HAVE `def14a_executive_comp`'s THREE-FISCAL-YEARS-PER-ACCESSION TRAP. 402(k)
requires the last completed fiscal year ONLY, and the measured shape agrees: `fiscal_year` is
unique per accession (6,386 accessions with one, 1,558 with none) and the primary key
`(ticker, accession_number, name)` has no fiscal-year leg. One row per director per filing, so
phase 4's grain warning does not apply and no de-duplication pass is needed.

⚠ SIX TICKERS HAVE ZERO ROWS HERE AND IT IS A PARSER DEFECT, not coverage: `APP`, `CRH`,
**`IBM`**, `PANW`, `VTRS`, `WDAY`. IBM has 31 `def14a_llm` filings, 379 director rows and 324
NEO rows, and not one director-comp row — its Item 402(k) table is not being parsed while
everything else in the same document is. This module does not fix the extraction (out of scope)
and it does not hide it either: the count is tallied at build time, because a family that is 91%
covered by filing and missing IBM entirely is not 91% covered. Written up for a `data_extract`
session in `reports/planning/2026-09-07-def14a-coverage-defects.md`.

NO TEMPORAL FILL OF PAY, ANYWHERE. Same restraint as D22 puts on the NEO grain: a director
absent from one year's table is missing, and inventing their retainer would invent the very
cross-sectional spread these features measure. The only repair is the within-row `total`
identity, which is arithmetic on disclosed components rather than an estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.pit import fundamentals_to_daily
from src.data_aggregate.utils.governance.def14a_impute import fill_total_from_components
from src.data_aggregate.utils.governance.staleness import (
    LEVEL_MAX_AGE_DAYS, expire_level_fields,
)

#: ⚠ The COLUMN PROJECTION for `def14a_director_comp` lives in `utils/common/sources.py` — one
#: registry for every cube step, asserted by `test_cube_incremental`, never a second copy here.

#: The SIX Item 402(k) components, against Item 402(c)'s seven. ⚠ The first is `fees_earned`
#: where the NEO table has `salary` and `bonus`: a non-employee director draws a retainer and
#: meeting fees, not a salary, and there is no bonus line in 402(k) at all.
DIRECTOR_COMPONENTS: tuple[str, ...] = (
    "fees_earned", "stock_awards", "option_awards",
    "non_equity_incentive", "pension_change", "other_compensation",
)

#: A ratio needs a positive denominator or it is not a ratio. Guarded at `> 0` and left NaN
#: otherwise — never zero-filled, never `inf`. A board whose disclosed median pay is 0 is a
#: parse failure, not a board that works for free.
_MIN_DENOMINATOR = 0.0

#: ⚠ THE NUMERATOR IS GUARDED AT `>= 0`, NOT `> 0`, AND THE DIFFERENCE IS THE WHOLE POINT.
#: `_ceo_ratio` previously guarded only the denominator, so a NEGATIVE `ceo_total_comp` would
#: have produced a negative pay ratio. EQT's 2009 proxy extracts
#: `ceo_total_comp = -8,920,166` against a `ceo_salary` of $649,036, so the case is real.
#: Negative pay is impossible; it is rejected here.
#:
#: ⚠ ZERO IS NOT REJECTED, even though the four tickers observed at 0 are all defects. A $0 CEO
#: total EXISTS: TSLA reports exactly that for Musk across five filings (2021-2025), with
#: `ceo_salary` also 0 and `median_employee_pay` populated each time. He takes no pay.
#:
#: The zeros in THIS column are a different story and none of them is provable from the value's
#: own range — every one needs the total compared against its components or against the same
#: CEO's neighbouring years, which is the post-write SCT sanity step's job, not a bound's:
#:
#:     F    2011  total 0, salary 0        between $13.6M (2009) and $15.5M (2012)  2,769 cells
#:     AIZ  2009  total 0, salary 0        followed by $6.4M, $7.9M, $9.4M            502 cells
#:     TTWO 2011  total 0, salary 0        neighbours are $7,051 and $11,424          501 cells
#:     COHR 2013  total 0, salary $628,000 SELF-CONTRADICTORY within the row           251 cells
#:
#: Rejecting zero here would blank all four AND any future Tesla-shaped truth, on a rule that
#: cannot tell them apart. A domain guard may only reject the IMPOSSIBLE.
_MIN_NUMERATOR = 0.0

#: The two pay-MIX shares are shares of a whole, so [0, 1], and the two are treated
#: DIFFERENTLY on purpose — this is the one deliberate clip in the governance panel.
#:
#: `director_equity_pay_pct` is BLANKED outside the range. Measured breach: **2,492 cells / 8
#: tickers spanning -0.907 to 2.951**. A share of 2.95 means `sum(stock_awards)` exceeds
#: `sum(total)` threefold, i.e. the denominator is wrong, and clipping it to 1.0 would publish
#: "this board was paid entirely in equity" as though it had been measured.
#:
#: `director_cash_fee_pct` is CLIPPED to [0, 1] and the clip is counted. Measured breach:
#: **492 cells / 2 tickers, entirely within 1.001-1.093** — traced to 41 individual director
#: rows where `fees_earned > total`, a summation-and-rounding artefact in the filed table
#: rather than a broken denominator. The values are otherwise usable and blanking 492 cells to
#: protect against a 9% overshoot would cost more than it saves. ⚠ If this breach ever exceeds
#: ~1.10 or spreads past a handful of tickers, it has stopped being a rounding artefact and the
#: clip must become a blank.
_SHARE_DOMAIN: tuple[float, float] = (0.0, 1.0)

#: Every field this module can emit — the exhaustiveness anchor.
#:
#: ⚠ `log_median_director_pay` DEVIATES from the plan's `median_director_pay` in name and in
#: basis, following phase 4's `log_ceo_total_comp` for the identical reason: a dollar level
#: spanning $149k (p10) to $360k (p90) is a scale, and every downstream consumer (the peer
#: winsorization, a linear sleeve, a monotone constraint) reads a log-dollar amount as a
#: comparable quantity and a raw one as a size proxy. The MEDIAN, not the mean, is the plan's
#: choice and it is kept: a lead-director or committee-chair retainer is a fat tail on a
#: ten-person board.
ALL_FIELDS: frozenset[str] = frozenset({
    "log_median_director_pay", "director_equity_pay_pct", "director_cash_fee_pct",
    "ceo_to_director_pay_ratio",
})

#: ⚠ NONE OF THESE IS AN EVENT (D21), and that claim STILL HOLDS. Director pay is a LEVEL — a
#: retainer structure persists between proxies, exactly as `ceo_pay_slice` and
#: `log_ceo_total_comp` do — so none of it belongs on the 548-day event clock. `EVENT_FIELDS` is
#: declared empty rather than omitted so the next reader can see the decision was taken.
#:
#: ⚠ WHAT DID NOT HOLD is the conclusion drawn from it: "not an event" was read as "nothing here
#: is aged out", and with no level horizon in the module to point at, an empty `EVENT_FIELDS`
#: meant an UNBOUNDED forward-fill. Measured cost: `f_log_median_director_pay` held ONE value
#: for 20.4 years of daily coverage on WMB and 18.5 on PSA, and Ford's
#: `f_ceo_to_director_pay_ratio` was exactly 0 for 2,769 consecutive trading days — a 2011 CEO
#: leg extracted as 0 welded to a director leg that then vanished for ten years.
#:
#: Phase 3 supplies `LEVEL_MAX_AGE_DAYS`, so the family now declares its levels AS levels and
#: expires them on the 1,095-day clock. Every field is one, which is the same statement
#: `ALL_FIELDS` makes — hence the alias rather than a second hand-maintained set that could
#: drift from it.
LEVEL_FIELDS: frozenset[str] = ALL_FIELDS

EVENT_FIELDS: frozenset[str] = frozenset()

#: ⚠ THE FIRST PEER LEGS ANY NEW FAMILY HAS EARNED SINCE PHASE 2, and they are earned on the
#: same measure that refuted every candidate in phases 3-5: the BETWEEN-SECTOR VARIANCE SHARE of
#: the level, against the four surviving legacy legs' 7.7%-12.4% and `profitMargins`' 9.0%.
#: Measured 2026-09-08 over the ten fields phase 6 adds:
#:
#:     director_cash_fee_pct      11.94%   <- peer leg
#:     director_equity_pay_pct     8.17%   <- peer leg
#:     board_age_dispersion        7.20%
#:     log_median_director_pay     6.84%
#:     board_tenure_dispersion     6.50%
#:     pct_long_tenured            5.48%
#:     ceo_to_director_pay_ratio   5.16%
#:     pct_overboarded             4.71%
#:     oldest_director_age         3.81%
#:     board_turnover              2.91%
#:
#: The two winners are the PAY-MIX shares, and that is economically the right shape: a director's
#: cash-versus-equity mix is set by a compensation consultant against a sector benchmark, so "how
#: much more cash than my sector" is a real question. `board_age_dispersion`'s 7.20% is the near
#: miss and is deliberately NOT taken -- the 7.7% floor belongs to `insider_ownership_pct`, kept
#: only because it is the most SIGN-STABLE signal in the whole panel, and a board's age spread has
#: no such record to lean on.
#:
#: ⚠ THE TWO ARE NOT ONE INFORMATION TYPE, which had to be checked before shipping both: pooled
#: correlation **-0.51** (mean per-date cross-sectional -0.47, never beyond -0.64), because
#: cash + equity sums to a median 0.98 but a p10 of only **0.66** -- option awards, pension
#: accruals and "other" take a third of the pay bill on many boards. Two complements would have
#: been one column twice over, which is the r=1.0 defect the cube already paid for once.
#:
#: `log_median_director_pay` and `ceo_to_director_pay_ratio` ship RAW: the first is a scale and
#: the second is a RATIO whose absolute level is the thesis -- 80x is entrenchment at any firm in
#: any sector, and re-centring it on a sector mean encodes the whole sector's excess as normal.
PEER_RELATIVE_FIELDS: frozenset[str] = frozenset({
    "director_cash_fee_pct", "director_equity_pay_pct",
})


def impute_director_comp(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fill a NULL per-director `total` from its six components. Returns `(copy, stats)`.

    The SAME identity `impute_exec_comp` implements, through the same helper rather than a
    second copy — the yield is small (917 rows, 98.3% -> 99.5%) and the reason to do it anyway
    is consistency of contract: `median_director_pay` divides by a `total`, and a total that is
    sometimes repaired and sometimes not would make its coverage depend on which grain a reader
    happened to look at.

    Non-destructive; `total_imputed` stamped on every row.
    """
    return fill_total_from_components(df, DIRECTOR_COMPONENTS)


def _per_filing_pay(dc: pd.DataFrame, tally: dict[str, int]) -> pd.DataFrame | None:
    """One row per filing: the median director total and the two pay-MIX shares.

    ⚠ THE SHARES ARE BOARD-LEVEL SUMS, not means of per-director ratios. A single director who
    joined in March draws a pro-rated $20k fee against a $0 stock award, and a mean of ratios
    lets that one row move the board's equity share by ten points; `sum(stock) / sum(total)` is
    the share of the board's whole pay bill that came as equity, which is the quantity the
    alignment argument is actually about. Both are computable; only one is stable.
    """
    need = {"ticker", "as_of", "total"}
    if dc is None or dc.empty or not need <= set(dc.columns):
        tally["skipped: no def14a_director_comp -> no director-pay family"] = 1
        return None
    d = dc.copy()
    d["as_of"] = pd.to_datetime(d["as_of"], errors="coerce")
    d["total"] = pd.to_numeric(d["total"], errors="coerce")
    d = d.dropna(subset=["ticker", "as_of"])
    if d.empty:
        return None
    keys = ["ticker", "as_of"]
    g = [d[k] for k in keys]
    out = pd.DataFrame({
        "median_director_pay": d["total"].groupby(g, sort=False).median(),
        "n_directors_paid": d["total"].notna().groupby(g, sort=False).sum(),
    })
    total_sum = d["total"].groupby(g, sort=False).sum(min_count=1)
    for name, col in (("director_equity_pay_pct", "stock_awards"),
                      ("director_cash_fee_pct", "fees_earned")):
        if col not in d.columns:
            tally[f"skipped: no {col} -> no {name}"] = 1
            continue
        part = pd.to_numeric(d[col], errors="coerce").groupby(g, sort=False).sum(min_count=1)
        share = part / total_sum.where(total_sum > _MIN_DENOMINATOR)
        lo, hi = _SHARE_DOMAIN
        outside = share.notna() & ((share < lo) | (share > hi))
        if name == "director_cash_fee_pct":
            # The deliberate exception -- clipped, not blanked. See `_SHARE_DOMAIN`.
            share = share.clip(lower=lo, upper=hi)
            tally[f"{name}: clipped into [0, 1]"] = int(outside.sum())
        else:
            share = share.where(~outside)
            tally[f"{name}: blanked outside [0, 1]"] = int(outside.sum())
        out[name] = share
    # The groupby was on named Series, so `reset_index` restores `ticker` / `as_of` by name.
    out = out.reset_index()
    tally["director-pay filings"] = len(out)
    tally["director-pay tickers"] = int(out["ticker"].nunique())
    tally["director-pay: median board size paid"] = int(out["n_directors_paid"].median())
    return out


def _ceo_ratio(pay: pd.DataFrame, def14a: pd.DataFrame | None,
               tally: dict[str, int]) -> pd.DataFrame | None:
    """`ceo_total_comp / median_director_pay`, joined WITHIN a filing.

    ⚠ IT CROSSES TWO TABLES and that is the only thing to get right about it: `ceo_total_comp`
    comes off the parent proxy row and the median off its 402(k) child, both keyed on the same
    `(ticker, as_of)` filing, so there is no time alignment to get wrong. What it DOES inherit is
    both parents' missingness, which is why it lands at ~59.5% against the family's 64% — the
    intersection of two independently incomplete disclosures, not a join defect.

    ⚠ THE NUMERATOR IS GUARDED TOO, but at `>= 0` rather than `> 0` — see `_MIN_NUMERATOR`
    for why a $0 CEO total is kept (TSLA reports one, five years running) while a NEGATIVE one
    is rejected (EQT 2009 extracts -$8.9M).
    """
    if def14a is None or def14a.empty or "ceo_total_comp" not in def14a.columns:
        tally["skipped: no ceo_total_comp -> no ceo_to_director_pay_ratio"] = 1
        return None
    p = def14a[["ticker", "as_of", "ceo_total_comp"]].copy()
    p["as_of"] = pd.to_datetime(p["as_of"], errors="coerce")
    p["ceo_total_comp"] = pd.to_numeric(p["ceo_total_comp"], errors="coerce")
    p = p.dropna(subset=["ticker", "as_of"])
    m = pay[["ticker", "as_of", "median_director_pay"]].merge(p, on=["ticker", "as_of"],
                                                              how="inner")
    if m.empty:
        tally["skipped: no filing carries both a CEO total and a director median"] = 1
        return None
    den = m["median_director_pay"].where(m["median_director_pay"] > _MIN_DENOMINATOR)
    num = m["ceo_total_comp"].where(m["ceo_total_comp"] >= _MIN_NUMERATOR)
    m["ceo_to_director_pay_ratio"] = num / den
    m = m.replace([np.inf, -np.inf], np.nan)
    tally["ceo_to_director_pay_ratio: filings with both legs"] = int(
        m["ceo_to_director_pay_ratio"].notna().sum())
    tally["ceo_to_director_pay_ratio: rejected (median <= 0)"] = int(
        (m["median_director_pay"].notna() & den.isna()).sum())
    tally["ceo_to_director_pay_ratio: rejected (ceo total < 0)"] = int(
        (m["ceo_total_comp"].notna() & num.isna()).sum())
    return m[["ticker", "as_of", "ceo_to_director_pay_ratio"]]


def director_pay_fields(director_comp: pd.DataFrame | None,
                        def14a: pd.DataFrame | None,
                        idx: pd.DatetimeIndex,
                        ) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """The four director-pay features (D42) as daily wide frames, plus the tallies.

    `director_comp` is expected to have been through `impute_director_comp` already — the caller
    owns the clean-on-read, exactly as it does for the NEO table.

    EVERY FIELD DEGRADES INDEPENDENTLY: a filing with a median but no CEO total still gets the
    three mix / level features, and an archive with no `stock_awards` column still gets the
    cash-fee share. Every skip is tallied with its reason, so "no feature" never reads as
    "feature built and empty".
    """
    tally: dict[str, int] = {}
    pay = _per_filing_pay(director_comp, tally)
    if pay is None:
        return {}, tally

    ratio = _ceo_ratio(pay, def14a, tally)
    if ratio is not None:
        pay = pay.merge(ratio, on=["ticker", "as_of"], how="left")

    # A log-dollar level: see `ALL_FIELDS` for why the plan's raw `median_director_pay` ships
    # logged. `> 0` guards the log itself; a non-positive median is a parse failure.
    pay["log_median_director_pay"] = np.log(
        pay["median_director_pay"].where(pay["median_director_pay"] > _MIN_DENOMINATOR))

    frames: dict[str, pd.DataFrame] = {}
    for name in sorted(ALL_FIELDS):
        if name not in pay.columns:
            continue
        h = pay[["ticker", "as_of", name]]
        if not h[name].notna().any():
            tally[f"skipped: {name} all null"] = 1
            continue
        daily = fundamentals_to_daily(h, name, idx)
        if daily.empty or not daily.notna().any().any():
            tally[f"skipped: {name} empty on the daily grid"] = 1
            continue
        frames[name] = daily
        tally[f"{name}: filings"] = int(h[name].notna().sum())

    undeclared = set(frames) - ALL_FIELDS
    if undeclared:
        raise AssertionError(f"director-pay fields not in ALL_FIELDS: {sorted(undeclared)}")

    # THE LEVEL HORIZON. Expired against `pay` -- this family's OWN per-filing frame -- so the
    # age of a director-pay cell is the age of the director-comp table that disclosed it, not
    # of whatever else the same proxy happened to contain. That distinction is the Ford case:
    # its `def14a_llm` row kept arriving every year while the 402(k) table was absent for ten,
    # and dating these cells off the proxy would have called a ten-year-old retainer fresh.
    frames, expiry = expire_level_fields(frames, pay, LEVEL_FIELDS)
    for name, (expired, before) in expiry.items():
        if expired:
            tally[f"expired >{LEVEL_MAX_AGE_DAYS}d: {name}"] = expired
            tally[f"non-null before expiry: {name}"] = before
    return frames, tally
