"""
THE VALIDITY GATE between the LLM proxy extraction and the governance feature columns
(`panel._DOMAIN` / `panel._gate`, plus the three director-side guards in `director_comp`).

Every fixture value below is a REAL breach measured on `cube_part_governance` (3,031,768 rows)
on 2026-09-08, named by ticker and filing date, so a future reader can go back to the filing:

    ABT   2012-03-15  pct_independent_directors = 1.273   -- 14 independent on an 11-seat board
    RCL   2021-04-22  ceo_equity_pay_pct        = 4.716   -- $3,042,000 of stock, $645,000 total
    GPN   2001-08-31  board_size                = 1
    MKC   1998-02-17  board_size                = 1
    GOOGL 2018-04-27  ceo_pay_ratio             = 197274  -- EXACTLY median_employee_pay: the
                                                             extraction read the second leg of
                                                             the disclosed "1 to 197,274" ratio
    META  2024        insider_ownership_pct     = 0.998   -- the "% of total voting power" column
    EQT   2009-03-05  ceo_total_comp            = -8.92M  -- negative pay, against a $649k salary

⚠ AND TWO VALUES THAT LOOK LIKE BREACHES AND ARE NOT. Both were blanked by a first cut of this
gate, and each now has a test asserting it SURVIVES:

    TSLA  2021-2025   ceo_pay_ratio             = 0       -- REAL. Musk takes no compensation:
                                                             five consecutive filings, salary
                                                             also 0, median employee pay
                                                             populated ($34,084-$57,243) each
                                                             time. All 1,064 zero cells in the
                                                             column are this one ticker, so an
                                                             exclusive `> 0` bound blanked
                                                             nothing BUT real values.
    LVS   2005-04-29  insider_ownership_pct     = 0.91    -- REAL. Adelson genuinely held ~88%
                                                             of a SINGLE-class company. It is
                                                             the 1 filing of 59 above the 0.90
                                                             bar with `dual_class_shares = 0`,
                                                             which is why the bound is
                                                             conditional rather than a band.

THE RULE THOSE TWO SETTLE, and it governs anything added to `_DOMAIN` later:
**a domain gate may only reject the IMPOSSIBLE.** Zero pay is possible; negative pay is not.
A zero that IS a defect (Ford 2011 between $13.6M and $15.5M; COHR 2013 with a 0 total beside a
$628,000 salary) is not provable from the value's own range and belongs to the post-write SCT
sanity step, which compares a total against its components and its neighbours.

Three properties are asserted, and the SECOND is the one that makes the gate safe to ship:
  1. every breach becomes NaN (or, for the one deliberate exception, is clipped and counted),
  2. an IN-DOMAIN value is returned BIT-IDENTICAL, so the gate is purely additive,
  3. the cost is COUNTED in the tally -- never a silent drop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.common.pit import fundamentals_to_daily
from src.data_aggregate.utils.governance.director_comp import (
    director_pay_fields,
)
from src.data_aggregate.utils.governance.panel import (
    _DOMAIN,
    _DOMAIN_ONLY_WHERE,
    _LEVEL_FIELDS,
    _RAW_DEF14A_FIELDS,
    _def14a_raw_fields,
    _gate,
    _governance_fields,
)

IDX = pd.bdate_range("2012-01-02", "2012-12-31")
AS_OF = pd.Timestamp("2012-03-15")
#: The dates the single proxy actually covers. ⚠ Every ticker is NaN BEFORE `as_of` and that is
#: correct point-in-time behaviour, not the gate -- so a "did the control survive" assertion has
#: to be made on this slice or it fails on the PIT discipline instead of on the code under test.
COVERED = IDX[IDX >= AS_OF]


def _proxy(**overrides) -> pd.DataFrame:
    """One proxy row per ticker on the same `as_of`, all levels in-domain unless overridden.

    Two tickers minimum: `GOOD` is the control whose values must come through untouched, and
    the breach is applied to `BAD` so every assertion is a per-ticker comparison rather than a
    whole-frame one.
    """
    base = {
        "ceo_pay_ratio": 120.0,
        "ceo_equity_pay_pct": 0.62,
        "pct_independent_directors": 0.85,
        "pct_female_directors": 0.30,
        "board_size": 11.0,
        "avg_board_tenure": 7.4,
        "insider_ownership_pct": 0.012,
        "ceo_since_year": 2005.0,
        "ceo_total_comp": 12_000_000.0,
        "ceo_is_founder": 0.0,
        "say_on_pay_support_pct": 0.94,
    }
    rows = [{"ticker": "GOOD", "as_of": AS_OF, **base},
            {"ticker": "BAD", "as_of": AS_OF, **{**base, **overrides}}]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 1. every measured breach is blanked                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("src, name, bad_value, label", [
    ("pct_independent_directors", "pct_independent_directors", 1.273, "ABT 2012-03-15"),
    ("pct_independent_directors", "pct_independent_directors", -0.01, "negative share"),
    ("ceo_equity_pay_pct", "ceo_equity_pay_pct", 4.716, "RCL 2021-04-22"),
    ("ceo_equity_pay_pct", "ceo_equity_pay_pct", -0.011, "measured minimum"),
    ("board_size", "board_size", 1.0, "GPN 2001-08-31 / MKC 1998-02-17"),
    ("board_size", "board_size", 2.0, "measured maximum below the bar"),
    ("board_size", "board_size", 200.0, "a future 10x parse error"),
    ("ceo_pay_ratio", "ceo_pay_ratio", -1.0, "negative pay is impossible (EQT 2009: -$8.9M)"),
    ("ceo_pay_ratio", "ceo_pay_ratio", 197_274.0, "GOOGL 2018-04-27 (== median pay)"),
    ("insider_ownership_pct", "insider_ownership_pct", 0.998, "META: voting power, not equity"),
    ("insider_ownership_pct", "insider_ownership_pct", 1.0000, "UHS / WDAY / HOOD"),
    ("pct_female_directors", "pct_female_directors", 1.5, "inert bound, still enforced"),
])
def test_measured_breach_is_blanked(src, name, bad_value, label):
    """The breach goes NaN for the offending ticker and ONLY for that ticker."""
    F = _governance_fields(_proxy(**{src: bad_value}), IDX, None, {})
    assert name in F, f"{name} vanished entirely -- the gate must blank cells, not fields"
    frame = F[name]
    assert frame["BAD"].isna().all(), f"{label}: {bad_value} survived the gate"
    assert frame.loc[COVERED, "GOOD"].notna().all(), \
        f"{label}: the gate leaked onto the control ticker"


def test_say_on_pay_support_is_gated_on_the_raw_path():
    """`_def14a_raw_fields` is a gated call site too, not just the `_LEVEL_FIELDS` loop.

    The bound fires on 0 cells in production today; it is here to catch a percent-vs-fraction
    regression on the one RAW fraction in the panel.
    """
    R = _def14a_raw_fields(_proxy(say_on_pay_support_pct=94.0), IDX, {})
    assert R["say_on_pay_support"]["BAD"].isna().all()
    assert R["say_on_pay_support"].loc[COVERED, "GOOD"].notna().all()


# --------------------------------------------------------------------------- #
# 2. the gate is ADDITIVE -- this is what makes it safe to ship               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("src, name", _LEVEL_FIELDS)
def test_in_domain_values_are_bit_identical(src, name):
    """An in-domain frame must come back EXACTLY as `fundamentals_to_daily` produced it.

    Not "close" -- bit-identical. This is the assertion that proves the gate perturbs none of
    the 106 features' in-range values, which is the whole basis for shipping it without
    re-screening every feature's IC.
    """
    hist = _proxy()
    expected = fundamentals_to_daily(hist, src, IDX)
    got = _governance_fields(hist, IDX, None, {})[name]
    pd.testing.assert_frame_equal(got, expected, check_exact=True)


@pytest.mark.parametrize("src, _name", _RAW_DEF14A_FIELDS)
def test_gate_is_a_noop_without_a_domain_entry(src, _name):
    """A field with no `_DOMAIN` entry is returned untouched -- the gate is opt-in.

    `ceo_is_founder` is the live example: it is a 1/0 flag and deliberately has no entry, so
    adding a level to `_RAW_DEF14A_FIELDS` never silently acquires a bound nobody chose.
    """
    frame = fundamentals_to_daily(_proxy(), src, IDX)
    tally: dict[str, int] = {}
    got = _gate(frame, src, tally)
    if src not in _DOMAIN:
        assert got is frame, f"{src} has no _DOMAIN entry but _gate copied the frame"
        assert tally == {}
    else:
        pd.testing.assert_frame_equal(got, frame, check_exact=True)


def test_a_well_paid_ceo_is_not_cut_by_the_1e5_bar():
    """The bar is on a MULTIPLE, not on dollars, and it cannot touch a realistic pay ratio.

    Measured over 981,463 cells / 491 tickers: p1 = 2.5, p50 = 183, p99 = 1,965. The highest
    value below the bar is TSLA's 18,043, then SBUX 6,666 / WELL 6,569 / AMZN 6,474 — against a
    bar of 100,000. Nothing lies between 18,043 and GOOGL's 197,274.

    ⚠ No dollar-denominated field is bounded at all: `ceo_total_comp`, `median_employee_pay`,
    `log_ceo_total_comp` and `log_median_director_pay` carry no upper bound anywhere.
    """
    # The realistic top of the distribution, plus TSLA's outlier, all survive.
    for ratio in (183.0, 1_965.0, 6_474.0, 18_043.0, 99_999.0):
        F = _governance_fields(_proxy(ceo_pay_ratio=ratio), IDX, None, {})
        kept = F["ceo_pay_ratio"].loc[COVERED, "BAD"]
        assert kept.notna().all(), f"the 1e5 bar cut a legitimate ratio of {ratio:,.0f}"
        assert (kept == ratio).all()
    # No pay LEVEL is bounded, so a $50M CEO total passes through untouched.
    assert not {"ceo_total_comp", "median_employee_pay", "log_ceo_total_comp",
                "log_median_director_pay"} & set(_DOMAIN)


def test_zero_ceo_pay_ratio_is_KEPT():
    """⚠ A $0 CEO PAY RATIO IS REAL, and a first cut of this gate blanked it.

    Every one of the 1,064 cells at exactly 0 is **TSLA**. Its proxies report Musk's
    `ceo_total_comp` as 0 with `ceo_salary` ALSO 0, across five consecutive filings
    (2021-08-26, 2022-06-23, 2023-04-06, 2024-04-29, 2025-09-17), each with a populated and
    plausible `median_employee_pay` ($34,084–$57,243). He genuinely takes no compensation.

    So the exclusive `> 0` bound this test replaced was blanking nothing BUT real values, and
    "a pay ratio is positive by construction" is a false premise. A domain gate rejects the
    IMPOSSIBLE; zero is possible, negative is not.

    The zeros that ARE defects — Ford 2011 between $13.6M and $15.5M, COHR 2013 with a $628,000
    salary against a 0 total — need the total compared against its components or its
    neighbours, which is the phase-2 sanity step, not a range check.
    """
    F = _governance_fields(_proxy(ceo_pay_ratio=0.0), IDX, None, {})
    kept = F["ceo_pay_ratio"].loc[COVERED, "BAD"]
    assert kept.notna().all(), "TSLA's genuine 0 pay ratio was blanked"
    assert (kept == 0.0).all()


def test_sub_one_pay_ratio_survives_on_purpose():
    """⚠ THE DELIBERATE NON-GATE, and it is asserted so nobody "tightens" it back.

    5,494 cells across 7 tickers carry `0 < ceo_pay_ratio < 1` -- XYZ 1.46e-05, EQT 8.64e-06,
    SMCI 0.095, ABNB 0.65, TTWO 0.75 -- and every one is `ceo_total_comp / median_employee_pay`
    computed CORRECTLY off a `ceo_total_comp` of $1 or $2.75 that is really a SALARY line.
    The ratio is not the broken leg. Raising the lower bound to 1.0 would blank 6,558 cells and
    HIDE the leg that is broken, which phase 2 repairs at source.
    """
    F = _governance_fields(_proxy(ceo_pay_ratio=8.64e-06), IDX, None, {})
    assert F["ceo_pay_ratio"].loc[COVERED, "BAD"].notna().all()


# --------------------------------------------------------------------------- #
# 3. the cost is COUNTED                                                       #
# --------------------------------------------------------------------------- #
def test_insider_ownership_bound_applies_only_to_dual_class():
    """⚠ LVS 2005-04-29: 0.91 insider ownership on a SINGLE-class company, and it is REAL.

    Sheldon Adelson genuinely held ~88% after the IPO. It is the ONE row of the 59 above the
    0.90 bar with `dual_class_shares = 0`, so an unconditional band would blank a real
    observation to catch the other 58.
    """
    hist = _proxy(insider_ownership_pct=0.91)
    hist["dual_class_shares"] = [1.0, 0.0]          # GOOD is dual-class, BAD (the "LVS") is not
    tally: dict[str, int] = {}
    F = _governance_fields(hist, IDX, None, tally)
    assert F["insider_ownership_pct"].loc[COVERED, "BAD"].notna().all(), \
        "the single-class 0.91 was blanked -- LVS 2005 is a real value"
    assert not [k for k in tally if k.startswith("domain-gated: insider_ownership_pct")]


def test_insider_ownership_bound_fires_on_a_dual_class_filer():
    """META 0.998 with the dual-class flag set: blanked, and counted."""
    hist = _proxy(insider_ownership_pct=0.998)
    hist["dual_class_shares"] = [0.0, 1.0]          # only BAD is dual-class
    tally: dict[str, int] = {}
    F = _governance_fields(hist, IDX, None, tally)
    assert F["insider_ownership_pct"]["BAD"].isna().all()
    assert F["insider_ownership_pct"].loc[COVERED, "GOOD"].notna().all()
    assert tally["domain-gated: insider_ownership_pct tickers"] == 1


def test_economic_ownership_is_COMPUTED_from_the_filed_share_count():
    """⚠ THE FIX FOR D12, and it is arithmetic on two filed numbers rather than an extraction.

    For a dual-class filer the economic percentage is **not a disclosed fact**. Alphabet's 2026
    ownership table prints `Class A Shares | Class A % | Class B Shares | Class B % | Total
    Voting Power %` and no combined column, so there is no cell any extraction could read that
    means "share of the company" — which is why re-extracting under a two-column schema still
    returned per-class figures (GOOGL 0.905-0.952 against voting 0.539-0.596).

    What IS disclosed, exactly, is the group's share COUNT. Dividing it by the filer's own
    reported shares outstanding reproduces the numbers this plan expected and the extraction
    could not reach:

        GOOGL  772,937,064 / 12,097,000,000 = 6.39%    (phase 5 expected ~6%)
        META   343,379,929 /  2,504,000,000 = 13.71%   (phase 5 expected ~14%, audit found 99.8%)

    Both legs are filed evidence; only the division is ours. `sharesOutstandingPit` is the
    denominator because it is the count knowable at the proxy's date, and the join is AS-OF in
    filing space — dividing two independently forward-filled daily frames would put a stale
    proxy's share count over a fresh quarter's outstanding and drift with every buyback.
    """
    from src.data_aggregate.utils.governance.panel import economic_ownership

    hist = pd.DataFrame([
        # the refined schema returns NULL for a per-class table, so the computation fills a
        # genuine hole rather than overwriting a disclosed number
        {"ticker": "GOOGL", "as_of": pd.Timestamp("2026-04-24"),
         "insider_shares": 772_937_064.0, "insider_ownership_pct": None},
        {"ticker": "META", "as_of": pd.Timestamp("2026-04-16"),
         "insider_shares": 343_379_929.0, "insider_ownership_pct": None},
        # no share count -> the extracted value is the best available and must survive
        {"ticker": "NOSH", "as_of": pd.Timestamp("2026-04-16"),
         "insider_shares": None, "insider_ownership_pct": 0.05},
        # a count on an incompatible basis (5x the shares outstanding) -> rejected, not clipped
        {"ticker": "BADBASIS", "as_of": pd.Timestamp("2026-04-16"),
         "insider_shares": 5.0e9, "insider_ownership_pct": 0.07},
        # ⚠ THE PRECEDENCE CASE. A DISCLOSED percentage is filed evidence and beats our
        # arithmetic, which matters because `sharesOutstandingPit` is out by up to 4x on some
        # tickers (APH 2021: 9.94% computed against 2.50% disclosed, ~598M shares and two 2:1
        # splits since). Measured over the 49 single-class filings carrying both, the two agree
        # to a median 0.0004 and within 1pp on 40 — so the derivation is right in the large and
        # the disclosed value is right always.
        {"ticker": "DISCLOSED", "as_of": pd.Timestamp("2026-04-16"),
         "insider_shares": 99_400_000.0, "insider_ownership_pct": 0.025},
    ])
    fun = pd.DataFrame([
        {"ticker": "GOOGL", "as_of": pd.Timestamp("2026-02-05"),
         "sharesOutstandingPit": 12_097_000_000.0},
        # ⚠ AFTER the proxy: an as-of join must NOT reach it
        {"ticker": "GOOGL", "as_of": pd.Timestamp("2026-04-30"),
         "sharesOutstandingPit": 12_116_000_000.0},
        {"ticker": "META", "as_of": pd.Timestamp("2026-01-29"),
         "sharesOutstandingPit": 2_504_000_000.0},
        {"ticker": "NOSH", "as_of": pd.Timestamp("2026-01-29"),
         "sharesOutstandingPit": 1_000_000.0},
        {"ticker": "BADBASIS", "as_of": pd.Timestamp("2026-01-29"),
         "sharesOutstandingPit": 1.0e9},
        # 99.4M / 1.0e9 = 9.94% computed, against 2.50% disclosed -- the APH shape
        {"ticker": "DISCLOSED", "as_of": pd.Timestamp("2026-01-29"),
         "sharesOutstandingPit": 1.0e9},
    ])
    tally: dict[str, int] = {}
    out = economic_ownership(hist, fun, tally).set_index("ticker")["insider_ownership_pct"]

    # ⚠ PER TICKER BY NAME. `merge_asof` resets the index, and a first cut realigned on the
    # original labels against a 0..n-1 result — silently giving GOOGL META's percentage and
    # META GOOGL's. Both values were individually plausible and the tally counted 2 either
    # way, so only a named-ticker assertion catches it.
    assert out["GOOGL"] == pytest.approx(772_937_064 / 12_097_000_000, rel=1e-9), \
        "GOOGL took the WRONG row's value or the post-dated share count"
    assert out["META"] == pytest.approx(343_379_929 / 2_504_000_000, rel=1e-9)
    assert out["NOSH"] == pytest.approx(0.05), "an extracted value was lost with no share count"
    assert out["BADBASIS"] == pytest.approx(0.07), "a >1 ratio was published instead of rejected"
    assert out["DISCLOSED"] == pytest.approx(0.025), \
        "our arithmetic (9.94%) overrode the FILER's own disclosed 2.50% — precedence inverted"
    assert tally["insider_ownership_pct: COMPUTED (no disclosed combined percentage)"] == 2
    assert tally["insider_ownership_pct: disclosed value PREFERRED over the computed one"] == 1

    print("\n=== SANITY CHECK: economic ownership computed from filed share counts ===")
    print(f"  GOOGL  772,937,064 / 12,097,000,000 = {out['GOOGL']:.4f}  (expected ~0.06)")
    print(f"  META   343,379,929 /  2,504,000,000 = {out['META']:.4f}  (audit found 0.998)")
    print("  GOOGL used the 2026-02-05 share count, NOT the 2026-04-30 one: as-of, not latest.")
    print(f"  DISCLOSED  computed 0.0994 vs the filer's own 0.0250 -> kept "
          f"{out['DISCLOSED']:.4f}: filed evidence beats our arithmetic.")
    print("  CONCLUSION: the percentage a dual-class proxy never prints, from two numbers it "
          "always does -- and ONLY where the filer prints nothing. Validated.")


def test_a_per_class_ownership_percentage_is_blanked_by_the_voting_leg():
    """⚠ THE SECOND HALF OF D12, AND EXTRACTION CANNOT FIX IT. Measured on Alphabet's own 2026
    proxy, whose ownership table has the columns

        Name | Class A Shares | Class A % | Class B Shares | Class B % | Total Voting Power %

    and NO combined economic column. Larry Page's `percent_of_class` of 0.465 is 389,051,160
    shares over Class B's ~837M -- literally correct for that column, and ~15x wrong as an
    economic stake (389M of ~12bn shares is ~3%). So for this filer shape an insider ECONOMIC
    percentage is not a disclosed fact at all, and the two-column split alone does not recover
    it: after re-extraction GOOGL still reported ownership 0.905-0.952 against voting
    0.539-0.596 over six filings.

    The discriminator is free and needs no threshold: an insider group holding SUPER-VOTING
    shares controls at least as large a share of the votes as of the equity, so `voting >=
    ownership` is an IDENTITY on a dual-class filer. `own > vote` therefore proves the two are
    on different denominators, which only happens when the ownership leg is per-class.
    """
    from src.data_aggregate.utils.governance.panel import repair_ownership_basis

    hist = _proxy()
    # GOOGL's real measured pair on BAD; GOOD carries a consistent (economic, voting) pair
    hist["insider_ownership_pct"] = [0.06, 0.922]
    hist["insider_voting_pct"] = [0.51, 0.543]
    hist["ceo_ownership_pct"] = [0.02, 0.465]
    hist["dual_class_shares"] = [1.0, 1.0]
    tally: dict[str, int] = {}
    out = repair_ownership_basis(hist, tally)

    bad = out[out["ticker"] == "BAD"]
    good = out[out["ticker"] == "GOOD"]
    assert bad["insider_ownership_pct"].isna().all(), "the per-class 0.922 survived"
    assert bad["ceo_ownership_pct"].isna().all(), \
        "the CEO leg came off the same columns and must inherit the basis error"
    assert bad["insider_voting_pct"].notna().all(), \
        "the voting leg is on ONE unambiguous basis and must be kept"
    assert good["insider_ownership_pct"].iloc[0] == pytest.approx(0.06), \
        "a consistent economic/voting pair was blanked"
    assert tally["insider_ownership_pct: blanked (per-class basis, own > vote)"] == 1

    print("\n=== SANITY CHECK: the per-class ownership basis ===")
    print("  GOOGL-shaped   own 0.922 > vote 0.543 -> impossible ordering -> BLANKED")
    print("  consistent     own 0.060 < vote 0.510 -> kept bit-identical")
    print("  the voting leg survives in both: it has one unambiguous denominator.")
    print("  CONCLUSION: blanked, never rescaled -- rescaling needs each class's shares "
          "outstanding at the filing date, which is a different source. Validated.")


def test_the_basis_repair_ignores_single_class_filers():
    """For a single-class filer the two legs are the same quantity by construction, so the
    comparison carries no information and any inequality is noise. LVS 2005 is why this
    matters: 0.91 economic ownership, single class, and it must survive untouched."""
    from src.data_aggregate.utils.governance.panel import repair_ownership_basis

    hist = _proxy()
    hist["insider_ownership_pct"] = [0.012, 0.91]
    hist["insider_voting_pct"] = [0.012, 0.50]      # BAD looks "impossible" but is single-class
    hist["dual_class_shares"] = [0.0, 0.0]
    tally: dict[str, int] = {}
    out = repair_ownership_basis(hist, tally)
    assert out["insider_ownership_pct"].notna().all(), \
        "a single-class filer was basis-repaired -- LVS 2005 is a real 0.91"
    assert not tally
    print("\n  single-class: the inequality is not applied, 0.91 survives. Validated.")


def test_insider_ownership_bound_fails_closed_without_the_discriminator():
    """No `dual_class_shares` column at all -> the bound applies UNCONDITIONALLY.

    An impossible value with no way to excuse it is still impossible. This is the direction
    that matters: the alternative would let a schema change silently disarm the gate.
    """
    hist = _proxy(insider_ownership_pct=0.998)
    assert "dual_class_shares" not in hist.columns
    tally: dict[str, int] = {}
    F = _governance_fields(hist, IDX, None, tally)
    assert F["insider_ownership_pct"]["BAD"].isna().all()
    # ...and it SAYS SO. A missing discriminator is a regression, not a normal state: the
    # column is in production today because `def14a_llm` has no projection in
    # `sources.SOURCE_COLUMNS` and loads in full. If that ever changes, the build log has to
    # be the thing that notices, not a coverage diff three weeks later.
    assert [k for k in tally if k.startswith("⚠ domain gate on insider_ownership_pct")], tally


def test_condition_follows_the_value_across_a_stale_forward_fill():
    """⚠ THE BUG THIS TEST EXISTS FOR: two columns forward-filled INDEPENDENTLY desynchronise.

    UHS's real history is the shape:

        1999-04-21   insider_ownership_pct = 0.996   dual_class_shares = 1
        2000-04-17   insider_ownership_pct = NULL    dual_class_shares = 0

    `fundamentals_to_daily` shows the last NON-NULL value, so dates in 2000-2001 carry the 1999
    ownership figure — but an unsubsetted condition frame carries the **2000** flag. The
    condition then describes a different filing from the value it is qualifying, and **515 cells
    of UHS's 0.996 sailed straight through the gate** before `_domain_condition` was made to
    subset to the rows that carry the value.
    """
    hist = pd.DataFrame([
        {"ticker": "UHS", "as_of": pd.Timestamp("2012-01-03"),
         "insider_ownership_pct": 0.996, "dual_class_shares": 1.0},
        {"ticker": "UHS", "as_of": pd.Timestamp("2012-06-01"),
         "insider_ownership_pct": np.nan, "dual_class_shares": 0.0},
        # A control, so the column survives the gate and the assertion is about the LEAK
        # rather than about the field vanishing.
        {"ticker": "GOOD", "as_of": pd.Timestamp("2012-01-03"),
         "insider_ownership_pct": 0.012, "dual_class_shares": 0.0},
    ])
    F = _governance_fields(hist, IDX, None, {})
    got = F["insider_ownership_pct"]
    stale = IDX[IDX >= pd.Timestamp("2012-06-01")]
    assert got.loc[stale, "UHS"].isna().all(), \
        "the stale 0.996 leaked: the condition came from a filing that did not supply the value"
    assert got["UHS"].isna().all()
    assert got.loc[COVERED, "GOOD"].notna().all()


def test_the_conditional_bound_is_the_only_one():
    """An exhaustiveness guard: if a second conditional bound appears, it needs its own tests.

    Every other entry in `_DOMAIN` is a flat range, which is why the parametrized breach test
    above can treat them uniformly.
    """
    assert set(_DOMAIN_ONLY_WHERE) == {"insider_ownership_pct"}
    assert set(_DOMAIN_ONLY_WHERE) <= set(_DOMAIN)


def test_tally_counts_cells_and_tickers():
    """Never a silent drop: the gate reports how much it blanked and on how many tickers."""
    tally: dict[str, int] = {}
    _governance_fields(_proxy(board_size=1.0, insider_ownership_pct=0.998), IDX, None, tally)

    cells = {k: v for k, v in tally.items() if k.startswith("domain-gated: board_size outside")}
    assert len(cells) == 1, tally
    # One ticker, every trading day from the proxy's `as_of` onward.
    n_covered = int((IDX >= AS_OF).sum())
    assert next(iter(cells.values())) == n_covered
    assert tally["domain-gated: board_size tickers"] == 1
    assert tally["domain-gated: insider_ownership_pct tickers"] == 1


def test_tally_is_silent_when_nothing_breaches():
    """A clean archive must not emit a gate key at all -- a 0 in the log reads as a finding."""
    tally: dict[str, int] = {}
    _governance_fields(_proxy(), IDX, None, tally)
    assert not [k for k in tally if k.startswith("domain-gated")], tally


def test_bounds_are_declared_for_every_measured_breach():
    """The exhaustiveness anchor: the seven columns the audit found breaching are all bounded.

    `ceo_to_director_pay_ratio`, `director_equity_pay_pct` and `director_cash_fee_pct` are the
    other three and are guarded in `director_comp.py`, beside the code that computes them.
    """
    assert set(_DOMAIN) >= {"ceo_equity_pay_pct", "pct_independent_directors", "board_size",
                            "ceo_pay_ratio", "insider_ownership_pct"}
    for field, (lo, hi, lo_inclusive) in _DOMAIN.items():
        assert lo < hi, field
        assert isinstance(lo_inclusive, bool), field


# --------------------------------------------------------------------------- #
# the three director-side guards                                               #
# --------------------------------------------------------------------------- #
def _director_comp(total: float, fees: float, stock: float) -> pd.DataFrame:
    """A three-director board on one filing, so the board-level SUM shares are what move."""
    return pd.DataFrame([
        {"ticker": "BAD", "as_of": AS_OF, "name": f"D{i}",
         "total": total, "fees_earned": fees, "stock_awards": stock}
        for i in range(3)
    ])


def test_director_cash_fee_share_is_clipped_and_counted():
    """⚠ THE ONE DELIBERATE CLIP IN THE GOVERNANCE PANEL.

    The measured breach is 1.001-1.093 on 492 cells / 2 tickers, traced to 41 director rows
    where `fees_earned > total` -- a summation-and-rounding artefact in the filed table, not a
    broken denominator. Clipping keeps 492 otherwise-usable cells; blanking them would cost
    more than a 9% overshoot does.
    """
    dc = _director_comp(total=100_000.0, fees=105_000.0, stock=0.0)
    frames, tally = director_pay_fields(dc, None, IDX)
    got = frames["director_cash_fee_pct"]["BAD"].dropna()
    assert len(got) > 0
    assert np.allclose(got, 1.0), "the mild breach must be CLIPPED, not blanked"
    assert tally["director_cash_fee_pct: clipped into [0, 1]"] == 1


def test_director_equity_share_is_blanked_not_clipped():
    """A 2.95 equity share means the denominator is wrong; clipping would fabricate a fact.

    Measured breach: 2,492 cells / 8 tickers spanning -0.907 to 2.951.
    """
    dc = _director_comp(total=100_000.0, fees=0.0, stock=295_000.0)
    frames, tally = director_pay_fields(dc, None, IDX)
    assert "director_equity_pay_pct" not in frames or \
        frames["director_equity_pay_pct"]["BAD"].isna().all()
    assert tally["director_equity_pay_pct: blanked outside [0, 1]"] == 1


def test_ceo_to_director_ratio_rejects_only_a_NEGATIVE_numerator():
    """EQT 2009 extracts `ceo_total_comp = -8,920,166` against a $649,036 salary.

    Negative pay is impossible, so it is rejected. ⚠ ZERO IS NOT: TSLA reports a genuine $0
    total five years running. All four tickers observed at 0 in this column (F, AIZ, TTWO, COHR)
    are defects, but not one is provable from the value's own range — Ford's 0 sits between
    $13.6M and $15.5M, COHR's sits beside a $628,000 salary — both of which need a comparison
    this guard cannot make. Rejecting 0 here would blank those four AND any future
    Tesla-shaped truth, on a rule that cannot tell them apart.
    """
    dc = _director_comp(total=235_424.0, fees=100_000.0, stock=135_424.0)
    neg = pd.DataFrame([{"ticker": "BAD", "as_of": AS_OF, "ceo_total_comp": -8_920_166.0}])
    frames, tally = director_pay_fields(dc, neg, IDX)
    assert "ceo_to_director_pay_ratio" not in frames
    assert tally["ceo_to_director_pay_ratio: rejected (ceo total < 0)"] == 1

    zero = pd.DataFrame([{"ticker": "BAD", "as_of": AS_OF, "ceo_total_comp": 0.0}])
    frames, tally = director_pay_fields(dc, zero, IDX)
    got = frames["ceo_to_director_pay_ratio"]["BAD"].dropna()
    assert len(got) > 0 and (got == 0.0).all(), "a genuine $0 CEO total must yield a 0 ratio"
    assert tally["ceo_to_director_pay_ratio: rejected (ceo total < 0)"] == 0


def test_ceo_to_director_ratio_keeps_a_healthy_pair():
    """The control: a real pair still produces the ratio, unchanged by the new guard."""
    dc = _director_comp(total=235_424.0, fees=100_000.0, stock=135_424.0)
    def14a = pd.DataFrame([{"ticker": "BAD", "as_of": AS_OF, "ceo_total_comp": 22_800_000.0}])
    frames, tally = director_pay_fields(dc, def14a, IDX)
    got = frames["ceo_to_director_pay_ratio"]["BAD"].dropna()
    assert np.allclose(got, 22_800_000.0 / 235_424.0)
    assert tally["ceo_to_director_pay_ratio: rejected (ceo total < 0)"] == 0
    assert not np.isinf(got.to_numpy()).any()


# --------------------------------------------------------------------------- #
# the correlated failure: the flag and the value are wrong on the SAME filing
# --------------------------------------------------------------------------- #

def _two_filings(bad_dual_flag: float) -> pd.DataFrame:
    """A dual-class filer with two filings, the LATER one carrying the per-class defect.

    Shaped on UHS: an early filing that discloses dual class correctly, then a later filing
    where the model missed the per-class structure -- and therefore reported both a per-class
    ownership percentage AND `dual_class_shares = 0`.
    """
    base = {"ceo_pay_ratio": 120.0, "board_size": 11.0}
    return pd.DataFrame([
        {"ticker": "BAD", "as_of": pd.Timestamp("2012-01-04"),
         "insider_ownership_pct": 0.031, "insider_voting_pct": 0.880,
         "dual_class_shares": 1.0, **base},
        {"ticker": "BAD", "as_of": AS_OF,
         "insider_ownership_pct": 0.9996, "insider_voting_pct": 0.908,
         "dual_class_shares": bad_dual_flag, **base},
    ])


def test_the_basis_repair_corroborates_dual_class_across_the_filers_history():
    """⚠ THE FLAG AND THE VALUE FAIL TOGETHER, so the repair cannot trust the flag on its own.

    A per-class ownership figure is produced exactly when the model did not read the table as
    per-class -- and that same miss writes `dual_class_shares = 0`. Keying the repair on the
    filing's own flag disables it on precisely the filings it exists to catch. Measured on the
    live archive after the 2026-09-09 re-extraction: UHS 2024-04-04 reported own 0.9996 against
    vote 0.9080 with the flag at 0, while 27 of UHS's other 31 filings disclose dual class.
    """
    from src.data_aggregate.utils.governance.panel import repair_ownership_basis

    # the flag is RIGHT on this filing -- repaired under either keying
    honest = repair_ownership_basis(_two_filings(1.0), {})
    assert pd.isna(honest["insider_ownership_pct"].iloc[1])

    # the flag is WRONG on the very filing that carries the bad value: only the ticker-level
    # corroboration can still see that this is a dual-class filer
    tally: dict[str, int] = {}
    out = repair_ownership_basis(_two_filings(0.0), tally)
    assert pd.isna(out["insider_ownership_pct"].iloc[1]), (
        "the per-class defect survived because the SAME defect cleared its own gate -- the "
        "repair must corroborate dual class across the filer's history, not per filing")
    assert out["insider_ownership_pct"].iloc[0] == 0.031, "the honest earlier filing was blanked"
    assert tally, "a silent repair is not a repair"
    print("\n  correlated failure: 0.9996 blanked on the filer's own history. Validated.")


def test_corroboration_still_spares_a_genuinely_single_class_filer():
    """LVS discloses single class in 23 of 23 filings, so promoting the condition to "ever
    disclosed dual class" must leave its real 0.91 exactly where it is. This is the direction
    that pays for the change: if it also blanked LVS, unconditional would be simpler."""
    from src.data_aggregate.utils.governance.panel import repair_ownership_basis

    lvs = pd.DataFrame([
        {"ticker": "LVS", "as_of": pd.Timestamp("2012-01-04"), "insider_ownership_pct": 0.885,
         "insider_voting_pct": 0.500, "dual_class_shares": 0.0, "board_size": 9.0},
        {"ticker": "LVS", "as_of": AS_OF, "insider_ownership_pct": 0.910,
         "insider_voting_pct": 0.500, "dual_class_shares": 0.0, "board_size": 9.0},
    ])
    tally: dict[str, int] = {}
    out = repair_ownership_basis(lvs, tally)
    assert out["insider_ownership_pct"].notna().all(), \
        "a filer that never disclosed dual class was basis-repaired -- LVS 0.91 is real"
    assert not tally
    print("\n  never-dual filer: 0.91 survives ticker-level corroboration. Validated.")


def test_the_domain_gate_also_corroborates_across_the_history():
    """The 0.90 band gate is keyed on the same flag as the repair and inherits the same
    correlated failure, so it is promoted the same way. REGN 2024-04-25 is the case with no
    voting leg at all: own 0.9740, flag 0, and 26 of its 31 filings disclosing dual class --
    the repair cannot see it (no `vote` to compare against), so the band gate has to."""
    hist = pd.DataFrame([
        {"ticker": "BAD", "as_of": pd.Timestamp("2012-01-04"), "insider_ownership_pct": 0.028,
         "dual_class_shares": 1.0, "board_size": 11.0},
        {"ticker": "BAD", "as_of": AS_OF, "insider_ownership_pct": 0.974,
         "dual_class_shares": 0.0, "board_size": 11.0},
    ])
    tally: dict[str, int] = {}
    F = _governance_fields(hist, IDX, None, tally)
    own = F["insider_ownership_pct"]["BAD"]
    after = own.loc[own.index >= AS_OF]
    assert after.isna().all(), (
        "0.974 reached the panel: the band gate trusted the same flag that the extraction "
        "defect had already corrupted")
    before = own.loc[own.index < AS_OF].dropna()
    assert not before.empty and (before == 0.028).all(), "the in-domain earlier value was lost"
    print("\n  band gate: REGN-shaped 0.974 blanked, 0.028 kept. Validated.")
