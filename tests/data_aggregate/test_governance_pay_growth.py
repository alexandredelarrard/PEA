"""Phase 1 — the two UNENCODED governance columns, and the two halves of their fix.

`f_ceo_pay_growth` and `f_ceo_pay_vs_revenue_growth` are the only members of
`panel._RAW_ONLY_COMPUTED`: no peer z, no self-history leg, therefore no ±8 clip anywhere
downstream. Until 2026-09-08 they shipped a completely untreated tail, and the numbers below
are the live ones, not illustrations.

MEASURED ON `cube_part_governance`, full table (2,455,297 non-null cells, 485 tickers):

    max                             280,621,540
    standard deviation                2,854,545
    cells above 100x                      4,534 on 16 tickers -- 0.18% of the column
    share of the column's TOTAL sum of squares held by those 4,534 cells   99.999999992%

A model fitted on that column is fitting sixteen tickers. The fix has two halves and neither
one is sufficient alone, which is the single most important thing these tests pin:

  1. THE CEO-IDENTITY GUARD removes ratios that are not quantities. The worst cell in the
     feature -- GOOGL, 251 trading days at 280,621,551x -- is Larry Page's real $1 followed by
     Sundar Pichai's real $280,621,552. Both numbers are correct; the ratio is not a growth
     rate, because the two packages belong to two different people. Measured on the live
     archive: 920 of 7,975 filing-level growth observations span a transition, expanding to
     259,177 daily cells on 409 tickers (12.50% of the column).
  2. THE CROSS-SECTIONAL 1%/99% TRIM bounds what is left. The guard alone only takes the
     maximum from 280,621,551 to 28,095,226 and the standard deviation from 3,105,939 to
     366,440, because ten of the twelve worst cells are a real package following a prior year
     filed as $0 or $1 by the SAME person -- SMCI 2025 (Charles Liang, $1 -> $28,095,227),
     C 2012 (Vikram Pandit, $1 -> $14,857,103), EQT 2021 (Toby Rice, $1 -> $7,526,515), AAPL
     2013 (Tim Cook, $1 -> $4,174,992) and TSLA 2019 (Musk, $49,920 -> $2,284,044,884, where
     BOTH figures are right). With the trim: max 39.59, sd 1.62, 2.28% of cells moved.

⚠ IT IS WINSORIZATION AND NOT A VALIDITY GATE, and the tests assert that distinction directly.
22 of the 30 filings with `ceo_total_comp <= 1` are internally consistent and most are real
(Jobs' $1 salary 2007-11, Page, Pandit, Musk's $0, Zelnick's $0). The inputs are fine; the
FUNCTIONAL FORM is what breaks. So `test_an_ordinary_pay_year_is_bit_identical` matters as much
as the bound: a trim that moved the body of the distribution would be a different, worse bug.

⚠ THE KNOWN RESIDUAL, recorded rather than fixed. AAPL 2013's true change was about −99% (from
Cook's real $377,996,537, which the extraction stored as $1) and the trim gives it a positive
value instead — a SIGN FLIP on one ticker. Phase 2's pay-table sanity step does not catch AAPL
2012 either, because that row is perfectly self-consistent ($1 total, $1 salary, zero
components); only an external cross-check would, and none is in scope.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.common.xs import winsorize_xs
from src.data_aggregate.utils.governance.names import ceo_identity_changed
from src.data_aggregate.utils.governance.panel import (
    _RAW_ONLY_COMPUTED,
    _ceo_pay_growth,
    _governance_fields,
)

IDX = pd.bdate_range("2011-01-03", "2012-12-31")
AS_OF_1 = pd.Timestamp("2011-03-15")
AS_OF_2 = pd.Timestamp("2012-03-15")
#: The window the SECOND proxy covers, i.e. where a one-filing growth can exist at all. Before
#: `AS_OF_2` every ticker is NaN and that is correct point-in-time behaviour, not the guard.
COVERED = IDX[IDX >= AS_OF_2]

#: 100 controls, so a per-date 1%/99% quantile has something to interpolate between. They all
#: carry the SAME +15% year, which makes both quantiles exactly 0.15: the trim then has to
#: leave every control untouched and can only bite the outlier. A fixture where the controls
#: were spread out would leave "was this control clipped?" unanswerable.
N_CONTROLS = 100
ORDINARY_GROWTH = 0.15


def _proxies(bad_comp: tuple[float, float] = (1.0, 280_621_552.0),
             bad_names: tuple[str | None, str | None] = ("Larry Page", "Sundar Pichai"),
             extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """`N_CONTROLS` well-behaved filers plus `BAD`, two annual proxies each.

    `BAD` defaults to the real GOOGL pair -- a real $1 followed by a real $280,621,552 across a
    real change of CEO -- because that single cell is what the whole phase is about.
    """
    rows = []
    for i in range(N_CONTROLS):
        t = f"C{i:03d}"
        rows += [{"ticker": t, "as_of": AS_OF_1, "ceo_total_comp": 1_000_000.0,
                  "ceo_name_proxy": "Jane Q. Control", "board_size": 9.0},
                 {"ticker": t, "as_of": AS_OF_2,
                  "ceo_total_comp": 1_000_000.0 * (1.0 + ORDINARY_GROWTH),
                  "ceo_name_proxy": "Jane Q. Control", "board_size": 9.0}]
    rows += [{"ticker": "BAD", "as_of": AS_OF_1, "ceo_total_comp": bad_comp[0],
              "ceo_name_proxy": bad_names[0], "board_size": 11.0},
             {"ticker": "BAD", "as_of": AS_OF_2, "ceo_total_comp": bad_comp[1],
              "ceo_name_proxy": bad_names[1], "board_size": 11.0}]
    df = pd.DataFrame(rows)
    if extra is not None:
        df = pd.concat([df, extra], ignore_index=True)
    return df


def _revenue(tickers: list[str]) -> pd.DataFrame:
    """One year-on-year revenue pair per ticker, so `infer_yoy_periods` returns 1 and
    `ceo_pay_vs_revenue_growth` is actually built. Flat revenue keeps the subtraction
    transparent: the difference column then equals the pay leg exactly."""
    rows = []
    for t in tickers:
        rows += [{"ticker": t, "as_of": AS_OF_1, "totalRevenue": 1.0e9},
                 {"ticker": t, "as_of": AS_OF_2, "totalRevenue": 1.0e9}]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# half 1 -- the CEO-identity guard                                            #
# --------------------------------------------------------------------------- #

def test_a_ceo_change_is_not_pay_growth():
    """The GOOGL cell, exactly as filed. $1 -> $280,621,552 across a change of CEO is NaN."""
    g = _ceo_pay_growth(_proxies(), IDX)
    assert g.loc[COVERED, "BAD"].isna().all(), "the transition was not guarded"
    assert np.allclose(g.loc[COVERED, "C000"], ORDINARY_GROWTH), "a control was disturbed"


def test_the_same_ceo_keeps_the_value_however_extreme():
    """SMCI's shape: a real $28M package following a prior year filed as $1, SAME person.

    The guard must NOT fire here -- it is not an outlier detector -- which is precisely why
    the trim has to exist as well. 28,095,226 is the guard-only maximum measured on the live
    archive."""
    g = _ceo_pay_growth(_proxies(bad_comp=(1.0, 28_095_227.0),
                                 bad_names=("Charles Liang", "Charles Liang")), IDX)
    assert np.allclose(g.loc[COVERED, "BAD"], 28_095_226.0), "the guard over-fired"


def test_a_zero_prior_year_is_NaN_not_infinity():
    """A prior year filed as $0 makes `pct_change` infinite, and an infinity is not a growth
    rate. Six live filings have this shape -- AIZ 2010, COHR 2014, F 2012 (Alan Mulally,
    $0 -> $15,499,993), FAST 2007, TTWO 2013 and VRTX 2008 -- and reproducing
    `fiscal_change_to_daily`'s `replace([inf, -inf], nan)` is what keeps them out.

    ⚠ THE TICKER DROPS OUT OF THE FRAME ENTIRELY here rather than appearing as a NaN column,
    because its only growth observation is the infinity and the pivot is built from the rows
    that survive. Asserted both ways so either outcome counts, and so a future `inf` leak
    fails loudly instead of hiding in a missing column."""
    g = _ceo_pay_growth(_proxies(bad_comp=(0.0, 15_499_993.0),
                                 bad_names=("Alan Mulally", "Alan Mulally")), IDX)
    assert "BAD" not in g.columns or g.loc[COVERED, "BAD"].isna().all()
    v = g.loc[COVERED].to_numpy(dtype="float64").ravel()
    assert np.isfinite(v[~np.isnan(v)]).all(), "an infinity reached the daily grid"
    assert np.allclose(g.loc[COVERED, "C000"], ORDINARY_GROWTH), "a control was disturbed"


def test_an_unknown_identity_keeps_the_value_and_is_counted():
    """This leg's asymmetry with `pay_features._comp_history`, asserted rather than described.

    That field nulls an unknown identity because it is the guarded one; this one keeps it
    because the trim already bounds its tail, and the tally is what makes the policy visible.
    The live population is currently zero -- every filing carrying a `ceo_total_comp` also
    carries a resolvable `ceo_name_proxy` on both sides -- so only a test can pin the branch.
    """
    tally: dict[str, int] = {}
    g = _ceo_pay_growth(_proxies(bad_names=("Larry Page", None)), IDX, tally)
    assert g.loc[COVERED, "BAD"].notna().all(), "an unknown identity must not blank the value"
    assert tally["ceo_pay_growth: kept on an UNKNOWN CEO identity (filings)"] == 1
    assert tally["ceo_pay_growth: nulled across a CEO change"] == 0


def test_the_flag_cannot_be_borrowed_from_another_filing():
    """THE DESYNC TEST, and the reason this builder is not `fiscal_change_to_daily`.

    `fundamentals_to_daily` forward-fills each column over the rows that CARRY it. A flag
    pivoted from a wider row set than its value therefore drifts out of step with it -- the bug
    that leaked 515 cells past the phase-0 insider-ownership gate. Here the value and the flag
    come out of one subset, so a THIRD filing under the new CEO must un-mask the column: the
    growth from Pichai's first year to his second is a real growth rate.
    """
    as_of_3 = pd.Timestamp("2012-06-15")
    extra = pd.DataFrame([{"ticker": "BAD", "as_of": as_of_3,
                           "ceo_total_comp": 322_214_785.0,
                           "ceo_name_proxy": "Sundar Pichai", "board_size": 11.0}])
    g = _ceo_pay_growth(_proxies(extra=extra), IDX)
    masked = IDX[(IDX >= AS_OF_2) & (IDX < as_of_3)]
    assert g.loc[masked, "BAD"].isna().all(), "the transition itself must stay masked"
    live = IDX[IDX >= as_of_3]
    assert g.loc[live, "BAD"].notna().all(), "the flag was carried past its own filing"
    assert np.allclose(g.loc[live, "BAD"], 322_214_785.0 / 280_621_552.0 - 1.0)


def test_the_guard_shares_one_definition_with_the_pay_family():
    """`names.ceo_identity_changed` is the ONLY definition of "the CEO changed", so the legacy
    leg and the guarded `ceo_comp_growth_1y` can never disagree about a transition. Spelling
    drift is the thing it has to survive: comparing raw strings manufactures 356 spurious
    turnovers out of 1,625 on the live archive."""
    h = pd.DataFrame({"ticker": ["X", "X", "X", "Y", "Y"],
                      "n": ["Timothy D. Cook", "Timothy Cook", "Tim Cook",
                            "Larry Page", "Sundar Pichai"]})
    changed = ceo_identity_changed(h["n"], h["ticker"])
    assert changed.tolist()[1:3] == [0.0, 0.0], "spelling drift read as a turnover"
    assert changed.iloc[4] == 1.0, "a real transition was missed"
    assert pd.isna(changed.iloc[0]) and pd.isna(changed.iloc[3]), "no prior filing -> unknown"


def test_the_tally_counts_cells_and_tickers():
    tally: dict[str, int] = {}
    _ceo_pay_growth(_proxies(), IDX, tally)
    assert tally["ceo_pay_growth: nulled across a CEO change"] == len(COVERED)
    assert tally["ceo_pay_growth: tickers with a nulled transition"] == 1


# --------------------------------------------------------------------------- #
# half 2 -- the cross-sectional trim                                          #
# --------------------------------------------------------------------------- #

def _shipped(hist: pd.DataFrame, with_revenue: bool = True) -> dict[str, pd.DataFrame]:
    rev = _revenue(sorted(hist["ticker"].unique())) if with_revenue else None
    return _governance_fields(hist, IDX, rev, {})


def test_a_same_ceo_outlier_is_bounded_by_the_trim():
    """The half the guard cannot do. SMCI's shape survives the guard at 28,095,226x and must
    still leave `_governance_fields` bounded -- here at the controls' own 0.15, because every
    control carries the same year and both quantiles therefore sit on it."""
    hist = _proxies(bad_comp=(1.0, 28_095_227.0),
                    bad_names=("Charles Liang", "Charles Liang"))
    f = _shipped(hist)["ceo_pay_growth"]
    assert np.isfinite(f.loc[COVERED, "BAD"].to_numpy()).all()
    assert np.allclose(f.loc[COVERED, "BAD"], ORDINARY_GROWTH)
    assert (f.loc[COVERED, "BAD"] <= f.loc[COVERED].quantile(0.99, axis=1) + 1e-9).all()


def test_an_ordinary_pay_year_is_bit_identical():
    """WINSORIZATION MUST BE INERT IN THE BODY. A +15% year is the ordinary case and the trim
    that bounds the GOOGL cell is only defensible if it leaves this one alone."""
    hist = _proxies(bad_comp=(1.0, 28_095_227.0),
                    bad_names=("Charles Liang", "Charles Liang"))
    f = _shipped(hist)["ceo_pay_growth"]
    controls = [c for c in f.columns if c.startswith("C")]
    assert len(controls) == N_CONTROLS
    pd.testing.assert_frame_equal(
        f.loc[COVERED, controls],
        pd.DataFrame(ORDINARY_GROWTH, index=COVERED, columns=controls),
        check_exact=False, check_names=False, atol=1e-12)


def test_the_difference_column_inherits_both_halves():
    """`ceo_pay_vs_revenue_growth` is built from the pay leg, so the guard reaches it by
    construction; the trim reaches it because it is applied PER SHIPPED COLUMN. That second
    half is not cosmetic -- the difference carries its own tail from the REVENUE leg, whose raw
    minimum on the live table is −212.44, which no bound on the pay leg can reach."""
    fields = _shipped(_proxies())
    assert "ceo_pay_vs_revenue_growth" in fields
    diff = fields["ceo_pay_vs_revenue_growth"]
    assert diff.loc[COVERED, "BAD"].isna().all(), "the guard did not reach the difference"
    # flat revenue -> the difference IS the pay leg
    assert np.allclose(diff.loc[COVERED, "C000"], ORDINARY_GROWTH)

    fields = _shipped(_proxies(bad_comp=(1.0, 28_095_227.0),
                               bad_names=("Charles Liang", "Charles Liang")))
    diff = fields["ceo_pay_vs_revenue_growth"]
    assert np.isfinite(diff.loc[COVERED, "BAD"].to_numpy()).all(), "the trim did not reach it"
    assert np.allclose(diff.loc[COVERED, "BAD"], ORDINARY_GROWTH)


@pytest.mark.parametrize("name", sorted(_RAW_ONLY_COMPUTED))
def test_every_declared_member_is_actually_trimmed(name: str):
    """The set DRIVES the trim, so it can no longer drift out of date as documentation. If a
    third unencoded column is added to `_RAW_ONLY_COMPUTED` it is bounded on the same line."""
    hist = _proxies(bad_comp=(1.0, 28_095_227.0),
                    bad_names=("Charles Liang", "Charles Liang"))
    f = _shipped(hist)[name]
    raw = _ceo_pay_growth(hist, IDX)
    assert raw.loc[COVERED, "BAD"].max() > 1e6, "the fixture stopped being extreme"
    assert f.loc[COVERED, "BAD"].max() < 1.0, f"{name} shipped an untrimmed tail"


def test_the_trim_touches_nothing_else_in_the_family():
    """A `_LEVEL_FIELDS` column shares the frame dict with the two trimmed ones and must come
    out untouched: `board_size` 11 against a hundred 9s is exactly what a 1%/99% row trim would
    flatten if the loop were applied to everything."""
    hist = _proxies()
    f = _shipped(hist)["board_size"]
    assert np.allclose(f.loc[COVERED, "BAD"], 11.0), "board_size was winsorized"
    assert np.allclose(f.loc[COVERED, "C000"], 9.0)
    assert not np.allclose(winsorize_xs(f).loc[COVERED, "BAD"], 11.0), (
        "the fixture no longer proves the loop is scoped -- a trim would not move BAD")
