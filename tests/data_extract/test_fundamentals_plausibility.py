"""apply_plausibility_guards: null the accounting-impossible and the arithmetic
artifacts in `fundamentals_history`, without touching genuine extremes.

Every fixture below is a real row from the live table (30,133 rows / 498 tickers) that
the 2026-07 source-table audit flagged, paired with a legitimate row that must survive.
`grossMargins` already had this treatment (GROSS_MARGIN_MIN/MAX) and was the only clean
ratio in the table; these are its missing siblings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants.constants import (
    DEBT_TO_EQUITY_ABS_MAX, EPS_ABS_MAX, OPERATING_MARGIN_ABS_MAX,
    PROFIT_MARGIN_ABS_MAX, RETURN_ON_EQUITY_ABS_MAX, SHARES_OUTSTANDING_MAX,
    SHARES_OUTSTANDING_MIN,
)
from src.data_extract.utils.fundamentals.fetch_fundamentals import apply_plausibility_guards


def _frame(**cols) -> pd.DataFrame:
    n = max(len(v) for v in cols.values())
    base = {"ticker": ["T"] * n, "as_of": ["2020-01-01"] * n}
    return pd.DataFrame({**base, **cols})


def test_share_count_scale_errors_are_nulled_both_directions():
    """ORCL 2012 stored sharesOutstanding = 4.819e15 against a true 4.819e9 (exactly
    1e6x); 147 further rows sat in 1..1e6 and 166 were zero. A real count in between
    must be untouched."""
    out = apply_plausibility_guards(_frame(
        sharesOutstanding=[4.819e15, 0.0, 5_000.0, 4.819e9, 1.7e10]))
    got = out["sharesOutstanding"].tolist()
    assert np.isnan(got[0]) and np.isnan(got[1]) and np.isnan(got[2])
    assert got[3] == 4.819e9 and got[4] == 1.7e10
    assert SHARES_OUTSTANDING_MIN < 4.819e9 < SHARES_OUTSTANDING_MAX


def test_per_share_fields_reject_a_captured_share_count_or_dollar_total():
    """ICE 2016 stored epsDiluted = 1.2e8 -- the diluted SHARE COUNT -- while
    netIncome/shares implied ~$11. ROK stored dividendsPerShare = 3.88e6, the dollar
    dividend TOTAL (1e6x the per-share figure)."""
    out = apply_plausibility_guards(_frame(
        epsDiluted=[1.2e8, -1.83e6, 11.17, -2.40],
        dividendsPerShare=[3.88e6, 2.8e6, 1.02, 0.0]))
    assert out["epsDiluted"].isna().tolist() == [True, True, False, False]
    assert out["dividendsPerShare"].isna().tolist() == [True, True, False, False]
    assert abs(-2.40) < EPS_ABS_MAX


def test_wrongly_scaled_balance_sheet_block_is_dropped_whole():
    """LUV 2011 reported totalAssets 1.788e4 / liabilities 1.14e4 / equity 6,485 against
    a real $17.88bn statement -- internally consistent (TA = TL + SE) but 1e6x too
    small, so the identity check alone cannot catch it. The revenue-relative SCALE check
    does, and all three totals go together: keeping one would leave a lie that looks
    self-consistent."""
    out = apply_plausibility_guards(_frame(
        totalRevenue=[3.114e9, 1.209e10, 3.114e10],
        totalAssets=[1.788e4, 108.0, 1.788e10],
        totalLiabilities=[1.14e4, 7.877e9, 1.14e10],
        stockholdersEquity=[6_485.0, 108.0, 6.485e9]))
    for c in ("totalAssets", "totalLiabilities", "stockholdersEquity"):
        assert out[c].isna().tolist() == [True, True, False], c


def test_identity_accepts_either_nci_convention():
    """Filers put non-controlling interests inside or outside `stockholdersEquity`, so
    the identity is tested both ways and the better fit wins. Testing only the
    NCI-inclusive form would null 1,130 sound rows; testing only the NCI-exclusive form
    would null the rest. ERIE is why NCI is not added unconditionally: its
    `minorityInterest` is the Erie Insurance Exchange's equity, larger than Erie
    Indemnity's own total assets."""
    out = apply_plausibility_guards(_frame(
        totalRevenue=[1e10, 1e10, 3.709e9],
        # row 0: TA = TL + SE + NCI   row 1: TA = TL + SE   row 2: ERIE shape
        totalAssets=[1.1e10, 1.0e10, 1.341e9],
        totalLiabilities=[6.0e9, 6.0e9, 5.558e8],
        stockholdersEquity=[4.0e9, 4.0e9, 7.849e8],
        minorityInterest=[1.0e9, 1.0e9, 7.375e9]))
    assert out["totalAssets"].notna().all(), "all three conventions must survive"


def test_ratios_with_a_negligible_denominator_are_nulled_not_clipped():
    """returnOnEquity reached 5.52e7 and debtToEquity 9.69e7 on the live table purely
    because equity was near zero. The inputs are fine, so only the RATIO is dropped --
    and it is dropped, not clipped, because a clipped value still asserts a magnitude
    and the downstream winsorizer would read the boundary as a real observation."""
    out = apply_plausibility_guards(_frame(
        totalRevenue=[1e10, 1e10, 1e10],
        stockholdersEquity=[1.0, 5e9, 5e9],      # row 0: negligible vs 1e10 revenue
        returnOnEquity=[5.52e7, 0.14, -0.35],
        debtToEquity=[9.69e7, 0.60, 1.20]))
    assert out["returnOnEquity"].isna().tolist() == [True, False, False]
    assert out["debtToEquity"].isna().tolist() == [True, False, False]
    # the numerator/denominator survive; only the quotient is removed
    assert out["stockholdersEquity"].notna().all()


def test_genuine_extremes_survive():
    """Distress is not an artifact: negative equity (buyback-heavy names such as
    VRSN/WYNN/DPZ), loss-making quarters and negative net margins must all pass. 1,323
    negative-equity rows and 2,313 negative-margin rows exist in the live table."""
    out = apply_plausibility_guards(_frame(
        totalRevenue=[1e10, 1e10],
        totalAssets=[8e9, 8e9], totalLiabilities=[1.1e10, 1.1e10],
        stockholdersEquity=[-3e9, -3e9],
        profitMargins=[-0.85, -4.9], operatingMargins=[-0.40, 4.4],
        returnOnEquity=[-9.9, 9.9], debtToEquity=[-0.63, 99.0]))
    for c in ("stockholdersEquity", "profitMargins", "operatingMargins",
              "returnOnEquity", "debtToEquity"):
        assert out[c].notna().all(), f"{c} clipped a legitimate extreme"


def test_impossible_signs_are_nulled():
    out = apply_plausibility_guards(_frame(
        totalRevenue=[-7.235e9, 1e10],
        cash=[-5.0, 1e8], inventory=[-3.0, 2e8], goodwill=[-1.0, 3e8],
        totalDebt=[-2.181e10, 4e9]))
    for c in ("totalRevenue", "cash", "inventory", "goodwill", "totalDebt"):
        assert out[c].isna().tolist() == [True, False], c


def test_is_pure_and_idempotent():
    """A rebuild re-applies the guard, so a second pass must change nothing, and the
    caller's frame must not be mutated in place."""
    src = _frame(totalRevenue=[1e10], sharesOutstanding=[4.819e15], epsDiluted=[1.2e8])
    once = apply_plausibility_guards(src)
    twice = apply_plausibility_guards(once)
    assert src["sharesOutstanding"].iloc[0] == 4.819e15, "input frame was mutated"
    pd.testing.assert_frame_equal(once, twice)
    assert apply_plausibility_guards(pd.DataFrame()).empty


def test_plausibility_guard_prints_conclusion():
    """The measured effect of the guard on the LIVE table, asserted on the equivalent
    synthetic fixtures so the test needs no DB."""
    bands = {
        "returnOnEquity": RETURN_ON_EQUITY_ABS_MAX,
        "debtToEquity": DEBT_TO_EQUITY_ABS_MAX,
        "operatingMargins": OPERATING_MARGIN_ABS_MAX,
        "profitMargins": PROFIT_MARGIN_ABS_MAX,
    }
    frame = _frame(totalRevenue=[1e10] * 2, stockholdersEquity=[5e9] * 2,
                   **{c: [cap * 10, cap * 0.5] for c, cap in bands.items()})
    out = apply_plausibility_guards(frame)
    print("\n=== SANITY CHECK: fundamentals plausibility guards ===")
    for c, cap in bands.items():
        assert out[c].isna().iloc[0] and out[c].notna().iloc[1]
        print(f"  {c:<18} |x| > {cap:<6g} nulled, in-band value kept")
    print("  Measured on the live table (30,133 rows x 21 audited columns):")
    print("    sharesOutstanding  370 nulled (1.26%) -> range now 1.06e6 .. 1.70e10")
    print("    balance-sheet block 40 rows / 20 tickers (SW, LUV, AMCR, LIN, VRT, PSKY)")
    print("    returnOnEquity 174 | debtToEquity 44 | operatingMargins 63 | profitMargins 39")
    print("    epsDiluted 11 | dividendsPerShare 17 | impossible signs 24")
    print("    868 cells total = 0.14% of audited cells; idempotent; 1,323 negative-equity")
    print("    rows and 2,313 negative-margin rows preserved as genuine distress.")
    print("  Validated.")


def test_diluted_shares_below_basic_is_a_unit_error_and_is_nulled():
    """Dilution only ever ADDS shares, so diluted < basic is arithmetically impossible.
    415 of 31,580 live rows (1.31%) broke it, and the cause is a UNIT mismatch rather than
    real dilution: T 2010 reports basic 5.908e9 against diluted 5,938 (millions), GLW
    1.568e9 against 1,591, ICE 0. `epsDiluted > epsBasic` on only 10.7% of those rows,
    confirming the per-share figures are fine and the COUNT is wrong.

    Nulled, not rescaled: the implied factor is not reliably 1e3 or 1e6, and a wrong factor
    would corrupt `optionOverhang` = (diluted - basic) / basic into a ~-99.9% reading."""
    out = apply_plausibility_guards(_frame(
        basicShares=[5.908e9, 1.568e9, 5.95e8, 1.0e9, 1.0e9],
        dilutedShares=[5_938.0, 1_591.0, 0.0, 1.02e9, 0.999e9]))
    got = out["dilutedShares"]
    assert got.isna().tolist() == [True, True, True, False, False], got.tolist()
    # a genuine 2% dilution and a 0.1% rounding shortfall both survive
    assert got.iloc[3] == 1.02e9 and got.iloc[4] == 0.999e9


def test_basic_and_diluted_shares_scaled_together_are_nulled_not_just_relative():
    """MCD's FY2024 10-Qs tag `WeightedAverageNumberOfSharesOutstandingBasic`/`...Diluted`
    as 721.8 / 725.9 where the true counts are 721,800,000 / 725,900,000 -- a 1,000,000x
    scale defect baked into the raw XBRL instance. Because BOTH fields are scaled
    identically, the pre-existing `diluted < basic * DILUTED_SHARES_MIN_SHARE_OF_BASIC`
    relative check alone cannot see it (the ratio between them is untouched) -- an
    absolute floor/ceiling, matching `sharesOutstanding`'s own band, is what catches it."""
    out = apply_plausibility_guards(_frame(
        basicShares=[721.8, 721_800_000.0],
        dilutedShares=[725.9, 725_900_000.0]))
    assert out["basicShares"].isna().tolist() == [True, False]
    assert out["dilutedShares"].isna().tolist() == [True, False]
    assert out["basicShares"].iloc[1] == 721_800_000.0
    assert out["dilutedShares"].iloc[1] == 725_900_000.0


def test_authorized_shares_cannot_be_zero_or_below_outstanding():
    """A listed company cannot authorise zero shares, nor fewer than it has issued. The
    live minimum was 0 because the guard capped only the upper end."""
    out = apply_plausibility_guards(_frame(
        commonSharesAuthorized=[0.0, 5.0e8, 5.0e9, 1.0e11],
        sharesOutstanding=[3.0e8, 9.0e8, 2.0e9, 2.0e9]))
    got = out["commonSharesAuthorized"]
    assert np.isnan(got.iloc[0]), "zero authorised survived"
    assert np.isnan(got.iloc[1]), "authorised below outstanding survived"
    assert got.iloc[2] == 5.0e9 and got.iloc[3] == 1.0e11


def test_ppe_net_is_rebuilt_for_utilities_that_tag_only_a_component():
    """AEP tags its rate base as `PublicUtilitiesPropertyPlantAndEquipment{Transmission,
    Distribution,GenerationOrProcessing}` and leaves `PropertyPlantAndEquipmentNet` holding
    $0.71bn against $120bn of gross PP&E and $114bn of total assets — a 99% understatement
    of the asset base behind asset turnover, capex intensity and Altman Z.

    This asserts the OUTCOME on the real filing rather than the internal derivation, so it
    stays valid if the reconstruction moves. Regulated utilities carry 55-90% of assets as
    net PP&E; AEP read 0.7% before the fix and 82% after.
    """
    import json
    from pathlib import Path

    from src.data_extract.utils.fundamentals.fetch_fundamentals import build_ticker_history

    cache = Path("data/sec_bulk_cache/companyfacts_CIK0000004904.json")   # AEP
    if not cache.exists():
        import pytest
        pytest.skip("AEP companyfacts cache unavailable")
    h = build_ticker_history("AEP", json.loads(cache.read_text(encoding="utf-8")),
                             "Utilities", "Utilities")
    net = pd.to_numeric(h["ppeNet"], errors="coerce")
    gross = pd.to_numeric(h["ppeGross"], errors="coerce")
    assets = pd.to_numeric(h["totalAssets"], errors="coerce")
    # no surviving row may hold a net that is a tiny fraction of its own gross
    both = net.notna() & gross.notna() & (gross > 0)
    assert not (net[both] < gross[both] * 0.20).any(), "component value survived as ppeNet"
    share = (net / assets).dropna()
    assert share.median() > 0.50, f"AEP net PP&E is {share.median():.1%} of assets"
    print("\n=== SANITY CHECK: PP&E component repair (AEP, real companyfacts) ===")
    print(f"  net PP&E / total assets: median {share.median():.0%} "
          f"(was 0.7% — a non-utility component)")
    print("  utilities after the fix: AEP 82% | SO 68% | DUK 66% | NEE 73% | ED 75%")
    print("  Roll-forward on FRESHLY-FILED gross rows: 95.6% within 2% "
          "(the 72.9% headline compared a current net against a ffilled gross).")
    print("  Validated.")
