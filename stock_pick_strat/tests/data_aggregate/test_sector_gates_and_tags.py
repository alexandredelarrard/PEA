"""
Steps 1-3 of the XBRL extraction / aggregation review, on REAL cached companyfacts
for five names chosen so every fixed regime is represented:

    JPM   Financials / Banks                      -> bank KPIs, modern CECL tags
    PGR   Financials / Insurance                  -> underwriting KPIs
    O     Real Estate / Equity REITs              -> FFO incl. impairment add-back
    XOM   Energy                                  -> EBITDAX (tags no OilAndGasProperty*,
                                                     so the OLD gate excluded it)
    CSCO  Information Technology                  -> control. Chosen because the LIVE
                                                     table (old code) gives it a bank
                                                     credit-loss provision (100% of
                                                     filings) and REIT rental income
                                                     (90%): it must now get neither.

What is asserted:
  1. dead tags   -- the 16 mapped-but-never-reported us-gaap names are gone, and the
                    modern replacements populate (JPM net charge-offs / HTM loss).
  2. provisions  -- trade bad debt (XOM) no longer lands in the bank credit-loss pool.
  3. FFO         -- NAREIT's real-estate impairment add-back raises O's FFO.
  4. GICS gates  -- each sector KPI exists ONLY inside its GICS scope. JPM tags
                    `OperatingLeaseLeaseIncome`, so under the old tag-presence gate a
                    BANK was handed REIT FFO/AFFO; XOM was denied EBITDAX.
  5. collisions  -- the fundamental and sector panels no longer share a feature name,
                    and `_merge_panel` now raises instead of emitting `_x`/`_y`.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data_aggregate.step_build_cube import FeatureCollisionError, StepBuildCube
from src.data_aggregate.utils.fundamentals.fundamental_features import build_fundamental_feature_panel
from src.data_aggregate.utils.fundamentals.sector_features import (
    SECTOR_KPI_COLS, build_sector_feature_panel, compute_sector_kpis,
)
from src.data_extract.utils.fundamentals.fetch_fundamentals import (
    DILUTED_SHARES_TAGS, EXTRA_FLOW_TAGS, EXTRA_STOCK_TAGS, FLOW_TAGS, STOCK_TAGS,
    build_ticker_history,
)

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "sec_bulk_cache"

# ticker -> (cik, GICS sector, GICS industry group)
TEST_NAMES: dict[str, tuple[str, str, str]] = {
    "JPM": ("0000019617", "Financials", "Banks"),
    "PGR": ("0000080661", "Financials", "Insurance"),
    "O": ("0000726728", "Real Estate", "Equity Real Estate Investment Trusts (REITs)"),
    "XOM": ("0000034088", "Energy", "Energy"),
    "CSCO": ("0000858877", "Information Technology", "Technology Hardware & Equipment"),
}

# us-gaap names the extractor mapped although NO filer of the 498 cached companyfacts
# reports them (measured, not guessed) -- each one silently killed its feature.
DEAD_TAGS = (
    "CommonEquityTierOneCapitalToRiskWeightedAssets",
    "DeferredPolicyAcquisitionCost",
    "DepreciationAndAmortizationRealEstate",
    "ExplorationAbandonmentAndDryHoleCosts",
    "FinancingReceivableAllowanceForCreditLossesWriteoff",
    "FinancingReceivableAllowanceForCreditLossesWriteoffAfterRecovery",
    "FinancingReceivableExcludingAccruedInterestNonaccrualStatus",
    "GainLossOnDispositionOfRealEstate",
    "HeldToMaturitySecuritiesAmortizedCostAfterAllowanceForCreditLoss",
    "OperatingAndNonoperatingRevenues",
    "OperatingIncome",
    "OperatingLoss",
    "PaymentsForRepurchaseOfCommonStockAndEmployeeShareRepurchases",
    "PolicyholderBenefitsAndClaimsIncurredHomeAndAutoAndOther",
    "ProvisionForCreditLossExpenseReversal",
    "WeightedAverageNumberOfDilutedSharesOutstandingAdjustment",
)

# feature names BOTH panels used to emit (different formulas) -> 20 `_x`/`_y` cube columns
COLLIDED = ("interest_coverage", "net_debt_to_ebitda", "gross_profitability",
            "cash_conversion_cycle", "sbc_intensity")


def _mapped_tags() -> set[str]:
    tags: set[str] = set(DILUTED_SHARES_TAGS)
    for mapping in (FLOW_TAGS, STOCK_TAGS, EXTRA_FLOW_TAGS, EXTRA_STOCK_TAGS):
        for candidates in mapping.values():
            tags.update(candidates)
    return tags


@pytest.fixture(scope="module")
def histories() -> dict[str, pd.DataFrame]:
    """Per-ticker point-in-time history rebuilt from the CACHED companyfacts (offline,
    real data -- real NaNs, real era-switching tags)."""
    out: dict[str, pd.DataFrame] = {}
    for ticker, (cik, sector, group) in TEST_NAMES.items():
        path = CACHE / f"companyfacts_CIK{cik}.json"
        if not path.exists():
            pytest.skip(f"companyfacts cache missing for {ticker} ({path.name})")
        facts = json.loads(path.read_text(encoding="utf-8"))
        hist = build_ticker_history(ticker, facts, sector, group)
        assert not hist.empty, f"{ticker}: empty history"
        out[ticker] = hist
    return out


@pytest.fixture(scope="module")
def fundamentals(histories) -> pd.DataFrame:
    return pd.concat(histories.values(), ignore_index=True)


@pytest.fixture(scope="module")
def kpis(fundamentals) -> pd.DataFrame:
    return compute_sector_kpis(fundamentals)


def _share(df: pd.DataFrame, ticker: str, col: str) -> float:
    """Non-null share of `col` for `ticker` over the last 5 fiscal years of filings."""
    d = df[df["ticker"] == ticker]
    if col not in d.columns or d.empty:
        return 0.0
    d = d[pd.to_datetime(d["as_of"]) >= pd.Timestamp("2020-01-01")]
    return 0.0 if d.empty else float(d[col].notna().mean())


# --------------------------------------------------------------------------- #
# 1. dead tags removed, modern replacements populate                           #
# --------------------------------------------------------------------------- #
def test_dead_tags_removed_and_modern_bank_tags_populate(histories):
    mapped = _mapped_tags()
    still_dead = sorted(t for t in DEAD_TAGS if t in mapped)
    assert not still_dead, f"still mapping tags no filer reports: {still_dead}"

    jpm = histories["JPM"]
    nco = _share(jpm, "JPM", "netChargeOffs")
    htm = _share(jpm, "JPM", "htmUnrealizedLoss")
    prov = _share(jpm, "JPM", "provisionForCreditLosses")
    assert nco > 0.5, f"JPM netChargeOffs still sparse ({nco:.0%})"
    assert htm > 0.5, f"JPM htmUnrealizedLoss sparse ({htm:.0%})"
    assert prov > 0.5, f"JPM provisionForCreditLosses sparse ({prov:.0%})"

    print(f"\n[1] dead tags dropped: {len(DEAD_TAGS)}/{len(DEAD_TAGS)} gone "
          f"({len(mapped)} candidate tags still mapped)")
    print(f"    JPM netChargeOffs {nco:.0%} non-null (was 0% -- both former "
          f"priority-1/2 names do not exist in us-gaap)")
    print(f"    JPM htmUnrealizedLoss {htm:.0%} non-null (new: the DISCLOSED "
          f"unrecognized HTM holding loss)")
    print("    SANITY CHECK: bank credit-quality tags now resolve -> "
          "net_charge_off_rate / htm_unrealized_loss_ratio are computable.")


# --------------------------------------------------------------------------- #
# 2. trade bad debt split out of the bank credit-loss pool                      #
# --------------------------------------------------------------------------- #
def test_trade_bad_debt_split_from_bank_provision(histories, kpis):
    """Trade bad debt is a DIFFERENT concept from a lending credit-loss provision and
    now has its own field. Note CSCO legitimately reports BOTH: Cisco Capital finances
    customer purchases, so it has real financing receivables. That is exactly why the
    KPI must be scoped by GICS rather than by "is the tag present" -- the raw field is
    genuinely dual-use, the bank RATIO (provision / loans) is not."""
    assert "ProvisionForDoubtfulAccounts" not in EXTRA_FLOW_TAGS["provisionForCreditLosses"], \
        "trade bad debt still coalesced into the lending credit-loss pool"
    assert EXTRA_FLOW_TAGS["provisionDoubtfulAccounts"] == ["ProvisionForDoubtfulAccounts"]

    bad = _share(histories["CSCO"], "CSCO", "provisionDoubtfulAccounts")
    jpm_bank = _share(histories["JPM"], "JPM", "provisionForCreditLosses")
    assert bad > 0.5, f"CSCO trade bad-debt expense not captured ({bad:.0%})"
    assert jpm_bank > 0.5, f"JPM lending provision lost ({jpm_bank:.0%})"

    # the leak that mattered: the bank RATIO built on it
    csco_rate = _share(kpis, "CSCO", "provision_rate")
    jpm_rate = _share(kpis, "JPM", "provision_rate")
    assert csco_rate == 0.0, f"CSCO still gets the bank provision_rate ({csco_rate:.0%})"
    assert jpm_rate > 0.5, f"JPM lost provision_rate ({jpm_rate:.0%})"
    assert _share(kpis, "CSCO", "bad_debt_intensity") > 0.5, \
        "bad_debt_intensity KPI not derived from the split-out field"

    print(f"\n[2] CSCO provisionDoubtfulAccounts {bad:.0%} in its OWN field; "
          f"bad_debt_intensity {_share(kpis, 'CSCO', 'bad_debt_intensity'):.0%}")
    print(f"    bank provision_rate: JPM {jpm_rate:.0%}, CSCO {csco_rate:.0%} "
          f"(live table gave CSCO a bank provision on 100% of filings)")
    print(f"    JPM provisionForCreditLosses {jpm_bank:.0%} -- lending pool intact")
    print("    SANITY CHECK: trade bad debt is kept as its own signal instead of being "
          "read as a lending provision, and the bank ratio built on it is now "
          "Banks-only -- CSCO reports both concepts, so only GICS can separate them.")


# --------------------------------------------------------------------------- #
# 3. NAREIT FFO includes the real-estate impairment add-back                    #
# --------------------------------------------------------------------------- #
def test_reit_ffo_adds_back_real_estate_impairment(histories, kpis):
    o = histories["O"]
    assert "realEstateImpairment" in o.columns, "ImpairmentOfRealEstate not extracted"
    imp = pd.to_numeric(o["realEstateImpairment"], errors="coerce")
    hit = int((imp > 0).sum())
    assert hit > 0, "no quarter with a property write-down for O"

    ko = kpis[(kpis["ticker"] == "O") & (imp.reindex(kpis.index[kpis["ticker"] == "O"]).notna())]
    row = kpis[(kpis["ticker"] == "O")].copy()
    row["imp"] = pd.to_numeric(row["realEstateImpairment"], errors="coerce")
    written_down = row[row["imp"] > 0]
    assert not written_down.empty

    # rebuild FFO without the add-back and confirm the fix strictly raises it
    ni = pd.to_numeric(written_down["netIncome"], errors="coerce")
    da = pd.to_numeric(written_down["depAmort"], errors="coerce").fillna(0.0)
    gains = pd.to_numeric(written_down["gainOnDispositions"], errors="coerce").fillna(0.0)
    rev = pd.to_numeric(written_down["totalRevenue"], errors="coerce")
    old_ffo_margin = ((ni + da - gains) / rev)
    new_ffo_margin = pd.to_numeric(written_down["ffo_margin"], errors="coerce")
    cmp = pd.DataFrame({"old": old_ffo_margin, "new": new_ffo_margin}).dropna()
    assert not cmp.empty and (cmp["new"] > cmp["old"]).all(), \
        "FFO margin not raised by the impairment add-back"

    lift = (cmp["new"] - cmp["old"]).max()
    print(f"\n[3] O: {hit} filings carry a real-estate impairment; on the "
          f"{len(cmp)} filings compared the add-back lifts ffo_margin by up to "
          f"{lift:.3f} ({100 * lift:.1f}pp)")
    print("    SANITY CHECK: FFO now excludes all three NAREIT property items "
          "(real-estate D&A, sale gains, impairment write-downs).")


# --------------------------------------------------------------------------- #
# 4. GICS gates: each KPI only inside its own sector                           #
# --------------------------------------------------------------------------- #
# KPI -> the only ticker of the five it may be non-null for
EXCLUSIVE = {
    "net_interest_margin": "JPM", "efficiency_ratio": "JPM", "loan_to_deposit": "JPM",
    "bank_roa": "JPM", "bank_operating_margin": "JPM", "npl_ratio": "JPM",
    "net_charge_off_rate": "JPM", "provision_rate": "JPM", "nii_growth": "JPM",
    "loss_ratio": "PGR", "expense_ratio": "PGR", "combined_ratio": "PGR",
    "investment_income_ratio": "PGR", "premium_growth": "PGR", "float_growth": "PGR",
    "ffo_margin": "O", "affo_margin": "O", "ffo_payout": "O", "rental_margin": "O",
    "net_debt_to_ebitdare": "O", "affo_dividend_coverage": "O",
    "ebitdax_margin": "XOM", "ddna_intensity": "XOM", "exploration_intensity": "XOM",
}


def test_sector_kpis_are_gics_scoped(kpis):
    leaks: list[str] = []
    for kpi, owner in EXCLUSIVE.items():
        if kpi not in kpis.columns:
            continue
        for ticker in TEST_NAMES:
            if ticker == owner:
                continue
            if _share(kpis, ticker, kpi) > 0.0:
                leaks.append(f"{kpi} leaked to {ticker}")
    assert not leaks, "sector KPIs outside their GICS scope: " + "; ".join(leaks)

    # the two directions the old tag-presence gate got wrong, proven on real filings
    assert _share(kpis, "JPM", "ffo_margin") == 0.0, \
        "a BANK still receives REIT FFO (JPM tags OperatingLeaseLeaseIncome)"
    xom_ebitdax = _share(kpis, "XOM", "ebitdax_margin")
    assert xom_ebitdax > 0.5, (
        f"XOM still has no EBITDAX ({xom_ebitdax:.0%}) -- it tags no "
        "OilAndGasProperty*, which is why the old gate covered 3 of 21 Energy names")
    # CSCO tags rentalIncome on 90% of its live rows -> the OLD gate gave a networking
    # company FFO / AFFO / implied cap rate; accumulatedOCI was ungated entirely.
    for kpi in ("net_interest_margin", "ffo_margin", "affo_margin", "ebitdax_margin",
                "loss_ratio", "aoci_to_equity", "rental_margin"):
        assert _share(kpis, "CSCO", kpi) == 0.0, f"CSCO (control) received {kpi}"

    print("\n[4] GICS scoping, non-null share of the last 5y of filings:")
    print(f"    {'KPI':26}" + "".join(f"{t:>7}" for t in TEST_NAMES))
    for kpi in ("net_interest_margin", "net_charge_off_rate", "combined_ratio",
                "ffo_margin", "affo_margin", "ebitdax_margin", "aoci_to_equity"):
        line = "".join(f"{_share(kpis, t, kpi):>6.0%} " for t in TEST_NAMES)
        print(f"    {kpi:26}{line}")
    print(f"    XOM ebitdax_margin {xom_ebitdax:.0%} (old tag gate: 0%); "
          f"JPM ffo_margin 0% (old tag gate: populated)")
    print("    SANITY CHECK: every sector KPI is confined to its GICS sector, and "
          "the starved Energy cohort is recovered.")


# --------------------------------------------------------------------------- #
# 5. no feature-name collision between the two panels                          #
# --------------------------------------------------------------------------- #
def test_panels_share_no_feature_name(fundamentals):
    assert not [c for c in COLLIDED if c in SECTOR_KPI_COLS], \
        "sector panel still claims a fundamental-panel feature name"

    tickers = list(TEST_NAMES)
    idx = pd.bdate_range("2021-01-04", "2026-06-30")
    close = pd.DataFrame(100.0, index=idx, columns=tickers)
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}

    fund = build_fundamental_feature_panel(fundamentals, peers, idx, stock_close=close)
    sect = build_sector_feature_panel(fundamentals, peers, idx)
    assert not fund.empty and not sect.empty

    overlap = (set(fund.columns) & set(sect.columns)) - {"date", "ticker"}
    assert not overlap, f"panels both emit {sorted(overlap)}"

    # the FUNDAMENTAL panel's own GICS gates (mask_columns): REIT / energy EV multiples
    def panel_share(panel: pd.DataFrame, ticker: str, col: str) -> float:
        if col not in panel.columns:
            return 0.0
        d = panel[panel["ticker"] == ticker]
        return 0.0 if d.empty else float(d[col].notna().mean())

    assert panel_share(fund, "O", "f_ffo_yield_xs") > 0.5, "REIT lost its FFO yield"
    assert panel_share(fund, "CSCO", "f_ffo_yield_xs") == 0.0, \
        "CSCO (tags rentalIncome) still gets a REIT FFO yield"
    assert panel_share(fund, "JPM", "f_ffo_yield_xs") == 0.0, \
        "JPM (tags OperatingLeaseLeaseIncome) still gets a REIT FFO yield"
    assert panel_share(fund, "XOM", "f_ebitdax_to_ev_xs") > 0.5, \
        "XOM still excluded from EV/EBITDAX"
    assert panel_share(fund, "CSCO", "f_ebitdax_to_ev_xs") == 0.0

    print(f"\n[5] fundamental panel {len(fund.columns) - 2} features, sector panel "
          f"{len(sect.columns) - 2} features, shared names: {len(overlap)}")
    for c in COLLIDED:
        owner = "fundamental" if f"f_{c}_vs_peers" in fund.columns else "-"
        print(f"    {c:24} owner={owner:12} in SECTOR_KPI_COLS={c in SECTOR_KPI_COLS}")
    print("    fundamental-panel gates: f_ffo_yield_xs O="
          f"{panel_share(fund, 'O', 'f_ffo_yield_xs'):.0%} JPM="
          f"{panel_share(fund, 'JPM', 'f_ffo_yield_xs'):.0%} CSCO="
          f"{panel_share(fund, 'CSCO', 'f_ffo_yield_xs'):.0%} | f_ebitdax_to_ev_xs XOM="
          f"{panel_share(fund, 'XOM', 'f_ebitdax_to_ev_xs'):.0%} CSCO="
          f"{panel_share(fund, 'CSCO', 'f_ebitdax_to_ev_xs'):.0%}")
    print("    SANITY CHECK: the 20 `_x`/`_y` cube columns can no longer be produced, "
          "and the REIT / energy EV multiples are confined to their GICS sector.")


def test_merge_panel_raises_on_collision():
    keys = ["date", "ticker"]
    existing = pd.DataFrame({"date": [pd.Timestamp("2026-01-02")], "ticker": ["AAPL"],
                             "f_interest_coverage_vs_peers": [1.0]})
    clashing = pd.DataFrame({"date": [pd.Timestamp("2026-01-02")], "ticker": ["AAPL"],
                             "f_interest_coverage_vs_peers": [2.0]})
    clean = pd.DataFrame({"date": [pd.Timestamp("2026-01-02")], "ticker": ["AAPL"],
                          "f_loss_ratio_vs_peers": [3.0]})
    step = SimpleNamespace(feature_panel=existing.copy())

    with pytest.raises(FeatureCollisionError) as err:
        StepBuildCube._merge_panel(step, clashing)
    assert "f_interest_coverage_vs_peers" in str(err.value)

    added = StepBuildCube._merge_panel(step, clean)
    assert added == 1
    assert not [c for c in step.feature_panel.columns if c.endswith(("_x", "_y"))]

    print("\n[5b] _merge_panel: collision -> FeatureCollisionError "
          f"({str(err.value)[:60]}...); clean panel -> +{added} column, no _x/_y")
    print(f"    keys preserved: {keys} -> {list(step.feature_panel.columns)}")
    print("    SANITY CHECK: a future duplicate feature name fails loudly instead "
          "of being silently split in two.")
