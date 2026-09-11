"""
cube_evidence_narrative.py  (scripts/)
--------------------------------------------------------------------------------------------
The hand-written sections of the `cube_part_fundamentals` evidence report, as structured
blocks so the PDF and the markdown render the SAME words.

Blocks are `(kind, ...)` tuples: `("h1", text)`, `("h2", text)`, `("p", text)`,
`("spacer", pts)`, `("pagebreak",)`, and
`("table", header, rows, column_weights, {col_index: alignment})`.

⚠ MEASUREMENTS COME IN THROUGH `meta`, THEY ARE NOT WRITTEN HERE. The one exception is the
`_sec` coverage table in section 3, which describes `fundamentals_history` rather than the
cube and so cannot be read from the cube profile; it is dated and its query is given, so it
can be re-measured. Everything else -- row counts, column counts, null rates, clip
saturation -- is substituted from the live measurement at render time. The entire reason this
report exists is that the codebase had accumulated claims nobody re-checked.
"""
from __future__ import annotations

# Measured 2026-09-05 against `fundamentals_history` with
#   select count("<col>")::float / count(*) from fundamentals_history
SEC_COVERAGE: list[tuple[str, str, str]] = [
    ("employees_sec", "75.7%", "*already used* — the whole workforce family"),
    ("regime_sec", "11.6%", "filer-type conditioning"),
    ("goodwill_sec", "10.2%",
     "a true ex-**goodwill** ROIC (today's deduction is goodwill **and** other "
     "intangibles, combined)"),
    ("intangiblesExGoodwill_sec", "7.1%", "the other half of that split"),
    ("accumulatedDepreciation_sec", "6.4%",
     "`asset_age`, `implied_useful_life` — the D&A-realism family"),
    ("minorityInterest_sec", "5.1%", "a cleaner EV minority leg"),
    ("ppeGross_sec", "4.7%", "the other half of D&A realism"),
    ("operatingLeaseLiability_sec", "3.5%",
     "lease-adjusted leverage without the `capital.py` estimate"),
    ("financeLeaseLiability_sec", "1.2%", "ditto"),
    ("netInterestIncome_sec", "1.1%",
     "**the bank family**: NIM, efficiency ratio, provision rate"),
    ("premiumsEarned_sec", "1.0%",
     "**the insurance family**: combined ratio, loss ratio"),
    ("netInvestmentIncome_sec", "0.8%", "insurance float economics"),
    ("noninterestIncome_sec", "0.6%", "bank revenue mix"),
    ("realizedInvestmentGains_sec", "0.5%", "insurance earnings quality"),
    ("rentalIncome_sec", "0.4%", "REIT revenue decomposition"),
]

MISSING: list[tuple] = [
    ("1", "Bank health — net interest margin, efficiency ratio, provision rate, NPL ratio, "
          "net charge-off rate",
     "Whether a bank earns a spread and prices credit correctly",
     "Financials are ~13% of the S&P 500 and are currently ranked on `bank_roa` alone. The "
     "2023 regional-bank dislocation was a spread-and-deposit story that ROA does not see.",
     "SEC XBRL: `InterestAndDividendIncomeOperating`, `InterestExpense`, "
     "`NoninterestExpense`, `ProvisionForLoanAndLeaseLosses`, "
     "`FinancingReceivableRecordedInvestmentNonaccrualStatus`",
     "High — bank tags are dimensioned by portfolio segment, so `companyfacts` drops many "
     "of them and the Financial Statement Data Sets are the real source"),
    ("2", "HTM unrealised loss ratio",
     "Held-to-maturity securities at fair value vs amortised cost",
     "The other half of the SVB signal. `aoci_to_equity` catches only the AFS mark; the HTM "
     "hole is where the capital actually went.",
     "SEC **NOTES** sets (`notes_num`) — footnote fair-value tags, the same table the "
     "pension features already read",
     "Medium — the plumbing exists (`load_tagged_facts`), only the tag list is new"),
    ("3", "Insurance health — combined ratio, loss ratio, float growth, reserve development",
     "Whether an insurer underwrites at a profit or buys revenue",
     "An insurer with a combined ratio above 100 is losing money on underwriting no matter "
     "how good the reported EPS looks. Currently invisible.",
     "SEC XBRL: `PremiumsEarnedNet`, `PolicyholderBenefitsAndClaimsIncurredNet`, "
     "`LiabilityForClaimsAndClaimsAdjustmentExpense`",
     "High — same dimensioned-tag problem as banks"),
    ("4", "Core / adjusted earnings",
     "Profit normalised for impairments, restructuring, disposal gains, litigation, "
     "unusual items",
     "The only non-recurring item SF1 isolates is `netIncomeDiscontinued`. The 72 rows where "
     "`netIncome > totalRevenue` (KKR to 14.1×, APO, ARES, BX — all real) are exactly what a "
     "core-earnings series exists to neutralise, and nothing does.",
     "SEC XBRL: `AssetImpairmentCharges`, `RestructuringCharges`, "
     "`GainLossOnDispositionOfAssets`, `LitigationSettlementExpense`, "
     "`UnusualOrInfrequentItem*`",
     "Medium — undimensioned income-statement tags, good `companyfacts` coverage"),
    ("5", "Debt-maturity ladder — 1y / 2-5y maturities, weighted average maturity",
     "How much debt must be rolled, and when",
     "`refinancing_risk` uses short-term debt as a proxy for the wall. The real ladder is a "
     "strictly better signal in a rate-shock regime, and it is disclosed.",
     "SEC XBRL: `LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths` and the four "
     "sibling tags",
     "Low — five undimensioned tags, well covered"),
    ("6", "Analyst estimate revisions (breadth + magnitude, 1m / 3m)",
     "The direction consensus is moving, not its level",
     "The cube has forward *level* (`fwd_eps_yield`, `forward_earnings_yield`) and realised "
     "*surprise*, but no revision momentum — historically one of the strongest medium-horizon "
     "anomalies and the obvious hole in the expectations theme.",
     "Third party: I/B/E/S, Refinitiv, or an FMP/Tiingo estimate history with vintages",
     "Medium — needs true point-in-time vintages; a current-snapshot feed introduces "
     "look-ahead and is worse than nothing"),
    ("7", "Remaining performance obligation (RPO) and its growth",
     "Contracted-but-unrecognised revenue under ASC 606",
     "The cleanest forward-revenue signal for subscription businesses. "
     "`deferred_rev_intensity` is only the *billed* portion; RPO is the whole book.",
     "SEC XBRL: `RevenueRemainingPerformanceObligation`",
     "Low — one undimensioned tag, mandatory since 2018"),
    ("8", "Capitalised software / internal-use software additions",
     "R&D that lands on the balance sheet instead of the income statement",
     "`rd_capitalized_roic` capitalises *expensed* R&D uniformly. Firms that already "
     "capitalise are double-counted; this tag is what separates them.",
     "SEC XBRL: `CapitalizedComputerSoftwareAdditions`", "Low"),
    ("9", "Segment-level revenue and margin dispersion",
     "How concentrated and how divergent the business lines are",
     "A conglomerate discount and a hidden-gem segment are both invisible in consolidated "
     "numbers.",
     "SEC XBRL segment dimensions (`us-gaap:StatementBusinessSegmentsAxis`)",
     "High — inherently dimensioned; `companyfacts` drops dimensioned facts entirely, so "
     "this needs the Frames/FSDS path"),
    ("10", "Cash taxes paid vs book tax expense",
     "The gap between GAAP tax and cash tax",
     "Persistent divergence is the single best tell for aggressive tax structuring, and it "
     "feeds directly into a cash-based ROIC.",
     "SEC XBRL: `IncomeTaxesPaidNet`", "Low"),
    ("11", "Supply-chain / customer concentration",
     "Revenue dependence on a small number of counterparties",
     "A concentration disclosure is a tail-risk feature no ratio captures.",
     "SEC 10-K Item 1A text extraction (LLM), or a vendor graph",
     "High — text extraction, and the DEF 14A run shows the per-filing cost model"),
    ("12", "Patent grants and citations",
     "Innovation output, not innovation spend",
     "`rd_intensity` measures the input. Output is the part that separates productive R&D "
     "from expensive R&D.",
     "Third party: USPTO PatentsView (free bulk)",
     "Medium — entity resolution from assignee names to tickers is the real work"),
]

# --------------------------------------------------------------------------------------- #
# NULL-RATE FORENSICS (section 4). Same exception as SEC_COVERAGE above: these describe
# `fundamentals_history`, `pension_facts`, `notes_num` and per-SECTOR cuts of the cube, none of
# which the cube profile can answer, so they cannot come through `meta`. Measured 2026-09-05;
# each query is named beside its table so every figure is re-derivable.
#
#   gate arithmetic   fill within scope = count(f_<kpi>_xs) / count(*) over the rows whose
#                     ticker is in SECTOR_KPI_SCOPE[family], joined cube -> sp500_tickers
#   floor             count("f_debtToEquity_xs") over cube_part_fundamentals (the best-covered
#                     FUNDAMENTALS column; the dividend columns are a PRICE-path anchor and
#                     give a misleading 0.61%)
# --------------------------------------------------------------------------------------- #

# family, null(_xs), GICS scope, share of rows, fill within scope, zero-coverage tickers
GATE_ARITHMETIC: list[tuple[str, str, str, str, str, str]] = [
    ("`energy_health`", "96.75%", "Energy", "4.25%", "76.5% / 94.9% of live span", "0 of 21"),
    ("`reit_health`", "95.07%", "Equity REITs", "5.67%", "86.9%", "**2 of 30** (AVB, EQR)"),
    ("`financials`", "87.32%", "Financials", "15.38%", "82.4%", "0 of 75"),
    ("`bank_roa`", "97.59%", "Banks", "2.63%", "91.8%", "0 of 13"),
    ("`capex_to_rate_base`", "94.61%", "Utilities", "6.27%", "85.8%", "—"),
]

# the D1/D6 before/after table, MEASURED after the rebuild (not projected):
# metric, Financials before, Financials after, RE before, RE after
D1_DELTA: list[tuple[str, str, str, str, str]] = [
    ("`cash` present (filing grain)", "0.281", "**1.000**", "0.274", "**1.000**"),
    ("`cash_to_debt` fill (listed rows)", "0.256", "**0.927**", "0.293", "**0.973**"),
    ("`net_debt_to_ebitda` fill", "0.276", "**0.923**", "0.288", "**0.944**"),
    ("`refinancing_risk` fill (**needs the SHORT leg**)", "0.287", "0.287", "0.310", "0.310"),
    ("`current_ratio` fill (**control** — must not move)", "0.287", "0.287", "0.310", "0.310"),
]

# D6: the classified column the repo read vs the unclassified one that was always there
CLASSIFIED_TRAP: list[tuple[str, str, str, str]] = [
    ("`currentAssets` / `currentLiabilities`", "0.281", "0.274", "`totalAssets` / `totalLiabilities` — **1.000**"),
    ("`shortTermInvestments` (`investmentsc`)", "0.277", "0.274", "`cashAndEquivalents` — **1.000**"),
    ("`longTermDebt` and `shortTermDebt`", "0.277", "0.274", "`totalDebt` — **1.000**"),
]

# the earnings_yield null decomposition: component, rows, share
EY_DECOMPOSITION: list[tuple[str, str, str]] = [
    ("no balance sheet (the floor)", "646,670", "16.79%"),
    ("**loss-making → masked to NaN by design**", "**263,652**", "**6.84%**"),
    ("`netIncome` missing on the row (`f_profitable` itself null)", "117,638", "3.05%"),
    ("profitable but no market cap", "10,511", "0.27%"),
    ("**total**", "", "**26.95%** vs 26.96% observed"),
]

# pension source depth: source, rows, reach, note
# why an empty row is empty: reason, rows, share of empties, share of panel
EMPTY_SPLIT: list[tuple[str, str, str, str]] = [
    ("**pre-listing padding** — the date is before the ticker's first price",
     "544,855", "84.3%", "14.14%"),
    ("**genuine warm-up** — listed, first filing not yet landed (~1 quarter)",
     "62,740", "9.7%", "1.63%"),
    ("no price row at all (AVB, EA, EQR — D3)", "23,652", "3.7%", "0.61%"),
    ("listed but never any fundamentals (BRK-B, BF-B — D3)", "15,423", "2.4%", "0.40%"),
    ("**after the first filing**", "**0**", "**0%**", "**0%**"),
]

# null rate on the full panel vs on rows where the ticker was actually listed
HONEST_DENOMINATOR: list[tuple[str, str, str]] = [
    ("rows", "3,852,470", "3,262,925"),
    ("`earnings_yield`", "26.96%", "**13.76%**"),
    ("`current_ratio`", "31.07%", "**19.24%**"),
    ("`interest_coverage`", "30.43%", "**18.25%**"),
    ("`altman_z`", "34.19%", "**22.30%**"),
    ("`pegy`", "53.51%", "**45.11%**"),
    ("`pension_retirement_liability`", "83.29%", "**80.32%**"),
]

# D5: what the four fabricated features carry on pre-listing rows
FABRICATED: list[tuple[str, str, str]] = [
    ("`dividend_yield`", "565,893", "`0.0`"),
    ("`dividend_payer`", "565,893", "`0.0`"),
    ("`shareholder_yield`", "565,893", "`0.0`"),
    ("`beneish_m_score`", "565,720", "`-2.48` (constant)"),
]

# D5: phantom contamination of the daily cross-section, by date, BEFORE and AFTER the fix
PHANTOM_DECAY: list[tuple[str, str, str, str, str]] = [
    ("1996-01-02", "300", "491", "**38.9%**", "**300** ✓"),
    ("2000-01-03", "344", "491", "**29.9%**", "**344** ✓"),
    ("2010-01-04", "417", "491", "**15.1%**", "**417** ✓"),
    ("2020-01-02", "478", "491", "2.6%", "**478** ✓"),
    ("2026-01-02", "491", "491", "0%", "**491** ✓"),
]

# D4: the measurement that overturned the un-masking. numerator, negative subset, positive
LOSS_SIGN_FLIP: list[tuple[str, str, str, str]] = [
    ("`netIncome`", "**-0.0297** (t = **-3.17**)", "+0.0116 (t = +3.31)", "37,116 / 8.5%"),
    ("`freeCashflow`", "**-0.0159** (t = **-2.44**)", "+0.0162 (t = +4.98)", "65,247 / 14.9%"),
    ("`stockholdersEquity`", "-0.0414 (t = -2.56)", "-0.0062 (t = -2.18)", "15,183 / 3.3%"),
]

# D4: what un-masking would actually hand the model. feature, merged spread, masked spread
MERGED_VS_MASKED: list[tuple[str, str, str]] = [
    ("`earnings_yield`", "0.0040", "**0.0053**"),
    ("`fcf_yield`", "0.0074", "**0.0079**"),
    ("`book_yield`", "-0.0092", "-0.0054"),
]

PENSION_SOURCES: list[tuple[str, str, str, str]] = [
    ("`pension_facts` (bulk FSDS)", "6,244", "125 tickers",
     "4 tags, 94 on the recognised noncurrent liability; median 17 years each"),
    ("`notes_num` `…FairValueOfPlanAssets`", "4,209", "214 CIKs", "widest single source"),
    ("`notes_num` `…BenefitObligation`", "2,428", "122 CIKs",
     "the PBO leg; caps `funded_ratio` at 98 tickers"),
    ("companyfacts `pensionDeficit`", "—", "**0**", "**the dead leg — DELETED, see D2**"),
]

# the Phase-6 resolution summary: defect, what it was, what was done, measured outcome
RESOLUTIONS: list[tuple[str, str, str]] = [
    ("**D1a** `cash` destroyed wherever `investmentsc` is absent",
     "new `sum_optional` field-map op: `cashneq + coalesce(investmentsc, 0)`",
     "`cash` **0.281 → 1.000** in Financials, 9,130 filings recovered (17.8%)"),
    ("**D1b** `liquid_assets` subtracted short-term investments **twice**",
     "dropped the duplicate leg; `cash` already contains them",
     "EV corrected; worst in IT, where median `investmentsc`/mcap is 1.30%"),
    ("**D1c** should bank/insurer cash be netted from EV?",
     "**decided: no.** Banks + Insurance (36 tickers) gated out of all three netting sites",
     "bank EV unchanged; `cash_to_debt` still gained — only netting was wrong"),
    ("**D2** the pension pool's second source was dead code",
     "deleted the `pensionDeficit` leg and the docstring's third source",
     "provably a no-op: **zero** pension columns exist on either fundamentals table"),
    ("**D3** BRK-B / BF-B invisible to every fundamental feature",
     "share-class symbol mapping in and out, plus a migration of 3,332 vendor rows",
     "489 → **491** tickers; **6** splits applied incl. `BRK-B 2010 ×50`"),
    ("**D4** the yields discard the magnitude of a loss",
     "**measured, and NOT un-masked** — the sign flips at zero; added `loss_intensity`",
     "mask intact on 0 rows either way; new column covers 98.8% of loss-makers"),
    ("**D5** four features fabricated on pre-listing rows",
     "masked to the listed window; Beneish keyed on data support",
     "1996 cross-section **491 → 300**; panel 3.85M → **3.32M** honest rows"),
    ("**D6** the distress DENOMINATOR had D1a's defect one level down",
     "`_combine_debt` now prefers the reconciled `totalDebt` via `capital.borrowings`",
     "`cash_to_debt` **0.256 → 0.927** (Financials), **0.293 → 0.973** (Real Estate)"),
]


def opening(m: dict) -> list[tuple]:
    """Everything before the feature table."""
    return [
        ("p",
         "This report exists because the audit it follows found that **65 of 179 defined "
         "features emitted nothing** and **three pairs sat at Pearson r = 1.0000** for a whole "
         "vintage of the table, with 68 tests passing throughout. A suite that never compares "
         "two cube features to each other cannot see that class of defect. So the standard "
         "here is not that the code looks right: every claim below is a number measured "
         "against the rebuilt table."),

        ("h1", "0. The rebuild, before and after"),
        ("p", f"Full-universe, full-history rebuild (`build-fundamentals --full`), "
              f"{m['rebuild_window']}."),
        ("table", ["", "before", "after"], [
            ["rows", f"{m['rows_before']:,}", f"**{m['rows']:,}**"],
            ["feature columns", str(m["cols_before"]), f"**{m['cols']}**"],
            ["characteristics", str(m["chars_before"]), f"**{m['chars']}**"],
            ["tickers", "502", "**489**"],
            ["dead features (defined, emit nothing)", "65", "**0**"],
            ["pairs at |r| ≥ 0.999", "5", "**0**"],
            ["mean peer-z clip saturation", "3.84%", f"**{m['sat_mean']}**"],
            ["tests/data_aggregate/", "21 failures", "**272 passed, 0 failed**"],
        ], [30, 12, 12], {1: "RIGHT", 2: "RIGHT"}),

        ("h2", "Per-builder output, from the run log"),
        ("table", ["builder", "features", "row coverage"], [
            ["peer-relative fundamental", "168", "100.0%"],
            ["sector KPI", "48", "81.4%"],
            ["earnings expectation", "11", "70.8%"],
            ["workforce", "8", "74.8%"],
            ["dividend", "14", "100.0%"],
        ], [30, 10, 12], {1: "RIGHT", 2: "RIGHT"}),

        ("h2", "The column diff is exactly the intended one"),
        ("p", "16 removed, 56 added, 193 kept."),
        ("p",
         "**Removed (16)** — `f_roic_*`, `f_roic_incl_goodwill_*`, `f_roic_ex_goodwill_*`, "
         "`f_goodwill_roic_drag_*` (renamed onto the `intangibles` basis, and the sector "
         "panel's duplicate `roic` deleted); `f_days_sales_outstanding_*`, "
         "`f_days_payable_outstanding_*`, `f_days_inventory_outstanding_*` (the sector panel's "
         "copies of `dso` / `dpo` / `dio` — one owner now); `f_pension_underfunding_to_mcap_*` "
         "(a strict subset of `pension_overhang_leverage` at r = 1.000000)."),
        ("p",
         "**Added (56)** — the workforce family (8), the revived sector KPIs (24: bank ROA, "
         "AOCI, book-value growth, the four REIT KPIs, the two energy KPIs, "
         "capex-to-rate-base, buyback intensity), `revenueGrowth` / `earningsGrowth` (4), the "
         "`intangibles_*` family (8), `roic_incl/ex_intangibles` (4), `ffo_yield` (3 views), "
         "`nci_income_share`, `sbc_to_buyback`, `acquisition_intensity`, `hyper_growth`."),
        ("p",
         "**Only two surviving columns moved by more than 1pp of null rate** — "
         "`f_pegy_vs_peers` (79.8% → 57.3%) and `f_pegy_xs` (70.2% → 53.5%). That is "
         "attributable, not incidental: PEGY masks its denominator to "
         "`growth% + dividend-yield% > 0`, and while `dividendsPaid` was stored "
         "outflow-negative the dividend leg was *subtracting* from the denominator and "
         "nulling the feature for payers. The other 191 kept columns are unchanged, which is "
         "the evidence that the audit did not disturb anything it was not aiming at."),

        ("h1", "1. How to read the feature table"),
        ("p",
         "Each **characteristic** (`fcf_yield`, `dso`, …) ships as up to three **views**: "
         "**peer-z** (`_vs_peers`, firm minus its embedding-peer basket, standardised, "
         "clipped ±8, then winsorised); **universe %ile** (`_xs`, cross-sectional percentile "
         "rank on the day); **self-history z** (`_vs_hist`, firm vs its own trailing "
         "1260-day mean and σ — the eight valuation yields only); and **raw** for the 0/1 "
         "regime flags, which are deliberately not standardised."),
        ("p",
         "The **null rate** quoted is the best-covered view, because the transforms can only "
         "lose rows, never add them."),
        ("p",
         "A high null rate is not automatically a defect. Three legitimate causes recur, and "
         "each row says which applies. **Masked on purpose**: `earnings_yield` is null for "
         "loss-makers because a negative E/P is not cheap, it is a loss — the `profitable` "
         "flag carries that regime instead. **Sector-gated**: `ffo_yield` is null outside the "
         "GICS equity-REIT group; a gate is a scope, not a gap. **Long look-back**: "
         "`operating_margin_5y_chg` and `dividend_growth_5y` need five years of "
         "point-in-time history before they can emit anything."),

        ("h2", "The @clip column"),
        ("p",
         "`peer_relative` clips the peer-z at ±8. A value sitting *on* the clip is no longer "
         "measuring a firm's distance from its peers — it is reporting that the "
         "standardisation ran out of room. `@clip` is the measured share of each column at "
         "|z| ≥ 8, and it is the most direct answer to “are the extreme values good?” for a "
         "standardised feature."),
        ("p",
         f"Across the {m['n_standardised']} standardised characteristics: **mean "
         f"{m['sat_mean']}, median {m['sat_median']}**, against a pre-audit mean of 3.84%. "
         f"That fall is the dispersion floor plus input winsorisation added in Phase 3 of the "
         f"audit."),
        ("p",
         "**What drives the remaining saturation is skew, not sparsity.** Across those "
         "columns, saturation correlates 0.50 with the peer-z mean/median gap and only 0.09 "
         "with the null rate. The 26 columns above 2% are almost all ratios floored at zero "
         "with a long right tail — `interest_coverage`, `cash_to_debt`, "
         "`revenue_per_employee`, `acquisition_intensity`, `sbc_to_buyback`, "
         "`option_overhang` — where p1 sits near −1 and p99 is pinned at exactly 8.00. "
         "Z-scoring a quantity bounded below at zero *cannot* be symmetric, so the right tail "
         "lands on the clip regardless of how well the peer basket is behaving. The "
         "counter-example is decisive: `pension_funded_ratio` is the sparsest column in the "
         f"table ({m['sparsest_null']} null) and saturates **0.0%**. For those features the "
         "`_xs` percentile view is the better read, and the model has both."),
    ]


def closing(m: dict) -> list[tuple]:
    """Everything after the feature table."""
    sec_rows = [[f"`{c}`", cov, why] for c, cov, why in SEC_COVERAGE]
    miss_rows = [[n, f"**{fam}**", what, why, src, cost] for n, fam, what, why, src, cost
                 in MISSING]
    return [
        ("h1", "3. What is NOT in the cube that should be"),
        ("p",
         "The audit deleted 65 dead features. They were not deleted because they were bad "
         "ideas — they were deleted because **`fundamentals_history` is Sharadar-first and "
         "SF1 does not carry their inputs**, so each measured exactly zero values while "
         "looking fully implemented. That is worse than absent: it is a feature the config "
         "references, the tests exercise on hand-built fixtures, and the model never sees."),

        ("h2", "3.1 The Sharadar side is exhausted"),
        ("p",
         "Measured, not assumed: of the 112 columns in `fundamentals_sharadar`, **every "
         "single data column is either already mapped or deliberately excluded**. Only the "
         "seven key/metadata columns (`ticker`, `date`, `reportperiod`, `calendardate`, "
         "`dimension`, `fiscalperiod`, `lastupdated`) are unreferenced by "
         "`configs/sharadar/sharadar_field_map.json`."),
        ("p",
         "The 48 excluded names are excluded *correctly* — they are ratios Sharadar derives "
         "from primitives we already map (`pe`, `roe`, `evebitda`, `netmargin`, …), "
         "split-adjusted per-share duplicates, or USD-translated twins. Recomputing them "
         "ourselves from the primitives is what keeps one definition of debt, EBITDA and "
         "invested capital across the whole cube. **There is no unharvested Sharadar field** "
         "— every gap below needs the SEC path or a vendor."),

        ("h2", "3.2 The SEC path: measured coverage of what already exists"),
        ("p",
         "The field map already declares 13 fields as `kind: \"sec\"` — it *knows* they "
         "cannot come from Sharadar. All 13 exist in `fundamentals_history` only as "
         "`_sec`-namespaced twins, and their live coverage is the reason nothing is built on "
         "them (measured 2026-09-05):"),
        ("table", ["`_sec` column", "coverage", "what it would unlock"],
         sec_rows, [16, 7, 55], {1: "RIGHT"}),
        ("p",
         "Ranking a 500-name universe on a column populated for 1% of rows produces a feature "
         "that is a *sector-and-vintage indicator wearing a ratio's name*. That is the "
         "specific trap this audit removed, so the honest statement is: **these return when "
         "the SEC extraction path is rebuilt to production coverage, not before.**"),

        ("h2", "3.3 The table"),
        ("p",
         "Ranked by what I would build first. “Expected edge” is a judgement, flagged as "
         "such; the coverage numbers are measured."),
        ("table", ["#", "Feature family", "What it measures", "Why it should be a strong "
                   "driver", "Source needed", "Cost / risk"],
         miss_rows, [2, 20, 16, 26, 22, 14], {0: "RIGHT"}),

        ("h2", "3.4 Two things worth saying plainly"),
        ("p",
         "**The GICS snapshot is a known look-ahead.** `sp500_tickers` holds today's "
         "classification with no date axis, so a name that changed sector carries its 2026 "
         "label back to 1995. It is accepted rather than unnoticed: it is the same look-ahead "
         "the peer basket already carries (one static 2026-vintage embedding basket over the "
         "whole history), so it adds no new *kind* of bias. A point-in-time GICS history "
         "would fix both at once, and neither alone justifies a vendor feed."),
        ("p",
         "**Some deletions should not come back.** `ebitdax_to_ev`, `implied_cap_rate` and "
         "`net_debt_to_ebitdare` are not on the list above. On any substrate lacking "
         "`explorationExpense` and `realEstateImpairment`, all three reduce to a sector-masked "
         "copy of a general-purpose ratio (measured r = 1.0000, 1.0000 and 0.9998). A gate is "
         "not a second feature — the model already gets sector membership from the GICS "
         "categorical. They should return only if the missing add-back legs return with them."),

        ("pagebreak",),
        ("h1", "4. What the null rates mean — and four defects to fix"),
        ("p",
         "Section 2 reports every feature's null rate. It does not say what a null rate *means*, "
         "and the six families with the highest ones turn out to have four different causes. "
         "This section separates them. It was written after the report was first delivered, "
         "in response to exactly the right question: *is this missing information, "
         "un-retrieved information, or an error?*"),
        ("p",
         "**All six defects it found have since been fixed and re-measured against a full "
         "rebuild.** Each subsection below states the symptom, the root cause, and the measured "
         "outcome. The evidence tables are kept in their *before* form because they are what "
         "diagnosed the defect; every “after” number is measured, not projected."),
        ("table", ["defect", "what was done", "measured outcome"],
         [list(r) for r in RESOLUTIONS], [30, 34, 34], {}),
        ("p",
         "**⚠ Null rates in this report are not comparable to the previous edition.** D5 removed "
         "537,435 rows from the panel — pre-listing padding that carried nothing but fabricated "
         "values — so the denominator shrank 14% and every whole-panel null rate fell by roughly "
         "12pp for that reason alone. Compare *fill within listed rows*, per sector."),

        ("h2", "4.1 Two constants govern every null rate"),
        ("p",
         "**16.79% of rows have no balance sheet — but only 1.6% of the panel is genuine "
         "warm-up.** `f_debtToEquity_xs` is the best-covered *fundamentals* column, so a row "
         "lacking it has no balance sheet. Those 646,670 rows split five ways, and the split "
         "matters because **84% of them are a ticker that was not yet public**:"),
        ("table", ["why the row is empty", "rows", "of empties", "of panel"],
         [list(r) for r in EMPTY_SPLIT], [52, 12, 12, 12],
         {1: "RIGHT", 2: "RIGHT", 3: "RIGHT"}),
        ("p",
         "The zero on the last row is the important one: **no mid-life gaps and no "
         "post-delisting tails** — the forward-fill is airtight. The panel *was* a near-dense "
         "grid (3,852,470 rows against 3,914,596 for a full 7,798 dates × 502 tickers = 98.4% "
         "dense), so every ticker carried rows back to 1995-09-01 whether or not it existed. "
         "Reddit had 7,182 rows before its IPO."),
        ("p",
         "**This is now fixed at the source rather than reported around.** Those padded rows "
         "survived only because four features wrote a value onto them (D5). Once masked, the "
         "rows were wholly NULL and `write_part(drop_empty=True)` removed them: **607,359 rows "
         "dropped, 15.5% of the grid, against 62,126 (1.6%) before.** The panel is now "
         "3,315,035 rows and *is* the honest denominator. The table below is what the old "
         "two-denominator comparison looked like, kept because it is what made the padding "
         "visible:"),
        ("table", ["", "old full panel", "old listed rows only"],
         [list(r) for r in HONEST_DENOMINATOR], [30, 14, 16], {1: "RIGHT", 2: "RIGHT"}),
        ("p",
         "(The 0.61% the old edition got from `f_dividend_yield_xs` was the wrong anchor for a "
         "different reason — see **D5**, which is *why* it was 0.61% at all.)"),
        ("p",
         "**A GICS-gated feature can never fill more than its sector's share of rows.** That is "
         "the design, and it is the entire explanation for five families. Each reconciles to "
         "three decimals against `1 − scope × fill`:"),
        ("table", ["family", "null (`_xs`)", "GICS scope", "share of rows",
                   "fill *within* scope", "zero-coverage tickers"],
         [list(r) for r in GATE_ARITHMETIC], [16, 9, 12, 10, 20, 16],
         {1: "RIGHT", 3: "RIGHT"}),
        ("p",
         "**Consequence for expectations**: a bank-only KPI cannot go below ~97.4% null however "
         "well it is extracted, and a REIT-only KPI cannot go below ~94.3%. Judge these "
         "families by *fill within scope*, never by the headline null rate. Nothing in this "
         "table is a defect except the two missing REITs, which are D3."),

        ("h2", "4.2 The two universe-wide families are where the findings are"),
        ("p",
         "**Distress (30.3%–36.3% null)** is the 16.79% floor plus two sectors that file an "
         "**unclassified balance sheet**. Banks, insurers and REITs are exempt from the "
         "current/non-current split (ASC 210-10-05-4), so Sharadar's `assetsc` / `liabilitiesc` "
         "are populated on only 28.2% / 27.7% of their filings against 100% for IT. "
         "`current_ratio` and `altman_z` genuinely cannot be built there — no SEC extract fixes "
         "it, because the split is not in the filing. But `cash_to_debt` and "
         "`net_debt_to_ebitda` die on the same rows and **they should not**: that is D1."),
        ("p",
         "**Pension (83.3%–93.5% null)** is *not* a gate. Only 199 of 489 tickers ever receive "
         "a value and mean per-ticker fill is 20.9% of the live span. Two causes compound: much "
         "of the index is DC-only and correctly has no projected benefit obligation to "
         "disclose, *and* one of the pool's three documented sources contributes literally "
         "nothing. That is D2."),

        ("h2", "4.3 D1 — `cash` is destroyed wherever `investmentsc` is absent"),
        ("p",
         "**Symptom.** In Financials and Real Estate, `fundamentals_history.cash` is null on "
         "*exactly* the rows where `currentAssets` is null — 5,647 + 2,438 rows, with **zero** "
         "rows where one is null and the other is not. Sharadar's `cashneq` is **100%** "
         "populated for those same filers."),
        ("p",
         "**Root cause.** `configs/sharadar/sharadar_field_map.json:55` defines "
         "`cash = cashneq + investmentsc`. `investmentsc` is a **current** asset, so it is "
         "absent for precisely the filers that do not classify their balance sheet. The sum is "
         "NaN-propagating, so a present `cashneq` is thrown away by an absent `investmentsc`. "
         "The field map calls this widening “best available” without noting that it *narrows* "
         "for two sectors."),
        ("p",
         "**Fixed** with a new field-map op, `sum_optional`: the first input is required and "
         "later ones are coalesced to 0, so a widening can never destroy the narrow line it "
         "widens. `sum` deliberately stays strict — `ebitda` without `depAmort` is EBIT, and "
         "widening *that* would silently mislabel the column. Recovered `cash` on **9,130 "
         "filings = 17.81% of `fundamentals_history` across 91 tickers** (the earlier 8,085 "
         "counted Financials and Real Estate only; `cashneq` is 100% in every sector)."),
        ("p",
         "**But `cash` alone did not fix the features, and finding out why produced D6.** The "
         "table below is the measured before/after over both fixes:"),
        ("table", ["measured (listed rows unless stated)", "Financials before", "after",
                   "Real Estate before", "after"],
         [list(r) for r in D1_DELTA], [34, 13, 9, 14, 9],
         {1: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT"}),
        ("p",
         "The last two rows are the controls and they did **not** move, which is correct: "
         "`current_ratio` needs the current/non-current split and `refinancing_risk` needs the "
         "short-term debt leg specifically. Neither is recoverable from a total, so ~0.29 is a "
         "substrate limit rather than a defect."),
        ("p",
         "**One feature was deleted as a consequence, and the redundancy guard is what caught "
         "it.** `net_debt_incl_offbs_to_ebitda` was the off-balance-sheet-inclusive twin of "
         "`net_debt_to_ebitda`. It survived only because the two computed debt differently — "
         "the twin already used the reconciled `totalDebt`, the distress version did not. Once "
         "both read the same column the last artificial difference went, and three of the four "
         "legs that were supposed to distinguish them (operating leases, finance leases, "
         "asset-retirement obligations) **have no column on this substrate at all**, while the "
         "fourth reaches 199 of 491 tickers. The pair came out at identical fingerprint hashes "
         "and r = 0.9955 across the live cube; the audit had already deleted "
         "`net_debt_to_ebitdare` at r = 0.9998 for exactly this. Nothing is lost — the pension "
         "overhang it folded in is carried undiluted by `pension_overhang_leverage` and "
         "`pension_retirement_liability`, which are separate columns."),
        ("p",
         "**The guard only worked once its fixture was made faithful.** It had never been "
         "passed `pension_facts` / `notes_num`, so every pension feature was silently absent "
         "from its matrix and the pair read as *exactly* 1.000000 — a fixture artefact rather "
         "than a measurement. That is the same unfaithful-fixture defect Phase 4 fixed in "
         "`test_composites_config`; this one was missed. Passing them took the comparable-feature "
         "count from 99 to 103 and turned a wrong number into the right one."),
        ("p",
         "**⚠ Impact analysis — the distress ratios are safe, enterprise value is not.** Adding "
         "`coalesce(x, 0)` cannot change a row where `x` was already present, and `investmentsc` "
         "is 89–100% populated in the eight sectors that classify their balance sheet, so the "
         "change is a **no-op** outside Financials and Real Estate. But `cash` is netted as a "
         "*non-operating liquid asset* by `capital.liquid_assets`, which feeds enterprise value, "
         "`net_debt` and `invested_capital` — so restoring bank cash shrinks EV and lifts every "
         "EV yield. Measured against market cap, `cashneq` is **10.99% at the Financials median "
         "and 101.37% at its p90**: for the top decile the restored cash exceeds the entire "
         "market cap, EV would go negative, and `positive_den=True` would null a row that has a "
         "value today. It also contradicts `liquid_assets`'s own intent, which already excludes "
         "`investmentSecurities` because *for a bank or insurer that is the core operating "
         "asset, not spare cash* — the same is true of “cash and due from banks”."),
        ("p",
         "**A second, pre-existing defect sits in the same expression (D1b).** "
         "`liquid_assets` returns `cash + shortTermInvestments + marketableSecuritiesCurrent`, "
         "but the field map *defines* `cash` as `cashAndEquivalents + shortTermInvestments` "
         "(line 177) and maps `shortTermInvestments` directly from `investmentsc` (line 179). "
         "**Short-term investments are subtracted twice** from EV, net debt and invested "
         "capital. The docstring's claim that `cash` is “investment-free” was true of the "
         "SEC-era extractor and is false on the Sharadar substrate. Median `investmentsc`/mcap "
         "is ~0 in nine sectors but **1.25% in IT**, so it understates EV most for exactly the "
         "cash-rich names where EV yields matter. Both landed in one rebuild, which is what "
         "makes the two effects attributable."),
        ("p",
         "**D1c was decided rather than assumed: bank and insurer cash is NOT netted.** Banks "
         "and Insurance (36 tickers) are gated out of **every** site that nets cash — enterprise "
         "value, net debt and invested capital — so EV cannot claim the cash is not spare while "
         "invested capital claims it is. The gate deliberately does **not** reach any cash "
         "*ratio*: a liquidity cushion is a real fact about a bank, and `cash_to_debt` is "
         "precisely the feature D1a exists to restore. Only netting was ever wrong."),

        ("h2", "4.4 D6 — the same defect one level down, in the DENOMINATOR"),
        ("p",
         "**Found by checking whether D1a had worked. It had not, on its own.** After the fix "
         "`cash` was 100% populated in every sector, and `f_cash_to_debt_xs` still filled only "
         "**0.256** of Financials' listed rows against a balance-sheet anchor of **0.986**. "
         "BRK-B, BX, AXP, BAC, C, BLK, AIG — every bank at a flat **0.0**, each with a complete "
         "balance sheet. The numerator was fixed; the denominator had the identical defect."),
        ("p",
         "`_distress_fields` built its debt as `longTermDebt + shortTermDebt`. Both are "
         "**current/non-current classifications**, so both are absent for exactly the filers "
         "exempt from that split — while the extractor's reconciled `totalDebt`, which "
         "`capital.borrowings()` already prefers, is populated everywhere. That is the "
         "“two definitions of debt that disagree” failure `capital.py`'s own docstring says it "
         "exists to prevent; `_distress_fields` simply never called it."),
        ("p",
         "**The pattern generalises, and it is the most useful thing in this section.** Three "
         "separate defects here are the same trap — a *classified* column read where an "
         "*unclassified* equivalent sits at full coverage:"),
        ("table", ["the classified column the repo read", "Financials", "Real Estate",
                   "the unclassified one that was always there"],
         [list(r) for r in CLASSIFIED_TRAP], [34, 11, 12, 43], {1: "RIGHT", 2: "RIGHT"}),
        ("p",
         "**Substituting `totalDebt` is safe because it is the same quantity, not a wider one.** "
         "Where both legs exist it equals their sum to within 1% on **100.00% of rows in all "
         "eleven sectors** (median relative gap 0.0). And it is genuinely debt rather than "
         "liabilities for the leg-less filers: median `totalDebt`/assets is **0.095** for "
         "Financials against `totalLiabilities`/assets of **0.877** — BAC's latest filing reads "
         "732bn of debt inside 3,198bn of liabilities on 3,499bn of assets, with deposits "
         "correctly excluded. Measured result: `cash_to_debt` **0.256 → 0.927** in Financials "
         "and **0.293 → 0.973** in Real Estate."),

        ("h2", "4.5 D2 — the pension pool's second source is dead code"),
        ("p",
         "The pool documents three gap-filling sources. Source 2 is companyfacts "
         "`pensionDeficit`, read at `capital.py:163` via `get(\"pensionDeficit\")` — but "
         "**`fundamentals_history` has no pension column at all** (an `information_schema` match "
         "on `%pension%` / `%opeb%` / `%benefit%` returns the empty list). That leg contributes "
         "exactly zero and always has; the docstring describes a three-source coalesce that is "
         "a two-source coalesce. Measured source depth, which is the ceiling any fix works "
         "against:"),
        ("table", ["source", "rows", "reach", "note"],
         [list(r) for r in PENSION_SOURCES], [26, 8, 12, 42], {1: "RIGHT"}),
        ("p",
         "**Fixed — no SEC path; the dead dependency was deleted.** The "
         "`get(\"pensionDeficit\")` fallback is gone from `capital.off_balance_sheet_obligations`, "
         "the leg is gone from the coalesce, and `_pension_pool`'s docstring now describes the "
         "two sources it actually has. **Zero change to any value** — that was the point, and it "
         "was proved at the source rather than by diffing a cube: an `information_schema` match "
         "for `%pension%` / `%opeb%` / `%benefit%` / `%retire%` across `fundamentals_history` "
         "**and** `fundamentals_sharadar` returns no columns at all, so the leg could not have "
         "fired on any live row. A unit test now pins it: feeding a `pensionDeficit` column must "
         "change nothing. Pension coverage stays at 199 / 489 tickers — a documented substrate "
         "limit, no longer an open defect."),

        ("h2", "4.6 D3 — share-class symbol reconciliation (BRK-B, BF-B)"),
        ("p",
         "Thirteen tickers carry no fundamentals, but **eleven of them are supposed to**. Eight "
         "(GEHC, GEV, HONA, KVUE, Q, SNDK, SOLV, VLTO) are in `INSUFFICIENT_HISTORY_TICKERS` — "
         "under four years of price history, excluded by `load_universe_tickers`. `EA` was "
         "acquired and sits in `redundant_ticks`. Those must not be “fixed”."),
        ("p",
         "**Only BRK-B and BF-B were a defect**: `sharadar_tickers` holds `BRK.B` and `BF.B`, the "
         "roster holds the dash form, and `fundamentals_sharadar` had **zero rows under either** "
         "— the SF1 fetch asked for the dash form and got nothing back. **The mismatch failed "
         "silently**: asking Sharadar for `BRK-B` returns HTTP 200 with an empty body, not a 403 "
         "and not an error, so nothing in the run log ever said so while BRK-B kept appearing in "
         "the cube via the price panel. A live test now pins that behaviour."),
        ("p",
         "**Fixed, and it was larger than the SF1 request.** `sharadar_actions` is fetched "
         "market-wide and is read by `split_events` to de-adjust share counts — it held **108 "
         "`BF.B` and 25 `BRK.B` rows** under the vendor spelling. Repairing only the fetch would "
         "have given both names a history whose share counts were de-adjusted against **no split "
         "events at all**: a silently wrong number rather than a missing one. So the symbol maps "
         "both ways — vendor spelling on the request, the caller's own ticker relabelled onto the "
         "response — and 3,332 already-persisted vendor rows were migrated. The inverse mapping "
         "is deliberately narrow (letters, one dot, one trailing letter): of 25,141 tickers in "
         "`sharadar_actions`, 429 match that share-class shape and 10 do not — foreign listings "
         "(`EVN.AX`, `MUV2.MI`, `TECHM.NS`), warrants (`OXY.WS`) and SPAC units (`AAC.U1`). A "
         "blanket rewrite would have corrupted all ten; 921 tickers already carry a dash and "
         "zero collide."),
        ("p",
         "**Measured**: `fundamentals_history` 489 → **491** tickers and 51,255 → 51,504 rows; "
         "BF-B 125 filings from 1995-09-06, BRK-B 124 from 1995-11-13; and **6 splits applied**, "
         "including **`BRK-B 2010-01-21 ×50.0`**. That last one is the proof the vendor-table "
         "migration was necessary rather than housekeeping: Berkshire's 50:1 B-share split is "
         "exactly the event whose absence would have put the share count — and therefore market "
         "cap, EV and every per-share feature — on the wrong basis for the largest financial in "
         "the index."),
        ("p",
         "⚠ **AVB and EQR are flagged, not actioned.** They are in neither exclusion list, yet "
         "are absent from the 500-row roster while still appearing in the cube via the price "
         "panel. They are the **two zero-coverage REITs** in the table in 4.1 — two large "
         "residential REITs missing from a 30-name sector. Neither stated rule explains them; "
         "raise it before the next roster refresh."),

        ("h2", "4.7 D4 — the valuation yields discard the magnitude of a loss"),
        ("p",
         "`earnings_yield` is 26.96% null, and it decomposes *exactly*. Note that **`netIncome` "
         "itself is only 3.87% null** at filing grain — it is not 27% null, and it is not the "
         "reason the yield is:"),
        ("table", ["component", "rows", "share"],
         [list(r) for r in EY_DECOMPOSITION], [46, 14, 18], {1: "RIGHT", 2: "RIGHT"}),
        ("p",
         "8.51% of filings report a *negative* net income, across **330 of 489 tickers**: two "
         "thirds of the index has posted a loss at some point. "
         "`fundamental_features.py:658` masks them deliberately — "
         "`net_income.where(net_income > 0)` — and for *that feature* the reasoning is sound: a "
         "negative E/P is not “cheap”, so ranking loss-makers by it is noise, and `f_profitable` "
         "carries the regime. The mask is airtight: zero loss-making rows carry a value."),
        ("p",
         "**But the mask is lossy and the flag does not replace it.** A firm losing 2% of its "
         "market cap a year and one losing 60% are both just `f_profitable = 0`. The same "
         "masking discards `fcf_yield` on **460,214 rows (11.95%)** and `book_yield` on "
         "**107,727 rows (2.80%)**."),
        ("p",
         "**The proposed fix was to un-mask the yields** on a one-line rule: *a quantity in the "
         "NUMERATOR over price keeps its sign; a quantity in the DENOMINATOR must be guarded "
         "positive.* The rule is sound in general, and it rests on a premise — **E/P is monotone "
         "across zero** — that was never measured. So it was measured before anything shipped."),
        ("p",
         "**The premise is false.** Spearman IC of the signed yield against `target_rank`, "
         "per date, month-end 1995–2026 (monthly, so overlapping forward returns cannot inflate "
         "the t-stat), computed **inside each sign subset**:"),
        ("table", ["numerator", "NEGATIVE subset (masked)", "positive subset (the design)",
                   "obs / share"],
         [list(r) for r in LOSS_SIGN_FLIP], [24, 28, 26, 18], {3: "RIGHT"}),
        ("p",
         "**The sign FLIPS at zero.** A deeper loss predicts a *better* forward return — the "
         "opposite of the direction E/P carries among profitable names. It is negative in **4 of "
         "4 sub-periods**, survives risk-orthogonalisation (IC −0.0284, t = −3.07 against "
         "`target_epsilon`), and Spearman makes it robust to the −54× tail. So “ranking "
         "loss-makers by E/P is noise” is wrong: it is not noise, it is **reversed** — which is a "
         "*stronger* reason to keep them out of the column, because merging two opposite "
         "relationships into one monotone rank averages them away. Measured directly as the "
         "Q5−Q1 spread in mean `target_rank`:"),
        ("table", ["feature", "MERGED (what un-masking ships)", "MASKED (kept)"],
         [list(r) for r in MERGED_VS_MASKED], [24, 30, 22], {1: "RIGHT", 2: "RIGHT"}),
        ("p",
         "**Decision: the mask stays, and one feature was added instead.** Un-masking shrinks "
         "the univariate edge on the two that matter, so it would have bought null-rate "
         "cosmetics with signal. `loss_intensity` = `-netIncome / mcap`, a positive magnitude "
         "(the house convention for `*_intensity`), defined **only** where earnings are negative "
         "— NaN for profitable names, never 0, since a zero would tie every profitable name at "
         "one value at the bottom of the rank, which is the identical defect D5 fixes on the "
         "dividend zero-fill. Given its own column the model can learn the reversed sign instead "
         "of having it cancel. Verified on the rebuilt cube: **zero** rows where both columns "
         "are defined, **zero** loss-makers carrying an `earnings_yield`, **zero** profitable "
         "names carrying a `loss_intensity`, and 260,689 of 263,776 loss-maker rows covered "
         "(98.8%; the rest have no market cap). The five yields are untouched."),
        ("p",
         "It is deliberately **not** a member of the `value` composite. That composite averages "
         "yields on a “high = cheap = good” thesis; `loss_intensity` is not a cheapness measure, "
         "and its measured direction is a distress/high-beta rebound that merely happens to point "
         "the same way. Folding it into a hand-weighted average would re-impose exactly the "
         "cancellation the separate column exists to prevent."),
        ("p",
         "**One finding recorded rather than actioned**: `book_yield`'s *positive* subset has IC "
         "**−0.0062 (t = −2.18)** over 1995–2026 — a high book yield predicts *worse* forward "
         "returns. That is the book-to-market factor inverted over this sample, not a data "
         "defect, and it belongs to a modelling review rather than a data audit."),
        ("p",
         "**The general lesson, which cost the least and is worth the most:** the plan for this "
         "defect required its own measurement before implementation, and that requirement is the "
         "only reason a five-line change that reduces the edge did not ship. An assertion in a "
         "docstring is not evidence, and neither is a rule that is sound in general."),

        ("h2", "4.8 D5 — four features are fabricated for companies that did not exist yet"),
        ("p",
         "**The most serious defect in this list**, and the only one that puts a *wrong* number "
         "rather than a missing one in front of the model. It was found by asking what the "
         "16.79% in 4.1 actually consists of. On the 565,893 pre-listing padded rows, four "
         "features are not null:"),
        ("table", ["feature", "non-null pre-listing rows", "value carried"],
         [list(r) for r in FABRICATED], [22, 22, 24], {1: "RIGHT"}),
        ("p",
         "Every other feature is correctly NaN there. The 10k–21k pre-listing values on the "
         "*fundamentals* features are **legitimate** — spin-offs such as OTIS and CARR file "
         "before they list (`hist_lo` 2020-02-07 vs `price_lo` 2020-03-19). Those must not be "
         "“fixed”."),
        ("p",
         "**Root cause — two independent paths, same shape.** `dividend_features.py:60` "
         "reindexes to `trading_index × universe` and `.fillna(0.0)`; lines 122/124 apply a "
         "second unconditional `.fillna(0.0)`, line 139 turns a boolean comparison (which can "
         "never be NaN) into `dividend_payer`, and line 152 propagates both into "
         "`shareholder_yield`. The docstring's intent — *non-payers get a real 0 so they rank "
         "correctly* — is **right**; the bug is the *scope* of the fill. Separately, "
         "`fundamental_features.py:307` seeds the Beneish M-score as `DataFrame(-4.84, …)` and "
         "fills each missing component with its neutral value, so a ticker with no data at all "
         "scores exactly **-2.48** — below the classic `M > -1.78` threshold, i.e. a clean bill "
         "of health for a company that does not exist."),
        ("p",
         "**Why it reaches the model even though the padded rows do not.** Padded rows are "
         "dropped at training for want of a label. But `_xs` is a percentile rank computed "
         "**per day across the universe** and `_vs_peers` is a per-day peer-z, so these "
         "fabricated values sit in the cross-section that prices the *real* names:"),
        ("table", ["date", "tickers listed", "carried a rank (before)",
                   "phantom share", "carry a rank (after)"],
         [list(r) for r in PHANTOM_DECAY], [14, 14, 18, 14, 18],
         {1: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT"}),
        ("p",
         "On 1996-01-02 the 191 phantoms were **all tied at rank 0.278**, so the 300 real names "
         "were compressed into [0.278, 1.000] — the bottom 27.8% of the scale occupied by "
         "companies that did not exist, and a real non-payer indistinguishable from "
         "Reddit-in-1996. The contamination decayed monotonically to zero, so **the feature's "
         "scale drifted systematically across the backtest**: the same economic state mapped to "
         "a different feature value in 1996 than in 2026. For a time-series split that is not "
         "noise, it is a trend."),
        ("p",
         "**Fixed** by masking all four to the listed window rather than widening the fill — the "
         "price frame already carried the answer, so no new input was needed: `listed = "
         "close.notna()` applied to `dividend_yield`, `dividend_payer` and `shareholder_yield`; "
         "and the M-score returned only `.where(rev.notna() & assets.notna())`, which is what "
         "its own docstring already promised. The Beneish support is keyed on **data**, not on "
         "listing, so the legitimate spin-off filings above are preserved. The mask was measured "
         "before it was written: it costs **448 rows** of genuine data across the whole 31-year "
         "panel — trading days inside a listed span with no quote, at most 2 per ticker, 0.014% "
         "of listed rows — and a day with no quote has no yield anyway."),
        ("p",
         "**The outcome is better than the plan predicted.** The four null rates were expected to "
         "rise to ~15%; they rose only to 1.59% and 6.39%, because the fabricated rows left the "
         "panel entirely. Those pre-listing rows carried *nothing but* these four values, so once "
         "masked they were wholly NULL and were dropped: **607,359 rows removed (15.5% of the "
         "grid) against 62,126 (1.6%) before**, taking the panel from 3,852,470 to **3,315,035** "
         "rows. The 1996 tie now holds **81 names** — the real non-payers, correctly tied at rank "
         "0.137 of a 300-name cross-section instead of 0.278 of a fictional 491. ⚠ Unlike the "
         "others this **changed values on rows the model trains on**, so the dividend family's "
         "historical IC was measured against a contaminated cross-section and should be expected "
         "to move; re-run the model comparison."),

        ("pagebreak",),
        ("h1", "5. Reproducing this report"),
        ("p",
         "Every number above is re-derivable against the live table; nothing is hard-coded. "
         "`python -m scripts.cube_fundamentals_evidence --out <dir> --format pdf` regenerates "
         "the artefacts and this document; `--check` measures and verifies coverage without "
         "writing anything, exiting non-zero if either check fails."),
        ("p",
         "The generator asserts **both** directions of coverage: every cube column must have "
         "a catalogue entry, and every catalogue entry must correspond to a live column. A "
         "deleted feature cannot leave documentation behind, and a new one cannot ship "
         "unexplained."),
        ("table", ["file", "what it is"], [
            ["`scripts/cube_fundamentals_evidence.py`",
             "the measurement and generator (profile, saturation, coverage assertion)"],
            ["`scripts/cube_feature_catalogue.py`",
             f"the prose half: {m['chars']} entries, no numbers that could go stale"],
            ["`scripts/cube_evidence_narrative.py`",
             "the hand-written sections, shared by the PDF and markdown renderers"],
            ["`scripts/cube_evidence_pdf.py`", "the PDF renderer"],
            ["`cube_part_fundamentals_profile.parquet`",
             "per-column profile (n, null rate, min/p1/p50/p99/max, mean, σ, distinct)"],
            ["`cube_part_fundamentals_saturation.parquet`",
             "per-column clip saturation and max |z|"],
        ], [26, 60], None),

        ("h2", "What this report does not establish"),
        ("p",
         "The evidence here is about **construction**: that each feature computes the formula "
         "it claims, that its nulls are explained, and that its tail behaves. It is not a "
         "filing-level audit — no value was traced back to a filed 10-Q and hand-recomputed. "
         "The vendor substrate underneath (`fundamentals_history`, Sharadar-first) is "
         "validated separately by the `fundamentals-validate` / `fundamentals-triage` path, "
         "and that remains the right tool for “is this number what the company actually "
         "reported?”."),
    ]
