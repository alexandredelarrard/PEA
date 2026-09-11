"""
cube_feature_catalogue.py  (scripts/)
--------------------------------------------------------------------------------------------
The PROSE half of the `cube_part_fundamentals` evidence report: what each characteristic
measures, why its null rate is the right one, and how to read its tail.
`cube_fundamentals_evidence.py` supplies the other half -- the measured numbers -- and joins
the two.

Keyed on the CHARACTERISTIC, i.e. the cube column minus the `f_` prefix and the
`_vs_peers` / `_xs` / `_vs_hist` suffix. The three views are one quantity seen three ways and
the economics belong to the quantity, not to the standardiser.

⚠ NO NUMBERS THAT COULD GO STALE. A measurement written here would be a claim about a table
this file cannot see, and the whole reason this report exists is that the codebase had
accumulated exactly that kind of claim. Coverage, percentiles and clip saturation all come
from the live table at generation time. The few figures that DO appear are properties of a
source or a definition, not of a build -- `employees_sec` at 75.7%, the r = 1.0000 pairs the
audit broke -- and each names what it measured.

The generator asserts BOTH directions: every cube column must have an entry here, and every
entry here must correspond to a live column. An unmatched name in either direction fails the
run, so a deleted feature cannot leave documentation behind and a new one cannot ship
unexplained.

⚠ BOTH DIRECTIONS ARE ASSERTABLE ONLY PER PART, which is why `CATALOGUES` at the bottom is a
registry of one dict per cube part rather than one dict for the cube. `CATALOGUE` here is
`cube_part_fundamentals`; `cube_governance_catalogue.GOVERNANCE` is `cube_part_governance`.
Merging them would put 85 governance names in the fundamentals generator's `unused` list and
fail every run -- the second direction is only meaningful against a named table.
"""

#: characteristic -> (family, what it does, why it is right, how to read the tail)
#:
#: ⚠ THIS DICT IS `cube_part_fundamentals` ONLY, and the scope is load-bearing rather than
#: incidental: `cube_fundamentals_evidence.py` asserts both directions against ONE table and
#: exits non-zero on either failure, so an entry describing another part's column would land in
#: its `unused` list and fail every future run. Both directions are only assertable PER PART.
#: Add another part's prose as its own module and register it in `CATALOGUES` below.
CATALOGUE: dict[str, tuple[str, str, str, str]] = {}


def add(name: str, family: str, what: str, why: str, tail: str) -> None:
    CATALOGUE[name] = (family, what, why, tail)


#: The suffixes a characteristic can wear, and the split that removes them. Defined HERE, in
#: the module that owns the keying convention, so the generator and every checker agree on what
#: a characteristic is -- two implementations of this would be two answers to "is
#: `f_board_size_vs_peers` documented?".
SUFFIXES = ("_vs_peers", "_vs_hist", "_xs")


def split(col: str) -> tuple[str, str]:
    """`f_fcf_yield_vs_peers` -> ('fcf_yield', '_vs_peers')."""
    base = col[2:] if col.startswith("f_") else col
    for s in SUFFIXES:
        if base.endswith(s):
            return base[: -len(s)], s
    return base, ""


# --------------------------------------------------------------------- value #
add("earnings_yield", "value", "TTM net income / market cap (E/P), masked to profitable names only.",
    "Nulls are loss-makers by design, and the mask is MEASURED rather than asserted. The old reason given -- 'ranking loss-makers by E/P is noise' -- is wrong: inside the loss subset the signed yield has IC -0.0297 (t = -3.17) against +0.0116 (t = +3.31) among profitable names, so it is not noise but REVERSED, which is a stronger reason to keep it out. See `loss_intensity`, which is where that information now lives.",
    "The high tail is deep-value/cyclical trough earnings; there is no negative tail because the numerator is masked at 0.")
add("loss_intensity", "value", "The annual LOSS as a positive share of market cap: -netIncome / mcap, defined only where TTM net income is negative.",
    "Carries what `earnings_yield`'s mask discards, in its own column so its sign can be learned separately. Measured Spearman IC of the signed yield vs `target_rank`, month-end 1995-2026, INSIDE each sign subset: -0.0297 (t = -3.17) among loss-makers against +0.0116 (t = +3.31) among profitable names -- a deeper loss predicts a BETTER forward return, negative in 4/4 sub-periods, and the same flip holds for free cash flow (-0.0159 vs +0.0162). Folding the two into one monotone column averages them away: merged Q5-Q1 spread 0.0040 vs the masked column's 0.0053. NaN for profitable names, never 0, so they do not pile into one tie at the bottom of the rank.",
    "High = a large loss relative to equity value. The measured direction is that the deep tail OUTPERFORMS -- a distress/high-beta rebound effect, not cheapness -- which is exactly why it must not share a column with E/P.")
add("sales_yield", "value", "TTM revenue / market cap (S/P).",
    "Lowest null rate of the yields: revenue is always positive so nothing is masked; nulls are only missing price or missing revenue.",
    "The high tail is low-margin distribution/retail (huge sales per dollar of equity value); the low tail is high-multiple software. Both are real business-model differences, which is why the feature is used peer-relative.")
add("book_yield", "value", "Positive stockholders' equity / market cap (B/P).",
    "Nulls are negative-book-equity names, masked deliberately; `negative_equity` is the flag that carries them.",
    "The high tail is financials and asset-heavy names trading near/below book; capped naturally because equity cannot exceed the asset base.")
add("fcf_yield", "value", "Positive TTM free cash flow (OCF - capex) / market cap.",
    "Nulls are cash-burning names, masked for the same monotonicity reason as E/P. NO LONGER identical to `intrinsic_yield`: that pair sat at r = 1.0000 until `revenueGrowth` was computed.",
    "The high tail is mature cash cows at trough multiples. Extreme values are bounded by the mask (FCF > 0) and by the peer-z clip.")
add("ebitda_to_ev", "value", "Positive TTM EBITDA / enterprise value -- the capital-structure-neutral cheapness measure.",
    "Nulls are negative-EBITDA names plus filers with no debt/cash tags to build EV from. EV is the fully-diluted build: diluted mcap + debt + leases + minority & redeemable NCI + preferred + pension deficit - liquid assets.",
    "The high tail is levered value; EV in the denominator is what stops a heavily-indebted name looking cheap on equity alone.")
add("fcf_to_ev", "value", "Positive TTM free cash flow / enterprise value.",
    "The cleanest cross-sector cash-valuation yield -- unlike E/P it is not distorted by D&A policy or capital structure.",
    "Tails match `ebitda_to_ev` in shape; the two differ where capex is heavy, which is the point of carrying both.")
add("eps_yield", "value", "Reported DILUTED EPS / price -- E/P net of preferred and after dilution.",
    "Reads `epsDiluted` (98.8% coverage) rather than dividing two independently forward-filled columns, so it cannot pair a stale share count with a fresh income figure.",
    "Differs from `earnings_yield` exactly by the preferred dividend and the diluted-vs-basic share gap, which is why both are kept.")
add("intrinsic_yield", "value", "Two-stage DCF on TTM free cash flow / price -- an intrinsic-value-to-price yield.",
    "The growth term is now real: it reads the computed `revenueGrowth`. While that column was never produced, the DCF grew every firm at the constant 2.5% terminal rate and the feature reduced to `fcf x constant`, correlating 1.0000 with `fcf_yield`. Measured r after the fix: 0.930.",
    "The tail is names whose modelled value is a large multiple of price -- deep value or a growth assumption the market rejects. The DCF's own terminal-growth cap bounds it.")
add("pegy", "value", "Trailing P/E divided by (EPS growth% + dividend yield%) -- growth- and income-adjusted P/E.",
    "The growth term PREFERS projected NTM/TTM EPS growth from the analyst archive and falls back to realised TTM net-income growth. The denominator is masked to > 0, so a shrinking, non-paying firm is null rather than negative -- PEGY is undefined there, not 'attractive'.",
    "Low = cheap for the growth bought. The masked denominator is what stops the classic PEG sign flip that makes a collapsing company look like the cheapest name in the universe.")
add("ffo_yield", "value", "REIT funds-from-operations (net income + D&A) / market cap, GICS-scoped to equity REITs.",
    "Null for ~94% of the panel BY DESIGN -- it is emitted only for the GICS equity-REIT group. Previously gated on tagging habits, which handed FFO to ~56 non-real-estate lessors and missed 9 of 31 REITs.",
    "Only the D&A leg of NAREIT FFO is available (no `gainOnDispositions`, no `realEstateImpairment` in SF1), so a REIT that sold property at a gain reads high. Documented, not silent.")

# ------------------------------------------------------------------- quality #
add("grossMargins", "quality", "Gross profit / revenue, straight from the history.",
    "A reported ratio, not a derived one -- nulls are filers who do not separate cost of revenue (many financials).",
    "Near 1.0 for software/pharma, near 0.1 for distributors. Used `_vs_peers` precisely because the level is an industry property.")
add("operatingMargins", "quality", "Operating income / revenue.", "Same source discipline as `grossMargins`.",
    "Negative for pre-profit biotech and post-impairment years -- both real states, kept rather than masked because the sign is informative here.")
add("profitMargins", "quality", "Net income / revenue (TTM).", "Reported ratio.",
    "The extreme high tail is alternative-asset managers (KKR, APO, ARES, BX) where consolidated `netIncome` exceeds `totalRevenue` -- 72 rows, all real, an artefact of investment-company accounting rather than an extraction bug.")
add("returnOnEquity", "quality", "Net income / stockholders' equity.",
    "NO LONGER identical to `sustainable_growth_rate`: that pair was r = 1.0000 while `dividendsPaid` was stored outflow-negative, because the payout clipped to 0 and retention to 1 for every payer. Measured r after the sign fix: 0.854.",
    "The tail is small-equity, high-return names and post-buyback capital structures; the `negative_equity` flag separates the ones where the ratio inverts meaning.")
add("gross_profitability", "quality", "Novy-Marx: gross profit / total assets (ex-lease asset base).",
    "Stays defined for loss-makers, which is exactly the cohort the earnings yields mask -- that is why it is carried alongside them.",
    "High tail is asset-light software; low tail is utilities and banks. The ex-lease base is what stops FY2019's ASC-842 adoption reading as a step change.")
add("fcf_margin", "quality", "Free cash flow / revenue.", "Derived from two well-covered columns.",
    "Can exceed 1.0 only where revenue is tiny relative to cash generation; the negative tail is genuine cash burn.")
add("accruals", "quality", "(Net income - free cash flow) / revenue -- the Sloan accrual anomaly.",
    "High accruals predict underperformance; the numerator is the profit not backed by cash.",
    "Both tails matter: a large positive value is aggressive revenue recognition, a large negative one is a firm whose cash beats its book profit.")
add("piotroski_f_score", "quality", "0-9 count of fundamental-health binaries (profitability, leverage/liquidity, efficiency).",
    "Scored ONLY where the core inputs (assets, net income, operating cash flow) all exist, so a data-poor name is not a false 0.",
    "Bounded 0-9 by construction. A 9 is a firm improving on every axis simultaneously and is genuinely rare.")
add("rule_of_40", "quality", "TTM revenue-growth% + FCF-margin% -- the software rule of thumb.",
    "Defined for any filer with revenue and cash flow; not sector-gated because the trade-off it encodes is universal.",
    "Above 40 is elite (fast grower OR cash cow). The extreme negatives are hyper-growth cash burners, which is the state the metric is designed to penalise.")
add("asset_growth", "quality", "YoY growth of the ex-lease total asset base (Fama-French CMA).",
    "On the EX-LEASE base deliberately: the FY2019 ASC-842 right-of-use recognition would otherwise read as a one-off investment spree across every lease-heavy name.",
    "Sign convention is -1 in the model: aggressive expanders subsequently underperform. The high tail is large acquisitions.")
add("debtToEquity", "quality", "Reported total debt / equity.", "A reported ratio.",
    "Explodes where equity is small or negative; `negative_equity` flags that regime and the peer-z clip bounds it.")

# ------------------------------------------------------------------ distress #
add("altman_z", "distress", "Altman Z (market-value variant): 1.2 WC/TA + 1.4 RE/TA + 3.3 EBIT/TA + 0.6 mcap/TL + 1.0 Sales/TA.",
    "On the ex-lease asset base -- otherwise the ASC-842 jump pushes every lease-heavy retailer toward 'distressed' in FY2019 for an accounting reason.",
    "Below ~1.8 is the classic distress zone. The high tail is asset-light, high-multiple names where mcap/TL dominates; that is a known property of the market-value variant, not a defect.")
add("net_debt_to_ebitda", "distress", "(Total debt - cash) / TTM EBITDA.",
    "Only meaningful for EBITDA > 0, so negative-EBITDA names are null rather than sign-flipped.",
    "The high tail is genuinely levered credits. Net cash names go negative, which reads correctly as 'more than debt-free'.")
# DELETED 2026-09-05: `net_debt_incl_offbs_to_ebitda`, the off-balance-sheet-inclusive twin.
# Its entry claimed the wedge against `net_debt_to_ebitda` "IS the off-balance-sheet
# obligation, which is the whole point" -- but three of the four legs that made the wedge
# (operating leases, finance leases, ARO) have no column on the Sharadar substrate at all,
# and the fourth reaches 199 of 491 tickers. It survived only because the distress block
# computed debt from `longTermDebt + shortTermDebt` while this one used the reconciled
# `totalDebt`; once both read `totalDebt` the two became the SAME NUMBER (identical
# fingerprint hashes, r = 0.9955 live). The pension overhang it folded in is still carried,
# undiluted, by `pension_overhang_leverage` and `pension_retirement_liability`.
add("interest_coverage", "distress", "TTM EBITDA / |interest expense|.",
    "Takes the absolute value of interest so a sign convention in the source cannot invert the ratio.",
    "Very high for debt-free names (small denominator); low single digits is the covenant-stress zone.")
add("ebit_interest_coverage", "distress", "Operating income / |interest expense| -- the stricter, EBIT-based coverage.",
    "Correlates 0.993 with the EBITDA version by construction (they differ only by D&A), and is kept because the D&A wedge is exactly what distinguishes a capital-intensive borrower.",
    "Below ~2x is structural distress risk; the two coverage measures diverge most for the names where that matters.")
add("current_ratio", "distress", "Current assets / current liabilities.", "A textbook liquidity ratio off two well-covered columns.",
    "Below 1 means short-term obligations exceed short-term assets. Banks sit low structurally, which is why it is read peer-relative.")
add("cash_to_debt", "distress", "Cash / total debt.", "Defined wherever both legs exist.",
    "Above 1 is a net-cash balance sheet; near 0 is a fully-drawn borrower.")
add("refinancing_risk", "distress", "Short-term debt / (cash + positive trailing free cash flow).",
    "The denominator is the self-funding capacity; the numerator is what must actually be rolled.",
    "Much greater than 1 means the firm must refinance a slug it cannot self-fund -- exposed to rate spikes and frozen credit markets.")

# ---------------------------------------------------------- growth / trend #
add("revenueGrowth", "growth", "YoY revenue growth, computed AT CUBE TIME against the filing nearest 365 calendar days back.",
    "A `CUBE_TIME_COLUMN`: the history build could only take a 4-ROW offset, and under the publication-event grain an amendment row makes four rows about nine months -- the wrong denominator for exactly the names that restate. The as-of match is NEAREST with a 180-day tolerance, whose worst case (as_of - 185 days) is still strictly before the row's own filing date, so both legs are public.",
    "Correlates 0.998 with `y_rev_growth`, which is the row-offset version -- the residual 0.2% is precisely the amendment/irregular-cadence cases the calendar match exists to fix.")
add("earningsGrowth", "growth", "YoY net-income growth, same cube-time 365-day as-of match.",
    "Same construction as `revenueGrowth`. A zero or negative prior is masked, not divided by: a swing from loss to profit has no percentage growth, and dividing by it manufactures a huge number with an arbitrary sign.",
    "Nulls are higher than for revenue because the prior-period mask bites on every loss year.")
add("y_rev_growth", "growth", "TTM revenue vs TTM one year ago, on the fiscal filing series (row offset).",
    "Seasonality-free by construction: same fiscal quarter each year, `yoy_periods` filings back.",
    "The row-offset twin of `revenueGrowth`; both are kept because they disagree exactly where the filing cadence is irregular.")
add("y_earnings_growth", "growth", "TTM net income vs TTM one year ago, fiscal series.", "As above.",
    "Sign flips around zero-earnings years are real; the peer-z clip keeps them from dominating.")
add("y_rev_growth_accel", "growth", "Change in the YoY revenue-growth rate -- second derivative of sales.",
    "Acceleration is a different signal from level growth and is what tends to move multiples.",
    "Symmetric tails; a large positive is an inflection, a large negative is a decelerating grower.")
add("q_rev_growth", "growth", "LATEST-QUARTER revenue YoY (discrete single quarter, not TTM).",
    "Reads the discrete `revenue_q` column, so it captures the inflection TTM smooths away.",
    "Noisier than the TTM version by design -- that is why both exist and the model chooses.")
add("q_earnings_growth", "growth", "Latest-quarter net income YoY.", "As above, on `netIncome_q`.",
    "Very wide tails around loss-to-profit quarters; the peer-z clip and winsorization bound them.")
add("rev_growth_accel", "growth", "Latest-quarter YoY growth minus the previous quarter's YoY growth.",
    "The quarterly acceleration term.", "Symmetric; the extremes are single-quarter inflections.")
add("q_margin_vs_ttm", "growth", "Latest-quarter net margin minus the TTM net margin -- margin inflection.",
    "Both legs come from the same filing, so no stale/fresh mismatch.",
    "Positive means the most recent quarter is running above the trailing year: an improving trend the TTM figure has not caught up with.")
add("y_margin_vs_ttm", "growth", "YoY change in the TTM profit margin.", "A `diff`, not a `pct` -- a margin is already a ratio.",
    "Percentage-point change; the tails are impairment and recovery years.")
add("gross_margin_chg", "growth", "YoY change in gross margin (percentage points).",
    "Deliberately on `_xs` not `_vs_peers`: a CHANGE in margin is comparable across industries even though the LEVEL is not.",
    "A software company and a grocer both improving 2pp is the same news, which is the rationale for the cross-sectional view.")
add("operating_margin_5y_chg", "growth", "Operating margin now vs ~5 trading years ago (percentage points).",
    "Asks whether margin expansion is STRUCTURAL rather than one good year. Higher null rate than the 1y version purely because it needs five years of point-in-time history.",
    "Positive = durable expansion. The tail is genuine business-model transitions.")
add("fcf_growth", "growth", "YoY growth in free cash flow, fiscal series.", "Standard YoY on a well-covered column.",
    "Wide tails around sign changes in FCF, bounded by winsorization.")
add("rd_intensity", "growth", "R&D expense / revenue.",
    "Null for filers who report no R&D -- absence of the tag here means the firm genuinely does not separate R&D, not a data gap.",
    "Above 0.2 is research-led (biotech, semis). Read `_vs_peers` because the level is a sector property.")
add("hyper_growth", "growth", "0/1 flag: YoY revenue growth above 25%. Emitted RAW, not peer-standardised.",
    "A NaN base gives a NaN flag, never a false 0 -- 'no data' must not read as 'not growing'. Now genuinely alive: it reads the `revenueGrowth` the cube computes.",
    "Binary by construction. The model conditions on the regime instead of averaging a feature whose meaning flips across it.")

# --------------------------------------------------------- capital / payout #
add("shares_growth", "capital_allocation", "YoY change in shares outstanding (net of buybacks and issuance).",
    "Net, so it can be masked by a firm that buys back exactly what it issues -- which is why `sbc_intensity` (the gross give-away) is carried too.",
    "Negative = net shrinkage (shareholder-friendly). The positive tail is dilutive equity raises.")
add("diluted_shares_growth", "capital_allocation", "YoY change in the fully-diluted share count.",
    "Catches dilution the basic count hides.", "Positive tail is convertible/option-heavy capital structures.")
add("option_overhang", "capital_allocation", "(Diluted - basic shares) / basic, computed by the extractor on periods where BOTH counts are reported.",
    "Deliberately NOT computed here from two independently forward-filled columns: that would compare a stale diluted count with a fresh basic one.",
    "The high tail is early-stage tech with heavy option pools -- exactly the dilution the net share count hides once buybacks offset issuance.")
add("sbc_intensity", "capital_allocation", "Stock-based compensation / revenue -- the GROSS equity give-away.",
    "Complements `shares_growth`, which is net and can read zero while a firm issues heavily and buys back in equal size.",
    "Above ~0.1 is a material transfer from shareholders to employees; concentrated in software.")
add("sbc_to_ocf", "capital_allocation", "SBC / operating cash flow -- how much reported operating cash is really non-cash comp.",
    "Both legs from the same cash-flow statement.",
    "Approaching 1 means reported OCF is largely an add-back rather than cash earned.")
add("sbc_to_buyback", "capital_allocation", "SBC / gross buyback magnitude. Above 1 = the repurchase is mopping up option dilution, not returning capital.",
    "The denominator is `capital.share_repurchases`, which floors Sharadar's NET issuance at 0 and negates it -- the source column is net issuance, so a repurchaser reads negative there.",
    "Above 1 is the diagnostic state. Names below 0.3 are genuinely returning capital.")
add("acquisition_intensity", "capital_allocation", "|Net cash paid for acquisitions| / ex-lease assets.",
    "Uses `.abs()` rather than a floor because both directions are M&A activity: 27,731 negative rows (purchases) to 9,039 positive (net disposal years). Intensity is unsigned by construction.",
    "The high tail is transformational deal years, which is precisely the cohort `intangibles_roic_drag` then judges.")
add("intangibles_growth", "ma_digestion", "YoY growth in the acquired-intangibles balance -- the balance-sheet trace of M&A.",
    "On Sharadar's COMBINED `intangibles`. The bare `goodwill` this used to read is written by no producer, so it grew nothing.",
    "Large positives are deal years; the exposure a future writedown lands on.")
add("roic_incl_intangibles", "ma_digestion", "NOPAT / invested capital, with acquired intangibles INCLUDED in the capital base.",
    "Invested capital includes capitalised leases, matching how leases are already treated as debt in EV and in the leverage ratios.",
    "This is the return the buyer actually earns on what was paid, goodwill and all.")
add("roic_ex_intangibles", "ma_digestion", "The same NOPAT over invested capital LESS acquired intangibles.",
    "NO LONGER identical to its twin: the pair was r = 1.0000 because the deduction read the bare `goodwill`, which subtracted nothing. Measured r after moving to Sharadar's `intangibles`: 0.549.",
    "Structurally higher than the incl-version (a smaller denominator). It is a WIDER deduction than a textbook ex-goodwill ROIC -- it also removes purchased patents, customer lists and brands -- but it is the same deduction for every ticker, which is what makes the cross-sectional rank meaningful.")
add("intangibles_roic_drag", "ma_digestion", "`roic_incl_intangibles - roic_ex_intangibles`: how much acquired intangibles dilute returns.",
    "STRUCTURALLY NEGATIVE, and the composite signs it '+' for that reason. It was identically zero across 3.04M rows while the deduction was empty.",
    "The most negative names are the serial acquirers who overpaid; nearer zero is organic growth or a well-digested deal.")
add("intangibles_to_assets", "ma_digestion", "Acquired intangibles / ex-lease total assets.", "A balance-sheet weight, not a flow.",
    "Above ~0.5 means most of the balance sheet is purchase accounting -- the writedown exposure.")
add("intangibles_to_equity", "ma_digestion", "Acquired intangibles / positive book equity.",
    "Equity masked to > 0 so the ratio does not invert on negative-book names.",
    "Above 1 means a full writedown would wipe out book equity. That is the tail the feature exists to find.")
add("sga_elasticity", "ma_digestion", "%dSG&A / %dRevenue.",
    "Guards against a near-flat revenue denominator (requires a >= 2% move), which would otherwise make the ratio meaningless and explode.",
    "Below 1 = synergies captured; around 1 = a bolt-on with no integration.")

# -------------------------------------------------------- forensic / quality #
add("dso", "accounting_quality", "Days sales outstanding: receivables / revenue x 365.",
    "Working-capital day counts have ONE owner (this panel); the sector panel used to compute them differently, producing `_x`/`_y` columns whose meaning depended on merge order.",
    "Rising DSO is the classic channel-stuffing tell. Very high values are genuine long-cycle industrials.")
add("dso_change", "accounting_quality", "YoY change in DSO (days).", "A difference, so it is unit-preserving.",
    "The positive tail is receivables growing faster than sales -- revenue recognised ahead of collection.")
add("dpo", "accounting_quality", "Days payable outstanding: payables / COGS x 365.", "Same ownership discipline as DSO.",
    "Rising DPO is supplier-funded growth: stretching vendors to finance the business.")
add("dpo_change", "accounting_quality", "YoY change in DPO (days).", "As above.", "A large positive is a working-capital-funded stretch.")
add("dio", "accounting_quality", "Days inventory outstanding: inventory / COGS x 365.",
    "Null for the many filers with no inventory (software, services, financials) -- absence is meaningful, not missing.",
    "The high tail is jewellery/aircraft/pharma-style long shelf lives.")
add("cash_conversion_cycle", "accounting_quality", "DSO + DIO - DPO: days of cash tied up in working capital.",
    "Emitted only where all three legs exist, so it is never a partial sum.",
    "Negative is the enviable state (Dell/Amazon working-capital model: paid before you pay). The high tail is inventory-heavy manufacturing.")
add("beneish_m_score", "accounting_quality", "Beneish (1999) 8-variable earnings-manipulation model.",
    "Missing indices default to NEUTRAL (1.0, TATA to 0), so M is defined wherever revenue and assets exist rather than dropping the whole score for one absent leg. Each index is clipped to +-10 before weighting.",
    "Above -1.78 is the classic manipulator screen. The near-100% coverage is a consequence of the neutral-default design and is intentional.")
add("nci_income_share", "accounting_quality", "Income attributable to non-controlling interests / |net income| -- the share of consolidated profit the parent's shareholders do not own.",
    "Reads `netIncomeToNci`, the LIVE column name. It was written `nciIncome`, which nothing has ever produced, so the feature emitted nothing -- and the column-existence diff could not see it because the argument is a loop variable, not a string literal.",
    "A high share means headline EPS overstates what accrues to the common holder. `netIncome` maps from Sharadar's `consolinc`, which includes NCI, so the ratio is well-posed.")
add("earnings_quality", "accounting_quality", "Operating cash flow / net income.",
    "Below 0.8 is accrual risk: profit not backed by cash.",
    "The negative tail is a profitable firm burning cash -- the state the metric exists to surface.")
add("accruals_ratio", "accounting_quality", "(Net income - operating cash flow) / assets -- Sloan accruals on the asset base.",
    "Complements `accruals` (same numerator, revenue denominator); the asset base is the Sloan formulation.",
    "Both tails are informative, as with `accruals`.")

# ------------------------------------------------------------ reinvestment #
add("da_to_capex", "reinvestment", "|D&A| / |capex|.",
    "Both legs made unsigned so a source sign convention cannot invert the ratio.",
    "Above 1 means the firm is consuming its asset base faster than it reinvests -- aging PP&E, a likely future capex cliff, and reported earnings flattered by low capex.")
add("da_minus_capex_growth", "reinvestment", "D&A growth minus capex growth (YoY).",
    "The second-order version: is the under-investment widening?",
    "Positive and large is the deteriorating case; the model learns the sign.")
add("capex_intensity", "reinvestment", "Capex / revenue.",
    "Repo convention stores cash outflows POSITIVE, so this is a positive intensity rather than a negative flow.",
    "Utilities and semis sit high, software near zero -- read `_vs_peers` for that reason.")
add("capex_to_dep", "reinvestment", "Capex / D&A -- the maintenance-vs-growth capex split.",
    "The reciprocal view of `da_to_capex`, kept because the two behave differently in the tails.",
    "Below 1 sustained is under-investment; well above 1 is a build-out.")
add("reinvestment_rate", "reinvestment", "(Capex - D&A + dNWC) / NOPAT -- net cash ploughed back per dollar of after-tax operating profit.",
    "NOPAT uses the effective tax rate clipped to [0, 50%] and defaulted to 21%, so it is defined universe-wide rather than dropping out on a single odd tax year.",
    "Above 1 means the firm reinvests more than it earns -- funded by debt or equity; the growth/leverage trade-off in one number.")
add("asset_turnover", "quality", "Revenue / AVERAGE total assets (mean of current and 1y-prior, falling back to period-end).",
    "Averaging the denominator is what makes it comparable for a firm that made a large acquisition mid-year.",
    "High for retail/distribution, low for utilities and REITs.")
add("gmroi", "quality", "Gross profit / average inventory -- gross margin return on inventory investment (retail).",
    "Emitted only where inventory is actually reported, so it does not manufacture a value for inventory-free businesses.",
    "The retail productivity measure; the high tail is fast-turning specialty retail.")
add("fixed_cost_coverage_margin", "quality", "(Gross profit - EBITDA) / revenue -- overhead intensity.",
    "The wedge between gross and EBITDA margin IS the operating-cost base.",
    "High means a heavy fixed-cost structure and therefore high operating leverage in both directions.")
add("effective_tax_rate", "quality", "Income tax expense / pretax income -- the RAW, unclipped, un-defaulted ratio.",
    "Deliberately distinct from the internal `_nopat_tax_rate`, which IS clipped to [0, 50%] and defaulted to 21%. Two different numbers for two different jobs; the shared name was a trap and was renamed.",
    "Values outside [0, 1] are real: loss carry-forwards, one-off settlements and the 2017 TCJA revaluation all produce them, and masking them would hide the event.")

# ---------------------------------------------------------------- operating #
add("sga_intensity", "operating", "SG&A / revenue.", "A reported-cost intensity.",
    "A strong sector property -- read `_vs_peers`.")
add("sga_growth", "operating", "YoY growth in SG&A.", "Fiscal-series YoY.",
    "Paired with revenue growth to form `operating_leverage`.")
add("operating_leverage", "operating", "Revenue growth minus SG&A growth.",
    "A difference of two growth rates, so it is scale-free and does not explode on a small denominator (unlike the elasticity form).",
    "Positive = sales growing faster than selling cost (scalable); negative = growth is being bought with rising SG&A.")
add("operating_leverage_elasticity", "operating", "%dOperating income / %dRevenue.",
    "Requires a >= 2% revenue move so a flat denominator cannot blow it up. Distinct from `operating_leverage`, which is a difference not a ratio.",
    "Above 1 = a scalable model (profit growing exponentially against linear sales); below 1 = diseconomies of scale.")
add("margin_expansion_delta", "operating", "(1y change in gross margin) - (1y change in EBITDA margin).",
    "Isolates whether margin gains are being kept or spent.",
    "Positive means gross margin is expanding while EBITDA margin lags -- overhead control is slipping. Both expanding is true pricing power.")
add("nwc_elasticity", "operating", "%dNet working capital / %dRevenue.",
    "Same >= 2% revenue guard as the other elasticities.",
    "Above 1 = cash-hungry growth: every extra dollar of sales locks up more than a dollar of working capital.")
add("deferred_rev_intensity", "operating", "Deferred revenue / revenue -- contracted but unrecognised billings.",
    "NOT sector-gated on purpose: deferred revenue is meaningful for any subscription or contract-backed model, and it is only reported by filers that have it.",
    "High is a subscription book with visibility; the tail is enterprise software and maintenance-heavy industrials.")

# --------------------------------------------------------------- workforce #
add("revenue_per_employee", "workforce", "TTM revenue / headcount, from the 10-K body-text employee count.",
    "Reads `employees_sec` -- SEC-owned in the Sharadar-first merged table (SF1 does not carry headcount), so `merge_history` namespaces it. Reading the bare `employees` returned an empty frame and killed all four workforce features on the first lookup, before revenue was ever read. Live coverage 75.7%.",
    "Enormous across sectors (a bank vs a restaurant chain) which is why it is read `_vs_peers`; within a peer set it is a genuine productivity measure.")
add("employee_growth", "workforce", "YoY change in headcount.",
    "Past-vs-past comparison, so leak-free; point-in-time from each 10-K's `as_of`.",
    "Large positives are acquisitions or hiring sprees; large negatives are restructurings -- both real corporate events.")
add("revenue_per_employee_growth", "workforce", "YoY growth in revenue per employee -- productivity trend.",
    "Asks whether revenue is outgrowing headcount (operating leverage) or merely scaling with the people pool.",
    "The positive tail is genuine automation/AI-era productivity gains, which is why the composite uses it.")
add("headcount_elasticity", "workforce", "%dEmployees / %dRevenue.",
    "Same >= 2% revenue guard as the other elasticities.",
    "Below 1 = revenue outgrowing the people pool (scale/synergies captured); around 1 = headcount scaling 1:1 with acquired revenue, i.e. integration not landing.")

# ---------------------------------------------------------------- pension #
add("pension_funded_ratio", "pension_risk", "Plan assets / projected benefit obligation, from the Financial-Statement-and-NOTES footnote tags.",
    "Sparse BY DESIGN -- only defined-benefit sponsors have a PBO, and the figure lives in a footnote tag (`notes_num`), not the primary statements. The bulk tables are now scoped to the fundamentals universe first, so filers with no balance sheet no longer graft themselves onto the panel.",
    "1.0 is fully funded; below ~0.8 is a material deficit. The legacy-industrial tail is real.")
add("pension_retirement_liability", "pension_risk", "The coalesced recognised net pension/OPEB deficit level.",
    "Gap-filled across three sources in order of directness (bulk FSDS recognised liability, companyfacts `pensionDeficit`, footnote PBO minus plan assets) using `combine_first` -- NOT a sum, so there is no OPEB double count.",
    "Floored at 0: only underfunding is debt-like, an overfunded plan is not a negative liability.")
add("pension_overhang_leverage", "pension_risk", "The coalesced net deficit / market cap.",
    "The equity-scaled version of the level -- the burden actually overhanging the stock.",
    "`pension_underfunding_to_mcap` was deleted from beside it: it scaled the footnote deficit, which is the LAST fallback inside this very pool, so the two were r = 1.000000 on their overlap at 8.3% coverage against this one's 28.3% -- a strict subset, not a second view.")
add("pbo_to_mcap", "pension_risk", "GROSS projected benefit obligation / market cap.",
    "Genuinely different from the deficit ratio: it flags rate and return sensitivity even for a FULLY FUNDED plan, which the net figure shows as zero.",
    "The tail is legacy manufacturers whose pension is larger than the equity it sits under. ⚠ THE WORST CLIP SATURATION IN THE TABLE (9.7%), and it is SKEW, not sparsity: peer-z mean +0.76 against a median of -0.21 with σ = 2.78. Sparsity is ruled out by `pension_funded_ratio`, which is sparser still (99.3% null) and saturates 0.0%. A gross obligation floored at zero with a handful of names at multiples of their own market cap cannot be symmetric, and z-scoring it puts that tail on the clip. Read the `_xs` percentile view for this one.")

# ----------------------------------------------------- shareholder return #
add("dividend_yield", "shareholder_return", "TTM dividends per share / price, RECONCILED against the SEC cash-flow `dividendsPaid` total.",
    "The per-share ex-date history is primary; the cash-flow total gap-fills. Non-payers get a real 0 rather than NaN so they rank correctly instead of dropping out of the cross-section.",
    "The high tail is REITs and utilities. A yield that looks extreme is usually a price collapse, which is why `dividend_coverage` is carried alongside.")
add("dividend_growth", "shareholder_return", "1-year growth in the dividend.", "Per-share history first, cash-flow total as fallback.",
    "Cuts are the informative tail and they are rare, which is what makes them predictive.")
add("dividend_growth_5y", "shareholder_return", "5-year dividend CAGR.",
    "Needs five years of point-in-time history, which is the whole reason for the higher null rate.",
    "Sustained double-digit growth is the dividend-growth cohort; negative is a cutter.")
add("dividend_payer", "shareholder_return", "0/1 flag: does the firm pay a dividend at all?",
    "Set from EITHER the ex-date history or a positive `dividendsPaid`, so a gap in one source does not misclassify a payer.",
    "Binary. It lets the model condition on the regime rather than average a yield across payers and non-payers.")
add("dividend_payout_ratio", "shareholder_return", "Total dividends / positive net income.",
    "Net income masked to > 0 so a loss year does not produce a negative payout ratio.",
    "Above 1 means the payout exceeds earnings -- funded from the balance sheet, the classic pre-cut state.")
add("dividend_coverage", "shareholder_return", "Free cash flow / dividends paid.",
    "The cash-based safety test, complementing the earnings-based payout ratio.",
    "Above 1 = free cash flow covers the payout; below 1 or negative = it does not, which is the cut signal.")
add("shareholder_yield", "shareholder_return", "Dividend yield + net buyback yield.",
    "Captures firms returning capital through repurchases instead of dividends, which a dividend yield alone scores as zero.",
    "The high tail is heavy repurchasers; negative means net issuance, i.e. the firm is taking capital in.")
add("payout_ratio", "shareholder_return", "TOTAL shareholder payout (dividends + buybacks) / net income.",
    "Both legs are positive magnitudes: `dividendsPaid` is stored outflow-positive (the field map now flips Sharadar's negative, matching `capex`) and `share_repurchases` floors net issuance at 0. While the dividend leg was negative this was a dividend payout ratio wearing a total-payout name.",
    "Above 1 means returning more than earned. The high tail is mature cash generators running down balance sheets.")
add("buyback_intensity", "shareholder_return", "Gross buyback magnitude / revenue.",
    "Derived from `equityIssuanceNet` negated and floored at 0 -- the source column is NET issuance, so a repurchaser reads negative there and an issuer reads positive.",
    "The high tail is large-cap repurchase programmes; zero is either no buyback or a net issuer.")
add("dps_growth", "shareholder_return", "YoY growth in declared dividends per share.",
    "From the fiscal `dividendsPerShare` series, so it is per-share and unaffected by share-count changes.",
    "Complements `dividend_growth`, which reconciles the ex-date and cash-flow views.")
add("sustainable_growth_rate", "growth", "ROE x retention ratio -- the maximum growth fundable without new equity or leverage.",
    "NO LONGER identical to `returnOnEquity`: the `clip(0, 1)` on the payout makes the `dividendsPaid` sign load-bearing, and while that column was outflow-negative every payer's payout clipped to exactly 0 and retention to 1. Measured r after the sign fix: 0.854.",
    "The clip is still correct -- a payout above 100% of earnings is not a negative retention -- but it only reads as a payout ratio because the outflow is now positive.")

# -------------------------------------------------------------- financials #
add("bank_roa", "bank", "Net income / assets, GICS-gated to banks.",
    "The ONLY bank KPI Sharadar can feed. Every other one needed `loans`, `depositsDomestic`, `noninterestExpense`, `provisionForCreditLosses` or `netInterestIncome`, none of which SF1 delivers; each measured ZERO values against the live table and was deleted rather than left dormant.",
    "For a bank, assets ARE the earning base, so ROA is the right survivor. Sparse because it is gated to one industry group.")
add("book_value_growth", "financials", "YoY growth in book equity, GICS-gated to Financials.",
    "For a financial, book value IS the productive asset, which is why the growth of it is the sector's growth measure.",
    "Sparse by design (one sector). The negative tail is mark-to-market capital erosion.")
add("aoci_to_equity", "financials", "Accumulated other comprehensive income / equity, GICS-gated to Financials. Signed.",
    "AOCI is mostly the available-for-sale mark: a large NEGATIVE value is unrealized securities losses eroding tangible capital. This is the 2023 SVB signal MINUS the held-to-maturity leg, which needs footnote fair values SF1 does not carry.",
    "The negative tail in 2022-23 is exactly the rate-shock capital erosion the metric exists to detect.")

# ------------------------------------------------------------------- reits #
add("ffo_margin", "reit_health", "FFO / revenue, GICS-gated to equity REITs.",
    "Only the D&A leg of NAREIT FFO is available; `gainOnDispositions` and `realEstateImpairment` are absent from SF1.",
    "Sparse by design. Directionally right and comparable across REITs, but not a strict NAREIT FFO -- documented rather than silent.")
add("ffo_payout", "reit_health", "Dividends / FFO -- the REIT payout ratio.",
    "The right denominator for a REIT, whose net income is depressed by depreciation on appreciating assets.",
    "Above 1 means distributing more than funds from operations.")
add("affo_margin", "reit_health", "AFFO (FFO - recurring capex) / revenue.",
    "The straight-line-rent and above/below-market lease amortisation legs the supplementals adjust for are likewise not in SF1.",
    "Stricter than FFO margin; the wedge is maintenance capex.")
add("affo_dividend_coverage", "reit_health", "AFFO / dividends -- does cash after maintenance capex cover the distribution?",
    "The REIT-specific dividend-safety test.",
    "Below 1 is the distribution-cut zone. Sparse by design (equity REITs only).")

# ------------------------------------------------------------------ energy #
add("ddna_intensity", "energy_health", "D&A / revenue, GICS-gated to Energy -- depletion intensity.",
    "`depletionDDA` is not in SF1, so total D&A stands in. For an E&P that IS overwhelmingly depletion, which is why the fallback was already the design.",
    "High means a reserve base being consumed quickly relative to the revenue it generates.")
add("ebitda_margin", "energy_health", "(Operating income + D&A) / revenue, GICS-gated to Energy.",
    "NOT named `ebitdax_margin` any more: EBITDAX adds back exploration expense so Successful-Efforts and Full-Cost filers are comparable, and `explorationExpense` is not in SF1, so nothing is added back. The old name promised a normalisation the data cannot perform.",
    "Correct for services and refiners; understated for a Successful-Efforts E&P by exactly the exploration it expensed. `ebitdax_to_ev` was DELETED for the same reason -- it reduced to `ebitda_to_ev` at r = 1.0.")

# --------------------------------------------------------------- utilities #
add("capex_to_rate_base", "utilities", "Capex / assets, GICS-gated to Utilities -- a rate-base growth proxy.",
    "`regulatoryAssets` and a standalone `goodwill` are not in SF1, so the base is NOT cleaned of either. It differs from the universal `capex_intensity` only in its denominator (assets, not revenue).",
    "A regulated utility only grows guaranteed earnings by expanding real infrastructure, so a high ratio is a structural long.")

# ---------------------------------------------------------------- research #
add("rd_capitalized_roic", "ai_capability", "ROIC with R&D treated as a 5-year intangible (Damodaran capitalisation) rather than expensed.",
    "AVAILABILITY-gated on R&D being reported, NOT GICS-gated -- capitalising R&D is meaningful for any research-intensive filer, in tech and industrials as much as biotech. The `pharma_gate` that used to sit beside it became unused when `patent_cliff` was deleted and was removed.",
    "Undoes GAAP's immediate expensing so organic innovators are comparable to serial acquirers -- the same comparison `roic_ex_intangibles` makes from the other side.")

# ------------------------------------------------------------ expectations #
add("fwd_eps_yield", "expectations", "Next-quarter consensus EPS / price.",
    "Point-in-time: a forward estimate applies only within its own quarter and the actual only after the report.",
    "The forward analogue of E/P; disagreement with the trailing version is the re-rating signal.")
add("forward_earnings_yield", "expectations", "NTM (next-twelve-months) consensus EPS / price.",
    "Built from the analyst-estimate archive with the same PIT discipline.",
    "Higher = cheaper on forward earnings.")
add("eps_expectation_growth", "expectations", "Next-quarter estimate / last actual - 1.",
    "The denominator is masked to > 0, so a loss quarter does not produce a sign-flipped growth rate.",
    "The expectations ramp; large positives are recovery stories.")
add("eps_surprise_last", "eps_beat", "Most recent quarter's EPS surprise (actual vs estimate).",
    "Realised, so it enters only after the report date.",
    "Post-earnings-announcement drift is one of the most durable anomalies; the tail is genuine beats and misses.")
add("eps_surprise_4q_avg", "eps_beat", "Average EPS surprise over the last four quarters.",
    "Smooths the single-quarter noise into a consistency measure.",
    "A persistently positive average is a management-guidance-conservatism signal.")
add("days_since_earnings", "expectations", "Calendar days since the most recent PAST earnings report, clipped to [0, 180].",
    "Leak-free: on date t it uses only report dates on or before t. NaN before a ticker's first report -- not 0, which would read as 'reported today'. The clip stops a late or skipped quarter producing an outlier.",
    "A near-zero value flags the post-earnings-announcement-drift window, which is the whole reason the feature exists; the 180 ceiling is the clip, not a real gap.")

# ------------------------------------------------------------- state flags #
add("profitable", "state", "0/1 flag: TTM net income > 0. Emitted RAW.",
    "A NaN base gives a NaN flag, never a false 0, so 'no data' is not read as 'unprofitable'.",
    "Binary. It carries the regime the masked earnings yields drop.")
add("fcf_positive", "state", "0/1 flag: TTM free cash flow > 0. Emitted RAW.", "Same NaN discipline.",
    "Binary; the companion regime for `fcf_yield`'s mask.")
add("negative_equity", "state", "0/1 flag: stockholders' equity <= 0. Emitted RAW.", "Same NaN discipline.",
    "Binary. Flags the names where `book_yield`, `returnOnEquity` and `debtToEquity` invert meaning -- post-buyback capital structures like MCD and HD, not distress.")


# --------------------------------------------------------------- the per-part registry #
#: part table -> its prose catalogue. The registry a both-directions check reads, so a new
#: documented part is one import and one entry here rather than a new checker.
#:
#: ⚠ IMPORTED AT THE BOTTOM, after every `add()` call above has run. `CATALOGUE` is populated
#: by module-level side effect, so a registry built at the top would capture an empty dict.
from scripts.cube_governance_catalogue import GOVERNANCE  # noqa: E402

CATALOGUES: dict[str, dict[str, tuple[str, str, str, str]]] = {
    "cube_part_fundamentals": CATALOGUE,
    "cube_part_governance": GOVERNANCE,
}
