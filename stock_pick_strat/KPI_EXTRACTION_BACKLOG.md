# KPI extraction backlog — inputs not yet available

The valuation-engine KPI audit (2026-07) implemented every KPI whose inputs are in the
current `fundamentals_history` extraction. The KPIs below are **blocked on inputs that are
not in standard financial-statement XBRL** (operational, clinical, forward, or free-text
MD&A data), or on a modelling-side assembly. Revisit these when extending SEC / alt-data
extraction.

| # | KPI | Missing input(s) | Where it would come from | Current fallback |
|---|-----|------------------|--------------------------|------------------|
| 1 | **Organic revenue growth** | Inorganic (acquired) revenue contribution per period | 10-K/10-Q MD&A free text ("acquisitions contributed $X") — not XBRL-tagged | `revenueGrowth` / `y_rev_growth` (total); `acquisition_intensity` flags inorganic activity |
| 14 | **Economic Profit / EVA** | **WACC** = cost of equity (rf + β·ERP) + cost of debt, capital-weighted | Modelling assembly: wire `betas` in + add rf/ERP config (NOT an extraction gap) | `roic` shipped; EVA = InvestedCapital·(ROIC − WACC) once WACC assembled |
| 21 | **NIM (exact)** | "Average earning assets" subset (loans + securities ex non-earning) | Bank supplements / call reports — not a clean XBRL tag | `net_interest_margin` uses **total assets** as the denominator |
| 23 | **CET1 ratio** | Broader CET1/Tier-1 tagging | Only ~6 banks XBRL-tag `Tier1RiskBasedCapitalToRiskWeightedAssets` today | `tier1_capital_ratio` (sparse; falls back to CET1 in the extractor) |
| 24 | **Combined ratio (exact)** | Reinsurance recoveries, ceding commissions, **gross** policy-acquisition costs | Insurance MD&A / schedules — not tagged | `loss_ratio` = gross claims/premiums; `expense_ratio` = (SG&A + DAC amort)/premiums |
| 31 | **Production cash margin / BOE** | BOE produced, lifting costs, production taxes, transportation | E&P operational tables in the 10-K (volumes + per-unit costs) — not financial XBRL | `ebitdax_margin`, `property_overvaluation_cushion` cover E&P valuation instead |
| 33 | **R&D pipeline efficiency** | Phase II/III trial counts & success rates | ClinicalTrials.gov / FDA (external clinical data) | `rd_capitalized_roic`, `patent_cliff` cover R&D economics instead |
| 34 | **Same-store sales (SSS/LFL)** | Comparable-store sales growth | Retail MD&A / press releases (free text) — not XBRL | `gmroi`, `asset_turnover`, `y_rev_growth` cover retail efficiency instead |

Notes:
- **Altman Z** is implemented as the listed-firm **market-value** variant (market value of equity
  in the 0.6·(MktCap/Total Liabilities) term); the private-firm "modified" variant would swap in
  book equity. No missing input — just a variant choice.
- Everything else in Parts 1–3 of the spec is now computed with the refined definitions
  (see `fundamental_features.py`, `sector_features.py`).

---

## Business-quality factors (2026-07) — remaining gaps

The #2 D&A realism, #5 forensic, #3 M&A digestion, #1 core/adjusted earnings and #4
AI-leverage factors are implemented in `fundamental_features.py` / `employee_features.py`
and wired into `comp_accounting_quality`, `comp_ma_digestion`, `comp_ai_leverage`
(build_cube.yml). These inputs remain missing or partial:

| Theme | Missing / partial input | Where it would come from | Current state |
|---|-----|------------------|------------------|
| #1 | **Non-recurring REVENUE** isolation (one-off licenses, bulk property sales, settlement income in revenue) | segment / MD&A text — one-offs sit *below* the revenue line in XBRL | `special_items` normalizes **profit**; a clean non-recurring-revenue % is not extractable (organic/inorganic needs sparse pro-forma acquisition-revenue tags) |
| #1 | **unusualItems** very sparse | `UnusualOrInfrequentItemNetGainLoss` tagged by only ~4/500 filers | pool = impairment + restructuring + litigation + disposals + bargain purchase + disc-ops |
| #3 | **Employees table empty** in the current DB | run `fetch_employees` | `headcount_elasticity` + `revenue_per_employee` (and the labor leg of AI score) populate after the employee fetch |
| #4 | **IT maturity** weak — capitalized software tagged by only ~15% of filers; no IT-spend disclosure | alt-data: tech job postings, tech-stack telemetry (BuiltWith / Revelio) | `capitalized_software_intensity` (sparse) + `rd_intensity` + SG&A/labor opportunity, sector-neutral |
| #4 | **Governance tech-leadership** flag (CTO/CIO/CDO, board technology committee) | add to `def14a_llm` extraction | not built |
| #5 | **Supplier finance / reverse factoring** (Greensill-style hidden debt) | debt/commitments footnote text (ASU 2022-04) — not XBRL | `dpo` + `dpo_change` are the indirect proxy |
| #5 | **Purchase obligations / take-or-pay / guarantees / VIEs** | contractual-obligations & commitments footnotes (text) | LLM footnote extraction (not built) |
| #5 | **Pension** partial | `pensionDeficit` covers filers with a recognized noncurrent DB deficit (~30%); overfunded / OPEB-only not captured | folded into `net_debt_incl_offbs_to_ebitda` when present |
| #14 | **WACC / EVA** | cost of equity (β·ERP) + cost of debt assembly | modelling-side (betas exist) | `roic_incl_goodwill` / `roic_ex_goodwill` shipped |

**Operational note — a re-fetch is required.** The new SEC tags (`goodwillImpairment`,
`gainOnSaleGeneric`, `litigationExpense`, `discontinuedOps`, `unusualItems`,
`bargainPurchaseGain`, `capitalizedSoftware`, `pensionDeficit`) are added to
`fetch_fundamentals` but populate the cube only after the next `fetch_fundamentals`
run **+ cube rebuild**. Until then the widened core-earnings pool, off-BS pension
leverage, goodwill-impairment digestion, and AI capitalized-software features degrade
gracefully to the existing-tag versions (guarded — no crash, just narrower inputs).
