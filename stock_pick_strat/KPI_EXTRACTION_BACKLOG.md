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
| 11 | **PEGY (projected)** | Forward/analyst consensus EPS growth | `analyst_estimates_history` (accruing, sparse) | `pegy` uses **TTM realized** EPS growth as the growth term |
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
