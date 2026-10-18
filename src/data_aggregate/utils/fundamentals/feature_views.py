"""Approved output views for ``cube_part_fundamentals``.

The raw characteristic is the common economic contract. Self-history is retained only for
the approved characteristics below, with two explicit peer comparisons.
"""

from __future__ import annotations

import pandas as pd

from src.data_aggregate.utils.common.panel import build_peer_relative_panel, peer_relative
from src.data_aggregate.utils.common.xs import HIST_MIN_PERIODS, HIST_WINDOW, winsorize_xs

RAW_FEATURES: frozenset[str] = frozenset(
    {
        "accruals",
        "accruals_ratio",
        "acquisition_intensity",
        "affo_dividend_coverage",
        "affo_margin",
        "altman_z",
        "aoci_to_equity",
        "asset_growth",
        "asset_turnover",
        "bank_roa",
        "beneish_m_score",
        "book_value_growth",
        "book_yield",
        "buyback_intensity",
        "capex_intensity",
        "capex_to_dep",
        "capex_to_rate_base",
        "cash_conversion_cycle",
        "cash_to_debt",
        "cash_runway_months",
        "current_ratio",
        "da_minus_capex_growth",
        "da_to_capex",
        "days_since_earnings",
        "ddna_intensity",
        "debtToEquity",
        "deferred_rev_intensity",
        "diluted_shares_growth",
        "dio",
        "dividend_coverage",
        "dividend_growth",
        "dividend_growth_5y",
        "dividend_payer",
        "dividend_payout_ratio",
        "dividend_yield",
        "dpo",
        "dpo_change",
        "dps_growth",
        "dso",
        "dso_change",
        "earnings_quality",
        "earnings_yield",
        "ebit_interest_coverage",
        "ebitda_margin",
        "ebitda_to_ev",
        "effective_tax_rate",
        "employee_growth",
        "reported_expected_eps_growth_qoq",
        "eps_surprise_4q_avg",
        "eps_surprise_last",
        "eps_yield",
        "fcf_growth",
        "fcf_margin",
        "fcf_positive",
        "fcf_to_ev",
        "fcf_yield",
        "ffo_margin",
        "ffo_payout",
        "ffo_yield",
        "fixed_cost_coverage_margin",
        "reported_expected_ttm_yield",
        "reported_expected_eps_yield",
        "gmroi",
        "grossMargins",
        "gross_margin_chg",
        "gross_profitability",
        "headcount_elasticity",
        "hyper_growth",
        "intangibles_growth",
        "intangibles_roic_drag",
        "intangibles_to_assets",
        "intangibles_to_equity",
        "interest_coverage",
        "intrinsic_yield",
        "loss_intensity",
        "margin_expansion_delta",
        "nci_income_share",
        "negative_equity",
        "net_debt_to_ebitda",
        "nwc_elasticity",
        "operatingMargins",
        "operating_leverage",
        "operating_leverage_elasticity",
        "operating_margin_5y_chg",
        "option_overhang",
        "payout_ratio",
        "pbo_to_mcap",
        "pegy",
        "pension_funded_ratio",
        "pension_overhang_leverage",
        "pension_retirement_liability",
        "piotroski_f_score",
        "profitMargins",
        "profitable",
        "q_earnings_growth",
        "q_margin_vs_ttm",
        "rd_capitalized_roic",
        "rd_intensity",
        "refinancing_risk",
        "reinvestment_rate",
        "returnOnEquity",
        "rev_growth_accel",
        "revenue_per_employee",
        "revenue_per_employee_growth",
        "roic_ex_intangibles",
        "roic_incl_intangibles",
        "rule_of_40",
        "sales_yield",
        "sbc_intensity",
        "sbc_to_buyback",
        "sbc_to_ocf",
        "sga_elasticity",
        "sga_growth",
        "sga_intensity",
        "shareholder_yield",
        "shares_growth",
        "sustainable_growth_rate",
        "y_earnings_growth_signed",
        "revenue_cagr_3y",
        "y_margin_vs_ttm",
        "y_rev_growth",
        "y_rev_growth_accel",
    }
)

RAW_FEATURES |= frozenset(
    {
        *[f"earnings_sue_lag{lag}" for lag in range(8)],
        "earnings_improvement_streak",
        "earnings_abret_3d",
        "earnings_abret_since_report",
        "profit_improvement_to_assets",
        "roa_vol_8q",
        "fcf_positive_fraction_8q",
        "ocf_yield",
    }
)

PEER_FEATURES: frozenset[str] = frozenset({"fcf_to_ev", "roic_incl_intangibles"})

HISTORY_FEATURES: frozenset[str] = frozenset(
    {
        "accruals",
        "accruals_ratio",
        "acquisition_intensity",
        "affo_dividend_coverage",
        "affo_margin",
        "aoci_to_equity",
        "asset_turnover",
        "bank_roa",
        "buyback_intensity",
        "capex_intensity",
        "capex_to_dep",
        "capex_to_rate_base",
        "cash_conversion_cycle",
        "cash_to_debt",
        "da_to_capex",
        "ddna_intensity",
        "debtToEquity",
        "deferred_rev_intensity",
        "dio",
        "dividend_coverage",
        "dividend_payout_ratio",
        "dividend_yield",
        "dpo",
        "dpo_change",
        "dso",
        "dso_change",
        "y_earnings_growth_signed",
        "earnings_quality",
        "ebit_interest_coverage",
        "ebitda_margin",
        "ebitda_to_ev",
        "effective_tax_rate",
        "fcf_growth",
        "fcf_margin",
        "fcf_to_ev",
        "fcf_yield",
        "ffo_margin",
        "ffo_payout",
        "ffo_yield",
        "fixed_cost_coverage_margin",
        "reported_expected_ttm_yield",
        "reported_expected_eps_yield",
        "gmroi",
        "grossMargins",
        "gross_profitability",
        "intangibles_roic_drag",
        "intangibles_to_assets",
        "intangibles_to_equity",
        "interest_coverage",
        "intrinsic_yield",
        "loss_intensity",
        "nci_income_share",
        "net_debt_to_ebitda",
        "nwc_elasticity",
        "operatingMargins",
        "operating_leverage_elasticity",
        "option_overhang",
        "payout_ratio",
        "pegy",
        "pension_funded_ratio",
        "pension_overhang_leverage",
        "pension_retirement_liability",
        "profitMargins",
        "rd_capitalized_roic",
        "rd_intensity",
        "refinancing_risk",
        "reinvestment_rate",
        "returnOnEquity",
        "revenue_per_employee",
        "roic_ex_intangibles",
        "roic_incl_intangibles",
        "sales_yield",
        "sbc_intensity",
        "sbc_to_buyback",
        "sbc_to_ocf",
        "sga_elasticity",
        "sga_intensity",
        "shareholder_yield",
        "y_margin_vs_ttm",
    }
)

EXCLUDED_FEATURES: frozenset[str] = frozenset({"q_rev_growth", "revenueGrowth", "earningsGrowth", "y_earnings_growth"})

FEATURE_COLUMNS: tuple[str, ...] = tuple(
    [f"f_{name}" for name in sorted(RAW_FEATURES)]
    + [f"f_{name}_vs_hist" for name in sorted(HISTORY_FEATURES)]
    + [f"f_{name}_vs_peers" for name in sorted(PEER_FEATURES)]
)


def build_fundamental_views(
    fields: dict[str, pd.DataFrame],
    peer_dict: dict,
    *,
    semantics: dict[str, str] | None = None,
    history_window: int = HIST_WINDOW,
    history_min_periods: int = HIST_MIN_PERIODS,
    history_fields: dict[str, pd.DataFrame] | None = None,
    output_since: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Emit the approved raw/history views while rejecting unreviewed characteristics."""
    unknown = set(fields) - RAW_FEATURES - EXCLUDED_FEATURES
    if unknown:
        raise KeyError(f"field(s) not in the approved fundamentals feature contract: {sorted(unknown)}")

    retained = {name: frame for name, frame in fields.items() if name in RAW_FEATURES}
    retained_semantics = {name: mode for name, mode in (semantics or {}).items() if name in retained}
    emission = {name: "raw+hist" if name in HISTORY_FEATURES else "raw" for name in retained}
    panel = build_peer_relative_panel(
        retained,
        peer_dict,
        emission=emission,
        semantics=retained_semantics,
        history_window=history_window,
        history_min_periods=history_min_periods,
        history_fields=history_fields,
        output_since=output_since,
    )
    selected_peers = PEER_FEATURES & retained.keys()
    if not selected_peers:
        return panel
    peer_columns: dict[str, pd.Series] = {}
    keys = pd.MultiIndex.from_frame(panel[["date", "ticker"]])
    for name in sorted(selected_peers):
        relative = winsorize_xs(peer_relative(retained[name], peer_dict))
        if output_since is not None:
            relative = relative.loc[relative.index >= pd.Timestamp(output_since)]
        column = relative.stack().astype("float32")
        assert isinstance(column, pd.Series)
        column.index.set_names(["date", "ticker"], inplace=True)
        peer_columns[f"f_{name}_vs_peers"] = column.reindex(keys).reset_index(drop=True)
    return pd.concat([panel, pd.DataFrame(peer_columns, index=panel.index)], axis=1)
