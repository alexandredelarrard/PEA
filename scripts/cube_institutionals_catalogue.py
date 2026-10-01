"""Static prose catalogue for `cube_part_institutionals` characteristics."""

from __future__ import annotations

INSTITUTIONALS: dict[str, tuple[str, str, str, str]] = {}


def _add_group(family: str, names: tuple[str, ...], source: str) -> None:
    for name in names:
        label = name.removeprefix("ic_").replace("_", " ")
        INSTITUTIONALS[name] = (
            family,
            label,
            f"Point-in-time {source}; availability is distinct from a missing constructed value.",
            "Interpret the tail within the feature's declared source support.",
        )


_add_group(
    "broad_13f",
    (
        "ic_inst_holders",
        "ic_inst_breadth_chg",
        "ic_inst_shares_chg",
        "ic_inst_new_buyer_ratio",
        "ic_inst_exit_ratio",
        "ic_inst_cluster_buying",
        "ic_inst_concentration",
        "ic_inst_net_options_ratio",
        "ic_inst_ownership_pct",
        "ic_inst_value_to_mcap",
    ),
    "all-filer 13F holdings",
)
_add_group(
    "elite_13f",
    (
        "ic_super_holders",
        "ic_super_breadth_chg",
        "ic_super_conviction_weight",
        "ic_super_max_conviction",
        "ic_super_conviction_chg",
        "ic_super_conviction_weight_yoy",
        "ic_super_holders_yoy",
        "ic_super_top10_holders",
        "ic_super_quarters_held",
        "ic_super_sp500_share",
        "ic_super_selection_score",
        "ic_super_shares_chg",
        "ic_super_new_top10",
        "ic_super_rank_jump",
        "ic_super_initiations",
        "ic_super_full_exits",
    ),
    "point-in-time selected-manager 13F books",
)
_add_group(
    "insider",
    (
        "ic_insider_buy_value_mcap_60d",
        "ic_insider_buy_value_mcap_180d",
        "ic_insider_distinct_buyers_120d",
        "ic_insider_cluster_buy_120d",
        "ic_insider_ceo_buy_mcap_180d",
        "ic_insider_cfo_buy_mcap_180d",
        "ic_insider_director_buy_mcap_180d",
        "ic_insider_purchase_pct_prior",
        "ic_insider_owner_surprise_120d",
        "ic_insider_net_buy_ratio_180d",
        "ic_insider_discretionary_sell_mcap_60d",
        "ic_insider_planned_sell_mcap_60d",
    ),
    "Forms 3/4/5 filing-date history",
)
_add_group(
    "beneficial_ownership",
    (
        "ic_act_initial_13d",
        "ic_act_amendment_intensity",
        "ic_act_repeat_activist",
        "ic_bo_new_holder",
        "ic_bo_escalation_13g_to_13d",
        "ic_bo_de_escalation_13d_to_13g",
    ),
    "Schedule 13D/13G event history; ownership numerics are intentionally excluded",
)
_add_group(
    "short_flow",
    (
        "ic_shortvol_ratio_5d",
        "ic_shortvol_ratio_20d",
        "ic_shortvol_ratio_60d",
        "ic_shortvol_acceleration",
        "ic_shortvol_turnover_20d",
        "ic_shortvol_high_x_weak_price",
        "ic_shortvol_high_x_strong_price",
        "ic_shortvol_market_coverage",
        "ic_ftd_to_adv20",
        "ic_ftd_persistence_30d",
    ),
    "lagged FINRA RegSHO and SEC fails-to-deliver history",
)
_add_group(
    "price_conditioning",
    (
        "ic_sig_super_age_days",
        "ic_sig_insider_age_days",
        "ic_sig_act_age_days",
        "ic_sig_super_ret_since",
        "ic_sig_insider_ret_since",
        "ic_sig_act_ret_since",
        "ic_sig_super_resid_ret_since",
        "ic_sig_insider_resid_ret_since",
        "ic_sig_act_resid_ret_since",
        "ic_sig_super_vol_scaled_move",
        "ic_sig_insider_vol_scaled_move",
        "ic_sig_act_vol_scaled_move",
        "ic_sig_insider_price_vs_buy",
        "ic_sig_insider_max_dd_since_buy",
        "ic_sig_insider_max_runup_since_buy",
    ),
    "the price path following public institutional events",
)
_add_group(
    "cross_source",
    (
        "ic_xs_bullish_available_family_count",
        "ic_xs_bearish_available_family_count",
    ),
    "raw availability support across independent source families",
)
for direction in ("bullish", "bearish"):
    name = f"ic_xs_{direction}_available_family_count"
    INSTITUTIONALS[name] = (
        "cross_source_control",
        f"Number of observable {direction} source families for this ticker-date.",
        "Persisted as raw support metadata for audit and denominator reconciliation.",
        "Integer-valued; zero means every declared family is provably unavailable, not missing.",
    )

# `src.validate.checks.catalogue` compares this standalone catalogue with persisted columns.
# Every accepted characteristic now emits exactly its raw economic representation.
CATALOGUE = {f"f_{name}": entry for name, entry in INSTITUTIONALS.items()}
