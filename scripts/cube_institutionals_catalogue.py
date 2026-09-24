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
        "ic_inst_flow_to_mcap",
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
        "ic_super_flow_to_mcap",
        "ic_super_new_top10",
        "ic_super_rank_jump",
        "ic_super_initiations",
        "ic_super_full_exits",
        "ic_super_exit_after_top10",
    ),
    "point-in-time selected-manager 13F books",
)
_add_group(
    "insider",
    (
        "ic_insider_buy_value_mcap_60d",
        "ic_insider_buy_value_mcap_180d",
        "ic_insider_buy_shares_so_180d",
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
        "ic_act_campaign_age_days",
        "ic_act_repeat_activist",
        "ic_act_purpose_board",
        "ic_act_purpose_strategic",
        "ic_bo_holder_count",
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
        "ic_shortvol_ratio_z252",
        "ic_shortvol_acceleration",
        "ic_shortvol_turnover_20d",
        "ic_shortvol_high_x_weak_price",
        "ic_shortvol_high_x_strong_price",
        "ic_shortvol_market_coverage",
        "ic_ftd_pct_so",
        "ic_ftd_to_adv20",
        "ic_ftd_z252",
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
        "ic_xs_bullish_family_ratio",
        "ic_xs_bullish_available_family_count",
        "ic_xs_bullish_actor_count",
        "ic_xs_bearish_family_ratio",
        "ic_xs_bearish_available_family_count",
        "ic_xs_conflict_ratio",
    ),
    "availability-aware agreement across independent source families",
)

INSTITUTIONALS["ic_xs_bullish_family_ratio"] = (
    "cross_source",
    "Share of available bullish families above their same-date 80th percentile.",
    "The numerator and denominator change together; the ratio is emitted only with at least three available families.",
    "Bounded [0, 1]; compare only dates with the persisted availability count.",
)
INSTITUTIONALS["ic_xs_bearish_family_ratio"] = (
    "cross_source",
    "Share of available bearish families above their same-date 80th percentile.",
    "Uses insider net selling, elite full exits and price-confirmed short flow, all jointly available from 2019.",
    "Bounded [0, 1]; a zero is measured only when all three families are observable.",
)
INSTITUTIONALS["ic_xs_conflict_ratio"] = (
    "cross_source",
    "The weaker of the bullish and bearish normalized family ratios.",
    "Defined only when both directional ratios meet the three-family evidence floor.",
    "High values mean independent bullish and bearish evidence coexist; NaN means insufficient evidence.",
)
for direction in ("bullish", "bearish"):
    name = f"ic_xs_{direction}_available_family_count"
    INSTITUTIONALS[name] = (
        "cross_source_control",
        f"Number of observable {direction} source families for this ticker-date.",
        "Persisted for audit and denominator reconciliation, not as a normalized alpha leg.",
        "Integer-valued; the corresponding ratio is unavailable below three.",
    )
