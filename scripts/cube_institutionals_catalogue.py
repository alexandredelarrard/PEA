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
INSTITUTIONALS["ic_insider_management_position_change_60d"] = (
    "insider",
    "Trailing 60-calendar-day net open-market P/S shares of supported direct CEO/CFO/COO common-stock accounts "
    "divided by pooled starting holdings, counted once per active owner/share-class account.",
    "Scales disclosed management buying and selling by the holdings supporting those trades; each lot keeps its "
    "original disclosure expiry and public corrections apply prospectively.",
    "Positive means net accumulation and negative means net reduction. Pooling requires compatible share classes "
    "and units; unsupported balances and uncertain shares abstain, while corroborated complete quiet windows may be zero.",
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
# Every characteristic emits its raw economic representation.
CATALOGUE = {f"f_{name}": entry for name, entry in INSTITUTIONALS.items()}
