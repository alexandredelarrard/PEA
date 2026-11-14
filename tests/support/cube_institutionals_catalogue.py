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
        "ic_insider_buy_value_mcap_120d",
        "ic_insider_buy_value_mcap_180d",
        "ic_insider_c_suite_buy_mcap_180d",
        "ic_insider_management_buy_mcap_180d",
        "ic_insider_ten_pct_buy_mcap_180d",
        "ic_insider_distinct_buyers_120d",
        "ic_insider_distinct_sellers_120d",
        "ic_insider_buyer_breadth_120d",
        "ic_insider_net_buy_ratio_60d",
        "ic_insider_net_buy_ratio_120d",
        "ic_insider_net_buy_ratio_180d",
    ),
    "Forms 3/4/5 filing-date history",
)
for _name in (
    "ic_insider_purchase_pct_prior_60d",
    "ic_insider_purchase_pct_prior_120d",
    "ic_insider_purchase_pct_prior_180d",
    "ic_insider_c_suite_purchase_pct_prior_180d",
    "ic_insider_management_purchase_pct_prior_180d",
):
    INSTITUTIONALS[_name] = (
        "insider",
        "Trailing-window net open-market P minus S shares of the supported direct common-stock accounts that traded, divided by "
        "those accounts' pooled holdings just before their first window trade; all insiders or one population.",
        "Scales disclosed insider buying and selling by the holdings supporting those trades; each lot keeps its "
        "original disclosure expiry and public corrections apply prospectively.",
        "Positive means net accumulation and negative means net reduction (never below -1). Pooling requires compatible share "
        "classes and units; every trading account must be supported, so unsupported balances abstain.",
    )
_add_group(
    "beneficial_ownership",
    (
        "ic_act_13d_count_126d_z",
        "ic_bo_13g_count_126d_z",
    ),
    "Schedule 13D (active) and 13G (passive) filing counts over the trailing 126 trading days, standardised across the universe each date and clipped at +/-5; filer identity is not used",
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
