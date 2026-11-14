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
for _name in ("ic_insider_conviction_60d", "ic_insider_conviction_120d"):
    INSTITUTIONALS[_name] = (
        "insider",
        "Largest account increase: the maximum, over the direct common-stock accounts (owner x class) that traded in the trailing "
        "window, of net open-market P minus S shares over that account's own balance when the window opened.",
        "Captures one conviction buyer instead of averaging them with sellers; accounts without a positive prior balance are left out.",
        "Positive means some account raised its own stake and negative means every qualifying account reduced; the ratio is bounded "
        "at the declared guard.",
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
        "ic_shortvol_ratio_20d",
        "ic_shortvol_ratio_60d",
        "ic_shortvol_ratio_120d",
        "ic_shortvol_ratio_180d",
        "ic_shortvol_acceleration_60d",
        "ic_shortvol_acceleration_120d",
        "ic_shortvol_turnover_20d",
        "ic_shortvol_high_x_weak_price",
        "ic_shortvol_high_x_strong_price",
        "ic_ftd_to_adv20",
        "ic_ftd_persistence_30d",
    ),
    "lagged FINRA RegSHO and SEC fails-to-deliver history",
)
# Every characteristic emits its raw economic representation.
CATALOGUE = {f"f_{name}": entry for name, entry in INSTITUTIONALS.items()}
