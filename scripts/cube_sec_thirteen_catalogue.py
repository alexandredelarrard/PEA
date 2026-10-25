"""Static prose catalogue for `cube_part_sec_thirteen` characteristics."""

from __future__ import annotations

from src.data_aggregate.utils.sec_thirteen.superinvestor_features import EMISSION as SUPERINVESTOR_EMISSION

SEC_THIRTEEN: dict[str, tuple[str, str, str, str]] = {}


def _add_group(family: str, names: tuple[str, ...], source: str) -> None:
    for name in names:
        label = name.removeprefix("ic_").replace("_", " ")
        SEC_THIRTEEN[name] = (
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
    tuple(SUPERINVESTOR_EMISSION),
    "fixed raw-CIK13F books;90-calendar-day first-public carry; genuine matched quarters for changes",
)
# Every characteristic emits its raw economic representation.
CATALOGUE = {f"f_{name}": entry for name, entry in SEC_THIRTEEN.items()}
