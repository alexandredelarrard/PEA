"""Static prose catalogue for `cube_part_sec_thirteen` characteristics: the per-cluster 13F features of `configs/cluster.yml`."""

from __future__ import annotations

from pathlib import Path

from src.data_aggregate.utils.sec_thirteen.cluster_config import cluster_specs
from src.utils.config import read_config

#: feature -> (what it measures, how to read the tail); the family is the cluster kind.
_FEATURES: dict[str, tuple[str, str]] = {
    "net_breadth": (
        "net breadth of the cluster: (new holders - exiting holders) / (holders now + exiting holders)",
        "-1 = every matched holder left, 1 = only new holders; a thin cluster is null below its min_holders",
    ),
    "new_position_rate": (
        "share of the cluster's matched managers that opened the position this quarter",
        "bounded 0 to 1; a high tail is a crowd arriving, not a size of the bet",
    ),
    "net_share_flow": (
        "net change in shares held by the cluster's matched managers (split-adjusted) over shares outstanding",
        "signed fraction of the float; the tails are the cluster buying or selling a large part of the company",
    ),
    "top10_chg": (
        "change in the share of matched managers holding the stock among their ten largest positions",
        "signed, bounded -1 to 1; high = the stock moved into the cluster's core books",
    ),
    "rank_gap": (
        "percentile rank of the cluster's aggregate weight in the stock minus the percentile rank of its market cap",
        "signed, bounded -1 to 1; high = the cluster is overweight against the stock's size",
    ),
    "rank_gap_chg": (
        "change of the rank gap against the same matched managers one quarter earlier",
        "signed, bounded -2 to 2; the difference of two rank gaps",
    ),
    "stickiness": (
        "share of the cluster's current holders that already held the stock four quarters earlier",
        "bounded 0 to 1; low = a recently built holder base",
    ),
    "cap_proximity": (
        "share of the cluster's holders for which the stock is at least 4% of their book",
        "bounded 0 to 1; high = the stock is a core position for the cluster",
    ),
    "option_intensity": (
        "call and put value over common plus option value in the cluster's books",
        "bounded 0 to 1; high = derivatives dominate the cluster's exposure to the stock",
    ),
    "flow_x_vol": (
        "net share flow times (percentile rank of 250-day volatility minus 0.5)",
        "signed, bounded -0.5 to 0.5; flow weighted toward volatile names",
    ),
}

SEC_THIRTEEN: dict[str, tuple[str, str, str, str]] = {}
for _spec in cluster_specs(read_config(str(Path(__file__).resolve().parents[2] / "configs")).cluster):
    for _feature in _spec.features:
        _what, _tail = _FEATURES[_feature]
        SEC_THIRTEEN[f"ic_cluster_{_spec.cluster_id}_{_feature}"] = (
            f"cluster_13f_{_spec.kind}",
            _what,
            "Point-in-time per-cluster 13F feature: one publication per report quarter from originals filed by the 45-day deadline, "
            "members fixed by the previous quarter's cluster label; null below the cluster's min_holders, never zero-filled.",
            _tail,
        )
# Every characteristic emits its raw economic representation.
CATALOGUE = {f"f_{name}": entry for name, entry in SEC_THIRTEEN.items()}
