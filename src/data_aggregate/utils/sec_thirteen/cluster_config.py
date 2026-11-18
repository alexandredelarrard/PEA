"""Cluster feature configuration: the validated `cluster` block of `configs/cluster.yml` as cluster specs, the cube column names and the size budget.

A spec is one cluster id with its feature list and `min_holders`; fund clusters take the ids 0..keep-1, every category cluster its own `id`.
Columns are `f_ic_cluster_<id>_<feature>`, in cluster-id then feature order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from omegaconf import DictConfig

from src.data_aggregate.utils.manager_profile.quarter_stats import STAT_COLUMNS

logger = logging.getLogger(__name__)

FUND = "fund"
COLUMN_PREFIX = "f_ic_cluster"
#: Feature names the cluster feature function can compute.
CLUSTER_FEATURES = (
    "net_breadth",
    "new_position_rate",
    "net_share_flow",
    "top10_chg",
    "rank_gap",
    "rank_gap_chg",
    "stickiness",
    "cap_proximity",
    "option_intensity",
    "flow_x_vol",
)


@dataclass(frozen=True)
class ClusterSpec:
    """One cluster of the cube: its kind (`fund`, `bank`, ...), id, ordered feature names and the managers needed for a value."""

    kind: str
    cluster_id: int
    features: tuple[str, ...]
    min_holders: int


def validate_cluster_config(cluster: DictConfig) -> None:
    """Raise ValueError unless the block is usable.

    No return column or unknown manager column among the clustering features, known per-cluster features, `exclude` keys among the
    clustering features, `1 <= keep <= k`, positive `min_holders`, and distinct category ids outside the fund range.
    """
    # Clustering features: names (plain or `{name, transform}`), no return column, only manager-details columns.
    names = [entry if isinstance(entry, str) else str(entry.name) for entry in cluster.features]
    returns = [name for name in names if "return" in name.lower()]
    if returns:
        raise ValueError(f"cluster.features: a return column cannot be a clustering feature: {returns}")
    unknown = [name for name in names if name not in STAT_COLUMNS]
    if unknown:
        raise ValueError(f"cluster.features: unknown manager columns {unknown}")
    outside = [name for name in cluster.exclude if name not in names]
    if outside:
        raise ValueError(f"cluster.exclude: keys {outside} are not clustering features")
    if not 1 <= int(cluster.keep) <= int(cluster.k):
        raise ValueError(f"cluster.keep must be between 1 and k={cluster.k}, got {cluster.keep}")

    # Per-cluster blocks: a fund block, known features, positive min_holders.
    blocks = dict(cluster.clusters)
    if FUND not in blocks:
        raise ValueError(f"cluster.clusters needs a {FUND!r} block")
    bad_features = [f"{kind}: {name}" for kind, block in blocks.items() for name in block.features if name not in CLUSTER_FEATURES]
    if bad_features:
        raise ValueError(f"cluster.clusters: unknown cluster features {bad_features}; expected names from {list(CLUSTER_FEATURES)}")
    low = [kind for kind, block in blocks.items() if int(block.min_holders) < 1]
    if low:
        raise ValueError(f"cluster.clusters: min_holders must be at least 1 for {low}")

    # Ids: the fund range 0..keep-1 is reserved, every category id is its own.
    category_ids = [int(block.get("id", -1)) for kind, block in blocks.items() if kind != FUND]
    taken = [*range(int(cluster.keep)), *category_ids]
    if min(category_ids, default=int(cluster.keep)) < int(cluster.keep) or len(set(taken)) != len(taken):
        raise ValueError(
            f"cluster.clusters: category id values {category_ids} must be distinct and at least keep={cluster.keep} (fund ids 0..keep-1)"
        )


def cluster_specs(cluster: DictConfig) -> list[ClusterSpec]:
    """The validated block as specs ordered by cluster id: `keep` fund clusters, then each category cluster."""
    validate_cluster_config(cluster)
    specs: list[ClusterSpec] = []
    for kind, block in cluster.clusters.items():
        ids = range(int(cluster.keep)) if kind == FUND else [int(block.id)]
        specs += [ClusterSpec(str(kind), cluster_id, tuple(block.features), int(block.min_holders)) for cluster_id in ids]
    return sorted(specs, key=lambda spec: spec.cluster_id)


def column_names(specs: list[ClusterSpec]) -> list[str]:
    """The cube columns `f_ic_cluster_<id>_<feature>` of the specs."""
    return [f"{COLUMN_PREFIX}_{spec.cluster_id}_{name}" for spec in specs for name in spec.features]


def column_budget(specs: list[ClusterSpec], cluster: DictConfig) -> int:
    """The number of cube columns of the specs; logs one WARNING above `max_clusters` clusters or `max_features_per_cluster` features in one."""
    n_columns = sum(len(spec.features) for spec in specs)
    widest = max((len(spec.features) for spec in specs), default=0)
    if len(specs) > int(cluster.max_clusters) or widest > int(cluster.max_features_per_cluster):
        logger.warning(
            "cluster features: %d clusters (limit %d) with up to %d features in one (limit %d) give %d cube columns; "
            "more columns than planned grow cube_part_sec_thirteen and its build memory",
            len(specs),
            int(cluster.max_clusters),
            widest,
            int(cluster.max_features_per_cluster),
            n_columns,
        )
    return n_columns
