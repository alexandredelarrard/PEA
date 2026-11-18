"""Cluster feature configuration: the validated `cluster` block of `configs/cluster.yml` as cluster specs, the cube column names and the size budget.

A spec is one cluster id with its feature list and `min_holders`; fund clusters take the ids 0..keep-1, every category cluster its own `id`.
Columns are `f_ic_cluster_<id>_<feature>`, in cluster-id then feature order.
The `labels` registry names every stored `cluster` id: the sentinels -2 and -1, the ids that build features, and the tracked-only ids
(a fitted fund cluster kept out of the cube, `dropped_by` an `exclude` key; a category with no feature block, `category`).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig

from src.data_aggregate.utils.manager_profile.quarter_stats import STAT_COLUMNS

logger = logging.getLogger(__name__)

FUND = "fund"
COLUMN_PREFIX = "f_ic_cluster"
#: Stored label of a row below the size floor and of an eligible row with a missing clustering feature.
BELOW_SIZE_FLOOR = -1
NOT_COMPUTABLE = -2
_SENTINELS = {NOT_COMPUTABLE: "not_computable", BELOW_SIZE_FLOOR: "below_size_floor"}
#: Feature names the cluster feature function can compute.
_CLUSTER_FEATURES = (
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


def feature_names(entries: Iterable[Any]) -> list[str]:
    """The names of the clustering features: an entry is a plain name or a `{name, transform}` mapping."""
    return [entry if isinstance(entry, str) else str(entry["name"]) for entry in entries]


def validate_cluster_config(cluster: DictConfig) -> None:
    """Raise ValueError unless the block is usable.

    No return column or unknown manager column among the clustering features, known per-cluster features, `exclude` keys among the
    clustering features, `1 <= keep <= k`, positive `min_holders`, distinct category ids outside the fund range, and a `labels` registry
    that names every id (see `_validate_labels`).
    """
    # Clustering features: no return column, only manager-details columns.
    names = feature_names(cluster.features)
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
    bad_features = [f"{kind}: {name}" for kind, block in blocks.items() for name in block.features if name not in _CLUSTER_FEATURES]
    if bad_features:
        raise ValueError(f"cluster.clusters: unknown cluster features {bad_features}; expected names from {list(_CLUSTER_FEATURES)}")
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
    _validate_labels(cluster, {str(kind): int(block.id) for kind, block in blocks.items() if kind != FUND})


def _validate_labels(cluster: DictConfig, feature_categories: dict[str, int]) -> None:
    """Raise ValueError unless `cluster.labels` names every stored id exactly once.

    Present: the sentinels -2 and -1 under their fixed names, the fund ids 0..keep-1 and each feature category under its kind. Every other id is
    tracked only: a fund cluster `dropped_by` an `exclude` key (`k - keep` of them) or a `category` that has no block in `cluster.clusters`.
    `features` is true exactly for the ids that build cube features.
    """
    entries = [dict(entry) for entry in cluster.get("labels") or []]
    if not entries:
        raise ValueError("cluster.labels is missing: name every stored cluster id there")
    ids = [int(entry["id"]) for entry in entries]
    repeated = sorted({cluster_id for cluster_id in ids if ids.count(cluster_id) > 1})
    names = [str(entry.get("name") or "") for entry in entries]
    if repeated:
        raise ValueError(f"cluster.labels: duplicate ids {repeated}")
    if "" in names or len(set(names)) != len(names):
        raise ValueError("cluster.labels: every id needs its own non-empty name")
    by_id = dict(zip(ids, entries, strict=True))

    # Sentinels, kept fund ids and feature categories are all present, the categories named after their kind.
    wrong = [f"{cluster_id} must be named {name!r}" for cluster_id, name in _SENTINELS.items() if by_id.get(cluster_id, {}).get("name") != name]
    wrong += [
        f"{cluster_id} must be named {kind!r}" for kind, cluster_id in feature_categories.items() if by_id.get(cluster_id, {}).get("name") != kind
    ]
    unnamed = [cluster_id for cluster_id in range(int(cluster.keep)) if cluster_id not in by_id]
    if wrong or unnamed:
        raise ValueError(f"cluster.labels: id {'; id '.join(wrong)}" if wrong else f"cluster.labels: no name for the fund ids {unnamed}")

    # Tracked-only ids: no feature block, and either the exclude rule that drops the fitted cluster or the category.
    built = {*range(int(cluster.keep)), *feature_categories.values()}
    tracked = [entry for entry in entries if int(entry["id"]) not in built and int(entry["id"]) not in _SENTINELS]
    in_blocks = [str(entry["category"]) for entry in entries if entry.get("category") is not None and entry["category"] in cluster.clusters]
    if in_blocks:
        raise ValueError(f"cluster.labels: category {in_blocks} has a block in cluster.clusters, so its only id is the one there")
    for entry in tracked:
        if (entry.get("dropped_by") is None) == (entry.get("category") is None):
            raise ValueError(f"cluster.labels: tracked id {entry['id']} needs exactly one of dropped_by (fund cluster) or category")
    rules = [str(entry["dropped_by"]) for entry in tracked if entry.get("dropped_by") is not None]
    if [rule for rule in rules if rule not in cluster.exclude]:
        raise ValueError(f"cluster.labels: dropped_by {sorted(set(rules) - set(cluster.exclude))} is not a cluster.exclude key")
    if len(rules) != int(cluster.k) - int(cluster.keep):
        raise ValueError(
            f"cluster.labels: {len(rules)} tracked fund ids, but k - keep = {int(cluster.k) - int(cluster.keep)} fitted clusters are dropped"
        )

    # Features are built exactly for the kept fund ids and the feature categories.
    flagged = {cluster_id for cluster_id, entry in by_id.items() if entry.get("features")}
    if flagged != built:
        raise ValueError(f"cluster.labels: features must be true exactly for the ids {sorted(built)}, got {sorted(flagged)}")


def label_names(cluster: DictConfig) -> dict[int, str]:
    """Stored id -> name for every id of the registry."""
    return {int(entry.id): str(entry.name) for entry in cluster.labels}


def fund_tracked_ids(cluster: DictConfig) -> dict[int, str]:
    """Ids of the fitted fund clusters kept out of the cube, in id order, each with the `exclude` key that drops it."""
    return {int(entry.id): str(entry.dropped_by) for entry in sorted(cluster.labels, key=lambda entry: int(entry.id)) if entry.get("dropped_by")}


def category_ids(cluster: DictConfig) -> dict[str, int]:
    """Category -> stored id: the feature categories of `cluster.clusters` and the tracked-only categories of `cluster.labels`."""
    ids = {str(kind): int(block.id) for kind, block in cluster.clusters.items() if kind != FUND}
    return ids | {str(entry.category): int(entry.id) for entry in cluster.labels if entry.get("category")}


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
