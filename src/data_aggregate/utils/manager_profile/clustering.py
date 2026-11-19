"""Manager clustering: which rows are fitted and labelled, which fitted clusters are kept, the saved ids and the label of each row.

Every labelled row has an id. A fund at or above the size floor takes the id of its nearest fitted cluster: 0..keep-1 for a kept cluster
(numbered by descending fit size) or a tracked id for a dropped one (numbered by the centroid value of the rule that drops it); a row of a category takes the id of its category; a row below the floor is -1; an eligible fund with a required
clustering feature missing is -2. `cluster.labels` in `configs/cluster.yml` names the ids.
A row is due when its `cluster` is null, it has a profile (`category`) and `period` plus the label delay has passed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_aggregate.utils.sec_thirteen.cluster_config import BELOW_SIZE_FLOOR, NOT_COMPUTABLE
from src.modelling.transformers.cluster import ClusterTransformer


def _passes_size(df: pd.DataFrame, filters: Mapping[str, Any]) -> pd.Series:
    """Rows with at least `min_positions` positions and `min_book_usd` of book (a null value fails)."""
    return df["n_positions"].ge(int(filters["min_positions"])) & df["book_value_usd"].ge(float(filters["min_book_usd"]))


def eligible_fund_rows(details: pd.DataFrame, filters: Mapping[str, Any]) -> pd.DataFrame:
    """The rows of the fitted category (`filters["category"]`) that pass the position and book filters."""
    return details[details["category"].eq(filters["category"]) & _passes_size(details, filters)]


def keep_clusters(transformer: ClusterTransformer, exclude: Mapping[str, Any], keep: int, tracked: Mapping[int, str]) -> dict[int, int]:
    """Fitted cluster -> stored id of every fitted cluster.

    A centroid at or above an `exclude` share drops its cluster; the others are kept and numbered 0..keep-1 by descending fit size. A dropped
    cluster takes a `tracked` id (id -> the exclude key that drops it, in id order): the clusters dropped by one key, highest centroid value
    of that key first, take that key's ids in order; a cluster over several limits goes by the first key of `exclude` it reaches.

    Raises ValueError unless exactly `keep` clusters remain and the dropped ones match the tracked ids key by key.
    """
    # Centroids back in raw feature units.
    raw = pd.DataFrame(np.array(transformer.centroids) * np.array(transformer.std) + np.array(transformer.mean), columns=transformer.features)

    # Dropped: the first exclude key whose share the centroid reaches.
    dropped_by: dict[int, str] = {}
    for feature, limit in exclude.items():
        for index in raw.index[raw[feature].ge(float(limit))]:
            dropped_by.setdefault(int(index), str(feature))

    # Kept clusters, largest fit sample first; a tie keeps the lower fitted index first.
    order = sorted(range(len(raw)), key=lambda index: (-transformer.sizes[index], index))
    kept = [index for index in order if index not in dropped_by]
    if len(kept) != keep:
        raise ValueError(
            f"the exclude rule {dict(exclude)} leaves {len(kept)} of {len(raw)} fitted clusters, {keep} wanted; "
            f"dropped {sorted(dropped_by)}; change cluster.k or cluster.exclude and refit"
        )
    mapping = {index: number for number, index in enumerate(kept)}

    # Dropped clusters: the ids of each key by descending centroid value of that key (a tie keeps the lower fitted index first).
    for feature in exclude:
        ids = [cluster_id for cluster_id, rule in tracked.items() if rule == feature]
        share = raw[feature].to_numpy()
        cluster_ids = sorted((index for index, rule in dropped_by.items() if rule == feature), key=lambda index: (-share[index], index))
        if len(cluster_ids) != len(ids):
            raise ValueError(
                f"{len(cluster_ids)} fitted clusters are dropped by {feature!r} but cluster.labels has {len(ids)} tracked ids for it ({ids}); "
                "fix the dropped_by entries of cluster.labels or the exclude limits and refit"
            )
        mapping.update(zip(cluster_ids, ids, strict=True))
    return mapping


def write_centroids(transformer: ClusterTransformer, mapping: dict[int, int], keep: int, path: Path) -> None:
    """Save the fit to `path` with, per fitted cluster, its `kept` flag, kept `number` and `tracked` id (the one that does not apply is null); the fit's own keys and hash are unchanged."""
    kept = [mapping[index] < keep for index in range(transformer.k)]
    extra = {
        "kept": kept,
        "number": [mapping[index] if flag else None for index, flag in enumerate(kept)],
        "tracked": [None if flag else mapping[index] for index, flag in enumerate(kept)],
    }
    transformer.save(path, extra)


def read_mapping(path: Path) -> dict[int, int]:
    """Fitted cluster -> stored id of every fitted cluster, from the file `write_centroids` wrote; raises ValueError when it records none."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "number" not in payload or "tracked" not in payload:
        raise ValueError(f"{path} records no cluster ids; run build-cluster with cluster.fit: true")
    pairs = zip(payload["number"], payload["tracked"], strict=True)
    return {index: int(number if number is not None else tracked) for index, (number, tracked) in enumerate(pairs)}


def due_rows(details: pd.DataFrame, delay_days: int, today: pd.Timestamp, relabel: bool) -> pd.DataFrame:
    """The profiled rows whose label is due on `today`: unlabelled ones, or every one when `relabel`."""
    old_enough = details["period"] + pd.Timedelta(days=delay_days) <= today
    unlabelled = pd.Series(True, index=details.index) if relabel else details["cluster"].isna()
    return details[old_enough & unlabelled & details["category"].notna()]


def assemble_labels(
    due: pd.DataFrame, raw_labels: pd.Series, mapping: dict[int, int], categories: Mapping[str, int], filters: Mapping[str, Any]
) -> pd.DataFrame:
    """`cik`, `period` and `cluster` of the due rows.

    A row below the size floor (`filters`) is -1, whatever its category. At or above it a fund takes the id of its nearest fitted cluster from
    `raw_labels` (nullable, indexed like `due`) through `mapping`, or -2 when that label is null (a required feature is missing); any other
    row takes the id of its category in `categories`. Raises ValueError for a category in neither `categories` nor the fitted one.
    """
    unknown = sorted(set(due["category"]) - set(categories) - {filters["category"]})
    if unknown:
        raise ValueError(f"categories {unknown} have no cluster id; add them to cluster.labels")

    # Fund: the nearest fitted cluster's id, -2 when there is none; the other categories take their own id.
    raw = pd.Series(raw_labels.reindex(due.index).to_numpy(dtype=float, na_value=np.nan), index=due.index)
    fund = raw.map(pd.Series(mapping, dtype=float)).fillna(NOT_COMPUTABLE)
    by_category = due["category"].map(categories).astype(float)
    label = fund.where(due["category"].eq(filters["category"]), by_category)

    # The size floor decides first.
    label = label.where(_passes_size(due, filters), BELOW_SIZE_FLOOR)
    return due[["cik", "period"]].assign(cluster=label.astype(int))
