"""Manager clustering: which rows are fitted and labelled, which fitted clusters are kept, the saved kept flags and the label of each row.

A label is a fund cluster 0..keep-1 (a fitted cluster numbered by descending fit size), a category cluster id, or -1 (assigned, in no cluster).
A row is due when its `cluster` is null, it has a profile (`category`) and `period` plus the label delay has passed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.modelling.transformers.cluster import ClusterTransformer

_NO_CLUSTER = -1


def _passes_size(df: pd.DataFrame, filters: Mapping[str, Any]) -> pd.Series:
    """Rows with at least `min_positions` positions and `min_book_usd` of book (a null value fails)."""
    return df["n_positions"].ge(int(filters["min_positions"])) & df["book_value_usd"].ge(float(filters["min_book_usd"]))


def eligible_fund_rows(details: pd.DataFrame, filters: Mapping[str, Any]) -> pd.DataFrame:
    """The rows of the fitted category (`filters["category"]`) that pass the position and book filters."""
    return details[details["category"].eq(filters["category"]) & _passes_size(details, filters)]


def keep_clusters(transformer: ClusterTransformer, exclude: Mapping[str, Any], keep: int) -> dict[int, int]:
    """Fitted cluster -> kept number: the clusters whose centroid reaches an `exclude` share are dropped, the rest numbered by descending fit size.

    Raises ValueError unless exactly `keep` clusters remain.
    """
    # Centroids back in raw feature units.
    raw = pd.DataFrame(np.array(transformer.centroids) * np.array(transformer.std) + np.array(transformer.mean), columns=transformer.features)

    # Dropped: any excluded feature at or above its share.
    dropped = pd.Series(False, index=raw.index)
    for feature, limit in exclude.items():
        dropped |= raw[feature].ge(float(limit))

    # Kept clusters, largest fit sample first; a tie keeps the lower fitted index first.
    order = sorted(range(len(raw)), key=lambda index: (-transformer.sizes[index], index))
    kept = [index for index in order if not dropped[index]]
    if len(kept) != keep:
        raise ValueError(
            f"the exclude rule {dict(exclude)} leaves {len(kept)} of {len(raw)} fitted clusters, {keep} wanted; "
            f"dropped {raw.index[dropped].tolist()}; change cluster.k or cluster.exclude and refit"
        )
    return {index: number for number, index in enumerate(kept)}


def write_centroids(transformer: ClusterTransformer, mapping: dict[int, int], path: Path) -> None:
    """Save the fit to `path` with each fitted cluster's `kept` flag and kept `number` (null when dropped); the fit's own keys and hash are unchanged."""
    kept = {"kept": [index in mapping for index in range(transformer.k)], "number": [mapping.get(index) for index in range(transformer.k)]}
    transformer.save(path, kept)


def read_mapping(path: Path) -> dict[int, int]:
    """Fitted cluster -> kept number from the file `write_centroids` wrote; raises ValueError when it records none."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "number" not in payload:
        raise ValueError(f"{path} records no kept clusters; run build-cluster with cluster.fit: true")
    return {index: int(number) for index, number in enumerate(payload["number"]) if number is not None}


def due_rows(details: pd.DataFrame, delay_days: int, today: pd.Timestamp, relabel: bool) -> pd.DataFrame:
    """The profiled rows whose label is due on `today`: unlabelled ones, or every one when `relabel`."""
    old_enough = details["period"] + pd.Timedelta(days=delay_days) <= today
    unlabelled = pd.Series(True, index=details.index) if relabel else details["cluster"].isna()
    return details[old_enough & unlabelled & details["category"].notna()]


def assemble_labels(
    due: pd.DataFrame, raw_labels: pd.Series, mapping: dict[int, int], categories: Mapping[str, int], filters: Mapping[str, Any]
) -> pd.DataFrame:
    """`cik`, `period` and `cluster` of the due rows.

    A fund takes its kept number from `raw_labels` (nearest fitted cluster, nullable, indexed like `due`) through `mapping`; a row of a
    category in `categories` takes that cluster id; both need the size filters. Every other row, a dropped cluster and a null feature give -1.
    """
    # Fund: the nearest fitted cluster renumbered, null for a dropped cluster or a missing label.
    raw = pd.Series(raw_labels.reindex(due.index).to_numpy(dtype=float, na_value=np.nan), index=due.index)
    fund = raw.map(pd.Series(mapping, dtype=float))

    # Category rows take their id; funds win on the fitted category; the size filters apply to both.
    by_category = due["category"].map(categories).astype(float)
    label = fund.where(due["category"].eq(filters["category"]), by_category)
    label = label.where(_passes_size(due, filters)).fillna(_NO_CLUSTER)
    return due[["cik", "period"]].assign(cluster=label.astype(int))
