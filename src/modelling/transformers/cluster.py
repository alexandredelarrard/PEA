"""
cluster.py  (src/modelling/transformers/cluster.py)
---------------------------------------------------
`ClusterTransformer`: a standalone, data-agnostic k-means. `fit(df)` takes the configured features (optionally
turned into a percentile rank within each `group_by` group), draws at most `max_rows_per_group` complete rows per
group with the seed, z-scores every feature (equal weight) and fits k-means++; `transform(df)` returns the nearest
centroid per row (null when a feature is null) without refitting. `save` / `load` round-trip the fit to JSON
(centroids, standardisation, feature specs, fit-sample sizes and a `version` hash that `load` verifies).
It imports neither the model families nor the context.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

RANK = "rank"


def _prepare(df: pd.DataFrame, features: list[str], ranked: list[str], group_by: str | None) -> pd.DataFrame:
    """The float feature matrix of `df`, the `ranked` columns replaced by their percentile rank within each `group_by` group."""
    # Columns the transform needs.
    needed = [*features, *([group_by] if group_by else [])]
    missing = [name for name in needed if name not in df.columns]
    if missing:
        raise ValueError(f"cluster input is missing columns {missing}")

    # Percentile rank inside the group (the whole frame when there is no group).
    matrix = df[features].astype(float)
    if ranked:
        groups = df[group_by].to_numpy() if group_by else np.zeros(len(df), dtype=int)
        matrix[ranked] = matrix[ranked].groupby(groups).rank(pct=True).to_numpy()
    return matrix


def _digest(specs: list[dict[str, str | None]], group_by: str | None, centroids: list[list[float]], mean: list[float], std: list[float]) -> str:
    """Short hash identifying a fit: what a label depends on (features, group, centroids and standardisation)."""
    payload = {"features": specs, "group_by": group_by, "centroids": centroids, "mean": mean, "std": std}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


class ClusterTransformer:
    def __init__(self, config: DictConfig) -> None:
        """Read and validate the block: `features` (names or `{name, transform: rank}`), `group_by`, `k`, `seed`, `n_init`, `max_rows_per_group`."""
        self._specs: list[dict[str, str | None]] = []
        for entry in config.get("features") or []:
            name, transform = (entry, None) if isinstance(entry, str) else (entry.get("name"), entry.get("transform"))
            if transform not in (None, RANK):
                raise ValueError(f"cluster feature {name!r}: unknown transform {transform!r}; expected {RANK!r} or none")
            if any(spec["name"] == name for spec in self._specs):
                raise ValueError(f"cluster feature {name!r} is listed twice")
            self._specs.append({"name": str(name), "transform": transform})
        if not self._specs:
            raise ValueError("cluster block needs at least one entry in features")

        self.features = [str(spec["name"]) for spec in self._specs]
        self.ranked = [str(spec["name"]) for spec in self._specs if spec["transform"] == RANK]
        self.group_by: str | None = config.get("group_by")
        self.k = int(config.k)
        self.seed = int(config.seed)
        self.n_init = int(config.n_init)
        self.max_rows_per_group = int(config.max_rows_per_group)
        if self.k < 2:
            raise ValueError(f"cluster k must be at least 2, got {self.k}")
        if self.n_init < 1 or self.max_rows_per_group < 1:
            raise ValueError("cluster n_init and max_rows_per_group must be at least 1")

        self.centroids: list[list[float]] = []
        self.mean: list[float] = []
        self.std: list[float] = []
        self.sizes: list[int] = []
        self.version: str | None = None

    def fit(self, df: pd.DataFrame) -> Self:
        """Fit k-means on the standardised sample of `df`; returns self."""
        # Complete rows only, then at most `max_rows_per_group` of them per group (seeded draw, order of `df` fixed).
        matrix = _prepare(df, self.features, self.ranked, self.group_by)
        complete = matrix.notna().all(axis=1).to_numpy()
        groups = df[self.group_by].to_numpy() if self.group_by else np.zeros(len(df), dtype=int)
        draw = pd.Series(np.random.default_rng(self.seed).random(len(df))[complete])
        within_group = draw.groupby(groups[complete]).rank(method="first").to_numpy()
        sample = matrix.to_numpy()[complete][within_group <= self.max_rows_per_group]
        if len(sample) < self.k:
            raise ValueError(f"cluster fit needs at least k={self.k} complete rows, got {len(sample)}")

        # Standardise: every feature gets mean 0 and std 1 on the sample.
        mean = sample.mean(axis=0)
        std = sample.std(axis=0)
        if (std == 0).any():
            raise ValueError(f"cluster features {[n for n, s in zip(self.features, std, strict=True) if s == 0]} are constant in the fit sample")
        model = KMeans(n_clusters=self.k, init="k-means++", n_init=cast(Any, self.n_init), random_state=self.seed).fit((sample - mean) / std)

        # Store the fit.
        self.centroids = model.cluster_centers_.tolist()
        self.mean = mean.tolist()
        self.std = std.tolist()
        self.sizes = np.bincount(cast(np.ndarray, model.labels_), minlength=self.k).tolist()
        self.version = _digest(self._specs, self.group_by, self.centroids, self.mean, self.std)
        logger.info(
            "cluster fit: %d of %d rows used (%d with a null feature), k=%d, sizes %s, version %s",
            len(sample),
            len(df),
            int((~complete).sum()),
            self.k,
            self.sizes,
            self.version,
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Nearest-centroid label per row of `df` (nullable int, null where a feature is null); nothing is refitted."""
        if self.version is None:
            raise RuntimeError("the cluster transformer is not fitted: call fit or load first")

        # Standardise with the stored parameters; rows with a null feature get no label.
        matrix = _prepare(df, self.features, self.ranked, self.group_by).to_numpy()
        z = (matrix - np.array(self.mean)) / np.array(self.std)
        null_row = np.asarray(np.isnan(z).any(axis=1), dtype=bool)
        valid = ~null_row

        # Nearest centroid: argmin of |c|^2 - 2 z.c (|z|^2 is the same for every centroid).
        centroids = np.array(self.centroids)
        nearest = np.zeros(len(df), dtype="int64")
        nearest[valid] = np.argmin((centroids**2).sum(axis=1) - 2.0 * z[valid] @ centroids.T, axis=1)
        return pd.Series(pd.arrays.IntegerArray(nearest, null_row), index=df.index)

    def save(self, path: Path) -> None:
        """Write the fit to `path` as JSON."""
        if self.version is None:
            raise RuntimeError("the cluster transformer is not fitted: nothing to save")
        payload: dict[str, Any] = {
            "version": self.version,
            "features": self._specs,
            "group_by": self.group_by,
            "k": self.k,
            "seed": self.seed,
            "n_init": self.n_init,
            "max_rows_per_group": self.max_rows_per_group,
            "mean": self.mean,
            "std": self.std,
            "centroids": self.centroids,
            "sizes": self.sizes,
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Self:
        """A fitted transformer read from the JSON `save` wrote; refuses a file whose content no longer matches its version."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        block = {key: payload[key] for key in ("features", "group_by", "k", "seed", "n_init", "max_rows_per_group")}
        transformer = cls(OmegaConf.create(block))
        digest = _digest(payload["features"], payload["group_by"], payload["centroids"], payload["mean"], payload["std"])
        if digest != payload["version"]:
            raise ValueError(f"{path}: content hash {digest} differs from the stored version {payload['version']}")
        transformer.centroids = payload["centroids"]
        transformer.mean = payload["mean"]
        transformer.std = payload["std"]
        transformer.sizes = payload["sizes"]
        transformer.version = payload["version"]
        return transformer
