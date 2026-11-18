"""cluster.yml: the shipped defaults, validation of the clustering features, the exact 80 cube columns and the size budget (AC-001)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from omegaconf import DictConfig, OmegaConf

from src.data_aggregate.utils.sec_thirteen.cluster_config import ClusterSpec, cluster_specs, column_budget, column_names, validate_cluster_config

ROOT = Path(__file__).resolve().parents[2]
LOGGER = "src.data_aggregate.utils.sec_thirteen.cluster_config"
FUND_FEATURES = ["net_breadth", "new_position_rate", "net_share_flow", "top10_chg", "rank_gap", "rank_gap_chg", "stickiness", "cap_proximity"]
FOUR = ["net_breadth", "new_position_rate", "net_share_flow", "top10_chg"]


def _config(**edits: Any) -> DictConfig:
    """The shipped `cluster` block, with top-level keys replaced by `edits`."""
    block = OmegaConf.to_container(OmegaConf.load(ROOT / "configs" / "cluster.yml").cluster, resolve=True)
    assert isinstance(block, dict)
    return OmegaConf.create({**block, **edits})  # type: ignore[return-value]


def _clusters(**changes: dict[str, Any] | None) -> dict[str, Any]:
    """The shipped `clusters` mapping with one entry replaced per keyword."""
    clusters = OmegaConf.to_container(_config().clusters)
    assert isinstance(clusters, dict)
    return {**{str(kind): block for kind, block in clusters.items()}, **changes}


def test_shipped_defaults() -> None:
    cluster = _config()
    assert cluster.fit is False and cluster.k == 10 and cluster.keep == 7
    assert cluster.seed == 0 and cluster.n_init == 30 and cluster.max_rows_per_quarter == 2000
    assert cluster.centroids_path == "cluster_centroids.json" and cluster.top_x is None
    assert cluster.cluster_label_delay_days == 90 and cluster.availability_buffer_days == 7 and str(cluster.history_start) == "2000-01-01"
    assert cluster.max_clusters == 11 and cluster.max_features_per_cluster == 10 and cluster.max_age_calendar_days == 100
    assert dict(cluster.fit_filters) == {"category": "fund", "min_positions": 10, "min_book_usd": 20_000_000}
    assert dict(cluster.exclude) == {"pct_etf": 0.40, "pct_equity_foreign": 0.40}
    assert len(cluster.features) == 10 and set(cluster.stability) == {"min_ari", "min_jaccard", "min_persistence", "min_share", "max_share"}
    validate_cluster_config(cluster)
    print(
        "SANITY: cluster.yml loads with fit false, k 10, keep 7, 10 clustering features, the stability thresholds and the budget keys; it validates"
    )


def test_return_columns_are_rejected_as_clustering_features() -> None:
    features = [*OmegaConf.to_container(_config().features), "book_return"]  # type: ignore[misc]
    with pytest.raises(ValueError, match="book_return"):
        validate_cluster_config(_config(features=features))
    ranked = [*OmegaConf.to_container(_config().features), {"name": "book_return", "transform": "rank"}]  # type: ignore[misc]
    with pytest.raises(ValueError, match="book_return"):
        validate_cluster_config(_config(features=ranked))
    print("SANITY: book_return is rejected among the clustering features, plain or ranked")


def test_unknown_features_are_rejected() -> None:
    with pytest.raises(ValueError, match="pct_unicorn"):
        validate_cluster_config(_config(features=["pct_cap_small", "pct_unicorn"]))
    bad = _clusters(bank={"id": 7, "features": [*FOUR, "momentum"], "min_holders": 8})
    with pytest.raises(ValueError, match="momentum"):
        validate_cluster_config(_config(clusters=bad))
    with pytest.raises(ValueError, match="pct_bond"):
        validate_cluster_config(_config(exclude={"pct_bond": 0.5}, features=["pct_etf", "pct_cap_small"]))
    print("SANITY: an unknown clustering feature, an unknown per-cluster feature and an exclude key outside the features are all rejected")


def test_cluster_ids_must_not_collide() -> None:
    clash = _clusters(bank={"id": 3, "features": FOUR, "min_holders": 8})
    with pytest.raises(ValueError, match="id"):
        validate_cluster_config(_config(clusters=clash))
    twin = _clusters(insurance={"id": 7, "features": FOUR, "min_holders": 4})
    with pytest.raises(ValueError, match="id"):
        validate_cluster_config(_config(clusters=twin))
    with pytest.raises(ValueError, match="min_holders"):
        validate_cluster_config(_config(clusters=_clusters(pension={"id": 10, "features": FOUR, "min_holders": 0})))
    print("SANITY: a category id inside the fund range, a duplicated category id and min_holders 0 are rejected")


def test_the_exact_80_column_names() -> None:
    specs = cluster_specs(_config())
    names = column_names(specs)

    expected = (
        [f"f_ic_cluster_{i}_{f}" for i in range(7) for f in FUND_FEATURES]
        + [f"f_ic_cluster_7_{f}" for f in [*FUND_FEATURES[:-1], "option_intensity"]]
        + [f"f_ic_cluster_8_{f}" for f in [*FUND_FEATURES[:-1], "flow_x_vol"]]
        + [f"f_ic_cluster_9_{f}" for f in FOUR]
        + [f"f_ic_cluster_10_{f}" for f in FOUR]
    )
    assert names == expected and len(names) == 80 == len(set(names))
    assert [(s.kind, s.cluster_id) for s in specs[:8]] == [
        ("fund", 0),
        ("fund", 1),
        ("fund", 2),
        ("fund", 3),
        ("fund", 4),
        ("fund", 5),
        ("fund", 6),
        ("bank", 7),
    ]
    assert {s.cluster_id: s.min_holders for s in specs} == {**dict.fromkeys(range(7), 8), 7: 8, 8: 4, 9: 5, 10: 4}
    print(
        "SANITY: the shipped config yields exactly the 80 f_ic_cluster_<id>_<feature> names: 7 fund x 8, bank 8, insurance 8, broker_dealer 4, pension 4"
    )


def test_budget_within_limits_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    specs = cluster_specs(_config())
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        n_columns = column_budget(specs, _config())
    assert n_columns == 80 and not caplog.records
    print("SANITY: 11 clusters and at most 8 features per cluster stay within the budget: 80 columns, no warning")


def test_budget_warns_above_11_clusters_and_still_builds(caplog: pytest.LogCaptureFixture) -> None:
    categories = {
        "bank": {"id": 8, "features": FOUR, "min_holders": 8},
        "insurance": {"id": 9, "features": FOUR, "min_holders": 4},
        "broker_dealer": {"id": 10, "features": FOUR, "min_holders": 5},
        "pension": {"id": 11, "features": FOUR, "min_holders": 4},
    }
    cluster = _config(keep=8, clusters={"fund": _clusters()["fund"], **categories})
    specs = cluster_specs(cluster)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        n_columns = column_budget(specs, cluster)

    assert len(specs) == 12 and n_columns == 8 * 8 + 4 * 4 == len(column_names(specs))
    assert any("12 clusters" in r.getMessage() and str(n_columns) in r.getMessage() for r in caplog.records), [r.getMessage() for r in caplog.records]
    print(f"SANITY: 12 clusters log one WARNING naming {n_columns} columns, and the specs and names are still built")


def test_budget_warns_above_10_features_per_cluster(caplog: pytest.LogCaptureFixture) -> None:
    eleven = ClusterSpec(kind="fund", cluster_id=0, features=tuple(f"feature_{i}" for i in range(11)), min_holders=8)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        n_columns = column_budget([eleven], _config())
    assert n_columns == 11 and any("11 features" in r.getMessage() for r in caplog.records), [r.getMessage() for r in caplog.records]

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        column_budget(cluster_specs(_config()), _config(max_features_per_cluster=7))
    assert any("8 features" in r.getMessage() for r in caplog.records)
    print("SANITY: a cluster of 11 features (or 8 against a limit of 7) logs the WARNING and the column count is still returned")
