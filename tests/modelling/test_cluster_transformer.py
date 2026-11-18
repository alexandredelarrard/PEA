"""ClusterTransformer: fit / transform / save / load on synthetic, non-13F data (AC-002).

The planted data has three groups whose centres differ on two features of very different scales plus one pure-noise feature of a huge
scale, so recovering the groups proves the z-score gives every feature the same weight. No DB, no context, no LightGBM.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf
from sklearn.cluster import KMeans

from src.modelling.transformers.cluster import ClusterTransformer

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ["big_noise", "small_a", "tiny_b"]


def _block(**overrides: Any) -> DictConfig:
    """A transformer config block over the planted features."""
    base: dict[str, Any] = {"features": list(FEATURES), "group_by": None, "k": 3, "seed": 7, "n_init": 5, "max_rows_per_group": 100_000}
    return OmegaConf.create({**base, **overrides})


def _planted(n_per_cluster: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Three planted clusters: informative `small_a` (scale 1) and `tiny_b` (scale 1e-3), nuisance `big_noise` (scale 1e6)."""
    rng = np.random.default_rng(seed)
    truth = np.repeat([0, 1, 2], n_per_cluster)
    centres_a = np.array([0.0, 1.0, 2.0])[truth]
    centres_b = np.array([0.0, 2e-3, 0.0])[truth]
    df = pd.DataFrame(
        {
            "big_noise": rng.normal(0.0, 1e6, truth.size),
            "small_a": centres_a + rng.normal(0.0, 0.05, truth.size),
            "tiny_b": centres_b + rng.normal(0.0, 5e-5, truth.size),
        }
    )
    return df, pd.Series(truth, index=df.index)


def _purity(labels: pd.Series, truth: pd.Series) -> float:
    """Share of rows in the majority truth class of their label (1.0 = planted clusters recovered up to renaming)."""
    table = pd.crosstab(labels.astype(int), truth)
    return float(table.max(axis=1).sum() / table.to_numpy().sum())


def test_recovers_planted_clusters_with_equal_feature_weight() -> None:
    df, truth = _planted()
    transformer = ClusterTransformer(_block()).fit(df)
    labels = transformer.transform(df)

    purity = _purity(labels, truth)
    raw_purity = _purity(pd.Series(KMeans(3, random_state=7).fit_predict(df.to_numpy())), truth)
    assert purity >= 0.99, f"standardised fit missed the planted clusters: purity {purity:.3f}"
    assert raw_purity < 0.7, f"control: unscaled k-means should be dominated by the 1e6 feature, purity {raw_purity:.3f}"
    assert labels.dtype == "Int64" and set(labels.unique()) == {0, 1, 2}
    print(f"SANITY: standardised k-means recovers the 3 planted clusters (purity {purity:.3f}); unscaled control {raw_purity:.3f}")


def test_standardisation_matches_the_fit_sample() -> None:
    df, _ = _planted()
    transformer = ClusterTransformer(_block()).fit(df)
    mean = np.array(transformer.mean)
    std = np.array(transformer.std)

    assert np.allclose(mean, df[FEATURES].mean().to_numpy(), rtol=1e-12, atol=0)
    assert np.allclose(std, df[FEATURES].std(ddof=0).to_numpy(), rtol=1e-12, atol=0)
    z = (df[FEATURES].to_numpy() - mean) / std
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-9) and np.allclose(z.std(axis=0), 1.0, atol=1e-9)
    print("SANITY: the stored mean/std standardise every feature to mean 0, std 1 on the fit sample (equal weight)")


def test_same_seed_same_centroids_and_version(tmp_path: Path) -> None:
    df, _ = _planted()
    first = ClusterTransformer(_block()).fit(df)
    second = ClusterTransformer(_block()).fit(df)

    assert np.array_equal(np.array(first.centroids), np.array(second.centroids))
    assert first.version == second.version
    first.save(tmp_path / "a.json")
    second.save(tmp_path / "b.json")
    assert (tmp_path / "a.json").read_text() == (tmp_path / "b.json").read_text()
    print(f"SANITY: two fits with seed 7 give identical centroids and identical JSON (version {first.version})")


def test_save_load_reproduces_labels_exactly(tmp_path: Path) -> None:
    df, _ = _planted()
    fitted = ClusterTransformer(_block()).fit(df)
    before = fitted.transform(df)
    path = tmp_path / "centroids.json"
    fitted.save(path)

    loaded = ClusterTransformer.load(path)
    after = loaded.transform(df)
    payload = json.loads(path.read_text())

    assert before.equals(after), "labels changed through save -> load"
    assert {"version", "features", "centroids", "mean", "std", "sizes", "k", "seed", "n_init", "group_by", "max_rows_per_group"} <= set(payload)
    assert loaded.version == fitted.version and loaded.centroids == fitted.centroids
    print(f"SANITY: save -> load -> transform reproduces all {len(df)} labels exactly; JSON keys {sorted(payload)}")


def test_tampered_json_is_refused(tmp_path: Path) -> None:
    df, _ = _planted(n_per_cluster=50)
    path = tmp_path / "centroids.json"
    ClusterTransformer(_block()).fit(df).save(path)
    payload = json.loads(path.read_text())
    payload["centroids"][0][0] += 1.0
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="version"):
        ClusterTransformer.load(path)
    print("SANITY: a centroid edited in the JSON no longer matches its version hash and load refuses it")


def test_rank_transform_is_within_group(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    base = rng.permutation(np.arange(1.0, 201.0))
    df = pd.DataFrame({"g": ["a"] * 200 + ["b"] * 200, "size": np.concatenate([base, base * 1e4])})
    block = _block(features=[{"name": "size", "transform": "rank"}], group_by="g", k=2)
    transformer = ClusterTransformer(block).fit(df)
    labels = transformer.transform(df)

    lower = labels[df.groupby("g")["size"].rank(pct=True) <= 0.5]
    upper = labels[df.groupby("g")["size"].rank(pct=True) > 0.5]
    assert lower.nunique() == 1 and upper.nunique() == 1 and lower.iloc[0] != upper.iloc[0]
    print("SANITY: a ranked feature is compared inside its group, so two groups 10,000x apart in scale label the same percentile alike")


def test_sampling_caps_rows_per_group() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"g": np.repeat(["a", "b", "c", "d"], 500), "x": rng.normal(size=2000), "y": rng.normal(size=2000)})
    block = _block(features=["x", "y"], group_by="g", k=2, max_rows_per_group=50)
    transformer = ClusterTransformer(block).fit(df)
    again = ClusterTransformer(block).fit(df)

    assert sum(transformer.sizes) == 200, f"expected 4 groups x 50 rows, fitted {sum(transformer.sizes)}"
    assert transformer.centroids == again.centroids
    print("SANITY: at most 50 rows per group enter the fit (200 of 2000), and the draw is reproducible with the seed")


def test_null_feature_gives_null_label() -> None:
    df, _ = _planted(n_per_cluster=50)
    transformer = ClusterTransformer(_block()).fit(df)
    probe = df.head(6).copy()
    probe.loc[probe.index[2], "small_a"] = np.nan
    labels = transformer.transform(probe)

    assert labels.isna().tolist() == [False, False, True, False, False, False]
    assert labels.index.equals(probe.index)
    print("SANITY: a row with a null feature gets a null label; the other rows keep theirs and the index is preserved")


def test_fit_ignores_rows_with_a_null_feature() -> None:
    df, _ = _planted(n_per_cluster=60)
    holed = df.copy()
    holed.loc[holed.index[::10], "small_a"] = np.nan
    transformer = ClusterTransformer(_block()).fit(holed)

    assert sum(transformer.sizes) == int(holed[FEATURES].notna().all(axis=1).sum())
    print(f"SANITY: {len(holed) - sum(transformer.sizes)} rows with a null feature are left out of the fit, {sum(transformer.sizes)} used")


def test_generic_validation(caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(ValueError, match="k"):
        ClusterTransformer(_block(k=1))
    with pytest.raises(ValueError, match="features"):
        ClusterTransformer(_block(features=[]))
    with pytest.raises(ValueError, match="transform"):
        ClusterTransformer(_block(features=[{"name": "small_a", "transform": "log"}]))
    with pytest.raises(ValueError, match="small_a"):
        ClusterTransformer(_block(features=["small_a", "small_a"]))
    df, _ = _planted(n_per_cluster=20)
    with pytest.raises(ValueError, match="missing"):
        ClusterTransformer(_block()).fit(df.drop(columns="tiny_b"))
    with pytest.raises(ValueError, match="rows"):
        ClusterTransformer(_block(k=3)).fit(df.head(2))
    with pytest.raises(RuntimeError, match="fit"):
        ClusterTransformer(_block()).transform(df)
    with caplog.at_level(logging.INFO, logger="src.modelling.transformers.cluster"):
        ClusterTransformer(_block()).fit(df)
    assert any("cluster fit" in record.getMessage() for record in caplog.records), "the fit logs a summary through the module logger"
    print("SANITY: k < 2, no features, an unknown transform, a duplicate feature, a missing column, too few rows and transform-before-fit all raise")


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    """Run `code` in a fresh interpreter from the repository root."""
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=False)


def test_importing_cluster_does_not_load_the_model_families() -> None:
    code = (
        "import sys\n"
        "import src.modelling.transformers.cluster\n"
        "heavy = sorted(m for m in ('lightgbm', 'src.modelling.transformers.base', 'src.modelling.transformers.lightgbm_model',"
        " 'src.modelling.transformers.linear_regression', 'src.modelling.transformers.families') if m in sys.modules)\n"
        "print('LOADED', heavy)\n"
    )
    result = _run_python(code)
    assert result.returncode == 0, result.stderr
    assert "LOADED []" in result.stdout, result.stdout
    print("SANITY: importing the cluster transformer loads neither LightGBM nor any model family nor base")


def test_import_order_with_the_manager_profile_step() -> None:
    first = _run_python("import src.data_aggregate.transformers.step_manager_profile\nimport src.modelling.transformers.cluster\nprint('OK')")
    second = _run_python("import src.modelling.transformers.cluster\nimport src.data_aggregate.transformers.step_manager_profile\nprint('OK')")
    assert first.returncode == 0 and second.returncode == 0, first.stderr + second.stderr
    print("SANITY: the manager profile step and the cluster transformer import in either order without a cycle")
