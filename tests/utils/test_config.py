"""Canonical configuration ownership, deterministic loading and early conflict refusal."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from src.utils.config import read_config


def _bootstrap(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.yml").write_text("seed: 4325\nlocal:\n  paths:\n    root: ./\n    data_store: data\n", encoding="utf-8")


def test_stage_files_have_one_canonical_owner() -> None:
    folder = Path("configs")
    required = {
        "config.yml",
        "data.yml",
        "data_extract.yml",
        "data_aggregate.yml",
        "data_peers.yml",
        "modelling.yml",
        "portfolio.yml",
        "universe.yml",
        "validate.yml",
    }
    assert required <= {p.name for p in folder.glob("*.yml")}
    assert not {"configs.yml", "paths.yml", "build_cube.yml", "peers.yml", "modellling.yml"} & {p.name for p in folder.glob("*.yml")}
    config = read_config(folder)
    assert config.seed == 4325 and config.local.paths.root == "./" and config.local.paths.data_store == "data"
    assert "data_aggregate" in config and "build_cube" not in config
    assert set(config.model.ensemble) == {"elasticnet", "lgbm", "random_forest"}
    assert {"linear", "lgbm", "random_forest"} <= set(config)
    print("SANITY: one bootstrap and one file per stage preserve seed, path keys and all saved model-family trees.")


def test_relative_and_absolute_config_directories_are_equivalent() -> None:
    relative = read_config("configs")
    absolute = read_config(Path("configs").resolve())
    assert OmegaConf.to_container(relative, resolve=True) == OmegaConf.to_container(absolute, resolve=True)
    print("SANITY: relative and absolute configuration directories resolve to identical values.")


def test_fragments_load_after_bootstrap_in_sorted_path_order(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    (tmp_path / "zeta.yml").write_text("zeta: 2\n", encoding="utf-8")
    (tmp_path / "alpha.yml").write_text("alpha: 1\n", encoding="utf-8")
    with patch.object(OmegaConf, "load", wraps=OmegaConf.load) as load:
        config = read_config(tmp_path)
    assert [Path(call.args[0]).name for call in load.call_args_list] == ["config.yml", "alpha.yml", "zeta.yml"]
    assert config.alpha == 1 and config.zeta == 2
    print("SANITY: filesystem creation order cannot change merge precedence.")


@pytest.mark.parametrize("value", [4325, 9])
def test_duplicate_namespace_is_refused_before_context(tmp_path: Path, value: int) -> None:
    _bootstrap(tmp_path)
    (tmp_path / "other.yml").write_text(f"seed: {value}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="seed.*config.yml.*other.yml"):
        read_config(tmp_path)
    print("SANITY: a second owner is refused even when its current value matches; no hidden override can appear later.")


def test_non_mapping_fragment_fails_before_context(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    (tmp_path / "invalid.yml").write_text("- invalid\n", encoding="utf-8")
    with pytest.raises(TypeError, match="mapping.*invalid.yml"):
        read_config(tmp_path)
    print("SANITY: malformed configuration fails before providers, database access or output writes.")


def test_disjoint_namespaces_preserve_lazy_cross_file_interpolation(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    (tmp_path / "alpha.yml").write_text("alpha:\n  local: 4\n  relative: ${.local}\n  external: ${seed}\n", encoding="utf-8")
    config = read_config(tmp_path)
    assert config.alpha.relative == 4 and config.alpha.external == 4325
    config.seed = 9
    assert config.alpha.external == 9
    print("SANITY: disjoint file assembly preserves relative, cross-file and lazily resolved interpolation.")
