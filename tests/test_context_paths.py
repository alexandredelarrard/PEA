"""Configured non-tabular peer paths remain independent of the working directory."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.context import define_global_paths


@pytest.mark.parametrize("directory", [None, "output/custom-peers", "absolute"])
def test_peer_directory_resolves_under_the_configured_data_root(directory: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROOT_PATH", raising=False)
    paths = {"root": str(tmp_path), "data_store": "warehouse"}
    if directory is not None:
        paths["peers_dir"] = str(tmp_path / "external-peers") if directory == "absolute" else directory
    config = OmegaConf.create({"local": {"paths": paths}})
    actual = define_global_paths(config)
    expected = tmp_path / "external-peers" if directory == "absolute" else tmp_path / "warehouse" / (directory or "output/peers")
    assert actual["SECTOR_PEERS_PATH"] == expected / "sector_peers.json"
    assert actual["OUTPUT_DIR"] == tmp_path / "warehouse/output"
    assert actual["MODELS_DIR"] == tmp_path / "warehouse/output/models"
    assert expected.is_dir()
    print(f"SANITY: peer directory {directory or 'default'} resolves from the configured data root; production output paths are preserved.")
