"""Artifact paths and metadata contract (src/modelling/utils/artifacts.py): the configured models
directory, the metadata `artifact_format` gate, and the ensemble loader's order and skips."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.modelling.utils.artifacts import (
    ARTIFACT_FORMAT,
    clear_members,
    load_ensemble,
    member_path,
    models_dir,
    read_metadata,
    safe_filename,
    save_member,
    write_metadata,
)


class _Member:
    def __init__(self, tag: str) -> None:
        self.tag = tag


def test_models_dir_resolution(tmp_path: Path) -> None:
    ctx = SimpleNamespace(paths={"DATA_STORE": tmp_path / "data", "MODELS_DIR": tmp_path / "data" / "output" / "models"})
    assert models_dir(ctx, OmegaConf.create({"model": {"models_dir": "output/models"}})) == tmp_path / "data" / "output" / "models"
    assert models_dir(ctx, OmegaConf.create({"model": {"models_dir": str(tmp_path / "abs")}})) == tmp_path / "abs"
    assert models_dir(ctx, OmegaConf.create({"model": {}})) == ctx.paths["MODELS_DIR"]
    print("\n=== SANITY CHECK: models_dir ===")
    print("  relative -> under DATA_STORE (== context MODELS_DIR for 'output/models'); absolute as-is; unset -> MODELS_DIR. Validated.")


def test_metadata_roundtrip_and_format_gate(tmp_path: Path) -> None:
    write_metadata(tmp_path, {"horizons": [30, 60], "train_end": "2023-06-30"})
    meta = read_metadata(tmp_path)
    assert meta["artifact_format"] == ARTIFACT_FORMAT and meta["horizons"] == [30, 60]

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "metadata.json").write_text(json.dumps({"horizons": [30]}))  # pre-refactor layout
    with pytest.raises(RuntimeError, match="retrain"):
        read_metadata(legacy)
    with pytest.raises(FileNotFoundError):
        read_metadata(tmp_path / "missing")
    print("\n=== SANITY CHECK: metadata gate ===")
    print(
        f"  written metadata carries artifact_format={ARTIFACT_FORMAT!r}; a pre-refactor metadata.json -> 'retrain' error; missing -> FileNotFoundError. Validated."
    )


def test_load_ensemble_order_and_skips(tmp_path: Path) -> None:
    for h, fam in ((60, "lgbm"), (30, "lgbm"), (30, "elasticnet")):
        save_member(_Member(f"{fam}{h}"), member_path(tmp_path, h, fam))
    got = load_ensemble(tmp_path, [30, 60, 90], ["elasticnet", "lgbm", "random_forest"])
    assert list(got) == [30, 60], "horizon 90 has no file -> absent"
    assert list(got[30]) == ["elasticnet", "lgbm"] and list(got[60]) == ["lgbm"], "family order kept, missing skipped"
    assert got[30]["lgbm"].tag == "lgbm30"
    with pytest.raises(FileNotFoundError):
        load_ensemble(tmp_path, [5], ["lgbm"])
    assert member_path(tmp_path, 30, "random_forest").name == "model_h30_random_forest.pkl"
    assert safe_filename("beta_USD/EUR 1") == "beta_USD_EUR_1"
    print("\n=== SANITY CHECK: load_ensemble ===")
    print(f"  {{h: families}} = { ({h: list(m) for h, m in got.items()}) }; no member at all -> FileNotFoundError. Validated.")


def test_clear_members_removes_only_member_pickles(tmp_path: Path) -> None:
    for h, fam in ((30, "lgbm"), (90, "elasticnet")):
        save_member(_Member(f"{fam}{h}"), member_path(tmp_path, h, fam))
    write_metadata(tmp_path, {"horizons": [30, 90]})
    (tmp_path / "model_h30_lgbm.txt").write_text("pre-refactor booster")  # never loaded, left alone
    removed = clear_members(tmp_path)
    assert sorted(p.name for p in removed) == ["model_h30_lgbm.pkl", "model_h90_elasticnet.pkl"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["metadata.json", "model_h30_lgbm.txt"]
    assert clear_members(tmp_path / "absent") == []
    print("\n=== SANITY CHECK: clear_members ===")
    print(f"  removed {[p.name for p in removed]}; metadata.json and non-member files kept; a missing directory is a no-op. Validated.")
