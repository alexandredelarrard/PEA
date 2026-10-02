"""`ls_model._load_models` (src/strategies/utils/ls_model.py) loads only the horizons the current
`metadata.json` was trained on: a member file left by an older run for another configured horizon
is never unpickled."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from src.modelling.utils.artifacts import member_path, save_member, write_metadata
from src.strategies.utils.ls_model import _load_models


class _Member:
    def __init__(self, tag: str) -> None:
        self.tag = tag


def test_only_metadata_horizons_are_loaded(tmp_path: Path) -> None:
    for h in (30, 60):
        save_member(_Member(f"enet{h}"), member_path(tmp_path, h, "elasticnet"))
    write_metadata(tmp_path, {"horizons": [30, 60], "model_types": ["elasticnet"], "target_type": "rank"})
    # an older run's h90 member (here: bytes that are not even a pickle -- loading them would raise)
    member_path(tmp_path, 90, "elasticnet").write_bytes(b"stale, unreadable member of a previous run")

    config = OmegaConf.create({"model": {"models_dir": str(tmp_path)}, "build_cube": {"targets": {"horizons": [30, 60, 90]}}, "strategy_ls": {}})
    context = SimpleNamespace(paths={"DATA_STORE": tmp_path, "MODELS_DIR": tmp_path})
    meta, models, target_type, horizons = _load_models(context, config)  # type: ignore[arg-type]

    assert horizons == [30, 60] and list(models) == [30, 60]
    assert models[30]["elasticnet"].tag == "enet30" and target_type == "rank"
    print("\n=== SANITY CHECK: ls_model loads the trained horizons only ===")
    print(
        f"  config horizons [30, 60, 90], metadata horizons {meta['horizons']} -> loaded {list(models)}; the stale h90 file is never read. Validated."
    )
