"""Run routing, immutable generations and precision-sensitive artifact identities."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.modelling.utils import artifacts
from src.modelling.utils.assessment import frame_digest, write_json

FAMILIES = ["elasticnet", "lgbm", "random_forest"]


class _Member:
    def __init__(self, tag: str) -> None:
        self.tag = tag


def _snapshot(parent: Path, name: str, tag: str) -> Path:
    directory = parent / name
    for family in FAMILIES:
        artifacts.save_member(_Member(tag + family), artifacts.member_path(directory, 60, family))
    artifacts.write_metadata(
        directory,
        {
            "horizons": [60],
            "model_types": FAMILIES,
            "feature_cols": ["x"],
            "run_id": "20261018_test",
            "full_history": True,
            "member_files": artifacts.member_inventory(directory, [60], FAMILIES),
        },
    )
    return directory


def test_run_paths_resolve_configured_roots_and_reject_unsafe_ids(tmp_path: Path) -> None:
    context = SimpleNamespace(paths={"DATA_STORE": tmp_path / "data"})
    config = OmegaConf.create({"model": {"assessment": {"output_dir": "output/modelling_backtest"}}})
    assert artifacts.run_directory(context, config, "20261018_test") == tmp_path / "data/output/modelling_backtest/20261018_test"
    config.model.assessment.output_dir = str(tmp_path / "alternate")
    assert artifacts.run_directory(context, config, "20261018_test") == tmp_path / "alternate/20261018_test"
    for bad in ("", "..", "../outside", "nested/run", "nested\\run", " bad", "CON", "run."):
        with pytest.raises(ValueError):
            artifacts.run_directory(context, config, bad)
    assert not (tmp_path / "alternate").exists(), "consumer path resolution must not create a run"
    print("SANITY: dated run paths honor configured roots and reject traversal, separators and reserved Windows filenames.")


def test_generation_switch_pins_a_complete_previous_reader(tmp_path: Path) -> None:
    production = tmp_path / "production"
    first = _snapshot(tmp_path, "first", "old-")
    second = _snapshot(tmp_path, "second", "new-")
    artifacts.publish_generation(production, first, "20261018_test", "a" * 32)
    previous = artifacts.read_metadata(production)
    artifacts.publish_generation(production, second, "20261018_test", "b" * 32)
    current = artifacts.read_metadata(production)
    assert current["artifact_format"] != artifacts.ARTIFACT_FORMAT
    old = artifacts.load_ensemble(production, [60], FAMILIES, metadata=previous)
    new = artifacts.load_ensemble(production, [60], FAMILIES, metadata=current)
    assert [old[60][family].tag for family in FAMILIES] == ["old-" + family for family in FAMILIES]
    assert [new[60][family].tag for family in FAMILIES] == ["new-" + family for family in FAMILIES]
    assert (production / previous["member_directory"]).is_dir()
    print("SANITY: one complete metadata switch exposes the new set while an already-pinned reader still loads the unchanged old set.")


def test_failed_publication_retains_previous_models_and_retry_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    production = tmp_path / "production"
    first = _snapshot(tmp_path, "first", "old-")
    second = _snapshot(tmp_path, "second", "new-")
    artifacts.publish_generation(production, first, "20261018_test", "a" * 32)
    previous = (production / artifacts.METADATA_FILE).read_bytes()
    original_write = artifacts.write_metadata

    def failed_write(directory: Path, meta: dict) -> Path:
        if directory == production:
            raise OSError("simulated metadata publication failure")
        return original_write(directory, meta)

    with monkeypatch.context() as local:
        local.setattr(artifacts, "write_metadata", failed_write)
        with pytest.raises(OSError, match="publication failure"):
            artifacts.publish_generation(production, second, "20261018_test", "b" * 32)
    assert (production / artifacts.METADATA_FILE).read_bytes() == previous
    failed = production / "generations/20261018_test" / ("b" * 32)
    assert failed.is_dir()
    artifacts.publish_generation(production, second, "20261018_test", "c" * 32)
    assert failed.is_dir(), "failed attempts stay available for diagnosis"
    with pytest.raises(FileExistsError):
        artifacts.publish_generation(production, second, "20261018_test", "c" * 32)
    current = artifacts.read_metadata(production)
    assert artifacts.load_ensemble(production, [60], FAMILIES, metadata=current)[60]["lgbm"].tag == "new-lgbm"
    print("SANITY: failed publication leaves active bytes intact; same-run retry uses a fresh generation and never overwrites retained attempts.")


def test_manifest_rejects_tampering_before_any_unpickle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    production = tmp_path / "production"
    snapshot = _snapshot(tmp_path, "snapshot", "saved-")
    artifacts.publish_generation(production, snapshot, "20261018_test", "a" * 32)
    metadata = artifacts.read_metadata(production)
    directory = production / metadata["member_directory"]
    artifacts.member_path(directory, 60, "random_forest").write_bytes(b"tampered")
    calls = []

    def forbidden_unpickle(path: Path) -> None:
        calls.append(path)
        raise AssertionError("No unpickle is allowed before validating the complete manifest")

    monkeypatch.setattr(artifacts, "load_member", forbidden_unpickle)
    with pytest.raises(ValueError, match="identity|hash|changed"):
        artifacts.load_ensemble(production, [60], FAMILIES, metadata=metadata)
    assert calls == []
    with pytest.raises(ValueError, match="directory|escape|relative"):
        artifacts.load_ensemble(production, [60], FAMILIES, metadata={**metadata, "member_directory": "../outside"})
    print("SANITY: changed member bytes and escaping generation references fail before any pickle is opened.")


def test_manifested_flat_snapshot_requires_every_member(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path, "snapshot", "saved-")
    metadata = artifacts.read_metadata(snapshot)
    artifacts.member_path(snapshot, 60, "random_forest").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        artifacts.load_ensemble(snapshot, [60], FAMILIES, metadata=metadata)
    assert list(artifacts.load_ensemble(snapshot, [60], FAMILIES)[60]) == ["elasticnet", "lgbm"]
    print("SANITY: saved-run manifests require the full declared set; explicit generic flat-file loading preserves its optional-member behavior.")


def test_cutoff_snapshot_cannot_be_published_as_production(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path, "snapshot", "cutoff-")
    metadata = artifacts.read_metadata(snapshot)
    artifacts.write_metadata(snapshot, {**metadata, "full_history": False})
    with pytest.raises(ValueError, match="full-history"):
        artifacts.publish_generation(tmp_path / "production", snapshot, "20261018_test", "a" * 32)
    assert not (tmp_path / "production").exists()
    print("SANITY: cutoff members cannot be promoted as full-history production models.")


def test_json_replace_uses_unique_closed_temporary_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "metadata.json"
    names = []
    original_replace = Path.replace

    def observed_replace(source: Path, target: Path) -> Path:
        names.append(source.name)
        with source.open("ab"):
            pass
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", observed_replace)
    write_json(path, {"value": 1})
    write_json(path, {"value": 2})
    assert len(set(names)) == 2
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 2}
    assert not list(tmp_path.glob("*.tmp"))
    print("SANITY: metadata writes replace from different closed files in the same directory and leave no temporary files.")


def test_float64_score_identity_detects_prediction_relevant_changes() -> None:
    first = pd.DataFrame({"date": pd.to_datetime(["2021-01-04"]), "ticker": ["A"], "x": [1.0000000001]})
    second = first.assign(x=1.0000000002)
    assert frame_digest(first, ["x"]) == frame_digest(second, ["x"])
    assert frame_digest(first, ["x"], downcast=False) != frame_digest(second, ["x"], downcast=False)
    assert frame_digest(first.iloc[::-1], ["x"], downcast=False) == frame_digest(first, ["x"], downcast=False)
    print("SANITY: scoring provenance preserves float64 changes while the existing normalized training identity remains compatible.")
