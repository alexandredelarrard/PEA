"""Run routing, immutable generations and precision-sensitive artifact identities."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.modelling.steps.step_assess_long_short import StepAssessLongShort
from src.modelling.steps.step_long_short import Schema, StepLongShort
from src.modelling.utils import artifacts
from src.modelling.utils.assessment import write_json
from src.utils.modelling_runs import read_run, run_directory
from tests.modelling.model_fixtures import make_config

FAMILIES = ["elasticnet", "lgbm", "random_forest"]


class _Member:
    def __init__(self, tag: str) -> None:
        self.tag = tag


def _open_fixture_run(tmp_path: Path, *, omit_train: bool = False) -> StepLongShort:
    step = object.__new__(StepLongShort)
    step._config = make_config(
        model={"horizons": [60], "target_type": "zscore", "assessment": {"run_id": "20261018_fixture"}},
        train={"start_date": "1999-01-01", "outcome_cutoff": "2021-12-31"},
        build_cube={"output": {"save_predictions": False, "save_signal": False, "save_cv_results": True}},
        local={"password": "must-not-persist"},
        data_extract={"api_key": "must-not-persist"},
        environment={"secret": "must-not-persist"},
    )
    step._config.linear.alpha = "${lgbm.lambda_l1}"
    if omit_train:
        del step._config.train
    step.families = list(step._config.model.ensemble)
    step._calendar = pd.bdate_range("2021-01-04", periods=10)
    step._training_as_of = step._calendar[-1]
    step._run_id = "20261018_fixture"
    step._run_path = tmp_path / step._run_id
    step.target_type, step.label_column = "zscore", "y"
    step._open_run(Schema([60], ["date", "ticker", "f0", "sector"], ["f0"], ["sector"], ["f0", "sector"], {}, {}), False)
    return step


def test_omitted_optional_configuration_block_keeps_existing_fitting_signature(tmp_path: Path) -> None:
    step = _open_fixture_run(tmp_path, omit_train=True)
    assert step._run_path is not None
    saved = json.loads((step._run_path / "training/configuration.json").read_text(encoding="utf-8"))
    assert saved["train"] is None
    read_run(step._run_path, training_required=False)
    print("SANITY: an omitted fixture training block remains null and reconstructs its original fitting signature without adding train:{}.")


def test_producer_saves_resolved_allowlisted_configuration_without_fitting(tmp_path: Path) -> None:
    step = _open_fixture_run(tmp_path)
    path = step._run_path / "training/configuration.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert set(saved) == {"seed", "train", "model", "linear", "lgbm", "random_forest"}
    assert saved["seed"] == step._config.seed and saved["linear"]["alpha"] == 1.0
    assert "assessment" not in saved["model"] and "models_dir" not in saved["model"]
    assert "must-not-persist" not in path.read_text(encoding="utf-8") and "${" not in path.read_text(encoding="utf-8")
    read_run(step._run_path, training_required=False)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        step._open_run(Schema([60], ["date", "ticker", "f0"], ["f0"], [], ["f0"], {}, {}), False)
    assert path.read_bytes() == before
    print("SANITY: producer saves only resolved fitting controls, excludes secrets/runtime routing and refuses reusing an existing run.")


@pytest.mark.parametrize("mutation", ["missing", "wrong-controls", "wrong-shape"])
def test_dated_run_refuses_missing_or_corrupt_configuration_before_consumption(tmp_path: Path, mutation: str) -> None:
    step = _open_fixture_run(tmp_path)
    path = step._run_path / "training/configuration.json"
    if mutation == "missing":
        path.unlink()
    else:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if mutation == "wrong-controls":
            saved["model"]["target_type"] = "rank"
        else:
            saved["unrelated"] = {}
        write_json(path, saved)
    with pytest.raises((ValueError, FileNotFoundError)):
        read_run(step._run_path, training_required=False)
    print(f"SANITY: {mutation} configuration is refused before fitted models are consumed.")


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
    assert run_directory(context, config, "20261018_test") == tmp_path / "data/output/modelling_backtest/20261018_test"
    config.model.assessment.output_dir = str(tmp_path / "alternate")
    assert run_directory(context, config, "20261018_test") == tmp_path / "alternate/20261018_test"
    for bad in ("", "..", "../outside", "nested/run", "nested\\run", " bad", "CON", "run."):
        with pytest.raises(ValueError):
            run_directory(context, config, bad)
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


@pytest.mark.parametrize("fold", [0, 1])
def test_assessment_requires_member_inventory_before_any_unpickle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fold: int) -> None:
    run = tmp_path / "run"
    directory = _snapshot(run, f"folds/h60/fold{fold}" if fold else "cutoff", "saved-")
    metadata = artifacts.read_metadata(directory)
    del metadata["member_files"]
    metadata.update(role="fold" if fold else "cutoff", fold=fold, full_history=False)
    artifacts.write_metadata(directory, metadata)
    step = object.__new__(StepAssessLongShort)
    step._output = run
    calls: list[Path] = []

    def forbidden_unpickle(path: Path) -> None:
        calls.append(path)
        raise AssertionError("No unpickle is allowed without the original member inventory")

    monkeypatch.setattr(artifacts, "load_member", forbidden_unpickle)
    training = {"run_id": metadata["run_id"], "model_types": FAMILIES}
    with pytest.raises(ValueError, match="inventory"):
        step._score_window(60, fold, pd.Timestamp("2022-01-03"), pd.Timestamp("2022-01-04"), 3, training, {})
    assert calls == []
    print(f"SANITY: {'fold' if fold else 'cutoff'} assessment refuses missing original member inventory before any unpickle.")


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
