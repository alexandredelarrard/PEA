"""Known-truth configuration reconciliation through the completed assessment checker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
from omegaconf import DictConfig, OmegaConf

from src.context import Context
from src.validate.checks import modelling


@pytest.fixture
def completed_assessment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Context, DictConfig, Path, Mock]:
    config = cast(
        DictConfig,
        OmegaConf.create(
            {
                "seed": 4325,
                "model": {
                    "ensemble": ["lgbm", "random_forest", "elasticnet"],
                    "label_column": "y",
                    "target_type": "zscore",
                    "weight_decay": {"enabled": False, "half_life_years": 10},
                    "cv": {"n_splits": 5, "embargo": None, "retain_models": False, "input_digest": False},
                    "assessment": {
                        "horizon": 60,
                        "outcome_cutoff": "2021-12-31",
                        "evaluation_start": "2022-01-03",
                        "test_windows": [["2000-01-01", "2002-12-31"]],
                        "prune_fraction": 0.2,
                        "max_rounds": 10,
                        "patience": 2,
                        "read_chunksize": 20,
                    },
                },
                "lgbm": {
                    "columns": ["legacy_tree"],
                    "columns_by_horizon": {60: ["tree_signal", "industry"]},
                    "categoricals": ["industry"],
                    "learning_rate": 0.01,
                    "monotonic": {"enabled": True, "features": [{"tree_signal": 1}]},
                },
                "random_forest": {
                    "columns": ["legacy_rf"],
                    "columns_by_horizon": {60: ["rf_signal"]},
                    "num_boost_round": 500,
                },
                "linear": {"columns": ["legacy_linear"], "columns_by_horizon": {60: ["linear_signal"]}, "alpha": 0.001},
            }
        ),
    )
    selected = {
        "lgbm": {"features": ["tree_signal", "industry"], "categoricals": ["industry"]},
        "random_forest": {"features": ["rf_signal"], "categoricals": []},
        "elasticnet": {"features": ["linear_signal"], "categoricals": []},
    }
    ledger = {
        "selected": selected,
        "fixed_parameters": {
            "lgbm": {"categoricals": ["industry"], "learning_rate": 0.01, "monotonic": {"enabled": True, "features": [{"tree_signal": 1}]}},
            "random_forest": {"num_boost_round": 500},
            "linear": {"alpha": 0.001},
        },
        "controls": {
            "seed": 4325,
            "shared": {
                "ensemble": ["lgbm", "random_forest", "elasticnet"],
                "label_column": "y",
                "target_type": "zscore",
                "weight_decay": {"enabled": False, "half_life_years": 10},
                "cv": {
                    "n_splits": 5,
                    "embargo": None,
                    "retain_models": True,
                    "input_digest": True,
                    "test_windows": [["2000-01-01", "2002-12-31"]],
                },
            },
            "assessment": OmegaConf.to_container(config.model.assessment, resolve=True),
        },
    }
    ledger["controls"]["assessment"].pop("read_chunksize")
    output = tmp_path / "_out"
    for directory in [output / "final", output / "eval"]:
        directory.mkdir(parents=True)
    report = b"<html>Known-truth artifact identities; expensive checks are isolated.</html>"
    (tmp_path / "report.html").write_bytes(report)
    for path, payload in [
        (output / "selection.json", ledger),
        (output / "final" / "result.json", {}),
        (output / "eval" / "result.json", {"metrics": {}}),
        (
            output / "report_manifest.json",
            {
                "report_sha256": hashlib.sha256(report).hexdigest(),
                "retained_tree_features": {"lgbm": selected["lgbm"]["features"], "random_forest": ["rf_signal"]},
            },
        ),
    ]:
        path.write_text(json.dumps(payload), encoding="utf-8")
    for name, value in [("_reconcile_selection", None), ("_reconcile_metrics", {}), ("_reconcile_cohorts", 1), ("_report_checks", None)]:
        monkeypatch.setattr(modelling, name, Mock(return_value=value))
    replay = Mock(return_value=1)
    monkeypatch.setattr(modelling, "_live_path_checks", replay)
    store = Mock()
    context = cast(Context, SimpleNamespace(store=store))
    return context, config, tmp_path, replay


@pytest.mark.parametrize(
    ("block", "kind", "values"),
    [
        ("lgbm", "columns", ["changed_tree"]),
        ("random_forest", "columns", ["tree_signal"]),
        ("linear", "columns", ["changed_linear"]),
        ("lgbm", "categoricals", []),
        ("random_forest", "categoricals", ["industry"]),
    ],
)
def test_changed_selected_pool_fails_before_source_replay(
    completed_assessment: tuple[Context, DictConfig, Path, Mock], block: str, kind: str, values: list[str]
) -> None:
    context, config, run, replay = completed_assessment
    if kind == "categoricals":
        if block == "lgbm":
            config.lgbm.categoricals = values
        else:
            config[block].columns_by_horizon[60] = ["rf_signal", *values]
    else:
        config[block].columns_by_horizon[60] = values
    result = modelling.check_modelling(context, out=run, config=config)
    assert result.status == "fail", result.summary()
    assert any(finding.field == f"{block}.{kind}_h60" for finding in result.findings)
    replay.assert_not_called()
    assert json.loads((run / "_out" / "validation.json").read_text(encoding="utf-8"))["verdict"] == "FAIL"
    print(f"SANITY: changed {block} h60 {kind} is a clear finding and prevents live source replay.")


@pytest.mark.parametrize(
    ("path", "value", "field"),
    [
        ("lgbm.learning_rate", 0.02, "lgbm.fixed_parameters"),
        ("random_forest.num_boost_round", 501, "random_forest.fixed_parameters"),
        ("linear.alpha", 0.002, "linear.fixed_parameters"),
        ("lgbm.monotonic.features", [{"tree_signal": -1}], "lgbm.fixed_parameters"),
        ("seed", 4326, "seed"),
        ("model.target_type", "rank", "model.target_type"),
        ("model.label_column", "other_y", "model.label_column"),
        ("model.ensemble", ["lgbm"], "model.ensemble"),
        ("model.weight_decay.enabled", True, "model.weight_decay"),
        ("model.cv.n_splits", 4, "model.cv"),
        ("model.assessment.outcome_cutoff", "2022-12-31", "model.assessment.outcome_cutoff"),
    ],
)
def test_changed_fixed_control_fails(completed_assessment: tuple[Context, DictConfig, Path, Mock], path: str, value: Any, field: str) -> None:
    context, config, run, replay = completed_assessment
    OmegaConf.update(config, path, value, merge=False)
    result = modelling.check_modelling(context, out=run, config=config)
    assert result.status == "fail", result.summary()
    assert any(finding.field == field for finding in result.findings)
    replay.assert_not_called()
    print(f"SANITY: changed fixed control {path} fails independently of the unchanged report artifacts.")


@pytest.mark.parametrize("string_key", [False, True])
def test_matching_configuration_respects_empty_pools_and_runtime_cv(
    completed_assessment: tuple[Context, DictConfig, Path, Mock], string_key: bool
) -> None:
    context, config, run, replay = completed_assessment
    config.model.cv.test_windows = [["2099-01-01", "2099-12-31"]]
    config.linear.columns_by_horizon = {"60" if string_key else 60: []}
    ledger_path = run / "_out" / "selection.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["selected"]["elasticnet"]["features"] = []
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    for block in ["lgbm", "random_forest"]:
        values = list(config[block].columns_by_horizon[60])
        config[block].columns_by_horizon = {"60" if string_key else 60: values}
    result = modelling.check_modelling(context, out=run, config=config)
    assert result.status == "pass", result.summary()
    replay.assert_called_once()
    print("SANITY: separate family pools, explicit numeric/category empties and int/string h60 keys agree; forced runtime CV keys do not flag YAML.")


@pytest.mark.parametrize("kind", ["columns"])
def test_missing_horizon_override_uses_actual_fallback(completed_assessment: tuple[Context, DictConfig, Path, Mock], kind: str) -> None:
    context, config, run, replay = completed_assessment
    del config.lgbm[f"{kind}_by_horizon"]
    result = modelling.check_modelling(context, out=run, config=config)
    assert result.status == "fail"
    finding = next(finding for finding in result.findings if finding.field == f"lgbm.{kind}_h60")
    assert finding.evidence["configured"] == list(config.lgbm[kind])
    replay.assert_not_called()
    print(f"SANITY: missing h60 {kind} override compares the legacy fallback instead of silently trusting frozen selection.")


def test_rf_fallback_is_checked_independently(completed_assessment: tuple[Context, DictConfig, Path, Mock]) -> None:
    context, config, run, replay = completed_assessment
    del config.random_forest.columns_by_horizon
    del config.random_forest.columns
    ledger_path = run / "_out" / "selection.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["selected"]["random_forest"] = ledger["selected"]["lgbm"]
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    manifest_path = run / "_out" / "report_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["retained_tree_features"]["random_forest"] = ["tree_signal", "industry"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = modelling.check_modelling(context, out=run, config=config)
    assert result.status == "pass", result.summary()
    replay.assert_called_once()
    print("SANITY: absent RF numeric/category declarations resolve to the actual LightGBM h60 pools without importing modelling code.")
