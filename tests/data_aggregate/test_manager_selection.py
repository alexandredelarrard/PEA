"""Known-truth configuration controls for the fixed raw-CIK cohort."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest


def _config(tmp_path: Path, managers: object) -> Path:
    directory = tmp_path / "superinvestors"
    directory.mkdir()
    (directory / "ranked_managers.json").write_text(json.dumps(managers), encoding="utf-8")
    return tmp_path


def test_fixed_prefix_preserves_user_order_without_replacements(tmp_path: Path) -> None:
    from src.data_aggregate.utils.sec_thirteen.superinvestor_roster import load_manager_cohorts

    roster = {"0000000001", "0000000002", "0000000003"}
    config_dir = _config(tmp_path, ["0000000003", "0000000001"])
    assert load_manager_cohorts(config_dir, roster, 1) == {"": frozenset(roster), "topk": frozenset({"0000000003"})}
    assert load_manager_cohorts(config_dir, roster, 2)["topk"] == frozenset({"0000000003", "0000000001"})
    print("SANITY: one ordered raw-CIK list fixes the top-K prefix; no manager is replaced from observed books.")


@pytest.mark.parametrize("managers", [[], {}, [1], ["1"], ["0000000001", "0000000001"], ["0000000009"]])
def test_invalid_manager_list_is_rejected(tmp_path: Path, managers: object) -> None:
    from src.data_aggregate.utils.sec_thirteen.superinvestor_roster import load_manager_cohorts

    with pytest.raises(ValueError):
        load_manager_cohorts(_config(tmp_path, managers), {"0000000001"}, 1)
    print("SANITY: malformed, duplicated or out-of-roster CIK lists fail before feature calculation.")


@pytest.mark.parametrize("k", [0, -1, 2, True, 1.5, "1"])
def test_invalid_k_is_rejected(tmp_path: Path, k: object) -> None:
    from src.data_aggregate.utils.sec_thirteen.superinvestor_roster import load_manager_cohorts

    with pytest.raises(ValueError):
        load_manager_cohorts(_config(tmp_path, ["0000000001"]), {"0000000001"}, cast(int, k))
    print("SANITY: K must be a positive integer bounded by the explicit configured list.")
