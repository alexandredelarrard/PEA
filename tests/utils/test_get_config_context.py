"""`get_config_context` on a missing config directory: logs the error and exits 1."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.context import get_config_context


def test_missing_config_dir_logs_error_and_exits_1(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    missing = tmp_path / "no_such_configs"

    with caplog.at_level(logging.ERROR, logger="src.context"), pytest.raises(SystemExit) as exc:
        get_config_context(str(missing), use_cache=False, save=False)

    assert exc.value.code == 1
    errors = [r for r in caplog.records if r.name == "src.context" and r.levelno == logging.ERROR]
    assert [r.getMessage() for r in errors] == [f"configuration file {missing} not found "]

    print("\n=== SANITY CHECK: get_config_context on a missing config dir ===")
    print(f"  SystemExit({exc.value.code}); one ERROR record from src.context: {errors[0].getMessage()!r}")
