"""One execution anchor and mode are established before constructing pipeline state."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pandas as pd

from src.context import get_config_context
from src.data_extract.utils.common.sessions import last_completed_session
from src.utils.window import WindowRun


def test_config_context_resolves_one_market_run_before_construction() -> None:
    instant = datetime(2026, 11, 16, 2, 30, tzinfo=UTC)
    with patch("src.context.Context") as constructor:
        config, context = get_config_context("configs", use_cache=False, save=False, as_of=instant, full=True)
    run = constructor.call_args.kwargs["run"]
    assert run.as_of == pd.Timestamp("2026-11-15") and run.mode == "full"
    assert run.instant == pd.Timestamp(instant)
    assert context is constructor.return_value and config.seed == 4325
    print("SANITY:02:30UTC is the prior New York date; one explicit full-run anchor reaches Context.")


def test_explicit_date_remains_stable_across_entry_calls() -> None:
    day = datetime(2026, 11, 10)
    with patch("src.context.Context") as constructor:
        get_config_context("configs", use_cache=False, save=False, as_of=day)
        first = constructor.call_args.kwargs["run"]
        get_config_context("configs", use_cache=False, save=False, as_of=day)
        second = constructor.call_args.kwargs["run"]
    assert first == second and first.as_of == pd.Timestamp("2026-11-10")
    assert first.mode == "incremental"
    print("SANITY: an explicit market date determines every entry call independently of wall-clock time.")


def test_run_preserves_the_entry_instant_for_settled_price_cutoffs() -> None:
    before = WindowRun("incremental", "before close", pd.Timestamp("2026-11-10 15:59", tz="America/New_York"))
    after = WindowRun("incremental", "after close", pd.Timestamp("2026-11-10 16:01", tz="America/New_York"))
    assert before.as_of == after.as_of == pd.Timestamp("2026-11-10")
    assert last_completed_session(before.instant) == pd.Timestamp("2026-11-09")
    assert last_completed_session(after.instant) == pd.Timestamp("2026-11-10")
    print("SANITY: one entry instant preserves pre/post-close behavior while every table shares its market date.")
