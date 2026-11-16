"""Manager profile leakage (AC-101): a profile row never depends on a report period after its own.

The build is replayed on a real sample with every `sec13f_hr` row of the periods after P removed; every row up to P must be identical.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from omegaconf import DictConfig

from src.data_aggregate.utils.manager_profile.profile import profile_rows
from src.utils.config import read_config
from tests.data_aggregate.test_manager_profile import live_reference, live_store, load_raw, run_quarters, sample_ciks

ROOT = Path(__file__).resolve().parents[2]
PERIODS = ["2008-12-31", "2024-06-30"]


@pytest.mark.parametrize("cut", PERIODS)
def test_rows_up_to_a_period_are_identical_without_later_periods(cut: str) -> None:
    cfg: DictConfig = read_config(str(ROOT / "configs")).build_cube.manager_profile
    store = live_store()
    end = pd.Timestamp(cut)
    quarters = [(end + pd.offsets.QuarterEnd(k)).strftime("%Y-%m-%d") for k in range(-4, 3)]
    raw = load_raw(store, sample_ciks(store, cut, 80), quarters)
    reference = live_reference(store)

    full = profile_rows(run_quarters(raw, reference, cfg), cfg)
    cut_raw = {period: frame for period, frame in raw.items() if pd.Timestamp(period) <= end}
    replay = profile_rows(run_quarters(cut_raw, reference, cfg), cfg)

    expected = full[full["period"] <= end].sort_values(["cik", "period"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(replay.sort_values(["cik", "period"]).reset_index(drop=True), expected, check_exact=True)
    later = full[full["period"] > end]
    assert len(raw) > len(cut_raw) and not later.empty, "the full run does see later periods"
    row = expected[expected["period"] == end]
    print(f"\n=== SANITY CHECK: leakage, cut at {cut} ===")
    print(
        f"{len(row)} managers at the cut period and {len(expected)} rows up to it are identical without {len(raw) - len(cut_raw)} later periods ({len(later)} later rows)."
    )
