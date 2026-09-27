"""Known-truth fiscal-period matching and source-age projection contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.common.pit import (
    fiscal_change_to_daily,
    fiscal_prior_to_daily,
    fiscal_prior_values,
    fundamentals_to_daily,
)


def _history() -> pd.DataFrame:
    """Irregular rows and amendments that make a filing-count lag wrong."""
    return pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "as_of": "2023-05-01",
                "fiscal_end": "2023-03-31",
                "value": 100.0,
            },
            {
                "ticker": "AAA",
                "as_of": "2023-08-01",
                "fiscal_end": "2023-06-30",
                "value": 130.0,
            },
            {
                "ticker": "AAA",
                "as_of": "2024-05-01",
                "fiscal_end": "2024-03-30",
                "value": 150.0,
            },
            # This amendment is published after the initial 2024 row. It must not alter that
            # row's predecessor, but it is available to the later 2024 amendment.
            {
                "ticker": "AAA",
                "as_of": "2024-06-01",
                "fiscal_end": "2023-03-31",
                "value": 120.0,
            },
            {
                "ticker": "AAA",
                "as_of": "2024-07-01",
                "fiscal_end": "2024-03-30",
                "value": 180.0,
            },
            {
                "ticker": "BBB",
                "as_of": "2023-05-01",
                "fiscal_end": "2023-04-01",
                "value": 40.0,
            },
            {
                "ticker": "BBB",
                "as_of": "2024-05-15",
                "fiscal_end": "2024-04-15",
                "value": 60.0,
            },
            {
                "ticker": "MISS",
                "as_of": "2024-05-15",
                "fiscal_end": None,
                "value": 10.0,
            },
        ]
    )


def test_fiscal_prior_matches_periods_not_filing_count_and_respects_amendments():
    history = _history()

    prior = fiscal_prior_values(history, "value", years=1, tolerance_days=45)

    # Initial 2024 row sees only the original prior-year filing, not the future amendment.
    assert prior.iloc[2] == pytest.approx(100.0)
    # Later amendment sees the amended prior-year value that is now public.
    assert prior.iloc[4] == pytest.approx(120.0)
    # BBB's 15-day fiscal drift is inside the 45-day window.
    assert prior.iloc[6] == pytest.approx(40.0)
    # Missing fiscal dates cannot be guessed from publication-row position.
    assert pd.isna(prior.iloc[7])
    # A filing-count predecessor for row 4 would be row 0 or row 1 depending on sort/order;
    # the matcher deterministically chooses the amended same fiscal period instead.
    assert prior.iloc[4] != history["value"].shift(4).iloc[4]

    print("\n=== SANITY CHECK: fiscal predecessor matching ===")
    print(
        "  initial 2024 filing -> original 2023 value 100; later 2024 amendment -> "
        "published 2023 amendment 120; 15-day fiscal drift matched; missing fiscal_end "
        "stayed null. Validated."
    )


def test_fiscal_change_becomes_visible_only_at_the_current_publication_date():
    history = _history()
    idx = pd.date_range("2024-04-29", "2024-05-03")

    daily = fiscal_change_to_daily(history, "value", idx, kind="pct")
    prior = fiscal_prior_to_daily(history, "value", idx)

    assert pd.isna(daily.loc[pd.Timestamp("2024-04-30"), "AAA"])
    assert daily.loc[pd.Timestamp("2024-05-01"), "AAA"] == pytest.approx(0.5)
    assert daily.loc[pd.Timestamp("2024-05-03"), "AAA"] == pytest.approx(0.5)
    assert pd.isna(prior.loc[pd.Timestamp("2024-04-30"), "AAA"])
    assert prior.loc[pd.Timestamp("2024-05-01"), "AAA"] == pytest.approx(100.0)

    print("\n=== SANITY CHECK: fiscal change publication clock ===")
    print(
        "  2024-vs-2023 growth is null through 2024-04-30 and becomes +50% exactly on the "
        "2024-05-01 filing date; the future 2023 amendment is not used. Validated."
    )


@pytest.mark.parametrize("max_age_days", [185, 460])
def test_daily_projection_expires_after_the_exact_source_age_boundary(
    max_age_days: int,
) -> None:
    filed = pd.Timestamp("2020-01-01")
    idx = pd.date_range(filed - pd.Timedelta(days=1), filed + pd.Timedelta(days=max_age_days + 1))
    history = pd.DataFrame(
        [
            {"ticker": "AAA", "as_of": filed, "value": 7.0},
            # A later null observation is not a producing observation and must not reset age.
            {"ticker": "AAA", "as_of": filed + pd.Timedelta(days=30), "value": np.nan},
        ]
    )

    capped = fundamentals_to_daily(history, "value", idx, max_age_days=max_age_days)
    uncapped = fundamentals_to_daily(history, "value", idx)

    assert pd.isna(capped.loc[filed - pd.Timedelta(days=1), "AAA"])
    assert capped.loc[filed + pd.Timedelta(days=max_age_days), "AAA"] == pytest.approx(7.0)
    assert pd.isna(capped.loc[filed + pd.Timedelta(days=max_age_days + 1), "AAA"])
    assert uncapped.loc[filed + pd.Timedelta(days=max_age_days + 1), "AAA"] == pytest.approx(7.0)

    print(f"\n=== SANITY CHECK: {max_age_days}-day source-age boundary ===")
    print(
        f"  value is absent before publication, alive through day {max_age_days}, null on day "
        f"{max_age_days + 1}; a later null row does not reset the clock and the default "
        "uncapped projection is unchanged. Validated."
    )
