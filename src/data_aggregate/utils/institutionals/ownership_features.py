"""Schedule 13D (active) and Schedule 13G (passive) filing counts on the target ticker.

Each feature counts the filings (one per `(ticker, accession_number)`, initial filings and amendments alike) stamped
on the ticker over the trailing `COUNT_WINDOW` trading days. Only fields present in every era are used, so the count
means the same thing across the whole history: no `date_of_event` (structured filings only) and no filer identity
(CIKs change with registrations). A filing counts from the first session on or after its filing date.

The filing volume itself is not stationary (13G holders multiply through 2016, the 2024 SEC rule change cuts routine
amendments), so each count is standardised across the universe on every date and clipped at `COUNT_CLIP`: the feature
is how unusual a ticker's filing flow is today, not a raw level. A covered, complete source with no filing in the
window is a count of 0 before standardising; an uncovered cell is NaN and sits outside the cross-section.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.availability import InstitutionalAvailability
from src.data_aggregate.utils.common.errors import _absent, _empty_panel
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.common.xs import xs_z
from src.data_aggregate.utils.price.frames import PriceFrames
from src.data_store.schema import Table, Tables

#: Trailing window in trading days (about six calendar months).
COUNT_WINDOW = 126

#: Clip of the cross-sectional z-score of a count.
COUNT_CLIP = 5.0

EMISSION: dict[str, str] = {
    "ic_act_13d_count_126d_z": "raw",
    "ic_bo_13g_count_126d_z": "raw",
}


def _filings(df_schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per `(ticker, accession_number)` with its filing date."""
    df_filings = df_schedule.assign(filing_date=pd.to_datetime(df_schedule["filing_date"], errors="coerce").dt.normalize())
    df_filings = df_filings.dropna(subset=["ticker", "accession_number", "filing_date"])
    return df_filings.groupby(["ticker", "accession_number"], as_index=False)["filing_date"].min()


def _filing_counts(df_filings: pd.DataFrame, idx: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """Filings per ticker in the trailing `COUNT_WINDOW` sessions; each lands on the first session on or after its filing date."""
    arrivals = np.zeros((len(idx), len(columns)))
    rows = idx.searchsorted(df_filings["filing_date"].to_numpy(), side="left")
    cols = columns.get_indexer(pd.Index(df_filings["ticker"]))
    keep = (rows < len(idx)) & (cols >= 0)
    np.add.at(arrivals, (rows[keep], cols[keep]), 1.0)
    cumulative = np.vstack([np.zeros((1, len(columns))), arrivals.cumsum(axis=0)])
    window_start = np.maximum(np.arange(len(idx)) + 1 - COUNT_WINDOW, 0)
    return pd.DataFrame(cumulative[1:] - cumulative[window_start], index=idx, columns=columns)


def _coverage_mask(
    frames: PriceFrames, idx: pd.DatetimeIndex, columns: pd.Index, source_start: pd.Timestamp, complete_through: pd.Timestamp
) -> pd.DataFrame:
    """Cells where the source is complete (from its start to its newest filing) and the ticker is listed."""
    listed = (
        frames.close_split.reindex(index=idx, columns=columns).notna()
        if frames.close_split is not None and not frames.close_split.empty
        else pd.DataFrame(True, index=idx, columns=columns)
    )
    return InstitutionalAvailability.combine(
        InstitutionalAvailability.date_mask(idx, columns, source_start),
        InstitutionalAvailability.through_mask(idx, columns, complete_through),
        listed,
    )


def build_ownership_feature_panel(
    frames: PriceFrames,
    sec_13d: pd.DataFrame | None,
    sec_13g: pd.DataFrame | None,
    *,
    availability: InstitutionalAvailability | None = None,
    complete_through_13d: pd.Timestamp | None = None,
    complete_through_13g: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Point-in-time 13D and 13G filing counts per ticker, masked by each schedule's source coverage."""
    idx = pd.DatetimeIndex(frames.trading_index).normalize().unique().sort_values()
    columns = pd.Index(frames.universe, name="ticker")
    schedules: tuple[tuple[str, pd.DataFrame | None, Table, pd.Timestamp | None], ...] = (
        ("ic_act_13d_count_126d_z", sec_13d, Tables.sec_13d, complete_through_13d),
        ("ic_bo_13g_count_126d_z", sec_13g, Tables.sec_13g, complete_through_13g),
    )

    # Count each usable schedule's filings, one leg per schedule.
    fields: dict[str, pd.DataFrame] = {}
    for name, df_schedule, table, complete_through in schedules:
        if _absent(df_schedule, {"ticker", "accession_number", "filing_date"}) or df_schedule is None:
            continue
        df_filings = _filings(df_schedule)
        if df_filings.empty:
            continue
        df_counts = _filing_counts(df_filings, idx, columns)

        # Outside the proven source coverage an absent filing is unknown, not zero.
        if complete_through is not None and pd.notna(complete_through):
            source_start = availability.source_date(table) if availability is not None else df_filings["filing_date"].min()
            df_counts = df_counts.where(_coverage_mask(frames, idx, columns, pd.Timestamp(source_start), pd.Timestamp(complete_through)))

        # Standardise across the universe each date so regime changes in filing volume cancel.
        fields[name] = xs_z(df_counts, clip=COUNT_CLIP, zero_sd_to_nan=True)
    if not fields:
        return _empty_panel()

    return build_peer_relative_panel(fields, frames.peers, emission={name: EMISSION[name] for name in fields}, availability=frames.availability)
