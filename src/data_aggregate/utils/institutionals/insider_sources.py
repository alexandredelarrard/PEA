"""Canonical quarterly-ZIP and daily-EDGAR insider source selection."""

from __future__ import annotations

import pandas as pd


def as_quarter(value: object) -> pd.Period | None:
    """Normalize a configured or stored quarter label."""
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    try:
        return pd.Period(str(value).upper(), freq="Q")
    except (TypeError, ValueError):
        return None


def overlay_insider_sources(
    bulk: pd.DataFrame | None,
    live: pd.DataFrame | None,
    *,
    bulk_authoritative_through: object = None,
) -> pd.DataFrame | None:
    """Overlay at accession grain, with live winning until its quarter is promoted.

    A quarterly ZIP is not allowed to replace an already staged EDGAR filing merely
    because it arrived. `bulk_authoritative_through` is advanced only after the retained
    completed-quarter parity report passes. Bulk-only and live-only accessions remain
    usable on either side; rows from one accession are never mixed across sources.
    """
    frames = [frame for frame in (bulk, live) if frame is not None and not frame.empty]
    if not frames:
        return None
    if bulk is None or bulk.empty:
        return live.copy() if live is not None else None
    if live is None or live.empty:
        return bulk.copy()

    bulk_accessions = set(bulk["accession_number"].dropna().astype(str))
    cutoff = as_quarter(bulk_authoritative_through)
    live_quarters = pd.to_datetime(live["filing_date"], errors="coerce").dt.to_period("Q")
    approved = live_quarters.le(cutoff) if cutoff is not None else pd.Series(False, index=live.index)
    overlaps_bulk = live["accession_number"].astype(str).isin(bulk_accessions)
    live_winning_accessions = set(live.loc[overlaps_bulk & ~approved, "accession_number"].dropna().astype(str))
    kept_bulk = bulk[~bulk["accession_number"].astype(str).isin(live_winning_accessions)]
    kept_live = live[~overlaps_bulk | live["accession_number"].astype(str).isin(live_winning_accessions)]
    return pd.concat([kept_bulk, kept_live], ignore_index=True, sort=False)


def bulk_complete_through(
    latest_quarter: object,
    bulk: pd.DataFrame | None,
    live: pd.DataFrame | None,
    *,
    bulk_authoritative_through: object = None,
) -> pd.Timestamp | None:
    """Cap bulk completeness before the first overlapping unpromoted live quarter."""
    latest = as_quarter(latest_quarter)
    if latest is None:
        return None
    frontier = latest.end_time.normalize()
    if bulk is None or bulk.empty or live is None or live.empty:
        return frontier

    bulk_accessions = set(bulk["accession_number"].dropna().astype(str))
    overlapping = live[live["accession_number"].astype(str).isin(bulk_accessions)]
    if overlapping.empty:
        return frontier
    cutoff = as_quarter(bulk_authoritative_through)
    overlap_quarters = pd.to_datetime(overlapping["filing_date"], errors="coerce").dt.to_period("Q")
    unapproved = overlap_quarters[overlap_quarters.notna()]
    if cutoff is not None:
        unapproved = unapproved[unapproved > cutoff]
    if unapproved.empty:
        return frontier
    conservative = unapproved.min().start_time.normalize() - pd.Timedelta(days=1)
    return min(frontier, conservative)
