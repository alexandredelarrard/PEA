"""Schedule 13D (active) and Schedule 13G (passive) filing counts on the target ticker.

Each feature counts the filings (one per `(ticker, accession_number)`, initial filings and amendments alike) about
the ticker over the trailing `COUNT_WINDOW` trading days. Only fields present in every era are used, so the count
means the same thing across the whole history: no `date_of_event` (structured filings only). The tables key a filing
by the ticker's CIK whether the ticker is its subject or its filer, so `_subject_side` drops the filings the ticker's
own entity filed about others (an approximation for markers: their only evidence is the accession prefix). A filing counts from its publication session: its configured source-publication date, rolled to the first session on or after it.

The filing volume itself is not stationary (13G holders multiply through 2016, the 2024 SEC rule change cuts routine
amendments), so each count is standardised across the universe on every date and clipped at `COUNT_CLIP`: the feature
is how unusual a ticker's filing flow is today, not a raw level. A covered, complete source with no filing in the
window is a count of 0 before standardising; an uncovered cell is NaN and sits outside the cross-section.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.data_aggregate.utils.common.availability import InstitutionalAvailability
from src.data_aggregate.utils.common.errors import _absent, _empty_panel
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.common.xs import xs_z
from src.data_aggregate.utils.price.frames import PriceFrames
from src.data_store.schema import Table, Tables
from src.utils.publication import publication_dates

#: Trailing window in trading days (about six calendar months).
COUNT_WINDOW = 126

#: Clip of the cross-sectional z-score of a count.
COUNT_CLIP = 5.0

#: A filer prefix counts as one institution's own submitter from this many accessions under one ticker ...
CONCENTRATED_MIN_FILINGS = 100

#: ... holding at least this share of all the prefix's accessions in the universe.
CONCENTRATED_SHARE = 0.5

#: A ticker whose surviving accessions are at least this share marker-only, with `CONCENTRATED_MIN_FILINGS` of them, is itself an
#: institution filing through shared agents: its marker-only accessions cannot be told from its own filings and are dropped.
MARKER_HEAVY_SHARE = 0.8

#: The columns the subject test reads beyond the counting columns.
SUBJECT_COLUMNS = ("ticker", "accession_number", "filing_date", "rp_seq", "cusip", "reporting_person_cik")

EMISSION: dict[str, str] = {
    "ic_act_13d_count_126d_z": "raw",
    "ic_bo_13g_count_126d_z": "raw",
}


@dataclass(frozen=True)
class ScheduleSource:
    """One schedule's raw filings and independently observed completeness frontier."""

    table: Table
    filings: pd.DataFrame | None
    complete_through: pd.Timestamp | None = None


@dataclass(frozen=True)
class SubjectIdentity:
    """Which `(ticker, CIK)` and `(ticker, CUSIP8)` pairs belong to the ticker's own lineage."""

    ciks: pd.MultiIndex
    cusips: pd.MultiIndex

    @classmethod
    def from_tables(cls, lineage: pd.DataFrame, lines: pd.DataFrame) -> SubjectIdentity:
        """Lineage CIKs from `universe_cik_symbol` and the non-excluded CUSIPs of `universe_cusip`."""
        windows = lineage.loc[lineage["role"].isin(("cik_window", "cik_event")), ["canonical_ticker", "cik"]].drop_duplicates()
        kept = lines.loc[lines["cusip"].notna() & lines["lineage_role"].ne("excluded"), ["canonical_company", "cusip"]]
        return cls(
            ciks=pd.MultiIndex.from_frame(windows),
            cusips=pd.MultiIndex.from_arrays([kept["canonical_company"], kept["cusip"].str[:8]]).drop_duplicates(),
        )


def _filings(df_schedule: pd.DataFrame, identity: SubjectIdentity | None = None) -> pd.DataFrame:
    """One row per `(ticker, accession_number)` about the ticker, with its filing date."""
    df_filings = df_schedule.assign(filing_date=pd.to_datetime(df_schedule["filing_date"], errors="coerce").dt.normalize())
    df_filings = df_filings.dropna(subset=["ticker", "accession_number", "filing_date"])
    if identity is not None and {"rp_seq", "cusip", "reporting_person_cik"} <= set(df_filings.columns):
        df_filings = df_filings.loc[_subject_side(df_filings, identity)]
    return df_filings.groupby(["ticker", "accession_number"], as_index=False)["filing_date"].min()


def _subject_side(df_filings: pd.DataFrame, identity: SubjectIdentity) -> pd.Series:
    """Rows of accessions about the ticker: the table keys filings by the ticker's CIK whether it is the subject or the filer.

    A parsed CUSIP of the ticker proves the subject. Otherwise an accession is the ticker's own filing about others when its number is prefixed
    by, or a reporting person is, one of the ticker's CIKs, when a CUSIP of another universe ticker is its subject, or (marker only) when its
    filer prefix is a concentrated institution-side submitter: at least `CONCENTRATED_MIN_FILINGS` accessions and `CONCENTRATED_SHARE` of the
    prefix's filings sit under this one ticker. A ticker left with `CONCENTRATED_MIN_FILINGS` or more marker-only accessions making up at least
    `MARKER_HEAVY_SHARE` of its surviving accessions is itself a filer through shared agents (an ordinary subject has mostly parsed accessions),
    so its marker-only accessions are dropped and only the parsed subject accessions count.

    The CONCENTRATED/MARKER_HEAVY classifications use full-history statistics, not date-local ones: an identity filter, so a rebuild with more
    data can change the classification, and with it old rows.
    """
    ticker, accession = df_filings["ticker"], df_filings["accession_number"]
    prefix = accession.str[:10]
    cusip8 = df_filings["cusip"].fillna("").str[:8]
    keys = pd.MultiIndex.from_arrays
    parsed = df_filings["rp_seq"].ne(-1)
    row = pd.DataFrame(
        {
            "ticker": ticker,
            "accession": accession,
            "prefix": prefix,
            "parsed": parsed,
            "subject": cusip8.ne("") & keys([ticker, cusip8]).isin(identity.cusips),
            "rp_own": parsed & keys([ticker, df_filings["reporting_person_cik"]]).isin(identity.ciks),
            "prefix_own": keys([ticker, prefix]).isin(identity.ciks),
        }
    )
    flags = row.groupby(["ticker", "accession"], as_index=False).agg(
        prefix=("prefix", "first"), parsed=("parsed", "any"), subject=("subject", "any"), rp_own=("rp_own", "any"), prefix_own=("prefix_own", "any")
    )
    other_subject = flags.groupby("accession")["subject"].transform("sum").sub(flags["subject"].astype(int)).gt(0)
    under_ticker = flags.groupby(["ticker", "prefix"])["accession"].transform("size")
    concentrated = (under_ticker >= CONCENTRATED_MIN_FILINGS) & (
        under_ticker / flags.groupby("prefix")["accession"].transform("size") >= CONCENTRATED_SHARE
    )
    flags["keep"] = flags["subject"] | ~(flags["prefix_own"] | flags["rp_own"] | other_subject | (concentrated & ~flags["parsed"]))
    survivors = flags.loc[flags["keep"]]
    marker_only = (~survivors["parsed"]).groupby(survivors["ticker"]).sum()
    marker_heavy = marker_only.index[
        (marker_only >= CONCENTRATED_MIN_FILINGS) & (marker_only / survivors.groupby("ticker").size() >= MARKER_HEAVY_SHARE)
    ]
    flags["keep"] &= flags["parsed"] | ~flags["ticker"].isin(marker_heavy)
    kept = flags.set_index(["ticker", "accession"])["keep"]
    return pd.Series(kept.reindex(keys([ticker, accession])).to_numpy(), index=df_filings.index)


def _visible_day(day: pd.Timestamp, table: Table, idx: pd.DatetimeIndex, config: DictConfig) -> pd.Timestamp:
    """The actual retained publication session of a source date."""
    return publication_dates(config, table, pd.DataFrame({"filing_date": [day]}), idx).iloc[0]


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
    schedules: Sequence[ScheduleSource],
    config: DictConfig,
    *,
    availability: InstitutionalAvailability | None = None,
    identity: SubjectIdentity | None = None,
) -> pd.DataFrame:
    """Point-in-time schedule filing counts, masked by each source's coverage."""
    idx = pd.DatetimeIndex(frames.trading_index).normalize().unique().sort_values()
    columns = pd.Index(frames.universe, name="ticker")
    if idx.empty:
        return _empty_panel()
    names = {Tables.sec_13d: "ic_act_13d_count_126d_z", Tables.sec_13g: "ic_bo_13g_count_126d_z"}

    # Count each usable schedule's filings, one leg per schedule.
    fields: dict[str, pd.DataFrame] = {}
    for source in schedules:
        table, df_schedule, complete_through = source.table, source.filings, source.complete_through
        name = names[table]
        if _absent(df_schedule, {"ticker", "accession_number", "filing_date"}) or df_schedule is None:
            continue
        df_filings = _filings(df_schedule, identity)
        if df_filings.empty:
            continue

        # Everything downstream reads the publication session of a filing, not its filing date.
        df_filings["filing_date"] = publication_dates(config, table, df_filings, idx)
        df_counts = _filing_counts(df_filings, idx, columns)

        # Outside the proven source coverage an absent filing is unknown, not zero.
        if complete_through is not None and pd.notna(complete_through):
            source_start = (
                _visible_day(availability.source_date(table), table, idx, config) if availability is not None else df_filings["filing_date"].min()
            )
            frontier = _visible_day(complete_through, table, idx, config)
            # A known complete prefix beyond the output calendar covers its last session.
            frontier = idx[-1] if pd.isna(frontier) else frontier
            df_counts = df_counts.where(_coverage_mask(frames, idx, columns, pd.Timestamp(source_start), frontier))

        # Standardise across the universe each date so regime changes in filing volume cancel.
        fields[name] = xs_z(df_counts, clip=COUNT_CLIP, zero_sd_to_nan=True)
    if not fields:
        return _empty_panel()

    return build_peer_relative_panel(fields, frames.peers, emission={name: EMISSION[name] for name in fields}, availability=frames.availability)
