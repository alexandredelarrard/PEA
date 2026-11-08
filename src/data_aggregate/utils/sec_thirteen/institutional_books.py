"""Bounded raw preparation and compact public-book history for the broad 13F panel."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.utils.institutional_vintages import full_13f_report, prepare_book_vintages, resolve_prepared_books

logger = logging.getLogger(__name__)


def compact_institutional_books(df_books: pd.DataFrame) -> pd.DataFrame:
    """Keep mapped amounts, unknown/zero rows and the positive-share CUSIP identity, one row per (period, manager,
    company) as stored: `sec13f_hr` holds one row per manager x company x accession in canonical-class shares (U-67)."""
    numeric = ["shares", "value_usd", "call_value", "put_value"]
    keys = ["period", "cik", "ticker"]
    df_mapped = df_books.loc[df_books.ticker.notna()]
    duplicated = df_mapped.duplicated(keys)
    if duplicated.any():
        sample = df_mapped.loc[duplicated, keys].head(3).to_dict("records")
        raise ValueError(f"13F books hold {int(duplicated.sum())} duplicate manager x company row(s), e.g. {sample}: run `identity --only propagate`")
    df_totals = df_mapped.set_index(keys)[numeric]
    df_totals["_cusips"] = pd.Series(df_mapped["cusip"].where(df_mapped["shares"].gt(0)).to_numpy(), index=df_totals.index, dtype="object")
    return df_totals


def _single_version_books(df_manifest: pd.DataFrame) -> pd.Series:
    """Manifest rows that are their book's only version, a public complete original/restatement."""
    versions = df_manifest.groupby(["cik", "period"])["accession_number"].transform("size")
    return versions.eq(1) & df_manifest["amendment_type"].isin(["original", "restatement"]) & df_manifest["public_at"].notna()


def _single_version_events(df_single: pd.DataFrame, full_report: pd.Series, df_compact: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """States of single-version books, one event per book on its public day, and their compact rows ordered by book
    with each state's row span; the resolver's result for a lone original/restatement."""
    restricted = df_single["restricted"].astype(bool).to_numpy()
    eligible = ~restricted & full_report.to_numpy(dtype=bool)
    df_states = pd.DataFrame(
        {
            "cik": df_single["cik"].to_numpy(),
            "period": df_single["period"].to_numpy(),
            "public_at": df_single["public_at"].to_numpy(),
            "last_observed_at": df_single["public_at"].to_numpy(),
            "accession_number": df_single["accession_number"].to_numpy(),
            "book_known": True,
            "zero_eligible": eligible,
            "comparison_eligible": eligible,
            "restricted": restricted,
            "resolution_status": "resolved",
            "_day": df_single["public_at"].to_numpy(),
        }
    )

    # Compact rows ordered by their book's state, each state spanning its own rows.
    books = pd.MultiIndex.from_frame(df_states[["cik", "period"]])
    codes = books.get_indexer(pd.MultiIndex.from_arrays([df_compact.index.get_level_values("cik"), df_compact.index.get_level_values("period")]))
    order = np.argsort(codes, kind="stable")
    df_compact, codes = df_compact.iloc[order], codes[order]
    df_states["_start"] = np.searchsorted(codes, np.arange(len(df_states)), side="left").astype("int64")
    df_states["_stop"] = np.searchsorted(codes, np.arange(len(df_states)), side="right").astype("int64")
    return df_compact, df_states


def _book_codes(df_manifest: pd.DataFrame, df_holdings: pd.DataFrame) -> tuple[pd.MultiIndex, NDArray[np.intp], NDArray[np.intp]]:
    """Number manager-quarters in first-appearance order; a position takes its quarter's code only under its own accession."""
    books = pd.MultiIndex.from_frame(df_manifest[["cik", "period"]]).unique()
    codes = books.get_indexer(pd.MultiIndex.from_frame(df_manifest[["cik", "period"]]))
    holding_codes = books.get_indexer(pd.MultiIndex.from_frame(df_holdings[["cik", "period"]]))
    accession_codes = df_holdings["accession_number"].map(pd.Series(codes, index=df_manifest["accession_number"].to_numpy()))
    holding_codes = np.where(accession_codes.to_numpy(dtype=float) == holding_codes, holding_codes, -1)
    return books, codes, holding_codes


def _resolve_public_days(df_manifest: pd.DataFrame, df_holdings: pd.DataFrame) -> tuple[list[pd.DataFrame], list[pd.DataFrame], pd.MultiIndex]:
    """Resolve and compact, once per public day, every manager-quarter published that day."""
    books, codes, holding_codes = _book_codes(df_manifest, df_holdings)
    df_events = pd.DataFrame({"_book": codes, "_day": df_manifest["public_at"].to_numpy()}).dropna().drop_duplicates()
    df_positions = df_holdings.drop(columns=[column for column in df_holdings if column.startswith("value_basis_") or column == "filing_date"])
    frames: list[pd.DataFrame] = []
    states: list[pd.DataFrame] = []

    # A quarter's book at a day depends only on its own versions, so one resolver call serves all quarters of that day;
    # provenance columns never reach the compact amounts, so they stay out of the per-accession resolver copies.
    for day, published in df_events.groupby("_day", sort=True)["_book"]:
        member = np.zeros(len(books) + 1, dtype=bool)  # the last slot is the unowned code -1
        member[published.to_numpy()] = True
        df_book, df_state = resolve_prepared_books(df_manifest.loc[member[codes]], df_positions.loc[member[holding_codes]], day)
        frames.append(compact_institutional_books(df_book))
        states.append(df_state.assign(_day=day))
    return frames, states, books


def _compact_batch_events(df_manifest: pd.DataFrame, df_holdings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Order compact public replacements by manager-quarter, then public day, with each state's row span."""
    frames, states, books = _resolve_public_days(df_manifest, df_holdings)
    if not states or all(state.empty for state in states):
        return pd.DataFrame(), pd.DataFrame({"_start": pd.Series(dtype="int64"), "_stop": pd.Series(dtype="int64")})

    # Rows keep their within-day order; the stable sort groups them by quarter first-appearance, then day.
    df_compact = pd.concat(frames)
    df_states = pd.concat(states, ignore_index=True)
    day_numbers = np.arange(len(frames))
    compact_books = books.get_indexer(
        pd.MultiIndex.from_arrays([df_compact.index.get_level_values("cik"), df_compact.index.get_level_values("period")])
    )
    compact_keys = compact_books * len(frames) + np.repeat(day_numbers, [len(frame) for frame in frames])
    state_books = books.get_indexer(pd.MultiIndex.from_frame(df_states[["cik", "period"]]))
    state_keys = state_books * len(frames) + np.repeat(day_numbers, [len(state) for state in states])
    order = np.argsort(compact_keys, kind="stable")
    df_compact, compact_keys = df_compact.iloc[order], compact_keys[order]
    df_states = df_states.iloc[np.argsort(state_keys, kind="stable")].reset_index(drop=True)
    state_keys = np.sort(state_keys, kind="stable")

    # Each state spans its own rows in the ordered compact frame.
    df_states["_start"] = np.searchsorted(compact_keys, state_keys, side="left").astype("int64")
    df_states["_stop"] = np.searchsorted(compact_keys, state_keys, side="right").astype("int64")
    return df_compact, df_states


def prepare_institutional_book_history(
    batches: Iterator[tuple[pd.DataFrame, pd.DataFrame]], trading_index: pd.DatetimeIndex, *, settle_days: int
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Retain only compact public events of the cleaned complete-book batches: single-version books in one pass,
    amended books through the shared resolver."""
    frames: list[pd.DataFrame] = []
    states: list[pd.DataFrame] = []
    offset = 0
    filings_count = 0
    raw_rows = 0

    for df_filings, df_holdings in batches:
        filings_count += len(df_filings)
        raw_rows += len(df_holdings)
        df_manifest = prepare_book_vintages(df_filings, df_holdings, trading_index, settle_days)
        single = _single_version_books(df_manifest)
        df_single = df_manifest.loc[single]
        single_rows = df_holdings["accession_number"].isin(set(df_single["accession_number"]))
        df_single_compact = compact_institutional_books(df_holdings.loc[single_rows])
        parts = (
            _single_version_events(df_single, df_single["report_type"].map(full_13f_report), df_single_compact),
            _compact_batch_events(df_manifest.loc[~single], df_holdings.loc[~single_rows]),
        )
        del df_filings, df_holdings, df_manifest, df_single, df_single_compact
        for df_compact, df_states in parts:
            frames.append(df_compact)
            states.append(df_states.assign(_start=df_states["_start"] + offset, _stop=df_states["_stop"] + offset))
            offset += len(df_compact)

    if not filings_count:
        logger.warning("No complete immutable 13F filings in scope; recover originals/amendments. Latest-only rows cannot supply PIT books.")
        return None
    frames = [frame for frame, state in zip(frames, states, strict=True) if not state.empty]
    states = [state for state in states if not state.empty]
    df_compact = pd.concat(frames) if frames else pd.DataFrame()
    df_states = pd.concat(states, ignore_index=True) if states else pd.DataFrame()
    logger.info(
        "Immutable broad 13F: %s accessions / %s raw rows reduced to %s mapped rows / %s dated book states.",
        filings_count,
        raw_rows,
        len(df_compact),
        len(df_states),
    )
    return df_compact, df_states
