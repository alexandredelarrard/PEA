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


def _book_codes(df_manifest: pd.DataFrame, df_holdings: pd.DataFrame) -> tuple[pd.MultiIndex, NDArray[np.intp], NDArray[np.bool_]]:
    """Number manager-quarters in first-appearance order; a position belongs to its quarter only under its own accession."""
    books = pd.MultiIndex.from_frame(df_manifest[["cik", "period"]]).unique()
    codes = books.get_indexer(pd.MultiIndex.from_frame(df_manifest[["cik", "period"]]))
    holding_codes = books.get_indexer(pd.MultiIndex.from_frame(df_holdings[["cik", "period"]]))
    accession_codes = df_holdings["accession_number"].map(pd.Series(codes, index=df_manifest["accession_number"].to_numpy()))
    return books, codes, accession_codes.to_numpy(dtype=float) == holding_codes


def _order_batch_events(frames: list[pd.DataFrame], states: list[pd.DataFrame], books: pd.MultiIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Order the amended books' compact public replacements by manager-quarter, then public day, with each state's
    row span."""
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


def _concat_events(parts: list[tuple[pd.DataFrame, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate the batches' compact rows and states, shifting each state's row span to the global frame."""
    offsets = np.cumsum([0, *(len(df_compact) for df_compact, _ in parts)])
    frames = [df_compact for df_compact, df_states in parts if not df_states.empty]
    states = [
        df_states.assign(_start=df_states["_start"] + offset, _stop=df_states["_stop"] + offset)
        for (_, df_states), offset in zip(parts, offsets[:-1].tolist(), strict=True)
        if not df_states.empty
    ]
    df_compact = pd.concat(frames) if frames else pd.DataFrame()
    df_states = pd.concat(states, ignore_index=True) if states else pd.DataFrame()
    return df_compact, df_states


def prepare_institutional_book_history(
    batches: Iterator[tuple[pd.DataFrame, pd.DataFrame]], trading_index: pd.DatetimeIndex, *, settle_days: int
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Retain only compact public events of the cleaned complete-book batches: single-version books in one pass,
    amended books through the shared resolver."""
    parts: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    filings_count = 0
    raw_rows = 0

    for df_filings, df_holdings in batches:
        # Publication facts; a single-version book is its own resolved state on its public day.
        filings_count += len(df_filings)
        raw_rows += len(df_holdings)
        df_manifest = prepare_book_vintages(df_filings, df_holdings, trading_index, settle_days)
        single = _single_version_books(df_manifest)
        df_single = df_manifest.loc[single]
        single_rows = df_holdings["accession_number"].isin(set(df_single["accession_number"]))
        df_single_compact = compact_institutional_books(df_holdings.loc[single_rows])
        parts.append(_single_version_events(df_single, df_single["report_type"].map(full_13f_report), df_single_compact))

        # Amended books: a quarter's book at a day depends only on its own versions, so one resolver call serves the
        # n-th public day of every quarter, each cut to its own day; provenance columns never reach the compact amounts.
        df_amended, df_amended_rows = df_manifest.loc[~single], df_holdings.loc[~single_rows]
        books, codes, owned = _book_codes(df_amended, df_amended_rows)
        df_positions = df_amended_rows.loc[owned, ~df_amended_rows.columns.str.match("value_basis_|filing_date|class_evidence")]
        df_events = pd.DataFrame({"_book": codes, "_day": df_amended["public_at"].to_numpy()}).dropna().drop_duplicates().sort_values("_day")
        frames: list[pd.DataFrame] = []
        states: list[pd.DataFrame] = []
        for _, df_round in df_events.groupby(df_events.groupby("_book").cumcount(), sort=True):
            cutoff = df_round.set_index("_book")["_day"].reindex(range(len(books))).to_numpy()
            df_book, df_state = resolve_prepared_books(
                df_amended.loc[df_amended["public_at"].le(cutoff[codes])], df_positions, df_round["_day"].max()
            )
            frames.append(compact_institutional_books(df_book))
            states.append(df_state.assign(_day=cutoff[books.get_indexer(pd.MultiIndex.from_frame(df_state[["cik", "period"]]))]))
        parts.append(_order_batch_events(frames, states, books))
        del df_filings, df_holdings, df_manifest, df_single, df_single_compact, df_amended_rows, df_positions

    # One frame of compact rows with global row spans.
    if not filings_count:
        logger.warning("No complete immutable 13F filings in scope; recover originals/amendments. Latest-only rows cannot supply PIT books.")
        return None
    df_compact, df_states = _concat_events(parts)
    logger.info(
        "Immutable broad 13F: %s accessions / %s raw rows reduced to %s mapped rows / %s dated book states.",
        filings_count,
        raw_rows,
        len(df_compact),
        len(df_states),
    )
    return df_compact, df_states
