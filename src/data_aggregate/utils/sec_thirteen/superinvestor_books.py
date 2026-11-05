"""Fresh public manager books and immutable, date-aware report-quarter archives."""

from __future__ import annotations

import logging
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Generator, Iterator, Mapping
from contextlib import closing
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd

from src.data_aggregate.utils.sec_thirteen.institutional_vintages import _iter_public_book_cache
from src.utils.institutional_vintages import prepare_book_vintages

type BookKey = tuple[str, pd.Timestamp]

_log = logging.getLogger(__name__)


def prepare_superinvestor_batches(
    packets: Generator[tuple[pd.DataFrame, pd.DataFrame]], trading_index: pd.DatetimeIndex
) -> Generator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Preserve global uniqueness on the source's ordered, complete CIK-quarter batches."""
    accessions: set[str] = set()
    manager: str | None = None
    holding_keys: set[tuple[str, str]] = set()
    peak_keys = 0
    try:
        with closing(packets):
            for df_filings, df_holdings in packets:
                if df_filings["accession_number"].isin(accessions).any():
                    raise ValueError("13F manifest accessions must be unique")
                df_manifest = prepare_book_vintages(df_filings, df_holdings, trading_index, settle_days=0)
                df_keys = df_holdings[["cik", "accession_number", "cusip"]]
                if manager is not None and holding_keys.intersection(
                    df_keys.loc[df_keys.cik.eq(manager), ["accession_number", "cusip"]].itertuples(index=False, name=None)
                ):
                    raise ValueError("13F holding vintages must be unique")
                accessions.update(df_filings["accession_number"])

                # Earlier CIKs cannot recur in the ordered source; only the trailing manager can straddle a batch.
                ciks: set[str] = set(df_filings["cik"]).union(df_holdings["cik"])
                if ciks:
                    trailing = max(ciks)
                    if trailing != manager:
                        holding_keys.clear()
                    manager = trailing
                    holding_keys.update(df_keys.loc[df_keys.cik.eq(manager), ["accession_number", "cusip"]].itertuples(index=False, name=None))
                    peak_keys = max(peak_keys, len(holding_keys))
                yield df_manifest, df_holdings
                del df_filings, df_holdings, df_manifest, df_keys
    finally:
        _log.info("13F identity validation: %s manifest accessions; peak trailing-manager holding keys=%s", len(accessions), peak_keys)


@dataclass(frozen=True, slots=True)
class BookSupport:
    """Resolver evidence determining which positions and absent positions are observable."""

    book_known: bool
    zero_eligible: bool
    comparison_eligible: bool
    restricted: bool
    resolution_status: str


@dataclass(frozen=True, slots=True, eq=False)
class PublicBook:
    """One public resolved revision; position/provenance columns stay untouched."""

    cik: str
    period: pd.Timestamp
    public_at: pd.Timestamp
    first_public_at: pd.Timestamp | None
    df_positions: pd.DataFrame
    support: BookSupport


@dataclass(frozen=True, slots=True)
class PublicBookSnapshot:
    """Current manager candidates and genuine archival books at one public/expiry event."""

    day: pd.Timestamp
    common_period: pd.Timestamp | None
    latest_period_by_manager: Mapping[str, pd.Timestamp]
    level_books: Mapping[str, PublicBook]
    archive: Mapping[BookKey, PublicBook]


@dataclass(frozen=True, slots=True)
class _PublicBookArchive(Mapping[BookKey, PublicBook]):
    """Share revision storage while exposing only each key's latest public version."""

    _revisions: Mapping[BookKey, tuple[PublicBook, ...]]
    day: pd.Timestamp

    def __getitem__(self, key: BookKey) -> PublicBook:
        versions = self._revisions[key]
        position = bisect_right(versions, self.day, key=lambda book: book.public_at)
        if position == 0:
            raise KeyError(key)
        return versions[position - 1]

    def __iter__(self) -> Iterator[BookKey]:
        return (key for key, versions in self._revisions.items() if versions[0].public_at <= self.day)

    def __len__(self) -> int:
        return sum(versions[0].public_at <= self.day for versions in self._revisions.values())


def _resolved_public_book(
    key: BookKey, day: pd.Timestamp, resolved: tuple[pd.DataFrame, pd.DataFrame], first_public_at: pd.Timestamp | None
) -> PublicBook:
    """Wrap the shared resolver output, assigning age only on first known resolution."""
    df_positions, df_state = resolved
    if df_state.empty:
        support = BookSupport(False, False, False, False, "unresolved")
    else:
        state = df_state.iloc[0]
        support = BookSupport(
            bool(state.book_known), bool(state.zero_eligible), bool(state.comparison_eligible), bool(state.restricted), str(state.resolution_status)
        )

    # Feature support cannot start or refresh the source book's age.
    if support.book_known and first_public_at is None:
        first_public_at = day
    return PublicBook(key[0], key[1], day, first_public_at, df_positions, support)


def _collect_public_revisions(packets: Generator[tuple[pd.DataFrame, pd.DataFrame]]) -> Mapping[BookKey, tuple[PublicBook, ...]] | None:
    """Resolve each complete prepared batch into the shared immutable revision archive."""
    revisions: dict[BookKey, list[PublicBook]] = defaultdict(list)
    filings_count = 0
    with closing(packets):
        for df_manifest, df_holdings in packets:
            filings_count += len(df_manifest)
            keys = set(df_manifest[["cik", "period"]].itertuples(index=False, name=None))
            if keys.intersection(revisions):
                raise ValueError("13F prepared batches must contain complete disjoint CIK-quarters")
            for day, _, changed in _iter_public_book_cache(df_manifest, df_holdings, history_quarters=None):
                for key, resolved in changed.items():
                    previous = revisions[key]
                    origin = previous[-1].first_public_at if previous else None
                    previous.append(_resolved_public_book(key, day, resolved, origin))
            del df_manifest, df_holdings
    if not filings_count:
        return None
    ordered = sorted(revisions, key=lambda key: revisions[key][0].public_at)
    return MappingProxyType({key: tuple(revisions[key]) for key in ordered})


def _public_book_schedule(
    revisions: Mapping[BookKey, tuple[PublicBook, ...]], trading_index: pd.DatetimeIndex, max_age_days: int
) -> tuple[dict[pd.Timestamp, list[PublicBook]], pd.DatetimeIndex]:
    """Add first-session expiry events independently of later publications."""
    publications: dict[pd.Timestamp, list[PublicBook]] = defaultdict(list)
    expiry_days: set[pd.Timestamp] = set()
    for versions in revisions.values():
        for book in versions:
            publications[book.public_at].append(book)

        origin = versions[-1].first_public_at
        if origin is None:
            continue
        expiry_position = int(trading_index.searchsorted(origin + pd.Timedelta(days=max_age_days), side="right"))
        if expiry_position < len(trading_index):
            expiry_days.add(trading_index[expiry_position])
    days = pd.DatetimeIndex(sorted(set(publications) | expiry_days))
    return publications, days


def _advance_latest_books(latest: dict[str, PublicBook], published: list[PublicBook]) -> None:
    """Advance only resolved quarter watermarks; restricted/partial books keep barriers."""
    for book in published:
        if not book.support.book_known:
            continue
        previous = latest.get(book.cik)
        if previous is None or book.period >= previous.period:
            latest[book.cik] = book


def prepare_superinvestor_book_history(
    packets: Generator[tuple[pd.DataFrame, pd.DataFrame]], trading_index: pd.DatetimeIndex, max_age_days: int
) -> tuple[PublicBookSnapshot, ...] | None:
    """Prepare causal public/expiry states from already cleaned, prepared whole books.

    The caller owns cleaning and prepare_book_vintages. Freshness limits current
    candidates only; archival lag/tenure evidence remains available at its public prefix.
    """
    if max_age_days < 0:
        raise ValueError("Superinvestor book age must be nonnegative")
    grid = pd.DatetimeIndex(trading_index).normalize().unique().sort_values()
    revisions = _collect_public_revisions(packets)
    if revisions is None:
        return None
    if not revisions or grid.empty:
        return ()
    publications, days = _public_book_schedule(revisions, grid, max_age_days)
    snapshots: list[PublicBookSnapshot] = []
    latest: dict[str, PublicBook] = {}

    for day in days:
        _advance_latest_books(latest, publications.get(day, []))
        level_books = {
            cik: book
            for cik, book in latest.items()
            if book.first_public_at is not None and day <= book.first_public_at + pd.Timedelta(days=max_age_days)
        }
        latest_periods = {cik: book.period for cik, book in latest.items()}
        common_period = max(latest_periods.values(), default=None)
        snapshots.append(
            PublicBookSnapshot(
                day, common_period, MappingProxyType(latest_periods), MappingProxyType(level_books), _PublicBookArchive(revisions, day)
            )
        )
    return tuple(snapshots)
