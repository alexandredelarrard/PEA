"""Fresh public manager books and immutable, date-aware report-quarter archives."""

from __future__ import annotations

import logging
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Generator, Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd

from src.utils.institutional_vintages import full_13f_report, prepare_book_vintages, resolve_prepared_books

type BookKey = tuple[str, pd.Timestamp]

_log = logging.getLogger(__name__)


def _trailing_holding_keys(
    df_filings: pd.DataFrame, df_holdings: pd.DataFrame, manager: str | None, keys: set[tuple[str, str]]
) -> tuple[str | None, set[tuple[str, str]]]:
    """Reject holding vintages repeated across batches; only the trailing manager can straddle the ordered source."""
    df_keys = df_holdings[["cik", "accession_number", "cusip"]]
    if manager is not None and keys.intersection(
        df_keys.loc[df_keys.cik.eq(manager), ["accession_number", "cusip"]].itertuples(index=False, name=None)
    ):
        raise ValueError("13F holding vintages must be unique")
    ciks: set[str] = set(df_filings["cik"]).union(df_keys["cik"])
    if not ciks:
        return manager, keys
    trailing = max(ciks)
    rows = set(df_keys.loc[df_keys.cik.eq(trailing), ["accession_number", "cusip"]].itertuples(index=False, name=None))
    return trailing, keys | rows if trailing == manager else rows


def prepare_superinvestor_batches(
    packets: Iterable[tuple[pd.DataFrame, pd.DataFrame]], trading_index: pd.DatetimeIndex
) -> Generator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Preserve global uniqueness on the source's ordered, complete CIK-quarter batches."""
    accessions: set[str] = set()
    manager: str | None = None
    keys: set[tuple[str, str]] = set()
    peak_keys = 0
    for df_filings, df_holdings in packets:
        if df_filings["accession_number"].isin(accessions).any():
            raise ValueError("13F manifest accessions must be unique")
        df_manifest = prepare_book_vintages(df_filings, df_holdings, trading_index, settle_days=0)
        manager, keys = _trailing_holding_keys(df_filings, df_holdings, manager, keys)
        accessions.update(df_filings["accession_number"])
        peak_keys = max(peak_keys, len(keys))
        yield df_manifest, df_holdings
    _log.info("13F identity validation: %s manifest accessions; peak trailing-manager holding keys=%s", len(accessions), peak_keys)


@dataclass(frozen=True, slots=True)
class BookSupport:
    """Resolver evidence determining which positions and absent positions are observable."""

    book_known: bool
    zero_eligible: bool
    comparison_eligible: bool
    restricted: bool
    resolution_status: str

    @property
    def comparable(self) -> bool:
        """Known, complete and unrestricted enough to support reported zeros, tenure and quarter comparisons."""
        return self.book_known and self.zero_eligible and self.comparison_eligible


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


def _single_version_books(
    df_manifest: pd.DataFrame, df_holdings: pd.DataFrame, full_report: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """The resolver's positions and states, without a replay, of quarters whose only version is a complete public
    original/restatement, and the mask of their manifest rows."""
    versions = df_manifest.groupby(["cik", "period"])["accession_number"].transform("size")
    single = versions.eq(1) & df_manifest["amendment_type"].isin(["original", "restatement"]) & df_manifest["parse_status"].eq("complete")
    single &= df_manifest["public_at"].notna()
    df_single = df_manifest.loc[single].set_index("accession_number")
    df_rows = df_holdings.loc[df_holdings["accession_number"].isin(df_single.index)]
    incomplete = df_rows["accession_number"].value_counts().reindex(df_single.index, fill_value=0).ne(df_single["parsed_row_count"].astype(int))
    if incomplete.any():
        raise ValueError(f"Incomplete stored 13F book: {incomplete.index[incomplete][0]}")

    # Positions take their quarter from the filing, in the resolver's (cik, period, cusip) order.
    filing = df_single.index.get_indexer(pd.Index(df_rows["accession_number"]))
    df_rows = df_rows.assign(period=df_single["period"].array.take(filing), cik=df_single["cik"].array.take(filing))
    restricted = df_single["restricted"].astype(bool)
    eligible = ~restricted & full_report.loc[single].astype(bool).to_numpy()
    df_states = df_single[["cik", "period"]].assign(
        book_known=True, zero_eligible=eligible, comparison_eligible=eligible, restricted=restricted, resolution_status="resolved"
    )
    return df_rows.sort_values(["cik", "period", "cusip"], kind="stable"), df_states, single


def _publication_rounds(
    df_manifest: pd.DataFrame, single: pd.Series, revisions: Mapping[BookKey, list[PublicBook]]
) -> tuple[list[BookKey], pd.DataFrame]:
    """Every public quarter in first-publication order, and the amended quarters' distinct public days numbered per quarter."""
    if not set(df_manifest[["cik", "period"]].itertuples(index=False, name=None)).isdisjoint(revisions):
        raise ValueError("13F prepared batches must contain complete disjoint CIK-quarters")
    df_events = df_manifest[["cik", "period", "public_at"]].assign(_single=single).dropna(subset=["public_at"])
    df_events = df_events.drop_duplicates(["cik", "period", "public_at"]).sort_values("public_at", kind="stable")
    df_events = df_events.assign(_round=df_events.groupby(["cik", "period"], sort=False).cumcount())
    order = list(df_events.loc[df_events["_round"].eq(0), ["cik", "period"]].itertuples(index=False, name=None))
    return order, df_events.loc[~df_events["_single"], ["cik", "period", "public_at", "_round"]]


def _append_revisions(revisions: dict[BookKey, list[PublicBook]], df_round: pd.DataFrame, df_books: pd.DataFrame, df_states: pd.DataFrame) -> None:
    """Append each quarter's resolved book at its public day; age starts at its first known resolution."""
    positions = {key: df.reset_index(drop=True) for key, df in df_books.groupby(["cik", "period"], sort=False)}
    states = {(state["cik"], state["period"]): state for state in df_states.to_dict("records")}
    for cik, period, day in df_round.itertuples(index=False, name=None):
        state = states.get((cik, period))
        support = BookSupport(False, False, False, False, "unresolved")
        if state is not None:
            known, zero, comparison, restricted = (bool(state[name]) for name in ("book_known", "zero_eligible", "comparison_eligible", "restricted"))
            support = BookSupport(known, zero, comparison, restricted, str(state["resolution_status"]))
        previous = revisions.setdefault((cik, period), [])
        origin = previous[-1].first_public_at if previous else None
        origin = day if support.book_known and origin is None else origin
        previous.append(PublicBook(cik, period, day, origin, positions.get((cik, period), df_books.iloc[:0].reset_index(drop=True)), support))


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
    packets: Iterable[tuple[pd.DataFrame, pd.DataFrame]], trading_index: pd.DatetimeIndex, max_age_days: int
) -> tuple[PublicBookSnapshot, ...] | None:
    """Prepare causal public/expiry states from already cleaned, prepared whole books.

    The caller owns cleaning and prepare_book_vintages. Freshness limits current
    candidates only; archival lag/tenure evidence remains available at its public prefix.
    """
    if max_age_days < 0:
        raise ValueError("Superinvestor book age must be nonnegative")
    grid = pd.DatetimeIndex(trading_index).normalize().unique().sort_values()
    revisions: dict[BookKey, list[PublicBook]] = {}
    filings_count = 0

    # A quarter's book at a day depends only on its own versions: one resolver call serves the n-th public day of
    # every amended quarter in the batch, each quarter cut to its own day.
    for df_manifest, df_holdings in packets:
        filings_count += len(df_manifest)
        df_rows, df_states, single = _single_version_books(df_manifest, df_holdings, df_manifest["report_type"].map(full_13f_report))
        order, df_events = _publication_rounds(df_manifest, single, revisions)
        books: dict[BookKey, list[PublicBook]] = {}
        _append_revisions(books, df_manifest.loc[single, ["cik", "period", "public_at"]], df_rows, df_states)
        df_replayed = df_holdings.loc[~df_holdings["accession_number"].isin(df_manifest.loc[single, "accession_number"])]
        for _, df_round in df_events.groupby("_round", sort=True):
            cutoff = pd.Series(df_round["public_at"].to_numpy(), index=pd.MultiIndex.from_frame(df_round[["cik", "period"]]))
            visible = df_manifest["public_at"].le(cutoff.reindex(pd.MultiIndex.from_frame(df_manifest[["cik", "period"]])).to_numpy())
            resolved = resolve_prepared_books(df_manifest.loc[visible.to_numpy()], df_replayed, df_round["public_at"].max())
            _append_revisions(books, df_round[["cik", "period", "public_at"]], *resolved)
        revisions.update((key, books[key]) for key in order)
    if not filings_count:
        return None
    if not revisions or grid.empty:
        return ()

    archive = MappingProxyType({key: tuple(revisions[key]) for key in sorted(revisions, key=lambda key: revisions[key][0].public_at)})
    publications, days = _public_book_schedule(archive, grid, max_age_days)
    snapshots: list[PublicBookSnapshot] = []
    latest: dict[str, PublicBook] = {}
    age = pd.Timedelta(days=max_age_days)
    for day in days:
        _advance_latest_books(latest, publications.get(day, []))
        level_books = {cik: book for cik, book in latest.items() if book.first_public_at is not None and day <= book.first_public_at + age}
        latest_periods = {cik: book.period for cik, book in latest.items()}
        common_period = max(latest_periods.values(), default=None)
        snapshots.append(
            PublicBookSnapshot(day, common_period, MappingProxyType(latest_periods), MappingProxyType(level_books), _PublicBookArchive(archive, day))
        )
    return tuple(snapshots)
