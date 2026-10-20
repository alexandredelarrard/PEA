"""Raw earnings-call transcripts from the HuggingFace dataset `defeatbeta/yahoo-finance-data` into
`earnings_call_sections`, one row per source paragraph; the speaker-turn split runs at aggregate time.

A run reads the parquet footer once, keeps the row groups whose `symbol` statistics meet the lineage scope,
maps their complete historical index to canonical tickers, diffs the index against the
stored calls and reads the `transcripts` column only for new or re-issued calls. Only `-F` also
deletes stored calls absent from the source; a re-issued or removed call has its derivatives
invalidated and a pending refresh marker written. Reads run on `read_workers` threads; every DB
write runs on the calling thread, one row group per batch.
"""

from __future__ import annotations

import logging
import threading
import time
from bisect import bisect_left
from collections import deque
from collections.abc import Callable, Collection, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import IO, Any, SupportsInt, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem
from omegaconf import DictConfig

from src.context import Context
from src.data_extract.utils.behavioral.utils_earnings_call_cache import (
    invalidate_earnings_call_derivatives,
    pending_refresh_markers,
)
from src.data_extract.utils.common.identity import AmbiguousSymbolTenureError, Identity, load_identity
from src.data_extract.utils.common.rate_limit import call_with_retries
from src.data_store.schema import Tables
from src.utils.string import squash, yahoo_symbol
from src.utils.universe import load_universe_tickers

logger = logging.getLogger(__name__)

DATASET_REPO = "defeatbeta/yahoo-finance-data"
DATASET_FILE = "data/US/stock_earning_call_transcripts.parquet"
SOURCE_KEY = f"{DATASET_REPO}/{DATASET_FILE}"

INDEX_COLUMNS = ("symbol", "fiscal_year", "fiscal_quarter", "report_date", "transcripts_id")
TRANSCRIPTS_COLUMN = "transcripts"
FIRST_PARAGRAPH = 1
_BLOCK_SIZE = 1 << 20
_RETRIES = 4
_RETRY_WAIT_SECONDS = 10.0
_TABLE = Tables.earnings_call_sections
_KEY = ["ticker", "quarter"]


@dataclass(frozen=True)
class TranscriptSource:
    """One pinned revision of the transcripts file and an opener for it."""

    revision: str
    opener: Callable[[], IO[bytes]]


@dataclass(frozen=True)
class ExtractSummary:
    """What one extraction run did."""

    full: bool
    revision: str
    row_groups: int
    calls_new: int
    calls_reissued: int
    rows_written: int
    calls_removed: int = 0


def resolve_hf_source() -> TranscriptSource:
    """The dataset's current commit and an opener for the transcripts file pinned to that commit."""
    api = HfApi()
    info = call_with_retries(lambda: api.dataset_info(DATASET_REPO), retries=_RETRIES, base_wait=_RETRY_WAIT_SECONDS, label="defeatbeta info")
    revision = str(info.sha)
    fs = HfFileSystem()
    path = f"datasets/{DATASET_REPO}@{revision}/{DATASET_FILE}"
    return TranscriptSource(revision=revision, opener=lambda: cast(IO[bytes], fs.open(path, "rb", block_size=_BLOCK_SIZE)))


def _is_text(dtype: pa.DataType) -> bool:
    return pa.types.is_string(dtype) or pa.types.is_large_string(dtype)


def check_source_schema(schema: pa.Schema) -> None:
    """Fail loudly when the source columns drift from the layout this module parses."""
    expected: dict[str, Callable[[pa.DataType], bool]] = {
        "symbol": _is_text,
        "fiscal_year": pa.types.is_integer,
        "fiscal_quarter": pa.types.is_integer,
        "report_date": _is_text,
        "transcripts_id": pa.types.is_integer,
    }
    problems = [f"{name}: missing" for name in [*expected, TRANSCRIPTS_COLUMN] if name not in schema.names]
    problems += [f"{name}: {schema.field(name).type}" for name, ok in expected.items() if name in schema.names and not ok(schema.field(name).type)]
    if TRANSCRIPTS_COLUMN in schema.names:
        dtype = schema.field(TRANSCRIPTS_COLUMN).type
        item = dtype.value_type if pa.types.is_list(dtype) or pa.types.is_large_list(dtype) else None
        fields = {item.field(i).name: item.field(i).type for i in range(item.num_fields)} if item is not None and pa.types.is_struct(item) else {}
        if not (
            pa.types.is_integer(fields.get("paragraph_number", pa.null()))
            and _is_text(fields.get("speaker", pa.null()))
            and _is_text(fields.get("content", pa.null()))
        ):
            problems.append(f"{TRANSCRIPTS_COLUMN}: {dtype}")
    if problems:
        raise ValueError(f"{SOURCE_KEY} schema drift ({'; '.join(problems)}); got {schema}")


def select_row_groups(metadata: pq.FileMetaData, symbols: Collection[str]) -> list[int]:
    """Footer symbol pruning, including share-class spellings; missing statistics never exclude a group."""
    spellings = sorted({form for symbol in symbols for form in (symbol, symbol.replace("-", "."), symbol.replace("-", "/"))})
    symbol_column = [metadata.schema.column(i).path for i in range(metadata.num_columns)].index("symbol")
    keep = []
    for group in range(metadata.num_row_groups):
        stats = metadata.row_group(group).column(symbol_column).statistics
        if stats is not None and stats.has_min_max:
            position = bisect_left(spellings, stats.min)
            if position == len(spellings) or spellings[position] > stats.max:
                continue
        keep.append(group)
    return keep


def _allows_rebased_call(identity: Identity, entity: str, day: pd.Timestamp) -> bool:
    """Without unambiguous dated issuer support, a reused alias cannot justify historical rebasing."""
    aliases = {
        symbol
        for symbol, rows in identity.symbol_intervals.items()
        for row in rows
        if row.entity == entity and row.status not in {"noise", "conflict"} and row.tape_symbol
    }
    for alias in aliases:
        hit = identity.tape_interval(alias, day)
        if hit is None or hit.entity != entity:
            continue
        try:
            if identity.entity_for(alias, day) == entity:
                return True
        except AmbiguousSymbolTenureError:
            continue
    for alias in aliases:
        try:
            owner = identity.entity_for(alias)
        except AmbiguousSymbolTenureError:
            return False
        if owner is not None and owner != entity:
            return False
    return True


def ticker_for_call(identity: Identity, symbol: str, day: pd.Timestamp) -> str | None:
    """Dated symbol ownership, allowing a vendor to retain/rebase a consolidating issuer's alias."""
    key = yahoo_symbol(symbol)
    if key in identity.redundant_symbols or pd.isna(day):
        return None
    intervals = identity.symbol_intervals.get(key, ())
    active = [row for row in intervals if row.covers(day) and row.status != "noise" and row.tape_symbol]
    if any(row.status == "conflict" for row in active):
        return None
    try:
        dated_owner = identity.entity_for(key, day)
    except AmbiguousSymbolTenureError:
        return None
    hit = identity.tape_interval(key, day)
    candidates = (
        {hit.entity}
        if hit is not None
        else {
            row.entity
            for row in intervals
            if row.status not in {"noise", "conflict"}
            and row.tape_symbol
            and any(window.cik == row.cik and window.owns(day) for window in identity.windows_by_entity.get(row.entity, ()))
        }
    )
    if not intervals and key in identity.roster_cik:
        entity = identity.universe_entity(key)
        candidates = (
            {entity}
            if any(window.cik == identity.roster_cik[key] and window.owns(day) for window in identity.windows_by_entity.get(entity, ()))
            else set()
        )
    if dated_owner is not None:
        candidates &= {dated_owner}
    if active:
        owners = {row.entity for row in active}
        if len(owners) != 1:
            return None
        candidates &= owners
    if len(candidates) != 1:
        return None
    entity = next(iter(candidates))
    if hit is None and not _allows_rebased_call(identity, entity, day):
        return None
    ticker = identity.ticker_by_entity.get(entity)
    if ticker is not None and "-" in ticker:
        stem, share_class = ticker.rsplit("-", 1)
        # A separately traded sibling class is not a predecessor symbol.
        if len(share_class) == 1 and key != ticker and squash(key).startswith(stem) and len(squash(key)) == len(stem) + 1:
            return None
    return ticker


def calls_from_index(index: pd.DataFrame, scope: set[str], identity: Identity) -> pd.DataFrame:
    """One row per scoped call: ticker, quarter, as_of, transcript_id and its row-group position.

    Resolve ownership before narrowing to the requested destination. A (ticker, quarter)
    published twice keeps the latest `report_date`, then the highest `transcripts_id`."""
    calls = index.assign(as_of=pd.to_datetime(index["report_date"], format="%Y-%m-%d"))
    calls["ticker"] = [ticker_for_call(identity, str(symbol), pd.Timestamp(day)) for symbol, day in zip(calls["symbol"], calls["as_of"], strict=True)]
    unresolved = calls["ticker"].isna()
    if unresolved.any():
        logger.warning(
            "Earnings calls: %d unresolved source call(s), symbols %s.", int(unresolved.sum()), sorted(calls.loc[unresolved, "symbol"].unique())
        )
    calls = calls[calls["ticker"].isin(scope)]
    calls = calls.assign(
        quarter=calls["fiscal_year"].astype(int).astype(str) + "Q" + calls["fiscal_quarter"].astype(int).astype(str),
        transcript_id=calls["transcripts_id"].astype("Int64"),
    )
    calls = calls.sort_values([*_KEY, "as_of", "transcript_id"], na_position="first").drop_duplicates(_KEY, keep="last")
    calls = one_call_per_date(calls.assign(ordinal=calls["fiscal_year"].astype(int) * 4 + calls["fiscal_quarter"].astype(int)))
    return calls[[*_KEY, "as_of", "transcript_id", "row_group", "row"]].reset_index(drop=True)


def one_call_per_date(calls: pd.DataFrame) -> pd.DataFrame:
    """Keep one fiscal label per (ticker, as_of), resolving each ticker from its latest date back.

    A multi-label date keeps the label whose `ordinal` (fiscal_year*4 + fiscal_quarter) is one
    below the label kept for the next later date, else the highest ordinal. The provider's
    fiscal-year relabels put 23 roster dates of 10 tickers (DG, DLTR, HD, LOW, LULU, TGT, ULTA,
    WSM, PCAR, RJF) under two labels; both would read as two calls on one date."""
    per_date = calls.groupby(["ticker", "as_of"])["ordinal"].transform("size")
    tickers = set(calls.loc[per_date > 1, "ticker"])
    if not tickers:
        return calls
    drop: list[Any] = []
    for _, chain in calls[calls["ticker"].isin(tickers)].groupby("ticker", sort=False):
        following: int | None = None
        for _, day in chain.sort_values("as_of", ascending=False).groupby("as_of", sort=False):
            ordinals = day["ordinal"]
            matched = ordinals[ordinals == following - 1] if following is not None else ordinals.iloc[0:0]
            kept = matched.index[0] if len(matched) else ordinals.idxmax()
            drop += [i for i in day.index if i != kept]
            following = int(ordinals[kept])
    return calls.drop(index=drop)


def diff_calls(calls: pd.DataFrame, stored: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(new, reissued) source calls. A re-issue keeps the key but changes its `transcript_id`
    or its call date; `old_as_of` carries the stored date."""
    if stored is None or stored.empty:
        return calls.copy(), calls.iloc[0:0].assign(old_as_of=pd.Series(dtype="datetime64[ns]"))
    stored = stored.assign(
        old_tid=pd.array(pd.to_numeric(stored["transcript_id"], errors="coerce"), dtype="Int64"),
        old_as_of=pd.to_datetime(stored["as_of"], errors="coerce").dt.normalize(),
    )[[*_KEY, "old_tid", "old_as_of"]].drop_duplicates(_KEY)
    merged = calls.merge(stored, on=_KEY, how="left", indicator=True)
    present = merged["_merge"] == "both"
    same_tid = (merged["transcript_id"] == merged["old_tid"]).fillna(False) | (merged["transcript_id"].isna() & merged["old_tid"].isna())
    same_date = merged["as_of"] == merged["old_as_of"]
    reissued = merged[present & ~(same_tid & same_date)]
    new = merged[~present]
    return new[calls.columns].reset_index(drop=True), reissued[[*calls.columns, "old_as_of"]].reset_index(drop=True)


def stale_calls(calls: pd.DataFrame, stored: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    """(gone, vanished): stored (ticker, quarter, as_of) whose key is absent from `calls`, and
    the stored tickers with no source call at all. Only meaningful when `calls` holds the whole
    source of every ticker in `stored` (a reconcile).

    A ticker whose every call left the source is reported, not removed: a provider symbol
    change or a partial file reads exactly like that, and it would delete the whole history."""
    empty = pd.DataFrame({"ticker": pd.Series(dtype=object), "quarter": pd.Series(dtype=object), "as_of": pd.Series(dtype="datetime64[ns]")})
    if stored is None or stored.empty:
        return empty, []
    keys = stored[_KEY].astype(str).assign(as_of=pd.to_datetime(stored["as_of"], errors="coerce").dt.normalize()).drop_duplicates(_KEY)
    source = calls[_KEY].astype(str)
    vanished = sorted(set(keys["ticker"]) - set(source["ticker"]))
    merged = keys[~keys["ticker"].isin(vanished)].merge(source, on=_KEY, how="left", indicator=True)
    return merged.loc[merged["_merge"] == "left_only", [*_KEY, "as_of"]].reset_index(drop=True), vanished


def explode_paragraphs(transcripts: pa.Array, calls: pd.DataFrame) -> pd.DataFrame:
    """Retained paragraph rows for `calls`, numbered contiguously after dropping content of length <= 2."""
    flat = pc.list_flatten(transcripts)
    owner = pc.list_parent_indices(transcripts).to_numpy(zero_copy_only=False)
    meta = calls.iloc[owner].reset_index(drop=True)
    rows = pd.DataFrame(
        {
            "ticker": meta["ticker"],
            "quarter": meta["quarter"],
            "paragraph": pd.array(flat.field("paragraph_number").to_pandas(), dtype="Int64"),
            "as_of": meta["as_of"],
            "transcript_id": meta["transcript_id"],
            "speaker": flat.field("speaker").to_pandas(),
            "content": flat.field("content").to_pandas(),
        }
    )
    rows = rows.loc[rows["paragraph"].notna() & (rows["content"].str.len() > 2)].drop_duplicates([*_KEY, "paragraph"], keep="first")
    # Filtering can remove paragraph 1, which identifies a stored call on resume.
    rows["paragraph"] = rows.groupby(_KEY)["paragraph"].rank(method="dense")
    return rows.astype({"paragraph": "int64"}).reset_index(drop=True)


class _Readers:
    """One `ParquetFile` per worker thread over the pinned file, sharing the parsed footer."""

    def __init__(self, source: TranscriptSource, metadata: pq.FileMetaData) -> None:
        self._source, self._metadata = source, metadata
        self._local = threading.local()
        self._lock = threading.Lock()
        self._handles: list[IO[bytes]] = []

    def read(self, group: int, columns: list[str]) -> pa.Table:
        """Read projected columns with the same retry policy for metadata and payloads."""

        def read_group() -> pa.Table:
            reader = getattr(self._local, "reader", None)
            if reader is None:
                handle = self._source.opener()
                with self._lock:
                    self._handles.append(handle)
                reader = pq.ParquetFile(handle, metadata=self._metadata)
                self._local.reader = reader
            return reader.read_row_group(group, columns=columns)

        return call_with_retries(
            read_group,
            retries=_RETRIES,
            base_wait=_RETRY_WAIT_SECONDS,
            label=f"defeatbeta {','.join(columns)} rg{group}",
        )

    def close(self) -> None:
        for handle in self._handles:
            handle.close()


def _read_index(readers: _Readers, group: int) -> pd.DataFrame:
    frame = readers.read(group, list(INDEX_COLUMNS)).to_pandas()
    return frame.assign(row_group=group, row=np.arange(len(frame)))


def _read_transcripts(readers: _Readers, group: int, calls: pd.DataFrame) -> pd.DataFrame:
    column = readers.read(group, [TRANSCRIPTS_COLUMN]).column(0)
    taken = column.combine_chunks().take(pa.array(calls["row"].to_numpy()))
    return explode_paragraphs(taken, calls.reset_index(drop=True))


def _ordered_parallel(groups: list[int], work: Callable[[int], pd.DataFrame], workers: int) -> Iterator[tuple[int, pd.DataFrame]]:
    """`work(group)` on `workers` threads, yielded in submission order with at most 2x`workers`
    results in flight, so a slow consumer bounds memory."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: deque[tuple[int, Future[pd.DataFrame]]] = deque()
        queue = iter(groups)
        for group in queue:
            pending.append((group, pool.submit(work, group)))
            if len(pending) >= 2 * workers:
                break
        while pending:
            group, future = pending.popleft()
            result = future.result()
            following = next(queue, None)
            if following is not None:
                pending.append((following, pool.submit(work, following)))
            yield group, result


def _stored_calls(context: Context, calls: pd.DataFrame, scope: set[str], full: bool) -> pd.DataFrame | None:
    """Stored (ticker, quarter, transcript_id, as_of), one row per call: every stored call of
    `scope` on a reconcile, else only the candidates' keys.

    Keyed on the candidates' quarters rather than on a date window: a re-issue can move a
    call's date out of any window, and its old paragraphs must still be found."""
    if full:
        where: dict[str, object] = {"ticker": list(scope), "paragraph": FIRST_PARAGRAPH}
    elif calls.empty:
        return None
    else:
        where = {"ticker": sorted(calls["ticker"].unique()), "paragraph": FIRST_PARAGRAPH, "quarter": sorted(calls["quarter"].unique())}
    stored = context.store.load(_TABLE, ["ticker", "quarter", "transcript_id", "as_of"], where=where, optional=True)
    if stored is None or full:
        return stored
    return stored.merge(calls[_KEY], on=_KEY, how="inner")


def _remove_calls(context: Context, calls: pd.DataFrame) -> None:
    """Invalidate caches and save refresh markers before deleting old paragraphs so interrupted work can safely retry."""
    invalidate_earnings_call_derivatives(context, calls)
    context.store.save(Tables.earnings_call_sentiment, pending_refresh_markers(calls))
    for ticker, group in calls.groupby("ticker", sort=False):
        context.store.delete(_TABLE, {"ticker": str(ticker), "quarter": group["quarter"].astype(str).tolist()})


def _write_batch(context: Context, paragraphs: pd.DataFrame, reissued: pd.DataFrame) -> int:
    """Preserve invalidation and the earliest refresh date before replacing old paragraphs."""
    if not reissued.empty:
        earliest = reissued[["as_of", "old_as_of"]].min(axis=1)
        _remove_calls(context, reissued.assign(as_of=earliest))
    return context.store.save(_TABLE, paragraphs) if not paragraphs.empty else 0


def extract_earnings_calls(
    context: Context,
    cfg: DictConfig,
    *,
    full: bool = False,
    tickers: list[str] | None = None,
    source: TranscriptSource | None = None,
) -> ExtractSummary:
    """Bring `earnings_call_sections` up to the current defeatbeta revision.

    `cfg` is the `earnings_calls` config block (`read_workers`). `tickers` narrows the scope
    (default: the analysis universe). Every run checks complete scoped metadata and fetches
    missing/reissued payloads. `full` additionally reconciles source-absent calls, retaining
    tickers that vanished entirely. `source` replaces the HuggingFace file (tests)."""
    log = context.log
    started = time.monotonic()
    scope = set(load_universe_tickers(context) if tickers is None else tickers)
    identity = load_identity(context)
    symbols = identity.universe_symbols(scope)
    src = source if source is not None else resolve_hf_source()

    with src.opener() as handle:
        footer = pq.ParquetFile(handle)
        metadata = footer.metadata
        check_source_schema(footer.schema_arrow)
    groups = select_row_groups(metadata, symbols)
    log.info(
        "Earnings calls: revision %s, %s run over %d tickers, %d/%d row groups, complete scoped metadata.",
        src.revision[:12],
        "full" if full else "incremental",
        len(scope),
        len(groups),
        metadata.num_row_groups,
    )

    workers = max(1, int(cfg.read_workers))
    readers = _Readers(src, metadata)
    rows_written = 0
    try:
        index_frames = [frame for _, frame in _ordered_parallel(groups, lambda g: _read_index(readers, g), workers)]
        df_index = pd.concat(index_frames, ignore_index=True) if index_frames else pd.DataFrame(columns=[*INDEX_COLUMNS, "row_group", "row"])
        df_index = df_index[df_index["symbol"].map(yahoo_symbol).isin(symbols)]
        df_calls = calls_from_index(df_index, scope, identity)
        df_stored = _stored_calls(context, df_calls, scope, full)
        df_new, df_reissued = diff_calls(df_calls, df_stored)
        df_gone, vanished = stale_calls(df_calls, df_stored if full else None)
        if vanished:
            log.warning("Earnings calls: %d stored ticker(s) have no call left in the source, kept as stored: %s", len(vanished), vanished[:20])
        log.info(
            "Earnings calls: %d source calls in scope, %d new, %d re-issued, %d removed from the source.",
            len(df_calls),
            len(df_new),
            len(df_reissued),
            len(df_gone),
        )
        if not df_gone.empty:
            log.warning(
                "Earnings calls: deleting %d stored call(s) absent from the source: %s", len(df_gone), df_gone[_KEY].head(10).to_dict("records")
            )
            _remove_calls(context, df_gone)

        df_needed = pd.concat(
            [
                df_new.assign(old_as_of=pd.Series(pd.NaT, index=df_new.index, dtype="datetime64[ns]"), is_reissued=False),
                df_reissued.assign(is_reissued=True),
            ],
            ignore_index=True,
        )
        by_group = {int(cast(SupportsInt, g)): df_part for g, df_part in df_needed.groupby("row_group", sort=True)}
        empty_calls = 0
        for group, df_paragraphs in _ordered_parallel(list(by_group), lambda g: _read_transcripts(readers, g, by_group[g]), workers):
            df_part = by_group[group]
            empty_calls += len(df_part) - df_paragraphs[_KEY].drop_duplicates().shape[0]
            rows_written += _write_batch(context, df_paragraphs, df_part[df_part["is_reissued"]])
        if empty_calls:
            log.warning("Earnings calls: %d call(s) carry no paragraph in the source.", empty_calls)
    finally:
        readers.close()

    log.info("Earnings calls: +%d paragraph rows in %.1fs.", rows_written, time.monotonic() - started)
    return ExtractSummary(
        full=full,
        revision=src.revision,
        row_groups=len(groups),
        calls_new=len(df_new),
        calls_reissued=len(df_reissued),
        rows_written=rows_written,
        calls_removed=len(df_gone),
    )
