"""
fetch_earnings_call_transcripts.py  (src/data_extract/utils/behavioral/fetch_earnings_call_transcripts.py)
----------------------------------------------------------------------------------------------------------
Raw earnings-call transcripts from the free HuggingFace dataset `defeatbeta/yahoo-finance-data`
(`data/US/stock_earning_call_transcripts.parquet`, ~2.27 GB, rebuilt about daily) into
`earnings_call_sections`, ONE ROW PER SOURCE PARAGRAPH. Nothing is cleaned or split here; the
speaker-turn split runs at aggregate time (`src/utils/earnings_call_split.py`).

The file is one parquet sorted by symbol: 1,195 row groups of ~200 calls, 551 of them touching
the roster. Footer 1.5-3 s, index columns ~0.29 s and a whole row group ~0.86 s per group. So a
run reads the footer once, keeps the row groups whose `symbol` statistics intersect the scope
(and, incrementally, whose `max(report_date)` reaches the frontier minus `lookback_days`),
reads only the five index columns of those groups, diffs them against the stored calls, and
reads the `transcripts` column only of the groups that hold a new or re-issued call.

  * No-op: the transcripts file's content hash is recorded with the run; an unchanged hash,
    an unchanged scope and no reconcile due skip everything after two metadata requests.
  * Reconcile: `-F`, a scope change, a cold table, or `reconcile_days` since the last full run
    compare every scoped call instead of the recent window.
  * Re-issue: a stored (ticker, quarter) whose `transcript_id` or call date changed has its old
    paragraphs deleted, its sentiment and embedding rows invalidated, and a pending refresh
    marker written. The date matters: `transcripts_id` is NULL on 2,914 of 33,676 roster calls
    (1,610 of them in 2025), so the id alone cannot see those calls change.
  * Reads run on `read_workers` threads, each with its own file handle; every DB write runs on
    the calling thread, one row group per batch, so a crash loses at most one batch.

Measured on the 2026-10-01 index (238,891 calls): (symbol, fiscal_year, fiscal_quarter) is
unique; the dedup rule (latest `report_date`, then highest `transcripts_id`) is defensive.
23 roster dates of 10 tickers carry two fiscal labels (provider relabels, e.g. DG 2025Q4 and
2026Q4 on 2026-03-12); `one_call_per_date` keeps one, chaining back from the next later call.
No symbol carries a `.` (BF-B is stored as `BF-B`); `.` is still mapped to `-` on read.
`paragraph_number` starts at 1 and is contiguous on 585/585 sampled calls, so the stored-call
diff reads only paragraph 1 of each call.
"""

from __future__ import annotations

import logging
import threading
import time
from bisect import bisect_left
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import IO, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem
from huggingface_hub.hf_api import RepoFile
from omegaconf import DictConfig

from src.constants.constants import NO_EARNINGS_CALL_TICKERS
from src.context import Context
from src.data_extract.utils.behavioral.utils_earnings_call_cache import (
    invalidate_earnings_call_derivatives,
    pending_refresh_markers,
)
from src.data_extract.utils.common.rate_limit import call_with_retries
from src.data_extract.utils.common.run_manifest import get_entry, manifest_window, record_run
from src.data_store.schema import Tables
from src.utils.universe import load_universe_tickers

logger = logging.getLogger(__name__)

DATASET_REPO = "defeatbeta/yahoo-finance-data"
DATASET_FILE = "data/US/stock_earning_call_transcripts.parquet"
# The manifest key under which the transcripts file's content hash is recorded.
SOURCE_KEY = f"{DATASET_REPO}/{DATASET_FILE}"

INDEX_COLUMNS = ("symbol", "fiscal_year", "fiscal_quarter", "report_date", "transcripts_id")
TRANSCRIPTS_COLUMN = "transcripts"
PARAGRAPH_FIELDS = ("paragraph_number", "speaker", "content")
FIRST_PARAGRAPH = 1
# Full history (the dataset's first call is 2005-10-11); the fallback `since` of a reconcile.
HISTORY_START = "2005-01-01"
_BLOCK_SIZE = 1 << 20
_RETRIES = 4
_RETRY_WAIT_SECONDS = 10.0
_TABLE = Tables.earnings_call_sections
_KEY = ["ticker", "quarter"]


@dataclass(frozen=True)
class TranscriptSource:
    """One pinned revision of the transcripts file: what to read and how to recognise it."""

    revision: str
    fingerprint: str
    opener: Callable[[], IO[bytes]]


@dataclass(frozen=True)
class ExtractSummary:
    """What one extraction run did."""

    noop: bool
    full: bool
    revision: str
    row_groups: int
    calls_new: int
    calls_reissued: int
    rows_written: int


@dataclass(frozen=True)
class RowGroupStats:
    """Footer statistics of one row group; None where the writer stored none."""

    index: int
    symbol_min: str | None
    symbol_max: str | None
    report_date_max: str | None


def resolve_hf_source() -> TranscriptSource:
    """The dataset's current commit, the transcripts file's content hash, and an opener pinned to that commit."""
    api = HfApi()
    info = call_with_retries(lambda: api.dataset_info(DATASET_REPO), retries=_RETRIES, base_wait=_RETRY_WAIT_SECONDS, label="defeatbeta info")
    revision = str(info.sha)
    entries = call_with_retries(
        lambda: api.get_paths_info(DATASET_REPO, [DATASET_FILE], repo_type="dataset", revision=revision),
        retries=_RETRIES,
        base_wait=_RETRY_WAIT_SECONDS,
        label="defeatbeta file",
    )
    files = [entry for entry in entries if isinstance(entry, RepoFile)]
    if not files:
        raise FileNotFoundError(f"{SOURCE_KEY} is absent at revision {revision}")
    fingerprint = files[0].lfs.sha256 if files[0].lfs is not None else files[0].blob_id
    fs = HfFileSystem()
    path = f"datasets/{DATASET_REPO}@{revision}/{DATASET_FILE}"
    return TranscriptSource(revision=revision, fingerprint=str(fingerprint), opener=lambda: fs.open(path, "rb", block_size=_BLOCK_SIZE))


def check_source_schema(schema: pa.Schema) -> None:
    """Fail loudly when the source columns drift from the layout this module parses."""

    def is_text(dtype: pa.DataType) -> bool:
        return pa.types.is_string(dtype) or pa.types.is_large_string(dtype)

    expected: dict[str, Callable[[pa.DataType], bool]] = {
        "symbol": is_text,
        "fiscal_year": pa.types.is_integer,
        "fiscal_quarter": pa.types.is_integer,
        "report_date": is_text,
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
            and is_text(fields.get("speaker", pa.null()))
            and is_text(fields.get("content", pa.null()))
        ):
            problems.append(f"{TRANSCRIPTS_COLUMN}: {dtype}")
    if problems:
        raise ValueError(f"{SOURCE_KEY} schema drift ({'; '.join(problems)}); got {schema}")


def _stat_text(value: Any) -> str | None:
    if value is None:
        return None
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def row_group_stats(metadata: pq.FileMetaData) -> list[RowGroupStats]:
    """Per row group `symbol` min/max and `report_date` max from the footer."""
    paths = [metadata.schema.column(i).path for i in range(metadata.num_columns)]
    symbol_col, date_col = paths.index("symbol"), paths.index("report_date")
    out = []
    for g in range(metadata.num_row_groups):
        group = metadata.row_group(g)
        symbol, date = group.column(symbol_col).statistics, group.column(date_col).statistics
        has_symbol = symbol is not None and symbol.has_min_max
        has_date = date is not None and date.has_min_max
        out.append(
            RowGroupStats(
                index=g,
                symbol_min=_stat_text(symbol.min) if has_symbol else None,
                symbol_max=_stat_text(symbol.max) if has_symbol else None,
                report_date_max=_stat_text(date.max) if has_date else None,
            )
        )
    return out


def select_row_groups(stats: list[RowGroupStats], symbols: list[str], since: str | None) -> list[int]:
    """Row groups that may hold a call of `symbols` (sorted) dated on or after `since`.

    A group without statistics is kept: pruning may only drop what it can prove irrelevant."""
    keep = []
    for group in stats:
        if group.symbol_min is not None and group.symbol_max is not None:
            i = bisect_left(symbols, group.symbol_min)
            if i == len(symbols) or symbols[i] > group.symbol_max:
                continue
        if since is not None and group.report_date_max is not None and group.report_date_max < since:
            continue
        keep.append(group.index)
    return keep


def dataset_symbols(tickers: list[str]) -> list[str]:
    """Repo tickers in every dataset spelling (`BRK-B` and `BRK.B`), sorted for the range test."""
    return sorted({form for ticker in tickers for form in (ticker, ticker.replace("-", "."))})


def calls_from_index(index: pd.DataFrame, scope: set[str], since: str | None) -> pd.DataFrame:
    """One row per scoped call: ticker, quarter, as_of, transcript_id and its row-group position.

    A (ticker, quarter) published twice keeps the latest `report_date`, then the highest
    `transcripts_id`."""
    calls = index.assign(ticker=index["symbol"].astype(str).str.replace(".", "-", regex=False))
    calls = calls[calls["ticker"].isin(scope)]
    if since is not None:
        calls = calls[calls["report_date"].astype(str) >= since]
    calls = calls.assign(
        quarter=calls["fiscal_year"].astype(int).astype(str) + "Q" + calls["fiscal_quarter"].astype(int).astype(str),
        as_of=pd.to_datetime(calls["report_date"], format="%Y-%m-%d"),
        transcript_id=pd.array(calls["transcripts_id"], dtype="Int64"),
    )
    calls = calls.sort_values([*_KEY, "as_of", "transcript_id"], na_position="first").drop_duplicates(_KEY, keep="last")
    calls = one_call_per_date(calls.assign(ordinal=calls["fiscal_year"].astype(int) * 4 + calls["fiscal_quarter"].astype(int)))
    return calls[[*_KEY, "as_of", "transcript_id", "row_group", "row"]].reset_index(drop=True)


def one_call_per_date(calls: pd.DataFrame) -> pd.DataFrame:
    """Keep one fiscal label per (ticker, as_of), resolving each ticker from its latest date back.

    A multi-label date keeps the label whose `ordinal` (fiscal_year*4 + fiscal_quarter) is one
    below the label kept for the next later date, else the highest ordinal. The provider's
    fiscal-year relabels put 23 roster dates of 10 tickers (DG, DLTR, HD, LOW, LULU, TGT, ULTA,
    WSM, PCAR, RJF) under two labels; both would read as two calls on one date. Every later
    call of a ticker sits inside any `since` window that holds an earlier one, so an
    incremental run resolves exactly as a full one."""
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


def explode_paragraphs(transcripts: pa.Array, calls: pd.DataFrame) -> pd.DataFrame:
    """Paragraph rows for `calls`, whose i-th row owns `transcripts[i]`."""
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
    rows = rows[rows["paragraph"].notna()].drop_duplicates([*_KEY, "paragraph"], keep="first")
    return rows.astype({"paragraph": "int64"}).reset_index(drop=True)


class _Readers:
    """One `ParquetFile` per worker thread over the pinned file, sharing the parsed footer."""

    def __init__(self, source: TranscriptSource, metadata: pq.FileMetaData) -> None:
        self._source, self._metadata = source, metadata
        self._local = threading.local()
        self._lock = threading.Lock()
        self._handles: list[IO[bytes]] = []

    def get(self) -> pq.ParquetFile:
        reader = getattr(self._local, "reader", None)
        if reader is None:
            handle = self._source.opener()
            with self._lock:
                self._handles.append(handle)
            reader = pq.ParquetFile(handle, metadata=self._metadata)
            self._local.reader = reader
        return reader

    def close(self) -> None:
        for handle in self._handles:
            handle.close()


def _read_index(readers: _Readers, group: int) -> pd.DataFrame:
    table = call_with_retries(
        lambda: readers.get().read_row_group(group, columns=list(INDEX_COLUMNS)),
        retries=_RETRIES,
        base_wait=_RETRY_WAIT_SECONDS,
        label=f"defeatbeta index rg{group}",
    )
    frame = table.to_pandas()
    return frame.assign(row_group=group, row=np.arange(len(frame)))


def _read_transcripts(readers: _Readers, group: int, calls: pd.DataFrame) -> pd.DataFrame:
    column = call_with_retries(
        lambda: readers.get().read_row_group(group, columns=[TRANSCRIPTS_COLUMN]).column(0),
        retries=_RETRIES,
        base_wait=_RETRY_WAIT_SECONDS,
        label=f"defeatbeta transcripts rg{group}",
    )
    taken = column.combine_chunks().take(pa.array(calls["row"].to_numpy()))
    return explode_paragraphs(taken, calls.reset_index(drop=True))


def _ordered_parallel(groups: list[int], work: Callable[[int], pd.DataFrame], workers: int) -> Iterator[tuple[int, pd.DataFrame]]:
    """`work(group)` on `workers` threads, yielded in submission order with at most 2x`workers`
    results in flight, so a slow consumer bounds memory."""
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        pending: deque[tuple[int, Future[pd.DataFrame]]] = deque()
        queue = iter(groups)
        for group in queue:
            pending.append((group, pool.submit(work, group)))
            if len(pending) >= 2 * max(1, workers):
                break
        while pending:
            group, future = pending.popleft()
            result = future.result()
            following = next(queue, None)
            if following is not None:
                pending.append((following, pool.submit(work, following)))
            yield group, result


def _scope(context: Context, tickers: list[str] | None) -> list[str]:
    names = tickers if tickers is not None else load_universe_tickers(context)
    return sorted({str(t).strip().upper() for t in names if str(t).strip()} - NO_EARNINGS_CALL_TICKERS)


def _stored_calls(context: Context, calls: pd.DataFrame, full: bool) -> pd.DataFrame | None:
    """Stored (ticker, quarter, transcript_id, as_of) of the candidate calls, one row per call.

    Keyed on the candidates' quarters rather than on a date window: a re-issue can move a
    call's date out of any window, and its old paragraphs must still be found."""
    if calls.empty:
        return None
    where: dict[str, object] = {"ticker": sorted(calls["ticker"].unique()), "paragraph": FIRST_PARAGRAPH}
    if not full:
        where["quarter"] = sorted(calls["quarter"].unique())
    stored = context.store.load(_TABLE, ["ticker", "quarter", "transcript_id", "as_of"], where=where, optional=True)
    if stored is None:
        return None
    return stored.merge(calls[_KEY], on=_KEY, how="inner")


def _write_batch(context: Context, paragraphs: pd.DataFrame, reissued: pd.DataFrame) -> int:
    """Replace re-issued calls, invalidate their derivatives, then upsert the batch."""
    store = context.store
    if not reissued.empty:
        for ticker, group in reissued.groupby("ticker", sort=False):
            store.delete(_TABLE, {"ticker": str(ticker), "quarter": group["quarter"].astype(str).tolist()})
        invalidate_earnings_call_derivatives(context, reissued)
        earliest = reissued[["as_of", "old_as_of"]].min(axis=1)
        store.save(Tables.earnings_call_sentiment, pending_refresh_markers(reissued.assign(as_of=earliest)))
    return store.save(_TABLE, paragraphs) if not paragraphs.empty else 0


def extract_earnings_calls(
    context: Context,
    cfg: DictConfig,
    *,
    full: bool = False,
    tickers: list[str] | None = None,
    source: TranscriptSource | None = None,
) -> ExtractSummary:
    """Bring `earnings_call_sections` up to the current defeatbeta revision.

    `cfg` is the `earnings_calls` config block (`lookback_days`, `read_workers`,
    `reconcile_days`). `tickers` narrows the scope (default: the analysis universe, minus the
    names that hold no call). `full` forces a reconcile of every scoped call. `source`
    replaces the HuggingFace file (tests)."""
    log = context.log
    started = time.monotonic()
    scope = _scope(context, tickers)
    src = source if source is not None else resolve_hf_source()

    _, reconcile_due = manifest_window(context, _TABLE, len(scope), pd.Timestamp(HISTORY_START), int(cfg.reconcile_days), tickers=scope)
    frontier = context.store.max_date(_TABLE)
    is_full = full or reconcile_due or frontier is None
    recorded = ((get_entry(context, _TABLE) or {}).get("identity_scope_fingerprints") or {}).get(SOURCE_KEY)
    if not is_full and recorded == src.fingerprint:
        log.info("Earnings calls: %s unchanged (%s) -> nothing to do (%.1fs).", SOURCE_KEY, src.fingerprint[:12], time.monotonic() - started)
        return ExtractSummary(noop=True, full=False, revision=src.revision, row_groups=0, calls_new=0, calls_reissued=0, rows_written=0)

    since = None if is_full else (frontier - pd.Timedelta(days=int(cfg.lookback_days))).strftime("%Y-%m-%d")
    with src.opener() as handle:
        footer = pq.ParquetFile(handle)
        metadata = footer.metadata
        check_source_schema(footer.schema_arrow)
    groups = select_row_groups(row_group_stats(metadata), dataset_symbols(scope), since)
    log.info(
        "Earnings calls: revision %s, %s run over %d tickers, %d/%d row groups%s.",
        src.revision[:12],
        "full" if is_full else "incremental",
        len(scope),
        len(groups),
        metadata.num_row_groups,
        "" if since is None else f" with calls since {since}",
    )

    workers = int(cfg.read_workers)
    readers = _Readers(src, metadata)
    rows_written = 0
    try:
        index_frames = [frame for _, frame in _ordered_parallel(groups, lambda g: _read_index(readers, g), workers)]
        index = pd.concat(index_frames, ignore_index=True) if index_frames else pd.DataFrame(columns=[*INDEX_COLUMNS, "row_group", "row"])
        calls = calls_from_index(index, set(scope), since)
        new, reissued = diff_calls(calls, _stored_calls(context, calls, is_full))
        log.info("Earnings calls: %d source calls in scope, %d new, %d re-issued.", len(calls), len(new), len(reissued))

        needed = pd.concat([new.assign(old_as_of=pd.NaT), reissued], ignore_index=True)
        by_group = {int(g): part for g, part in needed.groupby("row_group", sort=True)}
        empty_calls = 0
        for group, paragraphs in _ordered_parallel(list(by_group), lambda g: _read_transcripts(readers, g, by_group[g]), workers):
            part = by_group[group]
            empty_calls += len(part) - paragraphs[_KEY].drop_duplicates().shape[0]
            rows_written += _write_batch(context, paragraphs, part[part["old_as_of"].notna()].reset_index(drop=True))
        if empty_calls:
            log.warning("Earnings calls: %d call(s) carry no paragraph in the source.", empty_calls)
    finally:
        readers.close()

    record_run(
        context,
        _TABLE,
        len(scope),
        rows_written,
        is_full_rescan=is_full,
        identity_scope_fingerprints={SOURCE_KEY: src.fingerprint},
        tickers=scope,
    )
    log.info("Earnings calls: +%d paragraph rows in %.1fs.", rows_written, time.monotonic() - started)
    return ExtractSummary(
        noop=False,
        full=is_full,
        revision=src.revision,
        row_groups=len(groups),
        calls_new=len(new),
        calls_reissued=len(reissued),
        rows_written=rows_written,
    )
