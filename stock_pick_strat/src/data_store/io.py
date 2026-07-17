"""
io.py  (src/data_store/io.py)
-----------------------------
Shared source-file reader so the schema generator and the migrator see an
IDENTICAL frame for every table. Notably it promotes a single-level index to a
real column when the table's PK column lives in the index rather than the
columns (e.g. ticker_embeddings, whose ticker is the parquet index).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow.parquet as pq

from src.data_store.schema_registry import TableSpec


def read_source(spec: TableSpec, data_dir: Path, nrows: int | None = None) -> pd.DataFrame:
    path = data_dir / spec.source
    if path.suffix == ".csv":
        df = pd.read_csv(path, nrows=nrows)
    elif nrows is not None:
        # read only the first `nrows` without materializing the whole file
        # (cube.parquet is 5.7M x 159 — a full read would blow up a smoke test)
        batch = next(pq.ParquetFile(path).iter_batches(batch_size=nrows))
        df = batch.to_pandas()
    else:
        df = pd.read_parquet(path)

    return _promote_index(spec, df)


def _promote_index(spec: TableSpec, df: pd.DataFrame) -> pd.DataFrame:
    """Promote index -> column when a PK column is only in the index (e.g.
    ticker_embeddings, whose ticker is the parquet index)."""
    missing = [c for c in spec.pk if c not in df.columns]
    if missing and df.index.nlevels == 1:
        df.index.name = missing[0]
        df = df.reset_index()
    return df


def iter_source_batches(spec: TableSpec, data_dir: Path,
                        batch_size: int = 100_000) -> Iterator[pd.DataFrame]:
    """Stream a source file in row batches so huge tables (cube 5.7M x 159 ~ 7GB)
    never fully materialize in memory. CSVs are small enough to yield in one go."""
    path = data_dir / spec.source
    if path.suffix == ".csv":
        yield _promote_index(spec, pd.read_csv(path))
        return
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        yield _promote_index(spec, batch.to_pandas())
