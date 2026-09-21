"""
io.py  (src/validate/io.py)
--------------------------------------------------------------------------------------------
The only module in this package that touches disk. Every artifact a check produces lands
under `reports/validate/<slug>/{_cache,_out,plots}/`, passed in as `--out`; **nothing is ever
written under `src/`**, which is what the deleted `src/validate/{_cache,_out,plots}/`
directories were doing.

`pull` materialises one dated snapshot the whole sweep reuses. Measured stream throughput on
the live `pea_db` is 9.5k-15.3k rows/s, so a 2.6M-row part costs ~215-280 s per pass; nine
checks each re-reading it is most of an hour of pure I/O.

⚠ FLOAT32 IS THE CACHE'S DTYPE, AND TWO CHECKS MUST NOT USE IT. It halves the footprint and
sits far inside every threshold in this package, but an exact-equality test ("one column under
two names") and a grain test both ask whether two float64 values are bit-identical -- and a
float32 round-trip manufactures equalities that are not there. `redundancy` and `grain` read
float64 from the DB, cache or not.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.context import Context
from src.data_store.schema import Table, name_of, resolve
from src.validate.result import CheckResult, jsonable

log = logging.getLogger(__name__)

#: Rows per streamed chunk, and the unit of peak memory for every streaming check.
#:
#: ⚠ NOT the store's 200,000 default. Peak working set for a `pull` of
#: `cube_part_institutionals` (129 columns) measured 2.71 GB at 200,000 and 0.88 GB at 50,000
#: -- psycopg2's server-side cursor materialises `max_row_buffer` rows as Python objects
#: before pandas ever sees them, so the cost is rows x columns x ~24B of boxed floats, and it
#: scales with this number. 50,000 keeps the widest table (249 columns) inside the 2 GB budget
#: this package is built to.
CHUNK_ROWS = 50_000

CACHE_DIR, OUT_DIR, PLOTS_DIR = "_cache", "_out", "plots"


def run_dir(out: str | Path) -> Path:
    """`<out>/{_cache,_out,plots}`, created. Returns `<out>`."""
    root = Path(out)
    for sub in (CACHE_DIR, OUT_DIR, PLOTS_DIR):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def write_result(out: str | Path, result: CheckResult) -> Path:
    """`<out>/_out/<check>.json`, the file `data-check.md`'s report is assembled from."""
    path = run_dir(out) / OUT_DIR / f"{result.check}.json"
    path.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")
    return path


def cache_path(out: str | Path, table: Table | str) -> Path:
    return Path(out) / CACHE_DIR / f"{name_of(table)}.parquet"


def meta_path(out: str | Path) -> Path:
    return Path(out) / CACHE_DIR / "meta.json"


def _coerce(frame: pd.DataFrame, policy: dict[str, str]) -> pd.DataFrame:
    """Give every column the same dtype in every chunk, then float64 -> float32.

    ⚠ PANDAS INFERS DTYPES PER CHUNK, AND THAT BREAKS A CHUNKED PARQUET WRITE. A `double
    precision` column that happens to be entirely NULL inside one 200k-row window comes back
    as `object`, not `float64` -- measured on `cube_part_institutionals`, chunk 1, five legs
    (`f_ic_act_percent_of_class` and four siblings), every one of them `double precision` in
    Postgres. The writer then rejects chunk 1 for a schema mismatch against chunk 0.

    So the dtype is decided ONCE, by the first chunk that carries a non-null value, and every
    chunk is cast to it. A column that is null in every chunk is written as float: it holds no
    value to distort, and `profile` files it as a dead leg regardless.

    float32 is the storage dtype for floats -- see the module docstring for the two checks
    that must therefore not read this cache.
    """
    casts: dict[str, str] = {}
    for column in frame.columns:
        series = frame[column]
        if series.dtype == object:
            if column not in policy:
                present = series.dropna()
                policy[column] = ("object" if not present.empty
                                  and pd.to_numeric(present, errors="coerce").isna().any()
                                  else "float32")
            casts[column] = policy[column]
        elif str(series.dtype) == "float64":
            policy.setdefault(column, "float32")
            casts[column] = "float32"
    return frame.astype(casts) if casts else frame


def pull(context: Context, table: Table | str, out: str | Path, *,
         chunksize: int = CHUNK_ROWS) -> Path:
    """Stream `table` to `<out>/_cache/<table>.parquet` and record the pull in `meta.json`.

    Written chunk-by-chunk through a `ParquetWriter`: concatenating the chunks first would
    materialise the very frame this package exists to avoid (`cube_part_fundamentals` is
    5,369 MB on disk).
    """
    spec = resolve(table)
    root = run_dir(out)
    columns = context.store.columns(spec)
    if not columns:
        raise ValueError(f"{spec.name} has no columns -- is it built?")

    target = cache_path(root, spec)
    writer: pq.ParquetWriter | None = None
    policy: dict[str, str] = {}
    schema: pa.Schema | None = None
    rows = 0
    try:
        for chunk in context.store.iter_load(spec, columns=columns, chunksize=chunksize):
            chunk = _coerce(chunk, policy)
            batch = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
            if writer is None:
                schema = batch.schema
                writer = pq.ParquetWriter(target, schema, compression="snappy")
            writer.write_table(batch)
            rows += len(chunk)
            log.info("pull %s: %s rows written", spec.name, f"{rows:,}")
    finally:
        if writer is not None:
            writer.close()

    lo, hi = context.store.bounds(spec) if spec.date_col else (None, None)
    meta_path(root).write_text(json.dumps(jsonable({
        "table": spec.name,
        "pulled_at": pd.Timestamp.utcnow(),
        "rows": rows,
        "columns": columns,
        "pk": list(spec.pk),
        "date_col": spec.date_col,
        "first_date": lo,
        "last_date": hi,
        "max_date": context.store.max_date(spec) if spec.date_col else None,
        "float_dtype": "float32",
    }), indent=2), encoding="utf-8")
    return target


def read_meta(out: str | Path) -> dict | None:
    path = meta_path(out)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_cache(out: str | Path, table: Table | str, *,
               columns: Sequence[str] | None = None) -> pd.DataFrame | None:
    """The cached snapshot, or `None` when no `pull` has run into `<out>`.

    `columns=` is pushed into pyarrow, so a group-of-8 profile pass reads 8 columns off disk
    rather than 249."""
    path = cache_path(out, table)
    if not path.exists():
        return None
    return pq.read_table(path, columns=list(columns) if columns else None).to_pandas()


def iter_source(context: Context, table: Table | str, columns: Sequence[str], *,
                cache: str | Path | None = None,
                chunksize: int = CHUNK_ROWS) -> Iterator[pd.DataFrame]:
    """Chunks of `columns`, from the cache when one exists and from the DB otherwise.

    The single read path every streaming check uses, so "did this run read the cache?" has
    one answer per run rather than one per check."""
    if cache is not None and cache_path(cache, table).exists():
        parquet = pq.ParquetFile(cache_path(cache, table))
        for batch in parquet.iter_batches(batch_size=chunksize, columns=list(columns)):
            yield batch.to_pandas()
        return
    yield from context.store.iter_load(table, columns=list(columns), chunksize=chunksize)
