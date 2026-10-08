"""
batching.py  (src/utils/batching.py)
------------------------------------------------------------------------------------
Fixed-size chunking that runs on Python 3.12 (Airflow) and 3.13 (the project venv).

`itertools.batched` takes `strict=` only from 3.13, while ruff's B911 demands it be explicit;
the two cannot both be satisfied, so every caller chunks through `batched_tuples` instead.

Lives in `src/utils/` because several `src/` subfolders chunk keys and `src/` subfolders must
not import from one another.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice


def batched_tuples[T](values: Iterable[T], size: int) -> Iterator[tuple[T, ...]]:
    """Consecutive tuples of `size` items; the last one may be shorter."""
    if size < 1:
        raise ValueError("size must be at least 1")
    iterator = iter(values)
    while chunk := tuple(islice(iterator, size)):
        yield chunk
