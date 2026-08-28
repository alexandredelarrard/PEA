"""One config directory is ONE cache entry, however it is spelled.

Every loader in the fundamentals family is `@cache`d, and `functools.cache` keys on the
ARGUMENT -- so `load_catalogue()` and `load_catalogue("./configs")` were two entries
pointing at the same directory, and one `StepExtractAllData.run()` therefore parsed the
169 KB catalogue and ran all six of its validation passes twice. Both conventions exist in
the tree (no-arg in `fetch_fundamentals_sec` and `build_history`, explicit in `field_map`
and `validator`), so the fix is to normalise the key inside the loader rather than to pin a
convention nobody enforces.

Synthetic-free by nature: this is about cache identity, not about numbers, so it reads the
real `./configs` and asserts on object identity and miss counts.
"""
from __future__ import annotations

import os

import pytest

from src.data_extract.utils.fundamentals import cik_cutover, periods
from src.data_extract.utils.fundamentals.kpi_catalogue import (
    DEFAULT_CONFIG_DIR, _catalogue_at, load_catalogue, resolve_config_dir)
from src.data_extract.utils.fundamentals_sharadar import field_map

#: The three spellings of one directory that all reach these loaders in the live tree.
SPELLINGS = (None, DEFAULT_CONFIG_DIR, os.path.abspath(DEFAULT_CONFIG_DIR))


@pytest.mark.parametrize(("name", "load", "cached"), [
    ("load_catalogue", load_catalogue, _catalogue_at),
    ("load_guards", periods.load_guards, periods._guards_at),
    ("load_cutovers", cik_cutover.load_cutovers, cik_cutover._cutovers_at),
    ("load_field_map", field_map.load_field_map, field_map._field_map_at),
])
def test_one_directory_is_one_cache_entry(name, load, cached):
    """Three spellings, one parse. `load_field_map` is the one that had NO cache at all
    while calling the cached `load_catalogue` inside itself."""
    cached.cache_clear()
    results = [load(spelling) for spelling in SPELLINGS]

    misses = cached.cache_info().misses
    assert misses == 1, (
        f"{name} parsed its config {misses} times for {len(SPELLINGS)} spellings of one "
        f"directory")
    first = results[0]
    assert all(r is first for r in results), (
        f"{name} returned distinct objects for the same directory")

    print(f"\n=== SANITY CHECK: {name} cache key ===")
    print(f"  spellings asked for: {[str(s) for s in SPELLINGS]}")
    print(f"  cache misses: {misses}; all {len(results)} results are the same object")


def test_resolve_config_dir_is_the_single_normalisation():
    """The one function every loader routes through, so a fifth loader added later has an
    obvious thing to call rather than a convention to remember."""
    resolved = {resolve_config_dir(spelling) for spelling in SPELLINGS}
    assert len(resolved) == 1, f"three spellings resolved to {resolved}"
    assert os.path.isabs(next(iter(resolved))), "the cache key is not an absolute path"

    print("\n=== SANITY CHECK: resolve_config_dir ===")
    print(f"  {[str(s) for s in SPELLINGS]} -> {next(iter(resolved))}")
    print("  One absolute key for one directory. Validated.")
