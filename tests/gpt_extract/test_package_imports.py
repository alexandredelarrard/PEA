"""
test_package_imports.py  (tests/gpt_extract/test_package_imports.py)
----------------------------------------------------------------------
Every module under `src/gpt_extract/` must import cleanly. This is the test that would
have caught the `gpt_getter.py` IndentationError and the missing `GENERAL_SCHEMAS` before
either shipped: a broken module in this package fails silent until something tries to use
it, because nothing in `src/data_extract/` imports `gpt_extract` yet.
"""
from __future__ import annotations

import importlib
import pkgutil

import src.gpt_extract as gpt_extract


def _submodules() -> list[str]:
    names = [gpt_extract.__name__]
    for module_info in pkgutil.walk_packages(gpt_extract.__path__, prefix=f"{gpt_extract.__name__}."):
        names.append(module_info.name)
    return names


def test_every_module_under_gpt_extract_imports_cleanly():
    names = _submodules()
    assert len(names) >= 4, f"expected transformers/utils submodules, found only {names}"

    for name in names:
        importlib.import_module(name)

    print("\n=== SANITY: src/gpt_extract/ import sweep ===")
    for name in names:
        print(f"  OK  {name}")
    print(f"  {len(names)} module(s) imported cleanly, zero LangChain dependency.")
