"""The `src/modelling` layout and its import rules, read statically from the repository.

Files come from `git ls-files --cached --others --exclude-standard` (tracked or about to be, never
ignored build output), restricted to the ones present on disk. Imports are read with `ast`, so the
check needs no DB, no trained model and imports nothing it inspects.

- layout: `steps/` and `transformers/` hold exactly their modules, `utils/` only `.py` files, and
  `cli.py` is the only top-level module; `long_short/`, `long_book/` and `trend/` are gone.
- removed code: nothing under `src/`, `app/`, `scripts/`, `tests/` or `main.py` imports a deleted
  module or a deleted strategy class.
- layering: `utils` imports neither `transformers` nor `steps`; `transformers` does not import
  `steps`; the scoring consumers (`ls_model`, the app) reach modelling through `utils` only -- a
  pickled member unpickles to its own transformer class, so they never import one.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODELLING = "src/modelling"

EXPECTED_STEPS = {"__init__.py", "step_long_short.py"}
EXPECTED_TRANSFORMERS = {
    "__init__.py",
    "base.py",
    "lightgbm_model.py",
    "random_forest.py",
    "linear_regression.py",
    "monitor.py",
    "backtest.py",
}
REMOVED_MODULES = (
    "src.modelling.long_short",
    "src.modelling.long_book",
    "src.modelling.trend",
    "src.utils.trend",
    "src.strategies.step_long_book",
    "src.strategies.step_trend",
    "src.strategies.analysis.long_book_analysis",
    "src.strategies.analysis.trend_analysis",
)
REMOVED_NAMES = {"LongBookStrategy", "TrendCTAStrategy", "StepModelling"}
SCORING_CONSUMERS = ("src/strategies/utils/ls_model.py", "app/app.py")


def _repo_files(*roots: str) -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", *roots],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git ls-files unavailable: {exc}")
    return sorted({p for p in out.splitlines() if (ROOT / p).is_file()})


def _module_of(path: str) -> str:
    parts = Path(path).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imports(path: str) -> list[tuple[str, set[str]]]:
    """`(module, imported names)` for every import in `path`, relative imports resolved."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
    package = _module_of(path) if path.endswith("__init__.py") else _module_of(path).rpartition(".")[0]
    found: list[tuple[str, set[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, set()) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")[: len(package.split(".")) - (node.level - 1)]
                base = ".".join([*anchor, base] if base else anchor)
            names = {alias.name for alias in node.names}
            found.append((base, names))
            found.extend((f"{base}.{name}", set()) for name in names)  # `from pkg import submodule`
    return found


def _is_under(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def test_layout() -> None:
    files = _repo_files(MODELLING)
    rel = [p[len(MODELLING) + 1 :] for p in files]
    top = {p for p in rel if "/" not in p}
    dirs = {p.split("/", 1)[0] for p in rel if "/" in p}
    steps = {p.split("/", 1)[1] for p in rel if p.startswith("steps/")}
    transformers = {p.split("/", 1)[1] for p in rel if p.startswith("transformers/")}
    utils = {p.split("/", 1)[1] for p in rel if p.startswith("utils/")}

    assert dirs == {"steps", "transformers", "utils"}, f"unexpected sub-packages: {sorted(dirs - {'steps', 'transformers', 'utils'})}"
    assert top == {"cli.py"}, f"unexpected top-level files: {sorted(top - {'cli.py'})}"
    assert steps == EXPECTED_STEPS, f"steps/: extra {sorted(steps - EXPECTED_STEPS)}, missing {sorted(EXPECTED_STEPS - steps)}"
    assert transformers == EXPECTED_TRANSFORMERS, (
        f"transformers/: extra {sorted(transformers - EXPECTED_TRANSFORMERS)}, missing {sorted(EXPECTED_TRANSFORMERS - transformers)}"
    )
    assert utils and all("/" not in p and p.endswith(".py") for p in utils), f"utils/ must hold flat .py modules: {sorted(utils)}"

    print("\n=== SANITY CHECK: src/modelling layout ===")
    print(f"  top: {sorted(top)}; steps/: {sorted(steps)}")
    print(f"  transformers/: {sorted(transformers)}")
    print(f"  utils/: {sorted(utils)}")
    print("  long_short/, long_book/ and trend/ are gone. Validated.")


def test_no_removed_imports() -> None:
    files = [p for p in _repo_files("src", "app", "scripts", "tests", "main.py") if p.endswith(".py")]
    offenders = []
    for path in files:
        for module, names in _imports(path):
            if any(_is_under(module, removed) for removed in REMOVED_MODULES):
                offenders.append(f"{path}: {module}")
            elif hit := names & REMOVED_NAMES:
                offenders.append(f"{path}: from {module} import {sorted(hit)}")
    assert not offenders, "imports of removed code:\n  " + "\n  ".join(offenders)
    print("\n=== SANITY CHECK: no import of removed code ===")
    print(f"  {len(files)} Python files under src/, app/, scripts/, tests/, main.py scanned;")
    print(f"  none imports {', '.join(REMOVED_MODULES)} or {sorted(REMOVED_NAMES)}. Validated.")


def test_layering() -> None:
    rules = [
        (f"{MODELLING}/utils/", ("src.modelling.transformers", "src.modelling.steps")),
        (f"{MODELLING}/transformers/", ("src.modelling.steps",)),
    ]
    files = [p for p in _repo_files(MODELLING) if p.endswith(".py")]
    offenders = [
        f"{path}: {module}"
        for prefix, forbidden in rules
        for path in files
        if path.startswith(prefix)
        for module, _ in _imports(path)
        if any(_is_under(module, f) for f in forbidden)
    ]
    for consumer in SCORING_CONSUMERS:
        offenders += [
            f"{consumer}: {module}"
            for module, _ in _imports(consumer)
            if _is_under(module, "src.modelling") and module != "src.modelling" and not _is_under(module, "src.modelling.utils")
        ]
    assert not offenders, "layering violations:\n  " + "\n  ".join(offenders)
    consumer_imports = {c: sorted({m for m, _ in _imports(c) if _is_under(m, "src.modelling.utils")}) for c in SCORING_CONSUMERS}
    print("\n=== SANITY CHECK: modelling layering ===")
    print("  utils -> neither transformers nor steps; transformers -> not steps;")
    for consumer, mods in consumer_imports.items():
        print(f"  {consumer} reaches modelling only through {mods or 'nothing'}")
    print("  Validated.")
