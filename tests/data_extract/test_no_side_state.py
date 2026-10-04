"""No resume decision reads side state under `data/`: a source scan of `src/`."""

from __future__ import annotations

from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
#: The bulk-archive sidecars and their helpers, retired for `resume.archive_worklist`.
_SIDECAR_TOKENS = ("_universe.json", "mark_processed", "pending_periods", "_processed_scope")


def test_no_source_reads_or_writes_a_bulk_sidecar() -> None:
    hits = [
        f"{path.relative_to(SRC.parent)}:{n}: {line.strip()}"
        for path in sorted(SRC.rglob("*.py"))
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if any(token in line for token in _SIDECAR_TOKENS)
    ]
    assert not hits, "bulk sidecar references left in src/:\n" + "\n".join(hits)
    print("\n=== SANITY CHECK: no bulk sidecar in src/ ===")
    print(f"  {len(list(SRC.rglob('*.py')))} source files scanned for {_SIDECAR_TOKENS}: no hit. Validated.")
