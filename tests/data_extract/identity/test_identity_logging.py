"""Logging policy of the identity stage (phase 5, REQ-032, PD-14): every WARNING site is classified.

A WARNING is either the unreviewed action items of the publish review or an operational fault of the inputs (a partial
or unreadable cache); everything the build decides, performs or reviews logs at INFO or DEBUG. The scan reads the
identity build's own modules (the step, `data_extract/utils/identity/`, `identity_flags` and the shared identity
modules) and fails on any WARNING call not listed here with its class. Shared infrastructure the build calls (the
bulk-cache zip reader, the FTD and Notes zip readers, the store) keeps its own operational WARNINGs, out of this scan.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODULES = [
    ROOT / "src" / "data_extract" / "transformers" / "step_extract_identity.py",
    *sorted((ROOT / "src" / "data_extract" / "utils" / "identity").glob("*.py")),
    ROOT / "src" / "utils" / "identity_flags.py",
    ROOT / "src" / "data_extract" / "utils" / "common" / "identity_store.py",
    ROOT / "src" / "data_extract" / "utils" / "common" / "security_lines.py",
    ROOT / "src" / "data_extract" / "utils" / "common" / "identity.py",
]
#: `module.function` -> the class of its WARNING (PD-14).
CLASSIFIED = {
    "step_extract_identity.run_publish": "action: the unreviewed action items of the publish review (REQ-032)",
    "symbol_tenure.scan_form345_cache": "operational: fewer cached Form 3/4/5 quarters than expected, a partial history",
    "symbol_tenure._unreadable_quarter": "operational: a cached Form 3/4/5 zip without its submission member is skipped",
    "symbol_tenure._aggregate_submission": "operational: a cached Form 3/4/5 zip missing columns is skipped",
    "identity_propagate.retag_13f": "operational: a 13F accession whose source cannot be re-read keeps its stored rows",
}


def _warning_sites() -> dict[str, int]:
    """`{module.function: line}` of every `.warning(` / `.warn(` call in the scanned modules."""
    sites: dict[str, int] = {}
    for path in MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        for function in functions:
            calls = [
                c for c in ast.walk(function) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr in ("warning", "warn")
            ]
            sites.update({f"{path.stem}.{function.name}": call.lineno for call in calls})
    return sites


def test_every_identity_warning_site_is_classified() -> None:
    """PD-14: the WARNING sites are exactly the classified ones; the backlog, refused merges, symbol conflicts, stale
    rejections, cold-start states and propagation purges log at INFO or DEBUG."""
    sites = _warning_sites()

    assert set(sites) == set(CLASSIFIED), {"unclassified": sorted(set(sites) - set(CLASSIFIED)), "gone": sorted(set(CLASSIFIED) - set(sites))}
    actions = [site for site, kind in CLASSIFIED.items() if kind.startswith("action")]
    print("\n=== SANITY CHECK: identity WARNING sites (PD-14) ===")
    for site, kind in sorted(CLASSIFIED.items()):
        print(f"  {site:<40} line {sites[site]:>4}  {kind}")
    print(f"  {len(sites)} site(s) over {len(MODULES)} module(s): {len(actions)} action-item block(s), {len(sites) - len(actions)} operational")
    print("  OK: a healthy identity run warns only for unreviewed action items")
