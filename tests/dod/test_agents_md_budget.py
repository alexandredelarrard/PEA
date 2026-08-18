"""
test_agents_md_budget.py  (tests/dod/test_agents_md_budget.py)
--------------------------------------------------------------
`AGENTS.md` is capped at 70 lines, and the two copies of the contract (the generator's and the
hook's) must not drift apart.

Cheap and permanent. The budget matters because `AGENTS.md` loads into EVERY session, so every
new convention wants a line in it; catching the overrun in CI makes the trade-off explicit --
to add a line, remove one or push the detail into `docs/`.
"""
from __future__ import annotations

from pathlib import Path

from scripts.dod import report_common
from scripts.dod.refactor_metrics import AGENTS_MD_MAX_LINES

AGENTS_MD = Path(__file__).resolve().parents[2] / "AGENTS.md"


def test_agents_md_is_within_budget():
    lines = AGENTS_MD.read_text(encoding="utf-8").splitlines()
    n = len(lines)
    assert n <= AGENTS_MD_MAX_LINES, (
        f"AGENTS.md is {n} lines, budget is {AGENTS_MD_MAX_LINES}. Remove a line or move the "
        f"detail into docs/ — see docs/definition_of_done.md.")
    print(f"\n  AGENTS.md: {n}/{AGENTS_MD_MAX_LINES} lines "
          f"({AGENTS_MD_MAX_LINES - n} spare)")


def test_agents_md_points_at_the_contract():
    """The hard rule has to be reachable from the always-loaded file, or nobody follows it."""
    text = AGENTS_MD.read_text(encoding="utf-8")
    assert "docs/definition_of_done.md" in text
    assert "70 lines" in text


def test_the_gate_and_the_generator_agree_on_the_contract(dod_lib):
    """`dod_lib` duplicates the section headings and the hash rule because a hook must stay
    stdlib-only. The duplication is fine; a DIVERGENCE is not."""
    assert tuple(dod_lib.SECTIONS) == tuple(report_common.SECTIONS)
    assert dod_lib.SECTION_REGRESSIONS == report_common.SECTION_REGRESSIONS
    assert dod_lib.METRICS_FENCE == report_common.METRICS_FENCE
    assert dod_lib.EMPTY_SECTION_5_PREFIX == report_common.EMPTY_SECTION_5_PREFIX
    assert dod_lib.MIN_CHECKED_CHARS == report_common.MIN_CHECKED_CHARS
    assert tuple(dod_lib.KINDS) == tuple(report_common.KINDS)


def test_the_state_dir_formula_agrees(dod_lib, repo_root):
    """The hook WRITES the baseline the generators READ. If these two formulas disagree, every
    report silently falls back to a synthesised baseline and nobody notices."""
    assert dod_lib.state_root(repo_root) == report_common.state_dir(root=repo_root)
    print(f"\n  state dir: {report_common.state_dir(root=repo_root)}")
    print("\n=== SANITY CHECK: AGENTS.md budget + contract duplication ===")
    print("  AGENTS.md is inside its 70-line cap and still names the definition-of-done doc,")
    print("  so the rule is reachable from the always-loaded file.")
    print("  The hook (stdlib-only, .claude/hooks/dod_lib.py) and the generators")
    print("  (scripts/dod/report_common.py) are proven to agree on the section headings, the")
    print("  metrics fence, the section-5 escape clause, the report kinds, AND the")
    print("  out-of-repo state directory -- the duplication is deliberate, the drift is not.")
    print("  Validated.")
