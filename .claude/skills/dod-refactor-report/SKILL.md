---
name: dod-refactor-report
description: >
  Write the REFACTOR definition-of-done report after a restructure, cleanup, rename, bug fix,
  dependency change, test or tooling change, or any substantial edit that is not modelling and
  not data work. Also use when the DoD Stop hook says "classified REFACTOR" or asks for
  reports/<YYYY-MM-DD>/<slug>__REFACTOR.md.
---

# REFACTOR definition-of-done report

Contract: [docs/definition_of_done.md](../../../docs/definition_of_done.md). The generator fills
the numbers; you write §1, §5 and §6.

## 1. Run the generator

```bash
PY="$HOME/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe"
"$PY" scripts/dod/refactor_metrics.py --slug <kebab-slug> --tests tests/path/test_x.py
```

`--tests` is what makes **G1** a real check. Without it the generator infers `tests/…/test_*.py`
from the touched paths, and reports N/A if it finds none — an N/A G1 on a code refactor is a gap
you must name in §5.

## 2. Read the gates

| Gate | Meaning if it FAILS |
|---|---|
| G1 | your tests failed, or they passed but printed **no sanity-check conclusion** (`AGENTS.md`) |
| G2 | `src/data_store/` changed and `tests/data_store/test_store_boundary.py` is red |
| G3 | you added a `print(` under `src/` — use `self._log` / `context.log` |
| G4 | a public name disappeared while something still references it |
| G5 | `src/` changed but `AGENTS.md`/`README.md`/`docs/*.md` did not |
| G6 | **docstring lines shrank** in a touched file |
| G7 | `AGENTS.md` exceeds **70 lines** |

## 3. The three warnings that matter most here

**LOC is an observation, never a target.** It appears only in §3 and has no verdict column. Do
not set out to make files shorter. G6 exists because this codebase's docstrings carry
information that is nowhere else: `src/utils/xs.py` explains why three clip constants must *not*
be unified, and `src/validate/analyze_history.py::detect_level_outliers` documents two
false-positive bugs that a shorter version reintroduces. If you did remove docstring lines, that
is allowed — say which, and why they were not load-bearing, in §5.

**`AGENTS.md` is capped at 70 lines.** It loads into every session. To add a line, remove one or
push the detail into `docs/`. Do not "briefly" exceed it.

**Some duplication is deliberate.** `docs/coding_standard.md` and several docstrings document
places where two similar blocks must stay separate. The duplication number in §3 is a prompt to
go read those docstrings, not a mandate to merge.

## 4. Write §1, §5, §6

- **§1** — files and baseline are filled in. Add *what was asked*.
- **§5** — mandatory, non-empty. The call site you did not update, the test you did not run, the
  behaviour you knowingly changed, any gate that is N/A and shouldn't be. If genuinely nothing:
  `- None. Checked: <30+ chars>`.
- **§6** — next actions.

**Never edit the ` ```json dod-metrics ` block** — it is hash-checked.
