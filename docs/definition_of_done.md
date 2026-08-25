# Definition of done

An **important** task is not finished when the code runs. It is finished when a **report** exists
that a human can read months later and tell what changed, what was measured, and what was left
broken. This file is the contract; the `dod-*-report` skills are the procedure; a `Stop` hook checks
only that a fresh, valid report exists.

Light work triggers nothing: answering a question, reading code, a typo in a doc, a one-file comment
fix. The gate is for work that touches `src/`, `tests/`, `configs/`, `sql/`, `scripts/`, or that runs
a pipeline command.

## Three report types

| Type | The work | Generator |
|---|---|---|
| **MODELLING** | trained/retrained a model, changed a strategy, model or portfolio config | `scripts/dod/modelling_report.py` |
| **DATA** | extraction, aggregation, a new table or field, a validation/audit pass | `scripts/dod/data_profile.py` |
| **REFACTOR** | restructuring, cleanup, a bug fix, docs, tooling — everything else | `scripts/dod/refactor_metrics.py` |

Reports live at `reports/<YYYY-MM-DD>/<slug>__<TYPE>.md` — **one folder per day**, so everything
produced by a day's work (including a MODELLING run's copied plots under
`reports/<YYYY-MM-DD>/assets/<slug>/`) reads, reviews and prunes together. The date is the folder,
so it is not repeated in the filename. Reports are **tracked in git** and pruned at each release
(keep the last one per type plus anything a release note cites).

## The skeleton

Identical for all three types, so one validator covers all of them.

```markdown
---
type: MODELLING | DATA | REFACTOR
session_id: <the Claude session id>
generated_at: <ISO-8601>
baseline: {head_sha: <sha at session start>}
generator: scripts/dod/modelling_report.py@1
---
## 1. Scope
what was asked; files written; commands run; **SAMPLE SCOPE** (which tickers, which window,
full universe vs subset — a metric without its scope is not a measurement)

## 2. Gates
binary PASS/FAIL table. **Any FAIL means the work is NOT done.**

## 3. Metrics
observed numbers only. No pass/fail column.

## 4. Evidence
artifact paths, test output, plot links

## 5. Regressions, gaps and deliberate omissions
MANDATORY, non-empty

## 6. Next actions

```json dod-metrics
{ ..., "content_hash": "sha256:..." }
```
```

## Three rules the validator enforces structurally

**1. Gates are not metrics.** §2 carries a PASS/FAIL column; §3 must not. **`loc` may appear only in
§3.** LOC is an observation, never a target: an agent optimising it would delete this codebase's
load-bearing docstrings — `xs.py` exists to explain why three clip constants must *not* be unified,
and `src/validate/outliers.py::detect_level_outliers` documents two false-positive bugs that a shorter
version reintroduces. `refactor_metrics.py` gate **G6** fails when docstring lines shrink in a
touched file unless §5 justifies it.

**2. §5 cannot be silently empty.** At least one bullet. The only accepted way to say "nothing" is:

```
- None. Checked: <at least 30 characters describing what you actually looked at>
```

**3. Numbers come from the generator, prose from the agent.** The generator emits the fenced
` ```json dod-metrics ` block; the agent writes §1, §5 and §6 and **never edits the block**. Its
`content_hash` is `sha256` over the block's JSON with `content_hash` removed, serialised with sorted
keys and `(",", ":")` separators — the hook recomputes it.

*Honest limit:* an agent can still run a generator on a cherry-picked scope. That is why the scope
arguments are recorded verbatim inside the block, for the human reader.

## Two standing budgets

- **`AGENTS.md` ≤ 70 lines.** It is loaded into every session; every convention wants a line there.
  To add one, remove one or push the detail into `docs/`. Enforced by gate G7 and
  `tests/dod/test_agents_md_budget.py`.
- **A hook is exactly one process.** Stdlib only, `python -S -E`, and it **never shells out to git**.
  A process spawn costs ~450 ms on this machine (Git Bash + Defender + a OneDrive-synced tree), so a
  hook that pipes `git | grep | wc` costs 3–10 s of every single turn. Git belongs to the
  **generators**, which run once per task. Do not "just add a git call" to a hook.

## Escape hatches

The gate must never trap you. In precedence order: the harness's own `stop_hook_active` flag · the
kill switch `.claude/dod-disabled` (presence = off) · `PEA_DOD=off` in the environment · a 2-attempt
cap per session · a blanket `try/except` plus a 3 s wall-clock bail, so a broken hook fails **open**.

`/dod-skip <reason>` records a deliberate, loud skip. If the hook misclassifies your task, say so —
the classification evidence is printed in the refusal. Do not fight the hook.
