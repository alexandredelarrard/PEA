---
name: dod-modelling-report
description: >
  Write the MODELLING definition-of-done report after training or retraining a model, changing
  an ensemble member, a horizon, a feature set, a strategy sleeve, or any of
  configs/models.yml / modellling.yml / strategy/*.yml. Also use when the DoD Stop hook says
  "classified MODELLING" or asks for reports/<YYYY-MM-DD>/<slug>__MODELLING.md. Reads the run's
  existing diagnostics; never retrains.
---

# MODELLING definition-of-done report

Contract: [docs/definition_of_done.md](../../../docs/definition_of_done.md). The generator fills
the numbers; you write §1, §5 and §6.

## 1. Find the run — do not retrain

Diagnostics were already written by `StepModelling` to
`<OUTPUT_DIR>/diagnostics/<run_stamp>/`. The generator defaults to the newest run.

```bash
PY="$HOME/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe"
"$PY" scripts/dod/modelling_report.py --slug <kebab-slug>
```

Compare against the previous run whenever one exists — that is the only way gate **M5**
(OOS IC did not regress) can say anything:

```bash
"$PY" scripts/dod/modelling_report.py --slug <slug> --compare-run <previous_run_stamp>
```

If diagnostics are missing, **say so in §5** rather than reporting a model as done without
them. `AGENTS.md`: no model is "done" without TimeSeriesSplit CV + SHAP + printed OOS metrics.

## 2. Read the gates before writing prose

| Gate | Meaning if it FAILS |
|---|---|
| M1 | the run's `kpis.json`/`kpis.csv` are missing or unreadable — the run did not finish |
| M2 | a CV or OOS IC is NaN/inf — usually an empty fold or a horizon with no OOS window |
| M3 | a **booster** member has no SHAP. A linear member (elasticnet) never has SHAP; the gate states that itself and still passes |
| M4 | a booster has no PDP |
| M5 | OOS IC fell more than `--ic-tolerance` (default 0.002) below the compared run |

A FAIL is not something to work around. Either fix it or justify it explicitly in §5.

## 3. Write §1, §5, §6

- **§1** — the generator already filled the run, horizons and sample scope. Add *what was asked*.
- **§5** — mandatory, non-empty. Name the IC that dropped, the horizon you did not check, the
  member you removed, the config knob you changed and did not sweep. If genuinely nothing:
  `- None. Checked: <30+ chars of what you actually looked at>`.
- **§6** — the next experiment, in one line each.

## 4. Rules

- **Never edit the ` ```json dod-metrics ` block** — it carries a `content_hash` the hook
  recomputes. Edit it and the report is rejected.
- IC and Sharpe are **observations**, not targets. Do not tune a threshold so a gate passes.
- Numbers in your prose must match §3. If you want a number that is not there, add it to the
  generator, do not type it by hand.
