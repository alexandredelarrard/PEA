# Phase 2 — Diagnostics, measured from the DB ⬜

**Goal**: answer the three acceptance gates (D28) with printed numbers read **from Postgres**, not
from the API, and turn the zero-fill measurement into a per-field rule you approve.

**Prerequisite**: phase 1 verification passed — `fundamentals_sharadar` holds 29 tickers × 3 dimensions.
**Read first**: [README.md](README.md) — D28, D29, and the "measured facts" section.
**Next**: [phase-3-field-map.md](phase-3-field-map.md)

This phase **writes no production data**. It produces one report and one config file.

> ⚠ **This is NOT the SEC check scheme, and must not be wired into it.** `src/validate/`, its
> 35-check `CHECK_REGISTRY`, and the `fundamentals_check*` tables stay pointed at SEC data only
> (D25, and restated by the user on 2026-08-26). Nothing in this phase registers a check, writes a
> `fundamentals_check` row, imports from `src/validate/`, or is invoked by the validator CLI.
> It is a **standalone, read-only diagnostic** with exactly two consumers: your Full-tier purchase
> decision, and phase 3's `sharadar_zero_rules.json`. If implementing it starts to look like adding
> a check to the registry, stop — that is the wrong file.

---

## Why this phase exists

You are deciding whether to buy the Full history tier. That decision needs numbers from the data you
actually have, not from the vendor's disclaimers. Three gates (D28):

1. **Completeness** — no missing quarters per ticker (the spec's acceptance check #1).
2. **No implausible quarters** — the replacement for the spec's check #3, which is dead on arrival:
   ΣARQ == ARY at `+0.000%` on every year measured (README fact 2). Sharadar *constructs* Q4 as
   `ARY − Σ(Q1..Q3)`, so the identity is an identity. The real signal is whether that construction
   ever produces something absurd — the legacy Quandl docs show it yielding **ABT 2011 Q4 revenue of
   −$7.1bn**, annotated as intentional "to ensure that the quarterly and annual financials are
   aligned".
3. **Zero-fill prevalence, per field** — the input to the NULL rule you deferred.

---

## Scope

- **All 29 entitled tickers** for gates 1–3.
- **The 14 overlapping tickers** for anything comparing against SEC:
  `AAPL, AXP, BA, CAT, CSCO, GS, JNJ, JPM, MCD, MSFT, NVDA, PG, UNH, WMT`
  (DJIA-29 ∩ the 54-ticker `fundamentals_history_sec` roster).

⚠ **The CIK-cutover continuity test (D19) cannot run in this phase.** Its three tickers — APA,
GOOGL, ETN — are **none of them in the DJIA**. Write the test, mark it `skipif` on roster coverage,
and record in the phase-4 report that D19 is **unverified until the roster widens**. Do not quietly
drop it.

---

## Changes

### 1. `src/data_extract/utils/fundamentals_sharadar/diagnostics.py`

All reads through `self._context.store` with `columns=` and `where=`/`since=`. No `pd.read_sql`.

- [ ] `gate_completeness(context, tickers) -> pd.DataFrame`
      Per ticker: expected quarter count over its own observed window vs actual ARQ rows; every gap
      listed with its `reportperiod` boundary. A ticker whose history simply starts late is not a
      gap — measure against each ticker's own first row, not a global start.

- [ ] `gate_implausible_quarters(context, tickers) -> pd.DataFrame`
      Per (ticker, field) over ARQ, flag:
      - a **negative** value in a field that cannot be negative (`revenue`, `assets`, `cor`,
        `inventory`, `receivables`, `cashneq`, …) — the ABT failure mode;
      - a quarter whose magnitude exceeds the largest other quarter of the same fiscal year by more
        than the ratio already calibrated for the SEC path
        (`data_extract.fundamentals_periods.max_opposite_sign_q4_ratio: 3.0` — reuse it, do not
        invent a second threshold);
      - a **Q4-position** row that is systematically the outlier, which is the signature of a
        construction artefact rather than a real charge.
      Report the count, and the worst 20 with their actual numbers.

- [ ] `gate_zero_fill(context, tickers) -> pd.DataFrame`
      For each of the 41 documented zero-filled fields: `n_rows`, `n_zero`, `pct_zero`, and
      `n_tickers_all_zero` (a ticker where the field is 0 in **every** row — the strongest signal
      that it means "not applicable" rather than "zero this quarter").
      **Then split by whether the zero is defensible**, using the SEC layer on the 14 overlapping
      tickers: for each zero cell, is the SEC value also 0/absent (→ defensible) or a real number
      (→ **provably wrong**)?

      Expected shape, from the API-side pre-measurement (README fact 3): `deposits` 71%, `rnd` 52%,
      `inventory` 36% will be almost entirely *defensible* — banks have no inventory, retailers no
      R&D. **`intexp` at 25% will not be**: `intexp = 0` for JPM and GS is provably false. If the
      DB-side numbers disagree materially with these, stop — something in phase 1 is wrong.

- [ ] `cross_check_shares(context) -> pd.DataFrame`
      D-decision on `sharesOutstanding ← sharesbas`. Compare `sharesbas` against the SEC layer's
      share count on the 14 overlapping tickers. The question is whether `sharesbas` **sums multiple
      share classes**, which is undocumented and which this repo already solved painfully for 36
      multi-class tickers by summing the cover-page `dei:EntityCommonStockSharesOutstanding`.
      A systematic ratio (not noise) is the answer. Report the per-ticker median ratio.

- [ ] `confirm_sign_conventions(context) -> dict`
      Assert from stored data what was measured from the API: `capex <= 0` throughout, and
      `fcf == ncfo + capex` to the cent. **If either fails, the phase-3 field map is wrong and you
      must stop.** These are cheap assertions that protect an expensive mistake.

- [ ] `confirm_q4_tautology(context, tickers) -> pd.DataFrame`
      ΣARQ vs ARY per (ticker, fiscal year, field). Expected: `0.000%` everywhere. This is not a
      quality check — it is **evidence for the record** that the spec's acceptance check #3 is
      tautological on this vendor and must not be relied on. Report the max absolute deviation.

### 2. `src/data_extract/cli.py`

- [ ] `sharadar-diagnostics` — runs all six, prints a rich summary, writes the markdown report.

### 3. Outputs

- [ ] **`reports/planning/active-tasks/2026-08-26-sharadar-integration/phase-2-findings.md`** —
      the generated report. One section per gate, every claim a number.
- [ ] **`configs/sharadar/sharadar_zero_rules.json`** — the per-field decision, machine-proposed
      and **human-approved**. Shape:

```json
{
  "intexp":    {"rule": "null", "reason": "provably wrong: JPM/GS have material interest expense", "pct_zero": 0.254},
  "inventory": {"rule": "keep", "reason": "not applicable for banks/services; SEC agrees on 14/14", "pct_zero": 0.358}
}
```

Two rules only: `"null"` (treat 0 as unknown) and `"keep"` (0 is a real value). Every field in
`SHARADAR_ZERO_FILLED_FIELDS` must appear — no defaults, no silent omissions. Phase 3 reads it and
fails loudly on a missing field.

⚠ `configs/fundamentals/*.json` in this repo are **hand-formatted**; a `json.dumps` round-trip
reformats the whole file. Write `configs/sharadar/*.json` with a validated emitter or a text splice,
not a naive dump.

---

## Tests

`tests/data_extract/test_sharadar_diagnostics.py` — real data, and each prints its conclusion.

- [ ] `test_completeness_gate_runs` — asserts the frame is non-empty and prints the per-ticker gap count.
- [ ] `test_sign_conventions_hold` — the `capex <= 0` and `fcf == ncfo + capex` assertions. Prints both.
- [ ] `test_q4_identity_is_tautological` — asserts max deviation `< 0.01%` and **prints that this
      means check #3 carries no information**. A test that documents a dead check is worth having.
- [ ] `test_zero_rules_cover_every_flagged_field` — every field in `SHARADAR_ZERO_FILLED_FIELDS`
      has an entry in `sharadar_zero_rules.json`. Prints any missing.
- [ ] `test_cik_cutover_continuity` — written now, `skipif` no cutover ticker is in the roster.
      Prints the skip reason so the gap is visible rather than invisible.

---

## Verification

```bash
PY="$HOME/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe"
rtk "$PY" -m pytest tests/data_extract/test_sharadar_diagnostics.py -v -s
rtk "$PY" -m src data_extract sharadar-diagnostics -c ./configs
```

- [ ] `phase-2-findings.md` exists and every gate has numbers.
- [ ] `sharadar_zero_rules.json` covers all 41 fields, each with a reason.
- [ ] Sign conventions confirmed from stored data.
- [ ] Q4 tautology confirmed (max deviation < 0.01%).
- [ ] The `sharesbas` cross-check reports a per-ticker median ratio, and you have read it.
- [ ] The CIK-cutover test is present and **skipped with a printed reason**.

---

## The decision this phase hands back to you

The report should end with a plain answer to: **is this data good enough to buy the Full tier?**
Written as three findings and a recommendation, not as a table dump. Specifically:

- how many quarters are missing, and whether the gaps are structural or random;
- how many implausible quarters exist, and whether any is a *level* error rather than a construction
  artefact;
- which fields must be NULL-ruled, and what fraction of cells that removes.

**Do not proceed to phase 3 until you have read this and said go.**
