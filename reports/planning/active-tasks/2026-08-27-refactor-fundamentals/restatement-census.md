# Restatement census — would vintaging `quarterize` pay?

**Deliverable of [phase 3](phase-3-efficiency-memoisation.md) §3.5.** Measured 2026-08-28 on
the 8 frozen replay-sample tickers (MCD, ORCL, BA, BAC, KR, BRK-B, APA, VRT) at **full
history** — 9,345 windows across 517 publication events.

**Recommendation, up front: LEAVE IT.** Do not build the vintage redesign
([deferred.md](deferred.md) D-3). The measured restatement rate is **4–5x** the threshold the
plan pre-registered as the level at which a hybrid becomes attractive, and the fraction of the
work it could actually avoid is **~30 %**, not the ~98 % the design implicitly assumes.

## What was asked, and what "indicative" means here

Phase 3 §3.5 scoped this to the 8 sample tickers rather than the whole table, because
`fundamentals_facts` was being written while the plan was drafted. That is no longer true —
the facts walk finished and the table holds 491 tickers / 2,998,225 rows — but the census was
still run on the 8, because the frozen tier-B parquet is the input every other Phase 3 gate
used and re-freezing mid-phase is exactly what Phase 0 forbids.

**So: indicative, and biased toward restatement.** BAC and APA are in the sample *because*
they are known restaters. Treat 8.2 % as an upper-middle estimate, not a universe mean. The
whole-table re-run is in [post-run-checklist.md](post-run-checklist.md). It is unlikely to
change the recommendation: the number would have to fall below ~2 % to do that, and the
*cleanest* filer in this sample is already at 3.2 %.

## 1. Value vintages

A window's identity is `_latest_per_window`'s: its END, within `_SAME_PERIOD_DAYS` (7 days),
after sorting on `_WINDOW_ORDER`. A window has *n* vintages when *n* distinct `filing_date`s
report it. "Changed" is any difference beyond float noise (relative 1e-9); "material" is
relative > 1e-3, the level at which a restatement moves a peer z-score.

| measure | value |
|---|---|
| windows | 9,345 |
| with **> 1 vintage** | 8,107 (**86.8 %**) |
| with a **value-changing** vintage | 940 (**10.06 %**) |
| with a **material** (> 0.1 %) change | 768 (**8.22 %**) |
| vintages per window | median 2, mean 2.2, max 8 |

That 86.8 % is the expected part and is not itself a problem: a filer re-tags last year's
window as a comparative in every subsequent filing, and `_latest_per_window` exists precisely
to pick one. **The 8.22 % is the problem**, because those are the windows where "which vintage
you took" changes the number.

### Per filer

| ticker | windows | > 1 vintage | value-changing | material | max vintages |
|---|---|---|---|---|---|
| APA | 1,090 | 85.0 % | 19.54 % | **15.50 %** | 6 |
| BAC | 1,219 | 88.0 % | 15.01 % | **13.70 %** | 4 |
| VRT | 634 | 79.2 % | 8.83 % | 8.36 % | 8 |
| BA | 1,453 | 87.6 % | 10.19 % | 8.12 % | 4 |
| ORCL | 1,421 | 87.5 % | 6.05 % | 5.77 % | 3 |
| KR | 1,331 | 88.7 % | 6.24 % | 5.56 % | 8 |
| BRK-B | 790 | 85.2 % | 9.75 % | 7.59 % | 6 |
| MCD | 1,407 | 87.7 % | 6.68 % | **3.20 %** | 4 |

Worst is **APA at 15.5 %**; best is **MCD at 3.2 %**. Note the plan expected BAC to be the
upper bound — APA is worse, and APA was in the sample for an unrelated reason (a value that
once landed as the string `'1997000000.0'`). The spread across filers is a factor of ~5, so a
single global threshold would be the wrong shape for a hybrid anyway.

### Which fields, and how late

Material restatements land on the headline income-statement lines, not on obscure ones:

| field | material windows |
|---|---|
| `totalRevenue` | 128 |
| `incomeTaxExpense` | 77 |
| `pretaxIncome` | 74 |
| `netIncome` | 71 |
| `operatingIncome` | 66 |
| `operatingCashFlow` | 55 |
| `costOfRevenue` | 47 |
| `interestExpense` | 41 |
| `depAmort` | 40 |
| `basicShares` | 39 |

By shape: 351 quarterly, 194 annual, 118 ytd6, 105 ytd9 — every shape the ladder consumes.

**Lag from the first vintage to the value-changing one: median 364 days (1.0 year)**, p90 721
days, max 736. That is the signature of re-presentation in the next annual report, and it is
the reason a truncated sample understates the problem: tier A's 16-filing cut is ~4 years, so
it sees the effect, but any window in the last year of a filer's history has not yet had the
chance to be restated. The true rate is therefore a little **higher** than 8.22 %.

### The number that actually kills the hybrid

The plan's hybrid is "slice for clean fields, replay for dirty ones". Its unit is a field, not
a window, so the window rate is not the operative number:

| grouping | clean (never materially restated) | must be replayed |
|---|---|---|
| `(ticker, field)` pairs | 32 of 108 (**29.6 %**) | **70.4 %** |
| fields, ticker-agnostic | 1 of 19 (**5 %**) | **95 %** |

Only **`rentalIncome`** is never materially restated on any of the 8, and it is a niche REIT
line. So a per-field rule saves ~5 % of the work; even a per-`(ticker, field)` rule — which
means carrying a 491 x 48 cleanliness matrix, keeping it current, and being *wrong* in the
look-ahead direction the first time a clean pair restates — saves ~30 %.

Against that: `build_ticker` is already down to ~3,272 s of CPU for 8 full histories after
Phase 3, and Phase 4's process pool is a **4x** on the same work for a fraction of the risk.

## 2. Refusal flips

This is the half §3.5 said "decides whether vintaged refusals are tractable", and it is the
more damning of the two.

Per `(field, window, dc_code)`, the events at which the period engine refused it. A
**reversal** is a refusal that stops being emitted at a later event: the engine changed its
verdict on the same window as the prefix grew.

| ticker | events | refused triples | reversed | rate |
|---|---|---|---|---|
| APA | 69 | 9 | 2 | 22 % |
| BA | 69 | 2 | 1 | 50 % |
| BAC | 69 | 3 | 1 | 33 % |
| VRT | 34 | 4 | 2 | 50 % |
| MCD | 69 | 14 | 9 | 64 % |
| ORCL | 68 | 14 | 9 | 64 % |
| BRK-B | 69 | 9 | 6 | 67 % |
| KR | 70 | 7 | 7 | **100 %** |
| **TOTAL** | **517** | **62** | **37** | **60 %** |

| `dc_code` | refused | reversed |
|---|---|---|
| `split_basis_mismatch` | 25 | 18 (**72 %**) |
| `derived_basis_mismatch` | 13 | 8 (**62 %**) |
| `derived_sign_implausible` | 24 | 11 (**46 %**) |

**Three in five refusals are not stable facts about a window — they are facts about a
window *and* a visible set.** That is exactly what §3.1 Reason 3 predicted and it is
quantified here for the first time: `_is_ambiguous_duration` reads the `ends`/`values` arrays
of the visible nine-month cumulatives, and `_drop_annual_masquerading_as_quarter` compares
against the largest quarter observed *so far*, so both legitimately change their minds when a
later filing supplies the missing comparator. A vintaged design has to give each refusal a
`known_from` — and a verdict that reverses has no single one.

**State the base rate honestly**: 62 refused triples across 8 filers and 517 events is a small
population, so 60 % is a rate on a small base and the per-ticker rates (22 %–100 %) scatter
accordingly. But it is not a rounding artefact: the reversal fraction is above 45 % for every
`dc_code`, and the two most common codes are the two the plan named as the hard case.

## 3. Recommendation

**LEAVE IT.** Recorded against D-3.

Both halves fail their pre-registered test, independently:

- **Values.** 8.22 % of windows carry a material restatement against a "< 2 % makes a hybrid
  attractive" threshold, and the unit the hybrid would actually operate on — the
  `(ticker, field)` pair — is **70 % dirty**. The saving is ~30 %, not the ~98 % the design
  implicitly assumes.
- **Refusals.** 60 % of refusals reverse as the prefix grows, so the majority of them cannot
  be given a `known_from` at all. This is the harder blocker: the value half could in
  principle be salvaged with a cleanliness matrix, but a refusal that changes its verdict has
  no vintage to carry.

The redesign therefore trades a provably-correct point-in-time replay for a ~30 % saving on
the value path and an unsolved problem on the refusal path — in the one module where the
failure mode is publishing a restated number at a date before it was filed, which is the
look-ahead leak §3.1 rejected slicing over in the first place.

**Do Phase 4's process pool instead**: 4x on the same work, no semantic risk, and it is
already planned. Revisit D-3 only if the whole-table census comes back under ~2 % on BOTH
measures — and note the cleanest filer in this sample is already at 3.2 % on the value half,
so that is unlikely.
