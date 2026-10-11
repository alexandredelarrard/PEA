---
title: Cube aggregation
description: Incremental cube-part builders, feature families, target construction, and streamed final assembly.
type: module
tags:
  - wiki
  - module
---
# Cube aggregation

## Summary

`src/data_aggregate` converts normalized extract tables into a wide, point-in-time modelling cube. Seven domain sub-steps persist eight intermediate parts through the price maximum date, and a final assembler reports any part-edge mismatch before merging features, betas, peers, and wide targets at one row per date and ticker.

## Responsibilities

- Normalize prices and define the trading grid.
- Build forward targets and rolling factor loadings.
- Build fundamental, momentum, text, institutional, and governance feature panels.
- Apply peer-relative and cross-sectional transformations without future leakage, preserving binary states and structural zeros.
- Match financial changes by fiscal period, enforce publication-time visibility, and expire stale point-in-time values by source cadence.
- Recompute guarded trailing windows and detect schema drift.
- Warn on every missing, behind, or ahead registered part during assembly, while `cube-status` fails closed on any non-exact price-edge alignment.
- Stream the final cube to the store without materializing all output chunks twice.

## Public API / entry points

- `StepBuildCube.run()` and `cube_parts_status()` in [step_build_cube.py](../../src/data_aggregate/step_build_cube.py).
- Eight part commands plus status in [data_aggregate/cli.py](../../src/data_aggregate/cli.py).
- Domain sub-step classes live under `src/data_aggregate/transformers`; [step_cube_prices.py](../../src/data_aggregate/transformers/step_cube_prices.py) and [step_cube_institutionals.py](../../src/data_aggregate/transformers/step_cube_institutionals.py) show the shared orchestration contract.
- `StepCubeInstitutionals.run()` and `build_panel()` in [step_cube_institutionals.py](../../src/data_aggregate/transformers/step_cube_institutionals.py) are the institutional part's persistence and in-memory panel entry points.

## Key files

- [parts.py](../../src/data_aggregate/utils/common/parts.py) is the single part registry.
- [incremental.py](../../src/data_aggregate/utils/common/incremental.py) plans refresh windows and writes inclusive tails.
- [price_frames.py](../../src/data_aggregate/utils/common/price_frames.py), [pit.py](../../src/data_aggregate/utils/common/pit.py), [panel.py](../../src/data_aggregate/utils/common/panel.py), and [xs.py](../../src/data_aggregate/utils/common/xs.py) hold shared contracts. `pit.py` owns fiscal-period matching and source-age projection; `panel.py` applies declared continuous, binary, and structural-zero transform semantics.
- [step_cube_institutionals.py](../../src/data_aggregate/transformers/step_cube_institutionals.py) keeps the ordered institutional merge and persistence contract; [inputs.py](../../src/data_aggregate/utils/institutionals/inputs.py) owns projected and universe-scoped reads, [frontiers.py](../../src/data_aggregate/utils/institutionals/frontiers.py) resolves completeness boundaries, and [sink.py](../../src/data_aggregate/utils/institutionals/sink.py) carries source events and availability into the two derived panels.
- [step_assemble_cube.py](../../src/data_aggregate/transformers/step_assemble_cube.py) logs registry-wide part-versus-price edge warnings, left-joins wide targets onto the feature-led base, and writes chunks. [part_status.py](../../src/data_aggregate/utils/common/part_status.py) is the exact-alignment readiness gate.
- [short_flow_features.py](../../src/data_aggregate/utils/institutionals/short_flow_features.py) publishes FTD values atomically by the persisted source ZIP period, deriving availability from period end plus 15 days or a fresh latest-ZIP cache timestamp. It advances weekend/holiday availability to the next trading session and preserves unavailable `NaN` cells.

- [configs/build_cube.yml](../../configs/build_cube.yml) owns windows, targets, feature settings, and output switches.

## Earnings-call feature contract

`StepCubeText` combines cached sentiment and embedding call-grain metrics into exactly 12 model-facing `f_ec_*` columns in `cube_part_text`: eight raw values and four prior-only issuer-history scores. Earnings-call features deliberately do not use peer or cross-sectional normalization because transcript availability is sparse and non-synchronous. `ec_tone_delta`, `ec_length_delta` and the two embedding distances compare with the prior **consecutive** fiscal quarter and are null when that quarter has no call.

The text step reads raw paragraphs from `earnings_call_sections` (projected, scoped by ticker, ordered by paragraph) and cuts each call with `src/utils/earnings_call_split.py`, the one split shared by sentiment, embeddings and the validator:

- **Turns and Q&A start.** Same-speaker paragraphs merge into turns; placeholder speakers and texts are dropped. The Q&A starts at a strong operator or host hand-off that names the asker, or, once management has spoken 150 words, at a short Q&A-opening line or at a new speaker's short question.
- **Roles.** Each turn is labelled prepared, question or answer, with `exchange_idx` linking a question to its answers. A speaker who already asked a question stays a questioner.
- **Cleaning.** The sentence splitter never cuts decimals or abbreviations (`3.5%`, `U.S.`). Courtesy, backchannel, self-introduction, addressed-name and hand-off sentences are removed everywhere. Safe-harbor and IR boilerplate and the participant roster are removed from prepared remarks. Hedges are kept. Embedded question and answer turns need at least 10 cleaned words.
- **Sentiment inputs.** FinBERT scores the cleaned prepared remarks and a Q&A text made of **management answers only**.

Measured on samples of 585 and 2,940 calls: split `ok` on 99.3 %; analyst text in the Q&A sentiment string 27.1 % → 0 and operator text 4.2 % → 0; boilerplate 5.0 % → 1.55 % of prepared words (the rest carries figures); broken decimals 95,260 → 171 (the 171 are broken in the source). Removal precision is 19–20 of 20 hand-checked hits per rule. The split costs about 82 ms per call. Evidence: `reports/validate/2026-10-01-earnings-call-extraction-gaps/03-implementation.md`.

A call yields features only when its split status is `ok` and its cleaned texts pass `assess_earnings_call_sections` (both sections present, at least 100 cleaned words). Cached sentiment must be complete, cheap word and uncertainty metrics are refreshed from the current text, and embeddings are left-joined only onto quality-valid calls. A call with no prepared management text (for example a live-on-X call) or no management answer text drops out. A valid call becomes visible on the next trading session, remains available for 66 trading sessions, and then expires to null. Missing or malformed quarters remain null; a mathematically observed zero remains zero. Full and incremental runs both calculate on the complete trading calendar, while the shared writer alone slices the incremental refresh tail.

Sentiment rows carry the cache version `speaker-clean-v2` and embedding rows the model tag `text-embedding-3-small:speaker-clean-v2`; rows under another tag are rescored or re-embedded by the normal incremental pass. When a re-issued source call arrives, a durable pending marker carries its call date across the extract/aggregate boundary and is acknowledged only after the inclusive cube repair succeeds.

## Governance cleaning stage

`StepCubeGovernance` loads the raw governance tables (`def14a_llm`, `def14a_executive_comp`, `def14a_directors`, `def14a_director_comp`, `sec_8k_votes`, and `def14a_ownership` projected to ticker, accession, holder type and shares). It cleans them once with `clean_governance_inputs` in [clean.py](../../src/data_aggregate/utils/governance/clean.py). `build_governance_feature_panel` only builds features from the cleaned bundle. No impute, carry, repair or reconcile helper is called anywhere else; `test_governance_clean_stage.py` guards this.

The stage runs in a fixed order: child fills, board merge, `impute_def14a` (carries, identities, CEO stints), board provenance, ownership basis repair, economic ownership, ownership estimate, same-filing SCT pay growth, historical CEO pay backfill, say-on-pay vote fallback. A cleaned bundle is returned unchanged on a second call.

- **Provenance.** Every proxy column the stage writes carries `<col>_source_as_of` (the filing the value comes from) and `<col>_fill`: `filed`, `identity`, `carry`, `person`, `cross_source`, `estimate`, `backfill` or `interp`. A fill that uses a later filing also carries `<col>_evidence_as_of`. `staleness.expire_stale` ages a level by `_source_as_of`, so a carried value expires 1,095 days after its true source, not after the proxy it was carried onto.
- **CEO stints.** `names.ceo_stint_ids` splits a ticker's proxies on a change of CEO identity or of generational suffix (Jr., Sr., II, III); `person_key` is unchanged. `ceo_since_year` takes the value at or before the stint's first proxy year that the most filings of the stint disclose (ties go to the largest), else stays as filed. `ceo_is_founder` takes the stint's mode. Both are kind `person`, dated at the row's own `as_of`, and keep the filed value in `<col>_raw`. `ceo_is_board_chair` is carried only within a stint.
- **Ownership estimate.** A NULL `insider_ownership_pct` on a ticker never flagged dual-class is the sum of `director_officer` shares in `def14a_ownership` over point-in-time `sharesOutstandingPit`. Holder rows whose name reads as an entity (Inc, LLC, L.P., Trust, Fund, Group, Holdings, Capital, Vanguard, BlackRock, State Street, the "as a group" total, ...) are not summed: the extraction tags some 5% holders `director_officer`. It needs at least 5 remaining holders, a result in (0, 1] and at most 3x the ticker's largest filed value, and is kind `estimate`. Filed, computed and carried values are never overwritten.
- **Say-on-pay.** `f_say_on_pay_support` is gone; the 8-K vote series is the single say-on-pay source. A proxy with a filed support value and no 8-K tally in the prior 450 days adds one vote row (for = support, against = 1 − support, `source = proxy`), dated at the proxy.
- **SCT pay growth.** `ceo_comp_growth_sct` is log(total FY / total FY−1) from the CEO's row in the filing's own Summary Compensation Table, filer-stated totals only. `ceo_comp_growth_1y` and `ceo_pay_growth` prefer it and fall back to the cross-filing pair; the CEO-change mask still applies. A NULL parent `ceo_total_comp` takes the CEO's same-year SCT total (kind `identity`).

> ⚠ **Historical pay backfill is a deliberate lookahead.** A NULL CEO pay component or total is filled from a LATER proxy's SCT for the same CEO and fiscal year (`backfill`), or log-linearly inside one stint across at most 2 missing proxies (`interp`). In live data a value missing today is not filled this way, so history is more complete than the present. The step logs the counts as one WARNING. The lookahead is concentrated in early history: the share of proxies with at least one pay cell from a later filing is 35-54% in 2000-2005, 15-20% in 2006-2011 and 4-7% from 2012 (0 on any ticker's latest proxy), so a backtest on 2000-2011 sees denser pay history than live data and a train/test split straddling 2006 mixes the regimes. Two smaller lookaheads are also accepted: stint `person` values come from the stint's other filings, and the estimate's "ever dual-class" test and its filed-value ceiling read later proxies (they can only refuse a fill).

Every rule's count is logged as `governance clean | <stage>|<rule>: n`; `test_clean_stage_tally_keys` pins the stage names. Evidence: `reports/validate/2026-10-10-governance-fill-review/`.

## Dependencies

The module reads through [DataStore](./data-store.md), loads peer dictionaries from [peer deduction](./data-peers.md), and consumes the full set of normalized extract tables from [data extraction](./data-extract.md).

## Participates in

The module is the core of the [cube-build flow](../flows/cube-build.md). Its output feeds [modelling](./modelling.md), while its registry and incremental rules define the [cube-parts concept](../concepts/cube-parts.md).

## Related

- [Add a cube feature](../guides/add-a-cube-feature.md)
- [Point-in-time and quality controls](../architecture/point-in-time-and-quality.md)
- [Data access and storage](../guides/data-access.md)
