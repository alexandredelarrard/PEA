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

## Dependencies

The module reads through [DataStore](./data-store.md), loads peer dictionaries from [peer deduction](./data-peers.md), and consumes the full set of normalized extract tables from [data extraction](./data-extract.md).

## Participates in

The module is the core of the [cube-build flow](../flows/cube-build.md). Its output feeds [modelling](./modelling.md), while its registry and incremental rules define the [cube-parts concept](../concepts/cube-parts.md).

## Related

- [Add a cube feature](../guides/add-a-cube-feature.md)
- [Point-in-time and quality controls](../architecture/point-in-time-and-quality.md)
- [Data access and storage](../guides/data-access.md)

## SEC 13F fallback contract — isolated P2 candidate

The approved candidate in .worktrees/institutional-13f-pit uses the existing shared value_basis helper for both broad institutional and elite-manager panels. Missing/unusable common dollars, including value==shares, use positive reported shares × an eligible restored report-quarter-end as-traded quote. Zero/missing common quantities with positive filed values use once-normalized dollars / that quote. Inferred quantities never vote for units or rescale positive finite reported shares. An unsupported unit uses an explicit stored-dollar assumption flag/warning for inverse quantities. Quotes are nonfuture and at most seven calendar days old; no aggregation network fetch.

Unusable common legs with no quote are null/flagged; whole rows are excluded only when no supported typed sibling exposure remains. Option/debt siblings survive, leaving unsupported common totals conservatively unknown. Structural common zeros on option-only rows survive. Both builders log forward/inverse estimates, unverified units, whole-row/common exclusions and unresolved prices; all-excluded inputs return empty. Missing value columns reach fallback. Source data and publication clocks remain unchanged. Per-leg local float operands preserve pandas nullable missingness instead of converting it to zero.

This isolated candidate is not activated in production.50final controls and three complete quarter replays pass; the read-only vintage validator checks24recovered accessions separately. A unit1book can still contain materially wrong individual prices: BMO's84/473priced Q3positions account for73.06% of priced-slice filed dollars. Row-level value quality remains separate from units and completeness. Evidence is the local ignored reports/validate/2026-10-09-institutional-13f-audit/p2-fallbacks.html and its source/ratio artifacts. P3production adoption, P4genuine share basis and daily refresh remain separate.

## Ordinary immutable 13F consumers — isolated P5 candidate

The candidate in .worktrees/institutional-13f-pit now changes the actual broad and elite methods of StepCubeInstitutionals, rather than only a preview. Both load completed Tables.sec13f_filings and streamed whole-book Tables.sec13f_holdings_vintages. Resolve versions on their public clock, keep the greatest usable report quarter active, and compare the same eligible manager books with the latest public version of the adjacent prior quarter. New managers contribute levels without fabricating a prior comparison; older amendments change a still-used baseline prospectively.

Missing/restricted books and unsupported quantities cannot establish exits. A complete public empty book can. An inferred typed-leg omission remains unknown. Removed names emit zero/null to clear daily carry, option-only names retain their option feature, and elite expiry is scheduled from the existing four-quarter public clock even without a later filing. Date-only filings use the next session with the existing three-session deadline policy; complete same-day acceptance clocks determine order, otherwise original/amendment metadata supplies the fallback.

Normal source extraction now writes verified accession packets before compatibility-table reduction, including full selected ZIP books. These candidate consumers intentionally cannot reconstruct immutable history from legacy latest-only rows. A production rebuild requires source-version recovery for its intended scope. The code is on the isolated branch, with no merge or production cube write. Manager/CIK and security/CUSIP continuity, global day-45 significance, the user's price-cube refresh, and DAG full-versus-incremental proof remain in wiki/TODO.md. Current run evidence is reports/validate/2026-10-09-institutional-13f-audit.

## Insider canonical cleaning and feature contract

The insider branch of `StepCubeInstitutionals` calls one `clean_transactions` boundary before every aggregate. It keeps source values and publication/version provenance, normalizes identities/names/titles/flags/amounts, and receives one transaction-date as-traded quote frame restored through shared level/genuine-split helpers. Matching quote prices within±10% stay filed; gross corrections are estimated unless source evidence supports an exact amount. A missing execution price preserves known shares/after-holdings for account math while monetary dependencies remain unknown. [Cleaner](../../src/data_aggregate/utils/institutionals/insider_quality.py), [quote helper](../../src/data_aggregate/utils/institutionals/value_basis.py), [step](../../src/data_aggregate/transformers/step_cube_institutionals.py).

Comma/dotted title boundaries preserve explicit roles; named subsidiary/business-unit executives cannot automatically become issuer CEO/CFO/COO. Conservative compound/truncated titles may remain unknown. Missing officer titles use bounded strictly earlier public same-owner/issuer history and explicit DEF14 name/title evidence. CIK identity has priority over strict normalized names. Current contradictions/former roles block enrichment; a parent ceo_name_proxy that could be a highest-paid-person fallback is not sufficient leadership evidence. The ordinary scoped loader explicitly includes stored issuer_cik without a registry change. [Input seam](../../src/data_aggregate/utils/institutionals/inputs.py).

The twelve existing output names remain. Dollar flows sum their named60/180calendar disclosure-day window, then divide once by current PIT market cap. Counts stay distinct counts; net-buy ratio remains a dollar ratio. owner_surprise_120d contains an internal0–1 purchase-dollar rank against strictly earlier disclosure days, weighted by recent purchase dollars without temporal decay. Its contributors expire at their original disclosure anchor+120days; earlier history stays the comparison reference and the first disclosure day is undefined. purchase_pct_prior pools gross bought shares over60days and one implied starting holding per supported buying account, including unpriced known-share legs. The legacy decay argument remains caller-compatible but does not affect these cumulative insider fields. [Feature definitions and mathematics](../../src/data_aggregate/utils/institutionals/insider_features.py).

Exactly one added output, f_ic_insider_management_position_change_60d, pools cumulative net P−S shares over60calendar disclosure days, divided by one implied starting holding per supported active direct CEO/CFO/COO account. Holdings are the latest included economic-day terminal minus active net trades after visible balance-chain reconciliation; this is not a complete ledger exactlyD−60. Every lot keeps its own original disclosure expiry. Genuine splits normalize shares and holdings together; level/spinoff price actions cancel. Cross-account pooling requires one compatible class/unit basis: certified normalized units or the same economic date for scale-free unknown units. The class helper uses SEC security_title, not a CUSIP ledger. Zero/negative/full-exit observations remain valid; unsupported flows, inconsistent/ambiguous chains, incompatible units/classes, indirect/joint allocations and zero-prior openings abstain. Silent managers are absent. [Shared position math](../../src/data_aggregate/utils/institutionals/insider_positions.py).

Dated source assessments distinguish price-only contradictions from uncertain share quantities and preserve earlier information states. Later co-reporters and amendments are visible prospectively without refreshing original age. Broad Form4/A transaction-cell replacement remains a measured-materiality-unproved completeness limit. Filing day remains the user's information row for next-morning decisions; the existing target starts at that row's close, so it does not measure noon-entry P&L.

Integrated uncommitted cumulative content identity249e493a1918b47e71ff6ee8e3d72b0f5b059f99db80f79d9c93d179d6c86828 on base415528a38162966b7646243cc7dbf2a6bd1c8efd:100focused tests pass; an overlapping30-test ROOT integration check passes. Independent12-ticker seven-prefix/permutation replay compares5,988,060cells and both correctness/Ponytail reviews find no material calculation blocker. Final same-input full replay covers2,044,497source rows and2,374,616eligible cells; all13owned ROOT/WORK files independently match the manifest. Source/class-price, generic-validator and ambiguous-amendment limits remain conditional. No production cube/source-table write, configs/constants/store/DDL/fingerprint or model change. Local ignored evidence:reports/validate/2026-10-10-insider-feature-audit/report.html,07-cumulative-validation.md and06-final.md; prior daily-mean/decay evidence is preserved as historical. This scoped working-tree addendum does not certify a repository-wide wiki refresh.
