---
title: Cube aggregation
description: Incremental cube-part builders, feature families, target construction, and streamed final assembly.
type: module
tags:
  - wiki
  - module
source_commit: d4ebd98ffe561aee3f51f2c0b910fd2a627ae4f0
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
- **CEO stints.** `names.ceo_stint_ids` splits a ticker's proxies on a change of CEO, keyed on the first-listed name with honorifics (Dr., Mr., Prof., ...) stripped; a generational-suffix change (Jr., Sr., II, III) splits only together with a change of first name (WRB father and son), a bare surname is an unknown CEO, and one off-cycle proxy naming a CEO seen nowhere else in the ticker (another company's proxy) belongs to no stint. A co-CEO who becomes the first-listed CEO opens his own stint. `person_key` is unchanged. `ceo_since_year` takes the value at or before the stint's first proxy year that the most filings of the stint disclose (ties go to the largest), else stays as filed. `ceo_is_founder` takes the stint's mode. Both are kind `person`, dated at the row's own `as_of`, and keep the filed value in `<col>_raw`. `ceo_is_board_chair` is carried only within a stint.
- **Ownership estimate.** A NULL `insider_ownership_pct` on a ticker never flagged dual-class is the sum of `director_officer` shares in `def14a_ownership` over point-in-time `sharesOutstandingPit`. Holder rows whose name reads as an entity (Inc, LLC, L.P., Trust, Fund, Vanguard, BlackRock, State Street, a board-held block such as MetLife's Policyholder Trust, ...) are not summed: the extraction tags some 5% holders `director_officer`. Words that also appear in executive titles (Group, Holdings, Capital, Management, Investments, Securities, so also the "as a group" total) are tested on the name before its first comma only. It needs at least 5 remaining holders, a result in (0, 1] and at most 3x the ticker's largest filed value, and is kind `estimate`. Filed, computed and carried values are never overwritten.
- **Say-on-pay.** `f_say_on_pay_support` is gone; the 8-K vote series is the single say-on-pay source. A proxy with a filed support value and no 8-K tally in the prior 450 days adds one vote row (for = support, against = 1 − support, `source = proxy`), dated at the proxy.
- **SCT pay growth.** `ceo_comp_growth_sct` is log(total FY / total FY−1) from the CEO's row in the filing's own Summary Compensation Table, filer-stated totals only. `ceo_comp_growth_1y` and `ceo_pay_growth` prefer it and fall back to the cross-filing pair; the CEO-change mask still applies. A NULL parent `ceo_total_comp` takes the CEO's same-year SCT total (kind `identity`).

> ⚠ **Historical pay backfill is a deliberate lookahead.** A NULL CEO pay component or total is filled from a LATER proxy's SCT for the same CEO and fiscal year (`backfill`), or log-linearly inside one stint across at most 2 missing proxies (`interp`). In live data a value missing today is not filled this way, so history is more complete than the present. The step logs the counts as one WARNING. The lookahead is concentrated in early history: the share of proxies with at least one pay cell from a later filing is 35-54% in 2000-2005, 15-20% in 2006-2011 and 4-7% from 2012 (0 on any ticker's latest proxy), so a backtest on 2000-2011 sees denser pay history than live data and a train/test split straddling 2006 mixes the regimes. Two smaller lookaheads are also accepted: stint `person` values come from the stint's other filings, and the estimate's "ever dual-class" test and its filed-value ceiling read later proxies (they can only refuse a fill). An exact later SCT value also replaces a carried salary, and a bridged (`interp`) total feeds no cross-filing growth and is flagged `comp_imputed`.

> ⚠ **Run `build-governance -F` after each proxy season.** Because the stage reads later proxies (stint constants, the pay backfill), a new proxy can restate cleaned values on older rows. An incremental run rewrites only the last 7 trading days, so older rows keep their last-built values; the step logs a WARNING on every incremental run.

Every rule's count is logged as `governance clean | <stage>|<rule>: n`; `test_clean_stage_tally_keys` pins the stage names. Evidence: `reports/validate/2026-10-10-governance-fill-review/`.

## Dependencies

The module reads through [DataStore](./data-store.md), loads peer dictionaries from [peer deduction](./data-peers.md), and consumes the full set of normalized extract tables from [data extraction](./data-extract.md).

## Participates in

The module is the core of the [cube-build flow](../flows/cube-build.md). Its output feeds [modelling](./modelling.md), while its registry and incremental rules define the [cube-parts concept](../concepts/cube-parts.md).

## Related

- [Add a cube feature](../guides/add-a-cube-feature.md)
- [Point-in-time and quality controls](../architecture/point-in-time-and-quality.md)
- [Data access and storage](../guides/data-access.md)

## SEC 13F values and immutable books

The approved institutional code is merged into `dev` at `27e4f723f2e61a4dffd54d11f4785cee89e2b9ee`; its reviewed worktree remains at `reports/worktrees/institutional-13f-pit`. Both broad and elite panels use the shared [value repair](../../src/data_aggregate/utils/institutionals/value_basis.py): one supported monetary factor per manager/report-quarter/accession, checked against report-quarter-end as-traded prices. At least fifteen distinct priced securities support a unit-changing factor; unchanged thin books stay explicitly source-only. After scaling, common values missing, equal to shares, or outside the configured ten-percent price tolerance use reported shares times the eligible quote. Missing common shares with positive values use normalized dollars divided by that quote. Estimates do not vote for a unit factor. Call/put legs use their own underlying quantities; debt principal never uses an equity quote. Quotes are nonfuture and at most seven calendar days old. Logs distinguish scaling, estimates, exclusions and unresolved prices.

Only [`sec13f_hr` and `sec13f_manager_holdings`](../reference/table-catalog.md#existing-table-13f-versions--approved-storage-correction) store source versions. Positions and one hidden `cusip=_empty` completion record commit atomically per accession. Originals and amendments coexist; legacy rows remain unverified `_legacy` accessions. Source-complete empty books are observable. No auxiliary filing table or JSON data store is required.

The [ordinary consumers](../../src/data_aggregate/transformers/step_cube_institutionals.py) resolve each manager/report quarter using versions public by the observation day, retain the greatest usable quarter for levels, and compare against the latest public version of the adjacent prior quarter for the same eligible manager cohort. An old amendment can update that comparison prospectively without replacing newer levels or changing earlier daily observations. Same-day versions resolve to a single daily state; source clocks and amendment metadata order versions before emission.

A valid reconciled book with absent, blank or false confidentiality metadata is used normally. Only an affirmative source flag or omission statement activates the confidentiality absence safeguards. Large quarter changes do not exclude a book. Unknown/failed/partial books cannot establish exits; a verified complete empty book can. Unknown selected dollar weights propagate into elite comparisons, and zero selected comparison pools stay undefined.

Independent closure validation is conditional for bounded code/data correctness; production adoption is separate. The 11 October live inspection found neither existing destination had verified accession columns or completion packets. Historical use therefore requires non-destructive migration and replay of original source accessions before rebuilding the institutional part and assembled cube. Manager reporting perimeters, dated CUSIP continuity, broad timing significance and delayed-source refresh remain in [TODO](../TODO.md#13f-deferred-issues-from-the-october-2026-audit). Current evidence lives in `reports/validate/2026-10-09-institutional-13f-audit/`; merge status is recorded there, not implied by this contract.

## Insider canonical cleaning and feature contract

The insider branch of `StepCubeInstitutionals` calls one `clean_transactions` boundary before every aggregate. It keeps source values and publication/version provenance, normalizes identities/names/titles/flags/amounts, and receives one transaction-date as-traded quote frame restored through shared level/genuine-split helpers. Matching quote prices within±10% stay filed; gross corrections are estimated unless source evidence supports an exact amount. A missing execution price preserves known shares/after-holdings for account math while monetary dependencies remain unknown. [Cleaner](../../src/data_aggregate/utils/institutionals/insider_quality.py), [quote helper](../../src/data_aggregate/utils/institutionals/value_basis.py), [step](../../src/data_aggregate/transformers/step_cube_institutionals.py).

Comma/dotted and semicolon title boundaries preserve explicit roles. Ordinary corporate companions, including CFO, Sr. VP & Treasurer, VP, managing director and interim, preserve supported issuer CEO/CFO/COO roles. Explicit former/subsidiary/business-unit roles remain excluded; ambiguous issuer-name suffixes and rare unsupported companions remain unknown. Missing officer titles use bounded strictly earlier public same-owner/issuer history and explicit DEF14 name/title evidence. CIK identity has priority over strict normalized names. Current contradictions/former roles block enrichment; a parent ceo_name_proxy that could be a highest-paid-person fallback is not sufficient leadership evidence. The ordinary scoped loader explicitly includes stored issuer_cik without a registry change. [Input seam](../../src/data_aggregate/utils/institutionals/inputs.py).

The twelve existing output names remain. Dollar flows sum their named60/180calendar disclosure-day window, then divide once by current PIT market cap. Counts stay distinct counts; net-buy ratio remains a dollar ratio. owner_surprise_120d contains an internal0–1 purchase-dollar rank against strictly earlier disclosure days, weighted by recent purchase dollars without temporal decay. Its contributors expire at their original disclosure anchor+120days; earlier history stays the comparison reference and the first disclosure day is undefined. purchase_pct_prior pools gross bought shares over60days and one implied starting holding per supported buying account, including unpriced known-share legs. The legacy decay argument remains caller-compatible but does not affect these cumulative insider fields. [Feature definitions and mathematics](../../src/data_aggregate/utils/institutionals/insider_features.py).

Exactly one added output, f_ic_insider_management_position_change_60d, pools cumulative net P−S shares over60calendar disclosure days, divided by one implied starting holding per supported active direct CEO/CFO/COO account. Holdings are the latest included economic-day terminal minus active net trades after visible balance-chain reconciliation; this is not a complete ledger exactlyD−60. Every lot keeps its own original disclosure expiry. Genuine splits normalize shares and holdings together; level/spinoff price actions cancel. Cross-account pooling requires one compatible class/unit basis: certified normalized units or the same economic date for scale-free unknown units. The class helper uses SEC security_title, not a CUSIP ledger. Zero/negative/full-exit observations remain valid; unsupported flows, inconsistent/ambiguous chains, incompatible units/classes, indirect/joint allocations and zero-prior openings abstain. Silent managers are absent. [Shared position math](../../src/data_aggregate/utils/institutionals/insider_positions.py).

All thirteen fields share a confirmed-inactivity rule: a full named source window, then-public issuer disclosure, supported listing, supplied completeness frontier and required denominator/flag support allow quiet zero without a previous purchase or position. A ticker with no corroboration stays unknown. First-owner purchase ranks and zero-opening-holding ratios remain undefined; active unsupported quantities/classes/roles/holdings/units remain NaN. This zero convention for owner surprise means no recent purchase signal, rather than an observed rank. Conditioning consumes the final panel's values/masks. [Shared observability and emission](../../src/data_aggregate/utils/institutionals/insider_features.py).

Dated source assessments distinguish price-only contradictions from uncertain share quantities and preserve earlier information states. Unsupported records enter copy/amendment versioning before arithmetic filtering; later public corrections can retire uncertainty. Same-owner uncertain sales mask dependent active accounts; their original 60-day expiry cannot block later independent supported windows. Later co-reporters and amendments are visible prospectively without refreshing original age. Broad Form4/A transaction-cell replacement remains a measured-materiality-unproved completeness limit. Filing day remains the user's information row for next-morning decisions; the existing target starts at that row's close, so it does not measure noon-entry P&L.

Merged into `dev` in commit `d4ebd98ffe561aee3f51f2c0b910fd2a627ae4f0` from insider commit `0284541fc009a0270f034682c7c6773a8cf72f94`. The original 124 focused tests, independent 66 exact/48 raw-gap controls, 12-ticker seven-prefix replay on 5,988,060 cells and full 2,044,497-row/2,374,616-cell comparison remain frozen to their measured source checkpoints. Post-merge validation passes 34 affected tests, including proxy barriers, quiet zeros, amendment expiry, insider incremental equality and the newer three-year governance window; zero live writes and unchanged tested hashes. The sole merge conflict was the explicit loop binding required by Ruff in `load_insider_role_evidence`; the source check confirms every other application byte matches prior `dev` after newline normalization. Commit hooks, including Ruff and formatting, pass. Original checkpoint identities `dea433212ab916db12dbef5ba90b5f536c8a89ed6591e67062b9a4e7046c34a6` and `6754e31aabfa47642aba1f0bfc614d4e2cfcef7939989b4eef308d89f790b5a9` are historical evidence rather than final working-tree byte identities. Local ignored evidence: `reports/validate/2026-10-10-insider-feature-audit/report.html`, `06-final.md` and `_out/merge-insider-validation.json`. [Remaining adoption and source tasks](../TODO.md#insider-transactions-follow-ups) include the full stored-history rebuild: production source/cube tables were not written. Quiet-zero completeness still depends on the supplied max-date/resume-overlap frontier; absent accessions, class-price gaps and ambiguous amendments remain limitations. This refresh covers the insider contract, not a repository-wide data certification.
