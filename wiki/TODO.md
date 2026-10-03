---
title: TODO
description: Accepted backlog for source backfills and deferred data-contract work.
type: backlog
tags:
  - wiki
  - todo
  - data-quality
---
# TODO

Deferred work that requires new source history, authoritative eligibility evidence, or a broader data-contract change belongs here—not in production features with insufficient support.

## L/S point-in-time gates

### P0 — Purge the train-to-holdout label boundary

The [trainer](../src/modelling/steps/step_long_short.py) filters labelled rows by feature date through `train.end_date`, while the [L/S scorer](../src/strategies/utils/ls_model.py) starts at that same metadata date. The configured boundary is 2022-01-01 in [training](../configs/modellling.yml) and [portfolio](../configs/portfolio.yml). Forward 30/60/90-day labels near the training end can therefore contain returns from the nominal holdout; the per-fold CV embargo does not purge this separate boundary.

Before interpreting any backtest, make the first holdout decision strictly later than the latest return used by every training label (or remove the overlapping training rows). Test the boundary for every horizon and persist the effective label-end and first tradable holdout dates in run diagnostics.

### P0 — Rebuild historical universe and sector membership

The [price-part builder](../src/data_aggregate/transformers/step_cube_prices.py) uses the current S&P 500 roster for its historical grid, and [L/S sector neutralization](../src/strategies/step_ls.py) uses the current ticker-to-sector map. This can exclude former members and apply future classifications to past trades. Use a dated, point-in-time membership and sector history, reconciling the existing `sharadar_sp500` source before relying on it. Validate additions, removals, delistings, symbol changes, and the earliest eligible date per security; compare strategy results with the present survivor-only universe. Do not expand to Russell 1000 until this boundary is sound.

## Institutionals follow-up validation

- **P1 — Archive revisions and FTD timing.** Historical [FTD ZIP availability](./concepts/source-availability.md) dates are conservative estimates, not verified first-publication timestamps; a cached latest ZIP supplies only a short-lived local modification date. Test extra lags and an FTD-family ablation against the L/S holdout. Check whether revised source content can rewrite a pre-revision feature value; freeze prior published content if required by that evidence. Reassess the negative FTD monotonic constraint in [LightGBM configuration](../configs/models/lgbm_modelling.yml) against fold stability and economic meaning.
- **P1 — Full prefix replay and model re-evaluation.** Replay each institutional source family through the same historical prefix cutoff, then incremental-update it and prove pre-cutoff rows do not change except for an explicitly dated correction. Run the [institutionals quality driver](../scripts/institutionals_feature_quality.py) and a new model fit after the feature taxonomy changes. Keep correlated candidates for the model comparison, as agreed; remove one only on measured out-of-sample evidence.
- **P2 — Explain residual missingness.** Establish exact filing-discovery and absence frontiers for 13D/13G events; audit the remaining long interior hole runs one ticker and period at a time, including the five broad-13F coverage gaps. Separate source absence, ticker/CIK lineage, denominator ineligibility, and legitimate zero. Correct confirmed extraction or lineage defects and re-run coverage; never convert unknowns to zero solely to close a hole.



## Schedule 13D/13G ownership numerics

Reliable structured ownership values begin with the SEC mandate on 2024-12-17. The level and per-filer delta versions of `percent_of_class`, their cross-sectional transforms, and related power/aggregate fields remain excluded from `cube_part_institutionals`.

Restore them only after:

- parsing pre-mandate HTML/SGML cover pages point-in-time on `filing_date`;
- producing one canonical value per filing without summing repeated joint-filer values;
- reconciling pre-mandate output against overlapping XML and reporting disagreement rates;
- proving stable multi-year coverage across tickers and form variants;
- retaining enough history for representative training, validation, and test folds; and
- completing a new data-validation report.

Mean imputation over unavailable decades is not acceptable. Raw fields remain in the [source tables](./reference/table-catalog.md) for audit and future backfill.

## Pre-mandate beneficial-ownership event coverage

Investigate the roughly sixfold rise in Schedule 13D event rows around the structured-data mandate. Event-only features remain because filer identity and filing date exist before the mandate, but confirmed historical filing-discovery gaps must be backfilled before treating event intensity as time-comparable.

Acceptance evidence must separate a real regulatory/reporting shift from fetcher form-name or discovery coverage.

## Security-specific fails-to-deliver eligibility

The current FTD availability boundary is table-wide. Research and implement security-specific eligibility before interpreting pre-listing, post-delisting, suspended, or symbol-transition intervals as observed zero.

Required work:

- obtain authoritative primary-listing, delisting, suspension, and symbol-transition dates;
- reconcile those intervals with SEC FTD observations;
- keep vendor-backfilled price rows outside verified trading eligibility unavailable;
- build a 15-ticker edge-case table, including explicit treatment of the SMCI source-suspension gap; and
- prove in validation that no FTD feature appears before verified security eligibility.

Do not put these source-specific intervals in `symbol_tenure_manual.json`; that register has a different identity purpose.

## Point-in-time shares-outstanding split basis

Repair `fundamentals_history.sharesOutstandingPit` around stock splits before restoring guarded `ic_inst_ownership_pct` cells. The institutional rebuild on 2026-09-24 nulled 160 otherwise non-null cells above two times shares outstanding; bounded holes remained visible for AMCR, DD, DUK, and NVDA.

The numerator is present, but the point-in-time denominator is on an incompatible split basis. Clipping or forward-filling would fabricate ownership.

Required work:

- reconcile `sharesOutstandingPit` with split events and vendor-basis `sharesOutstanding`;
- backfill a corrected point-in-time denominator without applying future split knowledge;
- compare raw 13F shares against both denominator bases around every affected split;
- rebuild and validate `ic_inst_ownership_pct`; and
- remove the guard only if the corrected series is economically bounded and has no unexplained interior holes.

## Earnings-call tone drift over time

Transcript language drifts over the years, and nothing in the earnings-call features removes it. Measured on a stratified sample of 2,940 calls (140 per year, 2006–2026, 473 tickers), comparing 2006–10 with 2021–25:

- positive words rise in every role by 1.0 to 2.1 per 1,000 words per decade, and negative words fall by 0.5 to 0.9;
- courtesy sentences in analyst questions go from 8.6 to 13.9 per 1,000 words, as a step (2.6–3.5 in 2011–13, then 12–15 from 2015), which looks like a change in transcription style rather than in behaviour (unverified);
- courtesy in answers goes from 1.6 to 2.4 and in prepared remarks from 1.25 to 1.64 per 1,000 words.

The lexicon counts use a 40-word subset of each Loughran-McDonald list, not FinBERT; courtesy is now removed before scoring, but the positive/negative word drift is in the substantive text. The `f_ec_*` features have no cross-sectional normalization by design, so a common upward drift reaches the raw levels, and the prior-only issuer-history scores (`*_vs_hist`, 5-year window) carry it too: every issuer's current call reads above its own history. Evidence: `reports/validate/2026-10-01-earnings-call-extraction-gaps/01-research.md` (text-noise follow-up) and `_out/clean_politeness_by_year.png`.

Reconsider when any of these holds:

- full FinBERT scoring is done and the date-mean of `f_ec_tone` or `f_ec_tone_vs_hist` shows a trend over years;
- the earnings-call validator raises a PSI drift warning (`drift_psi_warn`) on a tone feature; or
- a model review finds the tone features' importance or sign unstable across CV folds by era.

Candidate fixes to compare then: a per-date cross-sectional demeaning of tone only, or a time-detrended issuer history. Keep the 12-column contract unless the measurement justifies a change.

## Documentation retirement

After this migration is reviewed:

- confirm every root instruction and package manifest resolves to [the wiki](./OVERVIEW.md);
- confirm the OpenKnowledge audit has no links into the retired documentation tree;
- delete only legacy docs that are fully represented here;
- keep this TODO as the accepted backlog; and
- append the retirement event to [wiki log](./log.md).

## Related

- [Data sources](./reference/data-sources.md)
- [Live database](./reference/live-database.md)
- [Source availability](./concepts/source-availability.md)
- [Cube aggregation](./modules/data-aggregate.md)


# extraction data
- short interest : extract also the Lit exchange NYSE /NASDAQ, from 2009 for all (now is 2018)
- insiders trading : sec form 3/4/5 -> sec since 2003, zip since Q1 2006 -> take sec. Done for the latest ones. So should be quick + add the Tickers fixed in gov
- earnings surprises starts 1999-08, but empty till ~2003
- financial notes (text & nums) 2009 from sec XBLR (zip), but possible directly from fillings (edgar)
- fix volume to be adjusted to spinoffs in price
- fix EC extraction history and gaps. 2/3 are available today vs 90% potential
- fix employee count, lots of MVs and gaps. Play with LLM extract.
- fine tune def 14 data extraction

# other data checks
- check data is consistent over time, even for latest 2026 month ?
- check dag data extract and data aggregate run smoothly end to end
- Review the price split / spinoff over time (price validation)
- refine modelling to be as stable as possible
- review periods when IC drops for few weeks / months
- add other strats decorrelated : - Super investors replica ?

# # Signals :
- include move from peers when new results are available -> move all peers info
- create variables based on clients stock move, geography graph, etc.

# universe expand:
- After the point-in-time S&P 500 gate above, evaluate Russell 1000 expansion with the same membership and delisting controls.
