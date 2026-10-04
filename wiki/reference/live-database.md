---
title: Live database
description: Dated measurements of the local PostgreSQL volume, coverage gaps, and operational caveats.
type: reference
snapshot: 2026-10-02
tags:
  - wiki
  - reference
  - database
  - operations
---
# Live database

## Purpose and freshness

This page records the latest migrated measurements of the local PostgreSQL volume. It describes observed state, not the desired schema. The [table catalog](./table-catalog.md) remains authoritative for meaning and grain.

> [!CAUTION]
> Most global measurements were taken in August 2026, with targeted institutional and identity updates through September 2026 and the earnings-call cutover on 2026-10-02. Re-query before relying on row counts, byte sizes, date maxima, or table presence. Never turn a historical measurement on this page into a hardcoded application assumption.

The measured environment was PostgreSQL 16 in container `pea_db`, database `pea`, local owner role `alexandre`, backed by volume `stock_pick_strat_pgdata`. At the August snapshot it occupied roughly 19 GB and held 41 physical tables in `public`.

## How to re-measure

Use the password-free Unix socket inside the container:

```bash
MSYS_NO_PATHCONV=1 docker exec pea_db psql -U alexandre -d pea -c "SELECT current_database();"
```

Prefer catalog queries that return table presence, row estimates, physical size, columns, distinct tickers, and min/max dates in one captured report. Keep the measurement date beside every result. Connection and container details are in [run the pipeline](../guides/run-the-pipeline.md).

## Largest observed datasets

| Table | Observed scale | Observed coverage | Operational implication |
| --- | --- | --- | --- |
| `sec13f_hr` | about 23.8M rows / 6.2 GB | period rows from 1987, usable broad coverage much later | Stream and scope every read; evaluate coverage on manager counts. |
| `prices` | about 1.78M rows | 2011-08 to 2026-08, 500 tickers | Price-dependent integration tests can run only when this table is present and current. |
| `earnings_call_sections` | 2,804,060 paragraph rows, 33,591 calls (2026-10-02 full load) | 2005-10 to the load date, 487 tickers | Text is the payload; project and scope by ticker. |
| `sec_filing_text` | about 34K rows / 1.2 GB | 2011-07 to 2026-08 | Do not perform unbounded text reads. |
| `insider_transactions` | about 2.01M canonical rows | filing coverage from 2006 through the latest completed bulk quarter | Canonical reads must apply scope/repair and bulk/live completeness. |
| `insider_footnotes` | about 1.89M rows | accession-linked, no ticker/date grain | Join at filing grain; quarantined accessions can leave explainable orphans. |

The 2026-09-23 targeted rebuild of `cube_part_institutionals` recorded roughly 3.27M rows, 120 feature columns, 491 tickers, and 1995-09 to 2026-09 coverage. Insider-dependent cross-source cells stopped at the measured bulk/live completeness frontier rather than being forward-filled.

## Earnings-call snapshot (2026-10-02)

The earnings-call tables were cut over to the defeatbeta paragraph grain on 2026-10-02. Evidence lives under `reports/validate/2026-10-01-earnings-call-extraction-gaps/` (`03-implementation.md`, `_out/p5-*`).

- `earnings_call_sections`: full load of 33,591 calls in 1,119 s (18.6 min, 1,091 s of it DB writes); an unchanged source revision is a no-op in about 1.3–3.9 s. Parity with the source index at the same revision is exact (0 calls only in source, 0 only in DB, 0 `as_of` mismatches). The validator measured split `ok` on 98.71 % of calls, prepared word share q05/q50/q95 = 0.165/0.363/0.581, and a clean grain.
- `earnings_call_sentiment` and `earning_calls_embedding` were recreated empty and are only partly filled: the 72-call cost probe, the 3-ticker end-to-end sample (ABNB, PLTR, CEG, 78 calls) and about 7,450 embedded calls. Full FinBERT scoring and the rest of the embeddings are pending; see [run the pipeline](../guides/run-the-pipeline.md#earnings-call-rebuild).
- `cube_part_text` currently holds only the 3 sample tickers (3,780 rows) with exactly the 12 approved `f_ec_*` columns. A full `build-text -F` after scoring restores the universe. The pre-cutover part is kept as `_cache/p5_cube_part_text.dump` in the evidence folder.
- The pre-cutover tables are renamed `earnings_call_sections_legacy` (110,994 rows, 1,618 MB), `earnings_call_sentiment_legacy` (55,863 rows, 11 MB) and `earning_calls_embedding_legacy` (1,379,014 rows, 8,658 MB), pending the user's confirmation to drop them. Nothing reads them.
- The `wiki_pageviews` table measured in August 2026 (about 1.70M rows) has no producer or reader left in the code.

## Fundamentals snapshot

The fundamentals layer was intentionally asymmetric during the recorded migration:

- `fundamentals_sharadar` and merged `fundamentals_history` had been rebuilt over roughly 489 tickers.
- `fundamentals_facts` and `fundamentals_history_sec` represented a narrower SEC replay roster.
- The merged table is the consumer surface; the SEC replay is the validation surface.
- SEC contribution to the merged table is smaller than Sharadar coverage by design.
- Publication-event grain and the enumerated SEC history contract reduced both row and column counts compared with a retired legacy table.
- Every SEC history null is expected to have an accompanying reason-code record.

The older warning that Sharadar data reflected a free-tier subset became stale after the subscription upgrade and re-extraction work. Treat individual counts here as dated evidence, not current entitlement.

## Institutional and identity snapshot

Important measured state:

- 13F manager coverage contained historical holes that ticker counts did not reveal. A later refill closed the identified empty filing months; validation now scores this axis.
- `sec13f_manager_holdings` covered the filing managers needed for denominator-quality analysis but not every roster code resolves to a filing CIK.
- CIK-first insider identity repair relabelled valid predecessor rows, admitted rows the symbol path missed, and quarantined mismatched entities instead of discarding evidence.
- `symbol_tenure` contains overlapping observed intervals; overlap is expected and must not be collapsed into a one-row lookup.
- `entity_lineage` is sparse by design: a missing CIK row means a singleton entity.
- Until the user runs the identity cutover ([run guide](../guides/run-the-pipeline.md)), the live `symbol_tenure` (primary key without `source`/`evidence_period`, no `dei` rows) and `entity_lineage` (one row per CIK, no `role`) keep the old shape. The accessor reads the old `entity_lineage` as membership rows (roster window plus event-only CIKs), so register chains reach consolidating listings only after the cutover, and `notes-download` and `identity-tables` need the recreated tables. Measured read-only on 2026-10-04 before the cutover: foreign filer rows only in `sec_8k` (1,749 accessions, 3,795 rows, 10 tickers) and `fundamentals_facts` (349 accessions, 31,280 rows, 8 tickers).
- On SEC-derived tables, `cik` means the filer that actually submitted the document, not the current roster CIK. Group company history by ticker and apply registrant policy.
- A targeted 2026-09-30 FTD audit read 1,079,328 stored rows and found 413 source ZIP periods, all with estimated vintage availability after metadata backfill. Exactly 36,299 rows (3.36%) have day-15 settlements carried by a `b` ZIP; no settlement date spans two stored ZIP periods. Feature publication must use persisted `period`, never infer the half from the settlement day. The [short-flow builder](../../src/data_aggregate/utils/institutionals/short_flow_features.py) enforces the persisted-period and single-ZIP-per-date contracts.

## Known table-presence and coverage traps

| Condition | Correct interpretation |
| --- | --- |
| Registered table is absent | A visible `TableMissingError`, not an empty source. Some destinations are created only on first write. |
| Table exists but is empty | Also an error by default; pass `optional=True` only for a genuinely optional source. |
| `sec_8k_votes` registered but not populated in an older snapshot | The parser had not run; do not interpret missing rows as no shareholder meetings. |
| DEF 14A parent covers many tickers but children cover only a smoke roster | Joining a child silently narrows the universe unless coverage is checked first. |
| `dividends` covers fewer tickers | Correct for non-payers; no row is not automatically missing data. |
| `earnings_surprises` has future dates | Scheduled calls are present; realized signals require non-null actual EPS. |
| Roster names with no earnings-call rows | BRK-B holds no calls; ED, EXPD and NVR are absent from the defeatbeta source. Their `f_ec_*` features are null, not stale. |
| Bulk insider/pension maxima lag today | Publication cadence, not automatically failed extraction. |
| Short-volume minimum predates the current provider window | The isolated stored date is not proof of continuously recoverable history. |
| `sec_def14a` code exists but table was removed in an older cutover | Check current table presence before designing a feature around Pay-versus-Performance history. |
| Price table missing on another machine | Real-data price fixtures skip; a green test suite then represents reduced coverage. |

## High-risk measured defects

### Insider transaction value

Raw `value_usd` is not safe to aggregate. A tiny number of convertible-note and malformed per-share rows dominated the raw sum even though medians appeared plausible. The institutional feature path therefore:

1. restricts to eligible common-stock, open-market, priced transaction codes;
2. separates scope rejection from price repair;
3. repairs suspect per-share prices against a local, per-ticker and per-share-class filed-price median; and
4. validates aggregate buy/sell magnitudes after repair.

The implementation lives in [insider_quality.py](../../src/data_aggregate/utils/institutionals/insider_quality.py). Any new consumer must reuse that logic or independently prove an equivalent contract.

### Schedule 13D/13G structured-data cliff

Pre-mandate rows reliably preserve event identity and filing dates, but numeric ownership fields are mostly unavailable. Post-mandate rows provide structured numerics, with joint-filer and comment edge cases. Event features can span the older history; numeric ownership features remain deferred in [TODO](../TODO.md).

### 13F source versus market coverage

Early stored periods can contain valid-looking holdings from only a small subset of managers. Use measured manager breadth, known hole/break masks, per-ticker prior-holder thresholds, and publication-date availability. Never infer completeness from ticker count or `min(period)`.

### Fundamentals fact grain

A single filing may report several comparative fiscal periods with the same fiscal labels. Facts must not be joined or deduplicated on labels alone; period end and accession-level publication state are part of the contract.

## Runtime state and infrastructure caveats

- The transferred volume's role differs from compose defaults; host connections require the actual password, while container-local socket access is trusted.
- PostgreSQL init scripts run only on an empty volume.
- An older running container may retain a stale bind mount even when [docker-compose.yml](../../docker-compose.yml) is correct. Inspect mounts before rebuilding an empty volume.
- Models, caches, and the database volume are non-recoverable risk zones. Back up affected tables or the volume before destructive maintenance.
- Never kill `python.exe` by image name; long-running source jobs must be stopped by PID.

## Updating this page

After a material extraction, rebuild, migration, or backfill:

1. re-measure presence, rows, size, columns, ticker count, and time coverage;
2. date every changed statement;
3. distinguish source limits from incomplete work;
4. update the corresponding [source](./data-sources.md) and [table](./table-catalog.md) contracts only if semantics changed; and
5. record deferred repairs in [TODO](../TODO.md).

## Related

- [Table catalog](./table-catalog.md)
- [Data sources](./data-sources.md)
- [Data access and storage](../guides/data-access.md)
- [Run the pipeline](../guides/run-the-pipeline.md)
- [Validation](../modules/validation.md)
