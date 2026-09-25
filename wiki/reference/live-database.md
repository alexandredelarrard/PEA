---
title: Live database
description: Dated measurements of the local PostgreSQL volume, coverage gaps, and operational caveats.
type: reference
snapshot: 2026-09-23
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
> Most global measurements were taken in August 2026, with targeted institutional and identity updates through September 2026. Re-query before relying on row counts, byte sizes, date maxima, or table presence. Never turn a historical measurement on this page into a hardcoded application assumption.

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
| `earning_calls_embedding` | about 1.38M rows / 8.6 GB | 2005-10 to 2026-07, 494 tickers | Always project; vectors are PostgreSQL arrays and cannot be bound by SQLite. |
| `sec13f_hr` | about 23.8M rows / 6.2 GB | period rows from 1987, usable broad coverage much later | Stream and scope every read; evaluate coverage on manager counts. |
| `prices` | about 1.78M rows | 2011-08 to 2026-08, 500 tickers | Price-dependent integration tests can run only when this table is present and current. |
| `earnings_call_sections` | about 110K rows / 1.5 GB | 2005-10 to 2026-07 | Text is the payload and is intentionally projected when scoring. |
| `sec_filing_text` | about 34K rows / 1.2 GB | 2011-07 to 2026-08 | Do not perform unbounded text reads. |
| `insider_transactions` | about 2.01M canonical rows | filing coverage from 2006 through the latest completed bulk quarter | Canonical reads must apply scope/repair and bulk/live completeness. |
| `insider_footnotes` | about 1.89M rows | accession-linked, no ticker/date grain | Join at filing grain; quarantined accessions can leave explainable orphans. |
| `wiki_pageviews` | about 1.70M rows | 2016-07 onward | Long windows have materially shorter history than prices. |

The 2026-09-23 targeted rebuild of `cube_part_institutionals` recorded roughly 3.27M rows, 120 feature columns, 491 tickers, and 1995-09 to 2026-09 coverage. Insider-dependent cross-source cells stopped at the measured bulk/live completeness frontier rather than being forward-filled.

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
- On SEC-derived tables, `cik` means the filer that actually submitted the document, not the current roster CIK. Group company history by ticker and apply registrant policy.

## Known table-presence and coverage traps

| Condition | Correct interpretation |
| --- | --- |
| Registered table is absent | A visible `TableMissingError`, not an empty source. Some destinations are created only on first write. |
| Table exists but is empty | Also an error by default; pass `optional=True` only for a genuinely optional source. |
| `sec_8k_votes` registered but not populated in an older snapshot | The parser had not run; do not interpret missing rows as no shareholder meetings. |
| DEF 14A parent covers many tickers but children cover only a smoke roster | Joining a child silently narrows the universe unless coverage is checked first. |
| `dividends` covers fewer tickers | Correct for non-payers; no row is not automatically missing data. |
| `earnings_surprises` has future dates | Scheduled calls are present; realized signals require non-null actual EPS. |
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
