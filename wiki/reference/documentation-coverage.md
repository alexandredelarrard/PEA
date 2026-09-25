---
title: Documentation coverage
description: Canonical subject map and deletion-readiness evidence for the retired documentation tree.
type: reference
tags:
  - wiki
  - reference
  - documentation
  - migration
---

# Documentation coverage

## Purpose

This page records where the former standalone documentation subjects live in the canonical wiki. It is a deletion-readiness map, not a second source of truth: follow the linked page for the maintained contract.

## Coverage map

| Former subject | Canonical wiki coverage | What was verified |
| --- | --- | --- |
| Agent context index | [Overview](../OVERVIEW.md), [coding standards](../guides/coding-standards.md), and [runbook](../guides/run-the-pipeline.md) | Task routing, non-negotiable repository rules, execution prerequisites, and links to every maintained area. |
| Architecture | [System overview](../architecture/system-overview.md), [orchestration and runtime](../architecture/orchestration-and-runtime.md), [data platform](../architecture/data-platform.md), and the [module catalog](../OVERVIEW.md#modules) | Directory ownership, Step construction, pipeline ordering, entry points, and current component boundaries. |
| Data schema | [Table catalog](./table-catalog.md), [live database](./live-database.md), and [table registry concept](../concepts/table-registry-and-store-boundary.md) | Registry ownership, primary keys, grains, date semantics, extract tables, cube parts, aggregate outputs, freshness, and schema drift caveats. |
| Live database state | [Live database](./live-database.md) | Dated row/coverage measurements, populated-versus-declared distinction, known defects, missing-table behavior, and bind-mount caveats. |
| Data sources | [Data sources](./data-sources.md), [SEC and LLM flow](../flows/sec-llm-extraction.md), [source availability](../concepts/source-availability.md), and [TODO](../TODO.md) | Provider map, shared transport, source-specific traps, institutional coverage boundaries, incremental behavior, cost discipline, and deferred backfills. |
| Data and database conventions | [Data access and storage](../guides/data-access.md), [point-in-time controls](../architecture/point-in-time-and-quality.md), and [cube parts](../concepts/cube-parts.md) | Single SQL boundary, bounded reads, missing-table semantics, writes, PIT integrity, incremental parts, SEC/XBRL rules, and artifact separation. |
| Configuration | [Configuration](./configuration.md) and [constants/configuration module](../modules/constants-and-configuration.md) | OmegaConf assembly, key ownership, availability/freshness, SEC identity evidence, validation semantics, cube/model/portfolio settings, and knob-placement rules. |
| Modelling, strategies, and portfolio | [Modelling, strategies, and portfolio](./modelling-and-portfolio.md), [modelling module](../modules/modelling.md), [strategies module](../modules/strategies.md), and [portfolio module](../modules/portfolio.md) | Layer contracts, training lifecycle, completion criteria, artifacts, bounded cube reads, sleeve construction, blend, ledger, and review checklist. |
| Coding standards | [Coding standards](../guides/coding-standards.md) | Naming, types, imports, docstrings, constants, logging, exception behavior, SQL boundary, risk zones, minimal-change discipline, reporting, and documentation synchronization. |
| Testing and validation | [Testing and validation](../guides/testing.md), [validate a change](../guides/validate-a-change.md), and [validation module](../modules/validation.md) | Real-data versus synthetic-test policy, fixtures, skips, mandatory sanity conclusions, guard tests, fingerprint baseline, validator use, and CI limitations. |
| Operations and backfills | [Run the pipeline](../guides/run-the-pipeline.md) and [large backfills and recovery](../guides/large-backfills-and-recovery.md) | Interpreter, Docker/PostgreSQL access, environment, complete CLI command families, cold-start order, Airflow, fundamentals resolution sweeps, registrant recovery, ownership/insider backfills, parity promotion, coverage evidence, validators, rollback rules, and operational gotchas. |
| Deferred data work | [TODO](../TODO.md) | All four accepted backlog items and their acceptance evidence. |

## Backlog location

The accepted backlog is [wiki TODO](../TODO.md). Its four data workstreams are:

1. Schedule 13D/13G ownership numerics before the structured-data mandate.
2. Pre-mandate beneficial-ownership event coverage.
3. Security-specific fails-to-deliver eligibility.
4. Point-in-time shares-outstanding split-basis repair.

The retirement checklist lives in the same page so it is not lost when the superseded documentation tree is removed.

## Deletion readiness

The old tree is removable when all of the following remain true:

- repository instructions and package metadata point to [the overview](../OVERVIEW.md);
- application code and tests contain no local path dependency on the retired tree;
- OpenKnowledge reports no broken links or orphaned wiki pages; and
- future documentation changes update the narrowest canonical wiki page and append [the wiki log](../log.md).
