---
title: Wiki Log
description: Append-only audit trail of wiki generation and refresh runs.
---

# Wiki Log

Append-only audit trail. Add one dated entry per generation or refresh run, recording the profile, the `source_commit` it was anchored to, and the coverage. The codebase-wiki skill describes the entry shape.

## 2026-09-25: generate

- Profile: internal/standard
- source_commit: 054f0e1
- Coverage: overview; five architecture areas; fourteen application, infrastructure, and test modules; five end-to-end flows; seven core concepts; five task guides
- Pages: [Overview](./OVERVIEW.md)
- Architecture: [System overview](./architecture/system-overview.md), [Data platform](./architecture/data-platform.md), [Orchestration and runtime](./architecture/orchestration-and-runtime.md), [Feature-to-portfolio pipeline](./architecture/feature-to-portfolio.md), [Point-in-time and quality controls](./architecture/point-in-time-and-quality.md)
- Modules: [Runtime and shared utilities](./modules/runtime-and-shared-utils.md), [Constants and configuration](./modules/constants-and-configuration.md), [Data store](./modules/data-store.md), [Data extraction](./modules/data-extract.md), [GPT extraction](./modules/gpt-extract.md), [Peer deduction](./modules/data-peers.md), [Cube aggregation](./modules/data-aggregate.md), [Modelling](./modules/modelling.md), [Strategies](./modules/strategies.md), [Portfolio](./modules/portfolio.md), [Validation](./modules/validation.md), [DAGs and infrastructure](./modules/dags-and-infrastructure.md), [Application and scripts](./modules/application-and-scripts.md), [Tests](./modules/tests.md)
- Flows: [Nightly data refresh](./flows/nightly-data-refresh.md), [SEC and LLM extraction](./flows/sec-llm-extraction.md), [Cube build](./flows/cube-build.md), [Model training and daily prediction](./flows/model-training-and-prediction.md), [Portfolio to trade ledger](./flows/portfolio-to-trade-ledger.md)
- Concepts: [Step pattern](./concepts/step-pattern.md), [Table registry and store boundary](./concepts/table-registry-and-store-boundary.md), [Point-in-time data](./concepts/point-in-time-data.md), [Cube parts](./concepts/cube-parts.md), [Peer-relative features](./concepts/peer-relative-features.md), [Source availability](./concepts/source-availability.md), [Strategy sleeves](./concepts/strategy-sleeves.md)
- Guides: [Run the pipeline](./guides/run-the-pipeline.md), [Add a data source](./guides/add-a-data-source.md), [Add a cube feature](./guides/add-a-cube-feature.md), [Add a model or sleeve](./guides/add-a-model-or-sleeve.md), [Validate a change](./guides/validate-a-change.md)

## 2026-09-25: refresh

- Profile: internal/standard
- source_commit: 054f0e1 (unchanged)
- Coverage: fully read and consolidated all 12 legacy `docs/` pages; added dense reference, operations, coding, testing, data-access, and backlog pages; rewired every wiki documentation link
- Primary pages: [Overview](./OVERVIEW.md), [TODO](./TODO.md), [Table catalog](./reference/table-catalog.md), [Data sources](./reference/data-sources.md), [Live database](./reference/live-database.md), [Configuration](./reference/configuration.md), [Modelling and portfolio](./reference/modelling-and-portfolio.md)
- Guides: [Run the pipeline](./guides/run-the-pipeline.md), [Data access](./guides/data-access.md), [Coding standards](./guides/coding-standards.md), [Testing and validation](./guides/testing.md), [Add a data source](./guides/add-a-data-source.md), [Add a model or sleeve](./guides/add-a-model-or-sleeve.md), [Validate a change](./guides/validate-a-change.md)
- Rewired sections: [architecture](./architecture/system-overview.md), [modules](./modules/data-store.md), [flows](./flows/nightly-data-refresh.md), and [concepts](./concepts/source-availability.md)
- Agent entry points: [AGENTS.md](../AGENTS.md), [CLAUDE.md](../CLAUDE.md)
- Retirement state: legacy `docs/` remains present for human review; package metadata and documentation-sync tooling now select the wiki

## 2026-09-25: refresh — documentation retirement audit

- Profile: internal/standard
- source_commit: 054f0e1 (unchanged)
- Coverage: removed active legacy-path references; verified all twelve former documentation subjects against canonical wiki destinations; made the backlog directly discoverable; corrected validation guidance against the inspected CLI; added source-grounded recovery procedures
- Pages: [Overview](./OVERVIEW.md), [TODO](./TODO.md), [Documentation coverage](./reference/documentation-coverage.md), [Run the pipeline](./guides/run-the-pipeline.md), [Large backfills and recovery](./guides/large-backfills-and-recovery.md)
- Agent entry points: [AGENTS.md](../AGENTS.md), [CLAUDE.md](../CLAUDE.md)
- Audit boundary: generated caches, tool-managed local state, secrets, report artifacts, and external provider URLs are not repository-documentation dependencies

## 2026-09-26: refresh — institutionals refactor

- Profile: internal/standard
- source_commit: 0d4ebd0
- Coverage: refreshed the institutional cube-part architecture, ordered build flow, completeness-frontier source references, and the insider outlier diagnostic after extracting input loading and frontier resolution from the step orchestrator
- Architecture: the top-level staged pipeline and persistence boundaries are unchanged; the institutional sub-step now delegates I/O to [inputs.py](../src/data_aggregate/utils/institutionals/inputs.py) and completeness decisions to [frontiers.py](../src/data_aggregate/utils/institutionals/frontiers.py)
- Pages: [Cube aggregation](./modules/data-aggregate.md), [Cube build](./flows/cube-build.md), [Source availability](./concepts/source-availability.md), [Application and scripts](./modules/application-and-scripts.md)
- Public contracts retained: `StepCubeInstitutionals.run()`, `build_panel()`, the ordered seven-panel merge, the shared conditioning sink, exact price/share projections, completeness semantics, and `cube_part_institutionals` persistence
