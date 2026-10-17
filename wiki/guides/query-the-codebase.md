---
title: Query the codebase with Graphify
description: Local Graphify setup, safe refresh commands, and graph-first source navigation for PEA.
type: guide
tags:
  - wiki
  - guide
  - graphify
---
## Goal

Use the local Graphify index to locate source symbols and their callers before opening source files. Read the relevant [wiki task page](../OVERVIEW.md#task-map) first, then query the graph and verify the returned source locations. The [agent rules](../../AGENTS.md) require this order.

## Steps

1. Run from the PEA repository root. Graphify is installed as the isolated uv tool `graphifyy[sql]==0.9.92`; its command is `graphify`.
2. Build a missing graph or refresh it when the source has changed. This command runs local AST extraction and community detection without an LLM:

   ```powershell
   rtk proxy graphify extract . --code-only --no-dedup --max-workers 4
   ```

3. Ask a focused question with a small output budget. Use exact symbol names when the question is too broad:

   ```powershell
   rtk proxy graphify query "StepCubeFundamentals" --budget 1500
   rtk proxy graphify explain "StepCubeFundamentals"
   rtk proxy graphify explain "schema.sql"
   rtk proxy graphify path "StepExtractAllData" "run_edgar_fetch" --undirected
   ```

4. Read the returned source locations to confirm signatures, control flow, and edge meaning before editing or making claims. If a query has no match, retry with a symbol or file name, then use a bounded source search.
5. Refresh the graph after source changes. Generate the report and interactive view without LLM labeling after a refresh:

   ```powershell
   rtk proxy graphify cluster-only . --no-label
   ```

The graph and HTML live under `graphify-out/`, which is ignored by [Git](../../.gitignore). The extraction command supports incremental reruns. A different checkout, new commit, or uncommitted source edits can make the graph stale; rerun extraction before relying on it. Keep the shrink guard: investigate extraction failures rather than forcing a smaller graph.

## Relevant code

- [.graphifyignore](../../.graphifyignore) defines the local scan boundary: Markdown, agent state, worktrees, reports, data, environment files, and Graphify outputs are excluded.
- [System overview](../architecture/system-overview.md) describes the pipeline stages.
- The source directories `src/`, `tests/`, `app/`, and `scripts/` supply the indexed code relationships.

## Gotchas

- In-scope Markdown reads and writes still go through OpenKnowledge MCP. Graphify's generic skill examples do not override the repository rules. The generated `graphify-out/GRAPH_REPORT.md` is Git-ignored by [.gitignore](../../.gitignore), so it is outside the OpenKnowledge content scope and may be written by Graphify itself. Keep generated reports and exports in that ignored directory.
- Use the explicit code-only extraction command above for refreshes. Keep `.graphifyignore` in place so generic extraction examples cannot read repository Markdown directly.
- `--no-dedup` preserves distinct code symbols instead of fuzzy-merging similarly named symbols. No paid semantic backend is needed for this code graph.
- Graph relationships can be inferred and dynamic Python calls can be missed. The default graph is undirected: use `path --undirected` for connectivity, and verify call direction in source. A graph edge is navigation evidence; source inspection establishes behavior. A missing edge does not establish the absence of a caller.
- On Windows, if an already-open terminal cannot find `graphify`, start a new terminal or call `rtk proxy "$env:USERPROFILE/.local/bin/graphify.exe" query "StepCubeFundamentals" --budget 1500`.
- Official Graphify skills are installed in the Windows user's Codex, Claude, and generic agent skill directories. Newly installed skills become available in subsequent agent turns/sessions.

## Local verification

The 16 October 2026 local verification (`graphify-out/verification.json`, an ignored runtime artifact) records Graphify 0.9.92 with its SQL parser, 15,389 nodes, 47,050 edges, 400 communities, and 81 SQL nodes at source commit `d1e41b6`. `query`, `explain`, and undirected `path` smoke checks passed. The read-only integrity check found no missing endpoints or duplicate edges; it retained 21 self-loops, including a source-verified recursive rebuild call. Codex, Claude, and generic agent skill files and the JSON, HTML, and Markdown report artifacts were verified. Markdown indexing was excluded. No application tests or database operations were needed for this tooling setup. These counts describe this local snapshot; refreshes can change them.

## Related

- [Run the pipeline](./run-the-pipeline.md)
- [Testing and validation](./testing.md)
- [Data access](./data-access.md)
