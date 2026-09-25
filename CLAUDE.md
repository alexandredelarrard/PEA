This project uses [AGENTS.md](AGENTS.md) as the always-loaded instruction contract.
Read it first, then use [wiki/OVERVIEW.md](wiki/OVERVIEW.md) to select the canonical architecture,
guide, module, reference, or TODO page for the task. The wiki is the only documentation
navigation surface.

All in-scope Markdown reads and writes must go through OpenKnowledge MCP. Read source code with
native workspace tools.

Always prefix shell commands with `rtk`.

<!-- rtk-instructions v2 -->
# RTK (Rust Token Killer) — token-optimized commands

## Golden rule

Prefix every shell command with `rtk`. If RTK has a dedicated filter it uses it; otherwise it
passes the command through unchanged. Apply the prefix to every independent command segment.

## Files and search

```bash
rtk ls <path>
rtk read <source-file>
rtk rg <pattern> <path>
rtk rg --files <path>
```

Use OpenKnowledge `exec`, `search`, `write`, and `edit` instead of these native commands for
Markdown files.
