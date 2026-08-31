# Implement Plan

You are tasked with fixing a bug step-by-step, following the phases outlined in the agent fundamentals-triage. This is Phase 2 bug identification and fix process.

## Initial Setup:

When this command is invoked, respond with:
```
🚀 Starting Bugfix (Phase 2 of bugfix workflow)

Please provide the path to your bugfix report plan from Phase 1 and the cluster ID to fix.
```

Repository live under C:\Users\de larrard alexandre\OneDrive - The Boston Consulting Group, Inc\Documents\repos_github\PEA\

## Implementation Process:

### 1. **Load and Review Plan**
- Read Claude.md and Agents.md
- **always use rtk in all your commands**, it saves a lot of token and make you more efficient
- Read .claude/agents/fundamentals-triage.md
- Read the cluster_id given by the user 
- Read the latest report document from `reports/validate/<YYYY-MM-DD>/run_id_<ID>.md`
- Read the newest `reports/validate/<date>/*.json`. **If none exists, stop and ask the user to
run `fundamentals-validate`** — do not run the validator yourself.

That file is a CONTRACT. For each of the top 5 it carries `cluster_id`, `ticker`, `field`,
`score`, `findings`, `checks_agreeing`, `severity_mix`, `tier_mix`, `period_range`,
`routing_hint`, `family_breadth`, `edgar_url`, `why` and `run_id`. **If a field is missing, say
so and stop rather than improvising** — a missing field is a defect in the report.


### 2. **Pre-Implementation Checks**

```bash
# Ensure clean working state
rtk git status
rtk git diff

# Run existing tests to ensure baseline
rtk python -m pytest

# Check current branch
rtk git branch --show-current
```

If not on a feature branch, suggest creating one:
```bash
rtk git checkout -b bugfix/cluster_{cluster_if}
```

### 3. **Phase-by-Phase Bugfixing**

**Read `src/validate/README.md` first** — especially §4, "when it does not work".
Follow strictly all the phases listed in te agent .claude/agents/fundamentals-triage.md

### 4. ** simplify after bugfix finished**

Run the command `/simplify` based on the `git diff` to ensure clean code handover
ANGLE — EFFICIENCY: Flag wasted work the diff introduces. (sub agent to run it) 