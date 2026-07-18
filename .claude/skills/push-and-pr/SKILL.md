---
name: push-and-pr
description: >-
  Commit the current working changes on a feature branch and open (or update) a
  GitHub pull request against the default branch, following this repo's commit/PR
  conventions. Use when the user asks to push changes, submit/open/raise a PR, or
  ship the current work.
---

# Push changes & submit a PR

Turn the current uncommitted work into a clean commit on a feature branch and open
(or update) a GitHub PR. Run every `git`/`gh` command from the repo root.

## 0. Preconditions
```bash
cd "$(git rev-parse --show-toplevel)"        # PEA root (changes may span docs + stock_pick_strat/)
gh auth status                               # must be authenticated; if not, stop and tell the user
git status --short && git branch --show-current
```
If `git status` is clean, there is nothing to ship — report that and stop.

## 1. Understand what changed (always, before committing)
```bash
git diff --stat                              # scope of change
git diff                                     # the actual changes — read them
git log --oneline -5
```
Summarize the change in one or two sentences. **Scan the diff for anything that must
not be committed** — secrets/tokens, `.env`, large data artifacts, anything under the
Postgres volume or `data/`. If `.env` or a secret appears staged, stop and warn.
Respect `.gitignore`; never `git add -f` an ignored path.

## 2. Choose the branch
- Base branch = the repo default (`main`).
- If the current branch is `main`/`master`: **create a feature branch first** — never
  commit directly to the default branch.
  ```bash
  git switch -c claude/<short-kebab-slug>
  ```
- If already on a feature branch or `dev`: stay on it.

## 3. Stage
Stage intentionally. Prefer `git add <paths>` for the files you inspected; use
`git add -A` only once you've confirmed the diff is all wanted.

## 4. Commit
Concise imperative subject (<~70 chars) + a short body of what/why. End with the
co-author trailer. Use a heredoc (Bash tool is POSIX sh — no PowerShell here-strings):
```bash
git commit -F - <<'EOF'
<imperative subject line>

- <what changed and why, bullet form>
- <second point if needed>

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
```

## 5. Push
```bash
git push -u origin "$(git branch --show-current)"
```

## 6. Open (or update) the PR
If a PR for this branch already exists, the push above already updated it — just
surface it. Otherwise create it:
```bash
gh pr view --json url -q .url 2>/dev/null || \
gh pr create --base main --head "$(git branch --show-current)" \
  --title "<imperative PR title>" \
  --body "$(cat <<'EOF'
## Summary
- <what this PR does, 1-3 bullets>

## Test plan
- <how it was verified: which tests ran / sanity checks / manual steps>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

## 7. Report
Return the PR URL as a clickable markdown link and one line on what it contains.

## Guardrails
- Only run this when the user asked to push/PR — that request is the authorization.
- Never commit secrets or `.env`; never force-push; never commit to `main` directly.
- Interactive git flags (`-i`) are unavailable here — don't use `rebase -i` / `add -i`.
- If the diff contains anything surprising or out of scope, pause and confirm before pushing.
- Keep the commit focused on the current work; don't sweep in unrelated changes.
