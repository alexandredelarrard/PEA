---
description: Deliberately skip the definition-of-done report for this task, with a stated reason.
argument-hint: <reason the report is not warranted>
---

The user has asked to skip the definition-of-done report for the current task.

Reason given: **$ARGUMENTS**

Do this, in order:

1. If the reason is empty, ask for one. A silent skip is exactly what
   [docs/definition_of_done.md](../../docs/definition_of_done.md) exists to prevent — the skip
   must be on the record.
2. Write the reason to the session's skip marker so the `Stop` hook stands down for this task:

   ```bash
   PY="$HOME/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe"
   "$PY" -S -E .claude/hooks/dod_stop.py --skip "$ARGUMENTS"
   ```

3. Tell the user, in one line, that the report is skipped and repeat the reason back so it is
   visible in the transcript.

Note for the assistant: this command is **user-invoked only**. Never run it on your own
initiative to get past the hook. If you believe the hook misclassified the task, say so in plain
words and let the user decide — `docs/definition_of_done.md` is explicit that you should not
fight the hook.
