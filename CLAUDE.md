## Risk zones — ask before editing

| File / directory | Why it's a risk zone |
|---|---|
| `src/context.py` | Imported by every step class — changes cascade everywhere |
| `src/utils/step.py` | Base class for all steps — changes break all inheritors |
| `src/constants/*.py` | Renaming any constant breaks all downstream references |
| `configs/*.yaml` | Structural changes must be mirrored in all consuming code |
| `data/` (existing files) | Overwriting saved parquet/model files is not recoverable |

For any of the above: propose the change and wait for approval before editing.

---

## Workflow for new features

1. Check `src/constants/*.py` before naming anything — add there first if missing
2. Check `src/utils/` for existing helpers before writing a new one
3. Implement the feature
4. Write the test alongside the implementation — not after
5. Run `pytest tests/path/to/new_test.py::test_function -v -s`
6. Show me **only the output of the new test**, not the full pytest summary
7. The test output must include the printed sanity check conclusion — if it doesn't, the work is not done

---

## How to communicate results

- When a task is done: show the new test output + the printed sanity check conclusion
- Do not show the full list of all passing tests — only the new ones
- If a refactor touches existing tests, tell me which ones and why, but don't dump all output
- For multi-step work: confirm each step before moving to the next

---

## What to do automatically

- Always check `src/constants/*.py` before introducing a new column name or string key
- After implementing a feature, propose the unit test before marking done
- Log via `self._context.logger`, never `print()`
- When writing a new config key, add it to the appropriate `configs/*.yaml`
- When a helper is useful across folders, place it in `src/utils/` not inline

---

## What NOT to do

- Do not replace OmegaConf with another config system
- Do not restructure the Step inheritance pattern
- Do not cross-import between `src/` subfolders (e.g. data_extraction importing from modelling)
- Do not hardcode strings or paths that belong in `src/constants/`
- Do not reformat code unrelated to the current task
- Do not say work is done without the printed sanity check conclusion in the test output