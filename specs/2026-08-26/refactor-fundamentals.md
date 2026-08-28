

## Ask / request

I want you to research how to refactor / simplify  src/data_extract/utils/fundamentals and all its dependencies. 
Making code, faster, more efficient, and cleaner, with less globals, large functions.

For context : 
- fundamentals fetch data from sec 
- fill fundamentals_facts 
- then fundamentals_history_sec 
- other sec extraction of tables 

### Goals: Research where those are not respected in the codebase

1. Readability & Simplification: Remove dead code, reduce nesting, use clear variable names, and follow best practices for python codebase.
2. Performance & Efficiency: Optimize time and space complexity, remove bottlenecks, and minimize memory overhead.
3. Maintainability: Break down large functions into smaller, single-responsibility functions.

## Context of what I see so far. (not exhaustive)

What I want you to focus on is EFFICIENCY of the code.

- I see that fetch_fundamentals_sec is slow and has lots of messages like, issue of connection or not found -> is there a bug there ? investigate 
"2026-08-27 09:49:52 - edgar.sgml.sgml_common - WARNING - sgml_common.py - SGML header declares 106 public document(s) but only 104 were parsed (accession 0000950170-24-023115). The submission may be truncated or malformed.
2026-08-27 09:51:17 - edgar.core - WARNING - _filings.py - SGML fetch failed for 0001104659-14-053327, falling back to homepage: peer closed connection without sending complete message body (incomplete chunked read)
2026-08-27 09:51:38 - edgar.sgml.sgml_common - WARNING - sgml_common.py - SGML header declares 124 public document(s) but only 122 were parsed (accession 0000950170-24-021195). The submission may be truncated or malformed.
2026-08-27 09:51:40 - edgar.sgml.sgml_common - WARNING - sgml_common.py - SGML header declares 203 public document(s) but only 201 were parsed (accession 0001013871-24-000005). The submission may be truncated or malformed."

- record_run is a method which should be usede in each fundamentals function saving info, and generalized such that latest info on extraction is saved in a clear json.
- ./configs folder is defined in configs / context, imported at first in large steps. Leverage it into sub functions / modules, instead of declaring it as a global function (CONFIG_DIR = "./configs" in severalg functions and tests)

- remove redundant functions replaced by a more generic function from utils (generalization principle)
- Class Step, declared in utils, should be used as a class recording all important variables, paths, client connections, then inherited by classes such that no need to reference anymore any direct path or new connections in each file. 

- constants.py is massive (1000 lines now) and hard to maintain, think if possibility to simplify it, put global variables in underlying functions if only called by one file (plus its test). 
- minimal docstring / comment to the strict minimum to what is needed, files have too much verbosity. 
- Remove the verbosity on the history of the checks and changes. The code needs to be seen as of today, not as a difference with context of what was before vs now (comments and docstrings to verify)


## proposed steps to reserach for futur plan build

- read Agents.md  
- read docs/coding_standard.md 
- leverage agent codebase-locator.md if necessary 
- Read the entire sets of functions used in class StepExtractFundamentals(Step)
- Read entire sets of functions used in class StepExtractPrices(Step) -> use it as a reference since its been ok as a refactor
- always think in a generic fashion, for reusability, clean code and lowest complexity possible.
- Build a clear and constructive research graph / definition and outcome, with sources. 
- The research will help then the step 2 : Plan 