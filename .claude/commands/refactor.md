Refactor [part of code to refactor given by user] and all the dependencies touching it, without changing its logic of data aggregation, feature creation and sanity checks.

Use the relevant Superpowers skills explicitly (extension pluggin).
Workflow:
1. Use OpenKnowledge and native workspace search to understand:
   - the subsystem architecture
   - entry points
   - callers and dependencies
   - public interfaces
   - persistence or protocol contracts
   - existing tests
   - related architecture decisions
2. Use brainstorming to define:
   - the specific refactoring goal
   - behavior that must remain unchanged
   - architectural invariants
   - non-goals
   - measurable completion criteria
3. Before modifying code:
   - inspect the current Git status
   - run the complete existing test suite
   - run the build, linter, formatter check, and type checker
   - record all existing failures separately
   - request a baseline code review
   - identify missing characterization tests
4. Use using-git-worktrees to create an isolated workspace unless
this session is already isolated.
5. Use writing-plans to create a step-by-step refactoring plan.
Each task must:
   - have a narrow responsibility
   - name exact files
   - preserve behavior
   - include its verification command
   - be independently reviewable
   - end with a small commit
6. Add characterization or regression tests before changing
unprotected behavior.
7. Execute using:
   - subagent-driven-development when appropriate, or
   - executing-plans when subagents are unavailable
8. Refactor incrementally. Prefer:
   - rename
   - extract function or class
   - move code
   - introduce an interface or seam
   - invert a dependency
   - remove verified dead code
   Avoid broad rewrites and unrelated feature changes.
9. After every task:
   - run the focused tests
   - run affected integration tests
   - inspect the diff
   - ensure no public contract changed unintentionally
10. If anything fails, use systematic-debugging and identify the
root cause before changing more code.
11. Before completion, use verification-before-completion:
    - run the complete test suite
    - run the full build
    - run linting and formatting checks
    - run type checking
    - compare behavior against the baseline
    - inspect the final Git diff
    - request a final code review
12. Refresh the OpenKnowledge codebase wiki for architecture,
module, flow, and source-reference changes.
Do not claim completion without fresh command output.
Do not change dependencies unless the approved plan requires it.
Do not combine the refactor with new product functionality.
