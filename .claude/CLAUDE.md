## Communication style
- Be concise. Skip preamble — go straight to the answer or the change.
- Show me diffs, not full file rewrites, for small changes.
- If something is ambiguous, ask one clarifying question before proceeding.

## My workflow
- I use python across all projects
- Functions called multiple times are in utils and imported from other class / methods 
- I have Step class where most important tasks are done with a run method to call all the workflow
- When I say "clean this up", I mean: improve readability and remove dead code only.
  Do not change logic or architecture.

## Coding defaults I apply everywhere
- Strict Typing for each function
- Named exports
- Error handling with typed errors
- Context to import 