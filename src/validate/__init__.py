"""
src/validate/ -- THE home for validation code, across every domain.

Part 2 of the three-part loop this repo's data runs on:

    1 EXTRACTION  (src/data_extract/)  ->  raw tables
    2 VALIDATION  (here)               ->  a ranked, explained finding queue. MUTATES NOTHING.
    3 BUGFIX      (an agent)           ->  a settled outcome recorded in configs/, then re-run 2

Read `README.md` before adding a check. It is the operating manual, and its most useful
section is the one on when the validator DOES NOT WORK.

Nothing here imports from another `src/` subfolder's internals beyond the published
catalogue / reason-code vocabulary, and nothing here writes to any table but
`fundamentals_check`.
"""
