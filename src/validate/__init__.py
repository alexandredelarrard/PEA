"""
src/validate/ -- THE home for validation code, across every domain.

Part 2 of the three-part loop this repo's data runs on:

    1 EXTRACTION  (src/data_extract/)  ->  raw tables
    2 VALIDATION  (here)               ->  a ranked, explained finding queue.
    3 BUGFIX      (an agent)           ->  a settled outcome recorded in configs/, then re-run 2

Nine table-agnostic checks, one CLI command each (`python -m src validate <check> -T <table>
-o reports/validate/<slug>`), and importable from here so a report's `_scripts/` can compose
them without a process boundary:

    from src.validate import check_redundancy
    result = check_redundancy(context, "cube_part_institutionals", config=config)
    if result.status != "pass":
        ...                        # `abstain` is not a pass -- see `result.py`

None of the check functions prints, exits or writes a file: `cli.py` owns all three, so the
exit-code contract (0 pass / 1 finding / 3 ABSTAINED) lives in exactly one place.
"""
from src.validate.checks import (check_bounds, check_catalogue, check_clip, check_coverage,
                                 check_grain, check_leakage, check_profile, check_redundancy,
                                 check_timeseries)
from src.validate.result import EXIT, CheckResult, Finding, gate, severity_for
from src.validate.spec import TableSpec, UndeclaredTableError, load_spec

__all__ = [
    # the nine checks
    "check_bounds", "check_catalogue", "check_clip", "check_coverage", "check_grain",
    "check_leakage", "check_profile", "check_redundancy", "check_timeseries",
    # the contract they return, and the exit codes it maps to
    "CheckResult", "Finding", "EXIT", "gate", "severity_for",
    # what a table is allowed to be assumed about
    "TableSpec", "UndeclaredTableError", "load_spec",
]
