"""
src/validate/checks/ -- the nine table-agnostic check functions.

Every one has the same signature and the same return type:

    check_<name>(context, table, *, config, cache=None, tickers=None, **kwargs) -> CheckResult

so `cli.py` can register all nine identically and a report's `_scripts/` can compose them.
None of them prints, exits, or writes a file -- `cli.py` owns all three.

The nine are the concepts that recurred across all five report families of the 2026-09-21
survey, each previously re-implemented three to five times with different thresholds and
different pass rules.
"""
from src.validate.checks.bounds import check_bounds
from src.validate.checks.catalogue import check_catalogue
from src.validate.checks.clip import check_clip
from src.validate.checks.coverage import check_coverage
from src.validate.checks.grain import check_grain
from src.validate.checks.leakage import check_leakage
from src.validate.checks.profile import check_profile
from src.validate.checks.redundancy import check_redundancy
from src.validate.checks.timeseries import check_timeseries

__all__ = [
    "check_bounds", "check_catalogue", "check_clip", "check_coverage", "check_grain",
    "check_leakage", "check_profile", "check_redundancy", "check_timeseries",
]
