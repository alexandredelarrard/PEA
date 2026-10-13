"""
command_line_interface.py  (src/constants/command_line_interface.py)
--------------------------------------------------------------------
Shared click option specs for the per-package `cli.py` modules (data_extract, data_peers,
data_aggregate, modelling, portfolio). Each is a `(*ARGS, **KWARGS)` pair so a command declares an option with
`@click.option(*X_ARGS, **X_KWARGS)`. The CLIs are the entry points the Airflow DAGs call via
`python -m src <package> <command> [options]`.
"""

from pathlib import Path
from typing import Any

import click

from src.constants.constants import DEFAULT_CONFIG_DIR

# path to the OmegaConf configs directory (every command needs it to build the Context)
CONFIG_ARGS = ("-c", "--config-path")
CONFIG_KWARGS: dict[str, Any] = dict(default=DEFAULT_CONFIG_DIR, show_default=True, help="Path to the OmegaConf configs directory.")

# optional comma-separated ticker subset; default = the full universe from `sp500_tickers`
TICKERS_ARGS = ("-t", "--tickers")
TICKERS_KWARGS: dict[str, Any] = dict(default=None, help="Comma-separated ticker subset (default: full sp500_tickers universe).")

# Replay the command's whole input window instead of resuming from stored progress.
FULL_ARGS = ("-F", "--full")
FULL_KWARGS: dict[str, Any] = dict(
    is_flag=True,
    default=False,
    help="Process the complete input window, including stored data; see the command help for source and write behavior.",
)

YEARS_ARGS = ("-y", "--years")
YEARS_KWARGS: dict[str, Any] = dict(
    type=int,
    default=None,
    help="Override data_extract.years_history for THIS run. A rebuild-from-scratch needs "
    "this to reach as far back as the incrementally-grown table it replaces.",
)

# Hidden run date for reproducible extraction runs.
AS_OF_ARGS = ("--as-of",)
AS_OF_KWARGS: dict[str, Any] = dict(
    type=click.DateTime(formats=["%Y-%m-%d"]), default=None, hidden=True, help="Run date YYYY-MM-DD (default: today)."
)

NO_CAP_ARGS = ("--no-cap",)
NO_CAP_KWARGS: dict[str, Any] = dict(is_flag=True, default=False, help="Lift the per-run document cap (one-time backlog runs only).")

DRY_RUN_ARGS = ("--dry-run",)
DRY_RUN_KWARGS: dict[str, Any] = dict(is_flag=True, default=False, help="Show the planned changes without applying them.")

TABLE_ARGS = ("--table", "table_name")
TABLE_KWARGS: dict[str, Any] = dict(default=None, help="One table; default: every table supported by this command.")

# ---- data extraction (`src/data_extract/cli.py`) ---- #
REPAIR_GAPS_ARGS = ("--repair-gaps", "repair")
REPAIR_GAPS_KWARGS: dict[str, Any] = dict(is_flag=True, default=False, help="Also re-read every day a key misses inside its stored span (one-time).")

FILING_WINDOW_ARGS = ("--filing-window",)
FILING_WINDOW_KWARGS: dict[str, Any] = dict(
    default=None,
    metavar="FROM:TO",
    help="Backfill a lost filing season, e.g. 2024-01-01:2024-03-01. Replaces the watermark and advances nothing. ONE EDGAR WALK AT A TIME.",
)

VINTAGES_ARGS = ("--vintages",)
VINTAGES_KWARGS: dict[str, Any] = dict(is_flag=True, default=False, help="Preserve explicit accession versions.")

CIK_ARGS = ("--cik", "ciks")
CIK_KWARGS: dict[str, Any] = dict(multiple=True, help="Filer CIK to recover (repeat for several filers).")

ACCESSION_ARGS = ("--accession", "accessions")
ACCESSION_KWARGS: dict[str, Any] = dict(multiple=True, help="Filing accession to recover (repeat for several accessions).")

PERIOD_START_ARGS = ("--period-start",)
PERIOD_START_KWARGS: dict[str, Any] = dict(type=str, default=None, help="First report period to recover.")

PERIOD_END_ARGS = ("--period-end",)
PERIOD_END_KWARGS: dict[str, Any] = dict(type=str, default=None, help="Last report period to recover.")

FILED_THROUGH_ARGS = ("--filed-through",)
FILED_THROUGH_KWARGS: dict[str, Any] = dict(type=str, default=None, help="Latest filing date eligible for recovery.")

MAX_FILINGS_ARGS = ("--max-filings",)
MAX_FILINGS_KWARGS: dict[str, Any] = dict(type=int, default=None, help="Maximum number of filings to recover.")

ARCHIVE_ARGS = ("--archive", "archives")
ARCHIVE_KWARGS: dict[str, Any] = dict(
    multiple=True, type=click.Path(exists=True, path_type=Path), help="Cached source archive (repeat for several archives)."
)

# Destructive facts recovery is distinct from a full, append-only filing replay.
REBUILD_ARGS = ("--rebuild",)
REBUILD_KWARGS: dict[str, Any] = dict(
    is_flag=True,
    default=False,
    help="Delete the requested tickers' SEC facts, history and reason codes, then refetch every filing. Use for a facts-resolution bug.",
)

VERIFY_HISTORY_ARGS = ("--verify-history",)
VERIFY_HISTORY_KWARGS: dict[str, Any] = dict(
    is_flag=True,
    default=False,
    help="Replay every ticker's SEC history and refuse if a published row would change. Writes only new events. Use after `fundamentals-facts -F`.",
)

PROPOSE_ARGS = ("--propose",)
PROPOSE_KWARGS: dict[str, Any] = dict(
    is_flag=True, default=False, help="Merge inert override candidates with approved: null; existing entries are never touched."
)

APPROVE_REKEY_ARGS = ("--approve-rekey", "approved_rekeys")
APPROVE_REKEY_KWARGS: dict[str, Any] = dict(
    multiple=True,
    metavar="OLD_ENTITY_ID:NEW_ENTITY_ID",
    help="Apply one exact older-CIK entity-id change that the build excluded and backlogged (WARNING log).",
)

REPAIR_AVAILABILITY_ARGS = ("--repair-availability",)
REPAIR_AVAILABILITY_KWARGS: dict[str, Any] = dict(
    is_flag=True,
    default=False,
    help="Rewrite notes available_at metadata using the historical estimate or cached download date; do not reparse ZIPs.",
)

TIMINGS_ARGS = ("--timings",)
TIMINGS_KWARGS: dict[str, Any] = dict(is_flag=True, default=False, help="Also print the plan-read timings.")

COUNT_ARGS = ("--count", "count_only")
COUNT_KWARGS: dict[str, Any] = dict(is_flag=True, default=False, help="Count only (the default without --delete).")

DELETE_ARGS = ("--delete",)
DELETE_KWARGS: dict[str, Any] = dict(is_flag=True, default=False, help="Delete the marker rows (rollback only).")

# ---- reporting ---- #
# A `validate` command prints its findings and writes a dated markdown report; these shape that
# output. Shared by `prices` and `institutionals`.
#
REPORT_PATH_ARGS = ("--report", "report_path")
REPORT_PATH_KWARGS: dict[str, Any] = dict(default=None, help="Where to write the markdown report (default: the command's report path).")

NO_WRITE_ARGS = ("--no-write",)
NO_WRITE_KWARGS: dict[str, Any] = dict(is_flag=True, default=False, help="Print only; write no file.")

# The D14 coverage gate. A feature below this share of non-null cells INSIDE its own
# availability window is listed for the cut -- never scored over all history, which would fail
# a 2024-12-17 feature for a property of the calendar.
#
# ⚠ NO `default` HERE ON PURPOSE. The number lives beside the gate that applies it
# (`src/validate/institutionals.COVERAGE_FLOOR`) and the call site passes it in, so the floor
# is declared once. A copy here would be a second declaration of D14 -- and the one that gets
# edited is never the one that is read.
COVERAGE_FLOOR_ARGS = ("--coverage-floor",)
COVERAGE_FLOOR_KWARGS: dict[str, Any] = dict(
    type=float, show_default=True, help="Coverage gate: a feature below this non-null share inside its availability window is listed for the cut."
)
