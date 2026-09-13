"""
command_line_interface.py  (src/constants/command_line_interface.py)
--------------------------------------------------------------------
Shared click option specs for the per-package `cli.py` modules (data_extract, data_peers,
data_aggregate, modelling, portfolio). Each is a `(*ARGS, **KWARGS)` pair so a command declares an option with
`@click.option(*X_ARGS, **X_KWARGS)`. The CLIs are the entry points the Airflow DAGs call via
`python -m src <package> <command> [options]`.
"""

from src.constants.constants import DEFAULT_CONFIG_DIR

# path to the OmegaConf configs directory (every command needs it to build the Context)
CONFIG_ARGS = ("-c", "--config-path")
CONFIG_KWARGS = dict(default=DEFAULT_CONFIG_DIR, show_default=True,
                     help="Path to the OmegaConf configs directory.")

# optional comma-separated ticker subset; default = the full universe from `sp500_tickers`
TICKERS_ARGS = ("-t", "--tickers")
TICKERS_KWARGS = dict(default=None,
                      help="Comma-separated ticker subset (default: full sp500_tickers universe).")

# force a full refresh / re-fetch instead of the default incremental (resume-from-DB) behaviour
FORCE_ARGS = ("-f", "--force")
FORCE_KWARGS = dict(is_flag=True, default=False, help="Force a full refresh (ignore incremental state).")

FULL_ARGS = ("-F", "--full")
FULL_KWARGS = dict(is_flag=True, default=False,
                    help="Ignore the run manifest and take the whole years-history window. "
                         "Needed for a chunked from-scratch backfill.")

YEARS_ARGS = ("-y", "--years")
YEARS_KWARGS = dict(
    type=int, default=None,
    help="Override data_extract.years_history for THIS run. A rebuild-from-scratch needs "
         "this to reach as far back as the incrementally-grown table it replaces.")

# ---- validate (`src/validate/cli.py`) ---- #
# A `validate` command prints its findings and writes a dated markdown report; these shape that
# output. Shared by `prices` and `institutionals`.
#
# `validate fundamentals` deliberately keeps its OWN copies of both: its `--report` writes a
# `.json` beside the markdown, and its `--no-write` also suppresses the append to
# `fundamentals_check` (so no delta is possible and the report says so). Those are different
# options that happen to share a name -- folding them in here would make one help string lie.
REPORT_PATH_ARGS = ("--report", "report_path")
REPORT_PATH_KWARGS = dict(default=None,
                          help="Where to write the markdown report "
                               "(default: reports/validate/<date>/<scope>.md).")

NO_WRITE_ARGS = ("--no-write",)
NO_WRITE_KWARGS = dict(is_flag=True, default=False, help="Print only; write no file.")

# The D14 coverage gate. A feature below this share of non-null cells INSIDE its own
# availability window is listed for the cut -- never scored over all history, which would fail
# a 2024-12-17 feature for a property of the calendar.
#
# ⚠ NO `default` HERE ON PURPOSE. The number lives beside the gate that applies it
# (`src/validate/institutionals.COVERAGE_FLOOR`) and the call site passes it in, so the floor
# is declared once. A copy here would be a second declaration of D14 -- and the one that gets
# edited is never the one that is read.
COVERAGE_FLOOR_ARGS = ("--coverage-floor",)
COVERAGE_FLOOR_KWARGS = dict(type=float, show_default=True,
                             help="Coverage gate: a feature below this non-null share inside "
                                  "its availability window is listed for the cut.")
