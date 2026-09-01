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
