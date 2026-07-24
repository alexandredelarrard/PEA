"""
command_line_interface.py  (src/constants/command_line_interface.py)
--------------------------------------------------------------------
Shared click option specs for the per-package `cli.py` modules (data_extract, data_peers,
data_aggregate). Each is a `(*ARGS, **KWARGS)` pair so a command declares an option with
`@click.option(*X_ARGS, **X_KWARGS)`. The CLIs are the entry points the Airflow DAGs call via
`python -m src <package> <command> [options]`.
"""

# path to the OmegaConf configs directory (every command needs it to build the Context)
CONFIG_ARGS = ("-c", "--config-path")
CONFIG_KWARGS = dict(default="./configs", show_default=True,
                     help="Path to the OmegaConf configs directory.")

# optional comma-separated ticker subset; default = the full universe from `sp500_tickers`
TICKERS_ARGS = ("-t", "--tickers")
TICKERS_KWARGS = dict(default=None,
                      help="Comma-separated ticker subset (default: full sp500_tickers universe).")

# force a full refresh / re-fetch instead of the default incremental (resume-from-DB) behaviour
FORCE_ARGS = ("-f", "--force")
FORCE_KWARGS = dict(is_flag=True, default=False, help="Force a full refresh (ignore incremental state).")
