import logging
import os

from io import StringIO
from pathlib import Path
import sys
from logging.config import dictConfig

from dotenv import load_dotenv, find_dotenv
from omegaconf import DictConfig, OmegaConf

from src.utils.utils_read_files import check_path_exist
from src.utils.config import read_config
from src.utils.seed import set_seed

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

def define_global_paths(config: DictConfig):

    global_paths = {}

    global_paths["ROOT"] = Path(os.getenv("ROOT_PATH", config.local.paths.root))
    data_store = config.local.paths.get("data_store", "data_store")
    global_paths["DATA_STORE"] = global_paths["ROOT"] / data_store

    # Prices & universe
    global_paths["PRICES_PATH"] = global_paths["DATA_STORE"] / "prices.parquet"
    global_paths["TICKERS_PATH"] = global_paths["DATA_STORE"] / "sp500_tickers.csv"
    global_paths["DIVIDENDS_PATH"] = global_paths["DATA_STORE"] / "dividends.parquet"

    # Fundamentals
    global_paths["FUNDAMENTALS_SNAPSHOT_PATH"] = global_paths["DATA_STORE"] / "fundamentals_latest.parquet"
    global_paths["FUNDAMENTALS_HISTORY_PATH"] = global_paths["DATA_STORE"] / "fundamentals_history.parquet"

    # Macro & news
    global_paths["MACRO_PATH"] = global_paths["DATA_STORE"] / "macro.parquet"
    global_paths["NEWS_PATH"] = global_paths["DATA_STORE"] / "news_latest.parquet"

    # Retail-attention alt-data (Wikipedia pageviews + Google Trends)
    global_paths["WIKI_PAGEVIEWS_PATH"] = global_paths["DATA_STORE"] / "wiki_pageviews.parquet"
    global_paths["GOOGLE_TRENDS_PATH"] = global_paths["DATA_STORE"] / "google_trends.parquet"

    # Analyst estimates
    global_paths["ANALYST_ESTIMATES_PATH"] = global_paths["DATA_STORE"] / "analyst_estimates.parquet"
    global_paths["ANALYST_ESTIMATES_HISTORY_PATH"] = global_paths["DATA_STORE"] / "analyst_estimates_history.parquet"

    # Management / ownership (yfinance snapshot; accrues history over time)
    global_paths["MANAGEMENT_PATH"] = global_paths["DATA_STORE"] / "management.parquet"
    global_paths["MANAGEMENT_HISTORY_PATH"] = global_paths["DATA_STORE"] / "management_history.parquet"

    # Employee counts (FMP historical-employee-count; genuinely historical, from filings)
    global_paths["EMPLOYEES_HISTORY_PATH"] = global_paths["DATA_STORE"] / "employees_history.parquet"

    # FMP historical endpoints (one call = full ticker history; see fetch_fmp_history)
    global_paths["ANALYST_GRADES_HISTORY_PATH"] = global_paths["DATA_STORE"] / "analyst_grades_history.parquet"
    global_paths["ANALYST_ACTIONS_HISTORY_PATH"] = global_paths["DATA_STORE"] / "analyst_actions_history.parquet"
    global_paths["EXEC_COMP_HISTORY_PATH"] = global_paths["DATA_STORE"] / "exec_compensation_history.parquet"
    global_paths["FMP_ESTIMATES_HISTORY_PATH"] = global_paths["DATA_STORE"] / "fmp_estimates_history.parquet"

    # Earnings surprises & PEAD
    global_paths["EARNINGS_SURPRISES_PATH"] = global_paths["DATA_STORE"] / "earnings_surprises.parquet"
    global_paths["PEAD_RESULTS_PATH"] = global_paths["DATA_STORE"] / "pead_results.parquet"

    # SEC filings
    global_paths["CIK_MAPPING_PATH"] = global_paths["DATA_STORE"] / "cik_mapping.csv"
    global_paths["SEC_FILINGS_INDEX_PATH"] = global_paths["DATA_STORE"] / "sec_filings_index.parquet"
    global_paths["SEC_FILINGS_TEXT_DIR"] = global_paths["DATA_STORE"] / "sec_filings_text"
    global_paths["SEC_BULK_CACHE_DIR"] = global_paths["DATA_STORE"] / "sec_bulk_cache"

    # Insider & institutional (SEC bulk data)
    global_paths["INSIDER_TRANSACTIONS_PATH"] = global_paths["DATA_STORE"] / "insider_transactions.parquet"
    global_paths["INSTITUTIONAL_HOLDINGS_PATH"] = global_paths["DATA_STORE"] / "institutional_holdings.parquet"
    global_paths["CUSIP_MAP_PATH"] = global_paths["DATA_STORE"] / "cusip_ticker_map.parquet"

    # Strategy outputs
    global_paths["BACKTEST_RESULT_PATH"] = global_paths["DATA_STORE"] / "backtest_result.parquet"

    # Cube / model pipeline outputs
    global_paths["OUTPUT_DIR"] = global_paths["DATA_STORE"] / "output"
    global_paths["SECTOR_PEERS_PATH"] = global_paths["OUTPUT_DIR"] / "sector_peers.json"
    global_paths["CUBE_PATH"] = global_paths["OUTPUT_DIR"] / "cube.parquet"
    global_paths["PREDICTIONS_PATH"] = global_paths["OUTPUT_DIR"] / "predictions.parquet"
    global_paths["CUBE_SIGNAL_PATH"] = global_paths["OUTPUT_DIR"] / "cube_signal.parquet"
    global_paths["CUBE_CV_RESULTS_PATH"] = global_paths["OUTPUT_DIR"] / "cube_cv_results.parquet"
    global_paths["CUBE_PANEL_PATH"] = global_paths["OUTPUT_DIR"] / "cube_panel.parquet"
    global_paths["MODELS_DIR"] = global_paths["OUTPUT_DIR"] / "models"

    for _, path in global_paths.items():
        if "https" not in str(path):
            check_path_exist(path)

    return global_paths


class Context:

    def __init__(self, config: DictConfig, use_cache: bool, save: bool):

        self._config = config

        # define paths
        self.paths = define_global_paths(self._config)

        # settup logging
        self._setup_logging()

        # env variables
        self._load_env()

        self._use_cache = use_cache
        self._save = save

    def _setup_logging(self):
        """Setup logging configuration"""

        # create logging buffer
        buffer = StringIO()
        handler = logging.StreamHandler(buffer)
        formatter = logging.Formatter(self._config.logging.formatters.file.format)
        handler.setFormatter(formatter)
        logging.root.addHandler(handler)
        self.log_buffer = buffer

        self.log = logging.getLogger(__name__)

    def _load_env(self):
        """Load environment variables"""
        dot_env_file = find_dotenv(usecwd=True)
        if dot_env_file:
            load_dotenv(dot_env_file)
            self.log.info(f"Loaded environment file: {dot_env_file}")
        else:
            self.log.warning("No .env file found")

    @property
    def config(self) -> DictConfig:
        return self._config

    @property
    def use_cache(self) -> bool:
        return self._use_cache

    @property
    def save(self) -> bool:
        return self._save

    @property
    def random_state(self) -> int:
        return self._config.seed


def get_config_context(config_path: str, use_cache: bool, save: bool):

    try:
        config = read_config(path="./configs")
        dictConfig(OmegaConf.to_container(config.logging))
        set_seed(config)
    except FileNotFoundError:
        print(f"configuration file {config_path} not found ", file=sys.stderr)
        sys.exit(1)

    context = Context(config=config, use_cache=use_cache, save=save)

    return config, context
