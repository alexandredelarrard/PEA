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
from src.utils.db import get_engine
from src.data_store.store import DataStore

os.environ['LC_ALL'] = "C"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

def define_global_paths(config: DictConfig):
    """Filesystem paths for NON-tabular artifacts only. All tabular data
    (prices, fundamentals, cube, predictions, ...) now lives in the database
    and is accessed via `context.store`; the parquet/CSV paths were removed
    when the pipeline moved to DB-only I/O.

    The `*_HISTORY_PATH` / `SEC_FILINGS_INDEX_PATH` / `DEF14A_LLM_PATH` entries
    are kept ONLY to anchor each fetcher's incremental `_meta.json` sidecar
    (built via `<path>.with_name("<stem>_meta.json")`); no parquet is written
    to them.
    """
    global_paths = {}

    global_paths["ROOT"] = Path(os.getenv("ROOT_PATH", config.local.paths.root))
    data_store = config.local.paths.get("data_store", "data_store")
    global_paths["DATA_STORE"] = global_paths["ROOT"] / data_store

    # Raw file caches (not tabular -> stay on disk)
    global_paths["SEC_BULK_CACHE_DIR"] = global_paths["DATA_STORE"] / "sec_bulk_cache"
    global_paths["SEC_FILINGS_TEXT_DIR"] = global_paths["DATA_STORE"] / "sec_filings_text"

    # Incremental meta-sidecar anchors (the parquet file itself is unused; only
    # its stem names the sibling "<stem>_meta.json" the fetcher reads/writes)
    global_paths["FUNDAMENTALS_HISTORY_PATH"] = global_paths["DATA_STORE"] / "fundamentals_history.parquet"
    global_paths["MANAGEMENT_HISTORY_PATH"] = global_paths["DATA_STORE"] / "management_history.parquet"
    global_paths["EMPLOYEES_HISTORY_PATH"] = global_paths["DATA_STORE"] / "employees_history.parquet"
    global_paths["DEF14A_LLM_PATH"] = global_paths["DATA_STORE"] / "def14a_llm.parquet"
    global_paths["SEC_FILINGS_INDEX_PATH"] = global_paths["DATA_STORE"] / "sec_filings_index.parquet"

    # Pipeline output artifacts (models, diagnostics, peer dict, CV results)
    global_paths["OUTPUT_DIR"] = global_paths["DATA_STORE"] / "output"
    global_paths["MODELS_DIR"] = global_paths["OUTPUT_DIR"] / "models"
    global_paths["SECTOR_PEERS_PATH"] = global_paths["OUTPUT_DIR"] / "sector_peers.json"
    global_paths["CUBE_CV_RESULTS_PATH"] = global_paths["OUTPUT_DIR"] / "cube_cv_results.parquet"

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

        # database-backed data store (replaces parquet I/O for tabular data;
        # `paths` are kept only for non-tabular artifacts: models, plots,
        # SEC JSON caches, filing text)
        self.store = DataStore(get_engine())

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
