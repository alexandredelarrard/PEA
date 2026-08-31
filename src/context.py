import logging
import os

from io import StringIO
from pathlib import Path
import sys
from logging.config import dictConfig

from dotenv import load_dotenv, find_dotenv
from omegaconf import DictConfig, OmegaConf
import requests

from src.utils.config import read_config
from src.utils.seed import set_seed
from src.utils.db import get_engine
from src.data_store.store import DataStore

os.environ['LC_ALL'] = "C"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

def check_path_exist(path):
    path = Path(path)
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)

def define_global_paths(config: DictConfig):
    """Filesystem paths for NON-tabular artifacts only. All tabular data
    (prices, fundamentals, cube, predictions, ...) now lives in the database
    and is accessed via `context.store`; the parquet/CSV paths were removed
    when the pipeline moved to DB-only I/O.
    """
    global_paths = {}

    global_paths["ROOT"] = Path(os.getenv("ROOT_PATH", config.local.paths.root))
    data_store = config.local.paths.get("data_store", "data_store")
    global_paths["DATA_STORE"] = global_paths["ROOT"] / data_store

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

    def __init__(self, config: DictConfig, use_cache: bool, save: bool, config_dir: Path):

        self._config = config
        self._config_dir = config_dir

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

        # EDGAR identity / SEC session (lazily built -- see the properties below)
        self._sec_user_agent: str | None = None
        self._edgar_identity_set = False
        self._sec_session: requests.Session | None = None

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
    def config_dir(self) -> Path:
        """The OmegaConf configs directory THIS context was built from, resolved to an
        absolute path so it matches whatever spelling (`-c ./configs`, `-c /abs/path`) reached
        the CLI. Fundamentals loaders that key an `@cache` on a config directory
        (`kpi_catalogue.resolve_config_dir` and its callers) need this exact value passed in
        explicitly -- their own default parameter is a different, redundant copy of
        `constants.DEFAULT_CONFIG_DIR` and does NOT see the CLI's `-c` flag."""
        return self._config_dir

    @property
    def use_cache(self) -> bool:
        return self._use_cache

    @property
    def save(self) -> bool:
        return self._save

    @property
    def random_state(self) -> int:
        return self._config.seed

    @property
    def sec_user_agent(self) -> str:
        """SEC EDGAR blocks requests without a descriptive User-Agent (name + email) --
        read from the env ONCE and cached. Replaces two hand-written copies of this same
        check (`edgar_driver._sec_headers` / `sec_utils._sec_headers`), which raised the
        same error from two places and re-read `os.getenv` on every single request."""
        if self._sec_user_agent is None:
            ua = os.getenv("SEC_USER_AGENT", "").strip()
            if not ua:
                raise RuntimeError(
                    "SEC_USER_AGENT is not set. SEC EDGAR blocks requests without a "
                    "descriptive User-Agent (name + email). Add it to your .env file, e.g.\n"
                    '  SEC_USER_AGENT="Your Name your.email@example.com"\n'
                    "See https://www.sec.gov/os/webmaster-faq#developers")
            self._sec_user_agent = ua
        return self._sec_user_agent

    def ensure_edgar_identity(self) -> None:
        """Configure edgartools' process-global identity from `sec_user_agent`. Idempotent:
        a no-op after the first call, replacing a `set_identity` that used to run once per
        fetch-run (or once per ticker, on the 13F path) for a value that never changes
        within a process. Imports `edgar` lazily -- `Context` is imported by every package,
        including ones that never touch SEC EDGAR."""
        if self._edgar_identity_set:
            return
        from edgar import set_identity
        set_identity(self.sec_user_agent)
        self._edgar_identity_set = True

    @property
    def sec_session(self) -> requests.Session:
        """One `requests.Session` with the SEC User-Agent pre-set on `session.headers`,
        shared by `sec_utils.sec_get` and `bulk_cache.ensure_zip`. Replaces a header dict
        rebuilt -- and `SEC_USER_AGENT` re-read from the env -- on every single request, and
        gives the multi-hundred-MB bulk-ZIP downloads connection reuse."""
        if self._sec_session is None:
            session = requests.Session()
            session.headers.update({
                "User-Agent": self.sec_user_agent,
                "Accept-Encoding": "gzip, deflate",
            })
            self._sec_session = session
        return self._sec_session


def get_config_context(config_path: str, use_cache: bool, save: bool):

    config_dir = Path(config_path).resolve()
    try:
        config = read_config(path=config_dir)
        dictConfig(OmegaConf.to_container(config.logging))
        set_seed(config)
    except FileNotFoundError:
        print(f"configuration file {config_path} not found ", file=sys.stderr)
        sys.exit(1)

    context = Context(config=config, use_cache=use_cache, save=save, config_dir=config_dir)

    return config, context
