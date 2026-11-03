import logging
import warnings
from abc import abstractmethod
from datetime import datetime

from omegaconf import DictConfig

from src.context import Context
from src.utils.string import camel_to_snake
from src.utils.tickers_universe.universe import Universe

warnings.filterwarnings("ignore", category=UserWarning)


class Step:
    def __init__(
        self,
        config: DictConfig,
        context: Context,
    ):
        # `type(self).__module__`, not `__name__`: bound here, `__name__` is always
        # "src.utils.step", so every step's lines were attributed to this file and no log
        # could say WHICH step emitted them.
        self._log: logging.Logger = logging.getLogger(type(self).__module__)
        self._config = config
        self._name = self.get_name()
        self._log.info(f"starting step {self}")
        self._today = datetime.today()
        self._context = context
        self._config_dir = context.config_dir

    @property
    def name(self):
        return self._name

    @property
    def _universe(self) -> Universe:
        """`context.universe`, read on first use, so constructing a step reads nothing."""
        return self._context.universe

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def get_name(cls):
        return camel_to_snake(cls.__name__)
