import logging
from abc import abstractmethod
from omegaconf import DictConfig
import warnings
from datetime import datetime

from src.context import Context
from src.utils.string import camel_to_snake

warnings.filterwarnings("ignore", category=UserWarning)

class Step:

    def __init__(
        self,
        config: DictConfig,
        context: Context,
    ):
        self._log: logging.Logger = logging.getLogger(__name__)
        self._config = config
        self._name = self.get_name()
        self._log.info(f"starting step {self}")
        self._today = datetime.today()
        self._context = context

    @property
    def name(self):
        return self._name

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def get_name(cls):
        return camel_to_snake(cls.__name__)
