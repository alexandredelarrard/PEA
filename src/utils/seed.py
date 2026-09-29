import logging
import random

from numpy import random as rd
from omegaconf import DictConfig


def set_seed(config: DictConfig):

    seed = config.seed
    if seed is None:
        raise ValueError("No seed was provided.")
    else:
        logging.getLogger().info(f"Set seed to value = {seed}")
        random.seed(seed)
        rd.seed(seed)
