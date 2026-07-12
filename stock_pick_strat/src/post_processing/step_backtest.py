import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.utils.step import Step
from src.context import Context


class StepModelling(Step):
    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    def run(self):
        pass