import click
from typing import List

from src.utils.cli_helper import assert_valid_url

CONFIG_ARGS = ("--config", "-c", "config_path")
CONFIG_KWARGS = {
    "default": "./configs",
    "show_default": True,
    "help": (
        "The path to the configuration folder for the run. "
        "The config is recursively created or merged with the help of Omegaconf python lib"
    ),
}
