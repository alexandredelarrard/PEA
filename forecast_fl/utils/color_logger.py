# -*- coding: utf-8 -*-
import logging
import time
from contextlib import contextmanager
from pathlib import Path

import click


class ColorHandler(logging.StreamHandler):
    """
    A color log handler.
    debug: magenta
    info: cyan
    warning: yellow
    error: red
    critical: red
    """

    def __init__(self, stream=None, colors=None):
        logging.StreamHandler.__init__(self, stream)
        colors = colors or {}
        self.colors = {
            "critical": colors.get("critical", "red"),
            "error": colors.get("error", "red"),
            "warning": colors.get("warning", "yellow"),
            "info": colors.get("info", "cyan"),
            "debug": colors.get("debug", "magenta"),
        }

    def _get_color(self, level):
        if level >= logging.CRITICAL:
            return self.colors["critical"]  # pragma: no cover
        if level >= logging.ERROR:
            return self.colors["error"]  # pragma: no cover
        if level >= logging.WARNING:
            return self.colors["warning"]  # pragma: no cover
        if level >= logging.INFO:
            return self.colors["info"]
        if level >= logging.DEBUG:  # pragma: no cover
            return self.colors["debug"]  # pragma: no cover

        return None  # pragma: no cover

    def format(self, record: str) -> str:
        """The handler formatter.
        Args:
            record: The record to format.
        Returns:
            The record formatted as a string.
        """
        text = logging.StreamHandler.format(self, record)
        color = self._get_color(record.levelno)
        return click.style(text, color)


class MakeFileHandler(logging.FileHandler):
    """
    A custom handler that creates a new logging file at each instantiation (usually beginning of run)
    """

    def __init__(self, filename, encoding=None):
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        version = time.strftime("%Y%m%d")
        versioned_filename = filepath.parent / (filepath.stem + f"_{version}" + filepath.suffix)
        logging.FileHandler.__init__(self, versioned_filename, mode="a", encoding=encoding, delay=0)


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.
    param highest_level: the maximum logging level in use. This would only need to be changed if a custom level greater than CRITICAL is defined.
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)
