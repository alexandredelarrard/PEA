import logging
from logging import LogRecord
import click
import time
from pathlib import Path
from contextlib import contextmanager


class ColorHandler(logging.StreamHandler):

    def __init__(self, stream=None, colors=None, **kwargs):
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
            return self.colors["critical"]
        if level >= logging.ERROR:
            return self.colors["error"]
        if level >= logging.WARNING:
            return self.colors["warning"]
        if level >= logging.DEBUG:
            return self.colors["debug"]
        if level >= logging.INFO:
            return self.colors["info"]

        return None

    def format(self, record: LogRecord) -> str:

        text = logging.StreamHandler.format(self, record)
        color = self._get_color(record.levelno)
        return click.style(text, color)


class MakeFileHandler(logging.FileHandler):

    def __init__(self, filename: str, encoding="utf-8"):
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        version = time.strftime("%Y-%m-%d_%H")

        versioned_filename = filepath.parent / (
            filepath.stem + f"_{version}" + filepath.suffix
        )
        logging.FileHandler.__init__(
            self, versioned_filename, mode="a", encoding=encoding, delay=False
        )


@contextmanager
def all_loggingLdisabled(highest_level=logging.CRITICAL):
    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)
