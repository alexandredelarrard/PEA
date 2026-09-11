import logging
import sys
from logging import LogRecord
import click
import time
from pathlib import Path
from contextlib import contextmanager


class ColorHandler(logging.StreamHandler):

    def __init__(self, stream=None, colors=None, **kwargs):
        # ⚠ A Windows console is cp1252, and `StreamHandler.emit` writes the formatted record
        # straight to it -- so ONE unencodable character raises UnicodeEncodeError inside emit
        # and the WHOLE line is dropped, not just the glyph. This codebase writes `⚠` into
        # tallies and warnings, i.e. exactly the lines that must not go missing: the
        # 2026-09-09 `cube_part_governance` rebuild silently lost its
        # `⚠ control_wedge < 0 (voting/ownership legs swapped): 4` alarm this way, leaving a
        # bare traceback with no indication of which message had died.
        #
        # `MakeFileHandler` below already defaults to utf-8, so only the console leg was
        # affected. Reconfiguring with errors="replace" costs one character on a terminal that
        # cannot render the glyph, instead of the message.
        stream = stream if stream is not None else sys.stderr
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError, AttributeError):
                pass
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
