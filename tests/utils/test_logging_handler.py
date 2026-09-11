"""A log line carrying `⚠` must survive a cp1252 console, because the alarms use that glyph.

The defect this pins is not cosmetic. `logging.StreamHandler.emit` writes the whole formatted
record in one call, so an unencodable character raises inside `emit` and the handler drops the
ENTIRE line -- message, level, timestamp -- leaving only a bare `--- Logging error ---`
traceback that never names which message died. The 2026-09-09 `cube_part_governance` rebuild
lost its `⚠ control_wedge < 0 (voting/ownership legs swapped): 4` tally exactly this way.
"""
from __future__ import annotations

import io
import logging

import pytest

from src.utils.logging_handler import ColorHandler

WARN = "⚠"
MESSAGE = f"{WARN} control_wedge < 0 (voting/ownership legs swapped): 4"


def cp1252_stream() -> io.TextIOWrapper:
    """A text stream with the encoding a Windows console actually has."""
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="")


def emit_through(handler: logging.Handler, msg: str) -> None:
    record = logging.LogRecord("t", logging.INFO, __file__, 1, msg, None, None)
    handler.handle(record)
    handler.flush()


def read_back(stream: io.TextIOWrapper) -> str:
    stream.flush()
    return stream.buffer.getvalue().decode(stream.encoding, errors="replace")


def test_a_plain_streamhandler_on_cp1252_loses_the_whole_line() -> None:
    """The control: this is the behaviour ColorHandler must NOT have."""
    stream = cp1252_stream()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    with pytest.raises(UnicodeEncodeError):
        # `handle` swallows the error via handleError, so go straight at emit's write path.
        stream.write(handler.format(logging.LogRecord(
            "t", logging.INFO, __file__, 1, MESSAGE, None, None)))
    assert "control_wedge" not in read_back(stream), (
        "the control is only meaningful if the message really is lost")


def test_colorhandler_keeps_the_message_when_the_glyph_cannot_be_encoded() -> None:
    stream = cp1252_stream()
    handler = ColorHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    emit_through(handler, MESSAGE)

    out = read_back(stream)
    # The glyph itself may or may not survive the terminal; the ALARM must.
    assert "control_wedge < 0 (voting/ownership legs swapped): 4" in out
    assert "--- Logging error ---" not in out


def test_an_ascii_message_is_unchanged() -> None:
    stream = cp1252_stream()
    handler = ColorHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    emit_through(handler, "governance tally | rows: say_on_pay: 6,788")
    assert "say_on_pay: 6,788" in read_back(stream)


def test_a_stream_that_cannot_be_reconfigured_is_still_accepted() -> None:
    """`reconfigure` is absent on StringIO and on many test doubles -- do not require it."""
    stream = io.StringIO()
    handler = ColorHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    emit_through(handler, MESSAGE)
    assert "control_wedge" in stream.getvalue()
