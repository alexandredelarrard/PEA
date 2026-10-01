"""Shared SEC HTML-to-text conversion for filing prose and visible table cells."""

from __future__ import annotations

import html
import re


def html_to_text(raw: str) -> str:
    """Strip EDGAR HTML/TXT markup while keeping visible text and table cells."""
    if not raw:
        return ""
    raw = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>", "\n", raw)
    raw = re.sub(r"(?i)</(p|div|tr|td|th|table|li|h[1-6])>", " ", raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(raw)
    text = text.replace("\xa0", " ")
    text = text.replace("​", "").replace("﻿", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()
