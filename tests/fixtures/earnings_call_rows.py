"""`earnings_call_sections` paragraph rows for tests.

Two sources:
  * `fixture_paragraphs` -- a hand-labelled real defeatbeta call from `earnings_calls/*.json`;
  * `synthetic_call` -- a minimal five-paragraph call (operator welcome, executive prepared
    remarks, operator hand-off naming the analyst, the analyst's question, the executive's
    answer) that `split_call` cuts with status `ok`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FIXTURE_DIR = Path(__file__).resolve().parent / "earnings_calls"
FIXTURE_PATHS = sorted(FIXTURE_DIR.glob("*.json"))
PARAGRAPH_COLUMNS = ["ticker", "quarter", "paragraph", "as_of", "transcript_id", "speaker", "content"]


def load_fixture(stem: str) -> dict:
    """The raw JSON of one hand-labelled call (truth + paragraphs)."""
    return json.loads((FIXTURE_DIR / f"{stem}.json").read_text(encoding="utf-8"))


def fixture_paragraphs(stem: str) -> pd.DataFrame:
    """Table rows of one hand-labelled call, keyed as the extractor writes them."""
    fx = load_fixture(stem)
    quarter = f"{fx['fiscal_year']}Q{fx['fiscal_quarter']}"
    return pd.DataFrame(
        [
            {
                "ticker": fx["symbol"],
                "quarter": quarter,
                "paragraph": int(p["paragraph_number"]),
                "as_of": fx["report_date"],
                "transcript_id": fx.get("transcripts_id"),
                "speaker": p["speaker"],
                "content": p["content"],
            }
            for p in fx["paragraphs"]
        ],
        columns=PARAGRAPH_COLUMNS,
    )


def synthetic_call(
    ticker: str,
    quarter: str,
    as_of: str | None,
    *,
    prepared: str,
    question: str,
    answer: str,
    analyst: str = "Jane Doe",
    executive: str = "John Smith",
    transcript_id: int = 1,
) -> pd.DataFrame:
    """One call as five paragraph rows; the split puts `prepared` in prepared_remarks and the
    hand-off, `question` and `answer` in qa."""
    paragraphs = [
        ("Operator", "Good day and welcome to the earnings conference call. I will now turn the call over to management."),
        (executive, prepared),
        ("Operator", f"Our first question comes from the line of {analyst} with Big Bank. Please proceed."),
        (analyst, question),
        (executive, answer),
    ]
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "quarter": quarter,
                "paragraph": number,
                "as_of": as_of,
                "transcript_id": transcript_id,
                "speaker": speaker,
                "content": content,
            }
            for number, (speaker, content) in enumerate(paragraphs, start=1)
        ],
        columns=PARAGRAPH_COLUMNS,
    )
