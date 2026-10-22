"""Known-truth transcript metadata and parenthesis cleaning."""

from __future__ import annotations

import pytest

from src.utils.text import text_metrics


def test_rosters_are_removed_without_discarding_financial_openings() -> None:
    records = [
        {"paragraph": 1, "speaker": "Transcript Operator", "content": "Transcript supplied by the provider."},
        {"paragraph": 2, "speaker": "Executives", "content": "Jane Doe CFO. John Smith CEO."},
        {"paragraph": 3, "speaker": "Analysts", "content": "Alex Roe Big Bank."},
        {"paragraph": 4, "speaker": "Operator", "content": "Our first question comes from Alex Roe."},
        {"paragraph": 5, "speaker": "Jane Doe", "content": "Revenue grew 12% to $4.37 billion."},
        {"paragraph": 6, "speaker": "Company Representatives", "content": "Jane Doe CFO."},
    ]
    clean, removed = text_metrics.clean_earnings_call_paragraphs(records)
    assert clean == records[3:5]
    assert removed == {"metadata paragraphs": 4}
    first = {**records[4], "paragraph": 1}
    assert text_metrics.clean_earnings_call_paragraphs([first])[0] == [first]
    assert records[0]["content"] == "Transcript supplied by the provider."
    print("SANITY: rosters at any position disappear; financial first paragraphs and structural Operator handoffs survive.")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Revenue (inaudible) grew 12%.", "Revenue grew 12%."),
        ("Revenue(edited (by transcriber))grew 12%.", "Revenue grew 12%."),
        ("Revenue grew (or 12% in dollars) this quarter.", "Revenue grew this quarter."),
        ("(Operator Instructions) Revenue grew (sic).", "Revenue grew."),
        ("Revenue rose 3.5% in the U.S. to $1.2 billion.", "Revenue rose 3.5% in the U.S. to $1.2 billion."),
        ("Revenue (unclosed financial speech", "Revenue (unclosed financial speech"),
        ("Revenue ) rose 12%.", "Revenue ) rose 12%."),
        ("(Technical issue)", ""),
    ],
)
def test_parenthetical_cleaning(raw: str, expected: str) -> None:
    clean, removed = text_metrics.clean_earnings_call_paragraphs([{"paragraph": 1, "speaker": "Jane Doe", "content": raw}])
    assert (clean[0]["content"] if clean else "") == expected
    assert bool(removed.get("parenthetical paragraphs")) == (expected != raw)
    print("SANITY: balanced parenthetical text is removed, nested notes included; decimals and unmatched financial text survive.")
