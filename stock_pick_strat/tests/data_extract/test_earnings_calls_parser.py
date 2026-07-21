"""
Pure parsing of the earnings-call fetcher (src/data_extract/utils/behavioral/
fetch_earnings_calls.py): transcript URL-slug -> (ticker, quarter, date) and the
HTML -> high-signal sections (prepared_remarks / qa / full) split. No network.
"""
from __future__ import annotations

from src.data_extract.utils.behavioral.fetch_earnings_calls import (
    _parse_link,
    _universe_slug_map,
    parse_transcript_sections,
)

_PREP = ("Good morning everyone and welcome to the call. Revenue grew and our cloud "
         "segment expanded nicely across every major region this quarter. " * 4)
_QA = ("Analyst: How is demand trending into next quarter and what are you seeing on "
       "pricing? CEO: Demand is strong and pricing held up well across the portfolio. " * 4)

_HTML = f"""<html><body>
<div class="transcript-content">
CALL PARTICIPANTS
Jane Doe -- Chief Executive Officer
John Smith -- Chief Financial Officer
Operator
{_PREP}
Questions and Answers
{_QA}
</div></body></html>"""


def test_parse_link_resolves_multitoken_ticker():
    slug_map = _universe_slug_map(["AAPL", "BRK-B"])
    rec = _parse_link("/earnings/call-transcripts/2024/02/15/"
                      "berkshire-hathaway-brk-b-q4-2023-earnings-call-transcript", slug_map)
    assert rec == {"ticker": "BRK-B", "quarter": "2023Q4", "call_date": "2024-02-15",
                   "url": "https://www.fool.com/earnings/call-transcripts/2024/02/15/"
                          "berkshire-hathaway-brk-b-q4-2023-earnings-call-transcript/"}
    aapl = _parse_link("/earnings/call-transcripts/2024/05/02/"
                       "apple-aapl-q2-2024-earnings-call-transcript", slug_map)
    assert aapl["ticker"] == "AAPL" and aapl["quarter"] == "2024Q2"
    # a ticker not in the universe -> None (filtered out of the index)
    assert _parse_link("/earnings/call-transcripts/2024/05/02/"
                       "someco-zzzz-q2-2024-earnings-call-transcript", slug_map) is None


def test_parse_sections_splits_prepared_and_qa():
    sec = parse_transcript_sections(_HTML)
    assert "full" in sec                                    # always kept (format-proof)
    assert "prepared_remarks" in sec and "qa" in sec
    assert "Good morning" in sec["prepared_remarks"]
    assert "How is demand" in sec["qa"]
    # the Q&A hand-off marker starts the qa section, so it must NOT leak into prepared
    assert "Questions and Answers" not in sec["prepared_remarks"]
    # too-short / no-content HTML -> empty dict
    assert parse_transcript_sections("<html><body><p>hi</p></body></html>") == {}

    print("\n=== SANITY CHECK: earnings-call parser ===")
    print("  URL slug -> (BRK-B, 2023Q4, 2024-02-15) with multi-token ticker resolution & "
          "universe filtering; transcript HTML splits into prepared_remarks vs qa at the "
          "operator's Q&A hand-off, always keeping 'full'. Validated.")


if __name__ == "__main__":
    test_parse_link_resolves_multitoken_ticker()
    test_parse_sections_splits_prepared_and_qa()
    print("\n=== SANITY CHECK: earnings-call parser ===")
    print("  URL slug -> (BRK-B, 2023Q4, 2024-02-15) with multi-token ticker resolution & "
          "universe filtering; transcript HTML splits into prepared_remarks vs qa at the "
          "operator's Q&A hand-off, always keeping 'full'. Validated.")
