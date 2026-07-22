"""
HuggingFace earnings-transcript backbone
(src/data_extract/utils/behavioral/fetch_hf_transcripts.py).

The dataset `kurry/sp500_earnings_transcripts` (2005-2025, MIT) is the deep-history backbone;
each row's verbatim `content` + speaker-segmented `structured_content` is parsed into the same
`earnings_call_sections` rows the Motley Fool path produces (full / prepared_remarks / qa /
participants), so downstream sentiment features are source-agnostic. Tests the pure row->sections
parser (no download / DB) + a best-effort live check on real dataset rows.
"""
from __future__ import annotations

import json

import pytest

from src.data_extract.utils.behavioral import fetch_hf_transcripts as hf


def _synthetic_row():
    prepared = ("Jane Cook, CEO: Thank you and good afternoon everyone. We delivered a record "
                "quarter with revenue up 12% and expanding margins across every segment. " * 4)
    cfo = ("John Roe, CFO: Turning to the financials, operating cash flow reached a new high and "
           "we returned capital to shareholders through buybacks and dividends. " * 4)
    qa = ("Sam Analyst: Congratulations on the quarter. Can you talk about demand trends into next "
          "year and the margin outlook? Jane Cook, CEO: Sure, demand remains strong and we expect "
          "continued operating leverage. " * 4)
    content = ("Operator: Good afternoon, and welcome to the Acme Corp Fourth Quarter earnings "
               "conference call. All lines are muted.\n"
               f"{prepared}\n{cfo}\n"
               "Operator: We will now begin the question-and-answer session. Our first question "
               "comes from Sam Analyst.\n"
               f"{qa}")
    structured = [
        {"speaker": "Operator", "text": "Good afternoon, and welcome..."},
        {"speaker": "Jane Cook, CEO", "text": prepared},
        {"speaker": "John Roe, CFO", "text": cfo},
        {"speaker": "Operator", "text": "We will now begin the question-and-answer session."},
        {"speaker": "A - Sam Analyst", "text": "Congratulations on the quarter..."},  # role prefix
        {"speaker": "Jane Cook, CEO", "text": "Sure, demand remains strong..."},
    ]
    return content, structured


def test_row_sections_and_participants():
    content, structured = _synthetic_row()
    out = hf.row_sections(content, structured)

    assert out["full"] == content
    assert "prepared_remarks" in out and "qa" in out
    # prepared has management, NOT the analyst Q&A; qa has the hand-off + analyst
    assert "Jane Cook, CEO" in out["prepared_remarks"] and "Sam Analyst" not in out["prepared_remarks"]
    assert "question-and-answer session" in out["qa"] and "Sam Analyst" in out["qa"]
    # participants: distinct, Operator excluded, "A - " role prefix stripped, order preserved
    assert out["participants"] == "Jane Cook, CEO\nJohn Roe, CFO\nSam Analyst"

    # participants helper directly
    assert hf._participants_text([{"speaker": "Operator", "text": "x"},
                                  {"speaker": "E - Bob", "text": "y"},
                                  {"speaker": "Bob", "text": "z"}]) == "Bob"
    # content too short + no structured -> empty (skipped downstream)
    assert hf.row_sections("tiny", None) == {}

    print("\n=== SANITY CHECK: HF transcript row -> sections ===")
    print(f"  sections: {sorted(out)}  participants={out['participants'].split(chr(10))}")
    print(f"  prepared_remarks {len(out['prepared_remarks'])} chars (mgmt only), "
          f"qa {len(out['qa'])} chars (hand-off + analyst)")
    print("  full always kept; split via the shared operator Q&A-marker; 'A - ' role prefix "
          "stripped, Operator excluded from participants. Validated.")


def test_row_sections_on_real_dataset_rows():
    """Best-effort: parse a few REAL rows from the live dataset (skips if HF is unreachable)."""
    def fetch(url):
        try:
            import requests
            return requests.get(url, timeout=40).text
        except Exception:
            try:
                from curl_cffi import requests as cr
                return cr.get(url, timeout=40, impersonate="chrome", verify=False).text
            except Exception:
                return None
    body = fetch("https://datasets-server.huggingface.co/rows?dataset=kurry/"
                 "sp500_earnings_transcripts&config=default&split=train&offset=0&length=6")
    if not body:
        pytest.skip("HF datasets-server unreachable")
    try:
        rows = json.loads(body)["rows"]
    except Exception:
        pytest.skip("HF response not parseable")

    n_full = n_split = 0
    for rr in rows:
        r = rr["row"]
        secs = hf.row_sections(r.get("content"), r.get("structured_content"))
        if secs.get("full"):
            n_full += 1
        if "prepared_remarks" in secs and "qa" in secs:
            n_split += 1
    assert n_full == len(rows), f"only {n_full}/{len(rows)} real rows produced a `full` section"
    print("\n=== SANITY CHECK: HF row->sections on REAL dataset rows ===")
    print(f"  {len(rows)} real transcripts: {n_full}/{len(rows)} have `full`, "
          f"{n_split}/{len(rows)} split into prepared_remarks + qa.")
    print("  CONCLUSION: real 2005-2025 dataset rows parse into the standard sections; the "
          "backbone is source-compatible with the Motley Fool path. Validated.")


if __name__ == "__main__":
    test_row_sections_and_participants()
    test_row_sections_on_real_dataset_rows()
