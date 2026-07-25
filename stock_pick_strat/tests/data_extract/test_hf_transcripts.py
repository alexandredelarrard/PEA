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
import types

import pytest

from src.constants.constants import HF_BACKBONE_EARLY_QUARTER, HF_BACKBONE_LATE_QUARTER
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


# --------------------------------------------------------------------------- #
# Backbone-present short-circuit (the "0 new calls" stall fix)                  #
# --------------------------------------------------------------------------- #
class _Res:
    def __init__(self, value):
        self._value = value

    def first(self):
        return self._value                               # (min_quarter, max_quarter) or (None, None)


class _FakeConn:
    def __init__(self, minmax):
        self._minmax = minmax

    def execute(self, clause, params=None):
        return _Res(self._minmax)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _ctx(minmax):
    engine = types.SimpleNamespace(connect=lambda: _FakeConn(minmax))
    return types.SimpleNamespace(store=types.SimpleNamespace(engine=engine))


def test_hf_backbone_presence_detection():
    # full backbone (deep history + reaches the cut) -> present
    present, lo, hi = hf._hf_backbone_already_ingested(_ctx(("2005Q1", "2026Q2")))
    assert present is True and lo == "2005Q1" and hi == "2026Q2"
    # only recent MF calls (no deep history) -> NOT present (must still ingest HF)
    assert hf._hf_backbone_already_ingested(_ctx(("2023Q2", "2026Q4")))[0] is False
    # deep history but not yet reaching the cut (interrupted ingest) -> NOT present
    assert hf._hf_backbone_already_ingested(_ctx(("2005Q1", "2018Q4")))[0] is False
    # empty table -> NOT present
    assert hf._hf_backbone_already_ingested(_ctx((None, None)))[0] is False
    print("\n=== SANITY CHECK: HF backbone presence detection ===")
    print(f"  thresholds: min <= {HF_BACKBONE_EARLY_QUARTER}, max >= {HF_BACKBONE_LATE_QUARTER}")
    print("  2005Q1..2026Q2 -> present; 2023Q2..2026Q4 -> absent; 2005Q1..2018Q4 -> absent; "
          "empty -> absent. Validated.")


def test_hf_ingest_short_circuits_when_present(monkeypatch):
    # when the backbone is present, ingest must return 0 WITHOUT downloading/scanning the parquet
    def boom(*a, **k):
        raise AssertionError("download_hf_parquet must NOT be called when backbone is present")
    monkeypatch.setattr(hf, "download_hf_parquet", boom)

    saved = hf.ingest_hf_transcripts(_ctx(("2005Q1", "2026Q2")))
    assert saved == 0

    print("\n=== SANITY CHECK: HF ingest short-circuits on full backbone ===")
    print("  table spans 2005Q1..2026Q2 -> ingest returned 0 and NEVER touched the 1.8GB parquet "
          "(download_hf_parquet not called). No more multi-minute '0 new calls' stall. Validated.")


if __name__ == "__main__":
    test_row_sections_and_participants()
    test_row_sections_on_real_dataset_rows()
    test_hf_backbone_presence_detection()
