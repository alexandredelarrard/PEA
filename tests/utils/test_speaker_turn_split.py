"""Speaker-turn split of earnings-call transcripts (src/utils/earnings_call_split.py).

Fixtures are real defeatbeta calls (tests/fixtures/earnings_calls/*.json) with hand-verified truth:
the first Q&A turn, the split status, which speakers are analysts / management, and the expected
tag of a few named turns. Long paragraphs are truncated in the fixtures (noted per file).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from src.constants.constants import EARNINGS_CALL_TAG_ANSWER, EARNINGS_CALL_TAG_PREPARED, EARNINGS_CALL_TAG_QUESTION
from src.utils.earnings_call_split import CallSplit, Turn, clean_paragraphs, label_turns, qa_start, split_call
from src.utils.text_metrics import assess_earnings_call_sections

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "earnings_calls"
FIXTURES = sorted(FIXTURE_DIR.glob("*.json"))
TURN_KEYS = {"section", "tag", "person", "text", "exchange_idx", "answer_idx"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_fixture_set_is_complete() -> None:
    names = {p.stem for p in FIXTURES}
    required = {
        "PLTR_2024Q4",
        "PNW_2024Q4",
        "AXON_2026Q2",
        "JNJ_2026Q2",
        "NOW_2026Q2",
        "UDR_2026Q2",
        "WST_2026Q2",
        "WELL_2025Q4",
        "ADP_2026Q3",
        "PSKY_2025Q2",
        "COIN_2026Q2",
        "IBM_2024Q1",
        "AXON_2025Q4",
        "CRH_2025Q2",
    }
    assert required <= names, sorted(required - names)
    assert len(names) >= 16
    print(f"SANITY fixtures: {len(names)} hand-labelled calls present, all 14 named hard/follow-up cases included")


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_qa_start_and_status_match_hand_labels(path: Path) -> None:
    fx = _load(path)
    truth = fx["truth"]
    turns = clean_paragraphs(fx["paragraphs"])
    k = qa_start(turns)
    split = split_call(fx["paragraphs"])
    assert split.status == truth["status"]
    first = truth["first_qa_turn"]
    if first is None:
        assert k is None and split.qa == ""
        got = "none"
    else:
        assert k is not None, "no Q&A start found"
        assert turns[k].speaker == first["speaker"], turns[k]
        assert turns[k].text.startswith(first["starts_with"]), turns[k].text[:120]
        assert split.qa.startswith(first["starts_with"])
        got = f"turn {k} {turns[k].speaker!r}"
    print(f"SANITY {path.stem}: status={split.status} Q&A start={got} matches the hand label")


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_roles_match_hand_labels(path: Path) -> None:
    fx = _load(path)
    truth = fx["truth"]
    turns = split_call(fx["paragraphs"]).turns
    analysts, mgmt = set(truth["analysts"]), set(truth["management"])
    wrong_analyst = [(t["person"], t["tag"], t["text"][:60]) for t in turns if t["person"] in analysts and t["tag"] != EARNINGS_CALL_TAG_QUESTION]
    wrong_mgmt = [(t["person"], t["text"][:60]) for t in turns if t["person"] in mgmt and t["tag"] == EARNINGS_CALL_TAG_QUESTION]
    assert not wrong_analyst, wrong_analyst
    assert not wrong_mgmt, wrong_mgmt
    for want in truth["tags"]:
        hits = [t["tag"] for t in turns if t["person"] == want["person"] and want["contains"] in t["text"]]
        assert hits == [want["tag"]], (want, hits)
    answers = Counter(t["exchange_idx"] for t in turns if t["tag"] == EARNINGS_CALL_TAG_ANSWER)
    if "max_answers_per_question" in truth:
        assert max(answers.values()) <= truth["max_answers_per_question"], answers.most_common(3)
    n_q = sum(t["tag"] == EARNINGS_CALL_TAG_QUESTION for t in turns)
    print(
        f"SANITY {path.stem}: {n_q} questions, all analyst turns are questions, no management question, "
        f"{len(truth['tags'])} named tags right, max answers per question {max(answers.values(), default=0)}"
    )


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_turns_follow_the_embedding_contract(path: Path) -> None:
    split = split_call(_load(path)["paragraphs"])
    assert isinstance(split, CallSplit)
    last_answer: dict[int, int] = {}
    for t in split.turns:
        assert set(t) == TURN_KEYS
        assert t["text"] and t["person"] is not None
        if t["tag"] == EARNINGS_CALL_TAG_PREPARED:
            assert (t["section"], t["exchange_idx"], t["answer_idx"]) == ("prepared_remarks", -1, -1)
        elif t["tag"] == EARNINGS_CALL_TAG_QUESTION:
            assert t["section"] == "qa" and t["exchange_idx"] >= 0 and t["answer_idx"] == 0
        else:
            assert t["tag"] == EARNINGS_CALL_TAG_ANSWER and t["section"] == "qa"
            assert t["answer_idx"] == last_answer.get(t["exchange_idx"], 0) + 1
            last_answer[t["exchange_idx"]] = t["answer_idx"]
    sections = [t["section"] for t in split.turns]
    assert sections == sorted(sections, key=lambda s: s == "qa"), "prepared turns come first"
    print(f"SANITY {path.stem}: {len(split.turns)} turns carry exactly {sorted(TURN_KEYS)} with consistent indices")


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_section_texts_are_header_free_and_pass_the_quality_gate(path: Path) -> None:
    fx = _load(path)
    split = split_call(fx["paragraphs"])
    speakers = {p["speaker"] for p in fx["paragraphs"]}
    lines = (split.prepared_remarks + "\n" + split.qa).splitlines()
    assert not [ln for ln in lines if any(ln.startswith(f"{s}:") for s in speakers)]
    quality = assess_earnings_call_sections({"prepared_remarks": split.prepared_remarks, "qa": split.qa})
    assert quality.valid == (split.status == "ok"), quality.reason
    print(f"SANITY {path.stem}: header-free texts, quality gate valid={quality.valid} agrees with status={split.status}")


def test_clean_paragraphs_known_truth() -> None:
    paras = [
        {"paragraph_number": 3, "speaker": "A - Jane Roe", "content": "Second paragraph of Jane."},
        {"paragraph_number": 1, "speaker": "AI Insights", "content": ""},
        {"paragraph_number": 2, "speaker": "\u200bJane  Roe", "content": "First paragraph of Jane."},
        {"paragraph_number": 4, "speaker": "Speaker 1", "content": "[Presentation]"},
        {"paragraph_number": 5, "speaker": "Speaker 2", "content": "Short placeholder line."},
        {"paragraph_number": 6, "speaker": "Speaker 0", "content": " ".join(["disclaimer"] * 20)},
        {"paragraph_number": 7, "speaker": "John Doe", "content": "  "},
    ]
    turns = clean_paragraphs(paras)
    assert turns == [Turn("Jane Roe", "First paragraph of Jane.\nSecond paragraph of Jane."), Turn("Speaker 0", " ".join(["disclaimer"] * 20))]
    print("SANITY clean_paragraphs: order restored, placeholders/empties dropped, role prefix and zero-width marks stripped, same speaker merged")


def test_split_call_terminal_statuses() -> None:
    assert split_call([]) == CallSplit("", "", [], "empty")
    assert split_call([{"speaker": "AI Insights", "content": ""}]).status == "empty"
    straight_to_qa = [
        {"speaker": "Operator", "content": "Our first question comes from Jane Roe with Big Bank."},
        {"speaker": "Jane Roe", "content": "How do you see demand developing in the second half of the year?"},
        {"speaker": "John Doe", "content": "Demand remains robust across every region and we expect it to stay that way."},
    ]
    split = split_call(straight_to_qa)
    assert split.status == "no_prepared" and split.prepared_remarks == ""
    assert [t["tag"] for t in split.turns] == [EARNINGS_CALL_TAG_QUESTION, EARNINGS_CALL_TAG_ANSWER]
    print("SANITY statuses: empty input -> empty, Q&A from the first turn -> no_prepared, never an exception")


def test_label_turns_role_fixes_on_minimal_exchange() -> None:
    qa = [
        Turn("Host Person", "Thanks. Up next, we have Will Power at Baird."),
        Turn("William Power", "Can you walk us through the drivers of the bookings growth this year?"),
        Turn("Chief Executive", "Bookings grew on the back of strong renewals and new logos in every segment."),
        Turn("William Power", "And as a follow-up, how should we think about margins for next year?"),
        Turn("Chief Executive", "Margins should expand modestly as the new products scale through the year."),
    ]
    turns = label_turns(qa, "qa", {"host person", "chief executive"})
    assert [(t["person"], t["tag"], t["exchange_idx"], t["answer_idx"]) for t in turns] == [
        ("William Power", "question", 0, 0),
        ("Chief Executive", "answer", 0, 1),
        ("William Power", "question", 1, 0),
        ("Chief Executive", "answer", 1, 1),
    ]
    print("SANITY label_turns: 'Up next, we have X' is a dropped hand-off, the analyst's follow-up stays a question")


def test_fixture_suite_conclusion() -> None:
    exact, analyst_turns, analyst_q, mgmt_q = 0, 0, 0, 0
    for path in FIXTURES:
        fx = _load(path)
        truth, turns = fx["truth"], clean_paragraphs(fx["paragraphs"])
        k = qa_start(turns)
        first = truth["first_qa_turn"]
        exact += (k is None) if first is None else (k is not None and turns[k].text.startswith(first["starts_with"]))
        labelled = split_call(fx["paragraphs"]).turns
        analyst_turns += sum(t["person"] in truth["analysts"] for t in labelled)
        analyst_q += sum(t["person"] in truth["analysts"] and t["tag"] == EARNINGS_CALL_TAG_QUESTION for t in labelled)
        mgmt_q += sum(t["person"] in truth["management"] and t["tag"] == EARNINGS_CALL_TAG_QUESTION for t in labelled)
    assert exact == len(FIXTURES) and analyst_q == analyst_turns and mgmt_q == 0
    print(
        f"SANITY CONCLUSION: exact Q&A start and status on {exact}/{len(FIXTURES)} hand-labelled calls; "
        f"{analyst_q}/{analyst_turns} analyst turns tagged question; {mgmt_q} management turns tagged question"
    )
