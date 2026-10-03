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
        "TXN_2025Q1",
        "BMY_2026Q1",
        "JPM_2024Q3",
    }
    assert required <= names, sorted(required - names)
    assert len(names) >= 19
    print(f"SANITY fixtures: {len(names)} hand-labelled calls present, all {len(required)} named hard/follow-up/Q&A-only-executive cases included")


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
        if first["speaker"] not in truth["management"]:  # qa holds management answers only
            assert first["starts_with"] not in split.qa
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


# ---- text cleaning (known truth) --------------------------------------------------------- #
DECIMALS = "Revenue grew 3.5% in the U.S. to $1.2 billion."
PREPARED_CEO = (
    "Good morning, everyone, and thank you for joining us. "
    "Joining me today are Jane Doe, our Chief Financial Officer, and Bob Lee, our COO. "
    "Before we begin, please note that today's remarks contain forward-looking statements and actual results may differ materially. "
    "A reconciliation of these non-GAAP measures is included in our press release, which is available on our website. "
    f"{DECIMALS} We expect revenue growth of 5% next year, driven by pricing and new customers in every region. "
    "Before I turn it over to Jane, I want to note that our backlog reached a record $4.2 billion this quarter. "
    "With that, I'll turn it over to Jane Doe, our CFO."
)
PREPARED_CFO = "Thanks, John. Gross margin expanded 120 basis points to 41.3%, the highest level in the company's history."
QUESTION = (
    "Hi, this is Mary Analyst from Big Bank. Thanks for taking my question and congrats on the quarter. "
    "How should we think about the margin trajectory into the second half of next year?"
)
ANSWER = (
    "Thanks, Mary. Yes, Mary, we expect margins to keep expanding as pricing flows through the order book. "
    "Thank you. The forward-looking statements in our filings with the SEC describe the risks."
)


def _call(qa_turns: list[tuple[str, str]]) -> list[dict]:
    rows = [
        ("Operator", "Good day and welcome to the earnings conference call. I will now turn the call over to management."),
        ("John Smith", PREPARED_CEO),
        ("Jane Doe", PREPARED_CFO),
        *qa_turns,
    ]
    return [{"paragraph_number": i, "speaker": s, "content": c} for i, (s, c) in enumerate(rows, start=1)]


def _default_call() -> list[dict]:
    return _call(
        [
            ("Operator", "Our first question comes from the line of Mary Analyst with Big Bank. Please proceed."),
            ("Mary Analyst", QUESTION),
            ("John Smith", ANSWER),
        ]
    )


def test_sentence_split_never_alters_decimals_or_abbreviations() -> None:
    from src.utils.earnings_call_split import split_sentences
    from src.utils.text_metrics import clean_earnings_call_text

    assert " ".join(split_sentences(DECIMALS)) == DECIMALS
    text = f"{DECIMALS} Thank you. EPS was $2.10, up 3.4% year-over-year in the U.S. and the U.K. markets."
    assert " ".join(split_sentences(text)) == text
    assert clean_earnings_call_text(DECIMALS) == DECIMALS
    split = split_call(_default_call())
    assert DECIMALS in split.prepared_remarks
    assert any(DECIMALS in t["text"] for t in split.turns if t["tag"] == EARNINGS_CALL_TAG_PREPARED)
    print("SANITY splitter: '3.5%', 'U.S.' and '$1.2' survive byte-identical through every cleaner")


def test_prepared_remarks_drop_boilerplate_roster_courtesy_and_handoffs_but_keep_business() -> None:
    prep = split_call(_default_call()).prepared_remarks
    for gone in ("thank you for joining", "Joining me today", "forward-looking", "non-GAAP", "turn it over to Jane Doe", "Thanks, John"):
        assert gone not in prep, gone
    assert "We expect revenue growth of 5% next year, driven by pricing and new customers in every region." in prep
    assert "Gross margin expanded 120 basis points to 41.3%" in prep
    kept = next(s for s in prep.split(". ") if "backlog" in s)
    assert "Jane" not in kept and "record $4.2 billion" in kept, kept
    print(f"SANITY prepared: boilerplate/roster/courtesy/hand-off removed, forward guidance and figures kept -> {prep[:90]!r}...")


def test_qa_text_is_management_answers_only_and_turns_are_cleaned() -> None:
    split = split_call(_default_call())
    assert "Big Bank" not in split.qa and "margin trajectory" not in split.qa and "first question" not in split.qa
    assert split.qa.startswith("Yes, we expect margins to keep expanding"), split.qa
    assert "forward-looking statements in our filings" in split.qa, "boilerplate is removed from prepared remarks only"
    assert "Thank you." not in split.qa, "a courtesy sentence in the middle of an answer is removed"
    q = next(t for t in split.turns if t["tag"] == EARNINGS_CALL_TAG_QUESTION)
    a = next(t for t in split.turns if t["tag"] == EARNINGS_CALL_TAG_ANSWER)
    assert q["text"] == "How should we think about the margin trajectory into the second half of next year?", q["text"]
    assert a["text"].startswith("Yes, we expect margins") and "Mary" not in a["text"], a["text"]
    assert (q["exchange_idx"], a["exchange_idx"], a["answer_idx"]) == (0, 0, 1)
    print("SANITY qa: only the management answer reaches sentiment; self-intro, courtesy and addressed names removed from turns")


def test_handoffs_are_dropped_but_requests_and_hedges_are_kept() -> None:
    from src.utils.earnings_call_split import CallNames, clean_turn_text

    names = CallNames(frozenset({"Bill", "Smith", "Jane", "Doe"}), ("Bill Smith", "Jane Doe"))
    assert clean_turn_text("With that, I'll turn it over to Jane Doe, our CFO, for the forecast.", names) == ""
    assert clean_turn_text("So can you provide more detail on that?", names) == "So can you provide more detail on that?"
    got = clean_turn_text("So Bill, if you could give us the contributions from the acquisitions.", names)
    assert got == "So, if you could give us the contributions from the acquisitions.", got
    got = clean_turn_text("I'll let Jane talk about the margin, but you know, it's kind of early.", names)
    assert got == "I'll let a colleague talk about the margin, but you know, it's kind of early.", got
    assert (
        clean_turn_text("Lastly, let's turn to the balance sheet and cash flow.", names) == "Lastly, let's turn to the balance sheet and cash flow."
    )
    print("SANITY hand-offs: pure hand-off dropped, analyst requests and topic transitions kept, hedges kept, names -> 'a colleague'")


def test_qa_turns_need_ten_words_and_exchanges_stay_contiguous() -> None:
    long_q = "Can you walk us through the drivers of the bookings growth in the second half?"
    long_a = "Bookings grew on the back of strong renewals and new logos in every single segment we serve."
    paras = _call(
        [
            ("Operator", "Our first question comes from the line of Mary Analyst with Big Bank. Please proceed."),
            ("Mary Analyst", long_q),
            ("John Smith", long_a),
            ("John Smith", "Okay. That is right, yes."),
            ("Operator", "Our next question comes from the line of Pete Short with Small Bank. Please proceed."),
            ("Pete Short", "What about margins next year?"),
            ("John Smith", "Margins should expand modestly as the new products scale through the year and costs fall."),
            ("Operator", "Our next question comes from the line of Ann Long with Mid Bank. Please proceed."),
            ("Ann Long", long_q.replace("bookings", "revenue")),
            ("Jane Doe", long_a.replace("Bookings", "Revenue")),
        ]
    )
    split = split_call(paras)
    qa = [(t["person"], t["tag"], t["exchange_idx"], t["answer_idx"]) for t in split.turns if t["section"] == "qa"]
    assert qa == [
        ("Mary Analyst", "question", 0, 0),
        ("John Smith", "answer", 0, 1),
        ("Ann Long", "question", 1, 0),
        ("Jane Doe", "answer", 1, 1),
    ], qa
    assert all(len(t["text"].split()) >= 10 for t in split.turns if t["section"] == "qa")
    assert "Margins should expand modestly" in split.qa, "an orphaned management answer still feeds the sentiment text"
    print("SANITY 10-word minimum: the 5-word question opens no exchange, its answer is not attached, indices stay 0..1")
