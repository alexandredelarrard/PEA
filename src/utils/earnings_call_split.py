"""
earnings_call_split.py  (src/utils/earnings_call_split.py)
----------------------------------------------------------
Pure speaker-turn split of one earnings-call transcript, given as source paragraphs
(`speaker`, `content`, optional `paragraph_number` / `paragraph`).

  clean_paragraphs -> cleaned speaker turns (placeholders dropped, same-speaker runs merged)
  qa_start         -> index of the first Q&A turn, or None
  label_turns      -> role-labelled turns {section, tag, person, text, exchange_idx, answer_idx}
  split_call       -> CallSplit: header-free prepared_remarks / qa texts, labelled turns, status

Measured on 585 defeatbeta calls (2024-2026, 54 symbols): status ok on 99.5 % (582; the other
3 are no_qa: two transcripts without a Q&A session and one fireside interview), every ok call passes
assess_earnings_call_sections, prepared share q05/q50/q95 = 0.19/0.34/0.57. Against the
embeddings `split_turns` labeller on the same turns, answer turns spoken by an analyst of the
same call fall from 8.7 % to 0.3 % and question turns over 300 words from 214 to 19.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, NamedTuple

from src.constants.constants import (
    EARNINGS_CALL_TAG_ANSWER,
    EARNINGS_CALL_TAG_PREPARED,
    EARNINGS_CALL_TAG_QUESTION,
)

_QA_SECTION, _PREP_SECTION = "qa", "prepared_remarks"
SplitStatus = Literal["ok", "no_qa", "no_prepared", "empty"]


class Turn(NamedTuple):
    """One speaker turn: consecutive source paragraphs of the same speaker, newline-joined."""

    speaker: str
    text: str


@dataclass(frozen=True)
class CallSplit:
    """Split of one call. Texts carry no "Speaker:" headers; `turns` is the embedding contract."""

    prepared_remarks: str
    qa: str
    turns: list[dict]
    status: SplitStatus


# ---- paragraph cleaning ------------------------------------------------------ #
_OPERATOR = re.compile(r"(?i)^(?:operator|moderator|coordinator|conference\s+operator)$")
_PLACEHOLDER_SPEAKER = re.compile(r"(?i)^(?:ai\s+insights|speaker\s+\d+|unknown|unidentified.*)$")
_PLACEHOLDER_TEXT = re.compile(r"(?i)^\[?(?:presentation|technical\s+difficulty|break\s+in\s+transmission|music)\]?$")
_ROLE_PREFIX = re.compile(r"^[A-Z]\s*-\s*")  # "A - Ana Soro" (answer/question role marker)
_INVISIBLE = re.compile(r"[\u200b-\u200f\u2060\ufeff]")  # zero-width marks split one speaker in two
_PLACEHOLDER_MAX_WORDS = 15  # a placeholder speaker with a real paragraph (a disclaimer) is kept


def _paragraph_order(p: Mapping[str, object]) -> int:
    n = p.get("paragraph_number", p.get("paragraph"))
    try:
        return int(float(str(n)))
    except ValueError:
        return 0


def clean_paragraphs(paragraphs: Iterable[Mapping[str, object]]) -> list[Turn]:
    """Drop empty and placeholder paragraphs, strip 'A - ' role prefixes, merge consecutive
    paragraphs of the same speaker into one turn."""
    turns: list[Turn] = []
    for p in sorted(paragraphs, key=_paragraph_order):
        speaker = " ".join(_ROLE_PREFIX.sub("", _INVISIBLE.sub("", str(p.get("speaker") or "")).strip()).split())
        text = str(p.get("content") or "").strip()
        if not text or _PLACEHOLDER_TEXT.match(text):
            continue
        if _PLACEHOLDER_SPEAKER.match(speaker) and len(text.split()) < _PLACEHOLDER_MAX_WORDS:
            continue
        if turns and turns[-1].speaker == speaker:
            turns[-1] = Turn(speaker, f"{turns[-1].text}\n{text}")
        else:
            turns.append(Turn(speaker, text))
    return turns


# ---- Q&A start ----------------------------------------------------------------- #
_NAME = r"[A-Z][A-Za-z.'’\-]+(?:\s+[A-Z][A-Za-z.'’\-]+){0,4}"
_QN = r"(?:questions?|q\s*(?:&|and)\s*a)"
_LOGISTICS_MAX = 400  # chars: hand-off / flow lines are short; longer turns are real content
# Phrases that open the Q&A queue ("we'll now turn to questions", "press star one", "the
# question-and-answer session"); intro previews use them too, hence the management-word floor.
_QA_OPENING_SRC = (
    r"press\s+(?:the\s+)?star|in\s+order\s+to\s+ask\s+a\s+question|poll\s+for\s+questions?"
    r"|(?:open|opening)\s+(?:up\s+)?(?:the\s+)?(?:floor|line|lines|call|phone\s+lines)\b"
    r"[^.]{0,25}?(?:for\s+)?" + _QN + r"|(?:we(?:'ll| will| are)|now|let's|i(?:'ll| will))\b[^.]{0,45}?"
    r"(?:begin|open|take|start|move\s+to|go\s+to|turn\s+[^.]{0,20}?to)[^.]{0,30}?" + _QN + r"|question[-\s]and[-\s]answer\s+session"
)
_QA_OPENING = re.compile(_QA_OPENING_SRC, re.I)
# A hand-off that only ever opens a Q&A exchange: it names the asker ("first question comes from",
# "from the line of", "up next, we have"), so it is never an intro preview.
_STRONG_HANDOFF = re.compile(
    r"(?i)(?:first|next|final|last)\s+(?:question|caller)\s+(?:today\s+)?"
    r"(?:(?:comes?|is|will\s+come|will\s+be|we\s+have)\s+)?(?:from|coming\s+from)\b"
    r"|from\s+the\s+line\s+of|up\s+(?:next|first),?\s+we\s+have|go\s+(?:ahead\s+)?to\s+(?:the\s+)?line\s+of"
)
_MIN_MGMT_WORDS = 150  # management words before a question can open the Q&A (skips intro previews)
_MAX_QUESTION_WORDS = 300  # an analyst's asking turn is short


def _is_op(speaker: str) -> bool:
    return bool(_OPERATOR.match(speaker))


def _is_short_question(text: str) -> bool:
    return "?" in text and len(text.split()) <= _MAX_QUESTION_WORDS


def _asks(turns: list[Turn], i: int, seen: set[str]) -> bool:
    """Turn i is a questioner's: the speaker asks a short question (in this turn or their own
    turn within the next two), never speaks longer than an asking turn, and the next other
    non-operator speaker already spoke (management answering). The last two conditions reject an
    executive's short rhetorical opening followed by a new executive or by their own long remarks."""
    spk = turns[i].speaker
    asks = any(s == spk and _is_short_question(t) for s, t in turns[i : i + 3])
    short = all(len(t.split()) <= _MAX_QUESTION_WORDS for s, t in turns if s == spk)
    nxt = next((s for s, _ in turns[i + 1 :] if s != spk and not _is_op(s)), None)
    return asks and short and nxt in seen


def qa_start(turns: list[Turn]) -> int | None:
    """Index of the first Q&A turn, the first of:
    - a strong hand-off naming the asker, said by the operator, an already-seen host, or a new
      speaker whose turn asks nothing (a host opening the call straight into questions);
    - once management has spoken `_MIN_MGMT_WORDS`: a short Q&A-opening turn by the operator or a
      seen host, or the first NEW non-operator speaker who asks a short question (see `_asks`),
      backed up one turn when the preceding turn is the operator/host hand-off."""
    seen: set[str] = set()
    words = 0
    for i, (spk, txt) in enumerate(turns):
        op = _is_op(spk)
        host = op or spk in seen or ("?" not in txt and len(txt.split()) <= _MAX_QUESTION_WORDS)
        if host and _STRONG_HANDOFF.search(txt):
            return i
        if words >= _MIN_MGMT_WORDS:
            if (op or spk in seen) and len(txt) <= _LOGISTICS_MAX and _QA_OPENING.search(txt):
                return i
            if not op and spk not in seen and _asks(turns, i, seen):
                prev, prev_txt = turns[i - 1]
                return i - 1 if _is_op(prev) or (prev in seen and len(prev_txt) <= _LOGISTICS_MAX) else i
        if not op:
            seen.add(spk)
            words += len(txt.split())
    return None


# ---- role labelling: operator / logistics turns ------------------------------------------ #
_HANDOFF = re.compile(
    r"\b(?:next|first|final|last|following)?\s*questions?\s+(?:comes?|is|will\s+come|will\s+be)\s+from\b"
    r"|up\s+(?:next|first),?\s+we\s+have\b|please\s+(?:go\s+ahead|proceed|stand\s+by)|" + _QA_OPENING_SRC,
    re.I,
)
# IR-host flow logistics / sign-off between or after questions: "Operator, next question please.",
# "we have time for one last question", "that wraps up the Q&A ... thank you for joining us."
_FLOW = re.compile(
    r"operator[,.\s][^.]{0,30}?(?:next|last|final|one\s+more)\s+question"
    r"|(?:next|last|final|one\s+more)\s+question[,.\s]*please"
    r"|we\s+have\s+time\s+for\b[^.]{0,25}?questions?"
    r"|that\s+(?:wraps?\s+up|concludes?|will\s+(?:wrap|conclude))\b[^.]{0,30}?"
    r"(?:q\s*(?:&|and)\s*a|call|session|portion)"
    r"|this\s+concludes\b|thank(?:s|\s+you)[^.]{0,20}?for\s+joining",
    re.I,
)
# The analyst NAME announced at a hand-off: the authoritative question-asker of the exchange.
# The phrase is case-insensitive; the captured name stays case-sensitive (a capitalised name).
_HANDOFF_NAME = re.compile(
    r"(?i:(?:questions?|q\s*(?:&|and)\s*a)\s+(?:comes?|is(?:\s+coming)?|will\s+come|will\s+be|coming)\s+from\s+"
    r"(?:the\s+line\s+of\s+)?)(" + _NAME + r")"
    r"|(?i:(?:go|turn|move)\s+(?:ahead\s+)?to\s+(?:the\s+)?line\s+of\s+)(" + _NAME + r")"
    r"|(?i:up\s+(?:next|first),?\s+we\s+have\s+)(" + _NAME + r")"
    r"|(?i:(?:first|next|final|last)\s+question[,:.]?\s+)(" + _NAME + r")"
)
_MIN_TURN = 25  # chars: ignore "thanks"/"good morning" fragments


def _is_operator(person: str, text: str) -> bool:
    """Operator hand-off / call-logistics turn (not content; marks a new exchange). An operator
    speaker is always logistics; otherwise the hand-off/flow phrase must be in a SHORT turn, so a
    long substantive remark that merely mentions 'we'll open it up for questions' is kept. A
    hand-off naming the asker is logistics up to a question's length when it asks nothing itself
    (a host's welcome that ends "we'll take our first question from X")."""
    if _is_op(person):
        return True
    if "?" not in text and len(text.split()) <= _MAX_QUESTION_WORDS and _STRONG_HANDOFF.search(text):
        return True
    return len(text) <= _LOGISTICS_MAX and bool(_HANDOFF.search(text) or _FLOW.search(text) or _HANDOFF_NAME.search(text))


# ---- role labelling: pleasantry cleaning ------------------------------------------------- #
# Analysts open with "Thanks for taking my question / congrats on the quarter" and close with
# "that's helpful / appreciate it / back in the queue"; management opens each answer with
# "Thanks, <name>. / Hi, <name>. / Yeah, <name>.".
_SENT = re.compile(r"[^.?!]+[.?!]*")
_PLEASANTRY_CUE = re.compile(
    r"^(?:hi|hey|hello|good\s+(?:morning|afternoon|evening)|morning|afternoon|thanks?|thank\s+you|"
    r"yeah|yep|yes|sure|okay|ok|great|perfect|excellent|terrific|wonderful|got\s+it|congrats?|"
    r"congratulations|awesome|understood|fair\s+enough)\b"
    r"|taking\s+(?:my|the|our|your)\s+questions?|congrat\w*|appreciate\s+it|back\s+in\s+(?:the\s+)?queue"
    r"|look\s+forward|nice\s+(?:quarter|results?)|great\s+(?:quarter|results?|answer|color|stuff)"
    r"|(?:that'?s|thats)\s+(?:helpful|great|all|it|fair)|thanks\s+so\s+much",
    re.I,
)
_NONINFO_Q = re.compile(
    r"^(?:do\s+you\s+have\s+(?:any\s+)?(?:other\s+|more\s+)?questions?|are\s+you\s+(?:okay|ok|good)"
    r"|any\s+(?:other|more|further)\s+questions?|no\s+(?:more|further)\s+questions?"
    r"|(?:i'?m|we'?re)\s+all\s+set|thank\s+you)\b",
    re.I,
)
# a sentence that OPENS with a greeting and carries a strong courtesy token is preamble even when
# longer ("Hey guys, congrats on the good prints here, ...").
_GREETING_START = re.compile(
    r"^(?:hi|hey|hello|good\s+(?:morning|afternoon|evening)|morning|afternoon|thanks?|thank\s+you|"
    r"yeah|yep|yes|sure|okay|ok|great|perfect|excellent|terrific|wonderful|congrats?|congratulations|"
    r"awesome)\b",
    re.I,
)
_STRONG_COURTESY = re.compile(
    r"congrat|nice\s+(?:quarter|results?|print)|great\s+(?:quarter|results?|print)|"
    r"good\s+(?:print|quarter|results?)|well\s+done|solid\s+(?:quarter|results?|print)|"
    r"thanks?\s+for\s+taking|thank\s+you\s+for\s+taking",
    re.I,
)


def _is_pleasantry(sent: str) -> bool:
    """A courtesy sentence with no question mark: a short (<=14w) sentence with any courtesy cue,
    or a longer opener (<=22w) that starts with a greeting AND carries a strong courtesy token."""
    if "?" in sent:
        return False
    n = len(sent.split())
    if n <= 14 and _PLEASANTRY_CUE.search(sent):
        return True
    return n <= 22 and bool(_GREETING_START.match(sent)) and bool(_STRONG_COURTESY.search(sent))


def _clean(text: str) -> str:
    """The substantive core of a turn: leading and trailing courtesy sentences dropped, whitespace collapsed."""
    sents = [s.strip() for s in _SENT.findall(re.sub(r"\s+", " ", text).strip()) if s.strip()]
    while sents and _is_pleasantry(sents[0]):
        sents.pop(0)
    while sents and _is_pleasantry(sents[-1]):
        sents.pop()
    return " ".join(sents).strip()


def _is_informative_question(text: str) -> bool:
    """A cleaned analyst turn that is a real question: as long as any kept turn, >= 4 words, and
    not 'do you have questions' / pure thanks. The question vector anchors every cosine of its
    exchange, so a meaningless question would corrupt the whole exchange."""
    t = text.strip()
    return len(t) >= _MIN_TURN and len(t.split()) >= 4 and not _NONINFO_Q.match(t)


def _turn(section: str, tag: str, person: str, text: str, exchange_idx: int, answer_idx: int) -> dict:
    return {"section": section, "tag": tag, "person": person, "text": text, "exchange_idx": exchange_idx, "answer_idx": answer_idx}


def label_turns(turns: list[Turn], section: str, mgmt_names: set[str] | None = None) -> list[dict]:
    """Role-label cleaned turns -> [{section, tag, person, text, exchange_idx, answer_idx}].

    * prepared_remarks: every non-logistics turn is 'prepared' (exchange_idx = answer_idx = -1).
    * qa: operator / logistics turns delimit exchanges and are dropped. Role per turn:
        question if the speaker is NAMED by a hand-off or already asked a question in this call,
        else answer if the speaker is management (`mgmt_names`, lower-cased prepared speakers),
        else question right after a hand-off or when, after an answer, a speaker new to the call
        asks a short question (a host hand-off phrased freely: "We've got X at Y"), else answer.
      A new informative question opens a new exchange; answer_idx = 0 for the question and
      1, 2, ... for the answer turns. Turns cleaned below `_MIN_TURN` chars are dropped."""
    if section != _QA_SECTION:
        out: list[dict] = []
        for per, txt in turns:
            body = _clean(txt)
            if not _is_operator(per, txt) and len(body) >= _MIN_TURN:
                out.append(_turn(section, EARNINGS_CALL_TAG_PREPARED, per, body, -1, -1))
        return out

    mn = mgmt_names or set()
    analyst_names = {next(g for g in m.groups() if g).strip().lower() for _per, txt in turns for m in _HANDOFF_NAME.finditer(txt)}
    askers: set[str] = set()
    spoken = set(mn)
    out = []
    ex, ans_i, cur, boundary, saw_answer = -1, 0, None, True, False
    for per, txt in turns:
        if _is_operator(per, txt):
            boundary = True
            continue
        perl = per.strip().lower()
        if perl in analyst_names or perl in askers:
            role = "q"
        elif perl in mn:
            role = "a"
        elif boundary or (saw_answer and perl not in spoken and _is_short_question(txt)):
            role = "q"
        else:
            role = "a"
        body = _clean(txt)
        if role == "q":
            if _is_informative_question(body):
                if boundary or saw_answer or perl != cur:  # a new question -> new exchange
                    ex, ans_i, cur, saw_answer = ex + 1, 0, perl, False
                askers.add(perl)
                spoken.add(perl)
                out.append(_turn(section, EARNINGS_CALL_TAG_QUESTION, per, body, ex, 0))
            boundary = False
        elif ex >= 0 and len(body) >= _MIN_TURN:  # management / specialist answer
            ans_i += 1
            saw_answer = True
            spoken.add(perl)
            out.append(_turn(section, EARNINGS_CALL_TAG_ANSWER, per, body, ex, ans_i))
    return out


def split_call(paragraphs: Iterable[Mapping[str, object]]) -> CallSplit:
    """Split one call: prepared_remarks = non-operator text before the Q&A start, qa = every turn
    from it; turns = labelled prepared turns then labelled Q&A turns (management names for the Q&A
    roles come from the prepared speakers). Status: empty (no turn), no_qa (no Q&A start found;
    the whole call is prepared), no_prepared (nothing before the Q&A start), ok."""
    turns = clean_paragraphs(paragraphs)
    if not turns:
        return CallSplit("", "", [], "empty")
    k = qa_start(turns)
    head, tail = (turns, []) if k is None else (turns[:k], turns[k:])
    prepared = "\n".join(t for s, t in head if not _is_op(s))
    qa = "\n".join(t for _, t in tail)
    prep_turns = label_turns(head, _PREP_SECTION)
    mgmt = {t["person"].strip().lower() for t in prep_turns if t["person"]}
    labelled = prep_turns + label_turns(tail, _QA_SECTION, mgmt)
    status: SplitStatus = "no_qa" if k is None else "no_prepared" if not prepared.strip() else "ok"
    return CallSplit(prepared, qa, labelled, status)
