"""
earnings_call_split.py  (src/utils/earnings_call_split.py)
----------------------------------------------------------
Pure speaker-turn split of one earnings-call transcript, given as source paragraphs
(`speaker`, `content`, optional `paragraph_number` / `paragraph`).

  clean_paragraphs -> cleaned speaker turns (placeholders dropped, same-speaker runs merged)
  qa_start         -> index of the first Q&A turn, or None
  split_sentences  -> lossless sentence split (never cuts "3.5%", "U.S. to", "$1.2")
  clean_turn_text  -> one turn's substantive text: courtesy / backchannel, self-introductions,
                      addressed names and hand-offs removed anywhere; in prepared remarks also
                      safe-harbor / IR boilerplate and participant rosters. Hedges are kept.
  label_turns      -> role-labelled turns {section, tag, person, text, exchange_idx, answer_idx}
  split_call       -> CallSplit: cleaned prepared_remarks, cleaned management-answer qa, turns, status

Measured on 585 defeatbeta calls (2024-2026, 54 symbols): status ok on 99.3 % (581; 3 no_qa: two
transcripts without a Q&A session and one fireside interview; 1 no_prepared: a live-on-X call whose
only prepared text is the safe harbor), every ok call passes assess_earnings_call_sections, prepared
share of call words q05/q50/q95 = 0.18/0.32/0.54, answer turns spoken by a questioner of the same call
2 of 11,594, persons tagged both question and answer 2, question turns over 300 words 11. On the
5,241 ok live calls since 2024, 3 carry a question-tagged person with >= 1,000 words: two verbose
analysts and one executive who speaks in a single exchange (CRM 2027Q1). On a 2,940-call sample
(2006-2026): 0 broken decimals introduced (the 130-171 left are the
source's own "3. 5"), boilerplate 5.0 % -> 1.5 % of prepared words (the rest carries figures), 9 ok
calls have no management answer text (defective transcripts) and fail the gate. Precision of the
removals, 20 random hits each: courtesy, self-introduction, addressed name 20/20; boilerplate,
roster, hand-off 19/20.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
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


# ---- sentence splitting ------------------------------------------------------------------ #
# A boundary is terminal punctuation (optionally closed by a quote or bracket) FOLLOWED BY whitespace
# and a character that is not lower-case: "3.5%", "$1.2", "U.S. to" are never cut, and
# " ".join(split_sentences(t)) is exactly the whitespace-collapsed t. An abbreviation before a capital
# ("U.S. Revenue") adds a boundary but never changes a character; titles ("Dr. Smith") never split.
_SENT_BOUNDARY = re.compile(r"(?:(?<=[.?!])|(?<=[.?!][\"'”’)\]]))(?<!\bDr\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bMrs\.)(?<!\bSt\.)\s+(?=[^a-z\s])")


def split_sentences(text: str) -> list[str]:
    """Sentences of `text` after whitespace collapse; rejoining them with one space is lossless."""
    collapsed = " ".join(str(text).split())
    return _SENT_BOUNDARY.split(collapsed) if collapsed else []


# ---- sentence cleaning: shared vocabulary ------------------------------------------------- #
# A sentence is noise when it has the class cue and nothing but function words, the class vocabulary
# (which holds every cue word), years, the call's participant names and at most one other capitalised
# token (a company or nickname), and it carries no figure ($, %, bps, million, ...).
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’&.\-]*")
_YEAR = re.compile(r"(?:19|20)\d\d")
_FIGURE = re.compile(
    r"\$\s?\d|\d(?:[\d,.]*\d)?\s?(?:%|percent\b|per\s+cent\b|basis\s+points?\b|bps\b|million\b|billion\b|thousand\b|cents?\b)",
    re.I,
)
_FUNCTION = frozenset(
    """a an the and or but so of for on in at to from with by as about into onto up out over back again too also just very
    really much all both any some this that these those it its is are was were be been being am have has had do does did
    i i'm i'll i'd i've me my we we're we'll we've we'd our us you you're you'll you've you'd your he she him her his they
    they're them their there here now then well oh ah uh um if can could would will should may might let let's get go
    going one don't didn't won't can't""".split()
)
_COURTESY_WORDS = frozenset(
    """thank thanks thx appreciate appreciated appreciating appreciation congrats congratulations congratulation congrat
    congratulate kudos good great nice terrific excellent fantastic impressive wonderful awesome amazing outstanding perfect
    super hi hey hello morning afternoon evening day everyone everybody guys folks team gentlemen ladies welcome pleasure glad
    question questions quarter results job color colour detail details answer answers insight insights comment comments
    time call conference taking take sharing joining helpful useful clear makes sense got gotcha yes yeah yep yup sure okay
    ok right alright absolutely correct exactly understood fair enough cool indeed definitely certainly course true fine
    sounds operator instructions queue hop jump look looking forward speaking talking seeing catching chatting soon later
    done sorry please ahead wish best luck today today's thing things everything bit lot q1 q2 q3 q4 that's print prints
    execution year stuff start update solid strong""".split()
)
_HANDOFF_WORDS = frozenset(
    """turn turning hand handing pass passing over back call floor mic microphone line lines open opening up now discuss
    discussing review reviewing cover covering walk walking through provide providing give giving share sharing talk
    talking speak speaking details detail more further additional financial financials results performance quarter
    quarterly ceo cfo coo cto chief executive operating officer president vice senior svp evp chairman chair head investor
    relations treasurer controller director founder who will then after before comments remarks few some opening closing
    prepared discussion questions question q&a operator take taking start begin conclude go ahead let like want add anything
    answer address comment handle jump chime slide slides page colleague colleagues highlights update overview business
    outlook guidance depth greater session portion""".split()
)
_NAME_RX = r"[A-Z][\w.'’\-]+(?:\s+[A-Z][\w.'’\-]+){0,2}"


class CallNames(NamedTuple):
    """Participant names of one call: single name tokens and full names (longest first)."""

    tokens: frozenset[str]
    full: tuple[str, ...]


_NO_NAMES = CallNames(frozenset(), ())
_NAME_STOP = frozenset(
    "Mr Mrs Ms Dr Jr Sr II III IV The And Of Inc LLC Ltd Corp Co Operator Analyst Unknown Speaker Executive Officer Chief "
    "President CEO CFO Company Representative Management Moderator Host Analysts Executives Participants Corporate "
    "I'll I'm".split()
)


def _is_pure(sent: str, vocab: frozenset[str], names: frozenset[str], max_lower: int = 0) -> bool:
    """No figure, and every token of `sent` is a function word, in `vocab`, a year or a participant
    name, except at most one other capitalised token and `max_lower` other words (a digit is a word)."""
    upper = lower = 0
    for tok in _TOKEN.findall(sent):
        t = tok.rstrip(".'’-")
        if t[-2:] in ("'s", "’s"):
            t = t[:-2]
        if not t or t in names:
            continue
        low = t.lower()
        if low in _FUNCTION or low in vocab or (len(t) == 4 and _YEAR.fullmatch(t)):
            continue
        if not t[0].isupper():
            lower += 1
            if lower > max_lower:
                return False
            continue
        upper += 1
        if upper > 1:
            return False
    return not _FIGURE.search(sent)


# ---- courtesy / backchannel ---------------------------------------------------------------- #
# Analysts open with "Thanks for taking my question / congrats on the quarter" and close with
# "that's helpful / back in the queue"; management opens answers with "Thanks, <name>." and both
# sides backchannel ("Okay.", "Yes, that's right.").
_COURTESY_CUE = re.compile(
    r"\b(?:thanks?|thank\s+you|thx|appreciate\w*|congrat\w*|kudos|good\s+(?:morning|afternoon|evening|day)|morning|afternoon"
    r"|hi|hey|hello|welcome|pleasure|well\s+done|operator\s+instructions?"
    r"|(?:great|good|nice|solid|strong|terrific|excellent|fantastic|impressive|wonderful)\s+(?:quarter|results?|prints?|job"
    r"|execution|year|questions?|color|stuff|answer|detail|update|start)"
    r"|taking\s+(?:my|the|our|your|these|those|a)\s+questions?|(?:back|hop)\s+(?:back\s+)?(?:in|into)\s+(?:the\s+)?queue"
    r"|look(?:ing)?\s+forward\s+to|(?:that'?s|that\s+is|very|super|really)\s+helpful|that'?s\s+(?:all|it)"
    r"|yes|yeah|yep|yup|okay|ok|sure|right|alright|all\s+right|absolutely|correct|exactly|understood|got\s+it|gotcha|perfect"
    r"|great|excellent|terrific|wonderful|awesome|fantastic|makes\s+sense|fair\s+enough|sounds\s+good|of\s+course"
    r"|definitely|certainly|indeed|cool)\b",
    re.I,
)
_COURTESY_QUESTION = re.compile(
    r"^(?:(?:hi|hey|hello|good\s+(?:morning|afternoon|evening)|thanks?|thank\s+you|okay|ok|yes|yeah)[\s,.!]*"
    r"(?:(?:everyone|all|guys|team|there)[\s,.!]+)?)?"
    r"(?:how\s+are\s+you(?:\s+(?:doing|all|guys|today))*|can\s+you\s+(?:hear|see)\s+me(?:\s+(?:now|okay|ok|all\s+right|alright))?"
    r"|(?:are\s+you|is\s+everyone)\s+(?:there|with\s+me)|do\s+you\s+have\s+(?:any\s+)?(?:other|more|further)\s+questions?"
    r"|any\s+(?:other|more|further)\s+questions?|did\s+i\s+lose\s+you|am\s+i\s+(?:on|there|audible))\W*$",
    re.I,
)
_NONINFO_Q = re.compile(
    r"^(?:do\s+you\s+have\s+(?:any\s+)?(?:other\s+|more\s+)?questions?|are\s+you\s+(?:okay|ok|good)"
    r"|any\s+(?:other|more|further)\s+questions?|no\s+(?:more|further)\s+questions?"
    r"|(?:i'?m|we'?re)\s+all\s+set|thank\s+you)\b",
    re.I,
)
_MAX_NOISE_WORDS = 40  # a longer sentence is never courtesy / hand-off noise


def is_pleasantry(sent: str, names: frozenset[str] = frozenset()) -> bool:
    """A courtesy or backchannel sentence: a non-informative question ("How are you?"), or a
    pure sentence (see `_is_pure`) with a courtesy cue. A real question is kept."""
    if len(sent.split()) > _MAX_NOISE_WORDS:
        return False
    if "?" in sent:
        return bool(_COURTESY_QUESTION.match(sent))
    return _is_pure(sent, _COURTESY_WORDS, names) and bool(_COURTESY_CUE.search(sent))


# ---- self-introductions and addressed names ------------------------------------------------ #
# "Hi, this is Jane Roe from Big Bank." / "Jane Roe on for John Doe." -- the clause is removed and
# the rest of the sentence kept ("..., I wanted to ask about margins").
_ORG_RX = r"(?:the\s+)?[A-Z0-9][\w.&'’\-]*(?:\s+(?!I\b)(?:[A-Z0-9&][\w.&'’\-]*|of|de|du|la|&))*"
_INTRO_PREP = r"(?i:on\s+for|filling\s+in\s+for|sitting\s+in\s+for|in\s+for|on\s+behalf\s+of)"
_SELF_INTRO = re.compile(
    r"^(?i:(?:hi|hey|hello|yes|yeah|good\s+(?:morning|afternoon|evening)|thanks?|thank\s+you)[,.!]?\s*"
    r"(?:(?:everyone|all|guys|team|there)[,.!]?\s*)?)?"
    r"(?:(?i:this\s+is|it'?s|it\s+is|i'?m|i\s+am)\s+(?i:actually\s+)?" + _NAME_RX + r"\s*,?\s+"
    r"(?:" + _INTRO_PREP + r"|(?i:calling\s+(?:in\s+)?(?:for|from)|from|with|at|for|here\s+(?:for|with|from))\b)"
    r"|" + _NAME_RX + r"\s+" + _INTRO_PREP + r")\s+" + _ORG_RX + r"\s*[,.;!]?\s*(?i:and\s+)?"
)
# "Thanks, Brian." / "Yes, John, we ..." -- the vocative is removed. A name that is not a participant
# of the call is only accepted after a courtesy cue; after "so / well / and" it must be a participant.
_ADDRESS_CUE_ANY = (
    r"(?i:thanks(?:\s+so\s+much)?|thank\s+you(?:\s+(?:so|very)\s+much)?|yes|yeah|yep|hi|hey|hello|sure|okay|ok|great"
    r"|good\s+(?:morning|afternoon|evening))"
)
_ADDRESS_CUE_KNOWN = r"(?i:sorry|well|so|and|right|absolutely|no|morning|afternoon)"
_ADDRESSED = re.compile(
    r"^(?P<cue>" + _ADDRESS_CUE_ANY + r"|" + _ADDRESS_CUE_KNOWN + r")(?:,\s*|\s+)"
    r"(?P<name>[A-Z][a-z'’\-]{1,15}(?:\s+[A-Z][a-z'’\-]{1,15})?)(?P<tail>\s*[,.!?](?:\s+|$)|$)"
)
_VOCATIVE = re.compile(r"^(?P<name>[A-Z][a-z'’\-]{1,15}(?:\s+[A-Z][a-z'’\-]{1,15})?),\s+")
_ADDRESS_NOT = frozenset(
    "And So Okay Yes Well The We Our Thanks Thank Good Great I That This Sure Yeah Very All Again Everyone Everybody Guys "
    "Operator Team Folks There Absolutely Just First Second Third Now Then Also Look Listen One In On As It Its You Your "
    "Overall Sorry Hi Hey Hello Right No Not Exactly Correct Perfect Ok Yep Understood Got Indeed Certainly Sir Gentlemen "
    "Ladies Morning Afternoon Evening Today Yesterday Tomorrow Monday Tuesday Wednesday Thursday Friday January February "
    "March April June July August September October November December Here What How Why When Where Which".split()
)


def _strip_self_intro(sent: str) -> str | None:
    """`sent` without a leading self-introduction clause; None when there is none."""
    m = _SELF_INTRO.match(sent)
    return None if m is None else sent[m.end() :]


def _strip_addressed(sent: str, names: frozenset[str]) -> str:
    """`sent` without a leading vocative name ("Yes, John, we ..." -> "Yes, we ...")."""
    m = _ADDRESSED.match(sent)
    if m is not None:
        toks = m.group("name").split()
        known = all(t in names for t in toks)
        generic = re.fullmatch(_ADDRESS_CUE_ANY, m.group("cue")) is not None and toks[0] not in _ADDRESS_NOT
        if known or generic:
            rest = sent[m.end() :]
            end = m.group("tail").strip()[:1]
            return f"{m.group('cue')}, {rest}" if rest else f"{m.group('cue')}{end if end in '.!?' else '.'}"
    v = _VOCATIVE.match(sent)
    if v is not None and all(t in names for t in v.group("name").split()):
        rest = sent[v.end() :]
        return rest[:1].upper() + rest[1:]
    return sent


# ---- hand-offs ----------------------------------------------------------------------------- #
# "With that, I'll turn it over to Jane Doe, our CFO." is dropped; a hand-off that also carries
# content keeps its content and loses the participant names.
# (keys, pattern, content words a pure hand-off may still carry): the pattern runs only when one key
# is a substring of the lower-cased sentence. "Turn it over to Mike for the forecast" is logistics;
# "Bill, if you could give us the contributions" is a request and must be pure.
_HANDOFF_PARTS = (
    (
        ("turn", "hand", "pass"),
        re.compile(
            r"\b(?:turn|turning|hand|handing|pass|passing)\s+(?:(?:it|this|things|everything|the\s+(?:call|floor|mic|microphone"
            r"|line|discussion|presentation))\s+(?:back\s+)?(?:over\s+)?|(?:back\s+)?over\s+|back\s+)to\b",
            re.I,
        ),
        2,
    ),
    (
        ("let ", "have ", "ask "),
        re.compile(
            r"\b(?i:let|have|ask)\s+" + _NAME_RX + r"\s+(?i:to\s+)?(?i:answer|take|address|comment|add|cover|talk|speak|handle"
            r"|jump|walk|provide|give|discuss|review|chime)"
        ),
        0,
    ),
    (
        ("want", "like", "why don", "can you", "could you", "maybe you", "if you", "perhaps you", "you can"),
        re.compile(
            r"^(?!(?:So|And|But|Well|Okay|Now|Then|Also|Maybe|Yes|Yeah|Great|Sure)\b)"
            + _NAME_RX
            + r",?\s+(?i:do\s+you\s+want|would\s+you\s+like|why\s+don'?t\s+you|can\s+you|could\s+you"
            r"|you\s+want|maybe\s+you\s+can|if\s+you\s+(?:want|could)|perhaps\s+you\s+can|you\s+can)\s+(?i:to\s+)?"
            r"(?i:take|add|answer|cover|comment|talk|address|jump|chime|speak|handle|provide|give|walk)"
        ),
        0,
    ),
    (("star", "question", "poll", "q&a", "q & a", "q and a"), _QA_OPENING, 2),
)
_HANDOFF_VOCAB = _HANDOFF_WORDS | _COURTESY_WORDS
_TURN_TO_NAME = re.compile(r"\b(?i:turn|turning|hand|handing|pass|passing)\s+to\s+([A-Z][\w'’\-]+)")


def _handoff_allowance(sent: str, names: frozenset[str]) -> int | None:
    """None when `sent` has no hand-off phrase ("turn it over to", "let Jane take that", "Jane, do
    you want to add", a Q&A opening, or a bare "turn to <participant>"); else the content words the
    sentence may carry and still be pure hand-off logistics."""
    low = sent.lower()
    allowed = [n for keys, rx, n in _HANDOFF_PARTS if any(k in low for k in keys) and rx.search(sent)]
    if any(m.group(1) in names for m in _TURN_TO_NAME.finditer(sent)):
        allowed.append(2)
    return max(allowed) if allowed else None


def _strip_names(sent: str, names: CallNames) -> str:
    """`sent` without participant names, tidied where a name was removed: a name before punctuation
    goes with its preposition ("before I turn it over to Jane, I want" -> "before I turn it over, I
    want"), a leading vocative goes, a name inside a clause becomes "a colleague" ("I'll let Lee talk
    about" -> "I'll let a colleague talk about")."""
    gap = "\x00"
    for full in names.full:
        sent = re.sub(rf"\b{re.escape(full)}(?:['’]s)?(?![\w'’])", gap, sent)
    sent = re.sub(r"\b[A-Z][\w'’\-]*", lambda m: gap if re.sub(r"['’]s$", "", m.group()) in names.tokens else m.group(), sent)
    if gap not in sent:
        return sent
    sent = re.sub(r"\x00(?:\s*\x00)+", gap, sent)
    sent = re.sub(r"(?:\s+(?:to|by|from|with))?\s*\x00\s*(?=[,;:!?]|\.(?:\s|$)|$)", "", sent)
    sent = re.sub(r"^\s*\x00[\s,]*", "", sent)
    sent = sent.replace(gap, "a colleague")
    sent = re.sub(r",(?:\s*,)+", ",", sent)
    sent = re.sub(r"^[\s,;:]+", "", sent).strip()
    return sent[:1].upper() + sent[1:]


# ---- prepared-remarks boilerplate and roster ---------------------------------------------- #
# Legal boilerplate is dropped whatever it carries; IR pointers (non-GAAP measures, webcast, press
# release, website, SEC filings, slides, the welcome line) and roster sentences only when they carry
# no figure, so "Non-GAAP operating margin increased to 25.5%" is kept.
_BOILER_LEGAL = re.compile(
    r"forward[-\s]looking\s+(?:statements?|information|comments|remarks|projections)"
    r"|\b(?:are|be|is|include|includes|contain|contains|constitute|constitutes|considered)\s+(?:\w+\s+){0,2}forward[-\s]looking"
    r"|safe[-\s]harbou?r|private\s+securities\s+litigation"
    r"|actual\s+(?:results|events|outcomes|performance)\s+(?:\w+\s+){0,3}(?:differ|vary)|differ\s+materially"
    r"|risk\s+factors|cautionary\s+(?:statements?|notes?|language)|undertakes?\s+no\s+(?:obligation|duty)"
    r"|(?:no|any)\s+(?:obligation|duty|intention)\s+to\s+(?:publicly\s+)?(?:update|revise)"
    r"|(?:subject\s+to|involves?)\s+(?:\w+\s+){0,3}risks\s+and\s+uncertainties|being\s+recorded|listen[-\s]only",
    re.I,
)
_POINTER_VERB = re.compile(
    r"\b(?:available|posted|found|locate\w*|accessible|access(?:ed)?|archived|visit|download\w*|go\s+to|going\s+to|listen"
    r"|refer|see|view|furnished|included|provided|attached|accompany\w*)\b",
    re.I,
)
_BOILER_IR = re.compile(
    r"non[-\s]?gaap\s+(?:financial\s+)?(?:measures?|metrics|information)"
    r"|\bweb\s?cast|\breplay\b"
    r"|(?:press|earnings|news)\s+release\s+(?:\w+\s+){0,3}(?:issued|distributed|released|posted|available|published|furnished"
    r"|filed|attached)"
    r"|(?:issued|distributed|posted|published|furnished|filed)\s+(?:\w+\s+){0,3}(?:press|earnings|news)\s+release"
    r"|(?:see|refer\s+to|found\s+in|included\s+in|provided\s+in|available\s+in|contained\s+in|detailed\s+in|described\s+in"
    r"|outlined\s+in|accompanying|conjunction\s+with)\s+(?:\w+\s+){0,3}(?:press|earnings|news)\s+release"
    r"|filings?\s+with\s+the\s+(?:sec|securities)|\bsec\s+filings?|\bform\s+(?:10-?[kq]|8-?k)|\b10-?[kq]s?\b|\b8-?k\b"
    r"|annual\s+report\s+on\s+form|securities\s+and\s+exchange\s+commission"
    r"|(?:slides?|slide\s+deck|deck|presentation|supplement\w*|materials?)\s+(?:\w+\s+){0,4}(?:available|posted)"
    r"|\bwelcome\b(?:\s+\w+){0,4}?\s+to\s+(?:[\w.'’&\-]+\s+){0,8}?(?:call|webcast|conference)\b"
    r"|\bthank(?:s|\s+you)\s+(?:\w+\s+){0,3}for\s+(?:joining|participating|attending|being\s+with|dialing)",
    re.I,
)
_BOILER_POINTED = re.compile(
    r"reconcil\w*|web\s?site|web\s+page|investor\s+relations\s+(?:section|site|page|portion|tab)|\bir\s+(?:website|site|page)|www\.",
    re.I,
)
_ROSTER = re.compile(
    r"\bjoined\s+(?:today\s+|here\s+|this\s+(?:morning|afternoon|evening)\s+|on\s+(?:the|today'?s|this)\s+call\s+)?by\b"
    r"|\bjoining\s+me\b|\bjoining\s+us\b(?=[^.]*?\b(?:are|is|will\s+be)\b)|\b(?:i'?m|we'?re|i\s+am|we\s+are)\s+joined\b"
    r"|\b(?:here\s+)?with\s+me\s+(?:today|here|now|on\s+(?:the|today'?s|this)\s+call|this\s+(?:morning|afternoon|evening)"
    r"|in\s+the\s+room)"
    r"|\b(?:also\s+)?on\s+(?:the|today'?s|this)\s+call\s+(?:today\s+|with\s+me\s+)?(?:are|is|we\s+have|will\s+be)\b"
    r"|\bparticipating\s+(?:on|in)\s+(?:the|today'?s|this)\s+call\b",
    re.I,
)


_LEGAL_KEYS = (
    "forward",
    "harbo",
    "litigation",
    "differ",
    "vary",
    "risk factor",
    "cautionary",
    "obligation",
    "duty",
    "intention",
    "uncertaint",
    "recorded",
    "listen",
)
_IR_KEYS = (
    "gaap",
    "webcast",
    "web cast",
    "replay",
    "release",
    "filing",
    "10-",
    "10k",
    "10q",
    "8-k",
    "8k",
    "annual report",
    "exchange commission",
    "available",
    "posted",
    "welcome",
    "thank",
)
_POINTED_KEYS = ("reconcil", "web", "investor relations", "ir site", "ir page", "www.")
_ROSTER_KEYS = ("joined", "joining", "with me", "on the call", "on today", "on this call", "participating")


def _has(low: str, keys: tuple[str, ...]) -> bool:
    return any(k in low for k in keys)


def _prepared_noise(sent: str, low: str) -> str | None:
    """'boilerplate' / 'roster' when a prepared-remarks sentence (`low` = lower-cased) is legal / IR
    boilerplate or a participant roster, else None. Substring keys gate each regex (speed only)."""
    if _has(low, _LEGAL_KEYS) and _BOILER_LEGAL.search(sent):
        return "boilerplate"
    if _FIGURE.search(sent):
        return None
    if _has(low, _IR_KEYS) and _BOILER_IR.search(sent):
        return "boilerplate"
    if _has(low, _POINTED_KEYS) and _BOILER_POINTED.search(sent) and _POINTER_VERB.search(sent):
        return "boilerplate"
    return "roster" if _has(low, _ROSTER_KEYS) and _ROSTER.search(sent) else None


# ---- the shared cleaner -------------------------------------------------------------------- #
def clean_turn_text(
    text: str,
    names: CallNames = _NO_NAMES,
    *,
    prepared: bool = False,
    removed: list[tuple[str, str]] | None = None,
) -> str:
    """The substantive text of one turn, sentence by sentence anywhere in the turn: courtesy /
    backchannel sentences, self-introductions, addressed names and pure hand-offs removed (names
    stripped from a hand-off that carries content); in prepared remarks also boilerplate and roster
    sentences. Hedges and fillers are kept. `removed` collects (class, fragment) for audits."""
    kept: list[str] = []

    def drop(cls: str, frag: str) -> None:
        if removed is not None:
            removed.append((cls, frag))

    for sentence in split_sentences(text):
        sent = sentence
        cls = _prepared_noise(sent, sent.lower()) if prepared else None
        if cls is not None:
            drop(cls, sent)
            continue
        rest = _strip_self_intro(sent)
        if rest is not None:
            drop("self_intro", sent[: len(sent) - len(rest)])
            sent = rest
            if not sent.strip():
                continue
        stripped = _strip_addressed(sent, names.tokens)
        if stripped != sent:
            drop("addressed_name", sent)
            sent = stripped
        if is_pleasantry(sent, names.tokens):
            drop("courtesy", sent)
            continue
        allowance = _handoff_allowance(sent, names.tokens)
        if allowance is not None:
            if len(sent.split()) <= _MAX_NOISE_WORDS and _is_pure(sent, _HANDOFF_VOCAB, names.tokens, allowance):
                drop("handoff", sent)
                continue
            no_names = _strip_names(sent, names)
            if no_names != sent:
                drop("handoff_name", sent)
                sent = no_names
        kept.append(sent)
    return " ".join(kept).strip()


_PERSON_SUFFIX = re.compile(r"\s+[-–—]\s+|,")  # "Name - Firm" / "Name, Title" speaker strings
_ANNOUNCE_KEYS = ("question", "q&a", "q & a", "q and a", "line of", "up next", "up first")


def _announced(turns: list[Turn]) -> set[str]:
    """The analyst names announced by hand-offs ("next question comes from Jane Roe")."""
    return {next(g for g in m.groups() if g).strip() for _, txt in turns if _has(txt.lower(), _ANNOUNCE_KEYS) for m in _HANDOFF_NAME.finditer(txt)}


def _call_names(turns: list[Turn], announced: set[str] | None = None) -> CallNames:
    """Participant names of a call: every non-operator speaker plus every analyst a hand-off announces."""
    speakers = {per for per, _ in turns if per and not _is_op(per) and not _PLACEHOLDER_SPEAKER.match(per)}
    full = {_PERSON_SUFFIX.split(per)[0].strip() for per in speakers} | (_announced(turns) if announced is None else announced)
    full = {f for f in full if f and f not in _NAME_STOP}
    tokens = {t for f in full for t in re.findall(r"[A-Z][A-Za-z'’\-]+", f) if t not in _NAME_STOP and t not in _ADDRESS_NOT}
    return CallNames(frozenset(tokens), tuple(sorted(full, key=len, reverse=True)))


# ---- role labelling -------------------------------------------------------------------------- #
_MIN_QA_WORDS = 10  # cleaned words: a shorter question or answer turn is not embedded
_MIN_ASK_WORDS = 4  # cleaned words: a shorter question turn does not make its speaker an asker


def _asks_something(text: str) -> bool:
    """A cleaned analyst turn that asks something: >= `_MIN_TURN` chars, >= 4 words, not
    'do you have questions' / pure thanks. Decides who is an asker of the call."""
    t = text.strip()
    return len(t) >= _MIN_TURN and len(t.split()) >= _MIN_ASK_WORDS and not _NONINFO_Q.match(t)


def _is_informative_question(text: str) -> bool:
    """A cleaned question turn worth embedding: it asks something and has >= `_MIN_QA_WORDS` words.
    The question vector anchors every cosine of its exchange, so a meaningless question would
    corrupt the whole exchange."""
    return _asks_something(text) and len(text.split()) >= _MIN_QA_WORDS


def _turn(section: str, tag: str, person: str, text: str, exchange_idx: int, answer_idx: int) -> dict:
    return {"section": section, "tag": tag, "person": person, "text": text, "exchange_idx": exchange_idx, "answer_idx": answer_idx}


_MIN_ANSWER_EXCHANGES = 3  # exchanges a speaker the hand-offs never name must speak in to be management
_ANALYST_LABEL = re.compile(r"\banalyst")  # "unidentified analyst" pools several askers under one label


def _qa_management(turns: list[Turn], analyst_names: set[str]) -> set[str]:
    """Lower-cased Q&A speakers who are management by evidence: they speak in at least
    `_MIN_ANSWER_EXCHANGES` exchanges (delimited by logistics turns and asker announcements), no
    hand-off names them, their label is not an analyst placeholder ("Unidentified Analyst" pools
    several askers), and none of their turns announces an asker (a host reading the queue).

    An analyst asks within one exchange; an executive who skipped the prepared remarks answers
    across many. Without this, such an executive is a questioner wherever a hand-off-like turn
    precedes them: an operator turn that swallowed the analyst's question (TXN 2025Q1), a
    management redirect ("Cristian, you want to start", BMY 2026Q1), or an interjection that ends
    in a question mark (JPM 2024Q3), and the sticky-asker rule then keeps them a questioner for
    the rest of the call. An executive who speaks in fewer exchanges keeps the old rules."""
    exchanges: dict[str, set[int]] = {}
    hosts: set[str] = set()
    ex = 0
    for per, txt in turns:
        perl = per.strip().lower()
        if _is_operator(per, txt):
            ex += 1
            continue
        if _STRONG_HANDOFF.search(txt) or _HANDOFF_NAME.search(txt):  # a host reads the next question
            hosts.add(perl)
            ex += 1
            continue
        exchanges.setdefault(perl, set()).add(ex)
    return {
        per
        for per, seen in exchanges.items()
        if len(seen) >= _MIN_ANSWER_EXCHANGES and per not in analyst_names and per not in hosts and not _ANALYST_LABEL.search(per)
    }


def _label_prepared(turns: list[Turn], section: str, names: CallNames) -> list[dict]:
    """Labelled prepared turns: every non-logistics turn whose cleaned text has >= `_MIN_TURN` chars."""
    prepared: list[dict] = []
    for per, txt in turns:
        if _is_operator(per, txt):
            continue
        body = clean_turn_text(txt, names, prepared=True)
        if len(body) >= _MIN_TURN:
            prepared.append(_turn(section, EARNINGS_CALL_TAG_PREPARED, per, body, -1, -1))
    return prepared


@dataclass
class _QaState:
    """Running state of the Q&A labelling pass over one call's turns."""

    analyst_names: set[str]
    management: set[str]
    spoken: set[str]
    askers: set[str] = field(default_factory=set)
    out: list[dict] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    ex: int = -1
    ans_i: int = 0
    cur: str | None = None
    boundary: bool = True
    saw_answer: bool = False
    open_ex: bool = False


def _is_question_turn(state: _QaState, perl: str, txt: str) -> bool:
    """Role of one Q&A turn by the precedence documented in `label_turns`."""
    if perl in state.analyst_names:
        return True
    if perl in state.management:
        return False
    if perl in state.askers:
        return True
    return state.boundary or (state.saw_answer and perl not in state.spoken and _is_short_question(txt))


def _label_question(state: _QaState, per: str, perl: str, body: str) -> None:
    """A question turn: marks an asker, opens an exchange when informative and new."""
    new = state.boundary or state.saw_answer or perl != state.cur
    if _asks_something(body):
        state.askers.add(perl)
        state.spoken.add(perl)
    if _is_informative_question(body):
        if new or not state.open_ex:  # a new question -> new exchange
            state.ex, state.ans_i, state.cur, state.saw_answer, state.open_ex = state.ex + 1, 0, perl, False, True
        state.out.append(_turn(_QA_SECTION, EARNINGS_CALL_TAG_QUESTION, per, body, state.ex, 0))
    elif new:  # a question too short to embed opens no exchange: its answers are orphaned
        state.cur, state.saw_answer, state.open_ex = perl, False, False
    state.boundary = False


def _label_answer(state: _QaState, per: str, perl: str, body: str) -> None:
    """A management / specialist answer turn: kept as answer text, attached to an open exchange."""
    if body:
        state.answers.append(body)
    if state.ex < 0 or len(body) < _MIN_TURN:
        return
    state.saw_answer = True
    state.spoken.add(perl)
    if state.open_ex and len(body.split()) >= _MIN_QA_WORDS:
        state.ans_i += 1
        state.out.append(_turn(_QA_SECTION, EARNINGS_CALL_TAG_ANSWER, per, body, state.ex, state.ans_i))


def _label_qa(turns: list[Turn], mgmt_names: set[str] | None, names: CallNames, announced: set[str] | None = None) -> tuple[list[dict], list[str]]:
    """(labelled Q&A turns, cleaned management answer texts) of one call; see `label_turns`."""
    analyst_names = {n.lower() for n in (_announced(turns) if announced is None else announced)}
    management = (mgmt_names or set()) | _qa_management(turns, analyst_names)
    state = _QaState(analyst_names, management, set(management))
    for per, txt in turns:
        if _is_operator(per, txt):
            state.boundary = True
            continue
        perl = per.strip().lower()
        question = _is_question_turn(state, perl, txt)
        body = clean_turn_text(txt, names)
        if question:
            _label_question(state, per, perl, body)
        else:
            _label_answer(state, per, perl, body)
    return state.out, state.answers


def label_turns(turns: list[Turn], section: str, mgmt_names: set[str] | None = None, names: CallNames | None = None) -> list[dict]:
    """Role-label turns -> [{section, tag, person, text, exchange_idx, answer_idx}], texts cleaned by
    `clean_turn_text` with the call's participant `names` (derived from `turns` when omitted).

    * prepared_remarks: every non-logistics turn is 'prepared' (exchange_idx = answer_idx = -1),
      kept when its cleaned text has >= `_MIN_TURN` chars.
    * qa: operator / logistics turns delimit exchanges and are dropped. Role per turn:
        question if the speaker is NAMED by a hand-off, else answer if the speaker is management
        (`mgmt_names`, lower-cased prepared speakers, plus the Q&A speakers `_qa_management` finds
        answering across exchanges), else question if they already asked a question in this call,
        else question right after a hand-off or when, after an answer, a speaker new to the call
        asks a short question (a host hand-off phrased freely: "We've got X at Y"), else answer.
      Question and answer turns need >= `_MIN_QA_WORDS` cleaned words. A kept question opens a new
      exchange (indices stay contiguous); answer_idx = 0 for the question and 1, 2, ... for the
      answers. A new question too short to keep opens nothing and its answers are not attached."""
    names = names if names is not None else _call_names(turns)
    if section == _QA_SECTION:
        return _label_qa(turns, mgmt_names, names)[0]
    return _label_prepared(turns, section, names)


def split_call(paragraphs: Iterable[Mapping[str, object]]) -> CallSplit:
    """Split one call at the Q&A start. prepared_remarks = the cleaned prepared turns, qa = the
    cleaned MANAGEMENT ANSWER turns (no operator, no analyst text), both newline-joined; turns =
    labelled prepared turns then labelled Q&A turns (management names for the Q&A roles come from the
    prepared speakers). Status: empty (no turn), no_qa (no Q&A start found; the whole call is
    prepared), no_prepared (no cleaned prepared text before the Q&A start), ok."""
    turns = clean_paragraphs(paragraphs)
    if not turns:
        return CallSplit("", "", [], "empty")
    k = qa_start(turns)
    head, tail = (turns, []) if k is None else (turns[:k], turns[k:])
    tail_announced = _announced(tail)
    names = _call_names(turns, _announced(head) | tail_announced)
    prep_turns = _label_prepared(head, _PREP_SECTION, names)
    mgmt = {t["person"].strip().lower() for t in prep_turns if t["person"]}
    qa_turns, answers = _label_qa(tail, mgmt, names, tail_announced)
    prepared = "\n".join(t["text"] for t in prep_turns)
    status: SplitStatus = "no_qa" if k is None else "no_prepared" if not prepared else "ok"
    return CallSplit(prepared, "\n".join(answers), prep_turns + qa_turns, status)
