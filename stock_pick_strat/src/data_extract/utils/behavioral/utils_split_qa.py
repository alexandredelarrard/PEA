import re 

# the operator's phrase that opens the analyst Q&A (splits prepared remarks from Q&A).
# Broadened across 2005-2025 transcript phrasings: the classic "question-and-answer session",
# operator-instructions, "(first) question comes/is/will be from" hand-off, "go to the line of X",
# "for our first question, we'll go/take", "we'll (now) take/go to our first question", the generic
# "we'll now begin/open/take/turn ... to questions", and "open the floor/line for questions".
_QA_MARKER = re.compile(
    r"(?i)"
    r"question[-\s]and[-\s]answer session|"
    r"questions?\s*(?:and|&)\s*answers?|"                 # standalone Q&A heading (no 'session')
    r"(?:first|next|final)\s+question\s+(?:comes?|is|will\s+come|will\s+be)\s+from|"
    r"(?:go|turn|move)\s+(?:ahead\s+)?to\s+(?:the\s+)?line\s+of\b|"          # "go to the line of X"
    r"(?:for\s+)?(?:our|your)\s+(?:first|next)\s+question,?\s+"              # "for our first question, we'll go"
    r"(?:we(?:'ll| will)|let's|i(?:'ll| will)|please)\b|"
    r"we(?:'ll| will)\s+(?:now\s+)?(?:go|move|turn|take)\s+(?:ahead\s+)?to\s+"  # "we'll take our first question"
    r"(?:our\s+)?(?:first|next)\s+(?:question|caller|line)|"
    r"(?:we(?:'ll| will| are going to| would like to)|now|let's|i(?:'ll| will))\b"
    r"[^.]{0,45}?(?:begin|open|take|start|conduct|move\s+to|go\s+to|turn[^.]{0,20}?to)"
    r"[^.]{0,30}?questions?|"
    r"open\s+(?:up\s+)?(?:the\s+)?(?:floor|line|lines|call)\b[^.]{0,25}?questions?|"
    r"\[?operator instructions\]?")


def split_prepared_qa(text: str) -> dict[str, str]:
    """Source-agnostic split of a full transcript TEXT into the high-signal sections funds
    analyse: `full` (ALWAYS kept -> format-proof), `prepared_remarks` (scripted management
    comments from call-open to the Q&A) and `qa` (the analyst Q&A, after the operator's
    hand-off). Used for BOTH the Motley Fool HTML-extracted text AND the HuggingFace
    `content` field. Call prose starts at the first 'Operator' line (skips logo/date/
    takeaways preamble); the Q&A hand-off is searched past the first ~2000 chars first
    (operators PREVIEW the Q&A in the intro), then anywhere. If NO phrase matches, fall back to the
    SECOND 'Operator' turn (the first opens the call, the second hands off to the Q&A) so a call with
    an unusual hand-off phrasing still yields a Q&A section. prepared/qa only when confidently split
    (>300 chars each)."""
    
    out: dict[str, str] = {"full": text}
    op = re.search(r"(?im)^\s*operator\b", text)
    prose = text[op.start():] if op else text
    m = _QA_MARKER.search(prose, 2000) or _QA_MARKER.search(prose)
    if m is None:                                        # no phrase matched -> second 'Operator' turn
        ops = list(re.finditer(r"(?im)^\s*operator\b", prose))
        m = ops[1] if len(ops) >= 2 else None
    if m:
        pre, post = prose[:m.start()].strip(), prose[m.start():].strip()
        if len(pre) > 300:
            out["prepared_remarks"] = pre
        if len(post) > 300:
            out["qa"] = post
    elif op:
        out["prepared_remarks"] = prose.strip()          # no Q&A hand-off found -> all remarks
    return out