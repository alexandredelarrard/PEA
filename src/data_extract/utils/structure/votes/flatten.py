"""
flatten.py  (src/data_extract/utils/structure/votes/flatten.py)
----------------------------------------------------------------
A filled `Item507Extract` -> `sec_8k_votes` rows: one row per proposal, with a director
election collapsed to a single row carrying per-role-category vote columns.

`_prepare_frame` here is NOT the same function as `def14a/flatten.py`'s: that one takes
`(rows, numeric, pk)`, this one takes `(rows)` and keys on `Tables.sec_8k_votes.pk`.
Unifying them would be a logic change, not a move.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from src.data_extract.utils.common.frame_sanitize import strip_nul
from src.data_extract.utils.schemas.vote_schema import (
    Item507Extract, PROPOSAL_TYPES, VOTE_STANDARDS,
)
from src.data_extract.utils.structure.def14a.gender import person_key
from src.data_extract.utils.structure.def14a.validate import clean_person_name, clean_text
from src.data_extract.utils.structure.votes.guard import (
    _grounded_nominees, _is_grounded, _VOTE_FIELDS, mentions_preliminary,
)
from src.data_store.schema import Tables

#: `non_employee`.
_ROLE_CATEGORIES = ("ceo", "exec_officer", "non_employee", "unmatched")

#: A nominee under 70% support is a governance event -- a genuine shareholder revolt
#: against a specific director -- so the count is a column rather than something a reader
#: has to recompute out of `nominee_votes_json`.
_LOW_SUPPORT_THRESHOLD = 0.70

#: Relative slack for `nominee_sum_matches`. Vote counts run to 10 significant figures
#: (Apple: 9,072,076,816), so an absolute tolerance is meaningless; 0.5% absorbs one
#: rounded cell without absorbing a permuted column.
_NOMINEE_SUM_RTOL = 0.005

#: Words a filer wraps around its own proposal label, stripped repeatedly so
#: "Stockholder Proposal #1" sheds both. Measured over the 302 distinct labels stored:
#: bare digits, `9.`, `4b`, `7.2`, `(a)`..`(i)`, `(1)`..`(3)`, `Item 1`, `Item 1.`,
#: `Proposal 1`, `Proposal No. 1`, `Proposal No.1`, `Proposal One`..`Proposal Seven`,
#: `Stockholder Proposal #1`, and the glued-on-title forms handled below.
_LABEL_NOISE_RE = re.compile(
    r"^(?:proposals|proposal|stockholders|stockholder|shareholders|shareholder|items|item|"
    r"matters|matter|no)\b\.?\s*")
_LABEL_PUNCT = "().,;:#–—- "

#: `4a` and `4b` are DIFFERENT proposals (Abbott 2021 split one bye-law amendment in two;
#: Accenture 2016 ran `7A/7B/8A/8B`, and one filer numbers `7.2`/`8.1`), and `(a)`..`(i)`
#: is another filer's whole numbering scheme. So the sub-item marker is KEPT and this
#: column stays TEXT: forcing a bare integer would either collide the two halves of a
#: split item or null every letter-only row.
_NUMBERED_LABEL_RE = re.compile(r"^(\d{1,2})\s*(?:\.(\d{1,2})|\.?\s*([a-z]))?$")
_LETTER_LABEL_RE = re.compile(r"^([a-z])$")

#: What a filer puts between its number and the title it glued on: `Proposal 1 - Election
#: of Directors`, `Proposal 3: Advisory Vote ...`, `Proposal 6– Shareholder proposal`.
_LABEL_TITLE_SEP_RE = re.compile(r"\s*[-–—:]\s*")

_NUMBER_WORDS = {
    w: str(i) for i, w in enumerate(
        ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve"), start=1)
}
_NUMBER_WORDS.update({
    w: str(i) for i, w in enumerate(
        ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
         "ninth", "tenth", "eleventh", "twelfth"), start=1)
})


def _as_ordinal(label: str) -> str | None:
    """`label` -> its canonical token, or None when it is not a bare ordinal on its own."""
    label = label.strip(_LABEL_PUNCT).lower()
    while True:
        shorter = _LABEL_NOISE_RE.sub("", label).strip(_LABEL_PUNCT)
        if shorter == label:
            break
        label = shorter
    if not label:
        return None
    numbered = _NUMBERED_LABEL_RE.match(label)
    if numbered:
        number, sub_number, sub_letter = numbered.groups()
        # `4.a` and `4b` are the same shape and must canonicalise alike; `7.2` cannot lose
        # its dot, since `72` is a different proposal.
        return number + (f".{sub_number}" if sub_number else (sub_letter or ""))
    if label in _NUMBER_WORDS:
        return _NUMBER_WORDS[label]
    return label if _LETTER_LABEL_RE.match(label) else None


def _clean_proposal_number(value: object) -> str | None:
    """The filer's own proposal label, normalised to a comparable token.

    One filing prints `Proposal No. 4`, the next `Item 4.`, the next `Proposal Four`, the
    next `Proposal 4 - Advisory Vote on Executive Compensation`, and all four mean the same
    thing: 302 distinct strings for what is at most a dozen concepts, which this collapses
    to 113 tokens. The number is read AS PRINTED by the model and canonicalised here, not
    in the prompt — the model's job is to report what the page says, not to tidy it.

    NOT redundant with `proposal_seq`. Measured over 6,828 rows / 1,413 filings / 94
    tickers: 4,113 normalise to the same value, 1,955 are null (the filing numbers
    nothing), 330 are a sub-item or a range with no single number to compare, 111 are
    letter-only, and 319 DISAGREE outright. The disagreements are informative, not noise:

      * A filer that numbers every director SEPARATELY. Carnival's 2023 meeting elects 11
        directors as proposals 1-11 and we collapse an election to ONE row, so its own
        numbering runs 12..22 against our 2..12 — a constant offset on 8 CCL filings, and
        the only place the filer's numbering survives that deliberate collapse.
      * A proposal that was numbered but never voted. Apple's 2013 item 2 "was withdrawn
        and no vote was taken" and its 2012 item 6 was never presented, so the filer's
        numbering skips where our gap-free ordinal structurally cannot.
      * A label that is not unique within the filing at all: Arch Capital's 2010 8-K puts
        TWO separate elections under one `Item 1`.

    An unrecognised shape is returned cleaned but otherwise untouched — a label this does
    not understand is source, not noise, and must not be silently dropped.
    """
    cleaned = clean_text(value)
    if cleaned is None:
        return None
    direct = _as_ordinal(cleaned)
    if direct:
        return direct

    # A glued-on title is dropped, but only when what follows the separator is PROSE. When
    # it is another ordinal the label is a RANGE — `Proposals 1 - 10`, `Proposals 1(a)-(g)`
    # — and that row aggregates every one of them, so reducing it to its first would
    # misstate what was voted on. Such a label is left verbatim.
    parts = _LABEL_TITLE_SEP_RE.split(cleaned, maxsplit=1)
    if len(parts) == 2:
        head, tail = _as_ordinal(parts[0]), _as_ordinal(parts[1])
        if head and not tail:
            return head
    return cleaned



_DIRECTOR_COLS = tuple(
    [f"n_nominees_{c}" for c in _ROLE_CATEGORIES]
    + [f"{f}_{c}" for c in _ROLE_CATEGORIES for f in _VOTE_FIELDS]
    + ["exec_officer_titles", "n_nominees", "min_support_pct", "min_support_name",
       "n_nominees_below_70pct"]
)

#: Columns coerced to numeric before the save. Must cover every vote column: a `None` in
#: an otherwise-float column arrives as `object` dtype and Postgres would take the column
#: as TEXT on a first-run CREATE.
_NUMERIC_COLS = (
    ("proposal_seq", "is_amendment", "mentions_preliminary", "is_preliminary_stated",
     "nominee_sum_matches")
    + _VOTE_FIELDS
    + tuple(c for c in _DIRECTOR_COLS if c not in ("exec_officer_titles", "min_support_name"))
)


def _director_columns(nominees: list[dict], roles: dict[str, str],
                      titles: dict[str, str]) -> dict:
    """The 20 per-category vote columns + the support summary, for one election row.

    NULL on every non-director row (~85% of the table). That is the shape you asked for:
    columns rather than an extra grain, and a director election is exactly one row per
    meeting, so the sparsity is bounded and known rather than open-ended.

    Returned in `_DIRECTOR_COLS` order, which is the same order the NULL fill for a
    non-election row uses. Without that, a frame's column ORDER would depend on whether
    its first proposal happened to be an election -- and on a first-run `CREATE TABLE`
    the frame's column order IS the DDL's.
    """
    row: dict = {}
    for cat in _ROLE_CATEGORIES:
        bucket = [n for n in nominees if roles.get(person_key(n["name"]), "unmatched") == cat]
        row[f"n_nominees_{cat}"] = float(len(bucket))
        for f in _VOTE_FIELDS:
            vals = [n[f] for n in bucket if n.get(f) is not None]
            row[f"{f}_{cat}"] = float(sum(vals)) if vals else None

    exec_titles = sorted({titles[k] for k in
                          (person_key(n["name"]) for n in nominees)
                          if k in titles and roles.get(k) == "exec_officer"})
    row["exec_officer_titles"] = " | ".join(exec_titles) or None
    row["n_nominees"] = float(len(nominees))

    support = [(n["votes_for"] / (n["votes_for"] + n["votes_against"]), n["name"])
               for n in nominees
               if n.get("votes_for") is not None and n.get("votes_against") is not None
               and (n["votes_for"] + n["votes_against"]) > 0]
    if support:
        worst = min(support)
        row["min_support_pct"] = float(worst[0])
        row["min_support_name"] = worst[1]
        row["n_nominees_below_70pct"] = float(
            sum(1 for pct, _ in support if pct < _LOW_SUPPORT_THRESHOLD))
    else:
        row["min_support_pct"] = None
        row["min_support_name"] = None
        row["n_nominees_below_70pct"] = None
    return {c: row[c] for c in _DIRECTOR_COLS}


def _nominee_sum_matches(nominees: list[dict]) -> float | None:
    """FLAG (never a filter): do the per-nominee totals agree with each other?

    Each nominee at one meeting faces the same shares-represented pool, so every
    nominee's four columns should sum to the same number. Computable on 96% of filings
    and holds on 91% -- which makes it a useful monitor and a bad gate, so it is stored
    and never acted on. None when fewer than two nominees carry a complete set.
    """
    totals = [sum(n[f] for f in _VOTE_FIELDS)
              for n in nominees
              if all(n.get(f) is not None for f in _VOTE_FIELDS)]
    if len(totals) < 2:
        return None
    lo, hi = min(totals), max(totals)
    return 1.0 if hi - lo <= _NOMINEE_SUM_RTOL * hi else 0.0


def _proposal_rows(ticker: str, filing: pd.Series, extract: Item507Extract, text: str,
                   roles: dict[str, str],
                   titles: dict[str, str]) -> tuple[list[dict], int]:
    """`(rows, guard rejections)` for one filing. One row per proposal, `proposal_seq`
    1-based in the filing's own order.

    `proposal_seq` is assigned over the KEPT rows, so it is gap-free and can never be
    null -- it is part of the primary key, and a NULL in a Postgres PK column aborts the
    whole insert rather than the one row.
    """
    rows: list[dict] = []
    rejected = 0
    preliminary = mentions_preliminary(text)
    for p in extract.proposals:
        description = clean_text(p.description)
        if not description:
            rejected += 1
            continue
        ptype = (p.proposal_type or "").strip().lower()
        if ptype not in PROPOSAL_TYPES:
            ptype = None
        nominees, n_bad = _grounded_nominees(p, text)
        rejected += n_bad

        votes = {f: getattr(p, f) for f in _VOTE_FIELDS}
        # `is_election` is `director_election` and NOTHING else, even though the frequency
        # vote below also arrives as a list of lines: an election is what earns a role map
        # and a `min_support_pct`, and computing either for a line called "One Year" would
        # be nonsense. The two types share `nominee_votes_json` and part company here.
        is_election = ptype == "director_election"
        is_bucket_vote = ptype == "say_on_pay_frequency"
        if is_election and not nominees:
            # an election whose every nominee failed the guard carries no tally at all
            rejected += 1
            continue
        if not is_election and not (_is_grounded(votes, text) or nominees):
            # nothing printed in the source backs this row -> it was written, not read
            rejected += 1
            continue

        # A frequency table is `1 Year | 2 Years | 3 Years | Abstain | Broker Non-Votes`
        # (Agilent writes `Every 1 Year`), so only its last two columns are proposal-level
        # -- the year buckets are lines and belong in `nominee_votes_json` with the
        # election's. Enforced HERE and not left to the prompt because the failure it
        # prevents is a column PERMUTATION, and the fabrication guard is blind to one: the
        # shifted numbers are all genuinely printed in the source, so every grounding test
        # passes while `2 Years` sits in `votes_against` reading as opposition.
        if is_election:
            proposal_votes = {f: None for f in _VOTE_FIELDS}
        elif is_bucket_vote and nominees:
            proposal_votes = {**votes, "votes_for": None, "votes_against": None}
        elif is_bucket_vote:
            # buckets absent -> the remaining counts cannot be labelled. Which column of
            # the source a lone number came from is exactly what is unknowable here, so
            # the row keeps its type, description and meeting and stores NO tally; the
            # loss stays countable as `nominee_votes_json IS NULL` on a frequency row.
            proposal_votes = {f: None for f in _VOTE_FIELDS}
        else:
            proposal_votes = dict(votes)

        standard = (p.vote_standard or "").strip().lower()
        row = {
            "ticker": ticker,
            "accession_number": filing["accession_number"],
            "proposal_seq": float(len(rows) + 1),
            "cik": filing.get("cik"),
            "form": clean_text(filing.get("form")),
            "filing_date": filing.get("filing_date"),
            "period_of_report": filing.get("period_of_report"),
            "meeting_date": clean_text(extract.meeting_date),
            "is_amendment": filing.get("is_amendment"),
            "mentions_preliminary": 1.0 if preliminary else 0.0,
            "is_preliminary_stated": (None if extract.is_preliminary is None
                                      else float(bool(extract.is_preliminary))),
            "proposal_number": _clean_proposal_number(p.proposal_number),
            "proposal_type": ptype,
            "description": description,
            "vote_standard": standard if standard in VOTE_STANDARDS else None,
            **proposal_votes,
            "nominee_sum_matches": (_nominee_sum_matches(nominees) if is_election
                                    else None),
            "nominee_votes_json": json.dumps(nominees) if nominees else None,
        }
        # The 20 category columns exist on every row so the DDL is one shape; they are
        # NULL off an election row rather than 0, because "no nominees in this bucket"
        # and "this proposal has no nominees" are different statements.
        row.update(_director_columns(nominees, roles, titles) if is_election
                   else {c: None for c in _DIRECTOR_COLS})
        rows.append(row)
    return rows, rejected


#: Concurrent LLM calls, for the same measured reason as the DEF 14A path: the work is pure
#: network wait on an API that accepts parallel requests, and a serial universe run is not
#: finishable. An Item 5.07 narrative is far smaller than a proxy (the longest in the
#: 333-filing baseline is 17,032 chars), so the per-call latency is lower -- but there are

def _prepare_frame(rows: list[dict]) -> pd.DataFrame:
    """Rows -> a save-ready frame: numeric columns coerced, NULs stripped, duplicates
    collapsed on the table's own primary key."""
    df = pd.DataFrame(rows)
    for c in _NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = strip_nul(df)                          # Postgres TEXT rejects NUL (\x00)
    for c in ("filing_date", "period_of_report", "meeting_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df.drop_duplicates(subset=list(Tables.sec_8k_votes.pk), keep="last")

