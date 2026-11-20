"""Flat vote-label, director-summary and row operations; the ticker worker owns their sequencing."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from src.data_extract.utils.structure.votes.guard import _VOTE_FIELDS
from src.data_extract.utils.structure.votes.vote_schema import VOTE_STANDARDS, Item507Extract
from src.utils.names import person_key
from src.utils.string import clean_text

_ROLE_CATEGORIES = ("ceo", "exec_officer", "non_employee", "unmatched")
_LOW_SUPPORT_THRESHOLD = 0.70
_NOMINEE_SUM_RTOL = 0.005
_LABEL_NOISE_RE = re.compile(r"^(?:proposals|proposal|stockholders|stockholder|shareholders|shareholder|items|item|matters|matter|no)\b\.?\s*")
_LABEL_PUNCT = "().,;:#–—- "
_NUMBERED_LABEL_RE = re.compile(r"^(\d{1,2})\s*(?:\.(\d{1,2})|\.?\s*([a-z]))?$")
_LETTER_LABEL_RE = re.compile(r"^([a-z])$")
_LABEL_TITLE_SEP_RE = re.compile(r"\s*[-–—:]\s*")
_NUMBER_WORDS = {
    word: str(i) for i, word in enumerate(("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"), 1)
}
_NUMBER_WORDS.update(
    {
        word: str(i)
        for i, word in enumerate(
            ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth"), 1
        )
    }
)
_DIRECTOR_COLS = tuple(
    [f"n_nominees_{category}" for category in _ROLE_CATEGORIES]
    + [f"{field}_{category}" for category in _ROLE_CATEGORIES for field in _VOTE_FIELDS]
    + ["exec_officer_titles", "n_nominees", "min_support_pct", "min_support_name", "n_nominees_below_70pct"]
)
_NUMERIC_COLS = (
    ("proposal_seq", "is_amendment", "mentions_preliminary", "is_preliminary_stated", "nominee_sum_matches")
    + _VOTE_FIELDS
    + tuple(column for column in _DIRECTOR_COLS if column not in ("exec_officer_titles", "min_support_name"))
)


def _clean_proposal_number(value: object) -> str | None:
    """Normalize a printed ordinal/title label while retaining split items, ranges and unknown labels."""
    # Normalize the whole label and possible title-separated parts with the same ordinal rules.
    cleaned = clean_text(value)
    if cleaned is None:
        return None
    labels = [cleaned, *_LABEL_TITLE_SEP_RE.split(cleaned, maxsplit=1)]
    ordinals: list[str | None] = []
    for label in labels:
        label = label.strip(_LABEL_PUNCT).lower()
        previous = None
        while label != previous:
            previous = label
            label = _LABEL_NOISE_RE.sub("", label).strip(_LABEL_PUNCT)
        match = _NUMBERED_LABEL_RE.match(label)
        if match:
            number, sub_number, sub_letter = match.groups()
            ordinals.append(number + (f".{sub_number}" if sub_number else sub_letter or ""))
        else:
            ordinals.append(_NUMBER_WORDS.get(label, label if _LETTER_LABEL_RE.match(label) else None))

    # A numeric tail is a range, not a prose title that can be discarded.
    if ordinals[0]:
        return ordinals[0]
    if len(ordinals) == 3 and ordinals[1] and not ordinals[2]:
        return ordinals[1]
    return cleaned


def _director_columns(nominees: list[dict[str, Any]], roles: dict[str, str], titles: dict[str, str]) -> dict[str, Any]:
    """Per-role totals, support summary and non-filtering sum flag for one grounded election."""
    # Preserve role precedence and NULL when a category has no disclosed count.
    row: dict[str, Any] = {}
    for category in _ROLE_CATEGORIES:
        bucket = [nominee for nominee in nominees if roles.get(person_key(nominee["name"]) or "", "unmatched") == category]
        row[f"n_nominees_{category}"] = float(len(bucket))
        for field in _VOTE_FIELDS:
            values = [nominee[field] for nominee in bucket if nominee.get(field) is not None]
            row[f"{field}_{category}"] = float(sum(values)) if values else None

    # Summarize grounded nominees without inventing proposal-level totals.
    exec_titles = sorted(
        {
            titles[key]
            for key in (person_key(nominee["name"]) for nominee in nominees)
            if key is not None and key in titles and roles.get(key) == "exec_officer"
        }
    )
    row["exec_officer_titles"] = " | ".join(exec_titles) or None
    row["n_nominees"] = float(len(nominees))
    support = [
        (nominee["votes_for"] / (nominee["votes_for"] + nominee["votes_against"]), nominee["name"])
        for nominee in nominees
        if nominee.get("votes_for") is not None and nominee.get("votes_against") is not None and nominee["votes_for"] + nominee["votes_against"] > 0
    ]
    worst = min(support) if support else (None, None)
    row["min_support_pct"], row["min_support_name"] = worst
    row["n_nominees_below_70pct"] = float(sum(pct < _LOW_SUPPORT_THRESHOLD for pct, _ in support)) if support else None
    totals = [sum(nominee[field] for field in _VOTE_FIELDS) for nominee in nominees if all(nominee.get(field) is not None for field in _VOTE_FIELDS)]
    row["nominee_sum_matches"] = float(max(totals) - min(totals) <= _NOMINEE_SUM_RTOL * max(totals)) if len(totals) >= 2 else None
    return {column: row[column] for column in (*_DIRECTOR_COLS, "nominee_sum_matches")}


def _proposal_rows(
    ticker: str, filing: pd.Series, extract: Item507Extract, prepared: list[dict[str, Any]], counts: set[int], preliminary: bool
) -> tuple[list[dict[str, Any]], int]:
    """Emit source-grounded rows from prepared facts; sequence only the kept proposals."""
    # Reject empty descriptions and ungrounded matters before assigning gap-free sequence values.
    rows: list[dict[str, Any]] = []
    rejected = 0
    for proposal in prepared:
        description = clean_text(proposal["description"])
        if not description:
            rejected += 1
            continue
        rejected += proposal["rejected"]
        nominees = proposal["nominees"]
        votes = {field: proposal[field] for field in _VOTE_FIELDS}
        is_election = proposal["proposal_type"] == "director_election"
        is_bucket_vote = proposal["proposal_type"] == "say_on_pay_frequency"
        grounded = any(int(round(float(value))) in counts for value in votes.values() if value is not None and pd.notna(value))
        if is_election and not nominees or not is_election and not (grounded or nominees):
            rejected += 1
            continue
        if is_election or is_bucket_vote and not nominees:
            proposal_votes: dict[str, Any] = dict.fromkeys(_VOTE_FIELDS)
        elif is_bucket_vote:
            proposal_votes = {**votes, "votes_for": None, "votes_against": None}
        else:
            proposal_votes = votes
        standard = (proposal["vote_standard"] or "").strip().lower()
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
            "mentions_preliminary": float(preliminary),
            "is_preliminary_stated": None if extract.is_preliminary is None else float(bool(extract.is_preliminary)),
            "proposal_number": proposal["proposal_number"],
            "proposal_type": proposal["proposal_type"],
            "description": description,
            "vote_standard": standard if standard in VOTE_STANDARDS else None,
            **proposal_votes,
            "nominee_sum_matches": None,
            "nominee_votes_json": json.dumps(nominees) if nominees else None,
        }
        row.update(proposal["director"] if is_election else dict.fromkeys(_DIRECTOR_COLS))
        rows.append(row)
    return rows, rejected
