"""
fetch_8k_votes_llm.py  (src/data_extract/utils/structure/fetch_8k_votes_llm.py)
--------------------------------------------------------------------------------
Parse the Form 8-K **Item 5.07** narratives ALREADY STORED in `sec_8k.item_text` into
`sec_8k_votes` — one row per proposal, with a director election collapsed to one row
carrying per-role-category vote columns.

No new download. `fetch_8k_edgar` already tags item 5.07 as high-signal and stores the
narrative: 6,657 rows over 405 tickers, 2010-03-01 -> 2026-08-19, 99.2% with `item_text`
longer than 200 chars, and nothing has ever parsed them. `item_text` carries zero HTML
tags but IS edgartools' rendered view, so column alignment survives (84.6% carry a
`U+2500` rule under the header, the rest are whitespace-aligned, one filing row per
text line) — which is why the model can read the tables out of plain text.

**There is no validation gate for a vote table, and this module does not pretend to
build one.** A vote table prints no independent total, and the dominant extraction error
is a column permutation, which is INVARIANT under sums. Measured on 57 filings: a
"total <= shares outstanding" bound is computable on only 9 (16%); per-nominee totals are
computable on 96% and hold on 91%; meeting-level totals are computable on 100% and hold
on 81%. Combining all three gives recall 0.56 / precision 0.56 — **7 of 16 known-bad
filings pass clean**. So the per-nominee sum lands as a FLAG (`nominee_sum_matches`), a
monitor and not a filter. Prior-year say-on-pay in the proxy narrative is not an
independent check either: it is lossy (Merck says "approximately 94%" where its own 8-K
says 93.50%) and the denominator convention varies by state of incorporation (XOM notes
New Jersey excludes abstentions from votes cast).

What replaces a gate is two hard rules, both of them the LLM's own measured failure modes:

1. **The fabrication guard.** The model invented an entire table ("John Doe" / "Jane
   Smith", 250,000,000 votes) for a filing whose `item_text` was truncated, and a fake
   row for another. So: emit nothing unless the text is long enough AND contains a
   comma-grouped number, and drop any nominee whose name is not in the source or whose
   numbers are all absent from it. See `rejection_reason` / `_grounded_nominees`.
2. **Never "latest wins" on an amendment.** Of 190 multi-filing meetings, 135 (71%) have
   an amendment carrying no vote numbers at all, so "latest wins" is right on 33/190
   (17%) while "union the group" is right on 173/190 (91%). Every filing's rows are
   therefore stored under its own accession and NOTHING is deduped across accessions; the
   reader unions on `(ticker, period_of_report)`. The trap worth naming: **in a contested
   election the FIRST 8-K is the preliminary one** — DIS `0000950157-24-000595` states
   "estimated preliminary voting results ... do not include shares voted on the blue
   proxy card distributed by Trian" and the 8-K/A carries the Inspector of Election's
   final results, switching from Against to Withhold on the way.

Role categorisation joins each nominee to the nearest prior proxy (`def14a_llm`,
`def14a_executive_comp`, `def14a_director_comp`) on the same `lastname|firstinitial` key
the DEF 14A gender consensus uses, so a nominee that matches there matches here. The
per-filing `unmatched` count is stored, not swallowed: that number IS the join's error
rate and it has to stay visible.

Accepted losses, recorded rather than repaired:
  * **62 rows corpus-wide (1.0%)** store a truncated `item_text` because the filer
    numbered its proposals `Item 1.` / `Item 2.` and edgartools' item splitter cut there
    (HWM 2019 stores 508 chars where the live filing has 10,922 and four tables). We do not
    re-fetch the HTML to repair them. Note what the LENGTH floor can and cannot do here:
    the shortest GENUINE tally in the baseline is 554 chars, so a floor high enough to
    reject a 508-char stub would destroy two correct filings. A stub cut before the tables
    is caught because it carries no share count, not because it is short.
  * **8.8% of Item 5.07 filings carry no vote table at all** — a 5.07(d) board-response
    filing discloses only the board's reaction to a say-on-pay frequency vote. Zero rows
    is a NORMAL outcome here, logged as such.
  * **9 of 190 multi-filing meetings (4.7%)** give no signal beyond the form type as to
    which filing supersedes which; only 14 filings corpus-wide say "preliminary", so
    `mentions_preliminary` resolves 8 of the 17 genuine restatements and no more.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from src.context import Context
from src.data_extract.utils.common.frame_sanitize import strip_nul
from src.data_extract.utils.common.llm_extractor import LLMExtractor
from src.data_extract.utils.common.sec_utils import existing_filings
from src.data_extract.utils.structure.def14a_gender import person_key
from src.data_extract.utils.structure.def14a_validate import clean_person_name, clean_text
from src.data_extract.utils.schemas.vote_schema import (
    Item507Extract, PROPOSAL_TYPES, ProposalVote, VOTE_STANDARDS,
)
from src.data_store.schema import Tables

logger = logging.getLogger(__name__)

#: The 8-K item code this module reads. Anything else in `sec_8k` is somebody else's row.
_ITEM = "5.07"

#: Columns read out of `sec_8k`. A narrow projection is mandatory (AGENTS.md): the table
#: holds ~197k item rows and `item_text` is the widest column in it, so an unprojected
#: read would pull every item code's narrative to select 3.4% of them.
_SOURCE_COLS = ("ticker", "cik", "accession_number", "form", "filing_date",
                "period_of_report", "is_amendment", "item_text")

#: Vote tallies in these filings are SHARE COUNTS and are always printed comma-grouped.
#: An Item 5.07 narrative with no comma-grouped number anywhere therefore has no table to
#: read, and asking the model to read one is how "John Doe / 250,000,000" gets invented.
_GROUPED_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+")

#: Floor on `item_text` length. It catches the emptiest stubs and NOT MUCH MORE, and the
#: reason is measured: a truncation stub is not reliably shorter than a real filing. The
#: shortest GENUINE tally in the 333-filing baseline is 554 chars -- TDG's 2014 and 2019
#: special meetings, each a complete `FOR / AGAINST / ABSTAIN` block on a single stock-option
#: plan -- while HWM 2019's truncated stub is 508. So a floor high enough to reject HWM would
#: destroy two correct filings, and this rule cannot be the one that catches a truncation.
#: What actually catches those is the comma-grouped-number test plus the per-row grounding
#: check: a stub that was cut before the tables carries no share count to read.
_MIN_ITEM_TEXT_CHARS = 400

#: A filing that says its own numbers are provisional. Only 14 filings corpus-wide do, so
#: this is a weak signal by construction -- it resolves 8 of the 17 genuine restatements
#: and is stored as a flag rather than used to choose between filings.
_PRELIMINARY_RE = re.compile(
    r"(?i)\b(preliminary|estimated\s+(?:preliminary\s+)?voting|"
    r"subject\s+to\s+(?:final\s+)?certification|not\s+yet\s+certified)\b")

#: The four role buckets a nominee lands in. `unmatched` is a bucket, not an error: it is
#: how the join's failure rate stays measurable instead of being silently folded into
#: `non_employee`.
_ROLE_CATEGORIES = ("ceo", "exec_officer", "non_employee", "unmatched")

#: The four vote columns, at proposal level and per role category.
_VOTE_FIELDS = ("votes_for", "votes_against", "votes_abstain", "votes_broker_non_votes")

#: A nominee under 70% support is a governance event -- a genuine shareholder revolt
#: against a specific director -- so the count is a column rather than something a reader
#: has to recompute out of `nominee_votes_json`.
_LOW_SUPPORT_THRESHOLD = 0.70

#: Relative slack for `nominee_sum_matches`. Vote counts run to 10 significant figures
#: (Apple: 9,072,076,816), so an absolute tolerance is meaningless; 0.5% absorbs one
#: rounded cell without absorbing a permuted column.
_NOMINEE_SUM_RTOL = 0.005

_VOTES_PROMPT = (
    "You are reading Item 5.07 of a Form 8-K: the certified results of a shareholder "
    "meeting. Extract every matter voted on and its vote counts.\n"
    "\n"
    "READ THE NUMBERS OFF THE PAGE. Every value you return must appear in the text. Do "
    "not compute, sum, average or infer any vote count. If a number is not printed, "
    "return null for it.\n"
    "\n"
    "The numbers are SHARE COUNTS, not percentages. Many filings also print a '% For' / "
    "'% Against' column, or stack a percentage row inside the table -- ignore every "
    "percentage. Some filings TRANSPOSE a non-director proposal into label/value lines "
    "('Votes Cast For: | 3,495,486,371 | 96.8 %'); read the share count, drop the "
    "percentage. Round fractional votes to whole shares.\n"
    "\n"
    "'WITHHELD' (or 'Withhold') is the column some filers use INSTEAD OF 'Against' in a "
    "director election. Put its count in `votes_against` and set `vote_standard` to "
    "'withheld'. It is NEVER the broker-non-vote column. Broker non-votes are printed as "
    "'Broker Non-Votes' or 'Non-Votes'; when the filing omits that column or prints "
    "'N/A', return null -- return 0 only when it prints a zero.\n"
    "\n"
    "A director election is ONE proposal with one entry per nominee in `nominees`. Leave "
    "the proposal-level vote fields null for it; do not add the nominees up.\n"
    "\n"
    "Some Item 5.07 filings report no tallies at all -- they only state the board's "
    "response to an earlier say-on-pay frequency vote. Return an empty `proposals` list "
    "for those. Never invent a table, a nominee or a number that is not in the text."
)


# --------------------------------------------------------------------------- #
# Rule 1 -- the fabrication guard                                              #
# --------------------------------------------------------------------------- #
def has_vote_numbers(text: str | None) -> bool:
    """True when `item_text` contains at least one comma-grouped number.

    Vote tallies are share counts and are always comma-grouped in these filings, so this
    is the cheapest test for "is there a table in here at all". Verified on the baseline:
    the genuine tally-free 5.07(d) board-response filings (NKE 2011/2017, BA 2011,
    TDG 2011/2014/2017/2019) contain none.
    """
    return bool(text) and bool(_GROUPED_NUMBER_RE.search(text))


def mentions_preliminary(text: str | None) -> bool:
    """True when the filing itself flags its numbers as provisional."""
    return bool(text) and bool(_PRELIMINARY_RE.search(text))


def rejection_reason(text: str | None) -> str | None:
    """Why this `item_text` must not be sent to the LLM, or None to proceed.

    Returning a REASON rather than a bool is what lets the run log distinguish the two
    populations that both produce zero rows: a truncated narrative (a known 1.0% loss we
    accepted) and a genuinely tally-free 5.07(d) filing (8.8%, a correct outcome). A
    single "0 rows" counter would have hidden both inside each other.
    """
    if text is None or not str(text).strip():
        return "empty item_text"
    if len(text) < _MIN_ITEM_TEXT_CHARS:
        return f"item_text truncated ({len(text)} chars)"
    if not has_vote_numbers(text):
        return "no comma-grouped number -- no vote table"
    return None


def _norm_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


#: Name tokens too short or too generic to be evidence on their own. A middle initial appears
#: somewhere in any proxy, so counting it as a hit would make the token fallback vacuous.
_WEAK_NAME_TOKENS = frozenset({"jr", "sr", "ii", "iii", "iv", "de", "la", "van", "der", "von"})


def _name_in_source(name: str | None, text: str) -> bool:
    """True when `name` appears in the source, whitespace- and case-insensitively.

    A verbatim name check is a valid hallucination test on filing text: 99.74% of the
    `directors[]` names extracted from DEF 14A bodies appear verbatim in their source.
    A nominee name is printed in the election table it was read from, so a name that is
    absent was not read -- it was written.

    But a CONTIGUOUS check alone over-rejects, and measurably: edgartools renders a narrow name
    column by wrapping it, and the vote numbers land BETWEEN the two halves of the name. PTC's
    2010 filing prints

        Paul                       100,753,338     1,735,851     7,486,441
        A. Lacy

    so "Paul A. Lacy" is not a substring of the rendered text at any whitespace normalisation,
    and three CORRECTLY read nominees were being discarded. Hence the token fallback: every
    substantial token of the name must appear in the source, which survives the wrap while
    still refusing a name the document never mentions. It is weaker than contiguity by exactly
    one thing -- a fabricated name assembled from words present elsewhere in the document --
    and the fabrication actually measured ("John Doe" / "Jane Smith" on a 508-char stub) fails
    it, because neither surname is anywhere in that stub.
    """
    cleaned = clean_person_name(name)
    if not cleaned:
        return False
    haystack = _norm_space(text)
    if _norm_space(cleaned) in haystack:
        return True
    tokens = [t for t in re.findall(r"[a-z]+", cleaned.lower())
              if len(t) > 1 and t not in _WEAK_NAME_TOKENS]
    return bool(tokens) and all(t in haystack for t in tokens)


def _number_in_source(value: float | None, text: str) -> bool:
    """True when `value` is printed in the source, comma-grouped or bare."""
    if value is None or pd.isna(value):
        return False
    n = int(round(float(value)))
    return f"{n:,}" in text or (n < 1000 and str(n) in text)


def _is_grounded(votes: dict, text: str) -> bool:
    """True when at least ONE of a row's vote counts is printed in the source.

    Deliberately ANY and not ALL. All-must-match would reject a whole row over a single
    cell the model rounded -- the measured LLM error set includes fractional votes
    truncated 1000x -- while the failure this guard exists to catch is wholesale
    invention, where a fabricated row's numbers appear nowhere. A row with no numbers at
    all is not grounded either: that is the shape of an invented nominee.
    """
    return any(_number_in_source(votes.get(f), text) for f in _VOTE_FIELDS)


# --------------------------------------------------------------------------- #
# Role categorisation                                                          #
# --------------------------------------------------------------------------- #
def _role_source(context: Context, ticker: str) -> dict[str, pd.DataFrame]:
    """The three narrow proxy reads the role map needs, done ONCE per ticker.

    Per (ticker, meeting) would re-read the same rows ~17 times for a ticker with 17
    annual meetings. Each read is projected to the columns the map actually uses -- an
    unprojected read of these tables is forbidden (AGENTS.md) and would pull the whole
    `def14a_json` blob along with it.
    """
    def _read(table, columns: list[str]) -> pd.DataFrame:
        df = context.store.load(table, columns=columns, where={"ticker": ticker},
                                optional=True)
        return df if df is not None else pd.DataFrame(columns=columns)

    return {
        "ceo": _read(Tables.def14a_llm, ["ticker", "as_of", "ceo_name_proxy"]),
        "exec": _read(Tables.def14a_executive_comp,
                      ["ticker", "as_of", "name", "title", "fiscal_year"]),
        "director": _read(Tables.def14a_director_comp, ["ticker", "as_of", "name"]),
    }


def _latest_before(df: pd.DataFrame, meeting_date: object) -> pd.DataFrame:
    """The rows of the single NEAREST PRIOR proxy, i.e. the one whose roster was on the
    ballot. A proxy filed AFTER the meeting describes the board the meeting elected, so
    using it would categorise a nominee by the outcome of the vote being categorised."""
    if df.empty or "as_of" not in df.columns or meeting_date is None:
        return df.iloc[0:0]
    as_of = pd.to_datetime(df["as_of"], errors="coerce")
    prior = df[as_of <= pd.Timestamp(meeting_date)]
    if prior.empty:
        return prior
    return prior[pd.to_datetime(prior["as_of"], errors="coerce") == as_of[prior.index].max()]


def _role_map(source: dict[str, pd.DataFrame],
              meeting_date: object) -> tuple[dict[str, str], dict[str, str]]:
    """`(person_key -> role, person_key -> title)` from the nearest prior proxy.

    Precedence is CEO > executive officer > non-employee director, because a
    CEO-and-director appears in two of the three sources and the more specific role is
    the informative one. Item 402(k) membership IS the definition of a non-employee
    director, so `def14a_director_comp` needs no independent independence test.
    """
    roles: dict[str, str] = {}
    titles: dict[str, str] = {}

    for _, r in _latest_before(source["director"], meeting_date).iterrows():
        key = person_key(r.get("name"))
        if key:
            roles[key] = "non_employee"

    execs = _latest_before(source["exec"], meeting_date)
    if not execs.empty and "fiscal_year" in execs.columns:
        years = pd.to_numeric(execs["fiscal_year"], errors="coerce")
        if years.notna().any():
            execs = execs[years == years.max()]
    for _, r in execs.iterrows():
        key = person_key(r.get("name"))
        if key:
            roles[key] = "exec_officer"
            title = clean_text(r.get("title"))
            if title:
                titles[key] = title

    for _, r in _latest_before(source["ceo"], meeting_date).iterrows():
        key = person_key(r.get("ceo_name_proxy"))
        if key:
            roles[key] = "ceo"

    return roles, titles


#: Every column `_director_columns` produces. Named once so a non-election row can be
#: filled with the identical NULL key set -- a row that simply OMITS them would make the
#: frame's column set depend on whether the first filing had an election.
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


# --------------------------------------------------------------------------- #
# Flatten                                                                      #
# --------------------------------------------------------------------------- #
def _grounded_nominees(proposal: ProposalVote, text: str) -> tuple[list[dict], int]:
    """`(kept nominees, rejected count)` after the fabrication guard.

    Both halves of the guard apply per nominee: the name must be printed in the source,
    and at least one of the four counts must be too.
    """
    kept, rejected = [], 0
    for n in proposal.nominees:
        name = clean_person_name(n.name)
        votes = {f: getattr(n, f) for f in _VOTE_FIELDS}
        if not name or not _name_in_source(name, text) or not _is_grounded(votes, text):
            rejected += 1
            continue
        kept.append({"name": name, **votes})
    return kept, rejected


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
        # `director_election` and NOTHING else, even when a per-person list came back. A
        # say-on-pay FREQUENCY proposal is also tallied per label -- JPM 2017 prints
        # `One Year | Two Years | Three Years | Abstain | Broker Non-Votes` -- and treating
        # that as an election would compute a role map and a `min_support_pct` for a
        # bucket called "One Year". Those buckets land in `nominee_votes_json` instead
        # (25.4% of filings carry the frequency vote, 4 buckets), which is exactly where
        # the design puts them: a recategorisation out of the JSON is free.
        is_election = ptype == "director_election"
        if is_election and not nominees:
            # an election whose every nominee failed the guard carries no tally at all
            rejected += 1
            continue
        if not is_election and not (_is_grounded(votes, text) or nominees):
            # nothing printed in the source backs this row -> it was written, not read
            rejected += 1
            continue

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
            "proposal_number": clean_text(p.proposal_number),
            "proposal_type": ptype,
            "description": description,
            "vote_standard": standard if standard in VOTE_STANDARDS else None,
            **{f: (None if is_election else votes[f]) for f in _VOTE_FIELDS},
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


#: Concurrent LLM calls per ticker, for the same measured reason as the DEF 14A path: the work
#: is pure network wait on an API that accepts parallel requests, and a serial universe run is
#: not finishable. An Item 5.07 narrative is far smaller than a proxy (the longest in the
#: 333-filing baseline is 17,032 chars), so the per-call latency is lower -- but there are 6,657
#: of them.
_LLM_WORKERS = 12

_T = TypeVar("_T")


def _map_filings(fn: Callable[[pd.Series], _T], filings: list[pd.Series],
                 workers: int) -> Iterable[_T]:
    """`fn` over `filings`, in order, `workers` at a time.

    Reads happen before the pool and the write happens after it, so a worker never touches the
    store -- which keeps `ensure_table`'s check-then-create race (concurrent writers on a COLD
    table can silently lose rows) off this path entirely.
    """
    if not filings:
        return []
    if workers <= 1 or len(filings) == 1:
        return [fn(f) for f in filings]
    with ThreadPoolExecutor(max_workers=min(workers, len(filings))) as pool:
        return list(pool.map(fn, filings))


def _process_filing(filing: pd.Series, extractor: LLMExtractor,
                    roles: dict[str, str],
                    titles: dict[str, str]) -> tuple[list[dict], int, str | None]:
    """`(rows, guard rejections, skip reason)` for one 8-K.

    A skip reason short-circuits BEFORE the LLM call, so a truncated or tally-free
    narrative costs nothing.
    """
    text = filing.get("item_text")
    reason = rejection_reason(text)
    if reason is not None:
        return [], 0, reason
    try:
        extract = extractor.extract(Item507Extract, text, instructions=_VOTES_PROMPT)
    except Exception as e:
        logger.warning("%s %s: Item 5.07 LLM extraction failed (%s)",
                       filing.get("ticker", ""), filing.get("filing_date", ""), e)
        return [], 0, "extraction failed"
    rows, rejected = _proposal_rows(filing["ticker"], filing, extract, text, roles, titles)
    return rows, rejected, None if rows else "no proposals survived the guard"


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


def fetch_8k_votes_llm(
    context: Context,
    config: DictConfig,
    tickers: list[str],
    model: str | None = None,
    max_chars: int | None = None,
    cache: bool | None = None,
    workers: int = _LLM_WORKERS,
) -> None:
    """Build/refresh `sec_8k_votes` from the stored Item 5.07 narratives, ticker by ticker.

    Reads `sec_8k` (never unprojected), skips accessions already stored, and upserts each
    ticker's rows before starting the next -- LLM calls are paid for, so nothing is
    batched. Skips gracefully when OPENAI_API_KEY is absent.

    `max_chars` defaults to `config.gpt.max_chars.sec8k_votes` (40k, not the DEF 14A
    path's 130k) because the input is one already-carved item, not a whole proxy: the
    longest Item 5.07 narrative in the 333-filing baseline is 17,032 chars, so 40k
    truncates nothing while keeping a runaway `item_text` from turning into a runaway bill.

    `model` / `cache` default to `config.gpt`; pass an explicit keyword to pin one for
    research without touching config.
    """
    model = model or config.gpt.llm_model[config.gpt.default_api]
    max_chars = config.gpt.max_chars.sec8k_votes if max_chars is None else max_chars
    cache = config.gpt.cache if cache is None else cache
    if not context.store.exists(Tables.sec_8k):
        context.log.warning("sec_8k does not exist yet — run the 8-K fetcher first; "
                            "Item 5.07 votes skipped")
        return

    source = context.store.load(Tables.sec_8k, columns=list(_SOURCE_COLS),
                                where={"item": _ITEM, "ticker": list(tickers)})
    if source is None or source.empty:
        context.log.info("no stored Item 5.07 narratives for the %d requested ticker(s)",
                         len(tickers))
        return

    seen = set(existing_filings(context, Tables.sec_8k_votes))
    todo = source[~source["accession_number"].isin(seen)]
    context.log.info("Item 5.07: %d stored filing(s) for %d ticker(s), %d already parsed, "
                     "%d to read", len(source), source["ticker"].nunique(),
                     len(source) - len(todo), len(todo))
    if todo.empty:
        return

    try:
        extractor = LLMExtractor(model=model, max_chars=max_chars, cache=cache)
    except EnvironmentError as e:
        context.log.warning("Item 5.07 vote extraction skipped: %s", e)
        return

    total_rows, total_rejected = 0, 0
    skips: dict[str, int] = {}
    for ticker, group in tqdm(todo.groupby("ticker"), desc="8-K votes"):
        role_source = _role_source(context, str(ticker))
        ticker_rows: list[dict] = []

        def _one(f: pd.Series) -> tuple[list[dict], int, str | None]:
            # the role map is a function of the MEETING, so it is rebuilt per filing -- but off
            # the three frames read once above, so a worker never touches the database
            roles, titles = _role_map(role_source, f.get("period_of_report"))
            return _process_filing(f, extractor, roles, titles)

        filings = [f for _, f in group.iterrows()]
        for rows, rejected, reason in _map_filings(_one, filings, workers):
            ticker_rows.extend(rows)
            total_rejected += rejected
            if reason:
                skips[reason] = skips.get(reason, 0) + 1

        if ticker_rows:
            written = context.store.save(Tables.sec_8k_votes, _prepare_frame(ticker_rows))
            total_rows += written
            unmatched = sum(r.get("n_nominees_unmatched") or 0 for r in ticker_rows)
            nominees = sum(r.get("n_nominees") or 0 for r in ticker_rows)
            context.log.info("%s: +%d vote row(s) from %d filing(s); %d/%d nominees unmatched",
                             ticker, written, len(group), int(unmatched), int(nominees))

    context.log.info("Item 5.07: %d row(s) written, %d row(s) rejected by the fabrication "
                     "guard", total_rows, total_rejected)
    for reason, n in sorted(skips.items(), key=lambda kv: -kv[1]):
        # every one of these is a filing that produced NO rows; the reason is what
        # separates an accepted loss from a correct empty answer
        context.log.info("Item 5.07: %d filing(s) skipped — %s", n, reason)
