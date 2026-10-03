"""
vote_schema.py  (src/data_extract/utils/structure/vote_schema.py)
------------------------------------------------------------------
Pydantic v2 schema for structured extraction of Form 8-K **Item 5.07** ("Submission of
Matters to a Vote of Security Holders") narratives — the certified shareholder-meeting
vote tallies.

Item 5.07 is the ONLY source of these numbers. The disclosure was moved out of 10-Q
Part II Item 4 into the 8-K by Rel. 33-9089 and so begins **2010-03**; no vote count is
ever tagged in XBRL (Apple's 2025 annual-meeting 8-K carries 21 facts, 100% of them
`dei:` cover-page tags), the SEC publishes no data set for it, and no free parse exists.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

#: The `proposal_type` vocabulary, reused verbatim from the retired `sec_def14a_votes`
#: table. Observed distribution over its 653 rows: shareholder_proposal 162,
#: company_proposal 151, director_election 139, auditor_ratification 114, say_on_pay 58,
#: equity_plan 24, say_on_pay_frequency 5.
PROPOSAL_TYPES = (
    "director_election",
    "say_on_pay",
    "say_on_pay_frequency",
    "auditor_ratification",
    "equity_plan",
    "shareholder_proposal",
    "company_proposal",
)

#: The two vote standards a director election can be reported under. 20.0% of filings
#: print "Withheld" rather than "Against"; the two are NOT the same thing legally (a
#: withhold is not a vote against under a plurality standard) and conflating them
#: silently would corrupt every support percentage computed downstream.
VOTE_STANDARDS = ("against", "withheld")


class NomineeVote(BaseModel):
    """One line of a per-line tally table: a director nominee, or a frequency bucket.

    The two share a shape — a label and up to four counts — so they share a model. A
    frequency bucket fills `name` and `votes_for` only; see `ProposalVote.nominees`.
    """

    name: str = Field(
        description="The label EXACTLY as printed at the head of this line — the "
        "nominee's name in an election table, or the frequency bucket "
        "('1 Year', 'Every 2 Years', '3 Years') in a say-on-pay "
        "frequency table"
    )
    votes_for: float | None = Field(
        None, description="Shares voted FOR this nominee, as printed. For a frequency bucket, the shares that chose THIS interval"
    )
    votes_against: float | None = Field(
        None,
        description="Shares voted AGAINST this nominee — or WITHHELD, when the table "
        "uses that column instead. Put the count here either way and record "
        "which word the filing used in the proposal's `vote_standard`",
    )
    votes_abstain: float | None = Field(None, description="Shares ABSTAINED / 'Abstentions' / 'Abstained' for this nominee")
    votes_broker_non_votes: float | None = Field(
        None,
        description="BROKER NON-VOTES for this nominee (also printed as 'Non-Votes'). "
        "Null when the filing does not report them; 0 ONLY when it prints a "
        "zero. NEVER put a 'Withheld' count here — 'Withheld' is a substitute "
        "for 'Against', not for broker non-votes",
    )


class ProposalVote(BaseModel):
    """One matter put to a vote. A director election is ONE proposal carrying every nominee."""

    proposal_number: str | None = Field(
        None, description="The proposal's number AS PRINTED ('1', '2a', 'Item 3', 'Proposal 4'). Null when the filing numbers nothing"
    )
    description: str = Field(description="What was voted on, in the filing's own words, condensed to one line")
    proposal_type: str = Field(
        description="One of: director_election, say_on_pay, say_on_pay_frequency, "
        "auditor_ratification, equity_plan, shareholder_proposal, company_proposal. "
        "`say_on_pay` is the advisory vote ON compensation; `say_on_pay_frequency` "
        "is the vote on HOW OFTEN that vote is held. A proposal submitted by a "
        "shareholder is `shareholder_proposal` whatever its subject; any other "
        "management item that fits none of the named types is `company_proposal`"
    )
    vote_standard: str | None = Field(
        None,
        description="For a director election only: 'against' if the table's second column "
        "is headed Against, 'withheld' if it is headed Withheld / Withhold. "
        "Null for every other proposal type",
    )
    votes_for: float | None = Field(
        None,
        description="Shares voted FOR, as printed. Null for a director election — the "
        "per-nominee counts go in `nominees` and must NOT be summed here. Null "
        "for a say_on_pay_frequency vote too: it is tallied per FREQUENCY "
        "BUCKET, and '1 Year' is not a vote FOR anything",
    )
    votes_against: float | None = Field(
        None,
        description="Shares voted AGAINST (or WITHHELD), as printed. Null for a director "
        "election, and null for a say_on_pay_frequency vote — '2 Years' is a "
        "choice of interval, NOT a vote against",
    )
    votes_abstain: float | None = Field(
        None,
        description="Shares ABSTAINED, as printed. On a say_on_pay_frequency vote this is "
        "that table's own 'Abstain' column, which sits BESIDE the year buckets "
        "— never the '3 Years' count",
    )
    votes_broker_non_votes: float | None = Field(
        None,
        description="BROKER NON-VOTES, as printed. Null when the filing does not report "
        "them or prints 'N/A'; 0 ONLY when it prints a zero. On a "
        "say_on_pay_frequency vote this is that table's own 'Broker Non-Votes' "
        "column — never the 'Abstain' count shifted along one",
    )
    nominees: list[NomineeVote] = Field(
        default_factory=list,
        description="EVERY per-line entry of the proposal's own tally table, for these two "
        "proposal types ONLY. For a director_election: one entry per nominee. For a "
        "say_on_pay_frequency vote: one entry per FREQUENCY BUCKET — `name` is the "
        "bucket label exactly as printed ('1 Year', 'Every 2 Years', '3 Years'), its "
        "count goes in `votes_for`, and the other three fields are null. Empty for "
        "every other proposal type",
    )


class Item507Extract(BaseModel):
    """Everything voted on at one shareholder meeting, as reported by one 8-K."""

    meeting_date: str | None = Field(None, description="Date of the shareholder meeting in YYYY-MM-DD form")
    is_preliminary: bool | None = Field(
        None,
        description="True if the filing itself says these results are PRELIMINARY, "
        "estimated, or subject to certification by the Inspector of Election",
    )
    proposals: list[ProposalVote] = Field(
        default_factory=list,
        description="Every matter voted on, in the order the filing presents them. Empty when "
        "the filing reports no tallies at all — some Item 5.07 filings only "
        "disclose the board's response to a say-on-pay frequency vote (5.07(d)) "
        "and carry no numbers. Returning nothing is a correct answer there",
    )
