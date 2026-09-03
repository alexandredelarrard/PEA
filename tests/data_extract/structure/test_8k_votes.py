"""Item 5.07 shareholder-vote parsing: the fabrication guard, the flatten, and the role map.

Synthetic-from-real. Every `item_text` in `fixtures/item507_texts.json` is a REAL narrative
captured out of `sec_8k` (10 filings, 2011-2026), and the `Item507Extract` objects below are
hand-read off those same texts. That split is deliberate: the LLM's reading cannot be tested
without paying for it, but everything downstream of the reading — the guard, the flatten, the
role categorisation, the flags — is pure, so it can be pinned exactly. The fixtures were chosen
to be the geometries that broke the deterministic parser, because they are the LLM's inputs too:

  aapl_2025  4-column per-nominee table; the auditor table drops to 3 columns (no broker column)
  jpm_2025   a `91.45 | 8.14 | 0.41` percentage row stacked INSIDE the table, `N/A` broker
             non-votes, and the CEO (James Dimon) on the ballot
  tdg_2020   `FOR | WITHHELD` instead of For/Against, plus the VERTICAL label:value layout the
             deterministic parser missed 100% of the time
  aee_2026   the dropped-header case: the director table's header row is just "Name"
  ge_2019    `Non-Votes`  ) the same filer, two vocabularies for one column
  ge_2024    `Broker Non-Votes`
  nke_2011   a genuine tally-free Item 5.07(d) board-response filing (553 chars, form 8-K/A)
  aapl_2020  `item_text` stored empty
  jpm_2017_freq  the say-on-pay FREQUENCY vote and its four buckets
                 (`One Year | Two Years | Three Years | Abstain | Broker Non-Votes`)
  tdg_2014_special  the SHORTEST genuine tally in the whole baseline, 554 chars — which is
                 why the length floor cannot be raised to catch a 508-char truncation stub
"""
from __future__ import annotations

import json
import pathlib
import re

import pandas as pd

from src.data_extract.utils.structure.votes.flatten import (
    _DIRECTOR_COLS, _LOW_SUPPORT_THRESHOLD, _ROLE_CATEGORIES, _VOTE_FIELDS,
    _prepare_frame, _proposal_rows,
)
from src.data_extract.utils.structure.votes.guard import (
    _MIN_ITEM_TEXT_CHARS, _name_in_source, has_vote_numbers, mentions_preliminary,
    rejection_reason,
)
from src.data_extract.utils.structure.votes.roles import _role_map
from src.data_extract.utils.schemas.vote_schema import (
    Item507Extract, NomineeVote, PROPOSAL_TYPES, ProposalVote,
)
from src.data_store.schema import Tables

_FIXTURES = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "item507_texts.json").read_text(encoding="utf-8")
)


def _filing(key: str, **overrides) -> pd.Series:
    """The stored `sec_8k` row for a fixture filing, as the fetcher sees it."""
    f = dict(_FIXTURES[key])
    text = f.pop("item_text")
    f.update(overrides)
    return pd.Series({**f, "item_text": text})


def _text(key: str) -> str:
    return _FIXTURES[key]["item_text"]


def _nominee(name: str, f: float, a: float, ab: float | None = None,
             bnv: float | None = None) -> NomineeVote:
    return NomineeVote(name=name, votes_for=f, votes_against=a, votes_abstain=ab,
                       votes_broker_non_votes=bnv)


def _election(nominees: list[NomineeVote], standard: str = "against",
              number: str = "1") -> ProposalVote:
    return ProposalVote(proposal_number=number, description="Election of directors",
                        proposal_type="director_election", vote_standard=standard,
                        nominees=nominees)


def _plain(description: str, ptype: str, f: float, a: float, ab: float | None = None,
           bnv: float | None = None, number: str = "2") -> ProposalVote:
    return ProposalVote(proposal_number=number, description=description, proposal_type=ptype,
                        votes_for=f, votes_against=a, votes_abstain=ab,
                        votes_broker_non_votes=bnv)


def _extract(*proposals: ProposalVote, meeting: str | None = None,
             preliminary: bool | None = None) -> Item507Extract:
    return Item507Extract(meeting_date=meeting, is_preliminary=preliminary,
                          proposals=list(proposals))


def _rows(key: str, extract: Item507Extract, roles: dict | None = None,
          titles: dict | None = None, **overrides) -> tuple[list[dict], int]:
    f = _filing(key, **overrides)
    return _proposal_rows(f["ticker"], f, extract, _text(key), roles or {}, titles or {})


# --------------------------------------------------------------------------- #
# The hand-read extracts                                                       #
# --------------------------------------------------------------------------- #
# Apple's 2025 annual meeting, read off `aapl_2025` line by line.
_AAPL_NOMINEES = [
    _nominee("Wanda Austin", 9_072_076_816, 40_131_307, 29_197_385, 3_038_264_304),
    _nominee("Tim Cook", 8_970_310_928, 153_141_693, 17_952_887, 3_038_264_304),
    _nominee("Alex Gorsky", 8_946_626_018, 165_324_875, 29_454_615, 3_038_264_304),
    _nominee("Andrea Jung", 8_546_796_776, 565_487_160, 29_121_572, 3_038_264_304),
    _nominee("Art Levinson", 8_479_896_928, 633_590_301, 27_918_279, 3_038_264_304),
    _nominee("Monica Lozano", 9_024_832_308, 87_408_524, 29_164_676, 3_038_264_304),
    _nominee("Ron Sugar", 8_632_486_843, 478_710_182, 30_208_483, 3_038_264_304),
    _nominee("Sue Wagner", 8_744_107_302, 368_677_410, 28_620_796, 3_038_264_304),
]
# The auditor table has NO broker-non-vote column at all -> null, not zero.
_AAPL_AUDITOR = _plain("Ratify the appointment of Ernst & Young LLP",
                       "auditor_ratification", 11_910_666_249, 221_074_424, 47_929_139)
_AAPL_SAY_ON_PAY = _plain("Advisory resolution to approve executive compensation",
                          "say_on_pay", 8_397_138_183, 691_312_529, 52_954_796,
                          3_038_264_304, number="3")

# JPMorgan's 2025 annual meeting. James Dimon is the CEO and is on the ballot.
_JPM_NOMINEES = [
    _nominee("Linda B. Bammann", 1_956_305_887, 53_454_668, 3_727_521, 357_873_002),
    _nominee("Michele G. Buck", 1_996_877_914, 12_350_979, 4_259_183, 357_873_002),
    _nominee("Stephen B. Burke", 1_868_304_460, 141_164_783, 4_018_833, 357_873_002),
    _nominee("Todd A. Combs", 1_942_363_468, 66_719_133, 4_405_475, 357_873_002),
    _nominee("Alicia Boler Davis", 2_001_885_628, 7_386_980, 4_215_468, 357_873_002),
    _nominee("James Dimon", 1_873_572_626, 124_438_267, 15_477_183, 357_873_002),
    _nominee("Alex Gorsky", 1_998_762_422, 10_469_740, 4_255_914, 357_873_002),
    _nominee("Mellody Hobson", 1_999_517_720, 9_514_453, 4_455_903, 357_873_002),
    _nominee("Phebe N. Novakovic", 1_993_131_287, 16_089_184, 4_267_605, 357_873_002),
    _nominee("Virginia M. Rometty", 1_982_392_832, 25_882_527, 5_212_717, 357_873_002),
    _nominee("Brad D. Smith", 2_003_349_948, 5_909_237, 4_228_891, 357_873_002),
    _nominee("Mark A. Weinberger", 1_979_784_700, 29_524_070, 4_179_306, 357_873_002),
]

# TransDigm 2020: FOR / WITHHELD, and a vertical label:value proposal.
_TDG_NOMINEES = [
    _nominee("David Barr", 48_694_503, 714_813),
    _nominee("Mervin Dunn", 35_388_087, 14_021_229),
    _nominee("Michael Graff", 35_186_600, 14_222_716),
    _nominee("Sean Hennessy", 36_649_801, 12_759_515),
    _nominee("Kevin Stein", 49_161_541, 247_775),
]
_TDG_SAY_ON_PAY = _plain("Advisory vote on compensation paid to the named executive officers",
                         "say_on_pay", 32_694_219, 16_684_559, 30_538, 1_531_934)


# --------------------------------------------------------------------------- #
# Rule 1 -- the fabrication guard                                              #
# --------------------------------------------------------------------------- #
def test_the_three_zero_row_populations_are_refused_for_distinct_reasons():
    """All three produce zero rows and the run log has to tell them apart: an empty stored
    narrative, a genuine 5.07(d) board response (a CORRECT empty answer, 8.8% of filings),
    and a stub too short to be anything. A single "0 rows" counter would hide each inside
    the others."""
    assert rejection_reason(_text("aapl_2020")) == "empty item_text"
    assert rejection_reason(_text("nke_2011")) == "no comma-grouped number -- no vote table"
    assert "truncated" in rejection_reason("Item 5.07. 1,234,567 votes.")
    assert rejection_reason(_text("aapl_2025")) is None


def test_a_narrative_with_no_comma_grouped_number_has_no_vote_table():
    """The cheapest fabrication guard there is. NKE's 5.07(d) response is prose about the
    board's decision and contains no share count anywhere; pad it past the length floor and
    the number test still catches it."""
    assert not has_vote_numbers(_text("nke_2011"))
    padded = _text("nke_2011") + " " * (_MIN_ITEM_TEXT_CHARS + 100)
    assert rejection_reason(padded) == "no comma-grouped number -- no vote table"
    assert has_vote_numbers(_text("aapl_2025"))


def test_a_fabricated_nominee_is_dropped_and_the_real_ones_survive():
    """The measured failure: the model returned "John Doe" / "Jane Smith" at 250,000,000 votes
    for a filing it could not read. Neither the name nor the number is in Apple's text."""
    fake = [_nominee("John Doe", 250_000_000, 1_000_000, 0, 0),
            _nominee("Jane Smith", 250_000_000, 1_000_000, 0, 0)]
    rows, rejected = _rows("aapl_2025", _extract(_election(_AAPL_NOMINEES + fake)))
    assert rejected == 2
    assert len(rows) == 1
    assert rows[0]["n_nominees"] == 8
    kept = {n["name"] for n in json.loads(rows[0]["nominee_votes_json"])}
    assert "John Doe" not in kept and "Jane Smith" not in kept
    assert "Tim Cook" in kept


def test_a_real_name_carrying_invented_numbers_is_still_dropped():
    """Half the guard is not enough. A permuted or invented tally under a name that IS in the
    document is the harder fabrication, so the numbers are checked independently."""
    rows, rejected = _rows("aapl_2025", _extract(_election(
        [_nominee("Tim Cook", 1_234_567_890, 987_654_321, 111_111_111, 222_222_222)])))
    # 2, not 1: the nominee is dropped, and an election left with no nominee at all carries
    # no tally, so the proposal goes with it
    assert rejected == 2 and rows == []


def test_a_name_wrapped_around_its_own_vote_numbers_is_not_rejected():
    """Measured on PTC's 2010 filing during the Phase-6 run: edgartools wraps a narrow name
    column, so the vote numbers land BETWEEN the two halves of the name --

        Paul                       100,753,338     1,735,851     7,486,441
        A. Lacy

    -- and a contiguous substring check discarded three CORRECTLY read nominees. The token
    fallback survives the wrap; the fabrications still fail it, which is the point.
    """
    wrapped = ("                          For          Withheld       Broker Non-Votes\n"
               "  Paul              100,753,338       1,735,851          7,486,441\n"
               "  A. Lacy\n"
               "  Michael            65,279,989      37,209,200          7,486,441\n"
               "  E. Porter\n")
    assert _name_in_source("Paul A. Lacy", wrapped)
    assert _name_in_source("Michael E. Porter", wrapped)
    assert not _name_in_source("John Doe", wrapped)
    assert not _name_in_source("Jane Smith", wrapped)
    # a middle initial is in every proxy, so it must not be what carries a match
    assert not _name_in_source("A. E. Nobody", wrapped)


def test_a_non_director_proposal_with_no_grounded_number_is_rejected():
    rows, rejected = _rows("aapl_2025", _extract(
        _plain("A proposal nobody filed", "company_proposal", 1_111_111_111, 2_222_222_222)))
    assert rejected == 1 and rows == []


def test_one_rounded_cell_does_not_cost_a_whole_row():
    """The guard is ANY-of-four, not all-of-four, on purpose: the measured LLM error set
    includes fractional votes truncated 1000x, and rejecting a row over one bad cell would
    throw away three good ones."""
    bent = list(_AAPL_NOMINEES)
    bent[1] = _nominee("Tim Cook", 8_970_310_928, 153_141_693, 17_952_887, 3_038_264)  # /1000
    rows, rejected = _rows("aapl_2025", _extract(_election(bent)))
    assert rejected == 0 and rows[0]["n_nominees"] == 8


def test_the_shortest_genuine_tally_in_the_corpus_is_not_mistaken_for_a_stub():
    """TDG's 2014 special meeting is 554 chars and COMPLETE: one stock-option plan,
    `FOR / AGAINST / ABSTAIN`, no broker column. It is 46 chars longer than HWM 2019's
    truncated stub, which is the whole reason the length floor sits at 400 and the
    comma-grouped-number test does the real work."""
    assert len(_text("tdg_2014_special")) == 554
    assert rejection_reason(_text("tdg_2014_special")) is None
    rows, rejected = _rows("tdg_2014_special", _extract(
        _plain("Approve and adopt the 2014 Stock Option Plan", "equity_plan",
               36_158_343, 11_920_062, 39_576, number="1")))
    assert rejected == 0
    assert rows[0]["votes_for"] == 36_158_343
    assert rows[0]["votes_broker_non_votes"] is None


def test_the_frequency_buckets_are_not_an_election():
    """JPM 2017 tallies `One Year | Two Years | Three Years | Abstain | Broker Non-Votes`.
    A per-label tally is not a per-person one: treating it as an election would build a role
    map and a `min_support_pct` for a bucket called "One Year". The buckets land in
    `nominee_votes_json`, where a recategorisation stays free, and the 20 category columns
    stay null."""
    buckets = [_nominee("One Year", 2_609_772_372, 0),
               _nominee("Two Years", 51_181_321, 0),
               _nominee("Three Years", 79_750_121, 0)]
    rows, rejected = _rows("jpm_2017_freq", _extract(ProposalVote(
        proposal_number="1", description="Frequency of future Say on Pay votes",
        proposal_type="say_on_pay_frequency", votes_abstain=7_681_114,
        votes_broker_non_votes=402_161_634, nominees=buckets)))
    assert rejected == 0
    row = rows[0]
    assert row["proposal_type"] == "say_on_pay_frequency"
    assert all(row[c] is None for c in _DIRECTOR_COLS)     # no n_nominees, no min_support
    assert row["nominee_sum_matches"] is None              # meaningless off an election
    assert row["votes_abstain"] == 7_681_114               # proposal-level votes SURVIVE
    assert {b["name"] for b in json.loads(row["nominee_votes_json"])} == {
        "One Year", "Two Years", "Three Years"}


# --------------------------------------------------------------------------- #
# The flatten                                                                  #
# --------------------------------------------------------------------------- #
def test_a_real_meeting_flattens_to_one_row_per_proposal():
    rows, rejected = _rows("aapl_2025", _extract(
        _election(_AAPL_NOMINEES), _AAPL_AUDITOR, _AAPL_SAY_ON_PAY,
        meeting="2025-02-25"))
    assert rejected == 0
    assert [r["proposal_seq"] for r in rows] == [1.0, 2.0, 3.0]     # gap-free, never null
    assert [r["proposal_type"] for r in rows] == [
        "director_election", "auditor_ratification", "say_on_pay"]
    assert all(r["accession_number"] == "0001140361-25-005876" for r in rows)


def test_the_election_row_carries_no_proposal_level_total():
    """Summing nominees into a proposal-level `votes_for` would invent a number the filing
    never printed — and it is not even well defined, since each nominee has their own tally."""
    rows, _ = _rows("aapl_2025", _extract(_election(_AAPL_NOMINEES), _AAPL_AUDITOR))
    election, auditor = rows
    assert all(election[f] is None for f in _VOTE_FIELDS)
    assert auditor["votes_for"] == 11_910_666_249


def test_an_absent_broker_column_is_null_and_a_printed_zero_would_be_zero():
    """Apple's auditor table has three columns; JPM prints `N/A`. Both mean "not reported",
    which is not the same statement as "no broker non-votes were cast"."""
    rows, _ = _rows("aapl_2025", _extract(_AAPL_AUDITOR, _AAPL_SAY_ON_PAY))
    auditor, say_on_pay = rows
    assert auditor["votes_broker_non_votes"] is None
    assert say_on_pay["votes_broker_non_votes"] == 3_038_264_304


def test_the_twenty_category_columns_are_null_on_a_non_election_row():
    """~85% of the table. NULL and not 0, so a reader cannot mistake "not an election" for
    "an election in which no nominee fell in this bucket"."""
    rows, _ = _rows("aapl_2025", _extract(_AAPL_AUDITOR))
    assert all(rows[0][c] is None for c in _DIRECTOR_COLS)


def test_the_withheld_standard_lands_in_votes_against_never_in_broker_non_votes():
    """20.0% of filings print Withheld rather than Against. The prompt-fixable error measured
    on ~62 rows was the model routing that count to broker non-votes instead."""
    rows, _ = _rows("tdg_2020", _extract(_election(_TDG_NOMINEES, standard="withheld")))
    row = rows[0]
    assert row["vote_standard"] == "withheld"
    stein = [n for n in json.loads(row["nominee_votes_json"]) if n["name"] == "Kevin Stein"][0]
    assert stein["votes_against"] == 247_775
    assert stein["votes_broker_non_votes"] is None
    assert row["votes_broker_non_votes_unmatched"] is None


def test_the_vertical_label_value_layout_is_an_ordinary_proposal_row():
    """`FOR  32,694,219` on its own line, one label per line. The deterministic parser missed
    this geometry 100% of the time; to the flatten it is just a proposal."""
    rows, rejected = _rows("tdg_2020", _extract(_TDG_SAY_ON_PAY))
    assert rejected == 0
    assert rows[0]["votes_for"] == 32_694_219
    assert rows[0]["votes_broker_non_votes"] == 1_531_934


def test_one_column_carries_both_of_ge_s_vocabularies():
    """The same filer switched from `Non-Votes` (2019) to `Broker Non-Votes` (2024). They are
    the same column and must not become two."""
    culp_2019 = _nominee("H. Lawrence Culp, Jr.",
                         4_546_353_762, 178_604_825, 28_064_806, 1_628_738_820)
    rows_2019, _ = _rows("ge_2019", _extract(_election([culp_2019])))
    assert rows_2019[0]["votes_broker_non_votes_unmatched"] == 1_628_738_820
    assert "Non-Votes" in _text("ge_2019") and "Broker Non-Votes" not in _text("ge_2019")
    assert "Broker Non-Votes" in _text("ge_2024")


def test_the_dropped_header_case_still_flattens():
    """AEE 2026's director table header is the single word "Name" — the four column headings
    are gone. That is what produced a silent column PERMUTATION in the deterministic parser;
    nothing downstream can detect it, so the only defence is that the numbers be verbatim."""
    rows, rejected = _rows("aee_2026", _extract(_election(
        [_nominee("Cynthia J. Brinkley", 211_811_213, 6_224_999, 509_328, 24_519_860),
         _nominee("Martin J. Lyons, Jr.", 211_634_204, 6_520_326, 391_010, 24_519_860)])))
    assert rejected == 0 and rows[0]["n_nominees"] == 2


# --------------------------------------------------------------------------- #
# Support summary + the flag                                                   #
# --------------------------------------------------------------------------- #
def test_min_support_picks_the_worst_nominee_off_the_real_tallies():
    rows, _ = _rows("aapl_2025", _extract(_election(_AAPL_NOMINEES)))
    row = rows[0]
    # Levinson: 8,479,896,928 / (8,479,896,928 + 633,590,301) = 0.9305
    assert row["min_support_name"] == "Art Levinson"
    assert 0.930 < row["min_support_pct"] < 0.931
    assert row["n_nominees_below_70pct"] == 0


def test_the_below_70pct_count_fires_on_a_real_revolt():
    """TDG 2020's weakest nominee sits at 71.2%, so the real answer there is 0. The threshold
    itself is checked by moving one nominee below it, which is arithmetic, not extraction."""
    rows, _ = _rows("tdg_2020", _extract(_election(_TDG_NOMINEES, standard="withheld")))
    assert rows[0]["min_support_name"] == "Michael Graff"
    assert 0.712 < rows[0]["min_support_pct"] < 0.713
    assert rows[0]["n_nominees_below_70pct"] == 0

    revolt = [n for n in _TDG_NOMINEES if n.name != "Michael Graff"]
    revolt.append(_nominee("Michael Graff", 14_222_716, 35_186_600))    # columns swapped
    rows2, _ = _rows("tdg_2020", _extract(_election(revolt, standard="withheld")))
    assert rows2[0]["n_nominees_below_70pct"] == 1
    assert rows2[0]["min_support_pct"] < _LOW_SUPPORT_THRESHOLD


def test_nominee_sum_matches_is_a_flag_and_never_drops_a_row():
    """Every nominee at one meeting faces the same shares-represented pool, so the four
    columns should sum alike. Computable on 96% of filings and true on 91% — a good monitor
    and a bad gate, so a 0 is stored and the row is kept."""
    ok, _ = _rows("aapl_2025", _extract(_election(_AAPL_NOMINEES)))
    assert ok[0]["nominee_sum_matches"] == 1.0

    bent = list(_AAPL_NOMINEES)
    bent[0] = _nominee("Wanda Austin", 9_072_076_816, 40_131_307, 29_197_385, 40_131_307)
    off, rejected = _rows("aapl_2025", _extract(_election(bent)))
    assert rejected == 0                                  # kept, not filtered
    assert off[0]["nominee_sum_matches"] == 0.0

    single, _ = _rows("aapl_2025", _extract(_election([_AAPL_NOMINEES[0]])))
    assert single[0]["nominee_sum_matches"] is None        # nothing to compare against


# --------------------------------------------------------------------------- #
# Role categorisation                                                          #
# --------------------------------------------------------------------------- #
def _proxy_source(ceo: str, execs: list[tuple[str, str]], directors: list[str],
                  as_of: str = "2025-04-01") -> dict[str, pd.DataFrame]:
    return {
        "ceo": pd.DataFrame([{"ticker": "JPM", "as_of": pd.Timestamp(as_of),
                              "ceo_name_proxy": ceo}]),
        "exec": pd.DataFrame([{"ticker": "JPM", "as_of": pd.Timestamp(as_of), "name": n,
                               "title": t, "fiscal_year": 2024} for n, t in execs]),
        "director": pd.DataFrame([{"ticker": "JPM", "as_of": pd.Timestamp(as_of), "name": n}
                                  for n in directors]),
    }


def test_the_ceo_on_the_ballot_lands_in_the_ceo_bucket_and_nobody_is_unmatched():
    """The plan's role test on a real meeting: JPM 2025 elected 12 directors, one of whom
    (James Dimon) is the CEO. `unmatched` has to be 0 here — that count IS the join's error
    rate, and it is a column precisely so a bad join cannot hide inside `non_employee`."""
    names = [n.name for n in _JPM_NOMINEES]
    source = _proxy_source(ceo="Jamie Dimon",                    # the proxy's own short form
                           execs=[("James Dimon", "Chairman and CEO"),
                                  ("Jeremy Barnum", "Chief Financial Officer")],
                           directors=[n for n in names if n != "James Dimon"])
    roles, titles = _role_map(source, pd.Timestamp("2025-05-20"))
    rows, rejected = _rows("jpm_2025", _extract(_election(_JPM_NOMINEES)), roles, titles)
    row = rows[0]
    assert rejected == 0
    assert row["n_nominees"] == 12
    assert row["n_nominees_ceo"] == 1
    assert row["n_nominees_non_employee"] == 11
    assert row["n_nominees_exec_officer"] == 0              # Barnum was not on the ballot
    assert row["n_nominees_unmatched"] == 0
    # the CEO bucket is Dimon's own line, not a share of the meeting's total
    assert row["votes_for_ceo"] == 1_873_572_626
    assert row["votes_against_ceo"] == 124_438_267


def test_a_nominee_the_proxy_never_names_is_unmatched_not_quietly_a_director():
    source = _proxy_source(ceo="Jamie Dimon", execs=[],
                           directors=["Linda B. Bammann", "Michele G. Buck"])
    roles, titles = _role_map(source, pd.Timestamp("2025-05-20"))
    rows, _ = _rows("jpm_2025", _extract(_election(_JPM_NOMINEES)), roles, titles)
    row = rows[0]
    assert row["n_nominees_non_employee"] == 2
    assert row["n_nominees_ceo"] == 1
    assert row["n_nominees_unmatched"] == 9
    assert sum(row[f"n_nominees_{c}"] for c in _ROLE_CATEGORIES) == row["n_nominees"]


def test_an_executive_on_the_ballot_contributes_a_title():
    source = _proxy_source(ceo="Jamie Dimon",
                           execs=[("James Dimon", "Chairman and CEO"),
                                  ("Todd A. Combs", "CEO, Consumer Lending")],
                           directors=["Linda B. Bammann"])
    roles, titles = _role_map(source, pd.Timestamp("2025-05-20"))
    rows, _ = _rows("jpm_2025", _extract(_election(_JPM_NOMINEES)), roles, titles)
    assert rows[0]["n_nominees_exec_officer"] == 1
    assert rows[0]["exec_officer_titles"] == "CEO, Consumer Lending"


def test_a_proxy_filed_after_the_meeting_cannot_categorise_it():
    """A proxy filed after the meeting describes the board the meeting elected, so using it
    would categorise a nominee by the outcome of the very vote being categorised."""
    later = _proxy_source(ceo="Jamie Dimon", execs=[],
                          directors=[n.name for n in _JPM_NOMINEES],
                          as_of="2026-04-01")
    roles, _ = _role_map(later, pd.Timestamp("2025-05-20"))
    assert roles == {}


def test_the_nearest_prior_proxy_wins_over_an_older_one():
    source = _proxy_source(ceo="Jamie Dimon", execs=[], directors=["Linda B. Bammann"],
                           as_of="2025-04-01")
    stale = _proxy_source(ceo="Somebody Else", execs=[], directors=["Old Director"],
                          as_of="2019-04-01")
    for k in source:
        source[k] = pd.concat([stale[k], source[k]], ignore_index=True)
    roles, _ = _role_map(source, pd.Timestamp("2025-05-20"))
    assert roles == {"bammann|l": "non_employee", "dimon|j": "ceo"}


# --------------------------------------------------------------------------- #
# Rule 2 -- amendments are unioned, never deduped                              #
# --------------------------------------------------------------------------- #
def test_an_amendment_is_stored_as_its_own_rows_and_nothing_is_deduped():
    """"Latest wins" is correct on 33 of 190 multi-filing meetings (17%); unioning the group
    is correct on 173 (91%), because 71% of amendments carry no vote numbers at all. So the PK
    carries the accession and the two filings coexist."""
    first, _ = _rows("aapl_2025", _extract(_election(_AAPL_NOMINEES)))
    amended, _ = _rows("aapl_2025", _extract(_election(_AAPL_NOMINEES)),
                       accession_number="0001140361-25-999999", form="8-K/A",
                       is_amendment=1.0)
    df = _prepare_frame(first + amended)
    assert len(df) == 2                                   # both survive the PK collapse
    assert set(df["accession_number"]) == {"0001140361-25-005876", "0001140361-25-999999"}
    assert df["period_of_report"].nunique() == 1          # ...and union on the same meeting
    assert set(Tables.sec_8k_votes.pk) == {"ticker", "accession_number", "proposal_seq"}


def test_mentions_preliminary_only_fires_on_a_filing_that_says_so():
    """A weak signal by construction: 14 filings corpus-wide say it, which resolves 8 of the
    17 genuine restatements. Stored as a flag, never used to pick between filings."""
    assert not mentions_preliminary(_text("aapl_2025"))
    contested = _text("aapl_2025").replace(
        "cast their votes as described below.",
        "cast their votes as described below. These are estimated preliminary voting results.")
    assert mentions_preliminary(contested)
    rows, _ = _rows("aapl_2025", _extract(_election(_AAPL_NOMINEES)))
    assert rows[0]["mentions_preliminary"] == 0.0


def test_an_unknown_proposal_type_is_nulled_not_invented():
    """The vocabulary is closed and reused verbatim from the retired `sec_def14a_votes`. A
    value outside it is a null, so nothing downstream has to guess what a new label meant."""
    rows, _ = _rows("aapl_2025", _extract(
        ProposalVote(proposal_number="4", description="Something new",
                     proposal_type="climate_proposal",
                     votes_for=11_910_666_249, votes_against=221_074_424)))
    assert rows[0]["proposal_type"] is None
    assert "climate_proposal" not in PROPOSAL_TYPES


# --------------------------------------------------------------------------- #
# The DDL, and a real upsert                                                   #
# --------------------------------------------------------------------------- #
def _ddl_columns(table: str) -> list[str]:
    """Column names declared for `table` in sql/schema.sql, in declaration order."""
    sql = (pathlib.Path(__file__).resolve().parents[3] / "sql" / "schema.sql").read_text(
        encoding="utf-8")
    m = re.search(r'CREATE TABLE IF NOT EXISTS "%s" \((.*?)\n\);' % re.escape(table),
                  sql, re.S)
    assert m, f"{table}: no CREATE TABLE block in sql/schema.sql"
    return re.findall(r'^\s+"([a-z0-9_]+)"\s', m.group(1), re.M)


def test_the_ddl_declares_exactly_what_the_flatten_writes():
    """A DDL that merely parses is not verified. A column the flatten writes but the table
    lacks fails the insert AFTER the tokens are paid; a column the DDL declares but nothing
    writes is permanently NULL, which reads downstream as "this company did not disclose it"
    rather than "we never extracted it"."""
    rows, _ = _rows("aapl_2025", _extract(_election(_AAPL_NOMINEES), _AAPL_AUDITOR))
    election, plain = rows
    assert list(election) == list(plain), (
        "an election row and a plain row must have the SAME column order — on a "
        "first-run CREATE TABLE the frame's column order IS the DDL's")

    ddl, produced = _ddl_columns("sec_8k_votes"), list(election)
    missing = [c for c in produced if c not in ddl]
    extra = [c for c in ddl if c not in produced]
    print("\n=== SANITY: sql/schema.sql vs the Item 5.07 flatten ===")
    print(f"  sec_8k_votes  ddl {len(ddl)} / code {len(produced)}  "
          f"{'OK' if not (missing or extra) else 'MISMATCH'}")
    if missing:
        print(f"    code writes but DDL lacks : {missing}")
    if extra:
        print(f"    DDL declares but unwritten: {extra}")
    assert not missing and not extra
    assert ddl == produced                       # order too, not just the set


def test_every_primary_key_column_is_non_null_in_a_built_row():
    """A NULL in a Postgres PK column aborts the WHOLE insert, not the one row. `proposal_seq`
    is assigned over the kept rows precisely so it cannot be null or have a gap."""
    rows, _ = _rows("aapl_2025", _extract(
        _election(_AAPL_NOMINEES), _AAPL_AUDITOR, _AAPL_SAY_ON_PAY))
    for r in rows:
        for c in Tables.sec_8k_votes.pk:
            assert r[c] is not None, f"{c} is NULL in a built row"
    assert [r["proposal_seq"] for r in rows] == [1.0, 2.0, 3.0]


def test_a_meeting_and_its_amendment_both_survive_a_real_upsert(sqlite_store):
    """Rule 2 through the actual store, not just `drop_duplicates`: the PK carries the
    accession, so the original and the 8-K/A coexist and the reader unions them on
    (ticker, period_of_report)."""
    first, _ = _rows("aapl_2025", _extract(_election(_AAPL_NOMINEES), _AAPL_AUDITOR))
    amended, _ = _rows("aapl_2025", _extract(_election(_AAPL_NOMINEES)),
                       accession_number="0001140361-25-999999", form="8-K/A",
                       is_amendment=1.0)
    sqlite_store.save(Tables.sec_8k_votes, _prepare_frame(first))
    sqlite_store.save(Tables.sec_8k_votes, _prepare_frame(amended))

    back = sqlite_store.load(Tables.sec_8k_votes)
    print("\n=== SANITY: one meeting, two filings, through a real upsert ===")
    print(back[["ticker", "accession_number", "form", "proposal_seq", "proposal_type"]]
          .to_string(index=False))
    assert len(back) == 3                                  # 2 proposals + 1, nothing collapsed
    assert back["accession_number"].nunique() == 2
    assert set(back["form"]) == {"8-K", "8-K/A"}

    # re-saving the same rows must UPDATE, not duplicate -- that is the incremental contract
    sqlite_store.save(Tables.sec_8k_votes, _prepare_frame(first))
    assert len(sqlite_store.load(Tables.sec_8k_votes)) == 3


if __name__ == "__main__":
    import sys
    import pytest as _pytest
    code = _pytest.main([__file__, "-q"])

    # ---- sanity conclusion, printed off the real fixtures ----
    print("\n" + "=" * 78)
    print("Item 5.07 fixtures — what the guard does to 10 REAL stored narratives")
    print("=" * 78)
    for key in sorted(_FIXTURES):
        f = _FIXTURES[key]
        reason = rejection_reason(f["item_text"])
        verdict = "SENT to the LLM" if reason is None else f"skipped: {reason}"
        print(f"  {key:10s} {f['ticker']:5s} {f['form']:6s} "
              f"{len(f['item_text']):6d} chars  ->  {verdict}")

    rows, rejected = _rows("aapl_2025", _extract(
        _election(_AAPL_NOMINEES), _AAPL_AUDITOR, _AAPL_SAY_ON_PAY))
    print(f"\n  AAPL 2025 hand-read -> {len(rows)} proposal rows, {rejected} guard rejections;"
          f" election row has {int(rows[0]['n_nominees'])} nominees,"
          f" min support {rows[0]['min_support_pct']:.4f} ({rows[0]['min_support_name']}),"
          f" nominee_sum_matches={rows[0]['nominee_sum_matches']}")
    fake = _AAPL_NOMINEES + [_nominee("John Doe", 250_000_000, 1_000_000, 0, 0)]
    _, n_bad = _rows("aapl_2025", _extract(_election(fake)))
    print(f"  the same extract with one invented nominee -> {n_bad} rejected, 8 kept")
    print("\nCONCLUSION: the two hard rules hold on real text — a truncated narrative and an "
          "empty one\nare refused for DISTINCT reasons before any LLM call, a tally-free "
          "5.07(d) filing is a\nnormal zero-row outcome, and an invented nominee cannot "
          "survive a source that never\nprints it. There is deliberately NO accuracy gate "
          "here: a vote table has no independent\ntotal and the dominant error is a column "
          "permutation, which is invariant under sums.")
    sys.exit(code)
