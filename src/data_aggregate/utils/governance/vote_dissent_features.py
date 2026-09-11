"""
vote_dissent_features.py  (src/data_aggregate/utils/governance/vote_dissent_features.py)
----------------------------------------------------------------------------------------
SHAREHOLDER DISSENT from the Item 5.07 vote record (`sec_8k_votes`) -- the only numbers in
EDGAR certified by the OWNERS rather than reported by the company. Four families, one source
table, one pass:

    say-on-pay          how much of the vote opposed the pay package
    director election   how much opposed the BOARD, per nominee and in aggregate
    CEO-specific        how much opposed the CEO in particular, net of the board
    auditor             how much opposed ratifying the audit firm

WHY THESE COME BEFORE COMPENSATION LEVELS. A pay ratio is a company characteristic; a 31%
say-on-pay dissent is a revealed institutional-investor OPINION about that characteristic,
priced by the people who can actually sell the stock.

TWO DENOMINATOR DECISIONS, both load-bearing, both easy to get silently wrong:

1. BROKER NON-VOTES ARE EXCLUDED (D11). A broker non-vote is an absent instruction from a
   beneficial owner, not an opinion. Putting it in the denominator would read as support
   dilution that nobody expressed, and it lands unevenly: under NYSE Rule 452 auditor
   ratification is a ROUTINE matter where brokers may vote uninstructed, so auditor rows
   carry broker non-votes on only 30.4% of rows while say-on-pay rows (non-routine) carry
   them nearly always. A shared denominator would make the two families incomparable.

2. WITHHELD IS ALREADY INSIDE `votes_against` (D12). 20.0% of election filings run a
   plurality standard and print "Withheld" instead of "Against"; the extractor writes that
   count into `votes_against` either way and records the basis in `vote_standard`. So
   `vote_standard` is INFORMATIONAL here -- adding a withhold term would double-count exactly
   those filings.

NO IMPUTATION (D25). Phase 2 §1.3 measured the source: 99.5% of say-on-pay rows and 99.4% of
auditor rows carry all three vote columns, and `nominee_votes_json` is filled on 100% of the
7,884 election rows. `sec_8k_votes` is never passed through an impute function, and a failed
extraction reads as NaN, never as zero dissent -- "we could not read the vote" and "nobody
objected" are opposite claims and the code keeps them apart everywhere below.

EVERY FIELD HERE IS AN EVENT (D21). A vote is evidence about the meeting that produced it,
not about a year in which no meeting happened, so all of them expire at
`GOVERNANCE_EVENT_MAX_AGE_DAYS` -- unlike the structural levels in `panel.py`, which describe
a board that keeps existing between proxies.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.pit import fiscal_change_to_daily, fundamentals_to_daily
from src.data_aggregate.utils.governance.staleness import (
    GOVERNANCE_EVENT_MAX_AGE_DAYS, expire_event_fields,
)

#: Fields that are already 0/1 indicators and must NOT be peer-z-scored or percentile-ranked:
#: the cross-sectional rank of a binary column is a re-encoding of its base rate, and a peer
#: z of a flag is dominated by how many peers happened to trip it. `panel.py` reads this set
#: to route them; membership is the ONLY thing that distinguishes a flag downstream, so a new
#: flag that is not listed here silently becomes a z-score.
RAW_FLAG_FIELDS: frozenset[str] = frozenset({
    "sop_dissent_gt_10", "sop_dissent_gt_20",
    "auditor_dissent_gt_05", "auditor_dissent_gt_10",
})

#: THE ONLY fields that also get a peer-relative leg. Everything this module emits ships RAW
#: as `f_<name>`; these ten additionally get `f_<name>_vs_peers`. Nothing gets `_xs`.
#:
#: ⚠ WHY NO `_xs` ANYWHERE IN THIS MODULE, measured 2026-09-07. `_xs` is
#: `rank(axis=1, pct=True)` -- a per-DATE monotone map of the raw level -- so the per-date
#: Spearman correlation between raw and `_xs` is **1.0000 on all 14 fields tested**, and a
#: within-date ranking model cannot tell them apart. That is an identity, not an estimate. The
#: one thing `_xs` can buy is removing a LEVEL TREND that a pooled model would latch onto, and
#: only the auditor levels have one (cross-sectional median 0.012 -> 0.047, a 3.9x secular
#: rise as proxy advisers turned on audit firms). The decision was to drop `_xs` regardless and
#: keep the magnitude instead: a dissent fraction is absolutely interpretable, and a pooled
#: model that needs de-trending can difference the raw level.
#:
#: ⚠ WHY ONLY THESE TEN GET PEERS. Between-sector share of cross-sectional variance is
#: **2.1%-6.6%** for every field here, against **9.0% for profitMargins** and **8.1% for
#: totalRevenue** on the identical measure -- shareholder dissent is a FIRM-SPECIFIC event, not
#: a sector characteristic. So the peer leg is kept only for the bounded LEVELS, where "high
#: for this basket" is at least a coherent question, and dropped for:
#:   * the seven `_delta_1y` fields and the five spreads/excesses -- already differences
#:     centred on zero, so peer-relativizing is a second relativization of the same quantity;
#:   * `management_dissent_spread`, whose peer leg measured 2.2% coverage on 102 tickers
#:     because `min_peers=3` cannot be met on a field present in 17.7% of ballots;
#:   * the four flags, where a peer z of a Bernoulli draw over ~7 peers reads as "how many of
#:     my peers also tripped this", close to the opposite of the intended signal.
#:
#: ⚠ `sop_against_pct` AND `auditor_vote_against_pct` LEFT THIS SET AND THE MODULE ENTIRELY on
#: 2026-09-08, taking their peer legs with them. `_dissent` is `(against + abstain) / valid` and
#: `_against_pct` is `against / valid`, so the two differ ONLY by abstentions -- a mean 0.668 pp
#: of the ballot on say-on-pay and 0.222 pp on the auditor vote. Measured on the live part:
#:
#:     pair                                                 Pearson    n_dissent   n_against
#:     f_sop_dissent            ~ f_sop_against_pct          0.9949    1,627,822   1,627,822
#:     f_auditor_vote_dissent   ~ f_auditor_vote_against_pct 0.9867    1,745,332   1,745,078
#:     f_sop_dissent_vs_peers   ~ f_sop_against_pct_vs_peers 0.9849    1,605,288   1,605,288
#:     f_auditor_..._vs_peers   ~ f_auditor_..._pct_vs_peers 0.9491    1,733,585   1,733,244
#:
#: WHICH TWIN SURVIVES, decided on coverage first. For the auditor pair `_dissent` is a STRICT
#: SUPERSET: 254 cells and 6 distinct values more, and NOT ONE cell where only `_against_pct`
#: exists (measured both directions -- `only_dissent` 254, `only_against` 0). Keeping it loses
#: nothing. The say-on-pay pair is an exact tie (identical non-null counts, identical 6,669
#: distinct values), so two other arguments decide it:
#:
#:   1. `sop_dissent` is the PARENT of five derived features -- `_excess_10`, `_excess_20`,
#:      `_gt_10`, `_gt_20`, `_delta_1y`. Dropping `_against_pct` costs nothing structurally;
#:      dropping `_dissent` would orphan five columns.
#:   2. ABSTENTION IS OPPOSITION in these two votes specifically. Auditor ratification is the
#:      cheapest available protest vote and say-on-pay is advisory, so an abstention is a
#:      deliberate withholding rather than an absence -- which is why ISS reads against +
#:      abstain. `_against_pct` throws that away.
#:
#: ⚠ THE FOURTH ROW IS NOT ITSELF EVIDENCE OF REDUNDANCY. At r = 0.9491 the auditor PEER pair
#: sits below the 0.985 line and was never flagged; `f_auditor_vote_against_pct_vs_peers` goes
#: because a peer z of a column that no longer exists cannot be built, not because it duplicates
#: anything. Four columns leave, not three: 106 -> 102 features (87 -> 85 raw, 18 -> 16 peer,
#: 1 self-history unchanged).
PEER_RELATIVE_FIELDS: frozenset[str] = frozenset({
    "sop_dissent",
    "board_dissent_mean", "board_dissent_median", "board_dissent_max", "board_dissent_p90",
    "board_dissent_breadth_10", "board_dissent_breadth_20",
    "board_pct_nominees_below_70_support", "ceo_director_dissent",
    "auditor_vote_dissent",
})

#: The say-on-pay family (proposal-level tallies).
_SOP_LEVELS: tuple[str, ...] = (
    "sop_dissent", "sop_dissent_excess_10", "sop_dissent_excess_20",
    "sop_dissent_gt_10", "sop_dissent_gt_20",
)
_SOP_DELTAS: dict[str, str] = {"sop_dissent_delta_1y": "sop_dissent"}

#: The board-wide (family 2) and CEO-specific (family 3) fields. They share a history frame
#: because `ceo_excess_dissent` differences the CEO's own dissent against the board median
#: computed from the SAME ballot -- reading the election rows twice would be both wasteful
#: and a chance for the two legs to drift apart.
_ELECTION_LEVELS: tuple[str, ...] = (
    "board_dissent_mean", "board_dissent_median", "board_dissent_max", "board_dissent_p90",
    "board_dissent_breadth_10", "board_dissent_breadth_20",
    "board_pct_nominees_below_70_support",
    "ceo_director_dissent", "ceo_excess_dissent",
    "management_dissent_spread", "ceo_vs_nonemployee_dissent",
)
_ELECTION_DELTAS: dict[str, str] = {
    "board_dissent_mean_delta_1y": "board_dissent_mean",
    "board_dissent_median_delta_1y": "board_dissent_median",
    "board_dissent_max_delta_1y": "board_dissent_max",
    "board_dissent_breadth_10_delta_1y": "board_dissent_breadth_10",
    "ceo_dissent_delta_1y": "ceo_director_dissent",
}

#: The auditor family. Its thresholds are deliberately TIGHTER than say-on-pay's: auditor
#: ratification normally passes with overwhelming support, so 5% opposition is already the
#: tail, where say-on-pay routinely runs to 10%.
_AUDITOR_LEVELS: tuple[str, ...] = (
    "auditor_vote_dissent",
    "auditor_dissent_gt_05", "auditor_dissent_gt_10",
)
_AUDITOR_DELTAS: dict[str, str] = {"auditor_vote_dissent_delta_1y": "auditor_vote_dissent"}

#: Every field this module emits. All of them expire (D21) -- there is no vote-derived
#: quantity that stays true about a year with no meeting.
EVENT_FIELDS: frozenset[str] = frozenset(
    _SOP_LEVELS + tuple(_SOP_DELTAS)
    + _ELECTION_LEVELS + tuple(_ELECTION_DELTAS)
    + _AUDITOR_LEVELS + tuple(_AUDITOR_DELTAS)
)


# --------------------------------------------------------------------------- primitives --
def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """`df[col]` as float, or an all-NaN column when the source does not carry it."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _valid_votes(against: pd.Series, abstain: pd.Series, favour: pd.Series) -> pd.Series:
    """The dissent DENOMINATOR: for + against + abstain, broker non-votes excluded (D11).

    NaN where the total is <= 0 -- an all-zero tally is a failed extraction, and dividing by
    it would manufacture a value out of nothing.

    ⚠ THE THREE LEGS ARE NOT INTERCHANGEABLE, and treating them as such fabricates revolts.
    A missing `votes_against` or `votes_abstain` is read as zero, which is right: a
    proposal that drew no abstentions is routinely printed with that line simply omitted, and
    31 say-on-pay rows / 33 auditor rows are exactly that shape (the `for` leg present, one of
    the other two absent). A missing `votes_FOR` is a different animal entirely -- with
    `fill_value=0` the ratio collapses to (against+abstain)/(against+abstain) = **1.0**, i.e.
    a unanimous revolt conjured out of a parse failure.

    Measured on the live table (2026-09-07): 9 say-on-pay and 8 auditor rows carry against and
    abstain but no `votes_for`, and every one of them landed at dissent == 1.0 before this
    guard -- IEX alone contributed six consecutive years of fictional 100% opposition, which
    is what the top-20 eyeball in the phase verification was there to catch. So the favour leg
    is REQUIRED and the other two are optional. See `2026-09-07-def14a-coverage-defects.md`
    for the upstream extraction bug; this guard is the consumer-side containment, not its fix.
    """
    total = favour.add(against, fill_value=0).add(abstain, fill_value=0)
    return total.where((total > 0) & favour.notna())


def _dissent(against: pd.Series, abstain: pd.Series, favour: pd.Series) -> pd.Series:
    """Share of the votes cast that opposed or abstained. NaN when the denominator is unusable."""
    return against.add(abstain, fill_value=0) / _valid_votes(against, abstain, favour)


def _sanitize(s: pd.Series, tally: dict[str, int], label: str) -> pd.Series:
    """NaN out values outside [0, 1] and COUNT them, rather than clipping them into range.

    A ratio above 1 or below 0 means the tally that produced it is internally inconsistent --
    a mis-parsed column, a transposed pair -- so the value carries no information and clipping
    it to the boundary would disguise a parse failure as an extreme but legitimate vote. The
    count is the data-quality signal that says how often that happens (GPT §2.1).
    """
    bad = s.notna() & ((s < 0.0) | (s > 1.0))
    n = int(bad.sum())
    if n:
        tally[f"out of [0,1] (nulled): {label}"] = n
    return s.mask(bad)


def _flag(x: pd.Series, threshold: float) -> pd.Series:
    """`x >= threshold` as 1.0/0.0, PRESERVING NaN.

    ⚠ `(x >= t).astype(float)` maps NaN to 0.0, turning "the vote could not be read" into
    "there was no dissent" -- the single most damaging silent bug available in this module,
    because it would populate a tail flag densest exactly where the data is worst.
    """
    return x.ge(threshold).astype("float64").where(x.notna())


def _collapse(hist: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Order the interim filing-grain frame so a duplicate (ticker, as_of) resolves the same
    way on every run.

    A company can file two 8-Ks on one day, or amend one to restate a tally.
    `fundamentals_to_daily` pivots with `aggfunc="last"`, which takes the last row IN FRAME
    ORDER -- deterministic only if the frame is sorted first. Accession then proposal
    sequence gives a stable total order; the later accession wins, which is what an
    amendment should do.
    """
    seq = (pd.to_numeric(rows["proposal_seq"], errors="coerce").to_numpy()
           if "proposal_seq" in rows.columns else 0.0)
    out = hist.assign(_acc=rows["accession_number"].astype(str).to_numpy(), _seq=seq)
    out = out.dropna(subset=["ticker", "as_of"])
    out = out.sort_values(["ticker", "as_of", "_acc", "_seq"], kind="mergesort")
    return out.drop(columns=["_acc", "_seq"]).reset_index(drop=True)


def _rows_of(votes: pd.DataFrame, proposal_type: str) -> pd.DataFrame:
    """The rows of one proposal family, with a usable filing date.

    A NULL `proposal_type` (the extractor returned a value outside `PROPOSAL_TYPES`) is
    skipped rather than bucketed anywhere -- an unrecognised proposal is not evidence about
    say-on-pay just because say-on-pay is the family being built.
    """
    if "proposal_type" not in votes.columns or "filing_date" not in votes.columns:
        return votes.iloc[0:0]
    rows = votes[votes["proposal_type"].astype("object") == proposal_type]
    return rows[pd.to_datetime(rows["filing_date"], errors="coerce").notna()]


def _as_of(rows: pd.DataFrame) -> pd.Series:
    """The FILING date, never `meeting_date`: a tally is not public until the 8-K carrying it
    is filed, and using the meeting date would leak the result by however many days the
    company took to file."""
    return pd.to_datetime(rows["filing_date"], errors="coerce")


def _proposal_legs(rows: pd.DataFrame, tally: dict[str, int], label: str,
                   ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """The three vote legs and the denominator for a proposal-level family, with the rows that
    lost their `votes_for` leg COUNTED rather than quietly dropped.

    The count is the whole point: these rows are an upstream extraction failure, and a
    consumer that silences them without saying so leaves the next reader to rediscover the
    same six years of IEX by hand.
    """
    against, abstain, favour = (_num(rows, "votes_against"), _num(rows, "votes_abstain"),
                                _num(rows, "votes_for"))
    lost = int((favour.isna() & (against.notna() | abstain.notna())).sum())
    if lost:
        tally[f"no votes_for leg (row dropped): {label}"] = lost
    return against, abstain, favour, _valid_votes(against, abstain, favour)


# ----------------------------------------------------------------------------- family 1 --
def _say_on_pay_history(votes: pd.DataFrame, tally: dict[str, int]) -> pd.DataFrame | None:
    """Interim filing-grain frame for the say-on-pay family.

    `say_on_pay_frequency` is deliberately NOT included: a frequency vote chooses an interval
    (1/2/3 years), so its `votes_for` / `votes_against` are NULL by construction and treating
    it as an approval vote would read a ballot-design question as dissent about pay.
    """
    rows = _rows_of(votes, "say_on_pay")
    tally["rows: say_on_pay"] = len(rows)
    if rows.empty:
        return None
    # `valid` is unpacked and unused: it was the denominator of the retired
    # `_against_pct` leg. Kept in the unpack rather than dropped from
    # `_proposal_legs`, whose other callers still need it.
    against, abstain, favour, _valid = _proposal_legs(rows, tally, "say_on_pay")
    dissent = _sanitize(_dissent(against, abstain, favour), tally, "sop_dissent")
    hist = pd.DataFrame({
        "ticker": rows["ticker"].astype(str),
        "as_of": _as_of(rows),
        "sop_dissent": dissent,
        # `sop_against_pct` (= against / valid) used to be emitted here. Removed rather than
        # left computed-but-unselected, because `_sanitize` writes tally entries and a build
        # log reporting the hygiene of a column nobody ships is a trap for the next reader.
        # See `PEER_RELATIVE_FIELDS` for the r = 0.9949 that retired it.
        # Excess-over-threshold: zero for a normal vote, and LINEAR in the tail beyond it, so
        # a 35% revolt is distinguishable from a 21% one where the flag below saturates.
        "sop_dissent_excess_10": (dissent - 0.10).clip(lower=0.0),
        "sop_dissent_excess_20": (dissent - 0.20).clip(lower=0.0),
        "sop_dissent_gt_10": _flag(dissent, 0.10),
        "sop_dissent_gt_20": _flag(dissent, 0.20),
    })
    return _collapse(hist, rows)


# ----------------------------------------------------------------------------- family 2 --
def _one_nominee_dissent(item: object) -> float:
    """Dissent for a single nominee entry of `nominee_votes_json`, or NaN if unusable."""
    if not isinstance(item, dict):
        return np.nan
    legs = []
    for key in ("votes_for", "votes_against", "votes_abstain"):
        try:
            legs.append(float(item.get(key)))          # type: ignore[arg-type]
        except (TypeError, ValueError):
            legs.append(np.nan)
    # Same asymmetry as `_valid_votes`: the FOR leg is required (without it the ratio is a
    # tautological 1.0), against and abstain may be absent and count as zero.
    if np.isnan(legs[0]):
        return np.nan
    valid = float(np.nansum(legs))
    if not valid > 0:
        return np.nan
    return float(np.nansum(legs[1:])) / valid          # (against + abstain) / valid


def _nominee_dissent(payload: object, tally: dict[str, int]) -> list[float]:
    """Per-nominee dissent for one election row. `[]` when the row carries no usable tally.

    A malformed blob is a SKIPPED ROW, never an exception: one bad JSON string out of 7,884
    must not take the whole cube build down, and the count of what was skipped is tallied so
    the loss is visible instead of inferred.
    """
    if payload is None or (isinstance(payload, float) and np.isnan(payload)):
        return []
    try:
        items = json.loads(payload)                    # type: ignore[arg-type]
    except (TypeError, ValueError):
        tally["nominee_votes_json malformed (row skipped)"] = (
            tally.get("nominee_votes_json malformed (row skipped)", 0) + 1)
        return []
    if not isinstance(items, list):
        tally["nominee_votes_json not a list (row skipped)"] = (
            tally.get("nominee_votes_json not a list (row skipped)", 0) + 1)
        return []

    out: list[float] = []
    for item in items:
        d = _one_nominee_dissent(item)
        if np.isnan(d):
            continue
        if not 0.0 <= d <= 1.0:
            tally["out of [0,1] (nulled): nominee dissent"] = (
                tally.get("out of [0,1] (nulled): nominee dissent", 0) + 1)
            continue
        out.append(d)
    return out


def _bucket_dissent(rows: pd.DataFrame, bucket: str, tally: dict[str, int]) -> pd.Series:
    """Dissent against one ROLE bucket of the ballot, from the stored per-bucket sums.

    ⚠ Gated on the bucket actually having a nominee. A CEO who did not stand -- a classified
    board electing a third of its directors, or a CEO who is not a director at all -- is
    UNKNOWN, not unopposed, and a zero here would be indistinguishable from a unanimous
    endorsement. The per-bucket vote columns can be 0 for both cases, so `n_nominees_<bucket>`
    is the only thing that separates them.
    """
    present = _num(rows, f"n_nominees_{bucket}") > 0
    d = _dissent(_num(rows, f"votes_against_{bucket}"), _num(rows, f"votes_abstain_{bucket}"),
                 _num(rows, f"votes_for_{bucket}"))
    return _sanitize(d.where(present), tally, f"{bucket}_dissent")


def _election_history(votes: pd.DataFrame, tally: dict[str, int]) -> pd.DataFrame | None:
    """Interim filing-grain frame for the board-wide AND CEO-specific families.

    Built in one pass because `ceo_excess_dissent` differences the CEO's dissent against the
    board median FROM THE SAME BALLOT: the two families are one computation wearing two
    names, and splitting them would mean parsing `nominee_votes_json` twice.
    """
    rows = _rows_of(votes, "director_election")
    tally["rows: director_election"] = len(rows)
    if rows.empty:
        return None

    lists = rows.get("nominee_votes_json", pd.Series(index=rows.index, dtype="object"))
    arrays = [np.asarray(_nominee_dissent(p, tally), dtype="float64") for p in lists]

    def agg(fn) -> pd.Series:
        return pd.Series([fn(a) if a.size else np.nan for a in arrays],
                         index=rows.index, dtype="float64")

    # A single-nominee election makes mean == median == max == p90. That is not a bug, it is
    # what a one-name ballot says; the peer panel's PEER_DISPERSION_FLOOR absorbs the low
    # dispersion that follows.
    #
    # ⚠ `board_dissent_max` IS NOT ALWAYS AN UNPOPULAR INCUMBENT. A dissident shareholder
    # nominee on the same ballot is crushed with near-zero support and pins the max at ~1.0 --
    # GM 2014/2015 (John Lauve, Dean Fitzpatrick: 1 vote for, 1.197 BILLION against) and AXP
    # 2010-2013 (Peter Lindner, 11 for) are the whole of it. Those values are CORRECT and must
    # not be filtered: the tally is real. But they mean the opposite thing economically -- a
    # challenger being rejected, not the board being repudiated -- so the max is the family's
    # least interpretable member. Measured 2026-09-07: 23 of 70,205 nominee dissents exceed
    # 0.90, touching 13 of 7,884 election rows (0.165%). Small enough to leave alone, large
    # enough that `board_dissent_mean` / `_median` are the ones to lean on.
    mean = agg(np.mean)
    median = agg(np.median)
    ceo = _bucket_dissent(rows, "ceo", tally)
    non_employee = _bucket_dissent(rows, "non_employee", tally)

    # `n_nominees_below_70pct` is derived from `min_support_pct`'s basis -- for/(for+against),
    # WITHOUT abstentions -- which is a different denominator from D11's. Naming this ratio
    # `..._support` rather than `..._dissent` is what keeps the two bases from being read as
    # one family; `min_support_pct` itself is deliberately not re-exported for the same reason.
    n_nominees = _num(rows, "n_nominees")
    below_70 = _num(rows, "n_nominees_below_70pct")

    hist = pd.DataFrame({
        "ticker": rows["ticker"].astype(str),
        "as_of": _as_of(rows),
        "board_dissent_mean": mean,
        "board_dissent_median": median,
        "board_dissent_max": agg(np.max),
        "board_dissent_p90": agg(lambda a: float(np.quantile(a, 0.90))),
        "board_dissent_breadth_10": agg(lambda a: float((a >= 0.10).mean())),
        "board_dissent_breadth_20": agg(lambda a: float((a >= 0.20).mean())),
        "board_pct_nominees_below_70_support": below_70 / n_nominees.where(n_nominees > 0),
        "ceo_director_dissent": ceo,
        # D15: the median INCLUDES the CEO's own nominee row, so this understates the excess
        # by roughly 1/n_nominees of it. Documented approximation, not an oversight -- the
        # exact leave-one-out version needs the CEO's name matched inside the JSON, which
        # phase 1 §4 measured as recovering 3 rows out of 908 and is not worth the coupling.
        "ceo_excess_dissent": ceo - median,
        "management_dissent_spread": _bucket_dissent(rows, "exec_officer", tally) - non_employee,
        # Exact, unlike `ceo_excess_dissent`: the non-employee bucket excludes the CEO by
        # construction, so there is no self-inclusion to correct for.
        "ceo_vs_nonemployee_dissent": ceo - non_employee,
    })

    _log_ceo_ceiling(rows, tally)
    return _collapse(hist, rows)


def _log_ceo_ceiling(rows: pd.DataFrame, tally: dict[str, int]) -> None:
    """Tally the three-way split behind the CEO leg's 76.5% coverage ceiling.

    ⚠ THIS IS A BALLOT PROPERTY, NOT A JOIN DEFECT, and the numbers exist so the next reader
    does not "fix" it. Phase 1 §4 tested the obvious repair -- matching the prior proxy's
    `ceo_name_proxy` against the names inside `nominee_votes_json` -- and it recovers 3 of the
    908 no-CEO-with-unmatched rows, moving coverage 76.5% -> 76.6%. Those ballots simply do
    not contain the CEO: under a staggered board only a subset stands each year, and the
    unmatched nominees are other directors missing from the prior proxy, not a hidden CEO.

    `n_nominees_unmatched` doubles as the role map's error rate, so the share of ballots that
    are mostly unmatched is tallied too: a family built on a bad role map should be visible.
    """
    n_ceo = _num(rows, "n_nominees_ceo").fillna(0.0)
    n_unmatched = _num(rows, "n_nominees_unmatched").fillna(0.0)
    n_nominees = _num(rows, "n_nominees")
    tally["elections: CEO resolved"] = int((n_ceo > 0).sum())
    tally["elections: no CEO, no unmatched (not on the ballot)"] = int(
        ((n_ceo <= 0) & (n_unmatched <= 0)).sum())
    tally["elections: no CEO, unmatched present"] = int(
        ((n_ceo <= 0) & (n_unmatched > 0)).sum())
    tally["elections: >50% of nominees unmatched"] = int(
        ((n_unmatched / n_nominees.where(n_nominees > 0)) > 0.5).sum())


# ----------------------------------------------------------------------------- family 4 --
def _auditor_history(votes: pd.DataFrame, tally: dict[str, int]) -> pd.DataFrame | None:
    """Interim filing-grain frame for auditor ratification.

    No `auditor_changed` and no audit-fee-direction feature: a change of audit firm is
    ambiguous without the Item 4.01 reason (a routine rotation and a disagreement over
    accounting look identical in the vote record), so `auditor_name` and the fee block stay
    unconsumed here.
    """
    rows = _rows_of(votes, "auditor_ratification")
    tally["rows: auditor_ratification"] = len(rows)
    if rows.empty:
        return None
    # `valid` is unpacked and unused: it was the denominator of the retired
    # `_against_pct` leg. Kept in the unpack rather than dropped from
    # `_proposal_legs`, whose other callers still need it.
    against, abstain, favour, _valid = _proposal_legs(rows, tally, "auditor_ratification")
    dissent = _sanitize(_dissent(against, abstain, favour), tally, "auditor_vote_dissent")
    hist = pd.DataFrame({
        "ticker": rows["ticker"].astype(str),
        "as_of": _as_of(rows),
        "auditor_vote_dissent": dissent,
        # `auditor_vote_against_pct` used to be emitted here; `_dissent` is a strict SUPERSET
        # of it (254 cells more, 0 the other way) at r = 0.9867. Same reason as say-on-pay's.
        "auditor_dissent_gt_05": _flag(dissent, 0.05),
        "auditor_dissent_gt_10": _flag(dissent, 0.10),
    })
    return _collapse(hist, rows)


# -------------------------------------------------------------------------------- daily --
def _family_frames(hist: pd.DataFrame, levels: tuple[str, ...], deltas: dict[str, str],
                   idx: pd.DatetimeIndex) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Daily wide frames for one family, plus the feature -> history-column map the expiry needs.

    Levels go through `fundamentals_to_daily` and deltas through `fiscal_change_to_daily`, so
    every point-in-time forward-fill in this module is the same tested one the fundamentals
    panels use -- the interim frame's `[ticker, as_of, <fields>]` shape exists precisely so no
    new daily-frame machinery has to be written here.

    `kind="diff"`, never `"pct"`: dissent is already a fraction, so 2% -> 11% is +9pp of the
    electorate, not +450% of anything meaningful.

    `periods=1`, and `infer_yoy_periods` is deliberately NOT called: it reads the median gap
    between filings, so one extra special meeting in a ticker's history would silently make
    its "year-over-year" change span two meetings. Annual meetings ARE the period here.
    """
    frames: dict[str, pd.DataFrame] = {}
    sources: dict[str, str] = {}
    for name in levels:
        f = fundamentals_to_daily(hist, name, idx)
        if not f.empty and f.notna().any().any():
            frames[name] = f
            sources[name] = name
    for name, src in deltas.items():
        f = fiscal_change_to_daily(hist, src, idx, kind="diff", periods=1)
        if not f.empty and f.notna().any().any():
            frames[name] = f
            # The delta's provenance is the LEVEL's filing -- the later leg is what made the
            # change knowable -- so the expiry ages it against the level's column, not against
            # a column named after the delta that does not exist in the history frame.
            sources[name] = src
    return frames, sources


def vote_dissent_fields(
    votes: pd.DataFrame | None, idx: pd.DatetimeIndex,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """(daily wide frames keyed by feature name, data-quality tallies).

    Empty dicts when `votes` is None/empty or carries no `proposal_type` column -- the
    caller's optional-source semantics. An absent vote archive means "no features", never an
    exception, because `sec_8k_votes` accrues as the 8-K fetcher runs and a half-built
    database must still produce a cube.

    The returned dict mixes peer-panelled fields and raw flags; `RAW_FLAG_FIELDS` is what
    tells them apart.
    """
    tally: dict[str, int] = {}
    if votes is None or votes.empty or "proposal_type" not in votes.columns:
        return {}, tally

    frames: dict[str, pd.DataFrame] = {}
    families = (
        (_say_on_pay_history(votes, tally), _SOP_LEVELS, _SOP_DELTAS),
        (_election_history(votes, tally), _ELECTION_LEVELS, _ELECTION_DELTAS),
        (_auditor_history(votes, tally), _AUDITOR_LEVELS, _AUDITOR_DELTAS),
    )
    for hist, levels, deltas in families:
        if hist is None or hist.empty:
            continue
        family, sources = _family_frames(hist, levels, deltas, idx)
        # Expire against THIS family's own history: the age of a say-on-pay cell is the age of
        # the say-on-pay filing, and a company that held an election but skipped the pay vote
        # must not have its stale pay dissent kept alive by the election's filing date.
        family, expiry = expire_event_fields(family, hist, EVENT_FIELDS, sources=sources)
        for name, (expired, before) in expiry.items():
            if expired:
                tally[f"expired >{GOVERNANCE_EVENT_MAX_AGE_DAYS}d: {name}"] = expired
                tally[f"non-null before expiry: {name}"] = before
        frames.update(family)
    return frames, tally
