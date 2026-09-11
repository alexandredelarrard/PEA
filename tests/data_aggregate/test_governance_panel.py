"""
Governance / executive-pay features from the DEF 14A LLM archive
(src/data_aggregate/utils/governance/panel.py).

Synthetic proxy history (annual) + a quarterly fundamentals history, to prove:
  * CEO total-comp YoY growth and the pay-vs-revenue-growth MISALIGNMENT signal,
  * board/pay level fields flow through, and
  * the panel is point-in-time and empty when the archive is absent.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.governance.panel import (
    _def14a_raw_fields,
    _governance_fields,
    build_governance_feature_panel,
)
from src.data_aggregate.utils.governance.staleness import LEGACY_EXEMPT_FROM_EXPIRY
from src.data_aggregate.utils.governance.vote_dissent_features import (
    PEER_RELATIVE_FIELDS as VOTE_PEER_FIELDS,
)


def _def14a() -> pd.DataFrame:
    """Three annual proxies for four tickers.

    ⚠ EVERY FIELD THAT CARRIES AN ENCODING VARIES ACROSS TICKERS, and that is a
    requirement of the contract test rather than decoration. `peer_relative` returns NaN for a
    basket whose peers all report the SAME value -- its declared zero-dispersion policy -- so a
    fixture with `board_size = 10` everywhere silently drops `f_board_size_vs_peers`, and the
    encoding test then fails against a fixture artefact instead of against the code. Likewise
    `avg_board_tenure` MOVES year to year, because `_vs_hist` divides by the firm's own trailing
    std and a constant history has none.
    """
    rows = []
    # (ticker, base CEO pay, CEO-since year, pct female, board size, equity-pay pct,
    #  insider-ownership pct, base board tenure) — all of them vary cross-sectionally
    specs = [("AAA", 10e6, 2010, 0.40, 9, 0.55, 0.0050, 6.0),
             ("BBB", 12e6, 2018, 0.20, 11, 0.70, 0.0120, 8.0),
             ("CCC", 8e6, 2005, 0.50, 13, 0.80, 0.0300, 9.5),
             ("DDD", 15e6, 2022, 0.30, 10, 0.62, 0.0020, 7.0)]
    # The phase-5 provision block, per ticker across the three proxies. Each column carries at
    # least one 0->1 and one 1->0 somewhere in the panel, so every transition flag has an event
    # to find; `avg_other_public_boards` moves so its delta exists.
    prov = {
        "AAA": {"classified_board": [0, 0, 1], "dual_class_shares": [0, 0, 1],
                "poison_pill": [0, 0, 1], "majority_voting": [1, 1, 0],
                "ceo_is_board_chair": [0, 0, 1], "independent_chair": [1, 1, 0],
                "lead_independent_director": [1, 1, 0], "avg_other_public_boards": [1.0, 1.3, 1.6],
                "auditor_name": ["Ernst & Young LLP"] * 3, "auditor_since_year": [np.nan] * 3},
        "BBB": {"classified_board": [1, 1, 0], "dual_class_shares": [1, 1, 1],
                "poison_pill": [1, 1, 0], "majority_voting": [0, 0, 1],
                "ceo_is_board_chair": [1, 1, 0], "independent_chair": [0, 0, 1],
                "lead_independent_director": [0, 0, 1], "avg_other_public_boards": [2.0, 1.8, 1.5],
                "auditor_name": ["KPMG LLP"] * 3, "auditor_since_year": [1998] * 3},
        "CCC": {"classified_board": [0, 0, 0], "dual_class_shares": [0, 0, 0],
                "poison_pill": [0, 0, 0], "majority_voting": [1, 1, 1],
                "ceo_is_board_chair": [0, 0, 0], "independent_chair": [1, 1, 1],
                "lead_independent_director": [1, 1, 1], "avg_other_public_boards": [0.5, 0.7, 0.9],
                "auditor_name": ["Deloitte & Touche LLP"] * 3, "auditor_since_year": [2011] * 3},
        "DDD": {"classified_board": [1, 1, 1], "dual_class_shares": [0, 0, 0],
                "poison_pill": [0, 0, 0], "majority_voting": [0, 0, 0],
                "ceo_is_board_chair": [1, 1, 1], "independent_chair": [0, 0, 0],
                "lead_independent_director": [0, 1, 1], "avg_other_public_boards": [3.0, 2.6, 2.2],
                "auditor_name": ["Grant Thornton LLP"] * 3, "auditor_since_year": [np.nan] * 3},
    }
    for tkr, pay0, since_yr, fem, board, eq, insider, tenure0 in specs:
        for i, yr in enumerate((2023, 2024, 2025)):
            rows.append({
                **{k: v[i] for k, v in prov[tkr].items()},
                "ticker": tkr, "as_of": pd.Timestamp(f"{yr}-04-01"),
                "ceo_total_comp": pay0 * (1.20 ** i),        # +20%/yr
                "ceo_pay_ratio": 200 + 50 * i,
                "ceo_equity_pay_pct": eq,
                "pct_independent_directors": 0.85,
                "pct_female_directors": fem,
                "board_size": board,
                "avg_board_tenure": tenure0 + 0.8 * i,       # a board that ages -> a self-history
                "say_on_pay_support_pct": 0.92,
                "insider_ownership_pct": insider,
                "ceo_is_founder": 1.0 if tkr in ("AAA", "CCC") else 0.0,
                "ceo_since_year": since_yr,
            })
    return pd.DataFrame(rows)


#: Real surnames, because `board_turnover` keys on `person_key` and `person_key("Director 1")`
#: is `director` -- placeholder names collapse to ONE key and a five-seat board digests as one.
_SEATS = ("Ann Alder", "Bob Birch", "Cara Cedar", "Dave Dogwood", "Erin Elm", "Finn Fir")


def _directors() -> pd.DataFrame:
    """The per-director child of `_def14a`, for phase 6's board-quality family.

    Ages, tenures and other-board counts all VARY ACROSS TICKERS for the reason the parent
    fixture's docstring gives: `peer_relative` returns NaN on a zero-dispersion basket, so a
    constant fixture would drop a peer leg and the encoding test would then be asserting against
    the fixture. One seat turns over in the last year so `board_turnover` is non-degenerate.
    """
    rows = []
    for k, tkr in enumerate(("AAA", "BBB", "CCC", "DDD")):
        for i, yr in enumerate((2023, 2024, 2025)):
            for s in range(4 + k % 2):
                name = _SEATS[(s + 2) % len(_SEATS)] if (s == 0 and yr == 2025) else _SEATS[s]
                rows.append({
                    "ticker": tkr, "accession_number": f"{tkr}-{yr}",
                    "as_of": pd.Timestamp(f"{yr}-04-01"), "name": name,
                    "age": 50.0 + 4.0 * s + 2.0 * k + i,
                    "tenure_years": 3.0 + 6.0 * s + k + i,
                    "is_independent": float(s > 0),
                    "other_public_company_boards": float((s + k) % 4),
                })
    return pd.DataFrame(rows)


def _director_comp() -> pd.DataFrame:
    """The Item 402(k) child, for phase 6's director-pay family. The cash/equity SPLIT varies by
    ticker -- that is what gives the two fields the peer dispersion their legs need."""
    rows = []
    for k, tkr in enumerate(("AAA", "BBB", "CCC", "DDD")):
        cash_share = 0.25 + 0.15 * k                     # 0.25 / 0.40 / 0.55 / 0.70
        for yr in (2023, 2024, 2025):
            for s in range(4 + k % 2):
                total = 240_000.0 + 20_000.0 * s
                rows.append({
                    "ticker": tkr, "accession_number": f"{tkr}-{yr}",
                    "as_of": pd.Timestamp(f"{yr}-04-01"), "name": _SEATS[s],
                    "total": total,
                    "fees_earned": total * cash_share,
                    "stock_awards": total * (1.0 - cash_share),
                    "option_awards": np.nan, "non_equity_incentive": np.nan,
                    "pension_change": np.nan, "other_compensation": np.nan,
                })
    return pd.DataFrame(rows)


def _votes() -> pd.DataFrame:
    """One annual meeting per ticker-year, carrying all THREE proposal types the four Item 5.07
    families read (`say_on_pay`, `director_election`, `auditor_ratification`).

    ⚠ This fixture exists because the encoding test's `votes=None` guard was VACUOUS without it:
    with no votes supplied in the positive case either, "no vote family survived" was true of
    both branches and proved nothing. Dissent varies by ticker AND rises over time, so the
    `_delta_1y` legs have something to difference and the peer baskets have dispersion.
    """
    rows = []
    for k, tkr in enumerate(("AAA", "BBB", "CCC", "DDD")):
        for i, yr in enumerate((2023, 2024, 2025)):
            acc = f"8K-{tkr}-{yr}"
            meeting = f"{yr}-05-10"
            sop_d = 0.04 + 0.06 * k + 0.03 * i          # 4%..40% against
            aud_d = 0.01 + 0.01 * k + 0.005 * i
            # five nominees, the FIRST of whom is the CEO, dissent rising across the slate
            nominees = []
            for s in range(5):
                d = 0.02 + 0.05 * s + 0.02 * k + 0.01 * i
                nominees.append({"name": _SEATS[s], "votes_for": 1_000.0 * (1.0 - d),
                                 "votes_against": 1_000.0 * d, "votes_abstain": 0.0,
                                 "votes_broker_non_votes": 200.0})
            ceo_d = 0.02 + 0.02 * k + 0.01 * i
            common = {"ticker": tkr, "accession_number": acc, "filing_date": meeting,
                      "vote_standard": None, "votes_broker_non_votes": 200.0,
                      "votes_abstain": 0.0, "nominee_votes_json": None, "n_nominees": None,
                      "n_nominees_below_70pct": None}
            for bucket in ("ceo", "exec_officer", "non_employee", "unmatched"):
                for leg in ("for", "against", "abstain", "broker_non_votes"):
                    common[f"votes_{leg}_{bucket}"] = None
                common[f"n_nominees_{bucket}"] = None
            rows.append({**common, "proposal_seq": 1.0, "proposal_type": "say_on_pay",
                         "votes_for": 1_000.0 * (1.0 - sop_d),
                         "votes_against": 1_000.0 * sop_d})
            rows.append({**common, "proposal_seq": 2.0, "proposal_type": "director_election",
                         "votes_for": None, "votes_against": None,
                         "nominee_votes_json": json.dumps(nominees), "n_nominees": 5.0,
                         "n_nominees_below_70pct": float(k >= 2),
                         "n_nominees_ceo": 1.0, "n_nominees_non_employee": 4.0,
                         "votes_for_ceo": 1_000.0 * (1.0 - ceo_d),
                         "votes_against_ceo": 1_000.0 * ceo_d, "votes_abstain_ceo": 0.0,
                         "votes_for_non_employee": 4_000.0 * (1.0 - 0.10),
                         "votes_against_non_employee": 4_000.0 * 0.10,
                         "votes_abstain_non_employee": 0.0})
            rows.append({**common, "proposal_seq": 3.0,
                         "proposal_type": "auditor_ratification",
                         "votes_for": 1_000.0 * (1.0 - aud_d),
                         "votes_against": 1_000.0 * aud_d})
    return pd.DataFrame(rows)


def _fundamentals() -> pd.DataFrame:
    rows = []
    for tkr in ("AAA", "BBB", "CCC", "DDD"):
        for i, q in enumerate(pd.date_range("2022-12-31", periods=14, freq="QE")):
            rows.append({"ticker": tkr, "as_of": q, "totalRevenue": 1000 * (1.03 ** i)})  # ~12%/yr
    return pd.DataFrame(rows)


def test_governance_fields_pay_growth_and_misalignment():
    idx = pd.date_range("2023-01-02", "2026-07-01", freq="B")
    F = _governance_fields(_def14a(), idx, _fundamentals())

    assert "ceo_pay_growth" in F and "ceo_pay_vs_revenue_growth" in F
    for name in ("ceo_pay_ratio", "pct_independent_directors", "pct_female_directors",
                 "avg_board_tenure", "insider_ownership_pct", "board_size"):
        assert name in F, f"missing level field {name}"
    # ⚠ `founder_ceo` is NOT here any more: it is a 1/0 flag and now ships RAW via
    # `_def14a_raw_fields`, because both peer encodings were degenerate on a binary (the peer z
    # of a Bernoulli draw over ~7 peers survived on 239 tickers / 32% of cells).
    assert "founder_ceo" not in F, "a binary must not be peer-panelled"
    # `say_on_pay_support` left with it, for the same reason in a different shape: 0.60 support
    # is a near-revolt at any firm in any sector, so the absolute fraction IS the signal.
    assert "say_on_pay_support" not in F, "an absolute fraction must not be peer-panelled"
    R = _def14a_raw_fields(_def14a(), idx)
    assert R["say_on_pay_support"]["AAA"].dropna().iloc[-1] == pytest.approx(0.92)
    assert R["founder_ceo"]["AAA"].dropna().iloc[-1] == 1.0
    assert R["founder_ceo"]["BBB"].dropna().iloc[-1] == 0.0

    # CEO tenure accrues by CALENDAR year (not a stale as_of snapshot): AAA CEO since
    # 2010 -> 15y on a 2025 date and 16y on a 2026 date; CCC (since 2005) outranks DDD (2022).
    assert "ceo_tenure" in F
    ten = F["ceo_tenure"]
    aaa_2025 = ten.loc[ten.index.year == 2025, "AAA"].dropna()
    aaa_2026 = ten.loc[ten.index.year == 2026, "AAA"].dropna()
    assert aaa_2025.iloc[-1] == pytest.approx(2025 - 2010)   # 15
    assert aaa_2026.iloc[-1] == pytest.approx(2026 - 2010)   # 16 -> grows with the calendar
    last = idx[-1]
    assert ten.loc[last, "CCC"] > ten.loc[last, "DDD"]       # 2005 vs 2022 start

    # CEO pay grew ~20%/yr; the latest observed pay_growth should be ~0.20
    pay_g = F["ceo_pay_growth"]["AAA"].dropna()
    assert pay_g.iloc[-1] == pytest.approx(0.20, abs=1e-6)
    # misalignment = pay growth (~20%) - revenue TTM growth (~12%) -> clearly positive
    mis = F["ceo_pay_vs_revenue_growth"]["AAA"].dropna()
    assert mis.iloc[-1] > 0.05

    print("\n=== SANITY CHECK: governance pay dynamics + CEO tenure ===")
    print(f"  ceo_pay_growth(last)={pay_g.iloc[-1]:.3f} (~0.20); "
          f"pay_vs_revenue_growth(last)={mis.iloc[-1]:.3f} (>0 = pay outpacing revenue).")
    print(f"  ceo_tenure AAA(since 2010): {aaa_2025.iloc[-1]:.0f}y in 2025 -> {aaa_2026.iloc[-1]:.0f}y in 2026 "
          f"(accrues by calendar year); CCC {ten.loc[last, 'CCC']:.0f}y > DDD {ten.loc[last, 'DDD']:.0f}y. Validated.")


def test_governance_panel_encoding_and_empty_guard():
    """The panel's ENCODING contract, which is the whole point of the 2026-09-08 review.

    Every governance feature ships RAW. On top of that exactly two encodings survived
    measurement, and `_xs` survived none:

      * `_vs_peers` on the EIGHTEEN fields with a real peer norm, on between-sector variance share:
        the four legacy levels (12.4% / 12.2% / 8.6% / 7.7%, against 9.0% for `profitMargins`),
        phase 3's twelve vote-dissent legs, and phase 6's two director pay-MIX shares
        (11.94% / 8.17%). Phases 4 and 5 added thirty-eight fields between them and earned not
        one peer leg, so the two that pass here are evidence the yardstick separates rather
        than simply refuses;
      * `_vs_hist` on `avg_board_tenure` alone -- the only field whose self-history z held its
        sign across both halves of the sample (+0.0128 then +0.0270) while raw and peer flipped;
      * NO `_xs` anywhere. It is a per-date monotone map of raw (per-date Spearman exactly
        1.0), and its one consumer -- the `governance` composite -- was deleted as carrying no
        predictive power.
    """
    idx = pd.date_range("2023-01-02", "2026-07-01", freq="B")
    peers = {t: {p: 1.0 for p in ("AAA", "BBB", "CCC", "DDD") if p != t}
             for t in ("AAA", "BBB", "CCC", "DDD")}
    panel, _ = build_governance_feature_panel(_def14a(), peers, idx,
                                              fundamentals_history=_fundamentals(),
                                              votes=_votes(),
                                              directors=_directors(),
                                              director_comp=_director_comp())
    cols = [c for c in panel.columns if c not in ("date", "ticker")]
    assert not panel.empty

    # 1. no `_xs` leg survives anywhere in the panel
    assert not [c for c in cols if c.endswith("_xs")], "the _xs encoding was retired"

    # 2. the differences ship RAW and are never peer-relativized: peer-z inverts their sign on
    #    29.3% / 36.9% of the positive cases, and the zero point IS the thesis
    for name in ("f_ceo_pay_vs_revenue_growth", "f_ceo_pay_growth"):
        assert name in cols
        assert f"{name}_vs_peers" not in cols
    # 3. a binary and an absolute fraction likewise
    assert "f_founder_ceo" in cols and "f_founder_ceo_vs_peers" not in cols
    assert "f_say_on_pay_support" in cols and "f_say_on_pay_support_vs_peers" not in cols
    # 4. the two features whose IC flips sign under every encoding ship raw only, and their
    #    monotone constraints were removed from configs/models/lgbm_modelling.yml with them
    for name in ("f_ceo_tenure", "f_pct_female_directors"):
        assert name in cols and f"{name}_vs_peers" not in cols
    # 5. the four with a genuine peer norm keep BOTH legs
    for name in ("f_board_size", "f_ceo_pay_ratio", "f_ceo_equity_pay_pct",
                 "f_insider_ownership_pct"):
        assert name in cols and f"{name}_vs_peers" in cols
    # 6. the one self-history survivor, and only it
    assert [c for c in cols if c.endswith("_vs_hist")] == ["f_avg_board_tenure_vs_hist"]

    # 7. ⚠ NO INTERACTION COLUMNS ANYWHERE IN THE PANEL (D16). GPT §13 specifies five products
    #    (dissent x pay, chair x dissent, ...); none is built, because LightGBM and the random
    #    forest find a product of two features they already have. The guard is on the assembled
    #    panel rather than on one module, so a later session cannot reintroduce one anywhere.
    assert not [c for c in cols if "_x_" in c], "an interaction feature was reintroduced"
    # and the one leg that existed ONLY inside a product now ships as a plain level
    assert "f_ceo_is_board_chair" in cols
    assert "f_ceo_is_board_chair_vs_peers" not in cols, "a Bernoulli has no peer norm"
    # 8. phase 5's families reached the panel, raw and raw only
    for name in ("f_classified_board_added", "f_governance_deterioration_count",
                 "f_net_governance_change", "f_board_independence_delta_1y",
                 "f_board_busyness", "f_auditor_tenure", "f_auditor_changed",
                 "f_auditor_is_big4"):
        assert name in cols, f"missing phase-5 field {name}"
        assert f"{name}_vs_peers" not in cols, f"{name} has no measured peer norm"

    # 9. phase 6's two families reached the panel, and their encoding is the MEASURED one: the
    #    six board-quality fields carry no peer leg (sector share 2.9%-7.2%, all below the 7.7%
    #    floor) while exactly two of the four director-pay fields do (11.94% / 8.17%). This is
    #    the first new family since phase 3 to earn a peer leg at all, which is also why the
    #    negative half has to be asserted: the yardstick separates, it does not simply refuse.
    for name in ("f_board_turnover", "f_pct_long_tenured", "f_pct_overboarded",
                 "f_board_tenure_dispersion", "f_board_age_dispersion",
                 "f_oldest_director_age"):
        assert name in cols, f"missing phase-6 board-quality field {name}"
        assert f"{name}_vs_peers" not in cols, f"{name} has no measured peer norm"
    for name in ("f_log_median_director_pay", "f_ceo_to_director_pay_ratio"):
        assert name in cols, f"missing phase-6 director-pay field {name}"
        assert f"{name}_vs_peers" not in cols, f"{name} has no measured peer norm"
    for name in ("f_director_cash_fee_pct", "f_director_equity_pay_pct"):
        assert name in cols and f"{name}_vs_peers" in cols, \
            f"{name} earned a peer leg on the sector share and did not get one"

    # 9b. the four Item 5.07 vote families reached the panel, and TWELVE of their fields carry a
    #     peer leg -- read off `vote_dissent_features.PEER_RELATIVE_FIELDS` rather than typed, so
    #     the assertion cannot drift from the declaration. ⚠ This is the half the plan's own
    #     record got wrong (D63/D75 said no new family had earned a peer leg since phase 2):
    #     phase 3 earned twelve, and 4 legacy + 12 vote + 2 director-pay is exactly the 18
    #     `_vs_peers` columns the built part carries.
    assert "f_sop_dissent" in cols and "f_ceo_director_dissent" in cols
    for name in sorted(VOTE_PEER_FIELDS):
        assert f"f_{name}" in cols, f"missing vote field {name}"
        assert f"f_{name}_vs_peers" in cols, f"{name} declares a peer leg and did not get one"
    assert len([c for c in cols if c.endswith("_vs_peers")]) == 4 + len(VOTE_PEER_FIELDS) + 2

    # 10. D3, asserted where it is actually observable: all TWELVE previously-shipped governance
    #     features are present under their exact old names. The list is read off
    #     `LEGACY_EXEMPT_FROM_EXPIRY` rather than typed here, so the two can never drift -- that
    #     frozenset is the same one `expire_stale` consults, and it is named on the FEATURE side.
    #     `test_impute_accrual.py` proves the VALUES are unmoved; this proves they still EXIST
    #     after governance was lifted out of `cube_part_institutionals` into its own part.
    missing_legacy = [n for n in sorted(LEGACY_EXEMPT_FROM_EXPIRY) if f"f_{n}" not in cols]
    assert not missing_legacy, f"legacy governance features lost by the move: {missing_legacy}"

    # no archive -> empty (optional-source semantics, never raises)
    empty, _ = build_governance_feature_panel(None, peers, idx)
    assert list(empty.columns) == ["date", "ticker"] and empty.empty
    # ...and `votes=None` is the OTHER optional source: the four Item 5.07 dissent families
    # vanish while every DEF 14A family still builds. Worth pinning because vote history begins
    # 2010-03, so "no votes" is not an error case -- it is fifteen years of this archive.
    no_votes, _ = build_governance_feature_panel(_def14a(), peers, idx,
                                                 fundamentals_history=_fundamentals(),
                                                 votes=None, directors=_directors(),
                                                 director_comp=_director_comp())
    nv_cols = [c for c in no_votes.columns if c not in ("date", "ticker")]
    assert "f_board_size" in nv_cols and "f_ceo_pay_ratio" in nv_cols
    assert not [c for c in nv_cols if c.startswith(("f_sop_", "f_director_dissent",
                                                    "f_ceo_dissent", "f_auditor_dissent"))], \
        "a vote family survived votes=None"
    # ...and the two phase-6 children are OPTIONAL in the same way: an archive with no director
    # rows must still build every other family rather than raise.
    no_kids, _ = build_governance_feature_panel(_def14a(), peers, idx,
                                                fundamentals_history=_fundamentals())
    kid_cols = [c for c in no_kids.columns if c not in ("date", "ticker")]
    assert "f_board_size" in kid_cols and "f_board_turnover" not in kid_cols

    print("\n=== SANITY CHECK: governance panel encoding ===")
    print(f"  {len(cols)} features: {len([c for c in cols if not c.endswith(('_vs_peers','_vs_hist'))])} raw"
          f" + {len([c for c in cols if c.endswith('_vs_peers')])} _vs_peers"
          f" + {len([c for c in cols if c.endswith('_vs_hist')])} _vs_hist + 0 _xs")
    print(f"  differences and flags ship raw only; the peer legs are 4 legacy levels + "
          f"{len(VOTE_PEER_FIELDS)} vote-dissent + 2 director pay-mix;")
    print("  avg_board_tenure is the sole _vs_hist survivor. None archive -> empty, no crash.")
    print("  ZERO columns contain `_x_`: GPT §13's five products are not built (D16), and")
    print("  ceo_is_board_chair -- the one leg that existed only inside one -- ships as a level.")
    print("  phase 6: 6 board-quality fields raw only (sector share 2.9%-7.2%), 4 director-pay")
    print("  fields of which the two pay-MIX shares earned a peer leg (11.94% / 8.17%).")
    print(f"  the two child tables are OPTIONAL: without them the panel still builds "
          f"{len(kid_cols)} features.")
    print(f"  all {len(LEGACY_EXEMPT_FROM_EXPIRY)} legacy features survive the move to their own "
          f"part under unchanged names (D3);")
    print(f"  votes=None -> {len(nv_cols)} features, every DEF 14A family intact and no vote "
          f"family left (vote history starts 2010-03).")
    print("  CONCLUSION: every encoding present is one a measurement justified. Validated.")
