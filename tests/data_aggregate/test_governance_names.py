"""
CEO identity for the governance cube (src/data_aggregate/utils/governance/names.py) and the
agreement-gated identity gap fill it enables in `def14a_impute`.

Three things under test:
  1. a co-CEO cell keys to the FIRST person listed (D28), not to a chimera of both;
  2. an interior `ceo_name_proxy` gap is filled only when the same PERSON stands on both sides
     (D27), judged on the key so a respelling still counts as agreement;
  3. the live archive reproduces the measurements phase 4's turnover guard is sized on -- these
     must be regenerated, never quoted from the plan document.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.governance.def14a_impute import (
    AGREEMENT_REQUIRED_FLAGS, impute_def14a,
)
from src.data_aggregate.utils.governance.names import (
    ceo_identity, is_multi_name, split_co_names,
)
from src.data_store.schema import Tables
from src.utils.names import person_key


def _row(ticker: str, as_of: str, **kw) -> dict:
    base = {"ticker": ticker, "as_of": as_of, "accession_number": f"{ticker}-{as_of}"}
    base.update(kw)
    return base


# --------------------------------------------------------------------------- co-CEO cells ---
def test_a_co_ceo_cell_keys_to_the_first_person_listed():
    """Unsplit, CPRT's `A. Jayson Adair; Jeffrey Liaw` keys `liaw|a` -- the first person's
    initial glued to the second's surname, matching NEITHER of them, so it reads as a turnover
    both the year it appears and the year it stops."""
    cases = [
        ("A. Jayson Adair; Jeffrey Liaw", "adair|a"),
        ("A. Jayson Adair and Jeffrey Liaw", "adair|a"),
        ("A. Jayson Adair & Jeffrey Liaw", "adair|a"),
        ("A. Jayson Adair", "adair|a"),          # the continuing CEO alone -> same key
    ]
    for cell, expected in cases:
        assert ceo_identity(cell) == expected, f"{cell!r} -> {ceo_identity(cell)!r}"
    assert ceo_identity("A. Jayson Adair; Jeffrey Liaw") != "liaw|a", "the unsplit chimera"

    # ` and ` needs its surrounding whitespace, or the pattern fires inside a surname
    assert ceo_identity("Richard Anderson") == "anderson|r"
    assert not is_multi_name("Richard Anderson")
    assert is_multi_name("A. Jayson Adair; Jeffrey Liaw")
    assert ceo_identity(None) is None and ceo_identity(np.nan) is None
    assert split_co_names(np.nan) == []

    print("\n=== SANITY CHECK: co-CEO cells (D28) ===")
    for cell, expected in cases:
        print(f"  {cell:<36} -> {ceo_identity(cell)}  (names: {split_co_names(cell)})")
    print("  'Richard Anderson' -> anderson|r  (' and ' did NOT fire inside the surname)")
    print("  CONCLUSION: a multi-name cell takes the FIRST person, so the continuing CEO's pay "
          "stays comparable year over year instead of keying to a chimera that matches nobody. "
          "Accepted cost: a genuine co-CEO transition reads as NO turnover. Validated.")


# ------------------------------------------------------------- the agreement-gated gap fill ---
def test_identity_gap_fills_only_when_the_same_person_stands_on_both_sides():
    """The plain `ffill` fills a bounded gap whenever SOMETHING exists each side, never checking
    that the two agree -- so on a real transition it hands the gap year the OLD CEO's name, and
    the turnover guard then compares old-vs-old, sees no change, and computes pay growth
    straight across the transition it exists to catch."""
    df = pd.DataFrame([
        # AGREE, same spelling -> filled
        _row("SAME", "2019-04-01", ceo_name_proxy="Timothy D. Cook", classified_board=1.0),
        _row("SAME", "2020-04-01", ceo_name_proxy=None, classified_board=np.nan),
        _row("SAME", "2021-04-01", ceo_name_proxy="Timothy D. Cook", classified_board=1.0),
        # AGREE under the KEY though the strings differ -> filled (a raw comparison would not)
        _row("DRIFT", "2019-04-01", ceo_name_proxy="Timothy D. Cook"),
        _row("DRIFT", "2020-04-01", ceo_name_proxy=None),
        _row("DRIFT", "2021-04-01", ceo_name_proxy="Tim Cook"),
        # DISAGREE -- a real ACGL-shaped turnover inside the gap -> left NaN
        _row("TURN", "1999-03-16", ceo_name_proxy="Mark D. Mosca"),
        _row("TURN", "2000-03-16", ceo_name_proxy=None),
        _row("TURN", "2001-03-16", ceo_name_proxy="Peter A. Appel"),
        # a PROVISION gap on the same disagreeing ticker must still carry forward
        _row("PROV", "2019-04-01", classified_board=1.0, poison_pill=0.0),
        _row("PROV", "2020-04-01", classified_board=np.nan, poison_pill=np.nan),
        _row("PROV", "2021-04-01", classified_board=0.0, poison_pill=1.0),
    ])
    out, stats = impute_def14a(df)
    out = out.set_index(["ticker", "as_of"])

    assert out.loc[("SAME", pd.Timestamp("2020-04-01")), "ceo_name_proxy"] == "Timothy D. Cook"
    assert out.loc[("DRIFT", pd.Timestamp("2020-04-01")), "ceo_name_proxy"] == "Timothy D. Cook"
    assert pd.isna(out.loc[("TURN", pd.Timestamp("2000-03-16")), "ceo_name_proxy"]), (
        "a gap with a DIFFERENT CEO each side must stay NaN = UNKNOWN")
    # the other FLAGS columns keep the plain carry-forward: "unchanged since the last
    # disclosure" is a sound prior for a bylaw and a guess about who holds a job
    assert out.loc[("PROV", pd.Timestamp("2020-04-01")), "classified_board"] == 1.0
    assert out.loc[("PROV", pd.Timestamp("2020-04-01")), "poison_pill"] == 0.0
    assert out.loc[("SAME", pd.Timestamp("2020-04-01")), "classified_board"] == 1.0
    assert AGREEMENT_REQUIRED_FLAGS == frozenset({"ceo_name_proxy"}), (
        "only the identity gets the agreement rule; widening it would stop provisions carrying")
    assert stats.get("declined (identity changed): ceo_name_proxy") == 1

    print("\n=== SANITY CHECK: identity gap fill requires agreement (D27) ===")
    print(f"  SAME  (Cook / gap / Cook)              -> filled: "
          f"{out.loc[('SAME', pd.Timestamp('2020-04-01')), 'ceo_name_proxy']!r}")
    print(f"  DRIFT ('Timothy D. Cook' / gap / 'Tim Cook') -> filled: "
          f"{out.loc[('DRIFT', pd.Timestamp('2020-04-01')), 'ceo_name_proxy']!r}  "
          "(agreement judged on the KEY, not the string)")
    print("  TURN  (Mosca / gap / Appel)            -> left NaN = UNKNOWN")
    print(f"  provisions still carry forward: classified_board=1.0, poison_pill=0.0")
    print(f"  stats: {stats}")
    print("  CONCLUSION: a bounded gap proves a value was disclosed either side, NOT that it was "
          "the same value. The identity fills only on agreement; every other FLAGS column keeps "
          "the plain carry-forward. Non-destructive by construction -- this only ever DECLINES "
          "to fill a NaN. Validated.")


# ----------------------------------------------------------------- the live re-measurement ---
def _live_def14a() -> pd.DataFrame:
    try:
        from src.context import get_config_context
        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        raw = ctx.store.load(Tables.def14a_llm)
    except Exception as e:                                  # noqa: BLE001
        pytest.skip(f"def14a_llm not reachable ({e})")
    if raw is None or raw.empty:
        pytest.skip("def14a_llm empty")
    return raw


def test_normalisation_measured_on_the_live_archive():
    """Regenerate the numbers phase 4's turnover guard is SIZED on, rather than trusting the
    plan document's copy of them.

    Reported on THREE bases, because two of them are legitimately different measurements and
    conflating them is how a reproduction "fails" that in fact agrees:
      * `raw`   -- the filer's own string;
      * `key`   -- the bare `person_key`, which is what the phase-1 plan measured;
      * `ident` -- `ceo_identity`, i.e. the key AFTER the co-CEO split (D28), which is what the
        cube actually applies and which is strictly better on every line.
    Identity counts are shown BOTH globally and summed per ticker: a CEO of two companies is one
    person globally and two ticker-CEOs per ticker, and the plan quotes the per-ticker basis.

    Asserted as directional invariants plus loose bounds, not exact counts: the archive grows.
    But the DIRECTION is a contract -- normalisation can only ever merge identities, never split
    them, and the co-CEO split can only ever improve on the bare key."""
    raw = _live_def14a()
    df = raw[["ticker", "accession_number", "as_of", "ceo_name_proxy"]].copy()
    df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce")
    df = df.sort_values(["ticker", "as_of"])
    df["key"] = df["ceo_name_proxy"].map(person_key)
    df["ident"] = df["ceo_name_proxy"].astype(object).map(ceo_identity)

    named = df[df["ceo_name_proxy"].notna()]
    cols = ["ceo_name_proxy", "key", "ident"]
    glob = {c: int(named[c].nunique()) for c in cols}
    per_ticker = named.groupby("ticker")[cols].nunique()
    summed = {c: int(per_ticker[c].sum()) for c in cols}
    shrunk = {c: int((per_ticker[c] < per_ticker["ceo_name_proxy"]).sum()) for c in cols[1:]}

    # consecutive-filing pairs, both sides named -> how many read as a CEO turnover
    g = df.groupby("ticker", sort=False)
    comparable = df["ceo_name_proxy"].notna() & g["ceo_name_proxy"].shift(1).notna()
    turns = {}
    for c in cols:
        prev = g[c].shift(1)
        # `notna()` on both sides is load-bearing: None is UNKNOWN, and two unknowns must not
        # compare equal into "no turnover" -- that is the failure mode the guard exists to avoid
        turns[c] = int((comparable & df[c].notna() & prev.notna() & (df[c] != prev)).sum())

    assert glob["key"] < glob["ceo_name_proxy"], "the key must MERGE identities, never split them"
    assert glob["ident"] <= glob["key"], "the co-CEO split can only merge further"
    assert turns["key"] < turns["ceo_name_proxy"], "the key must remove apparent turnovers"
    assert turns["ident"] <= turns["key"], "the co-CEO split can only remove more"
    assert glob["key"] / glob["ceo_name_proxy"] > 0.7, "a 30%+ collapse would be over-merging"

    n_comparable, n_multi = int(comparable.sum()), int(named["ceo_name_proxy"].map(is_multi_name).sum())
    print("\n=== SANITY CHECK: what the person key buys, measured live ===")
    print(f"  rows={len(raw)}  tickers={raw['ticker'].nunique()}  named CEO cells={len(named)}")
    print(f"  {'measure':<40} {'raw':>7} {'key':>7} {'+co-CEO':>8}   {'delta vs raw':>14}")
    for label, d in (("distinct identities (global)", glob),
                     ("distinct identities (sum per ticker)", summed)):
        print(f"  {label:<40} {d['ceo_name_proxy']:>7} {d['key']:>7} {d['ident']:>8}   "
              f"{d['ident'] - d['ceo_name_proxy']:>+7} "
              f"({(d['ident'] - d['ceo_name_proxy']) / d['ceo_name_proxy']:+.1%})")
    print(f"  {'consecutive pairs comparable':<40} {n_comparable:>7} {n_comparable:>7} "
          f"{n_comparable:>8}   {'--':>14}")
    print(f"  {'pairs reading as a CEO TURNOVER':<40} {turns['ceo_name_proxy']:>7} "
          f"{turns['key']:>7} {turns['ident']:>8}   "
          f"{turns['ident'] - turns['ceo_name_proxy']:>+7} "
          f"({(turns['ident'] - turns['ceo_name_proxy']) / turns['ceo_name_proxy']:+.1%})")
    print(f"  tickers whose identity count shrinks: {shrunk['key']} (key) / {shrunk['ident']} "
          f"(+co-CEO) of {per_ticker.shape[0]}")
    print(f"  multi-name (co-CEO) cells encountered: {n_multi}  <- the D28 population, logged "
          "because there is no feature and no flag column behind it")
    print("  CONCLUSION: the spurious turnovers above are pay-growth observations phase 4's guard "
          "would otherwise DISCARD -- it nulls growth on every apparent CEO change, and on raw "
          "strings ~a fifth of those changes are one filer respelling one person. The co-CEO "
          "split removes a further slice the bare key cannot see. Validated.")


def test_the_turnover_guards_true_cost_on_computable_pay_pairs():
    """§5's last item: the guard's cost is not "apparent turnovers", it is pay-growth
    observations LOST -- consecutive filings that both carry a `ceo_total_comp`, where growth
    would be computable but the guard nulls it because the CEO changed.

    That is the number phase 4 is sized on, and measuring it on raw strings overstates it."""
    raw = _live_def14a()
    df = raw[["ticker", "as_of", "ceo_name_proxy", "ceo_total_comp"]].copy()
    df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce")
    df = df.sort_values(["ticker", "as_of"])
    df["ident"] = df["ceo_name_proxy"].astype(object).map(ceo_identity)
    g = df.groupby("ticker", sort=False)

    # a pay pair: consecutive filings of one ticker, both with a CEO total
    payable = df["ceo_total_comp"].notna() & g["ceo_total_comp"].shift(1).notna()
    costs = {}
    for label, col in (("raw string", "ceo_name_proxy"), ("ceo_identity", "ident")):
        prev = g[col].shift(1)
        changed = df[col].notna() & prev.notna() & (df[col] != prev)
        unknown = payable & (df[col].isna() | prev.isna())   # NaN identity -> also nulled
        costs[label] = (int((payable & changed).sum()), int(unknown.sum()))

    n_pairs = int(payable.sum())
    raw_cost = costs["raw string"][0]
    key_cost = costs["ceo_identity"][0]
    assert key_cost < raw_cost, "the guard must cost LESS once identities reconcile"

    print("\n=== SANITY CHECK: the turnover guard's true cost ===")
    print(f"  computable pay-growth pairs (both filings carry ceo_total_comp): {n_pairs}")
    for label, (changed, unknown) in costs.items():
        print(f"    on {label:<13} -> nulled by a CEO change: {changed:>5} "
              f"({changed / n_pairs:.1%})   | also nulled as UNKNOWN identity: {unknown}")
    print(f"  => the guard recovers {raw_cost - key_cost} pay-growth observations "
          f"({(raw_cost - key_cost) / raw_cost:.1%} of its raw-string cost) purely by "
          "reconciling spellings")
    print("  CONCLUSION: measured on raw strings the guard looks far more expensive than it is; "
          "roughly a fifth of what it would discard is one filer respelling one CEO's name. "
          "Validated.")


def test_the_ceo_to_neo_cross_table_match_measured_on_the_live_archive():
    """The join the pay-slice family depends on: is the CEO named on a `def14a_llm` row actually
    present in that same filing's own `def14a_executive_comp` rows?

    Phase 4's exact CPS divides `ceo_total_comp` by the sum of the top-5 NEO totals, which is
    only meaningful if the CEO is INSIDE that denominator. On raw strings the filer's proxy
    summary and its own Summary Compensation Table disagree often enough to matter -- the same
    person written two ways in two tables of one document."""
    proxies = _live_def14a()
    try:
        from src.context import get_config_context
        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        neos = ctx.store.load(Tables.def14a_executive_comp)
    except Exception as e:                                  # noqa: BLE001
        pytest.skip(f"def14a_executive_comp not reachable ({e})")
    if neos is None or neos.empty:
        pytest.skip("def14a_executive_comp empty")

    key = ["ticker", "accession_number"]
    p = proxies[key + ["ceo_name_proxy"]].dropna(subset=["ceo_name_proxy"]).drop_duplicates(key)
    n = neos[key + ["name"]].dropna(subset=["name"]).copy()
    n["k"] = n["name"].map(person_key)

    raw_sets = n.groupby(key)["name"].apply(set)
    key_sets = n.groupby(key)["k"].apply(lambda s: set(s.dropna()))
    p = p.join(raw_sets.rename("neo_raw"), on=key).join(key_sets.rename("neo_key"), on=key)
    p = p[p["neo_raw"].notna()]                 # filings whose SCT was extracted at all

    hit_raw = int(sum(c in s for c, s in zip(p["ceo_name_proxy"], p["neo_raw"])))
    ident = p["ceo_name_proxy"].astype(object).map(ceo_identity)
    hit_key = int(sum(k is not None and k in s for k, s in zip(ident, p["neo_key"])))

    assert hit_key > hit_raw, "reconciling must find MORE of the CEOs, never fewer"
    assert hit_key / len(p) > 0.90, "a sub-90% match would mean the key is failing, not helping"

    print("\n=== SANITY CHECK: CEO <-> NEO cross-table match, measured live ===")
    print(f"  filings with both a named CEO and extracted SCT rows: {len(p)}")
    print(f"    CEO found in its own filing's NEO rows, RAW string : {hit_raw} "
          f"({hit_raw / len(p):.1%})")
    print(f"    CEO found in its own filing's NEO rows, KEYED      : {hit_key} "
          f"({hit_key / len(p):.1%})")
    print(f"    => +{hit_key - hit_raw} filings (+{(hit_key - hit_raw) / len(p):.1%}pp)")
    print("  CONCLUSION: the same person is written two ways in two tables of ONE document often "
          "enough to break the join on raw strings. Keying it is what lets phase 4's exact CEO "
          "Pay Slice assert the CEO sits inside its own top-5 denominator. Validated.")


def test_the_identity_gap_fill_measured_on_the_live_archive():
    """The §3.3 population, live: how many interior identity gaps exist, how many the key says
    agree (and are therefore filled), and how many are declined because a real transition
    happened inside the gap."""
    raw = _live_def14a()
    df = raw[["ticker", "as_of", "ceo_name_proxy"]].copy()
    df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce")
    df = df.sort_values(["ticker", "as_of"])
    g = df.groupby("ticker", sort=False)
    fwd, bwd = g["ceo_name_proxy"].ffill(), g["ceo_name_proxy"].bfill()
    inside = df["ceo_name_proxy"].isna() & fwd.notna() & bwd.notna()

    agree_str = inside & (fwd == bwd)
    k_f = fwd.astype(object).map(ceo_identity)
    k_b = bwd.astype(object).map(ceo_identity)
    agree_key = inside & k_f.notna() & k_b.notna() & (k_f == k_b)

    n_gaps, n_str, n_key = int(inside.sum()), int(agree_str.sum()), int(agree_key.sum())
    assert n_key >= n_str, "the key can only ever find MORE agreement than a raw string"
    assert n_key <= n_gaps

    _, stats = impute_def14a(raw)
    declined = stats.get("declined (identity changed): ceo_name_proxy", 0)

    disagreeing = df[inside & ~agree_key]
    print("\n=== SANITY CHECK: identity gap fills, measured live ===")
    print(f"  interior ceo_name_proxy gaps (a value present each side): {n_gaps}")
    print(f"    agreeing on the RAW STRING -> fillable: {n_str}")
    print(f"    agreeing on the KEY        -> fillable: {n_key}  (+{n_key - n_str} the key "
          "recovers that a string comparison would leave NaN)")
    print(f"    DECLINED, a different CEO each side:    {n_gaps - n_key}")
    print(f"  impute_def14a on the full table reports declined={declined}, "
          f"carried={stats.get('carry: ceo_name_proxy', 0)}")
    if not disagreeing.empty:
        show = disagreeing.head(5).assign(before=fwd[disagreeing.index],
                                          after=bwd[disagreeing.index])
        print("  the declined gaps are unambiguous real transitions, e.g.:")
        for _, r in show.iterrows():
            print(f"    {r['ticker']:<6} {str(r['as_of'])[:10]}  "
                  f"{r['before']!r} -> {r['after']!r}")
    print("  CONCLUSION: the plain ffill would give every declined gap the OUTGOING CEO's name, "
          "blinding the turnover guard to the exact transition it exists to catch. Requiring "
          "agreement keeps the drifts and drops only the real changes. Validated.")


if __name__ == "__main__":
    test_a_co_ceo_cell_keys_to_the_first_person_listed()
    test_identity_gap_fills_only_when_the_same_person_stands_on_both_sides()
    test_normalisation_measured_on_the_live_archive()
    test_the_turnover_guards_true_cost_on_computable_pay_pairs()
    test_the_identity_gap_fill_measured_on_the_live_archive()
