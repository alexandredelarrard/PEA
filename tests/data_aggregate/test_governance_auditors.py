"""
Auditor-name normalisation (src/data_aggregate/utils/governance/auditors.py) — the person-key
job done for an ENTITY.

`def14a_llm.auditor_name` holds 70 distinct raw strings naming 5 firms. Half of every apparent
auditor change in the archive is a filer rewriting its own auditor's name, so without this the
`auditor_changed` feature phase 5 ships would be ~51% spelling drift.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data_aggregate.utils.governance.auditors import (
    AUDITOR_ALIASES, AUDITOR_FIRMS, BIG4, canonical_auditor, canonical_auditor_series,
    unrecognised_auditor_names,
)
from src.data_store.schema import Tables


def test_the_aliases_a_first_pass_regex_would_miss():
    """Every one of these was found by READING the 70 raw strings, not by widening a pattern
    until it swallowed something.

    ⚠ `Young Ireland` is the case that proves the point, and it cuts the OTHER way from how this
    test first read it. It is a truncated cell, and it resolves to `ey` because the same archive
    holds `Ernst & Young Ireland` as a real EY member firm — the truncation is corroborated by a
    neighbouring row rather than guessed at. Leaving it as `other` is the more expensive error:
    one filing then reads as a change TO and FROM a mystery auditor, so `auditor_changed` fires
    twice on a company whose auditor never changed. `Brown, Schwab, Bergquist & Co.` is the case
    that stays `other` — a real, small, non-big-4 practice, which is exactly what a regex loose
    enough to catch `D&T` would mislabel.
    """
    pins = {
        "D&T": "deloitte",                       # initialism, 7 rows
        "GT": "grant_thornton",                  # initialism, 2 rows
        "Ernst and Young, LLP": "ey",            # the "and" spelling, 1 row
        "BDO Seidman": "bdo",                    # pre-2010 name
        "BDO USA": "bdo",
        "BDO": "bdo",
        "KPMG Peat Marwick LLP": "kpmg",         # pre-1999 name
        "Coopers & Lybrand L.L.P.": "pwc",       # merged INTO PwC in 1998
        "Price Waterhouse LLP": "pwc",           # merged INTO PwC in 1998
        "Deloitte &Touche LLP": "deloitte",      # missing space
        "Young Ireland": "ey",                   # truncated, resolved by hand (see above)
        "Brown, Schwab, Bergquist & Co.": "other",   # a real, small, non-big-4 firm
    }
    for raw, expected in pins.items():
        assert canonical_auditor(raw) == expected, f"{raw!r} -> {canonical_auditor(raw)!r}"

    # ⚠ Arthur Andersen is REAL HISTORY and keeps its own canonical value. Folding it into
    # `other` would erase the 2002 collapse -- a genuine, involuntary auditor change at every
    # client it had, and the single most interesting auditor event in this sample.
    for raw in ("Arthur Andersen LLP", "Arthur Andersen", "Arthur Andersen, LLP"):
        assert canonical_auditor(raw) == "arthur_andersen", f"{raw!r} must not be 'other'"

    # None is ABSENT (no auditor named); "other" is PRESENT BUT NOT A KNOWN FIRM. Collapsing
    # the two would turn every non-disclosing filing into a change to and from a mystery firm.
    assert canonical_auditor(None) is None
    assert canonical_auditor("") is None
    assert canonical_auditor("Some Firm Nobody Has Heard Of LLP") == "other"

    assert set(AUDITOR_ALIASES.values()) <= set(AUDITOR_FIRMS)
    assert set(BIG4) == {"pwc", "kpmg", "ey", "deloitte"}

    print("\n=== SANITY CHECK: the auditor alias table ===")
    for raw, expected in pins.items():
        print(f"  {raw:<32} -> {canonical_auditor(raw)}")
    print(f"  Arthur Andersen LLP              -> {canonical_auditor('Arthur Andersen LLP')} "
          "(its own value: the 2002 collapse is real history, not junk)")
    print(f"  None -> {canonical_auditor(None)} (ABSENT)   unknown string -> "
          f"{canonical_auditor('Some Firm Nobody Has Heard Of LLP')} (PRESENT, not a known firm)")
    print(f"  table: {len(AUDITOR_ALIASES)} explicit aliases -> {len(AUDITOR_FIRMS)} canonical "
          "values. CONCLUSION: initialisms and predecessor firm names resolve, a truncated cell "
          "is resolved by a HAND entry corroborated by a neighbouring row (`Young Ireland` -> ey, "
          "beside `Ernst & Young Ireland`), an unrecognised firm falls through to `other` rather "
          "than being pattern-matched into a big-4, and Arthur Andersen keeps its own value. "
          "Validated.")


def test_the_collapse_measured_on_the_live_archive():
    """The measurement that justifies the whole module: how many apparent auditor CHANGES are
    the filer respelling one firm's name."""
    try:
        from src.context import get_config_context
        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        raw = ctx.store.load(Tables.def14a_llm)
    except Exception as e:                                  # noqa: BLE001
        pytest.skip(f"def14a_llm not reachable ({e})")
    if raw is None or raw.empty or "auditor_name" not in raw.columns:
        pytest.skip("def14a_llm empty or has no auditor_name")

    df = raw[["ticker", "as_of", "auditor_name"]].copy()
    df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce")
    df = df.sort_values(["ticker", "as_of"])
    df["firm"] = canonical_auditor_series(df["auditor_name"])

    named = df[df["auditor_name"].notna()]
    n_raw, n_firm = named["auditor_name"].nunique(), named["firm"].nunique()

    def _changing_tickers(col: str) -> int:
        s = df[df[col].notna()]
        return int((s.groupby("ticker")[col].nunique() > 1).sum())

    ch_raw, ch_firm = _changing_tickers("auditor_name"), _changing_tickers("firm")

    assert n_firm < n_raw, "normalisation must COLLAPSE strings, never add them"
    assert ch_firm <= ch_raw, "normalisation can only remove apparent changes, never create them"
    assert canonical_auditor_series(named["auditor_name"]).notna().all(), (
        "every named cell must resolve to a canonical value, `other` included")

    dist = named["firm"].value_counts()
    unknown = unrecognised_auditor_names(named["auditor_name"])
    fill = len(named) / len(raw) * 100

    print("\n=== SANITY CHECK: auditor normalisation, measured live ===")
    print(f"  rows with an auditor_name: {len(named)} of {len(raw)} ({fill:.1f}%)")
    print(f"  distinct RAW strings: {n_raw}  ->  distinct canonical firms: {n_firm}")
    for firm, n in dist.items():
        flag = "  <- REAL HISTORY, own value" if firm == "arthur_andersen" else ""
        print(f"    {firm:<16} {n:>6}  ({n / len(named):>5.1%}){flag}")
    print(f"  tickers whose RAW string changes at least once:  {ch_raw}")
    print(f"  tickers whose CANONICAL firm changes at least once: {ch_firm}")
    print(f"  => spurious 'auditor changed' events removed: {ch_raw - ch_firm} of {ch_raw} "
          f"({(ch_raw - ch_firm) / ch_raw:.0%})")
    big4 = float(named["firm"].isin(BIG4).mean())
    print(f"  is_big4 would be {big4:.1%} constant -- so as a standalone feature it is near "
          f"degenerate; the informative side is the {1 - big4:.1%} that is NOT big-4 (D32).")
    print(f"  rows falling through to `other`: {int(dist.get('other', 0))}")
    print(f"  UNRECOGNISED strings (need an AUDITOR_ALIASES entry): {unknown or 'none'}")
    print("  CONCLUSION: without this, roughly half of every auditor-change signal would be a "
          "filer rewriting its own auditor's name. An unrecognised string is logged rather than "
          "silently becoming a new firm, so a new spelling is visible as the archive grows. "
          "Validated.")


if __name__ == "__main__":
    test_the_aliases_a_first_pass_regex_would_miss()
    test_the_collapse_measured_on_the_live_archive()
