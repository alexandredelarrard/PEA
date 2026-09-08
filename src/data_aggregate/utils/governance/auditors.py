"""
auditors.py  (src/data_aggregate/utils/governance/auditors.py)
--------------------------------------------------------------------------------
The same job `names.py` does for a PERSON, done for an ENTITY: `def14a_llm.auditor_name` holds
70 distinct raw strings naming 5 audit firms, and half of every apparent auditor change is a
filer rewriting its own auditor's name.

Measured 2026-09-07 over `def14a_llm` (10,498 rows carry an `auditor_name`, 85.1%):

    distinct raw strings                                          70
    tickers whose RAW auditor string changes at least once        336
    tickers whose CANONICAL firm changes at least once            165
    => spurious "auditor changed" events removed          171 of 336  (51%)

An alias TABLE, not a regex. Every one of the 70 observed strings is listed below, keyed on its
own lower-cased, whitespace-collapsed form -- the `constants.CUSIP_TICKER_OVERRIDES` pattern:
data, not cleverness. The point is not brevity, it is that an UNRECOGNISED string falls through
to `other` and gets logged: as the archive grows, a new spelling of Deloitte must be visible as
a missing table entry rather than silently becoming a new firm. A regex wide enough to swallow
`D&T` and `GT` is also wide enough to swallow `Brown, Schwab, Bergquist & Co.`, and confidently
mislabel a real small practice as a big-4 firm. A truncated cell like `Young Ireland` is resolved
by a HAND entry instead, corroborated by another row of the same table -- which is a reading of
the evidence, where a pattern would have been a guess applied to everything that matched it.

TWO JUDGEMENTS, both about firm continuity rather than spelling:

  * PREDECESSOR NAMES FOLD IN. `Price Waterhouse` (16 rows) and `Coopers & Lybrand` (25) merged
    in 1998 to FORM PricewaterhouseCoopers, `KPMG Peat Marwick` (51) is KPMG's own former name,
    `BDO Seidman` (11) is BDO USA's. The client's auditor was continuous through each rename, so
    folding them in is what stops 1998 reading as an auditor change at every PW and C&L client.
  * ⚠ `Arthur Andersen` (197 rows) does NOT fold anywhere and is NOT `other`. It gets its own
    canonical value because the 2002 collapse forced a genuine, involuntary auditor change at
    every client it had -- the single most interesting auditor event in this sample, and folding
    it into `other` would erase exactly the transitions a future study would look for.

A cell naming two genuinely different firms (3 rows: PwC+EY, Deloitte+EY, EY+PwC) takes the
FIRST firm listed, consistent with `names.ceo_identity`'s co-CEO rule -- at worst that mislabels
a joint audit, whereas `other` would manufacture a change in both directions.

The canonical firm is a CLEANED COLUMN on the read frame. `def14a_llm` is never mutated.
"""
from __future__ import annotations

import pandas as pd

from src.utils.string import clean_text

#: The closed canonical set. `other` is the explicit escape hatch, not a firm.
BIG4 = ("pwc", "kpmg", "ey", "deloitte")
AUDITOR_FIRMS = BIG4 + ("arthur_andersen", "grant_thornton", "bdo", "other")

#: Every `auditor_name` string observed in `def14a_llm`, keyed on `_lookup_key` of itself.
#: Grouped by firm; the trailing count is that string's row count on 2026-09-07.
AUDITOR_ALIASES: dict[str, str] = {
    # -- Ernst & Young (3,215 rows) -----------------------------------------------------------
    "ernst & young llp": "ey",                          # 2857
    "ernst & young": "ey",                              # 284
    "ey": "ey",                                         # 42
    "e&y": "ey",                                        # 14
    "ernst & young us": "ey",                           # 6
    "ernst & young, llp": "ey",                         # 5
    "ernst and young, llp": "ey",                       # 1  the "and" spelling
    "ernst & young ireland": "ey",                      # 2  member firm
    "ernst & young chartered accountants": "ey",        # 1  member firm
    "ernst & young accountants llp": "ey",              # 1  member firm (NL)
    "ey accountants b.v.": "ey",                        # 1  member firm (NL)
    "ernst & young (e&y)": "ey",                        # 1
    # ⚠ A TRUNCATED CELL, mapped by HAND and on purpose. `Young Ireland` is not a firm; the same
    # table holds `ernst & young ireland` (2 rows) as a real EY member firm, so the truncation is
    # corroborated by the archive itself rather than guessed at. Mapping it keeps one filing from
    # reading as a change TO and FROM a mystery auditor -- which is what `other` would have made
    # it, and that is the more expensive error: `auditor_changed` would fire twice on a company
    # whose auditor never changed.
    "young ireland": "ey",                              # 1
    # -- PricewaterhouseCoopers (2,872 rows, incl. the two 1998 predecessor firms) -------------
    "pricewaterhousecoopers llp": "pwc",                # 2581
    "pricewaterhousecoopers": "pwc",                    # 139
    "pwc": "pwc",                                       # 62  (`PwC` 61 + `PWC` 1)
    "pricewaterhousecoopers sa": "pwc",                 # 18  member firm
    "pricewaterhousecoopers, llp": "pwc",               # 8
    "pricewaterhousecoopers ag": "pwc",                 # 8   member firm
    "pwc llp": "pwc",                                   # 8
    "pricewaterhousecoopers llc": "pwc",                # 2
    "pricewaterhousecoopers accountants n.v.": "pwc",   # 2   member firm (NL)
    "pricewaterhousecoopers llp and pricewaterhousecoopers ag": "pwc",   # 2  two member firms
    "coopers & lybrand l.l.p.": "pwc",                  # 23  predecessor, merged 1998
    "coopers & lybrand llp": "pwc",                     # 1
    "coopers & lybrand": "pwc",                         # 1
    "price waterhouse llp": "pwc",                      # 14  predecessor, merged 1998
    "price waterhouse l.l.p.": "pwc",                   # 1
    "price waterhouse": "pwc",                          # 1
    # -- Deloitte (2,321 rows) ----------------------------------------------------------------
    "deloitte & touche llp": "deloitte",                # 1952 (`...LLP` 1950 + `...llp` 2)
    "deloitte": "deloitte",                             # 235
    "deloitte & touche": "deloitte",                    # 109
    "deloitte llp": "deloitte",                         # 10
    "d&t": "deloitte",                                  # 7   initialism a regex would miss
    "deloitte & touche, llp": "deloitte",               # 6
    "deloitte and touche llp": "deloitte",              # 3
    "deloitte & touche (ireland)": "deloitte",          # 3   member firm
    "deloitte &touche llp": "deloitte",                 # 1   missing space
    "deloitte touche tohmatsu": "deloitte",             # 1   the global network's own name
    "deloitte touche tohmatsu japan": "deloitte",       # 1   member firm
    "deloitte sa": "deloitte",                          # 1   member firm
    "deloitte u.s.": "deloitte",                        # 1
    "deloitte & touche llp and deloitte llp": "deloitte",   # 1  two entities of ONE firm
    # -- KPMG (1,761 rows, incl. the pre-1999 Peat Marwick name) ------------------------------
    "kpmg llp": "kpmg",                                 # 1375
    "kpmg": "kpmg",                                     # 334
    "kpmg peat marwick llp": "kpmg",                    # 49  former name, renamed 1999
    "kpmg peat marwick, llp": "kpmg",                   # 1
    "kpmg peat marwick": "kpmg",                        # 1
    "kpmg, llc": "kpmg",                                # 1
    # -- Arthur Andersen (197 rows) -- REAL HISTORY, its own value, never `other` -------------
    "arthur andersen llp": "arthur_andersen",           # 187
    "arthur andersen": "arthur_andersen",               # 9
    "arthur andersen, llp": "arthur_andersen",          # 1
    # -- Grant Thornton (98 rows) -------------------------------------------------------------
    "grant thornton llp": "grant_thornton",             # 84
    "grant thornton": "grant_thornton",                 # 11
    "gt": "grant_thornton",                             # 2   initialism a regex would miss
    "grant thornton, llp": "grant_thornton",            # 1
    # -- BDO (17 rows, incl. the pre-2010 Seidman name) ---------------------------------------
    "bdo seidman": "bdo",                               # 9   former name, renamed 2010
    "bdo usa": "bdo",                                   # 3
    "bdo": "bdo",                                       # 2
    "bdo usa, llp": "bdo",                              # 1
    "bdo seidman, llp": "bdo",                          # 1
    "bdo seidman llp": "bdo",                           # 1
    # -- joint audits naming two DIFFERENT firms -> the first listed (3 rows) ------------------
    "pricewaterhousecoopers llp and ernst & young llp": "pwc",      # 1
    "deloitte & touche llp and ernst & young llp": "deloitte",      # 1
    "ernst & young; pricewaterhousecoopers": "ey",                  # 1
    # -- genuinely not a big-4/GT/BDO firm, and not a garbled cell either (4 rows) -------------
    # Real, small audit practices. `other` is the honest answer: the closed set has no slot for
    # them, and inventing one per firm would be a per-ticker column, not a feature.
    "brown, schwab, bergquist & co.": "other",          # 2
    "alpern, rosenthal & company": "other",             # 1
    "lane gorman trubitt, llc": "other",                # 1
    # ⚠ `Young Ireland` is NOT here: it is a truncated cell resolved BY HAND to `ey` up in the
    # EY block, because the same archive holds `ernst & young ireland` and that corroborates the
    # truncation. `other` would have made one filing read as a change to AND from a mystery
    # auditor, firing `auditor_changed` twice on a company whose auditor never changed.
}


def _lookup_key(value: object) -> str | None:
    """The alias table's key: whitespace-collapsed and lower-cased, nothing else.

    Mechanical, not interpretive. Stripping punctuation here would shorten the table by
    conflating `Deloitte & Touche` with `Deloitte and Touche` -- and would equally conflate
    strings the table is supposed to keep apart, which is the property that makes an unknown
    spelling visible.
    """
    cleaned = clean_text(value)
    return cleaned.lower() if cleaned else None


def canonical_auditor(value: object) -> str | None:
    """One of `AUDITOR_FIRMS` for a raw `auditor_name` cell, or None when there is no cell.

    None is ABSENT (the 14.9% of proxies that name no auditor) and `other` is PRESENT BUT NOT
    A KNOWN FIRM -- collapsing the two would turn every non-disclosing filing into a change to
    and from a mystery auditor.
    """
    key = _lookup_key(value)
    if key is None:
        return None
    return AUDITOR_ALIASES.get(key, "other")


def canonical_auditor_series(values: pd.Series) -> pd.Series:
    """`canonical_auditor` over a column. `.astype(object)` for the same pyarrow reason as
    `names.ceo_identity_series`: a returned None must stay None, not become `pd.NA`, or the
    transition test that reads this column compares to neither True nor False."""
    return values.astype(object).map(canonical_auditor)


def unrecognised_auditor_names(values: pd.Series) -> dict[str, int]:
    """The distinct raw strings that fell through to `other`, with their row counts.

    Logged at build time. A NEW unrecognised string appearing as the archive grows means
    `AUDITOR_ALIASES` needs an entry -- not that a company hired an unknown auditor -- and this
    is the only thing that makes that visible, since `other` is a legitimate value too.
    """
    keys = values.astype(object).map(_lookup_key)
    unknown = values[keys.notna() & ~keys.isin(AUDITOR_ALIASES)]
    return {str(k): int(v) for k, v in unknown.value_counts().items()}
