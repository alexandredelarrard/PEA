"""
roles.py  (src/data_extract/utils/structure/votes/roles.py)
------------------------------------------------------------
Join each nominee to the nearest prior proxy so a vote row knows whether it is about the
CEO, another executive officer, or a non-employee director.

The key is `person_key` from `def14a/gender.py` -- the SAME definition the DEF 14A gender
consensus groups on, so a nominee that matches there matches here. The per-filing
`unmatched` count is stored rather than swallowed: that number IS this join's error rate.
"""
from __future__ import annotations

import pandas as pd

from src.context import Context
from src.data_extract.utils.structure.def14a.gender import person_key
from src.data_extract.utils.structure.def14a.validate import clean_text
from src.data_store.schema import Tables

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
