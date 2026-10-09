"""
anchor_check.py  (src/data_extract/utils/prices/anchor_check.py)
------------------------------------------------------------------
Detects a restated `close_split` basis in `prices` from the data. A split or spinoff restates every earlier
bar, but the incremental fetch only notices an event dated after a ticker's last stored bar. Yahoo's current
`close_split` for the bars of a ticker's first stored calendar year (the anchor) is compared with the stored
one: a stored anchor off by a factor means the ticker's whole history sits on a stale basis.

One `actions=False` download per distinct first year. A missing or empty response is missing data, never
evidence: it triggers nothing. A ticker with no usable common bars in its first year is re-anchored on the next
later calendar year of its own stored data, one grouped download per candidate year; one still unanchored
after `_MAX_FALLBACK_YEARS` tries is named in a WARNING.
"""

import logging
from collections.abc import Callable

import pandas as pd

from src.context import Context
from src.data_store.schema import Tables

logger = logging.getLogger(__name__)

#: Median `abs(fresh / stored - 1)` over the anchor bars above which a ticker is restated. Float noise between
#: two downloads of the same basis is ~1e-15; the smallest split Yahoo records is 1.0017 (a 0.17% move).
ANCHOR_TOLERANCE = 1e-4
#: Fewer common anchor bars than this are judged on their maximum deviation, not their median.
_MIN_COMMON_BARS = 3
#: Later calendar years tried, after the first stored one, to anchor a ticker Yahoo no longer serves there.
_MAX_FALLBACK_YEARS = 4

Download = Callable[[list[str], pd.Timestamp, pd.Timestamp], pd.DataFrame]


def _common_dates(old: pd.Series, new: pd.Series) -> pd.Index:
    """Dates on which both series hold a positive bar."""
    return old.dropna()[lambda s: s > 0].index.intersection(new.dropna()[lambda s: s > 0].index)


def anchor_restated(stored: pd.Series, fresh: pd.Series, tol: float) -> bool:
    """Whether the stored `close_split` anchor disagrees with Yahoo's (both Series indexed by date).

    NaN and non-positive bars are dropped on both sides; no common date means no evidence and no trigger."""
    old, new = stored.dropna(), fresh.dropna()
    common = _common_dates(old, new)
    if common.empty:
        return False
    deviation = (new.loc[common] / old.loc[common] - 1).abs()
    return bool((deviation.median() if len(common) >= _MIN_COMMON_BARS else deviation.max()) > tol)


def anchor_groups(context: Context, keys: list[str]) -> dict[int, list[str]]:
    """`{first stored calendar year: keys}` of the keys that have a stored `prices` row."""
    stats = context.store.key_stats(Tables.prices, "ticker", where={"ticker": keys})
    stats = stats.dropna(subset=["first"])
    years = stats["first"].dt.year
    return {int(year): sorted(stats.loc[years == year, "key"]) for year in sorted(years.unique())}


def stored_anchor(context: Context, keys: list[str], since: pd.Timestamp, until: pd.Timestamp) -> pd.DataFrame:
    """Stored `[ticker, date, close_split]` of `keys` over `[since, until]`; empty when there is none."""
    rows = context.store.load(
        Tables.prices, columns=["ticker", "date", "close_split"], where={"ticker": keys}, since=since, until=until, optional=True
    )
    if rows is None or rows.empty:
        return pd.DataFrame(columns=["ticker", "date", "close_split"])
    return rows.assign(date=pd.to_datetime(rows["date"]))


def _compare_year(context: Context, members: list[str], year: int, until: pd.Timestamp, download: Download) -> tuple[list[str], list[str]]:
    """`(restated, unanchored)` members for the bars of calendar `year`; unanchored have no usable common bar.

    Members with no stored bar in `year` are left out of the download and reported unanchored."""
    since, end = pd.Timestamp(year=year, month=1, day=1), min(pd.Timestamp(year=year, month=12, day=31), until)
    stored = stored_anchor(context, members, since, end)
    present = sorted(set(stored["ticker"]))
    if not present:
        return [], list(members)
    fresh = download(present, since, end)
    if fresh.empty:
        logger.warning("Anchor download for %d ticker(s) of year %d returned nothing; not checked", len(present), year)
        return [], list(members)
    restated: list[str] = []
    unanchored = [member for member in members if member not in present]
    for ticker, old in stored.groupby("ticker"):
        old_close = old.set_index("date")["close_split"]
        new_close = fresh.loc[fresh["ticker"] == ticker].set_index("date")["close_split"]
        if _common_dates(old_close, new_close).empty:
            unanchored.append(str(ticker))
        elif anchor_restated(old_close, new_close, ANCHOR_TOLERANCE):
            restated.append(str(ticker))
    logger.info("Anchor check, year %d: %d ticker(s) compared, %d restated", year, len(present), len(restated))
    return restated, sorted(unanchored)


def anchor_restated_keys(context: Context, keys: list[str], until: pd.Timestamp, download: Download) -> list[str]:
    """Keys whose anchor `close_split` disagrees with a fresh Yahoo download.

    Each key is anchored on its first stored year; a key Yahoo does not serve there moves to its next later
    year, one grouped download per candidate year, up to `_MAX_FALLBACK_YEARS` later years."""
    triggered: list[str] = []
    pending: dict[str, int] = {}
    for year, members in (anchor_groups(context, keys) if keys else {}).items():
        restated, unanchored = _compare_year(context, members, year, until, download)
        triggered.extend(restated)
        pending.update({ticker: year for ticker in unanchored})
    for _ in range(_MAX_FALLBACK_YEARS):
        reachable = {ticker: year + 1 for ticker, year in pending.items() if year + 1 <= until.year}
        retry = {ticker: year for ticker, year in pending.items() if ticker not in reachable}
        for year in sorted(set(reachable.values())):
            members = sorted(ticker for ticker, tried in reachable.items() if tried == year)
            restated, unanchored = _compare_year(context, members, year, until, download)
            triggered.extend(restated)
            retry.update({ticker: year for ticker in unanchored})
        pending = retry
    if pending:
        logger.warning("%d ticker(s) could not be anchor-checked, no served year with common bars: %s", len(pending), ", ".join(sorted(pending)))
    if triggered:
        logger.warning("%d ticker(s) whose anchor close_split disagrees with Yahoo: %s", len(triggered), ", ".join(sorted(triggered)))
    return sorted(triggered)
