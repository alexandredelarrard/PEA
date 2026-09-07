"""
sessions.py  (src/data_extract/utils/common/sessions.py)
------------------------------------------------------------------
Which US trading session is finished, in the fetchers' terms.

Every price fetcher used to end its download window at `pd.Timestamp.today()`, which on any
weekday before 16:00 ET means "include a session that is still trading". yfinance answers
that with a real-looking bar carrying a partial-session OHLC and a fraction of the day's
volume -- measured on the live table, the last stored bar had 0.535x its own Jul-Aug median
volume. Nothing downstream can tell that bar from a settled one, so it flows into returns,
momentum, betas and the labels at full weight.

This module holds the one function that answers "the last close that has actually printed",
so the clamp is defined once instead of at each call site.
"""
from datetime import time
from zoneinfo import ZoneInfo

import pandas as pd

#: US equity regular-session close, in exchange-local time. The half-days (1:00pm ET on the
#: sessions before Independence Day / after Thanksgiving / Christmas Eve) close EARLIER, so
#: this constant is conservative on them too -- it can only ever wait longer, never less.
US_MARKET_CLOSE_ET = time(16, 0)

MARKET_TZ = ZoneInfo("America/New_York")


def last_completed_session(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """The last US session whose CLOSE has printed, tz-naive and normalized.

    Holidays need no handling: yfinance returns no bar for one, so a clamp that lands on a
    holiday simply fetches nothing extra, and the next run's window (which always reaches
    back over the recent tail) picks the real sessions up. The clamp only has to be
    CONSERVATIVE -- it must never include a session still trading, which is the bar the
    unclamped `until=today` wrote.

    `now` is injectable so the clock positions can be tested without freezing time; a naive
    `now` is read as exchange-local, since that is the only frame in which "before the close"
    is a meaningful question."""
    et = pd.Timestamp.now(tz=MARKET_TZ) if now is None else pd.Timestamp(now)
    et = et.tz_localize(MARKET_TZ) if et.tzinfo is None else et.tz_convert(MARKET_TZ)

    day = et.normalize().tz_localize(None)
    if et.time() < US_MARKET_CLOSE_ET:      # today's close has not printed yet
        day -= pd.Timedelta(days=1)
    while day.weekday() >= 5:               # roll back over the weekend
        day -= pd.Timedelta(days=1)
    return day
