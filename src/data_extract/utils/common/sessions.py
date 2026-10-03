"""The last US trading session whose close has printed, so price fetchers never store a partial bar."""

from datetime import time

import pandas as pd

from src.constants.constants import MARKET_TIMEZONE

#: US equity regular-session close, exchange-local; conservative on early-close half-days.
US_MARKET_CLOSE_ET = time(16, 0)


def last_completed_session(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """The last US session whose close has printed, tz-naive and normalized; weekends roll back.

    Holidays are not handled (no bar exists for one). A naive `now` is read as exchange-local."""
    et = pd.Timestamp.now(tz=MARKET_TIMEZONE) if now is None else pd.Timestamp(now)
    et = et.tz_localize(MARKET_TIMEZONE) if et.tzinfo is None else et.tz_convert(MARKET_TIMEZONE)

    day = et.normalize().tz_localize(None)
    if et.time() < US_MARKET_CLOSE_ET:
        day -= pd.Timedelta(days=1)
    while day.weekday() >= 5:
        day -= pd.Timedelta(days=1)
    return day
