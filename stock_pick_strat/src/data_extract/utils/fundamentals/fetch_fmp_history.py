"""
Historical analyst & governance data from Financial Modeling Prep (FMP).

Four endpoints, each returning a ticker's FULL history in ONE call (no
pagination), fetched via the shared `fmp_client` (multi-key rotation +
`fetched_at` incremental):

  analyst grades      grades-historical              MONTHLY rating distribution
                      (strongBuy/buy/hold/sell/strongSell counts per month, 2018+)
  analyst actions     grades                          EVENT-LEVEL upgrades/downgrades
                      (dated rating changes per broker, 2012+)
  exec compensation   governance-executive-compensation  ANNUAL, per named officer
                      (salary/bonus/stock/total per fiscal year, from proxies, 2002+)
  estimates           analyst-estimates (annual)      ANNUAL forward consensus
                      (revenue/EBITDA/netIncome/EPS avg + #analysts, ~5y back + fwd)

Point-in-time notes
-------------------
* grades-historical / grades / exec-comp are genuinely historical: each row
  carries the date it became public (`as_of`), so features built from them are
  backtestable.
* analyst-estimates has NO per-row estimate date (only the fiscal period), so it
  is a forward consensus snapshot. We stamp `as_of` = the pull date and accrue a
  point-in-time series over successive runs (same pattern as the yfinance
  estimate snapshot); only the forward horizon is meaningfully usable live.

Run:
    python -m src.data_extract.fetch_fmp_history
"""
from __future__ import annotations

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.fmp_client import fetch_incremental


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.normalize()


# --------------------------------------------------------------------------- #
# analyst grades: monthly rating distribution (grades-historical)
# --------------------------------------------------------------------------- #
_GRADE_COLS = ["ticker", "as_of", "strong_buy", "buy", "hold", "sell", "strong_sell"]


def normalize_grades(records: list[dict], ticker: str) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=_GRADE_COLS)
    rows = [{
        "ticker": ticker, "as_of": r.get("date"),
        "strong_buy": r.get("analystRatingsStrongBuy"),
        "buy": r.get("analystRatingsBuy"),
        "hold": r.get("analystRatingsHold"),
        "sell": r.get("analystRatingsSell"),
        "strong_sell": r.get("analystRatingsStrongSell"),
    } for r in records if r.get("date")]
    df = pd.DataFrame(rows, columns=_GRADE_COLS)
    if df.empty:
        return df
    df["as_of"] = _to_dt(df["as_of"])
    for c in ("strong_buy", "buy", "hold", "sell", "strong_sell"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["as_of"])
    return (df.sort_values(["ticker", "as_of"])
              .drop_duplicates(subset=["ticker", "as_of"], keep="last")
              .reset_index(drop=True))


def fetch_analyst_grades(context: Context, tickers: list[str], pause: float = 0.3,
                         refetch_window_days: int = 30, session=None) -> pd.DataFrame:
    return fetch_incremental(
        context, tickers, endpoint="grades-historical",
        normalize=normalize_grades, dedup_keys=["ticker", "as_of"],
        path_key="ANALYST_GRADES_HISTORY_PATH",
        refetch_window_days=refetch_window_days, pause=pause, session=session,
        desc="analyst grades (monthly)")


# --------------------------------------------------------------------------- #
# analyst actions: dated upgrades / downgrades (grades)
# --------------------------------------------------------------------------- #
_ACTION_COLS = ["ticker", "as_of", "grading_company", "previous_grade",
                "new_grade", "action"]


def normalize_actions(records: list[dict], ticker: str) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=_ACTION_COLS)
    rows = [{
        "ticker": ticker, "as_of": r.get("date"),
        "grading_company": r.get("gradingCompany"),
        "previous_grade": r.get("previousGrade"),
        "new_grade": r.get("newGrade"),
        "action": r.get("action"),
    } for r in records if r.get("date")]
    df = pd.DataFrame(rows, columns=_ACTION_COLS)
    if df.empty:
        return df
    df["as_of"] = _to_dt(df["as_of"])
    df = df.dropna(subset=["as_of"])
    return (df.sort_values(["ticker", "as_of"])
              .drop_duplicates(subset=["ticker", "as_of", "grading_company",
                                       "new_grade", "action"], keep="last")
              .reset_index(drop=True))


def fetch_analyst_actions(context: Context, tickers: list[str], pause: float = 0.3,
                          refetch_window_days: int = 30, session=None) -> pd.DataFrame:
    return fetch_incremental(
        context, tickers, endpoint="grades",
        normalize=normalize_actions,
        dedup_keys=["ticker", "as_of", "grading_company", "new_grade", "action"],
        path_key="ANALYST_ACTIONS_HISTORY_PATH",
        refetch_window_days=refetch_window_days, pause=pause, session=session,
        desc="analyst actions (upgrades/downgrades)")


# --------------------------------------------------------------------------- #
# executive compensation: annual per officer (governance-executive-compensation)
# --------------------------------------------------------------------------- #
_EXEC_COLS = ["ticker", "as_of", "fiscal_year", "name_and_position",
              "salary", "bonus", "stock_award", "total"]


def normalize_exec_comp(records: list[dict], ticker: str) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=_EXEC_COLS)
    rows = [{
        "ticker": ticker,
        # public when the proxy is accepted/filed
        "as_of": r.get("acceptedDate") or r.get("filingDate"),
        "fiscal_year": r.get("year"),
        "name_and_position": r.get("nameAndPosition"),
        "salary": r.get("salary"),
        "bonus": r.get("bonus"),
        "stock_award": r.get("stockAward"),
        "total": r.get("total"),
    } for r in records if (r.get("acceptedDate") or r.get("filingDate"))]
    df = pd.DataFrame(rows, columns=_EXEC_COLS)
    if df.empty:
        return df
    df["as_of"] = _to_dt(df["as_of"])
    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce").astype("Int64")
    for c in ("salary", "bonus", "stock_award", "total"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["as_of", "fiscal_year", "name_and_position"])
    return (df.sort_values(["ticker", "fiscal_year", "as_of"])
              .drop_duplicates(subset=["ticker", "fiscal_year", "name_and_position"],
                               keep="last")
              .reset_index(drop=True))


def fetch_exec_comp(context: Context, tickers: list[str], pause: float = 0.3,
                    refetch_window_days: int = 30, session=None) -> pd.DataFrame:
    return fetch_incremental(
        context, tickers, endpoint="governance-executive-compensation",
        normalize=normalize_exec_comp,
        dedup_keys=["ticker", "fiscal_year", "name_and_position"],
        path_key="EXEC_COMP_HISTORY_PATH",
        refetch_window_days=refetch_window_days, pause=pause, session=session,
        desc="executive compensation (annual)")


# --------------------------------------------------------------------------- #
# analyst estimates: annual forward consensus (analyst-estimates)
# --------------------------------------------------------------------------- #
_EST_COLS = ["ticker", "as_of", "fiscal_date", "eps_avg", "eps_high", "eps_low",
             "revenue_avg", "ebitda_avg", "net_income_avg",
             "num_analysts_eps", "num_analysts_rev"]


def normalize_estimates(records: list[dict], ticker: str) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=_EST_COLS)
    # no per-row estimate date -> snapshot; as_of = the pull date (accrues PIT)
    as_of = pd.Timestamp.today().normalize()
    rows = [{
        "ticker": ticker, "as_of": as_of, "fiscal_date": r.get("date"),
        "eps_avg": r.get("epsAvg"), "eps_high": r.get("epsHigh"), "eps_low": r.get("epsLow"),
        "revenue_avg": r.get("revenueAvg"), "ebitda_avg": r.get("ebitdaAvg"),
        "net_income_avg": r.get("netIncomeAvg"),
        "num_analysts_eps": r.get("numAnalystsEps"),
        "num_analysts_rev": r.get("numAnalystsRevenue"),
    } for r in records if r.get("date")]
    df = pd.DataFrame(rows, columns=_EST_COLS)
    if df.empty:
        return df
    df["fiscal_date"] = _to_dt(df["fiscal_date"])
    for c in ("eps_avg", "eps_high", "eps_low", "revenue_avg", "ebitda_avg",
              "net_income_avg", "num_analysts_eps", "num_analysts_rev"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["fiscal_date"])
    return (df.sort_values(["ticker", "fiscal_date"])
              .drop_duplicates(subset=["ticker", "fiscal_date"], keep="last")
              .reset_index(drop=True))


def fetch_estimates(context: Context, tickers: list[str], pause: float = 0.3,
                    refetch_window_days: int = 30, session=None) -> pd.DataFrame:
    return fetch_incremental(
        context, tickers, endpoint="analyst-estimates",
        normalize=normalize_estimates,
        dedup_keys=["ticker", "fiscal_date", "as_of"],
        path_key="FMP_ESTIMATES_HISTORY_PATH", params={"period": "annual"},
        refetch_window_days=refetch_window_days, pause=pause, session=session,
        desc="analyst estimates (annual)")


if __name__ == "__main__":
    from src.context import get_config_context
    from src.data_extract.fetch_prices import get_sp500_tickers

    _, ctx = get_config_context("./configs", use_cache=False, save=False)
    tks = get_sp500_tickers(ctx)
    fetch_analyst_grades(ctx, tks)
    fetch_analyst_actions(ctx, tks)
    fetch_exec_comp(ctx, tks)
    fetch_estimates(ctx, tks)
