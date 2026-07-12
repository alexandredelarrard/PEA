# Quant Factor Dashboard (S&P 500)

A local Streamlit app for a value/growth factor investing strategy: pulls
price history, fundamentals, macro data, and news, scores stocks on
value/growth factors, backtests a long-only top-N portfolio, and displays
everything in an interactive dashboard.

## Quick start

```bash
cd quant_dashboard
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then add your free FRED API key
```

Get a free FRED API key (takes 2 minutes, no cost): https://fred.stlouisfed.org/docs/api/api_key.html

Then fetch data (first run takes a while — this hits real APIs, be patient
and don't hammer them):

```bash
python -m data.fetch_prices          # 10yr OHLCV for all S&P 500 tickers
python -m data.fetch_fundamentals    # snapshot fundamentals (yfinance)
python -m data.fetch_macro           # CPI + Fed balance sheet from FRED
python -m data.fetch_news            # recent headlines per ticker
```

Then launch the dashboard:

```bash
streamlit run app.py
```

## IMPORTANT: free-data limitations you need to know about

**1. Fundamentals history is shallow with free sources.**
`yfinance` only exposes ~4 years of annual/quarterly financials
(revenue, EBITDA, R&D, etc.) per ticker — Yahoo doesn't give more for free.
True 10-year fundamental history isn't available from any free, ToS-compliant
API I'm aware of. Two practical paths:
  - **Use SimFin's free tier** (https://simfin.com) — they offer bulk CSV
    downloads of ~10+ years of income statement / balance sheet / cashflow
    data for US public companies, free for personal/non-commercial use.
    `data/fetch_fundamentals.py` has a stub function `load_simfin_bulk()`
    you can point at a downloaded SimFin bulk export.
  - **Build history yourself going forward.** Every time you run
    `fetch_fundamentals.py`, it appends a dated snapshot to
    `data_store/fundamentals_history.parquet`. Run it monthly/quarterly and
    in a couple years you'll have your own history. Not useful today, but
    free and compounding.
  - PE ratio, market cap, and current growth metrics ARE available today
    per-ticker via yfinance `.info` — those work fine for a cross-sectional
    (point-in-time) factor screen even without deep history.

**2. Historical news is essentially unavailable for free at 10-year depth.**
`yfinance`'s `.news` only returns a handful of recent headlines per ticker,
no archive. For real historical news at scale, free options are limited to:
  - **GDELT Project** (https://www.gdeltproject.org) — free, huge historical
    news event database, but heavier to work with (BigQuery / raw files).
  - **NewsAPI.org** free tier — only covers the last ~30 days, not history.
  `data/fetch_news.py` implements current headlines via yfinance so the
  dashboard has *something* live, and documents the GDELT path for anyone
  who wants to build real historical news coverage later.

**3. Inflation and Fed balance sheet are fully solved for free.**
FRED (`fredapi`) gives clean, complete, free daily/monthly series:
  - CPI: series `CPIAUCSL`
  - Fed balance sheet total assets: series `WALCL`
  10 years of both, no limitations, no paid tier needed.

**4. Price history is fully solved for free.**
`yfinance` gives full daily OHLCV for 10+ years for any ticker, no API key
needed. This part just works.

## Project structure

```
quant_dashboard/
├── app.py                       # Streamlit dashboard (entry point)
├── config.py                    # paths, env var loading
├── requirements.txt
├── .env.example
├── data/
│   ├── fetch_prices.py          # S&P 500 tickers + 10yr OHLCV via yfinance
│   ├── fetch_fundamentals.py    # yfinance snapshot fundamentals + SimFin stub
│   ├── fetch_macro.py           # FRED: CPI, Fed balance sheet
│   └── fetch_news.py            # yfinance recent headlines
├── strategy/
│   ├── factors.py                # value/growth factor scoring
│   └── backtest.py               # top-N monthly-rebalance backtest vs SPY
└── data_store/                   # parquet/csv cache written by fetch scripts
```

## Strategy logic (starting point — tune as you like)

Composite z-score across, per stock:
  - Value: trailing P/E (lower = better), EV/EBITDA (lower = better)
  - Growth: revenue growth YoY (higher = better)
  - Quality/spend: R&D as % of revenue (context factor, not scored by default)
  - Size: market cap (used for filtering/liquidity, not scored)

Stocks are ranked by composite score each rebalance date; the backtest goes
long the top N (default 30), equal-weighted, rebalanced monthly, compared
against SPY buy-and-hold.

This is a reasonable, standard starting factor model — not investment
advice, and you should sanity-check every factor definition and the
backtest for lookahead bias before trusting the numbers.
