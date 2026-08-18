# Persona & Goal
I want you to refactor the file 
src.data_extract.utils.prices.fetch_prices. 

I see many points of improvement. 

I want you to start refactoring fetch_price_history function. 
- It should take how many years of history need to download 
- then read the latest date of extraction prices was done from the db, per ticker, and only get the oldest date. 
- extract all tickers daily past the oldest date (no problem if some tickers already have the date extract, will be handled at upload / upsert time).
- Download the prices 
- save the prices .

The function to know since when the download needs to go back to need to be generic to be reusable afterward.

# Core Constraints & Guardrails
Refactor fetch_prices.py::fetch_price_history and its dependencies. 

Simplify the flow and create a function called fetch_dividends that handles the dividend download from yfinance, 

'
# dividends piggy-back on the SAME download (actions=True); skipped for the
    # market/macro tickers, which are not part of the equity universe
    if download_dividends:
        _save_dividends(context, new)
'

dividends, should be in fetch_dividends.py file in utils/prices folder

The if empty should be directly handled in the data store level.
context.store.save("prices", new) -> should check if new is empty and give a logging.warning('table is empty xxxx') instead of having the condition if not empty in each python function

# Input Context & Tools
Read docs/conding-standard.md and docs/data_conventions on top of Agents.md before starting. 

Plan the list of steps before refactoring. 