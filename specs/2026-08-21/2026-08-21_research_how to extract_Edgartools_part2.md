
You are the best SEC filling expert also quant and python beast. 
Your role is to come up with best research plan to build the best, most accurate and fast running validator on SEC fundamentals extracted and compiled in PEA codebase.

## Ask / request - research best way to validate extracted fields / computed fundamentals

My goal is to have a step, validating all key financial fields from the SEC (revenue, income, assets, shareoutstanding, etc.) very consistently for all tickers in a ticker list (sp500 for now, but soon russell 1000), for all quarters.

I am doing all this data extract check to build best possible quant models, because I think my edge vs quant large firms is that I own end to end from data extraction, aggregation, model, etc. 
So I need you to create the most bullet proof strategy to validate extraction fundamentals are rigorous.

Consistence means that each financial KPI is :
- comparable quarter to quarter amongst the sam ticker 
- is equivalent between tickers (same information) 
- Null value if the value is not existing at all or can be explained clearly (expexted)
- time series makes sense, no strange outliers due to wrong definition or modification in accounting filling standard from the ticker or the norm

Several research and plan has been conducted to build a strong extraction scheme from SEC leveraging edgartools. 
- The build of the fundamentals extraction is done up to phase 3 from the plan (cf below), missing the Q4 deduction to have a robust TTM extraction (expected)

I want you to build a FundamentalsValidator class that : 
- ensure all quarters expected are extracted (quarterly, yearly)
- create a tool agent will use to assess that a list of tickers and list of fundamentals have all the values right, accurate and aligned between tickers. 

## What the research needs to do

- Start by reading the 3 reports @reports/research/financial-data giving all the context and findings regarding how the data is extracted and limitations seen along the way 
- Then read the @reports/planning/active-tasks/2026-08-21-fundamentals-rebuild-plan.md -> which is the plan currently built (end of step 3 right now)
- Then check in the internet what are best strategies to put in place to verify extracted values are meaningful : 
    - no missing value between quarters
    - the tickers value's are consistent over time, quarter to quarter, seen as TTM 
    - the fundamentals are comparable between tickers of the same sector, enabling strong quant strategy down the road 
- Look at how best quant and data sec fundamentals provider structurally and automatically check millions of value's consistency, over time, ticker universe and fundamental space 
- Read the code regarding fundamentals/ and tests/ to check what is already in place
- Then come up with the key findings such that the plan phase (next phase) can propose a very strong way to build the validation tool (to be called valiudation_toolkit.py) that any agent will call to double check the numbers are fully correct.

**Garbage in = garbage out** in a model, you need to fully think of how to identify the data extracted is robust, after running function fetch_fundamentals_sec on tickers.


## What not to do
 
- No implementation of any kind 
- No plan, just research 
- No code change
- No approximate definition from universal Wikipaedia or other generic sources. 
- Do not invent, if not possible raise it and propose trade offs: I will make the decision. 


## proposed steps to build the plan

- read Agents.md + relevant docs/**.md files to get repo context. 
- research from the agents to strongly understand the structure of edgar tools for, for each filling file.
- Look at the code for files and dependencies listed 
- Look at the internet to answer the specific questions 
- Build a clear and constructive research graph / definition and outcome, with sources. 
- The research will help then the step 2 : Plan 
