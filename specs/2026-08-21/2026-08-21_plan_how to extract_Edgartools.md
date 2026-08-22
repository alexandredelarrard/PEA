
## Ask / request - plan best way to extract ticker sec fundamentals 

My goal is to have a step, extracting all key financial fields from the SEC (revenue, income, assets, shareoutstanding, etc.) very consistently for all tickers in a ticker list (sp500 for now, but soon russell 1000).

Consistence means that each financial KPI is :
- comparable quarter to quarter amongst the sam ticker 
- is equivalent between tickers (same information) 
- Null value if the value is not existing at all 
- Same accounting definitions

For context, they will be used down the road to create a zscore or a rank vs peers and fed into ML models to predict futur stock moves, so need to have same scale, same information. 

Each KPI to be extracted will have to pass 3 checks : 
- extracted for all quarters of same ticker (until today) 
- No weird kicks due to change of definition or field tag modified 
- Deduction of Q4 from Y -(Q1+Q2+Q3) should align with previous quarter -> avoid kincks. 

I already have an implementation of fundamentals_facts and fundamentals_history tables that I don't like in current PEA codebase. 
I want you to build the best plan possible doing the following.
The plan will have 2 parts:
- plan to clean current code to rebuild fully the fundamentals tables -> remove old code too complex
- plan to create the best new architecture now we know all the caveats, with the clear scope.

## What the plan needs to do

- Start by reading the research outcome from phase 1: @2026-08-21-fundamentals-extraction.md
- Read /docs/coding_standard.md before planing the refactor / feature recreation 
    - The plan should take into account coding standards mainly : 
        - keep constants / global variable at the start of each file if only used in the file, otherwise push into constants.py 
        - Doctrings short at top of file / in each function -> has to be read by human and maintained 
        - Write code for beginners, clear, clean and modular. Utils folder is the go to if function are used many times.
- Look at /src/data_extract/fundamentals/*/*py -> all the functions used to currently extract fundamentals facts & build wide table from edgartools call. 

- 1. Plan to remove all the implementations that is tool long, complex, erroneous with wrong architecture.
    - remove the python files from data_extraction related to fundamentals extraction (only the part related to fundamental extraction, keeping earnings surprises or table unrelated to creation of the tables fundamentals)
    - remove also the tests and checks linked to those files. 
    - Plan to remove any unused file, variable from constants.py as well 

- 2. Once the plan to clean code is robust and structured, I want you to plan for the build of best codebase / architecture to extract facts and store in fundamentals_facts table (long format), then build the fundamentals_history (wide format) as a second step taking the following into consideration: 
    - Plan to steal edgartools  Q4 formula in edgar/ttm when deducing Q4 from Y. Plan should leverage decumulate_quarterly_flow's and rewrite it. 
    - 1. Breadth vs comparability: I want option B. Universal + regime-gated (36-45 fields, Tiers 1-3 with a regime column). 
    - 2. Resolution mechanism. drive roll-ups from the filer's own calculation linkbase (§3.2) with the tag list as fallback: xbrl.calculation_linkbase()
    - 3. Substrate. Per-filing filing.xbrl() (current)
    - 4. Debt definition. totalDebt= Gross debt + operating leases + finance leases
    - 5. Cash definition.  cash + short-term investments ( cash equivalents, and amounts generally described as restricted cash or restricted cash equivalents.)
    - 6. 
    - 7. zero-vs-missing get decided: Adopt a Compustat-style _DC reason code per value, with a "combined into" destination
    - 8. Plan only ends with a 30 ticker test rebuild of both fundamentals_facts and fundamentals_history.

- Instead of tag list in a python file, should be a json in configs, read when needed. 
- Plan should update schemas.sql to only take into consideration the fundamentals from Tiers 1-3 and needed fields to have robust extraction / calculation
- Plan should include docs/*.md files update after / during refactoring 
- Plan  should include the refactor of the src/validate  and tests/ folders, to ensure right value creation, non erroneous. Checks should include strict rules on : 
    - All tickers (490 = sp500 tickers - redudant tickers - too short history tickers) should have fundamentals / quarterly data extracted
    - All quarters from a ticker sould be deduced up to run time 
    - All fundamentals should be extracted for all tickers for similar period than quarterly dates, except if flagged in exception file (specific sectors) 
    - Values should evolve in a range, detecting outliers over time (time series quarter to quarter). If need decision, let's discuss. 
- Plan should include a last refactoring step / sub agent, Amongst the last steps, add an EFFICIENCY-focused code quality review step which reviews  an in-progress refactor in the Python repo. Get the diff with: `git diff HEAD -- fetch_xxxx.py`. ANGLE — EFFICIENCY: Flag wasted work the diff introduces. (sub agent to run it) 

In the plan, this validator should be a dedicated class, running after the data extraction / creation of fundamentals_history, and flag any error / outlier in a report table to be further sanity checked. 

**When building the plan, keep in mind extractions will have to run daily, during the night, to have an automatic trading algorithm reading newly extracted data, run ML models and take positions in stock.**

## What not to do
 
- No implementation of any kind 
- No code change, just plan
- No approximate definition from universal Wikipaedia or other generic sources. 
- Do not invent, if not possible raise it and propose trade offs: I will make the decision. 
- Do not assume, verify from pyton files or context codebase.

## proposed steps to build the plan

- read Agents.md + relevant docs/**.md files to get repo context. 
- Look at the code for files and dependencies listed 
- Look at the internet to check best coding practices in case of doubt
- Focus the plan on the 40+ financial KPIs listed in the research phase
- Come up with the clear definition for all key (40+ ) financial KPIs needed to build financial KPIs for quant strategy down the road. 
- The plan will help then the step 3 : Implementation 
- If you need to check tables, I renamed: fundamentals_history is now fundamentals_history_legacy, fundamentals_facts is now fundamentals_facts_legacy
