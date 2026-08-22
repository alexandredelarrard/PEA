
## Ask / request - research best way to extract fundamentals

My goal is to have a step, extracting all key financial fields from the SEC (revenue, income, assets, shareoutstanding, etc.) very consistently for all tickers in a ticker list (sp500 for now, but soon russell 1000).

Consistence means that each financial KPI is :
- comparable quarter to quarter amongst the sam ticker 
- is equivalent between tickers (same information) 
- Null value if the value is not existing at all 
- Same accounting definitions

For context, they will be used down the road to create a zscore or a rank vs peers, so need to have same scale, same information. 

Each KPI to be extracted will have to pass 3 checks : 
- extracted for all quarters of same ticker (until today) 
- No weird kicks due to change of definition or field tag modified 
- Deduction of Q4 from Y -(Q1+Q2+Q3) should align with previous quarter -> avoid kincks. 

I already spent some time to have a strong implementation of edgartools financial KPIs but bumped into lots of issues: 
- code was huge and difficult to understand 
- when looking at example of tickers and KPI, the time series was wront (missing quarters, value off, definition not homogeneous over time because ticker filled it differently over time).
- I started with a too large list of KPIs.

Before doing any implementation or plan, here is what I need you to do.

## What the research needs to do

- Review the current implementation of @@step_extract_structure::fetch_fundamentals_edgartools which extracts the fields and tags and save to the table fundamentals_facts
- Review the current implementation of @@step_extract_structure::rebuild_fundamentals_history which transform the long format to a large format to be used after in the cube_part_fundamentals, and write to fundamentals_history table
- look at @src/data_extraction/utils/fundamentals/ utils functions, especially fundamentals_tags.py which list the fiel tag to extract 
- Look at edgartools deeply and think on how to best handle edge cases per industry to have consistent KPIs 
    - A possible strategy I think of would be to have a specific tag list per ticker to handle edge cases, then a fallback to the tag list. 
    - come up with the best possible solution

From the internet and edgartool repo (https://github.com/dgunning/edgartools):
- Does edgar tools enable to extract financial KPIs consistently (look at the fundamentals_history columns to check what KPIs I am thinking of from schemas.sql) -> this was primary list with lots of issues.
- How does SEC / Bloomberg or other data provider handle such a large diversity of definitions to reconcile ?

**Biggest question I want you to answer** is: 
- **What is the best definition to take into consideration to have a unified definition amongst all Tickers in sp500** ? 
    - How to define CAPEX universaly
    - How to define Assets, Debt, cash, fcf, etc ? 
- Start with the most common financial KPIs and have a clear universal definition to follow.

- Research what key KPI will be shortlisted in this first phase and are core for any quant strategy to understand companies difficulties. Only the universal KPIs first (all sp500 companies have). Here is the start of a shortlist.
    - revenue 
    - operating cost in detail 
    - income 
    - r&d
    - capex 
    - debt 
    - shareholders infos 
    - cash / free cash flow 
    - liabilities / leases, hidden debt basically 

Amongst the issues i had : 
- current Assets not available or total Assets has to sum current and non current for some tickers 
- Share outstanding sometimes without A /B /C class just the B class -> need to have all the shares 
- Revenue for REITs and banks can be difficult, differ from the regular stock revenue (come from rents and interest)
- Capex deduction -> need to reconstruct on most cases. 
- Most difficult tickers were MAA (missing quarters), Banks, REIT, Insurance, Oil and Gas for specific capex


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
- Shortlist the 40+ financial KPIs to focus first on the extraction 
- Come up with the clear definition for all key (40+ ) financial KPIs needed to build financial KPIs for quant strategy down the road. 
- Build a clear and constructive research graph / definition and outcome, with sources. 
- The research will help then the step 2 : Plan 
