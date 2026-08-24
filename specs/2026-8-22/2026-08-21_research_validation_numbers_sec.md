
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

- Review the first research result @2026-08-21-fundamentals-extraction.md -> conclusion of the specs/2026-08-21_research_how to extract_Edgartools.md request 
- Review the first 2 parts of the plan @2026-08-21-fundamentals-rebuild-plan.md, implemented so far. 
- Review and research17 of 53 fields carry authority: "UNVERIFIED". The research established verbatim FASB/Reg S-X/ASC authority for what it investigated (revenue, debt, capex, D&A, cash, R&D, the required-vs-elective captions) and never touched the ordinary Tier-2/3 lines. I will need you to produce a new research .md answering those questions pending. 
- Research the reason why the following: 
3. The absence register is measured per regime, not per GICS sector — the sector matrix couldn't be translated since "Financials" spans four regimes. Re-measured off the 7.8M facts: Assets present for 441/441 in every regime; bank/insurer 100% absent for currentAssets/currentLiabilities/inventory/R&D (exactly as 17 CFR 210.1-02(bb)(1)(i) predicts); but utility/energy 0% absent for currentAssets — they do file classified balance sheets, so they must not be grouped with banks for that exception. And bank.capex is .43 absent, not ~1.0 — the plan's "not reliably reconstructible" is about intermittency, which I kept excused with a written override because a mixed TTM makes FCF wrong rather than missing.
- Give all the elements to refine / enrich the created jsons, and verify first information given there is correct : 
    - fundamentals_kpis 
    - fundamentals_regimes
    - fundamentals_exceptions 
- Create a research .md file answering all the comments, that implementation will be able to leverage from. (I will skip the plan phase since It is mostly information retreival to have best KPI construction).


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
