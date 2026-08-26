
## Ask / request - research best way to extract fundamentals from shardar

My goal is to have a step, extracting all key financial fields from the shardar (revenue, income, assets, shareoutstanding, etc.) very consistently for all tickers in a ticker list (sp500 for now, but soon russell 1000).

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

I already have in the codebase a strong structure to download sec data using edgartools in src/data_extract/fundamentals and a validator in src/validate
which has a process to identify gaps, record them in fundamentals_check and solve with a sub agent. 

But there are too many errors and time to fix them all will take months. 
I would like instead to create a new step in data_extract that will extract the full history of fundamentals from shardar 
URL = https://sharadar.com/docs

I will subscribe for the full history depth but just the fundamentals since other offers concern: 
- price -> I have with yahoo finance 
- institutionals and insiders -> already extract from sec. 

Before doing any implementation or plan, here is what I need you to do.

## What the research needs to do

- Read Agent.md and docs/Readme.md 
- Read current state of the codebase starting from src/data_extract/fundamentals
- Read /data_store and sql/schemas.sql to understand all the fields needed to be extracted from shardar (specialised fields are missing, that is ok)
- Research on how to organize the codebase to add a step called fundamentals: 
    - need to create a new table fundamentals_shardar 
    - need to create a new step in data_extract -> should not be under fundamentals but an entire new folder called fundamentals_shardar
    - Check how the missing fields will impact the data_aggregation down the road. 
- Research how then to create a fundamentals_history which will be first shardar full extraction, then complemented with the fundamentals_facts extracted from current sec process (for the specific industry features for instance)
- I want shardar first (source of truth) then our sec extraction (still messy). 
- Do not take care of the data_aggregation (both steps and tests), that I will fix later since the codebase changes. 
- Research how you can leverage the trial data (DOW 30) to build a proof of concept, check the flow , missing fields, quality of data. 
- Research all the edge cases shardar has, gap in definitions vs what we built with fundamentals_facts

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

Keep in mind, doing this to have best quality fundamentals data (so research quality of shardar), in history depth (since 1999) fundamentals width (all important KPIs and industry ones), for russell 1000. 