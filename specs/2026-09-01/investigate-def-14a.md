you are the sec expert regarding financial fundamentals. 
Your role is to review and research def_14a / c data extraction done in the codebase today and identify any data gap from edgar tools (then what can we do) or our own codebase (what would be the fixes) 

# status 
- def 14 is fetched with @src/data_extract/utils/structure/fetch_def14a_edgar.py   
- it produces 5 different tables : sec_def_14a, *_director_comp, *_executive_comp, *_ownership, *_votes. 
- Those tables have variable filling rate with gaps identified so far : 

## sec_def_14a
- Company name not filled consistently
- peo_name, sporadic filling (correlated to company_name missing filling), same for peo_total_comp
- peo_actually_paid_comp sometimes negative 
- auditor data sporadically filled 
- table overall looks very little filled. How can we increase filling rate 

## sec_def14a_director_comp 

- total seems well filled 
- fee earned as well 
- however stock awards / option awards / pension change, other comp is mostly missing 
- total should be te sym of all fields, including those missing, why are they missing and can we retreive them more consistently ? 
- salary for ticker ='A' is unrealistically high (> Billions). Why, can it be fixed or constrained ? 


## sec_def14a_executive_comp 

- name is well filled, but sometimes has 'Former' instead of name, why ? 
- title is very sporadically filled. Can it be more consistent, can we use previous name title to backfill or foward fill the title (edge case will be when title change for same person)
- does the total comp of ceo align with director comp total ? 

## sec_def14a_votes

- say_on_pay has lots of null for board_recommendation, why ? 
- can we have better filling rate for column board_recommendation ? 
- is ther a way to also get the number of votes for, and votes against to check how close and divided the board is ? 

# General questions

- Is there a date threshold, like before X date, format of filling of def 14a changed, explaining why data filled is sporadic. 

- lots of warnings from edgar.core : example 
"2026-09-01 10:28:44 - edgar.core - WARNING - _filings.py - SGML fetch failed for 0000874761-15-000021, falling back to homepage: SEC returned HTML or XML content instead of expected SGML filing data. This may indicate an invalid request or temporary SEC server issue." -> why ? any way to fix this ? 

- Is it a pure edgar tool issue or a filling issue (format unconsistent over time and between fillings ? )

- How accurate are those data ? Can I trust it to build strong alphas for quant strategy down the road ? 


## proposed steps to research for futur plan build

- read Agents.md  
- read docs/data_source.md + docs/data_schemas.md
- Read the def14a functions used to extract those tables
- Research how edgartools extract this data & how the sec fill those data in XML / HTML formats 
- Look for format change in time or between tickers 
- Research if possible to extract info more robustly
- Investigate deeply your assumptions and check it works on at least 10-15 tickers and 5 random years picked between 2000 to today to check format variation handling. 
- **This is research only phase, which report will be used to plan as a second step.**