
## Ask / request - refactor fetch_xxx_edgar.py
I want you to refactor the @src/data_extract/utils/structure/fetch_xxx_edgar.py scripts. 

You will need to review and refactor the following: 
- @fetch_13d_edgar.py 
- @fetch_8k_edgar.py 
- @fetch_def14a_edgar.py 
- @fetch_filing_text.py 

It works fine but can be easily refactored, improved speed and remove redundancy.
EFFICIENCY-focused code quality review step which reviews each file listed above and dependencies, move redundant pieces, speed focus, etc.

I noted : 
- def _worker(ticker: str, cik: str) -> redundant, in utils
- load_cik_mapping -> can it read directly tickers list instead of doing filtering after ? 
- _cik_num -> seems redundant with padding/ can be reused ? 
- _COLS list in each file, is it listing only needed coluns ? 
- refactor _filing_row in each file to use common parts from utils, and restrict to only specific parts
- reduce verbosity of each file (docstring / file header -> minimal to what is needed)

I want you to only focus on the data_extraction, I will handle the data aggregation, modelling, etc. later for refactoring. I want you to write the minimal code / comment, version, clean and working. 

## What refactor needs to do

Refactor should do the following : 
- read the table already built (Tables.sec_xxx) 
- rename the table 'sec_xxx' (to help me understand from where the table comes from)
- Only download the dates missing not already stored in the table (only the fillings missing from fillings date)
- extract dates / fillings not already downloaded, reuse from utils
- cleaning steps respective to each filling file (8k, 13d, etc.)
- save the extracted fillings into the table (upsert)
- update the record_run json to keep track of extractions.
- All the variables called from constants.py should be moved to the respective file if only called in the file or through its tests. constants.py is only for constants used through multiple files.
- verify you have all the available history for past 15 years

## What not to do
 
- Fix more than strict dependency in data_aggregation or modelling functions / files (I will focus on refactoring those parts later). 
- Increase docstrings, history of the refactor in docstrings / comments
- Introduce regressions in any kind (data history depth, extraction, olumns, tickers, etc.)
- Mark as done a code not fully tested with real sample of data (sufficiently substancial sample to not be too slow, but represent a valid strong test).

## proposed steps to build the plan

- read Agents.md + relevant docs/**.md files to get repo context. 
- research from the agents to strongly understand the structure of edgar tools for, for each filling file.
- plan the effort to refactor, reuse all the functions already written / in utils / data.store. 
- always think in a generic fashion, for reusability , clean code. 
- minimal docstring / comment to the strict minimum to what is needed.
- Amongst the last steps, add an EFFICIENCY-focused code quality review step which reviews  an in-progress refactor in the Python repo. Get the diff with: `git diff HEAD -- xxxx.py`. YOUR ANGLE — EFFICIENCY: Flag wasted work the diff introduces. (sub agent to run it) 
- Work on the list of findings flagged.
