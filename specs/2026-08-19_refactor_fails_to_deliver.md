
## Ask / request - refactor fetch_fails_to_deliver.py
I want you to refactor the @src/data_extract/utils/prices/fetch_fails_to_deliver.py script. 

It works fine. Download zips , open extract and save. 
I would like you to replace the zip download with a direct edgar tools call to get a more dynamic, up to date extraction instead of zip downloads. 

I want you to only focus on the data_extraction, I will handle the data aggregation, modelling, etc. later for refactoring. I want you to write the minimal code / comment, version, clean and working. 

## What refactor needs to do

Refactor should do the following : 
- read the table already built (Tables.fails_to_deliver) 
- rename the table 'sec_fails_to_deliver' (to help me understand from where the table comes from)
- Only download the dates missing not already stored in the table (only the fillings missing from fillings date)
- extract dates
- cleaning steps, to ensure the fails quantity and fails value have the right unit (refine the tests if needed)
- save the extracted fillings into the table (upsert)
- update the record_run json to keep track of extractions.
- All the variables called from constants.py should be moved to the fetch_13f.py file if only called in the file or through its tests. constants.py is only for constants used through multiple files.
- verify you have all the available history for past 15 years (if possible - expected to have up to mid 2017 from the zips)

## What not to do
 
- Fix more than strict dependency in data_aggregation or modelling functions / files (I will focus on refactoring those parts later). 
- Increase docstrings, history of the refactor in docstrings / comments
- Introduce regressions in any kind (data history depth, extraction, olumns, tickers, etc.)
- Mark as done a code not fully tested with real sample of data (sufficiently substancial sample to not be too slow, but represent a valid strong test).

## proposed steps to build the plan

- read Agents.md + relevant docs/**.md files to get repo context. 
- verifying if edgar tools give the exact same fields, with same history. If anything important is missing, I need to approve. 
- if yes, plan the effort to refactor, reuse all the functions already written / in utils / data.store. 
- always think in a generic fashion, for reusability , clean code. 
- minimal docstring / comment to the strict minimum to what is needed.
- propose a refined set of tests to ensure the new refactor works as before or better -> for instance the thousand deduction is corrected 
- check if following utils are still used somewhere and delete if no longer in use in any other file
  from src.data_extract.utils.common.bulk_cache import (
    cache_dir, ensure_zip, ingested_periods, read_zip_text,
)
    from src.data_extract.utils.common.sec_utils import (
    load_processed_universe)
- Amongst the last steps, add an EFFICIENCY-focused code quality review step which reviews  an in-progress refactor in the Python repo. Get the diff with: `git diff HEAD -- fetch_fails_to_deliver.py`. YOUR ANGLE — EFFICIENCY: Flag wasted work the diff introduces. (sub agent to run it) 
- Then work on the list of findings flagged.
