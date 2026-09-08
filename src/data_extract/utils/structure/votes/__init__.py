"""
votes  (src/data_extract/utils/structure/votes/)
------------------------------------------------
Form 8-K **Item 5.07** shareholder-vote tallies, read out of the narratives `fetch_8k_edgar`
has already stored in `sec_8k.item_text`. No new download.

    guard.py    the fabrication guard -- what stops the model inventing a vote table
    roles.py    joins each nominee to the nearest prior proxy for its role category
    flatten.py  a filled `Item507Extract` -> `sec_8k_votes` rows
    fetch.py    orchestration: read, skip what is stored, extract, save

This package is NOT independent of `def14a/`: it imports `clean_person_name` / `clean_text` from
`def14a/validate.py` and `person_key` from `def14a/gender.py` on purpose. The vote role map and
the DEF 14A gender consensus agreeing depends on there being exactly ONE definition of the person
key -- which now lives in `src/utils/names.py` and is re-exported by those two modules, because
the governance cube joins on the same key and may not import this package.

⚠ `roles.py` buckets each nominee with that key, and `flatten.py` writes the result as the twenty
stored `*_ceo` / `*_exec_officer` / `*_non_employee` columns on `sec_8k_votes`. Redefining the key
re-partitions those STORED columns without recomputing them, so a change there is a re-extraction.
"""
from src.data_extract.utils.structure.votes.fetch import fetch_8k_votes_llm

__all__ = ["fetch_8k_votes_llm"]
