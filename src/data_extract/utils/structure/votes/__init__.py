"""
votes  (src/data_extract/utils/structure/votes/)
------------------------------------------------
Form 8-K **Item 5.07** shareholder-vote tallies, read out of the narratives `fetch_8k_edgar`
has already stored in `sec_8k.item_text`. No new download.

    guard.py    the fabrication guard -- what stops the model inventing a vote table
    roles.py    joins each nominee to the nearest prior proxy for its role category
    flatten.py  a filled `Item507Extract` -> `sec_8k_votes` rows
    fetch.py    orchestration: read, skip what is stored, extract, save

This package is NOT independent of `def14a/`: `clean_person_name` / `clean_text` live in
`def14a/validate.py` and `person_key` in `def14a/gender.py`, and both are imported here on
purpose. The vote role map and the DEF 14A gender consensus agreeing depends on there being
exactly ONE definition of the person key.
"""
from src.data_extract.utils.structure.votes.fetch import fetch_8k_votes_llm

__all__ = ["fetch_8k_votes_llm"]
