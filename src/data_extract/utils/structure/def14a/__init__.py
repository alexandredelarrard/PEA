"""
def14a  (src/data_extract/utils/structure/def14a/)
--------------------------------------------------
Everything that reads a DEF 14A proxy (or the DEF 14C information statement a controlled
company files instead), split by job:

    carve.py     anchors -> the `=== LABEL ===` payload the model reads; called by the
                 `sec-def14a-text` walk, which stores the payloads in `sec_def14a_text`
    flatten.py   a filled `Def14AExtract` -> the parent row and the four child tables
    fetch.py     orchestration: read the stored payloads, extract the open sections, save,
                 then the cross-ticker gender consensus
    tables.py    deterministic HTML table extraction the carve embeds as TSV
    validate.py  row cleaning and the fee repairs; re-exports the two name cleaners
    gender.py    the cross-filing gender consensus; re-exports `person_key`

⚠ `person_key` / `clean_person_name` / `clean_text` no longer live in this package. They are
shared vocabulary between the extraction that writes these tables and the governance cube that
joins them, so they moved to `src/utils/{names,string}.py` -- `src/data_aggregate/` may not
import `src/data_extract/`, and a second copy would let the write side and the read side key
the same human differently. `gender.py` and `validate.py` re-export them, so nothing here or in
`votes/` had to change.
"""

from src.data_extract.utils.structure.def14a.carve import prepare_def14a_sections

__all__ = ["prepare_def14a_sections"]
