"""
def14a  (src/data_extract/utils/structure/def14a/)
--------------------------------------------------
Everything that reads a DEF 14A proxy (or the DEF 14C information statement a controlled
company files instead), split by job:

    carve.py     anchors -> the `=== LABEL ===` payload the model reads
    flatten.py   a filled `Def14AExtract` -> the parent row and the four child tables
    fetch.py     orchestration: list, skip what is stored, extract, save, then the
                 cross-ticker gender consensus
    tables.py    deterministic HTML table extraction the carve embeds as TSV
    ecd.py       the Pay-versus-Performance / ECD inline-XBRL block (filer-tagged, no LLM);
                 used by `fetch_def14a_edgar`, which is a different fetcher entirely
    validate.py  row cleaning and the fee/name repairs
    gender.py    `person_key` and the cross-filing gender consensus

`person_key` lives in `gender.py` and is imported by `votes/roles.py`: the DEF 14A gender
consensus and the 8-K vote role map agreeing depends on there being ONE definition.
"""
from src.data_extract.utils.structure.def14a.carve import prepare_def14a_sections
from src.data_extract.utils.structure.def14a.fetch import fetch_def14a_llm

__all__ = ["fetch_def14a_llm", "prepare_def14a_sections"]
