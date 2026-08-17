"""
DEF 14A LLM save: strings must be NUL-free before the Postgres upsert
(src/data_extract/utils/structure/fetch_def14a_llm.py::_strip_nul).

DEF 14A filings are HTML/PDF-derived, so LLM-extracted strings (company_name,
ceo_name_proxy, the def14a_json dump) can carry a stray NUL (\x00). Postgres TEXT
columns reject NUL -> psycopg2 'a string literal cannot contain NUL characters'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_extract.utils.structure.fetch_def14a_llm import _strip_nul


def test_strip_nul_removes_nul_and_preserves_everything_else():
    df = pd.DataFrame([
        {"ticker": "AAA", "company_name": "Acme\x00 Corp", "ceo_name_proxy": "Jane\x00 Doe",
         "def14a_json": '{"name": "a\x00b"}', "board_size": 9, "note": None},
        {"ticker": "BBB", "company_name": "Clean Co", "ceo_name_proxy": "John Roe",
         "def14a_json": '{"name": "ok"}', "board_size": 7, "note": "fine"},
    ])
    out = _strip_nul(df.copy())

    # no NUL survives in ANY cell of ANY column (dtype-agnostic: pandas 2 object + pandas 3 str)
    assert not any(isinstance(v, str) and "\x00" in v
                   for c in out.columns for v in out[c]), "NUL still present"
    # NULs stripped exactly (chars around them kept)
    assert out.loc[0, "company_name"] == "Acme Corp"
    assert out.loc[0, "ceo_name_proxy"] == "Jane Doe"
    assert out.loc[0, "def14a_json"] == '{"name": "ab"}'
    # untouched values preserved (clean strings, null, numerics)
    assert out.loc[1, "company_name"] == "Clean Co"
    assert pd.isna(out.loc[0, "note"])
    assert list(out["board_size"]) == [9, 7]

    print("\n=== SANITY CHECK: DEF 14A NUL stripping ===")
    print("  \\x00 removed from company_name / ceo_name_proxy / def14a_json before the "
          "Postgres upsert (fixes 'a string literal cannot contain NUL characters'); "
          "clean strings, None and numeric columns are left untouched. Validated.")


if __name__ == "__main__":
    test_strip_nul_removes_nul_and_preserves_everything_else()
