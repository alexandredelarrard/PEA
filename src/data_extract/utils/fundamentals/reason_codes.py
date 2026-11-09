"""
reason_codes.py  (src/data_extract/utils/fundamentals/reason_codes.py)
--------------------------------------------------------------------------------------------
Resolver-internal `dc_code` shared by the facts layer: a `Resolution` carrying a code is unresolved
(`Resolution.resolved`). The other codes live next to the mechanism that sets them (`xbrl_linkbase.py`,
`periods.py`); none is written to a table.
"""

from __future__ import annotations

#: The filer tags nothing for this field anywhere in the filing; set by `_materialise` and by `resolve_field`'s
#: final fallback.
NOT_DISCLOSED = "not_disclosed"
