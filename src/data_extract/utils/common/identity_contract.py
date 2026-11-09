"""The identity row contract shared by the identity builders and the `Identity` reader.

Role and source labels of the `universe_cik_symbol` / `symbol_tenure` rows, the stored open-start sentinel, the
identity error types and the CIK -> `company_id` row helpers. Pure values and pandas only, no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

#: Stored value of an open start: `valid_from` is a PK column and cannot be NULL.
SENTINEL_START = pd.Timestamp("1900-01-01")

ROLE_WINDOW = "cik_window"
ROLE_EVENT = "cik_event"
ROLE_SYMBOL = "symbol"

FORM345_SOURCE = "form345"
MANUAL_SOURCE = "manual"
#: Notes cover-page (`dei:TradingSymbol`) partition, one row set per Notes zip period.
DEI_SOURCE = "dei"


class IdentityError(ValueError):
    """Base of every identity failure, so one `except` covers the whole layer."""


class CikInTwoEntitiesError(IdentityError):
    """One CIK carries two `company_id`s; asserted by the build, since the PK does not guarantee it."""


class TwoUniverseTickersOneEntityError(ValueError):
    """A merge would put two universe tickers in one entity; the fix is a curated row, never a wider merge."""


def entity_or_singleton(entity_by_cik: Mapping[str, str], cik: str) -> str:
    """The stored entity of a padded CIK; a CIK with no stored row is its own entity `E{cik}`."""
    return entity_by_cik.get(cik, f"E{cik}")


def entity_by_cik_map(lineage: pd.DataFrame) -> dict[str, str]:
    """`{cik: company_id}` from `universe_cik_symbol` rows with padded CIKs (several rows per CIK share one entity)."""
    return dict(zip(lineage["cik"].astype(str), lineage["company_id"].astype(str), strict=False))


def check_one_entity_per_cik(lineage: pd.DataFrame) -> None:
    """Raise `CikInTwoEntitiesError` when one (padded) CIK carries two company ids, which would make `entity_of` order-dependent."""
    per_cik = pd.DataFrame({"cik": lineage["cik"].astype(str), "company_id": lineage["company_id"].astype(str)}).drop_duplicates()
    clashes = per_cik[per_cik.duplicated("cik", keep=False)]
    if not clashes.empty:
        raise CikInTwoEntitiesError(
            f"identity: {clashes['cik'].nunique()} CIK(s) carry two company ids in universe_cik_symbol -- "
            f"{clashes.sort_values(['cik', 'company_id']).to_dict('records')}. One CIK belongs to one company."
        )
