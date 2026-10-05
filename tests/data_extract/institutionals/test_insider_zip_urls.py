"""The insider zip is fetched from either SEC hosting path: the likelier one first, the other on a miss."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from src.constants.constants import SEC_INSIDER_SWAP_YEAR, SEC_INSIDER_URL_NEW_TEMPLATE, SEC_INSIDER_URL_TEMPLATE
from src.data_extract.utils.common.bulk_cache import ensure_zip
from src.data_extract.utils.institutionals.fetch_insider_transactions import zip_urls


def test_each_quarter_lists_both_paths_with_the_likelier_first():
    before, after = f"{SEC_INSIDER_SWAP_YEAR - 1}q4", f"{SEC_INSIDER_SWAP_YEAR}q1"
    assert zip_urls(before) == (SEC_INSIDER_URL_TEMPLATE.format(quarter=before), SEC_INSIDER_URL_NEW_TEMPLATE.format(quarter=before))
    assert zip_urls(after) == (SEC_INSIDER_URL_NEW_TEMPLATE.format(quarter=after), SEC_INSIDER_URL_TEMPLATE.format(quarter=after))
    print(f"SANITY: {before} tries the old path first and {after} the new one first; both quarters keep the other path as a fallback.")


def test_a_404_on_the_new_path_falls_back_to_the_old_one(tmp_path):
    quarter = f"{SEC_INSIDER_SWAP_YEAR}q1"
    new, old = zip_urls(quarter)
    requested: list[str] = []

    def _get(url: str, **kwargs: object) -> SimpleNamespace:
        requested.append(url)
        if url == new:
            return SimpleNamespace(status_code=404)
        return SimpleNamespace(status_code=200, iter_content=lambda chunk_size: iter([b"zip bytes"]))

    context = cast(Any, SimpleNamespace(sec_session=SimpleNamespace(get=_get)))
    path = ensure_zip(context, tmp_path / f"{quarter}.zip", zip_urls(quarter), label=f"insider {quarter}")

    assert requested == [new, old]
    assert path is not None and path.read_bytes() == b"zip bytes"
    print(f"SANITY: {quarter} missed on the new path (HTTP 404), was downloaded from the old path, and is cached as {path.name}.")
