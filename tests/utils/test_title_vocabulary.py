"""The shared title vocabulary changes neither the governance name cleaner nor the insider title classifiers."""

import pytest

from src.data_aggregate.utils.institutionals.insider_quality import officer_role, title_population
from src.utils.names import clean_person_name

# Expected values recorded from the classifiers before the vocabulary was shared.
TITLES = {
    "Chief Executive Officer": (1, "CEO"),
    "Chairman, President & CEO": (1, "CEO"),
    "Chief Financial Officer": (1, "CFO"),
    "Executive Vice President": (1, "VP_LEVEL"),
    "Senior Vice President": (2, "VP_LEVEL"),
    "Vice President": (2, "VP_LEVEL"),
    "Vice Chairman and President": (1, "PRESIDENT"),
    "General Counsel": (2, "other_named_officer"),
    "Treasurer": (2, "VP_LEVEL"),
    "Group President": (1, "other_named_officer"),
    "Director": (2, "other_named_officer"),
    "SVP": (2, "VP_LEVEL"),
    "Former CEO": (4, "other_named_officer"),
}
NAMES = {
    "Luca Maestri Former Senior Vice President": "Luca Maestri",
    "Bob De Lange Group President": "Bob De Lange",
    "Jane Q. Doe Chief Executive Officer": "Jane Q. Doe",
    "Tim Cook CEO": "Tim Cook",
    "Jane Q. Doe General Counsel": "Jane Q. Doe",
    "Jane Q. Doe Co-CEO": "Jane Q. Doe",
    "Jane Q. Doe Executive Vice President": "Jane Q. Doe",
    "Jane Q. Doe Director": "Jane Q. Doe Director",
}


@pytest.mark.parametrize("title", sorted(TITLES))
def test_insider_title_classification_is_unchanged(title: str) -> None:
    assert (title_population(title), officer_role(title)) == TITLES[title]
    print(f"SANITY: {title!r} -> population/role {TITLES[title]} as before the shared vocabulary.")


@pytest.mark.parametrize("raw", sorted(NAMES))
def test_glued_title_stripping_is_unchanged(raw: str) -> None:
    assert clean_person_name(raw) == NAMES[raw]
    print(f"SANITY: {raw!r} -> {NAMES[raw]!r} as before the shared vocabulary.")
