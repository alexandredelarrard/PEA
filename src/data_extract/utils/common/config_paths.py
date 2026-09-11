"""
config_paths.py (src/data_extract/utils/common/config_paths.py)
-----------------------------------------------------------------------------------
Where a config directory resolves to, for every `@cache`d loader in `data_extract`.

Lives in `common/` because the registrant register is a cross-cutting concern -- tiers A, B
and C all read it -- and `common/` is the lowest layer, so a loader there cannot reach up
into `fundamentals/` for a path helper without inverting the dependency. `kpi_catalogue`
re-exports the name so its dozen existing importers are unaffected.
"""
from __future__ import annotations

from pathlib import Path

from src.constants.constants import DEFAULT_CONFIG_DIR


def resolve_config_dir(config_dir: str | None = None) -> str:
    """One canonical absolute path for a config directory, whatever spelling reached us.

    Every `@cache`d loader in this package keys on its ARGUMENT, so `None`, `"./configs"`
    and an absolute path pointing at the same directory were three cache entries -- and one
    `StepExtractAllData.run()` parsed the 169 KB catalogue and ran all six validation passes
    twice, because the no-arg and explicit conventions both exist in the tree. Resolving
    first makes that mistake cheap instead of doubling the work.
    """
    return str(Path(config_dir or DEFAULT_CONFIG_DIR).resolve())
