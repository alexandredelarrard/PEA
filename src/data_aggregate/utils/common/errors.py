"""Empty-panel and optional-source contracts for cube builders."""

import pandas as pd


#: D5: every builder answers an absent source with the SAME empty frame. A fresh object each
#: call, never a module-level constant -- `PanelMerger.add` and several callers reindex or
#: assign onto what they get back, and a shared instance would be mutated across builds.
def _empty_panel() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "ticker"])


def _absent(df: pd.DataFrame | None, need: set[str] | None = None) -> bool:
    """Whether a source is missing, empty or lacks an unconditionally required column."""
    if df is None or df.empty:
        return True
    return bool(need) and not need.issubset(df.columns)
