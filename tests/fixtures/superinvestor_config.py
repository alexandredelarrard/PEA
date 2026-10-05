"""A fixture `<config_dir>/superinvestors/` (hand overrides, optional roster history) for the roster tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Appaloosa's filer succession: Appaloosa Management LP files to 2015-12-31, Appaloosa LP from 2016-03-31.
APPALOOSA_OLD, APPALOOSA_NEW = "0001006438", "0001656456"
APPALOOSA_CHAIN = {APPALOOSA_OLD: [{"cik": APPALOOSA_OLD, "to": "2015-12-31"}, {"cik": APPALOOSA_NEW, "from": "2016-03-31"}]}


def write_roster_config(root: Path, overrides: dict[str, Any] | None = None, snapshots: list[dict[str, Any]] | None = None) -> str:
    """Write `<root>/configs/superinvestors/overrides.json` (empty `cik_overrides` / `unresolvable` unless given) and,
    when `snapshots` is given, `dataroma_roster_history.json`; return the config dir."""
    folder = root / "configs" / "superinvestors"
    folder.mkdir(parents=True, exist_ok=True)
    blob = {"cik_overrides": {}, "unresolvable": {}, **(overrides or {})}
    (folder / "overrides.json").write_text(json.dumps(blob), encoding="utf-8")
    if snapshots is not None:
        history = {"_README": ["fixture"], "snapshots": snapshots}
        (folder / "dataroma_roster_history.json").write_text(json.dumps(history), encoding="utf-8")
    return str(root / "configs")
