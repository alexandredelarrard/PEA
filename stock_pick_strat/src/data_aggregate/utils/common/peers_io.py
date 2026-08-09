"""
peers_io.py  (src/data_aggregate/utils/common/peers_io.py)
-------------------------------------------------------
Read the peer baskets every cube step needs.

`StepDeducePeers.run()` returns the cached dict when `SECTOR_PEERS_PATH` exists, so calling
it is cheap and does NOT recompute the correlation/embedding peer groups. But it is a Step
in another `src/` subfolder, and the assemble step needs the peer dict only to write the
`peers` JSON column -- it should not have to construct a Step (and, in the old code, run
the entire price prologue) to get it.

This reads the cache directly and falls back to the deduce step when it is absent, so the
dependency is one function instead of a cross-folder Step instantiation.
"""
from __future__ import annotations

import json
import logging

from omegaconf import DictConfig
from src.data_peers.step_deduce_peers import StepDeducePeers
from src.context import Context

logger = logging.getLogger(__name__)

def load_peers(context: Context, config: DictConfig | None = None) -> dict:
    """The peer dict, from the `SECTOR_PEERS_PATH` cache; recomputed via `StepDeducePeers`
    only when the cache is missing and a config is supplied."""
    path = context.paths["SECTOR_PEERS_PATH"]
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("peer cache at %s is unreadable -> recomputing", path)
    if config is None:
        return {}
    # imported lazily: only the cache-miss path depends on the peers package
    return StepDeducePeers(context=context, config=config).run()


def load_peers_or_raise(context: Context, config: DictConfig | None = None) -> dict:
    """`load_peers`, but a missing/empty peer dict is an error: every feature panel is
    peer-relative, so building a cube without peers would silently emit all-NaN features."""
    peers = load_peers(context, config)
    if not peers:
        raise RuntimeError(
            f"no peer baskets at {context.paths['SECTOR_PEERS_PATH']} -> run "
            "`python -m src data_peers deduce-peers` first")
    return peers
