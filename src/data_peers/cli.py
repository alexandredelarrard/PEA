"""
cli.py  (src/data_peers/cli.py)
-------------------------------
Peer-basket step for the aggregation DAG. Builds the return-correlation + business-description
peer graph ONCE and persists it (to SECTOR_PEERS_PATH, on the mounted ./data volume), so every
downstream feature / target / assemble step just LOADS it instead of recomputing.

    python -m src data_peers deduce-peers [-c ./configs]
"""
import click

from src.constants.command_line_interface import CONFIG_ARGS, CONFIG_KWARGS
from src.context import get_config_context
from src.utils.cli_helper import SpecialHelpOrder
from src.data_peers.step_deduce_peers import StepDeducePeers


@click.group(cls=SpecialHelpOrder)
def cli() -> None:
    """DATA PEERS — build + persist the peer baskets used by the cube."""


@cli.command(help="Deduce peer baskets (correlation + embeddings) and persist the peer dict.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
def deduce_peers(config_path: str) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    StepDeducePeers(context=context, config=config).run()   # saves to SECTOR_PEERS_PATH
