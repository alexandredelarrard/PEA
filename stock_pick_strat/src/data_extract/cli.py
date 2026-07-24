
import click

# TODO
# from src.constants.command_line_interface import (
# )

from src.context import get_config_context
from src.utils.cli_helper import SpecialHelpOrder



@click.group(cls=SpecialHelpOrder)
def cli():
    """
    HART PIPELINE STEPS
    """


@cli.command(
    help="Crawling AUCTIONS for any seller",
    help_priority=3,
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*CRAWL_THREADS_ARG, **CRAWL_THREADS_KWARG)
@click.option(*SELLER_ARGS, **SELLER_KWARGS)
@click.option(*START_DATE_ARGS, **START_DATE_KWARGS)
@click.option(*END_DATE_ARGS, **END_DATE_KWARGS)
def step_crawling_auctions(
    config_path,
    threads: int,
    seller: str,
    start_date: str,
    end_date: str,
):

    config, context = get_config_context(config_path, use_cache=False, save=False)
    crawl = StepCrawlingAuctions(
        config=config,
        context=context,
        threads=threads,
        sellers=seller.split(","),
        start_date=start_date,
        end_date=end_date,
    )

    # get crawling_function
    crawl.run(crawl.get_auctions_urls_to_crawl(), crawl.crawling_auctions_iteratively)

    # python -m src datacrawl step-crawling-auctions -t 6 --seller subdrouot,christies,sothebys,phillips,bonhams,millon


@cli.command(
    help="Crawling ITEMS for any seller",
    help_priority=3,
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*CRAWL_THREADS_ARG, **CRAWL_THREADS_KWARG)
@click.option(*SELLER_ARGS, **SELLER_KWARGS)
def step_crawling_items(config_path, threads: int, seller: str):

    config, context = get_config_context(config_path, use_cache=False, save=False)
    crawl = StepCrawlingItems(
        config=config,
        context=context,
        threads=threads,
        sellers=seller.split(","),
    )

    # get crawling_function
    crawl.run(crawl.get_list_items_to_crawl(), crawl.crawl_items_iteratively)

    # python -m src datacrawl step-crawling-items -t 6 --seller subdrouot,christies,sothebys,phillips,bonhams,millon


@cli.command(
    help="Crawling details for any seller",
    help_priority=6,
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*SELLER_ARGS, **SELLER_KWARGS)
@click.option(*CRAWL_THREADS_ARG, **CRAWL_THREADS_KWARG)
@click.option(*REDO_PICT_ARGS, **REDO_PICT_KWARGS)
def step_crawling_details(
    config_path, threads: int, seller: str, redo_picture: bool = False
):

    config, context = get_config_context(config_path, use_cache=False, save=False)
    crawl = StepCrawlingDetails(
        config=config,
        context=context,
        threads=threads,
        sellers=seller.split(","),
    )
    if not redo_picture:
        crawl.run(crawl.get_list_items_to_crawl(), crawl.crawling_details_function)
    else:
        crawl.run(crawl.get_list_items_missing_pict(), crawl.crawling_details_function)
    # python -m src datacrawl step-crawling-details -t 15 -s subdrouot,bonhams,millon -rdp True
    # python -m src datacrawl step-crawling-details -t 15 -s christies,sothebys,phillips


@cli.command(
    help="Crawling Pictures for any seller",
    help_priority=5,
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*SELLER_ARGS, **SELLER_KWARGS)
@click.option(*CRAWL_THREADS_ARG, **CRAWL_THREADS_KWARG)
def step_crawling_pictures(config_path, threads: int, seller: str):

    config, context = get_config_context(config_path, use_cache=False, save=False)
    crawl = StepCrawlingPictures(
        config=config,
        context=context,
        threads=threads,
        sellers=seller.split(","),
    )

    # get crawling_function
    crawl.run(crawl.get_list_items_to_crawl(), crawl.crawling_picture)
    # python -m src datacrawl step-crawling-pictures -t 15 --seller subdrouot,christies,sothebys,phillips,bonhams,millon


@cli.command(
    help="Crawling Artist list",
    help_priority=5,
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*CRAWL_THREADS_ARG, **CRAWL_THREADS_KWARG)
def step_crawling_artists(config_path, threads: int):

    config, context = get_config_context(config_path, use_cache=False, save=False)
    crawl = StepCrawlingArtists(config=config, context=context, threads=threads)

    # get crawling_function
    crawl.run(crawl.get_list_items_to_crawl(), crawl.crawling_artists)
    # python -m src datacrawl step-crawling-artists -t 1
