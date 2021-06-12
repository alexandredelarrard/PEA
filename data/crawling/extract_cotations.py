from pathlib import Path as pl
import time
from data.crawling.crawling import Crawling
from data.crawling.extract_informations import extract_infos

def create_urls(datas, base_url):
    liste_urls = []
    mapping = datas["mapping_reuters"]

    for k in mapping["CODE"]:
        liste_urls.append(f"{base_url}/companies/{k}")
    return liste_urls


def main_extract(configs, datas, proxy=False, cores =1, sub_loc="vente/seloger"):
    """
    projects = 1 louer, 2 = acheter
    types=1,2 (appart et maison )
    sort=d_dt_crea (date de creation)
    """

    # extract all infos on all pages
    base_path = pl(configs["resources"]["base_path"])

    # initialize crawler
    base_url = configs["urls"]["reuters"]

    # create urls_list
    list_urls = create_urls(datas, base_url)

    # initialize queue of urls
    crawling = Crawling(base_path, proxy=proxy, cores=cores)
    crawling.initialize_queue_drivers()
    crawling.start_threads_and_queues(extract_infos, sub_loc=sub_loc)

    t0 = time.time()
    crawling.initialize_queue_urls(urls=list_urls)
    print('*** Main thread waiting')
    crawling.queues["urls"].join()
    print('*** Done in {0}'.format(time.time() - t0))
    crawling.close_queue_drivers()