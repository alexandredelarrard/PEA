from pathlib import Path as pl
import time
import os 
from datetime import datetime
from crawling.crawling import Crawling
from crawling.extract_informations import extract_infos

def create_urls(data, base_url):
    liste_urls = []
    today = datetime.today().strftime("%Y-%m-%d")

    data = data.loc[~data["REUTERS_CODE"].isnull()]

    for k in data["REUTERS_CODE"]:
        if not os.path.isdir(base_url / pl("companies") / k / today / "PEOPLE.csv"):
            liste_urls.append(f"{base_url}/companies/{k}")
    return liste_urls


def main_extract_financials(configs_general, datas, proxy=False, cores =2, sub_loc=""):

    # extract all infos on all pages
    base_path = pl(configs_general["resources"]["base_path"])

    # initialize crawler
    base_url = configs_general["urls"]["reuters"]

    # create urls_list
    list_urls = create_urls(datas, base_url)

    # initialize queue of urls
    # driver= crawling.initialize_driver_chrome()
    crawling = Crawling(base_path, proxy=proxy, cores=cores)
    crawling.initialize_queue_drivers()
    crawling.start_threads_and_queues(extract_infos, sub_loc=sub_loc)

    t0 = time.time()
    crawling.initialize_queue_urls(urls=list_urls)
    print('*** Main thread waiting')
    crawling.queues["urls"].join()
    print('*** Done in {0}'.format(time.time() - t0))
    crawling.close_queue_drivers()
