from pathlib import Path as pl
import time
import pandas as pd
import os 
from datetime import datetime

from utils.general_functions import check_save_path
from crawling.crawling import Crawling
from crawling.extract_informations import extract_infos

def create_urls(data, base_path, base_url):

    liste_urls = []
    today = datetime.today().strftime("%Y-%m-%d")

    data = data.loc[~data["REUTERS_CODE"].isnull()]

    for k in data["REUTERS_CODE"]:
        check_save_path(base_path / pl("data/extracted_data") / k)

        dirs = os.listdir(base_path / pl("data/extracted_data") / k)
        if len(dirs) > 0:
            max_date = max(set(dirs) - set(["financials"])) 

            if (datetime.today() - pd.to_datetime(max_date)).days > 15:
                liste_urls.append(f"{base_url}/companies/{k}")
            else:
                if not os.path.isfile(base_path / pl("data/extracted_data") / k / max_date / "PEOPLE.csv"):
                    liste_urls.append(f"{base_url}/companies/{k}")
        else:
            liste_urls.append(f"{base_url}/companies/{k}")

    print(f"TO CRAWL {len(liste_urls)}")

    return liste_urls


def main_extract_financials(configs_general, data, proxy=False, cores =1, sub_loc=""):

    # extract all infos on all pages
    base_path = pl(configs_general["resources"]["base_path"])

    # initialize crawler
    base_url = configs_general["urls"]["reuters"]

    # create urls_list
    liste_urls = create_urls(data, base_path, base_url)

    crawling = Crawling(base_path, proxy=proxy, cores=cores)
    # driver = crawling.initialize_driver_chrome()
    crawling.initialize_queue_drivers()
    crawling.start_threads_and_queues(extract_infos, sub_loc=sub_loc)

    t0 = time.time()
    crawling.initialize_queue_urls(urls=liste_urls)
    print('*** Main thread waiting')
    crawling.queues["urls"].join()
    print('*** Done in {0}'.format(time.time() - t0))

    # redo for missed urls 
    t0 = time.time()
    crawling.initialize_queue_urls(urls=crawling.missed_urls)
    print('*** Main thread waiting')
    crawling.queues["urls"].join()
    print('*** Done in {0}'.format(time.time() - t0))
    crawling.close_queue_drivers()

    print(f"REMAINING MISSED : {crawling.missed_urls}")

    return crawling.missed_urls
