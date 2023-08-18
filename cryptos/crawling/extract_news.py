from pathlib import Path as pl
import time
import pandas as pd
import os 
from datetime import datetime

from utils.general_functions import check_save_path
from crawling.crawling_step import Crawling
from crawling.extract_informations import extract_infos


if __name__ == "__main__":

    crawling = Crawling(proxy=False, cores=2)
    driver = crawling.initialize_driver_chrome()

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
    