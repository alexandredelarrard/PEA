import time
import pandas as pd 
from xml.dom import minidom
from datetime import datetime
import glob
import requests
from tqdm import tqdm

from crawling.crawling_step import Crawling


def get_all_xmls(path):

    response = requests.get(path)
    xml_content = response.content
    p1 = minidom.parseString(xml_content)
    tagname= p1.getElementsByTagName('loc')

    all_paths = []
    for x in tagname:
        all_paths.append(x.firstChild.data)

    return all_paths

def get_list_articles(all_articles):

    list_articles = []
    for url_articles in tqdm(all_articles):
        day_articles = get_all_xmls(url_articles)
        list_articles += day_articles

    list_articles = pd.DataFrame(list_articles, columns=["url"])
    list_articles["type"] = list_articles["url"].apply(lambda x: str(x).split("/")[3] if len(str(x).split("/"))>=4 else "")
    list_articles = list_articles.loc[list_articles["type"].isin(["markets", "tech", "policy", "business", "consensus-magazine"])]

    list_articles["date"] = list_articles[["url", "type"]].apply(lambda x: "/".join(x["url"].split("https://www.coindesk.com/" + x["type"] + "/")[1].split("/", 3)[:3]), axis=1)

    return list_articles 


def load_downloaded_articles():

    articles = glob.glob("./data/history/news/*.csv")

    all = pd.DataFrame()
    for f in articles:
        all = pd.concat([all, pd.read_csv(f)])
    
    return all


def extract_infos(driver):
    return driver.find_element_by_tag_name("article").text

if __name__ == "__main__":

    crawling = Crawling(proxy=False, cores=5)

    all_paths = get_all_xmls("https://www.coindesk.com/arc/outboundfeeds/sitemap-index/?outputType=xml")
    specific_articles = get_list_articles(all_paths)

    #filtered to already downloaded
    already_there = load_downloaded_articles()
    if already_there.shape[0]>0:
        specific_articles = specific_articles.loc[~specific_articles["url"].isin(already_there["url"].tolist())]
        print(f"already crawled {already_there.shape[0]} articles / remaining {specific_articles.shape[0]}")

    crawling.initialize_queue_drivers()
    crawling.start_threads_and_queues(extract_infos, sub_loc="./data/history/news")

    t0 = time.time()
    crawling.initialize_queue_urls(urls=specific_articles.to_dict(orient="records"))
    print('*** Main thread waiting')
    crawling.queues["urls"].join()
    print('*** Done in {0}'.format(time.time() - t0))
    crawling.close_queue_drivers()

    # redo for missed urls 
    # t0 = time.time()
    # crawling.initialize_queue_urls(urls=crawling.missed_urls)
    # print('*** Main thread waiting')
    # crawling.queues["urls"].join()
    # print('*** Done in {0}'.format(time.time() - t0))
    
    print(f"REMAINING MISSED : {crawling.missed_urls}")
