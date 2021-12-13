import pandas as pd 
import time
from crawling.crawling import Crawling
import Levenshtein as lev

stock = {"GB" : ".L",
        "DE"  : ".DE",   
        "SE" : ".ST",
        "CH" : [".ZU", ".S"],
        "IT"  : ".MI",
        "NL" : ".AS",
        "ES" : [".MA", ".MC"],
        "DK" : ".CO",
        "FI"  : ".HE",
        "NO"  : ".OL",
        "BE" : ".BR",
        "PL" : ".WA",
        "IE" : ".I",
        "AT" : ".VI",
        "PT" : ".LS",     
        "LU" : ".LU"}


def find_info(driver):

    table = driver.find_elements_by_class_name("search-table-data")
    symbols = {}
    if len(table)>0:
        elements = table[0].find_elements_by_tag_name("tr")
        for tr in elements:
            td = tr.find_elements_by_tag_name("td")
            if len(td) == 3:
                if td[0].text not in symbols.keys():
                    symbols[td[0].text] = [td[1].text]
                else:
                    symbols[td[0].text].append(td[1].text)

    return symbols


def handle_cookie(driver):
    cookie = driver.find_elements_by_xpath("//div[@id='onetrust-consent-sdk']")

    if len(cookie) == 1:
        try:
            cookie = cookie[0]
            button = cookie.find_element_by_tag_name("button")
            button.click()
        except Exception:
            pass

    time.sleep(1)


def find_best_match(key, symbols, extensions):

    if len(symbols) == 0:
        return ""

    best_key = ""
    best_len = 100
    for k, value in symbols.items():
        distance = lev.distance(key.lower(), k.lower())
        if distance < best_len:
            best_key = k
            best_len = distance

    proposition = []
    if best_key != "":
        values = symbols[best_key]
        for places in values:
            if "." in places:
                if "." + places.split(".")[1] in extensions:
                    proposition.append(places)
        return ", ".join(proposition)
    else:
        return symbols


if __name__ == "__main__":

    url = "https://www.reuters.com/finance/stocks/lookup?searchType=any&comSortBy=marketcap&sortBy=&dateRange=&search="
    data = pd.read_csv(r"C:\Users\de larrard alexandre\Documents\repos_github\PEA\data\data_for_crawling\old\missing.csv", sep=";")
    base_path = pl(configs_general["resources"]["base_path"])

    cr = Crawling(base_path=base_path, proxy=False, cores=1)
    driver = cr.initialize_driver_chrome(proxy=False)

    data["DEDUCED"] = ""

    for key, country in data[["NAME", "Country"]].values:

        short_key = "+".join(key.split(" ")[:1])
        driver.get(url + short_key)

        handle_cookie(driver)
        symbols = find_info(driver)
        deduced = str(find_best_match(key, symbols, stock[country]))
        data.loc[data["NAME"]== key,"DEDUCED"] = deduced 
        print(key, deduced)

    data.to_csv(r"C:\Users\de larrard alexandre\Documents\repos_github\PEA\data\data_for_crawling\mapping_europe_yahoo.csv", sep=";")