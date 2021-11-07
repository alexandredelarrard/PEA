import time
import pandas as pd
import os 
from datetime import datetime
from pathlib import Path as pl
import logging

def get_pricing_infos(driver):

    pricing_infos = {}
    pricing_driver = driver.find_elements_by_xpath("//div[@class='container']/div[1]/div[1]/table/tbody/tr")
    
    if len(pricing_driver)> 0:
        for table_info in pricing_driver:
            split = table_info.text.split("\n")
            if len(split) == 2:
                sp1 = split[1].replace(",", "")
                if sp1.replace(".","").isdigit():
                    pricing_infos[split[0]] = float(sp1)
                else:
                    pricing_infos[split[0]] = sp1
    
    return pricing_infos


def get_news(driver, current_url):

    def extract_item_news(driver, news_driver, news_infos):

        if len(news_driver) == 1:
            for scrol in range(100, 2900, 250):
                driver.execute_script(f"window.scrollTo(0,{scrol})")
                time.sleep(0.2)

            news_items = news_driver[0].find_elements_by_xpath("//div[@class='item']")
            for news in news_items:
                split = news.text.split("\n")
                if len(split) == 3:
                    news_infos.append({"TITLE" : split[0], "SUMMARY" : split[1], "DATE" : split[2]})
                else:
                    news_infos.append({"TITLE" : "", "SUMMARY" : news.text, "DATE" : ""})

        return news_infos

    news_infos = []

    #key developments 
    driver.get(current_url + "/key-developments")
    news_driver = driver.find_elements_by_xpath("//div[@id='__next']/div/div[4]/div/div/div/section/div/div[2]")
    news_infos = extract_item_news(driver, news_driver, news_infos)
    
    # old news 
    driver.get(current_url + "/news")
    news_driver = driver.find_elements_by_xpath("//div[@id='__next']/div/div[4]/div/div/div/div/div[2]")
    news_infos = extract_item_news(driver, news_driver, news_infos)

    return news_infos


def get_events(driver, current_url):

    dividendes_infos = []
    events_infos = []

    driver.get(current_url + "/events")
    events_driver = driver.find_elements_by_xpath("//div[@class='container']/section")

    for i, section in enumerate(events_driver):
        if i == 1:
            dividendes_driver = section.find_elements_by_xpath("//div/div[3]/table/tbody/tr")
            for div_time in dividendes_driver:
                rows = div_time.find_elements_by_tag_name("td")
                if len(rows) == 5:
                    dividendes_infos.append({"ANNOUNCE_DATE" : rows[0].text, "EXPECTED_DATE" : rows[1].text, 
                                            "RECORDED_DATE" : rows[2].text, "PAYMENT_DATE" : rows[3].text, 
                                            "AMOUNT" : rows[4].text})

        if i == 0:
            time_list = []
            event_list = []
            for div_event in section.find_elements_by_tag_name("span"):
                event_list.append(div_event.text)
            for div_event in section.find_elements_by_tag_name("time"):
                time_list.append(div_event.text)

            if len(time_list) == len(event_list):
                for i in range(len(time_list)):
                    events_infos.append({"EVENT_TITLE" : event_list[i], "EVENT_DATE" : time_list[i]})


    return dividendes_infos, events_infos


def get_people(driver, current_url):

    people_infos = []

    driver.get(current_url + "/people")
    people_driver = driver.find_element_by_tag_name("tbody")
    people = people_driver.find_elements_by_tag_name("tr")

    for member in people:
        rows = member.find_elements_by_tag_name("td")
        if len(rows) == 4:
            people_infos.append({"NAME" : rows[0].text, "AGE" : rows[1].text, "POSITION" : rows[2].text,
                                "APPOINTED" : rows[3].text})

    return people_infos


def get_key_metrics(driver, current_url):

    kpi_infos = {}

    driver.get(current_url + "/key-metrics")
    kpi_driver = driver.find_elements_by_xpath("//div[@id='__next']/div/div[4]/div/div/div/div/div")

    for sub_driver in kpi_driver[:8]:
        title = sub_driver.find_element_by_tag_name("h3").text
        table = sub_driver.find_elements_by_tag_name("tr")
        kpi_infos[title.upper().replace(" ","_")] = {}

        for row in table:
            split = row.text.split("\n")
            if len(split) == 2:
                sp1 = split[1].replace(",", "")
                if sp1.replace(".","").isdigit():
                    kpi_infos[title.upper().replace(" ","_")][split[0]] = float(sp1)
                else:
                    kpi_infos[title.upper().replace(" ","_")][split[0]] = sp1

    return kpi_infos


def get_financials(driver, current_url):

    financials_infos = {}

    for analysis in ["income-statement-annual", "balance-sheet-annual", "cash-flow-annual",
                     "income-statement-quarterly", "balance-sheet-quarterly", "cash-flow-quarterly"]:

        driver.get(f"{current_url}/financials/{analysis}")
        financials_infos[analysis] = {}

        column_driver = driver.find_element_by_tag_name("thead")
        columns = column_driver.text.split("\n")[0].split(" ")
        for i in range(len(columns)):
            financials_infos[analysis][columns[i]] = {}

        row_driver = driver.find_element_by_tag_name("tbody")
        rows = row_driver.find_elements_by_tag_name("tr")

        for row in rows:
            subcat = row.find_element_by_tag_name("th").text
            extract = row.find_elements_by_tag_name("td")[:-1]
            if len(extract) == len(columns):
                for i in range(len(columns)):
                    financials_infos[analysis][columns[i]][subcat] = extract[i].text
    
    return financials_infos


def save_to_disk(company, results, save_path):

    today = datetime.today().strftime("%Y-%m-%d")

    if not os.path.isdir(save_path):
         os.mkdir(save_path)

    if not os.path.isdir(save_path / pl(f"{company}")):
            os.mkdir(save_path / pl(f"{company}"))

    if not os.path.isdir(save_path / pl(f"{company}/{today}")):
            os.mkdir(save_path / pl(f"{company}/{today}"))

    final_path = save_path / pl(f"{company}/{today}")
    
    for key in results["financials_infos"].keys():
        pd.DataFrame(results["financials_infos"][key]).to_csv(final_path / pl(f"{key.upper()}.csv"))

    pd.DataFrame(results["people_infos"]).to_csv(final_path / pl("PEOPLE.csv"), index=False)
    pd.DataFrame(results["dividendes_infos"]).to_csv(final_path / pl("DIVIDENDS.csv"), index=False)
    pd.DataFrame(results["events_infos"]).to_csv(final_path / pl("EVENTS.csv"), index=False)
    pd.DataFrame(results["news_infos"]).to_csv(final_path / pl("NEWS.csv"), index=False)
    pd.DataFrame(results["kpi_infos"]).to_csv(final_path / pl("KPIS.csv"))

    logging.info(f"EXTRACTED {company}")


def validate_cookis(driver):

    cookie = driver.find_elements_by_xpath("//div[@id='onetrust-consent-sdk']")

    if len(cookie) == 1:
        try:
            cookie = cookie[0]
            button = cookie.find_element_by_tag_name("button")
            button.click()
        except Exception:
            pass
    
        time.sleep(1)
    

def extract_infos(driver, sub_loc):

    current_url = driver.current_url
    company  = current_url.split("/")[-1]

    validate_cookis(driver)

    pricing_infos_dict = get_pricing_infos(driver)
    time.sleep(0.5)

    # extract news :
    news_infos = get_news(driver, current_url)
    time.sleep(0.5)

    # events 
    dividendes_infos, events_infos = get_events(driver, current_url)
    time.sleep(0.5)

    # people 
    people_infos = get_people(driver, current_url)
    time.sleep(0.5)

    # key metrics 
    kpi_infos = get_key_metrics(driver, current_url)
    time.sleep(0.5)

    # financials 
    financials_infos = get_financials(driver, current_url)
    time.sleep(0.5)

    save_to_disk(company,  {"pricing_infos" : pricing_infos_dict, 
                       "news_infos": news_infos,
                       "dividendes_infos": dividendes_infos,
                       "events_infos" : events_infos,
                       "people_infos" : people_infos,
                       "kpi_infos" : kpi_infos,
                       "financials_infos" : financials_infos}, 
                       sub_loc)

    return driver, "Done"