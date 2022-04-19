import time
import pandas as pd
import os 
from datetime import datetime
from pathlib import Path as pl
import logging

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

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

    # TODO: scrowl down 2/4/10 times until stabilized ...

    def extract_item_news(driver, news_driver, news_infos, index):

        if len(news_driver) == 1:

            if index ==1:
                for scrol in range(100, 7500, 250):
                    driver.execute_script(f"window.scrollTo(0,{scrol})")
                    time.sleep(0.1)
                time.sleep(1.5)

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
    time.sleep(2)
    news_driver = driver.find_elements_by_xpath("//div[@id='__next']/div/div[4]/div/div/div/section/div/div[2]")
    news_infos = extract_item_news(driver, news_driver, news_infos, 0)
    
    # old news 
    driver.get(current_url + "/news")
    time.sleep(2)
    news_driver = driver.find_elements_by_xpath("//div[@id='__next']/div/div[4]/div/div/div/div/div[2]")
    news_infos = extract_item_news(driver, news_driver, news_infos, 1)

    return news_infos


def get_events(driver, current_url):

    dividendes_infos = []
    events_infos = []

    driver.get(current_url + "/events")
    time.sleep(2)
    WebDriverWait(driver, 15).until(EC.visibility_of_element_located((By.CLASS_NAME, 'container')))

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
    time.sleep(2)
    WebDriverWait(driver, 30).until(EC.visibility_of_element_located((By.CLASS_NAME, 'table-container')))

    try:
        people_driver = driver.find_element_by_tag_name("tbody")
        people = people_driver.find_elements_by_tag_name("tr")

        for member in people:
            rows = member.find_elements_by_tag_name("td")
            if len(rows) == 4:
                people_infos.append({"NAME" : rows[0].text, "AGE" : rows[1].text, "POSITION" : rows[2].text,
                                    "APPOINTED" : rows[3].text})

    except Exception:
        people_infos = {}
        pass 

    return people_infos


def get_key_metrics(driver, current_url):

    kpi_infos = {}

    driver.get(current_url + "/key-metrics")
    time.sleep(3)
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
        time.sleep(1)
        WebDriverWait(driver, 30).until(EC.visibility_of_element_located((By.CLASS_NAME, 'tables-container')))

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

        # get currency 
        if analysis == "income-statement-annual":
            section = driver.find_elements_by_xpath("//section[@class='section']/div/span")
            if len(section)>0:
                currency = section[0].text
            else:
                currency = ""
    
    return financials_infos, currency


def get_profile(driver, current_url):

    profile_infos = {}

    WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.CLASS_NAME, 'Profile-about-1d-H-')))
    profile_driver = driver.find_elements_by_xpath("//div[@class='container']/div/div/table")

    # main metrics 
    if len(profile_driver) >= 1:
        for row in profile_driver[0].find_elements_by_tag_name("tr"):
            text =row.text.split("\n")
            profile_infos[text[0].upper()] = text[1]

    # desc
    desc_driver = driver.find_elements_by_xpath("//div[@class='Profile-about-1d-H-']/p")
    if len(desc_driver) >= 1:
        profile_infos["PRESENTATION"] = desc_driver[0].text

    # company name
    desc_driver = driver.find_elements_by_xpath("//div[@class='Profile-about-1d-H-']/h3")
    if len(desc_driver) >= 1:
        profile_infos["COMPANY_NAME"] = desc_driver[0].text

    # analyst ratings 
    ratings_driver = driver.find_elements_by_xpath("//div[@class='container section']/div")
    if len(ratings_driver) >= 1:
        profile_infos["RATING"] = ratings_driver[0].text
    
    return {"PROFILE" : profile_infos}


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

    pd.DataFrame(results["profile_infos"]).to_csv(final_path / pl("PROFILE.csv"))
    pd.DataFrame(results["people_infos"]).to_csv(final_path / pl("PEOPLE.csv"), index=False)
    pd.DataFrame(results["dividendes_infos"]).to_csv(final_path / pl("DIVIDENDS.csv"), index=False)
    pd.DataFrame(results["events_infos"]).to_csv(final_path / pl("EVENTS.csv"), index=False)
    pd.DataFrame(results["news_infos"]).to_csv(final_path / pl("NEWS.csv"), index=False)
    pd.DataFrame(results["kpi_infos"]).to_csv(final_path / pl("KPIS.csv"))

    logging.info(f"EXTRACTED {company}")


def validate_cookis(driver):

    cookie = driver.find_elements_by_xpath("//div[@class='ot-btn-container']")
    
    if len(cookie) == 1:
        try:
            cookie = cookie[0]
            button = cookie.find_element_by_tag_name("button")
            button.click()
        except Exception:
            pass
    

def extract_infos(driver, sub_loc):

    redo_finance = False
    current_url = driver.current_url
    company  = current_url.split("/")[-1]

    # get profile infos
    time.sleep(1.5)
    validate_cookis(driver)
    profile_infos_dict = get_profile(driver, current_url)

    # pricing infos
    time.sleep(1.5)
    validate_cookis(driver)
    pricing_infos_dict = get_pricing_infos(driver)

    # extract news :
    time.sleep(1.5)
    validate_cookis(driver)
    news_infos = get_news(driver, current_url)

    # events 
    try:
        dividendes_infos, events_infos = get_events(driver, current_url)
    except Exception:
        print(f"no event for {company}")
        dividendes_infos = []
        events_infos = []
        pass

    # people 
    try:
        people_infos = get_people(driver, current_url)
    except Exception:
        print(f"no people for {company}")
        people_infos = [{"NAME" :  "", "AGE" : "", "POSITION" : "",
                                    "APPOINTED" : ""}]
        pass

    # financials 
    try:
        financials_infos, currency = get_financials(driver, current_url)
        if currency != "":
            profile_infos_dict["PROFILE"]["CURRENCY"] = currency
    except Exception:
        time.sleep(15)
        redo_finance = True
        pass

    # key metrics 
    kpi_infos = get_key_metrics(driver, current_url)

    if redo_finance:
        try:
            financials_infos, currency = get_financials(driver, current_url)
            if currency != "":
                profile_infos_dict["PROFILE"]["CURRENCY"] = currency
        except Exception:
            print(f"no finance info for {company}")
            financials_infos = []
            pass

    if len(financials_infos) > 0:
        save_to_disk(company, {"profile_infos": profile_infos_dict, 
                        "pricing_infos" : pricing_infos_dict, 
                        "news_infos": news_infos,
                        "dividendes_infos": dividendes_infos,
                        "events_infos" : events_infos,
                        "people_infos" : people_infos,
                        "kpi_infos" : kpi_infos,
                        "financials_infos" : financials_infos}, 
                        sub_loc)
    else:
        return driver, "ERROR"

    return driver, "Done"