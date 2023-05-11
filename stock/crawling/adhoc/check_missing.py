import pandas as pd 
import time
from crawling.crawling import Crawling
import crawling.extract_informations as info


data = pd.read_csv(r"C:\Users\de larrard alexandre\Documents\repos_github\PEA\data\data_for_crawling\old\missing.csv", sep=",")
base_path = pl(configs_general["resources"]["base_path"]) 

cr = Crawling(base_path=base_path, proxy=False, cores=1)
driver = cr.initialize_driver_chrome(proxy=False)

for current_url in ["https://www.reuters.com/companies/EVKN.DE", 
                   "https://www.reuters.com/companies/CAPP.PA",
                   "https://www.reuters.com/companies/WISEA.L",
                    "https://www.reuters.com/companies/VOLVB.ST",
                    "https://www.reuters.com/companies/SWEDA.ST",
                    "https://www.reuters.com/companies/SKFB.ST",
                    "https://www.reuters.com/companies/GIRO.PA",
                    "https://www.reuters.com/companies/SECUB.ST",
                    "https://www.reuters.com/companies/G24N.DE",
                    "https://www.reuters.com/companies/SBSTA.OL",
                    "https://www.reuters.com/companies/RB.L",
                    "https://www.reuters.com/companies/PSMGN.DE",
                    "https://www.reuters.com/companies/PEUP.PA",
                    "https://www.reuters.com/companies/NIBEB.ST",
                    "https://www.reuters.com/companies/TIGOSDB.ST",
                    "https://www.reuters.com/companies/MGOC.AX",
                    "https://www.reuters.com/companies/LEGN.DE",
                    "https://www.reuters.com/companies/KYGA.I",
                    "https://www.reuters.com/companies/IFXGN.DE",
                    "https://www.reuters.com/companies/HMB.ST",
                    "https://www.reuters.com/companies/GJFS.OL",
                    "https://www.reuters.com/companies/GETIB.ST",
                    "https://www.reuters.com/companies/FPE3.DE",
                    "https://www.reuters.com/companies/FLTRE.I",
                    "https://www.reuters.com/companies/BALDB.ST",
                    "https://www.reuters.com/companies/ECP.PA",
                    "https://www.reuters.com/companies/ESSITYA.ST",
                    "https://www.reuters.com/companies/EDV.AX",
                    "https://www.reuters.com/companies/EKTAB.ST",
                    "https://www.reuters.com/companies/DPWGN.DE",
                    "https://www.reuters.com/companies/DARK.L",
                    "https://www.reuters.com/companies/CZR.N",
                    "https://www.reuters.com/companies/BNLD.L",
                    "https://www.reuters.com/companies/BNRGN.DE",
                    "https://www.reuters.com/companies/BPTB.L",
                    "https://www.reuters.com/companies/ATCOA.ST"]:
    driver.get(current_url)

    company  = current_url.split("/")[-1]

    info.validate_cookis(driver)
    time.sleep(1)

    try:
        # get profile infos
        profile_infos_dict = info.get_profile(driver, current_url)
        time.sleep(0.5)

        # pricing infos
        pricing_infos_dict = info.get_pricing_infos(driver)
        time.sleep(0.5)

        # extract news :
        news_infos = info.get_news(driver, current_url)
        time.sleep(0.5)

        # events 
        dividendes_infos, events_infos = info.get_events(driver, current_url)
        time.sleep(0.5)

        # people 
        people_infos = get_people(driver, current_url)
        time.sleep(0.5)

        # key metrics 
        kpi_infos = info.get_key_metrics(driver, current_url)
        time.sleep(2.5)

        # financials 
        financials_infos = info.get_financials(driver, current_url)
        time.sleep(0.5)

        info.save_to_disk(company, {"profile_infos": profile_infos_dict, 
                            "pricing_infos" : pricing_infos_dict, 
                            "news_infos": news_infos,
                            "dividendes_infos": dividendes_infos,
                            "events_infos" : events_infos,
                            "people_infos" : people_infos,
                            "kpi_infos" : kpi_infos,
                            "financials_infos" : financials_infos}, 
                            sub_loc)

        print(current_url)
    
    except Exception as e:
        print(current_url, e)