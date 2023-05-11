import pandas as pd
from modelling.picking.stocks.stocks_data_prep import ajouter_date_prediction, kpis_past
import matplotlib.pyplot as plt


def add_external_data(stocks, dict_macroeconomy_data):

    stocks["DATE_MONTH"] = stocks["Date"].dt.to_period("M")

    # merge with date to get country & sector
    stocks = pd.merge(stocks, dict_macroeconomy_data["COMPANIES"][["REUTERS_CODE", "Country", "SECTOR"]], 
                            left_on="COMPANY", right_on="REUTERS_CODE", how="left", validate="m:1")

    # merge with date / country to get inflation 
    mapping_country_inflation = {'US' : "USA", 'FR' : "FRA", 'GB' : "GBR", 'DE':"DEU", 
                                'SE' : "SWE", 'CHF' : "CHE", 'AUS' : "G-7", 'IT' : "ITA", 'NL' : "NLD", 
                                'ES' : "ESP", 'DK' : "DNK",
                                'CH' : "CHN", 'FI' : "FIN", 'BE' : "BEL", 'NO' : "NOR", 'KS' : "KOR", 
                                'TW': "CHN", 'AT':"AUT", 'IE' : "IRL", 'JPY' : "JPN", 'PL' : "POL", 'PT' : "PRT",
                                'LU' : "LUX"}

    stocks["COUNTRY"] = stocks["Country"].map(mapping_country_inflation)
    stocks = stocks.merge(dict_macroeconomy_data["INFLATION"], 
                        left_on=["COUNTRY", "DATE_MONTH"],
                        right_on=["LOCATION", "TIME"],
                        how="left")
    
    # merge with date / country to get lt rates 
    mapping_country_inflation["AUS"] = "AUS"
    mapping_country_inflation["CH"] = "KOR"
    mapping_country_inflation["TW"] = "KOR"
    stocks["COUNTRY"] = stocks["Country"].map(mapping_country_inflation)

    stocks = stocks.merge(dict_macroeconomy_data["LT_RATES"], 
                        left_on=["COUNTRY", "DATE_MONTH"],
                        right_on=["LOCATION", "TIME"],
                        how="left",
                        suffixes=("_INFLATION", "_LT_RATES"))

    # merge with date / country to get st rates 
    mapping_country_inflation["AUS"] = "AUS"
    mapping_country_inflation["CH"] = "CHN"
    mapping_country_inflation["TW"] = "CHN"
    stocks["COUNTRY"] = stocks["Country"].map(mapping_country_inflation)

    stocks = stocks.merge(dict_macroeconomy_data["ST_RATES"], 
                        left_on=["COUNTRY", "DATE_MONTH"],
                        right_on=["LOCATION", "TIME"],
                        how="left")

    # merge with date / country to get ppi 
    mapping_country_inflation["AUS"] = "G-7"
    mapping_country_inflation["CH"] = "KOR"
    mapping_country_inflation["TW"] = "KOR"
    mapping_country_inflation["SE"] = "NOR"
    mapping_country_inflation["US"] = "G-7"
    stocks["COUNTRY"] = stocks["Country"].map(mapping_country_inflation)

    stocks = stocks.merge(dict_macroeconomy_data["PPI_RATES"], 
                        left_on=["COUNTRY", "DATE_MONTH"],
                        right_on=["LOCATION", "TIME"],
                        how="left",
                        suffixes=("_ST_RATES", "_PPI_RATES"))

    # merge with date / country to get unemployment 
    mapping_country_inflation["AUS"] = "AUS"
    mapping_country_inflation["CH"] = "KOR"
    mapping_country_inflation["TW"] = "KOR"
    mapping_country_inflation["CHF"] = "NOR"
    mapping_country_inflation["US"] = "USA"

    stocks["COUNTRY"] = stocks["Country"].map(mapping_country_inflation)

    dict_macroeconomy_data["UNEMPLOYMENT_RATES"] = dict_macroeconomy_data["UNEMPLOYMENT_RATES"].\
                                            rename(columns={"LOCATION" : 'LOCATION_UNEMPLOYMENT_RATES',
                                                            "Value" : 'Value_UNEMPLOYMENT_RATES',
                                                            "TIME" : 'TIME_UNEMPLOYMENT_RATES'})

    stocks = stocks.merge(dict_macroeconomy_data["UNEMPLOYMENT_RATES"], 
                        left_on=["COUNTRY", "DATE_MONTH"],
                        right_on=["LOCATION_UNEMPLOYMENT_RATES", "TIME_UNEMPLOYMENT_RATES"],
                        how="left")

    stocks = stocks.drop(['COUNTRY', 'LOCATION_INFLATION',
                        'TIME_INFLATION', 'LOCATION_LT_RATES',
                        'TIME_LT_RATES', 'LOCATION_ST_RATES', 'TIME_ST_RATES',
                        'LOCATION_PPI_RATES', 'TIME_PPI_RATES',
                        'LOCATION_UNEMPLOYMENT_RATES', 'TIME_UNEMPLOYMENT_RATES'
                        ], axis=1)

    # fill missing values by country 
    stocks = stocks.sort_values(["Country", "COMPANY", "DATE_MONTH"])
    for col in ["Value_INFLATION", "Value_LT_RATES", "Value_ST_RATES", "Value_PPI_RATES", "Value_UNEMPLOYMENT_RATES"]:
        stocks[col] = stocks[["Country", col]].groupby("Country").ffill().bfill()

    for col in ["Value_INFLATION", "Value_LT_RATES", "Value_ST_RATES", "Value_PPI_RATES", "Value_UNEMPLOYMENT_RATES"]:
        stocks = stocks.rename(columns={col : col.upper()})

    return stocks


def concatenate_comos(dict_macroeconomy_data):

    for i, commo in enumerate(['GOLD',  's&p500', 'WHEAT', "SILVER", "SUGAR", "COTTON", "CACAO", 
                'BRENT', 'GAS', 'COPPER', "VIX"]):
        df_commo = dict_macroeconomy_data[commo]
        del df_commo["Open"]
        
        if i == 0:
            full_commos = df_commo
            full_commos = full_commos.rename(columns={"Close": commo})
        else: 
            df_commo = df_commo.rename(columns={"Close": commo})
            full_commos = full_commos.merge(df_commo, on="Date", how="left", validate="1:1")

    full_commos = full_commos.ffill().bfill()

    return full_commos


def add_commodities(stocks, dict_macroeconomy_data):

    # add commodities variations for past 5 days #  'COFFEE', 'CACAO', , 
    for commo in ['GOLD',  's&p500', 'WHEAT', 'SUGAR', 'SILVER','COTTON',
                'BRENT', 'GAS', 'COPPER', "VIX"]:
        df_commo = dict_macroeconomy_data[commo]
        try:
            del df_commo["Open"]
        except Exception:
            pass

        df_commo = ajouter_date_prediction(df_commo)
        df_commo = kpis_past(df_commo, "Close", commo, k_days=5)

        stocks = stocks.merge(df_commo, on="Date", how="left", validate="m:1")

    return stocks