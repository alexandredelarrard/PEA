import numpy as np
import yfinance as yf
from yahoo_fin.stock_info import *
from datetime import datetime
from pathlib import Path as pl
import tqdm
from utils.general_cleaning import check_folder

# Fetch the data
def main_extract_stock_pe(data):

    today = datetime.today().strftime("%d-%m-%Y")
    data = data.loc[~data["YAHOO_CODE"].isnull()]
    to_extract = data["YAHOO_CODE"].tolist()

    mapping_yahoo_reuters = data[["REUTERS_CODE", "YAHOO_CODE"]]
    mapping_yahoo_reuters.index = data["YAHOO_CODE"].tolist()
    mapping_yahoo_reuters = mapping_yahoo_reuters["REUTERS_CODE"].to_dict()

    stocks = []
    missing_ticks = []

    for company in tqdm.tqdm(to_extract):
        try:
            ticker_company = yf.Ticker(company)
            infos = ticker_company.info
            stocks.append([company,
                           infos["longName"], 
                           infos["fullTimeEmployees"], 
                           infos["longBusinessSummary"], 
                           infos["profitMargins"], 
                           infos["netIncomeToCommon"],
                           infos["revenueGrowth"], 
                           infos["totalRevenue"] ,
                           infos["operatingMargins"],
                           infos["debtToEquity"], 
                           infos["forwardPE"], 
                           infos["trailingPE"], 
                           infos["revenuePerShare"],
                           infos["dividendYield"],
                           infos["floatShares"],
                           infos["marketCap"],
                           infos['totalDebt'],
                           infos["priceToBook"]
                           ])

        except Exception:
            stocks.append([company, 
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan
                            ])
            missing_ticks.append(company)
            pass 

    columns= ["COMPANY", "NAME", "EMPLOYEES", "SUMMARY", "PROFIT_MARGINS", "NET_INCOME", "REVENUE_GROWTH", "TOTAL_REVENUE", "OPERATING_MARGIN", 
            "DEBT_TO_EQUITY", "FOREWARD_PE", "PE", "REVENUE_PER_SHARE", "DIVIDEND_YIELD", "NBR_SHARES", "MARKET_CAP", "TOTAL_DEBT", "PRICE_TO_BOOK"]
    stocks = pd.DataFrame(stocks, columns=columns)
    stocks.to_csv(rf"C:\Users\de larrard alexandre\OneDrive - The Boston Consulting Group, Inc\Documents\repos_github\PEA\data\yahoo_finance\company_infos_{today}.csv", sep=";", index=False, encoding="latin1")

    print(f"NO STOCK FOUND FOR {missing_ticks}")

    return missing_ticks
        
