import pandas as pd 
import os
from datetime import timedelta, datetime
import seaborn as sns
import numpy as np
import tqdm

group_indus = {'Information Services': 'Information Services',
            'Biotechnology': "sante",
            'Industrials': "Industrials",
            'Communication Services': "communication service",
            'Energy': "Energy",
            'Construction and Materials': "btp",
            'Consumer Staples': "consommation",
            'Electronic Components': "transport et electronique",
            'Utilities': "Energy",
            'Automobiles and Parts': "transport et electronique",
            'Health Care Services': "sante",
            'Real Estate': "immobilier",
            'Hotels & Restaurants': "immobilier",
            'Apparel, Accessories & Luxury Goods': "luxe",
            'Semiconductors':  "transport et electronique",
            'Health Care Equipment': "sante",
            'Airlines': "transport et electronique",
            'Application Software':  'Information Services',
            'Insurance': "financieres",
            'Asset Management':"financieres",
            'Banks': "financieres",
            'Financials': "financieres",
            'Vinters & Tobacco': "consommation",
            'Aerospace & Defense': "transport et electronique",
            'Oil & Gas': "Energy",
            'Materials': "btp",
            'Telecommunications': "communication service",
            'Industrial Machinery': "Industrials",
            'Chemicals': "btp"}
    

def load_portefeuille(configs_general, portefeuille, date_pivot, date_finale=None):

    stocks = {}
    base_path = configs_general["resources"]["base_path"] + "/data/extracted_data"
    date_pivot = pd.to_datetime(date_pivot, format="%Y-%m-%d")

    to_skip = []

    if not date_finale:
        date_finale = datetime.today()

    for ticker in portefeuille:
        liste_dates_company = os.listdir(base_path + f"/{ticker}")
        finance_date = max(liste_dates_company)

        try:
            a = pd.read_csv(base_path + f"/{ticker}/{finance_date}/STOCK.csv")
            
            if a.shape[0] <=200:
                to_skip.append(ticker)
            else:
                stocks[ticker] = a
                stocks[ticker] = stocks[ticker][["Date", "Close"]]
                stocks[ticker]["Date"] = pd.to_datetime(stocks[ticker]["Date"], format="%Y-%m-%d")
                stocks[ticker] = stocks[ticker].loc[stocks[ticker]["Date"].between(date_pivot, date_finale)]
                
                closest_value = (stocks[ticker].loc[stocks[ticker]["Date"]
                                                .between(date_pivot,
                                                        date_pivot + timedelta(days=4)), "Close"]
                                                .mean())
                if pd.isnull(closest_value) : 
                    min_date = stocks[ticker]["Date"].min()
                    print(f"CLOSEST_VALUE for {ticker} is for date {min_date}")
                    closest_value = stocks[ticker].loc[stocks[ticker]["Date"] == min_date].values[0]
                
                stocks[ticker]["Close"] = stocks[ticker]["Close"]*100 / closest_value
            
        except Exception:
            to_skip.append(ticker) 

    return stocks, to_skip


def load_stocks_index(configs_general, date_pivot, date_finale):

    base_path = configs_general["resources"]["base_path"] + "/data/extracted_data"
    date_pivot = pd.to_datetime(date_pivot, format="%Y-%m-%d")

    if not date_finale:
        date_finale = datetime.today()

    stocks = {}

    #cac  & s&p
    for index in ["cac40", "s&p500"]:
        stocks[index] = pd.read_csv(base_path + f"/commodities/{index}.csv")
        stocks[index] = stocks[index][["Date", "Close"]]
        stocks[index]["Date"] = pd.to_datetime(stocks[index]["Date"], format="%Y-%m-%d")
        stocks[index] = stocks[index].loc[stocks[index]["Date"].between(date_pivot, date_finale)]
        stocks[index]["Close"] = stocks[index]["Close"]*100 / stocks[index].loc[stocks[index]["Date"].between(date_pivot,
                                                            date_pivot + timedelta(days=4)), "Close"].values[0]

    return stocks


def aggregate_portfolio(stocks, portefeuille):

    for i, (k, v) in enumerate(stocks.items()):                                     
        if i == 0:
            df_portefeuille = v.rename(columns={"Close" : k})
        else:
            df_portefeuille = df_portefeuille.merge(v.rename(columns={"Close" : k}), 
                                                              on="Date", 
                                                              how="left", 
                                                              validate="1:1")

    df_portefeuille["STRATEGY"] = df_portefeuille[portefeuille].mean(axis=1)

    return df_portefeuille


def portfolios(configs_general, portefeuille, date_pivot, date_finale=None):

    # cac & sp500
    stocks_index = load_stocks_index(configs_general, date_pivot, date_finale)

    #shares from portfolio
    stocks_share, _ = load_portefeuille(configs_general, portefeuille, date_pivot, date_finale)
    stocks = {key: stocks_share[key] for key in portefeuille}
    df_portefeuille = aggregate_portfolio(stocks, portefeuille)
    
    df_portefeuille = df_portefeuille.merge(stocks_index["cac40"].rename(columns={"Close" : "CAC40"}))
    df_portefeuille = df_portefeuille.merge(stocks_index["s&p500"].rename(columns={"Close" : "S&P"}))

    # plot analysis
    df_portefeuille.set_index("Date")[["CAC40", "S&P", "STRATEGY"]].plot(figsize=(10,10))

#### strategy 0

def deduce_porfolio_kpis(liste_tickers, stocks):

    df_portefeuille = aggregate_portfolio(stocks, liste_tickers)

    # get kpis 
    std_porteuf = df_portefeuille["STRATEGY"].std()
    gain = df_portefeuille["STRATEGY"].iloc[-1] - 100 

    correl  = df_portefeuille.drop("STRATEGY", axis=1).corr()
    correl = correl[correl != 1]
    final_correl = correl.mean(axis=1).mean()

    return [gain, std_porteuf, final_correl, liste_tickers] 


def tirage_par_industrie(data):

    sector_count =  data["SECTEUR_0"].value_counts()
    
    selection = []
    for k, v in sector_count.items():
        tickers = data.loc[data["SECTEUR_0"] ==k, "REUTERS_CODE"].to_list()
        selection.append(tickers[np.random.randint(0, v)])
    
    return selection


def top_performers(stocks_share, data, date_pivot, date_finale):

    data["SECTEUR_0"] = data["SECTOR"].map(group_indus)

    if not date_finale:
        date_finale = datetime.today()
    else: 
        date_finale = pd.to_datetime(date_finale, format="%Y-%m-%d")

    final_stock = {}
    sharp = {}
    nbr_years = (date_finale - pd.to_datetime(date_pivot, format="%Y-%m-%d")).days /365

    for k, v in stocks_share.items():
        dividend = data.loc[data["REUTERS_CODE"] == k, "DIV"].values[0]*0.8/100
        pe = data.loc[data["REUTERS_CODE"] == k, "PE"].values[0]
        sharp[k] = (v["Close"].iloc[-1] + nbr_years*v["Close"].median()*dividend  - 100) / v["Close"].std()
        final_stock[k] = sharp[k]/np.sqrt(pe)

    data["SHARP"] = data["REUTERS_CODE"].map(sharp)
    data["FINAL_STOCK"] = data["REUTERS_CODE"].map(final_stock)
    data = data.sort_values(["SECTEUR_0", "FINAL_STOCK"], ascending = [0,0])
    data["ID"] = 1
    data["ID"] = data[["SECTEUR_0", "ID"]].groupby("SECTEUR_0").cumsum()

    return data.loc[data["ID"] <= 5]


def main_heuristique(configs_general, data, date_pivot, date_finale, nbr_iter = 3000):

    # get them all  
    stocks_share, to_skip = load_portefeuille(configs_general, data["REUTERS_CODE"].tolist(), date_pivot, date_finale)
    print(f"MISSING {len(to_skip)} tickers out of {len(data['REUTERS_CODE'].tolist())}")

    # remove to skip 
    sub_data = top_performers(stocks_share, data, date_pivot, date_finale)

    kpis = []
    
    for i in tqdm.tqdm(range(nbr_iter)):

        liste_tickers = tirage_par_industrie(sub_data)
        stocks = {key: stocks_share[key] for key in liste_tickers}
        new_kpi = deduce_porfolio_kpis(liste_tickers, stocks)

        kpis.append(new_kpi)

    results = pd.DataFrame(kpis)
    results.columns = ["GAIN", "STD", "CORR", "TICKERS"]
    
    results["SHARP"] = results["GAIN"] / results["STD"]
    results = results.sort_values(["SHARP"], ascending = [0])
    
    return results 
